#!/usr/bin/env python3
"""Driver for the RDMSim ablation grid.

The full grid is 7 scenarios x 28 arms x 5 seeds = 980 runs, which is far too much to
launch as a single all-or-nothing job: a run is minutes long, several JVMs run at once, and
each JVM wants its own heap. So this driver is built around two ideas.

1. **Select a subset.** Every dimension of the grid (scenario, learning config, lambda,
   seed, baseline) is a filter, so any slice can be run on its own and resumed later.
2. **Bound the resources.** ``--jobs`` and ``--heap`` are explicit and checked against
   system memory *before* anything launches, because the JVM's default heap is a quarter of
   system RAM and N of those at once will thrash or be killed. ``--batch-size`` and
   ``--max-wall`` let the grid be walked over several sittings.

A manifest (``experiments/manifest.json``) records the status of every cell, so resuming is
based on recorded state rather than guessing from whatever files happen to exist. A cell
killed halfway through is re-run rather than silently counted as finished.

See docs/ABLATION.md for the guide; ``--help`` for the flags.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import zlib
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Arm registry
#
# Adding a learning config here is all that is needed to add it to the grid: the config
# keys are written verbatim into the cell's solver.config. Keep the keys in sync with
# readConfigFile() in src/RDMSim_TAAS.java.
# ---------------------------------------------------------------------------

LEARNING_CONFIGS: dict[str, dict[str, str]] = {
    # Model frozen at the seeded Dirichlet prior. Bottom rung: isolates whether the
    # learning loop does any work at all.
    "static": {"staticModel": "true", "useSurpriseUpdating": "false",
               "surpriseMeasureForGamma": "MIP"},
    # Classic Bayesian updating: Dirichlet +1, no surprise weighting.
    "bayes": {"staticModel": "false", "useSurpriseUpdating": "false",
              "surpriseMeasureForGamma": "MIP"},
    # SMiLe surprise-weighted updating, one arm per surprise signal.
    "smile-CC": {"staticModel": "false", "useSurpriseUpdating": "true",
                 "surpriseMeasureForGamma": "CC"},
    "smile-BF": {"staticModel": "false", "useSurpriseUpdating": "true",
                 "surpriseMeasureForGamma": "BF"},
    "smile-MIP": {"staticModel": "false", "useSurpriseUpdating": "true",
                  "surpriseMeasureForGamma": "MIP"},
}

# Fixed-policy baselines. They bypass the solver, so they cost almost nothing to run, and
# they pin the endpoints of the MST-RT mixture line that every algorithm arm is read
# against. See the ground-truth table in docs/ABLATION.md.
BASELINE_POLICIES = ["mst", "rt", "random"]

DEFAULT_LAMBDAS = [0.0, 2.0, 5.0, 10.0, 20.0]
DEFAULT_SEEDS = [222, 223, 224, 225, 226]
ALL_SCENARIOS = [0, 1, 2, 3, 4, 5, 6]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEMPLATE = PROJECT_ROOT / "src" / "solver.config"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    """One run: a scenario, an arm, and a seed."""

    scenario: int
    arm: str                      # e.g. "smile-MIP_lam5" or "fixed_mst"
    seed: int
    timesteps: int
    crn: bool
    learning: str | None = None   # None for baseline arms
    lam: float | None = None      # None for baseline arms
    baseline: str | None = None   # None for solver arms
    status: str = STATUS_PENDING
    wallclock_s: float | None = None
    # Horizon the recorded wallclock_s was measured at. Restored from the manifest, which
    # may hold a timing from a shorter pilot; run time is superlinear in the horizon, so a
    # timing is only comparable when this matches the horizon now requested.
    measured_timesteps: int | None = None
    peak_rss_mb: float | None = None
    attempts: int = 0
    command: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable identifier, also the manifest key."""
        return f"s{self.scenario}/{self.arm}/seed{self.seed}"

    def directory(self, output_root: Path) -> Path:
        return output_root / f"s{self.scenario}" / self.arm / f"seed{self.seed}"

    @property
    def is_baseline(self) -> bool:
        return self.baseline is not None


def arm_sort_key(arm: str) -> tuple:
    """Baselines first, then learning configs in ladder order, then by lambda."""
    if arm.startswith("fixed_"):
        return (0, arm, 0.0)
    for index, learning in enumerate(LEARNING_CONFIGS):
        prefix = f"{learning}_lam"
        if arm.startswith(prefix):
            try:
                lam = float(arm[len(prefix):].replace("_", "."))
            except ValueError:
                lam = 0.0
            return (1, index, lam)
    return (2, arm, 0.0)


def arm_name(learning: str, lam: float) -> str:
    """Directory-safe arm name. 5.0 renders as lam5, 2.5 as lam2_5."""
    text = f"{lam:g}".replace(".", "_")
    return f"{learning}_lam{text}"


# ---------------------------------------------------------------------------
# Selection parsing
# ---------------------------------------------------------------------------

def parse_int_list(spec: str, valid: list[int] | None, name: str) -> list[int]:
    """Parse "all", "0,3,5", "0-6", or a mix of those into a sorted unique int list."""
    if spec.strip().lower() == "all":
        if valid is None:
            raise SystemExit(f"--{name}=all is not meaningful; give explicit values")
        return list(valid)
    values: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", part)
        if match:
            low, high = int(match.group(1)), int(match.group(2))
            if high < low:
                raise SystemExit(f"--{name}: range '{part}' is reversed")
            values.extend(range(low, high + 1))
        else:
            try:
                values.append(int(part))
            except ValueError:
                raise SystemExit(f"--{name}: '{part}' is not an integer or range")
    if valid is not None:
        bad = sorted(set(values) - set(valid))
        if bad:
            raise SystemExit(f"--{name}: {bad} outside allowed {valid}")
    return sorted(set(values))


def parse_float_list(spec: str, name: str) -> list[float]:
    """Lambda values. Ranges are not supported: the useful values are not evenly spaced."""
    if spec.strip().lower() == "all":
        return list(DEFAULT_LAMBDAS)
    values: list[float] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            raise SystemExit(f"--{name}: '{part}' is not a number")
        if values[-1] < 0:
            raise SystemExit(f"--{name}: lambda must be >= 0; got {values[-1]}")
    return sorted(set(values))


def parse_name_list(spec: str, valid: list[str], name: str) -> list[str]:
    """Parse "all", "none", or a comma list of names, case-insensitively."""
    text = spec.strip()
    if text.lower() == "all":
        return list(valid)
    if text.lower() in ("none", ""):
        return []
    lookup = {v.lower(): v for v in valid}
    out: list[str] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if part.lower() not in lookup:
            raise SystemExit(f"--{name}: '{part}' unknown; choose from {valid} (or all/none)")
        out.append(lookup[part.lower()])
    # Preserve the registry order so cell listings are stable.
    return [v for v in valid if v in out]


def parse_arm_specs(spec: str, name: str) -> list[tuple[str, float]]:
    """Parse explicit "learning:lambda" pairs, e.g. "smile-MIP:5,bayes:0"."""
    pairs: list[tuple[str, float]] = []
    lookup = {k.lower(): k for k in LEARNING_CONFIGS}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"--{name}: '{part}' must be learning:lambda, e.g. smile-MIP:5")
        learning_text, lam_text = part.rsplit(":", 1)
        learning_key = learning_text.strip().lower()
        if learning_key not in lookup:
            raise SystemExit(
                f"--{name}: unknown learning config '{learning_text}'; "
                f"choose from {sorted(LEARNING_CONFIGS)}")
        try:
            lam = float(lam_text)
        except ValueError:
            raise SystemExit(f"--{name}: '{lam_text}' is not a number in '{part}'")
        pairs.append((lookup[learning_key], lam))
    return pairs


def parse_exclusions(spec: str) -> tuple[set[tuple[str, float]], set[int]]:
    """Split --exclude into arm pairs and whole scenarios ("scenario=4")."""
    arm_parts: list[str] = []
    scenarios: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"scenario\s*=\s*(\d+)", part, re.IGNORECASE)
        if match:
            scenarios.add(int(match.group(1)))
        else:
            arm_parts.append(part)
    arms = set(parse_arm_specs(",".join(arm_parts), "exclude")) if arm_parts else set()
    return arms, scenarios


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def parse_size(text: str) -> int:
    """Parse a JVM-style size ("1g", "768m", "512k", "1073741824") into bytes."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmgt]?)b?", text.strip(), re.IGNORECASE)
    if not match:
        raise SystemExit(f"--heap: '{text}' is not a size like 1g, 768m or 512k")
    value = float(match.group(1))
    unit = match.group(2).lower()
    factor = {"": 1, "k": 1024, "m": 1024 ** 2, "g": 1024 ** 3, "t": 1024 ** 4}[unit]
    return int(value * factor)


def format_size(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(num_bytes) < 1024 or unit == "TiB":
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}TiB"


def total_system_memory() -> int | None:
    """Total physical RAM in bytes, or None if it cannot be determined."""
    if platform.system() == "Windows":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            return None
        return None
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def check_memory_budget(jobs: int, heap_bytes: int, max_memory: str | None,
                        force: bool) -> None:
    """Refuse to start if jobs x heap would oversubscribe RAM.

    This is the guard that stops the grid from being killed by the OOM killer halfway
    through: the JVM reserves its heap lazily but will happily grow into it, and N
    concurrent JVMs each sized for the whole machine is the default failure mode.
    """
    requested = jobs * heap_bytes
    total = total_system_memory()

    if max_memory is not None:
        budget = parse_size(max_memory)
        source = f"--max-memory {max_memory}"
    elif total is not None:
        budget = int(total * 0.75)
        source = f"75% of {format_size(total)} system RAM"
    else:
        print("warning: cannot determine system RAM; skipping the memory budget check",
              file=sys.stderr)
        return

    print(f"Memory plan: {jobs} jobs x {format_size(heap_bytes)} heap = "
          f"{format_size(requested)} (budget {format_size(budget)}, {source})")

    if requested <= budget:
        return

    safe_jobs = max(1, budget // heap_bytes)
    message = (
        f"refusing to start: {jobs} jobs x {format_size(heap_bytes)} heap = "
        f"{format_size(requested)} exceeds the budget of {format_size(budget)} ({source}).\n"
        f"  Either run --jobs {safe_jobs} at this heap, or lower --heap.\n"
        f"  Each JVM solves a POMDP per timestep; starving it of heap makes runs slower, "
        f"not smaller, so prefer fewer jobs over a smaller heap.\n"
        f"  Pass --force-memory to override."
    )
    if force:
        print("warning: " + message, file=sys.stderr)
        return
    raise SystemExit("error: " + message)


# ---------------------------------------------------------------------------
# Config rendering
# ---------------------------------------------------------------------------

def render_config(template_text: str, overrides: dict[str, str]) -> str:
    """Return the template with each override applied, replacing or appending as needed.

    Commented-out keys are left alone: this rewrites only live ``key=value`` lines, so the
    template's documentation survives into every cell config, which matters because those
    comments carry the lambda calibration table.
    """
    remaining = dict(overrides)
    lines_out: list[str] = []
    for line in template_text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                lines_out.append(f"{key}={remaining.pop(key)}")
                continue
        lines_out.append(line)
    if remaining:
        lines_out.append("")
        lines_out.append("############ SET BY run_ablation.py ############")
        for key, value in remaining.items():
            lines_out.append(f"{key}={value}")
    return "\n".join(lines_out) + "\n"


def cell_overrides(cell: Cell, output_dir_relative: str) -> dict[str, str]:
    overrides: dict[str, str] = {
        "outputDirectory": output_dir_relative,
        "runSeed": str(cell.seed),
    }
    if cell.is_baseline:
        # A baseline never consults the model, so the learning keys are irrelevant; pin
        # them to something inert so the config reads unambiguously.
        overrides.update({
            "fixedPolicy": cell.baseline,
            "lambda": "0.0",
            "staticModel": "false",
            "useSurpriseUpdating": "false",
        })
    else:
        overrides.update(LEARNING_CONFIGS[cell.learning])
        overrides["lambda"] = f"{cell.lam:g}"
        overrides["fixedPolicy"] = ""
    return overrides


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_cells(args: argparse.Namespace) -> list[Cell]:
    scenarios = parse_int_list(args.scenarios, ALL_SCENARIOS, "scenarios")
    seeds = parse_int_list(args.seeds, None, "seeds")
    baselines = parse_name_list(args.baselines, BASELINE_POLICIES, "baselines")
    excluded_arms, excluded_scenarios = parse_exclusions(args.exclude or "")

    if args.arms:
        arm_pairs = parse_arm_specs(args.arms, "arms")
    else:
        learnings = parse_name_list(args.learning, list(LEARNING_CONFIGS), "learning")
        lambdas = parse_float_list(args.lambdas, "lambdas")
        arm_pairs = [(lrn, lam) for lrn in learnings for lam in lambdas]

    arm_pairs = [p for p in arm_pairs if p not in excluded_arms]
    scenarios = [s for s in scenarios if s not in excluded_scenarios]

    cells: list[Cell] = []
    for scenario in scenarios:
        for learning, lam in arm_pairs:
            for seed in seeds:
                cells.append(Cell(
                    scenario=scenario, arm=arm_name(learning, lam), seed=seed,
                    timesteps=args.timesteps, crn=args.crn,
                    learning=learning, lam=lam))
        for policy in baselines:
            for seed in seeds:
                cells.append(Cell(
                    scenario=scenario, arm=f"fixed_{policy}", seed=seed,
                    timesteps=args.timesteps, crn=args.crn, baseline=policy))

    return order_cells(cells, args.order, args.seeds)


def order_cells(cells: list[Cell], order: str, seed_spec: str) -> list[Cell]:
    """Run order. Interleaving across scenarios means a partial grid still covers breadth.

    That matters when the grid is walked in batches: stopping after 200 of 980 cells should
    leave every scenario partially populated rather than S0-S1 complete and S2-S6 empty.
    """
    if order == "sequential":
        return cells
    if order == "random":
        # zlib.crc32, not hash(): Python randomises str hashes per process, so hash()
        # would give a different 'random' order on every invocation.
        rng = random.Random(zlib.crc32(seed_spec.encode("utf-8")))
        shuffled = list(cells)
        rng.shuffle(shuffled)
        return shuffled

    buckets: dict[int, list[Cell]] = {}
    for cell in cells:
        buckets.setdefault(cell.scenario, []).append(cell)
    out: list[Cell] = []
    longest = max(len(bucket) for bucket in buckets.values())
    for index in range(longest):
        for scenario in sorted(buckets):
            bucket = buckets[scenario]
            if index < len(bucket):
                out.append(bucket[index])
    return out


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"warning: could not read manifest {path} ({exc}); starting fresh",
              file=sys.stderr)
        return {}
    entries = data.get("cells", {})
    # A 'running' entry means a previous driver was killed mid-flight. That cell's output
    # may be half-written, so it must be re-run rather than trusted.
    reset = 0
    for entry in entries.values():
        if entry.get("status") == STATUS_RUNNING:
            entry["status"] = STATUS_PENDING
            reset += 1
    if reset:
        print(f"Reset {reset} stale 'running' cell(s) from a previous driver to pending")
    return entries


def save_manifest(path: Path, entries: dict[str, dict]) -> None:
    """Write the manifest atomically (temp file + replace), retrying on file locks.

    This repo lives in a OneDrive-synced folder, and on Windows ``os.replace`` raises
    PermissionError while anything - the sync client, an antivirus scan, an open editor -
    holds the destination. Crashing here would abandon the running JVMs mid-batch, so retry
    with backoff and, failing that, warn and carry on: every save rewrites the whole
    manifest, so the next successful one catches up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cells": entries,
    }
    temp = path.with_suffix(".json.tmp")
    delay = 0.1
    for _ in range(8):
        try:
            with temp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=1, sort_keys=True)
            temp.replace(path)
            return
        except PermissionError:
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
    print(f"warning: could not update {path} (file locked?); "
          f"will try again on the next save", file=sys.stderr)


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

# Set in main() from resolve_java(). Module-level so build_command stays a pure function of
# the cell.
JAVA_EXE = "java"


def resolve_java(override: str | None) -> str:
    """Return the path of the real JVM executable, bypassing launcher shims.

    On Windows, ``java`` on PATH is usually Oracle's ``javapath\\java.exe`` shim, which
    starts the real JDK java.exe as a CHILD process and waits for it. Killing the shim -
    which is what a timeout, --stop-on-failure or Ctrl+C does - leaves the real JVM running,
    holding its heap. The JVM reports its own home directory, so ask it and launch
    ``<java.home>/bin/java`` directly. kill_tree() is the backstop for anything else that
    wraps the JVM.
    """
    if override:
        if not Path(override).exists():
            raise SystemExit(f"--java: {override} does not exist")
        return override
    try:
        probe = subprocess.run(["java", "-XshowSettings:properties", "-version"],
                               capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SystemExit(f"error: could not run 'java' ({exc}); install a JDK or pass --java")
    for line in (probe.stderr + probe.stdout).splitlines():
        stripped = line.strip()
        if stripped.startswith("java.home"):
            home = stripped.split("=", 1)[1].strip()
            name = "java.exe" if platform.system() == "Windows" else "java"
            candidate = Path(home) / "bin" / name
            if candidate.exists():
                return str(candidate)
    print("warning: could not determine java.home; launching 'java' from PATH. "
          "Kills will still take the whole process tree.", file=sys.stderr)
    return "java"


def kill_tree(process: subprocess.Popen) -> None:
    """Kill a launched cell and everything it started."""
    if process.poll() is not None:
        return
    if platform.system() == "Windows":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    try:
        process.kill()
    except Exception:
        pass


def classpath() -> str:
    separator = ";" if platform.system() == "Windows" else ":"
    return separator.join(["bin", str(Path("libraries") / "*")])


def build_command(cell: Cell, config_path: Path, output_dir_relative: str,
                  heap: str) -> list[str]:
    return [
        JAVA_EXE,
        f"-Xmx{heap}",
        "-cp", classpath(),
        f"-DconfigPath={config_path.as_posix()}",
        f"-DoutputDirectory={output_dir_relative}",
        f"-Dscenario={cell.scenario}",
        f"-Dtimesteps={cell.timesteps}",
        f"-Dcrn={'true' if cell.crn else 'false'}",
        # charts.html is ~5 MB a run; across the grid that is several GB of plots nobody
        # opens. Cross-cell figures come from results.csv instead.
        "-Dcharts=false",
        "RDMSim_TAAS",
    ]


def start_rss_sampler(process: subprocess.Popen, cell: Cell,
                      interval: float = 2.0) -> threading.Thread | None:
    """Sample the JVM's RSS in a daemon thread, recording the peak on the cell.

    The pilot needs this number: peak RSS, not wall-clock, is what sets the real --jobs
    ceiling, and -Xmx bounds only the Java heap, not the JVM's total footprint. Requires
    psutil (``pip install psutil``); without it the column is left empty rather than
    guessed, and nothing else changes.
    """
    try:
        import psutil
    except ImportError:
        return None

    def sample() -> None:
        try:
            proc = psutil.Process(process.pid)
        except Exception:
            return
        peak = 0.0
        while process.poll() is None:
            try:
                rss = proc.memory_info().rss
            except Exception:
                break
            peak = max(peak, rss / (1024 ** 2))
            time.sleep(interval)
        if peak:
            cell.peak_rss_mb = peak

    # Returned so finish() can join it: otherwise the peak may land on the cell only after
    # its manifest entry has been written, and never be persisted.
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    return thread


def launch(cell: Cell, args: argparse.Namespace, template_text: str) -> subprocess.Popen:
    output_root = Path(args.output_root)
    cell_dir = cell.directory(output_root)
    cell_dir.mkdir(parents=True, exist_ok=True)

    # outputDirectory must be relative: configureDirectories() resolves it against the
    # working directory and rejects absolute paths.
    try:
        relative = cell_dir.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        raise SystemExit(
            f"error: --output-root must sit inside the project ({PROJECT_ROOT}); "
            f"RDMSim rejects an absolute outputDirectory")

    config_path = cell_dir / "solver.config"
    config_path.write_text(render_config(template_text, cell_overrides(cell, relative)),
                           encoding="utf-8")

    # Delete any stale summary before launching, so its presence afterwards always means
    # THIS run wrote it. This is load-bearing: runCaseRDM() catches RuntimeException and
    # only logs it, so a cell that fails on a bad config still exits 0. Without this, a
    # re-run (--force) over a previously successful cell could leave the old summary in
    # place and be scored as a success.
    stale_summary = cell_dir / "runSummary.txt"
    if stale_summary.exists():
        stale_summary.unlink()

    cell.command = build_command(cell, config_path, relative, args.heap)
    log_path = cell_dir / "run.log"
    log_handle = log_path.open("w", encoding="utf-8", errors="replace")
    process = subprocess.Popen(cell.command, cwd=PROJECT_ROOT, stdout=log_handle,
                               stderr=subprocess.STDOUT,
                               start_new_session=(platform.system() != "Windows"))
    process._log_handle = log_handle  # type: ignore[attr-defined]
    process._cell = cell             # type: ignore[attr-defined]
    process._started = time.time()   # type: ignore[attr-defined]
    process._rss_thread = start_rss_sampler(process, cell)  # type: ignore[attr-defined]
    return process


def keep_system_awake(enable: bool) -> bool:
    """Ask Windows not to idle-sleep while a batch runs. Returns True if the request held.

    A full grid is many hours; left alone, a laptop idles into (Modern) Standby, which
    suspends the JVMs and stalls the grid. ES_SYSTEM_REQUIRED blocks *idle* sleep only -
    closing the lid or choosing Sleep still sleeps, which run_grid() survives by not
    counting suspended time. No-op elsewhere; use caffeinate / systemd-inhibit there.
    """
    if platform.system() != "Windows":
        return False
    try:
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        flags = es_continuous | (es_system_required if enable else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except Exception:
        return False


def close_log(process: subprocess.Popen) -> None:
    try:
        process._log_handle.close()  # type: ignore[attr-defined]
    except Exception:
        pass


def run_grid(cells: list[Cell], args: argparse.Namespace, manifest: dict[str, dict],
             manifest_path: Path, template_text: str) -> bool:
    """Run the pending cells. Returns True if the batch was interrupted with Ctrl+C."""
    pending = [c for c in cells if c.status != STATUS_DONE]
    if args.batch_size is not None:
        pending = pending[: args.batch_size]
        print(f"Batch limited to {len(pending)} cell(s) by --batch-size")

    if not pending:
        print("Nothing to run: every selected cell is already marked done.")
        return False

    # Retries are a per-invocation budget. The manifest's attempt count is history: without
    # this reset, a cell that failed in an earlier sitting would start today already past
    # its retry limit and get a single attempt.
    for cell in pending:
        cell.attempts = 0

    started_at = time.time()
    queue = list(pending)
    active: list[subprocess.Popen] = []
    completed = 0
    failed = 0
    total = len(queue)
    stop_launching = False

    def finish(process: subprocess.Popen, status: str) -> None:
        nonlocal completed, failed
        cell: Cell = process._cell  # type: ignore[attr-defined]
        close_log(process)
        sampler = getattr(process, "_rss_thread", None)
        if sampler is not None:
            sampler.join(timeout=5)
        cell.wallclock_s = round(time.time() - process._started, 1)  # type: ignore[attr-defined]
        cell.measured_timesteps = cell.timesteps
        cell.status = status
        manifest[cell.key] = manifest_entry(cell)
        save_manifest(manifest_path, manifest)
        completed += 1
        if status == STATUS_DONE:
            print(f"  [{completed}/{total}] {cell.key} done in {cell.wallclock_s}s")
        else:
            failed += 1
            print(f"  [{completed}/{total}] {cell.key} {status.upper()} "
                  f"after {cell.wallclock_s}s (see {cell.directory(Path(args.output_root))/'run.log'})",
                  file=sys.stderr)

    def abandon(reason: str) -> None:
        """Kill everything in flight, return those cells to pending, and stop the batch.

        Used by Ctrl+C and --stop-on-failure. Killed cells must not be retried (the old
        loop requeued them, so --stop-on-failure never stopped) and must not stay marked
        'running' in the manifest.
        """
        if active:
            print(f"{reason}: stopping {len(active)} running cell(s); they stay pending",
                  file=sys.stderr)
        for process in active:
            kill_tree(process)
        for process in active:
            try:
                process.wait(timeout=30)
            except Exception:
                pass
            close_log(process)
            cell = process._cell  # type: ignore[attr-defined]
            cell.status = STATUS_PENDING
            manifest[cell.key] = manifest_entry(cell)
        # Cells waiting in the retry queue were marked 'running' when they last launched;
        # record them as pending too, or the next startup misreports them as stale.
        for cell in queue:
            cell.status = STATUS_PENDING
            manifest[cell.key] = manifest_entry(cell)
        active.clear()
        queue.clear()
        save_manifest(manifest_path, manifest)

    # A gap this long between two of our own 0.4 s polls means the machine was asleep:
    # this process was suspended along with the JVMs. Wall-clock kept running, so without
    # this correction a healthy cell would be killed by --timeout on wake (a 16-minute
    # 'static' cell after any sleep over ~45 min), and the sleep would be booked as its
    # run time and poison --dry-run's estimates.
    suspend_gap_s = 30.0
    last_tick = time.time()

    interrupted = False
    try:
        while queue or active:
            # Honour --max-wall by stopping new launches; in-flight cells are allowed to
            # finish so no partial output is left behind.
            if args.max_wall is not None and not stop_launching:
                if time.time() - started_at >= args.max_wall:
                    print(f"--max-wall reached; not launching further cells "
                          f"({len(queue)} left pending)")
                    stop_launching = True

            while queue and len(active) < args.jobs and not stop_launching:
                cell = queue.pop(0)
                cell.attempts += 1
                cell.status = STATUS_RUNNING
                manifest[cell.key] = manifest_entry(cell)
                save_manifest(manifest_path, manifest)
                print(f"-> {cell.key} (attempt {cell.attempts})")
                active.append(launch(cell, args, template_text))

            if not active:
                break

            time.sleep(0.4)

            # Suspend check goes HERE, between the sleep and the timeout checks: the driver
            # spends nearly all its time in the sleep above, so that is where a real suspend
            # lands, and the timeout checks below must see the corrected start times.
            now = time.time()
            gap = now - last_tick
            last_tick = now
            if gap > suspend_gap_s:
                for process in active:
                    process._started += gap  # type: ignore[attr-defined]
                started_at += gap
                print(f"  (system was suspended for ~{gap/60:.0f} min; that time is not "
                      f"counted against --timeout, --max-wall or cell timings)")

            for process in list(active):
                cell = process._cell  # type: ignore[attr-defined]
                elapsed = time.time() - process._started  # type: ignore[attr-defined]

                if process.poll() is None:
                    if args.timeout is not None and elapsed > args.timeout:
                        kill_tree(process)
                        process.wait()
                        active.remove(process)
                        if cell.attempts <= args.retries:
                            close_log(process)
                            print(f"  {cell.key} timed out after {int(elapsed)}s; requeueing")
                            cell.status = STATUS_PENDING
                            manifest[cell.key] = manifest_entry(cell)
                            queue.append(cell)
                        else:
                            finish(process, STATUS_TIMEOUT)
                    continue

                active.remove(process)
                if process.returncode == 0 and summary_path(cell, args).exists():
                    finish(process, STATUS_DONE)
                elif cell.attempts <= args.retries:
                    close_log(process)
                    print(f"  {cell.key} failed (exit {process.returncode}); requeueing")
                    cell.status = STATUS_PENDING
                    manifest[cell.key] = manifest_entry(cell)
                    queue.append(cell)
                else:
                    finish(process, STATUS_FAILED)
                    if args.stop_on_failure:
                        abandon("--stop-on-failure")
                        break
    except KeyboardInterrupt:
        interrupted = True
        abandon("Interrupted")

    elapsed = time.time() - started_at
    print(f"\nBatch {'interrupted' if interrupted else 'finished'} after {elapsed/60:.1f} min: "
          f"{completed - failed} done, {failed} failed/timed out.")
    return interrupted


def summary_path(cell: Cell, args: argparse.Namespace) -> Path:
    return cell.directory(Path(args.output_root)) / "runSummary.txt"


def manifest_entry(cell: Cell) -> dict:
    entry = asdict(cell)
    entry.pop("command", None)
    entry["command"] = " ".join(cell.command)
    return entry


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "scenario", "learning", "lambda", "baseline", "seed", "timesteps", "crn",
    "MON1_rate", "MON2_rate", "MON3_rate", "joint_rate",
    "MON1_belief", "MON2_belief", "MON3_belief",
    "rt_share", "mean_links", "mean_bw", "mean_ttw",
    "mean_abs_qgap", "mean_gamma", "state_change_rate",
    "wallclock_s", "peak_rss_mb", "status",
]


def read_summary(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "\t" in line:
                key, value = line.split("\t", 1)
                values[key.strip()] = value.strip()
    except OSError:
        return {}
    return values


def mean_of_column(path: Path, column: int) -> float | None:
    """Mean of a whitespace-separated column in a per-step log, or None if unavailable.

    Used for mean |Q-gap| (qvalues.txt col 4) and mean gamma (gamma.txt col 2). Both files
    are legitimately empty for some arms - baselines never solve, and the static-model arm
    never computes gamma - so an absent value is information, not an error.
    """
    if not path.exists():
        return None
    total = 0.0
    count = 0
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) <= column:
                continue
            try:
                total += float(parts[column])
                count += 1
            except ValueError:
                continue
    except OSError:
        return None
    return total / count if count else None


def read_flags(path: Path) -> dict[int, int] | None:
    """timestep -> satisfied flag from a MON{1,2,3}Sat.txt ("timestep value flag")."""
    if not path.exists():
        return None
    flags: dict[int, int] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                flags[int(float(parts[0]))] = int(float(parts[2]))
            except ValueError:
                continue
    except OSError:
        return None
    return flags or None


def joint_rate(directory: Path) -> float | None:
    """Fraction of timesteps at which MON1, MON2 and MON3 are all satisfied at once.

    Not the product of the per-NFR rates: within a step the three are negatively
    correlated (more active links helps MON1 and hurts MON2/MON3), so the joint rate sits
    well below the product. Computed from the per-step flags, joined on timestep; None
    when any of the three logs is missing (e.g. pruned with --keep-logs summary).
    """
    per_nfr = [read_flags(directory / f"MON{i}Sat.txt") for i in (1, 2, 3)]
    if any(flags is None for flags in per_nfr):
        return None
    common = set(per_nfr[0]) & set(per_nfr[1]) & set(per_nfr[2])
    if not common:
        return None
    hits = sum(1 for t in common if all(flags[t] == 1 for flags in per_nfr))
    return hits / len(common)


def cell_from_manifest(key: str, entry: dict) -> Cell | None:
    """Rebuild a Cell from a manifest entry, for cells outside the current selection."""
    try:
        return Cell(
            scenario=int(entry["scenario"]), arm=entry["arm"], seed=int(entry["seed"]),
            timesteps=int(entry.get("timesteps", 0)), crn=bool(entry.get("crn", True)),
            learning=entry.get("learning"), lam=entry.get("lam"),
            baseline=entry.get("baseline"), status=entry.get("status", STATUS_PENDING),
            wallclock_s=entry.get("wallclock_s"),
            measured_timesteps=entry.get("timesteps"),
            peak_rss_mb=entry.get("peak_rss_mb"))
    except (KeyError, TypeError, ValueError):
        print(f"warning: manifest entry '{key}' is malformed; skipped in aggregation",
              file=sys.stderr)
        return None


def aggregate(cells: list[Cell], args: argparse.Namespace,
              manifest: dict[str, dict] | None = None) -> int:
    """Rebuild the CSV from every cell the manifest knows about, not just this selection.

    The CSV is rewritten whole each time, so it has to cover everything on disk: otherwise
    running one narrow slice would silently drop the rows from every previous slice. Cells
    in the current selection win over their manifest entry, since they carry fresh timings.
    """
    merged: dict[str, Cell] = {}
    for key, entry in (manifest or {}).items():
        rebuilt = cell_from_manifest(key, entry)
        if rebuilt is not None:
            merged[key] = rebuilt
    for cell in cells:
        merged[cell.key] = cell

    rows: list[dict] = []
    for cell in sorted(merged.values(), key=lambda c: (c.scenario, arm_sort_key(c.arm), c.seed)):
        directory = cell.directory(Path(args.output_root))
        summary = read_summary(directory / "runSummary.txt")
        if not summary:
            continue
        rows.append({
            "scenario": summary.get("scenario", cell.scenario),
            "learning": cell.learning or "",
            "lambda": f"{cell.lam:g}" if cell.lam is not None else "",
            "baseline": cell.baseline or "",
            "seed": summary.get("runSeed", cell.seed),
            "timesteps": summary.get("timesteps", cell.timesteps),
            "crn": summary.get("crn", str(cell.crn).lower()),
            "MON1_rate": summary.get("MON1_satisfaction_rate", ""),
            "MON2_rate": summary.get("MON2_satisfaction_rate", ""),
            "MON3_rate": summary.get("MON3_satisfaction_rate", ""),
            "joint_rate": fmt(joint_rate(directory)),
            "MON1_belief": summary.get("MON1_mean_belief_prob", ""),
            "MON2_belief": summary.get("MON2_mean_belief_prob", ""),
            "MON3_belief": summary.get("MON3_mean_belief_prob", ""),
            "rt_share": summary.get("action_RT_share", ""),
            "mean_links": summary.get("mean_active_links", ""),
            "mean_bw": summary.get("mean_bandwidth_consumption", ""),
            "mean_ttw": summary.get("mean_time_to_write", ""),
            "mean_abs_qgap": fmt(mean_of_column(directory / "qvalues.txt", 3)),
            "mean_gamma": fmt(mean_of_column(directory / "gamma.txt", 1)),
            "state_change_rate": summary.get("state_change_rate", ""),
            "wallclock_s": "" if cell.wallclock_s is None else cell.wallclock_s,
            "peak_rss_mb": "" if cell.peak_rss_mb is None else round(cell.peak_rss_mb, 1),
            "status": cell.status,
        })

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} row(s) to {csv_path}")
    return len(rows)


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.6g}"


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def print_listing(cells: list[Cell], args: argparse.Namespace, estimate: bool) -> None:
    by_status: dict[str, int] = {}
    for cell in cells:
        by_status[cell.status] = by_status.get(cell.status, 0) + 1

    print(f"{len(cells)} cell(s) selected")
    for status in (STATUS_DONE, STATUS_PENDING, STATUS_FAILED, STATUS_TIMEOUT,
                   STATUS_RUNNING):
        if by_status.get(status):
            print(f"  {status}: {by_status[status]}")

    solver_cells = [c for c in cells if not c.is_baseline]
    baseline_cells = [c for c in cells if c.is_baseline]
    print(f"  solver arms: {len(solver_cells)}, baseline arms: {len(baseline_cells)}")

    if args.verbose:
        for cell in cells:
            print(f"    {cell.status:8s} {cell.key}")

    if not estimate:
        return

    # Estimate per ARM, not globally. Arms differ by more than an order of magnitude: a
    # learned model sharpens and collapses the value function to 1-3 alpha vectors, while
    # the frozen-prior 'static' arm carries ~11, and Perseus' cost scales with that. A
    # single global mean would badly misprice any selection whose arm mix differs from the
    # cells already measured.
    # Only measurements taken at THIS horizon are comparable: run time is superlinear in
    # the number of timesteps, so a 200-step timing would badly underprice a 1000-step cell.
    measured_by_arm: dict[str, list[float]] = defaultdict(list)
    for cell in cells:
        if cell.wallclock_s and cell.measured_timesteps == args.timesteps:
            measured_by_arm[cell.arm].append(cell.wallclock_s)

    # Nominal rates, measured on scenario 3 at 1000 steps (see docs/ABLATION.md):
    # smile-MIP lam=0 122 s, lam=5 70 s, static lam=0 962 s, fixed_mst 13 s. The static
    # arm is the expensive one because a frozen prior keeps ~11 alpha vectors where a
    # learned model collapses to ~2, and Perseus' cost scales with that.
    nominal_solver = 120.0 * (args.timesteps / 1000.0)
    nominal_static = 960.0 * (args.timesteps / 1000.0)
    nominal_baseline = 13.0 * (args.timesteps / 1000.0)

    def per_cell_estimate(cell: Cell) -> tuple[float, bool]:
        samples = measured_by_arm.get(cell.arm)
        if samples:
            # Median: robust to an occasional outlier (e.g. a cell that spanned a sleep
            # before suspend correction existed, or a machine under other load).
            return statistics.median(samples), True
        if cell.is_baseline:
            return nominal_baseline, False
        if cell.learning == "static":
            return nominal_static, False
        return nominal_solver, False

    remaining = [c for c in cells if c.status != STATUS_DONE]
    serial_s = 0.0
    measured_arms = 0
    for cell in remaining:
        seconds, from_measurement = per_cell_estimate(cell)
        serial_s += seconds
        measured_arms += 1 if from_measurement else 0

    remaining_solver = len([c for c in remaining if not c.is_baseline])
    remaining_baseline = len(remaining) - remaining_solver
    parallel_s = serial_s / max(1, args.jobs)

    print(f"\nEstimate ({measured_arms}/{len(remaining)} cell(s) priced from measured "
          f"timings for their own arm, the rest from nominal rates):")
    print(f"  {remaining_solver} solver + {remaining_baseline} baseline cell(s) remaining")
    print(f"  serial ~{serial_s/3600:.1f} h; at --jobs {args.jobs} ~{parallel_s/3600:.1f} h")
    print(f"  peak memory ~{format_size(args.jobs * parse_size(args.heap))}")
    if measured_arms < len(remaining):
        print("  Nominal rates: ~2 min per 1000 steps for a learning arm, ~16 min for a")
        print("  'static' arm; run a pilot to replace them with measurements.")

    slowest = sorted(
        ((statistics.median(v), arm) for arm, v in measured_by_arm.items()), reverse=True)[:3]
    if slowest:
        print("  slowest measured arms: "
              + ", ".join(f"{arm} {seconds:.0f}s" for seconds, arm in slowest))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_ablation.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run a selected subset of the RDMSim ablation grid.",
        epilog="""\
examples:
  # smoke test one cell, short horizon
  run_ablation.py --scenarios 3 --arms smile-MIP:0 --baselines none --seeds 222 --timesteps 50

  # the scenario-3 pilot: all 28 arms, 5 seeds
  run_ablation.py --scenarios 3 --seeds 222-226 --jobs 4 --heap 1g

  # low-memory machine
  run_ablation.py --jobs 2 --heap 768m

  # walk the full grid in sittings
  run_ablation.py --batch-size 60        # repeat until --list shows nothing pending

  # re-run one suspect cell
  run_ablation.py --scenarios 4 --arms bayes:10 --seeds 224 --force

  # see what would run, and what it would cost
  run_ablation.py --dry-run

See docs/ABLATION.md for the full guide.""")

    selection = parser.add_argument_group("cell selection")
    selection.add_argument("--scenarios", default="all",
                           help="scenarios to run: all, 3, 0,3, 5-6 (default: all)")
    selection.add_argument("--learning", default="all",
                           help=f"learning configs: all or {','.join(LEARNING_CONFIGS)} "
                                "(default: all)")
    selection.add_argument("--lambdas", default=",".join(f"{x:g}" for x in DEFAULT_LAMBDAS),
                           help="lambda values for the cross (default: 0,2,5,10,20)")
    selection.add_argument("--baselines", default="all",
                           help=f"fixed-policy arms: all, none, or {','.join(BASELINE_POLICIES)} "
                                "(default: all)")
    selection.add_argument("--arms", default="",
                           help="explicit learning:lambda pairs, e.g. 'smile-MIP:5,bayes:0'; "
                                "overrides --learning/--lambdas")
    selection.add_argument("--exclude", default="",
                           help="subtract arms or scenarios, e.g. 'static:20,scenario=0'")
    selection.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                           help="seeds: 222-226 or 222,223 (default: 222-226)")
    selection.add_argument("--timesteps", type=int, default=1000,
                           help="horizon per run (default: 1000)")
    crn_group = selection.add_mutually_exclusive_group()
    crn_group.add_argument("--crn", dest="crn", action="store_true", default=True,
                           help="use common random numbers so arms are paired (default)")
    crn_group.add_argument("--no-crn", dest="crn", action="store_false",
                           help="let RDMSim draw its own environment (confirmation runs)")

    resources = parser.add_argument_group("resource bounds")
    resources.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                           help="concurrent JVMs (default: half the cores)")
    resources.add_argument("--heap", default="1g",
                           help="per-JVM -Xmx, e.g. 1g or 768m (default: 1g)")
    resources.add_argument("--max-memory", default=None,
                           help="memory budget for jobs x heap (default: 75%% of RAM)")
    resources.add_argument("--force-memory", action="store_true",
                           help="warn instead of refusing when over the memory budget")
    resources.add_argument("--batch-size", type=int, default=None,
                           help="run at most this many cells, then exit resumably")
    resources.add_argument("--max-wall", type=parse_duration, default=None,
                           help="stop launching after this long, e.g. 6h or 90m")
    resources.add_argument("--timeout", type=parse_duration, default=3600,
                           help="per-cell kill after this long (default: 3600s)")
    resources.add_argument("--retries", type=int, default=1,
                           help="relaunch a failed cell this many times (default: 1)")
    resources.add_argument("--allow-sleep", action="store_true",
                           help="do not ask Windows to stay awake while the batch runs")
    resources.add_argument("--stop-on-failure", action="store_true",
                           help="abandon the batch on the first unrecoverable cell")

    bookkeeping = parser.add_argument_group("inspection and bookkeeping")
    bookkeeping.add_argument("--list", action="store_true",
                             help="print the selected cells and their status, run nothing")
    bookkeeping.add_argument("--dry-run", action="store_true",
                             help="as --list, plus a wall-clock and memory estimate")
    bookkeeping.add_argument("--force", action="store_true",
                             help="re-run cells already marked done (default: skip them)")
    bookkeeping.add_argument("--aggregate-only", action="store_true",
                             help="rebuild the CSV from cells on disk, launching nothing")
    bookkeeping.add_argument("--order", choices=("interleaved", "sequential", "random"),
                             default="interleaved",
                             help="run order; interleaved rotates across scenarios (default)")
    bookkeeping.add_argument("--keep-logs", choices=("all", "summary", "none"), default="all",
                             help="per-step logs to retain per cell (default: all)")
    bookkeeping.add_argument("--output-root", default=str(PROJECT_ROOT / "experiments" / "cells"),
                             help="where cell directories live")
    bookkeeping.add_argument("--csv", default=str(PROJECT_ROOT / "experiments" / "results.csv"),
                             help="aggregated results CSV")
    bookkeeping.add_argument("--manifest", default=str(PROJECT_ROOT / "experiments" / "manifest.json"),
                             help="per-cell status manifest")
    bookkeeping.add_argument("--java", default=None,
                             help="path to the java executable (default: the real JVM behind "
                                  "'java' on PATH, resolved via java.home)")
    bookkeeping.add_argument("--template", default=str(DEFAULT_TEMPLATE),
                             help="solver.config to use as the per-cell template")
    bookkeeping.add_argument("--verbose", "-v", action="store_true",
                             help="list every cell, not just the counts")

    return parser


def parse_duration(text: str) -> float:
    """Accept bare seconds or a suffixed duration: 90, 90s, 30m, 6h."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smh]?)", str(text).strip(), re.IGNORECASE)
    if not match:
        raise argparse.ArgumentTypeError(f"'{text}' is not a duration like 600, 30m or 6h")
    value = float(match.group(1))
    return value * {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2).lower()]


def prune_logs(cells: list[Cell], args: argparse.Namespace) -> None:
    """Apply --keep-logs after the batch.

    Only cells marked done are pruned, so a failed cell keeps everything needed to diagnose
    it. runSummary.txt and solver.config are never deleted: the first is what the CSV and
    resume logic read, the second is the record of what the cell actually ran.

      all      keep every per-step log (default; ~1-2 MB per cell)
      summary  delete per-step logs, keep runSummary.txt, solver.config and run.log
      none     as summary, and delete run.log too
    """
    if args.keep_logs == "all":
        return

    always_keep = {"runSummary.txt", "solver.config"}
    if args.keep_logs == "summary":
        always_keep.add("run.log")

    removed = 0
    for cell in cells:
        if cell.status != STATUS_DONE:
            continue
        directory = cell.directory(Path(args.output_root))
        if not directory.is_dir():
            continue
        for entry in directory.iterdir():
            if entry.name in always_keep:
                continue
            if entry.is_file():
                entry.unlink()
                removed += 1
            elif entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
    if removed:
        print(f"--keep-logs {args.keep_logs}: removed {removed} file(s) from completed cells")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.retries < 0:
        raise SystemExit("--retries must be >= 0")

    template_path = Path(args.template)
    if not template_path.exists():
        raise SystemExit(f"template config not found: {template_path}")
    template_text = template_path.read_text(encoding="utf-8")

    if not (PROJECT_ROOT / "bin" / "RDMSim_TAAS.class").exists():
        print("warning: bin/RDMSim_TAAS.class not found - build first:\n"
              '  javac -d bin -cp "libraries/*" $(find src -name "*.java")',
              file=sys.stderr)

    cells = build_cells(args)
    if not cells:
        raise SystemExit("no cells selected; check --scenarios/--learning/--lambdas/--exclude")

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    stale_horizon: list[tuple[str, int]] = []
    for cell in cells:
        entry = manifest.get(cell.key)
        if entry:
            cell.status = entry.get("status", STATUS_PENDING)
            cell.wallclock_s = entry.get("wallclock_s")
            cell.measured_timesteps = entry.get("timesteps")
            cell.peak_rss_mb = entry.get("peak_rss_mb")
            cell.attempts = entry.get("attempts", 0) if not args.force else 0
        # A cell completed at a different horizon is not this cell: the satisfaction rates
        # are means over timesteps, so a 200-step result cannot stand in for a 1000-step
        # one. Without this check, a short smoke run would silently leave holes in a later
        # full-horizon grid.
        if (cell.status == STATUS_DONE and cell.measured_timesteps is not None
                and cell.measured_timesteps != args.timesteps):
            stale_horizon.append((cell.key, cell.measured_timesteps))
            cell.status = STATUS_PENDING
        if args.force:
            cell.status = STATUS_PENDING

    if stale_horizon:
        print(f"{len(stale_horizon)} cell(s) were completed at a different horizon than "
              f"--timesteps {args.timesteps} and will be re-run:")
        for key, horizon in stale_horizon[:5]:
            print(f"  {key} (was {horizon} steps)")
        if len(stale_horizon) > 5:
            print(f"  ... and {len(stale_horizon) - 5} more")

    if args.list or args.dry_run:
        print_listing(cells, args, estimate=args.dry_run)
        return 0

    if args.aggregate_only:
        aggregate(cells, args, manifest)
        return 0

    check_memory_budget(args.jobs, parse_size(args.heap), args.max_memory,
                        args.force_memory)

    global JAVA_EXE
    JAVA_EXE = resolve_java(args.java)
    print(f"JVM: {JAVA_EXE}")

    if not args.allow_sleep and keep_system_awake(True):
        print("Asked Windows not to idle-sleep while the batch runs "
              "(closing the lid still sleeps; --allow-sleep to disable)")
    try:
        interrupted = run_grid(cells, args, manifest, manifest_path, template_text)
    finally:
        if not args.allow_sleep:
            keep_system_awake(False)
    prune_logs(cells, args)
    aggregate(cells, args, manifest)
    if interrupted:
        print("Completed cells are saved. Re-run the same command to resume.")
        return 130

    failures = [c for c in cells if c.status in (STATUS_FAILED, STATUS_TIMEOUT)]
    if failures:
        print(f"\n{len(failures)} cell(s) need attention:", file=sys.stderr)
        for cell in failures[:10]:
            print(f"  {cell.status}: {cell.key}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
