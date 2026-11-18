"""Loading: NFR thresholds, single-run logs, the ablation grid and per-step series.

Thresholds come from ``config_log_files/configuration.json``, the same file RDMSim reads,
converted to absolute values exactly as RDMSim's ``NetworkProperties.loadNetworkSettings()``
does. Nothing here hard-codes a threshold value.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import pickle
import sys
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config_log_files" / "configuration.json"
DEFAULT_CSV = PROJECT_ROOT / "experiments" / "results.csv"
DEFAULT_CELLS = PROJECT_ROOT / "experiments" / "cells"
TIMESERIES_CACHE = PROJECT_ROOT / "experiments" / "analysis" / "timeseries_cache.pkl"

# Per-link capacity factors RDMSim multiplies the percentage thresholds by. They are
# constants inside the RDMSim jar (NetworkProperties), not configuration; the same values
# appear in src/pomdp/POMDP.java:322-325 (number_of_links*30, number_of_links*20).
BANDWIDTH_PER_LINK = 30.0     # Gbps per link
WRITE_TIME_PER_LINK = 20.0    # ms per link


# ---------------------------------------------------------------------------
# Reuse experiments/analyse.py and run_ablation.py rather than duplicating them
# ---------------------------------------------------------------------------

def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered before executing: dataclasses resolve string annotations through
    # sys.modules[cls.__module__], which would otherwise be missing.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


analyse = _load_module("analyse", PROJECT_ROOT / "experiments" / "analyse.py")
run_ablation = _load_module("run_ablation", PROJECT_ROOT / "experiments" / "run_ablation.py")


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Thresholds:
    """Absolute NFR thresholds. MON1 is satisfied at >=, MON2 and MON3 at <=."""

    links: float        # MON1, active links
    bandwidth: float    # MON2, bandwidth consumption (Gbps)
    write_time: float   # MON3, time to write (ms)
    # Network capacity each threshold is a percentage of (300 links, 9000 Gbps, 6000 ms for
    # 25 mirrors), so titles can state "105 (35% of 300 links)".
    capacity: tuple[float, float, float] = (math.nan, math.nan, math.nan)

    def for_nfr(self, index: int) -> float:
        return (self.links, self.bandwidth, self.write_time)[index]

    def percent(self, index: int) -> float:
        """Threshold as a percentage of capacity (what configuration.json specifies)."""
        cap = self.capacity[index]
        return 100.0 * self.for_nfr(index) / cap if cap and math.isfinite(cap) else math.nan


def thresholds_from_config(config_path: str | Path | None = None,
                           overrides: tuple[float | None, float | None, float | None] = (None, None, None)
                           ) -> Thresholds:
    """Absolute thresholds from configuration.json, as RDMSim computes them.

    number_of_links = m(m-1)/2 for m = mirror_number; then
      MON1 = link_threshold% of number_of_links
      MON2 = bandwidth_threshold% of number_of_links * 30
      MON3 = writing_times_threshold% of number_of_links * 20
    An explicit override (the --mon*-threshold flags RDMSim_TAAS.java passes) wins.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"RDMSim configuration not found: {path}")
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    try:
        mirrors = int(config["mirror_number"])
        link_pct = float(config["link_threshold"])
        bandwidth_pct = float(config["bandwidth_threshold"])
        write_pct = float(config["writing_times_threshold"])
    except KeyError as exc:
        raise KeyError(f"{path} is missing {exc}; cannot derive NFR thresholds") from exc

    links_total = mirrors * (mirrors - 1) / 2
    capacity = (links_total, links_total * BANDWIDTH_PER_LINK, links_total * WRITE_TIME_PER_LINK)
    derived = [pct / 100.0 * cap for pct, cap in zip((link_pct, bandwidth_pct, write_pct), capacity)]
    final = [o if o is not None else d for o, d in zip(overrides, derived)]
    for name, o, d in zip(("MON1", "MON2", "MON3"), overrides, derived):
        if o is not None and abs(o - d) > 1e-6:
            logger.warning("{} threshold overridden to {} (configuration.json gives {})", name, o, d)
    return Thresholds(*final, capacity=capacity)


def check_run_thresholds(run_dir: str | Path, thresholds: Thresholds) -> bool:
    """Warn if a run recorded different thresholds than the config now gives.

    That happens when configuration.json is edited after the runs: the plotted line would
    then not be the threshold those runs were scored against. Never changes the value.
    """
    summary = Path(run_dir) / "runSummary.txt"
    if not summary.exists():
        return True
    recorded = {}
    for line in summary.read_text(encoding="utf-8", errors="replace").splitlines():
        if "\t" in line:
            key, value = line.split("\t", 1)
            recorded[key.strip()] = value.strip()
    pairs = [("threshold_active_links", thresholds.links),
             ("threshold_bandwidth_consumption", thresholds.bandwidth),
             ("threshold_time_to_write", thresholds.write_time)]
    ok = True
    for key, expected in pairs:
        if key in recorded:
            try:
                value = float(recorded[key])
            except ValueError:
                continue
            if abs(value - expected) > 0.01:
                ok = False
                logger.warning(
                    "THRESHOLD MISMATCH in {}: run recorded {}={} but configuration.json gives {}. "
                    "Was the config edited after this run?", summary, key, value, expected)
    return ok


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

# Files read_run_dir() should not merge into the per-timestep frame: wide or categorical
# files with their own shape, the solver dump, and this script's own output.
SKIP_FILES = {"state_transitions.txt", "monitorables.txt", "charts.html", "runSummary.txt",
              "qvalues.txt"}


def read_run_dir(folder_path: str | Path) -> pd.DataFrame:
    """Merge every per-step output file of one run into a single per-timestep frame.

    Each file's role is inferred from its column count and name (the convention
    RDMSim_TAAS.java logs with): 2-column files are "timestep value"; MIPBounds.txt is
    "timestep lower upper"; MON{1,2,3}Sat.txt are "timestep measuredValue satisfiedFlag".
    """
    folder_path = os.path.abspath(folder_path)
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Output directory not found: {folder_path}")

    logger.info("Stage: Scanning {} for monitorable/surprise/gamma output files", folder_path)
    frames = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path) or filename in SKIP_FILES or not filename.endswith(".txt"):
            continue
        if os.path.getsize(file_path) == 0:
            logger.warning("Skipping empty file: {}", filename)
            continue
        try:
            df = pd.read_csv(file_path, sep=r"\s+", header=None, on_bad_lines="skip")
        except Exception as e:  # malformed or unreadable: report and carry on
            logger.warning("Skipping file {} due to error: {}", filename, e)
            continue
        if df.empty:
            continue

        name = filename.split(".")[0].lower()
        if df.shape[1] == 3 and name == "mipbounds":
            df.columns = ["timestep", "mip_lower", "mip_upper"]
        elif df.shape[1] == 3 and name.endswith("sat"):
            df.columns = ["timestep", name, name + "flag"]
        elif df.shape[1] == 2:
            df.columns = ["timestep", name]
        else:
            logger.debug("Skipping file {} with unexpected shape {}", filename, df.shape)
            continue
        logger.info("Read {} ({} rows)", filename, len(df))
        frames.append(df)

    if not frames:
        raise ValueError(f"No usable monitorable output files found in {folder_path}")

    longest = max(len(df) for df in frames)
    merged = reduce(lambda left, right: pd.merge(left, right, on="timestep"), frames)
    logger.info("Stage: Merged {} files into a single dataframe ({} rows)", len(frames), len(merged))
    if len(merged) < longest:
        logger.warning(
            "Merged dataframe ({} rows) is shorter than the longest input file ({} rows) - "
            "the inner join on 'timestep' dropped rows. Check for a stray/short file in {}",
            len(merged), longest, folder_path)
    return merged


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def load_grid(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Completed cells of results.csv as a DataFrame, using analyse.load_rows' parsing.

    Adds ``lam`` (float, NaN for baselines) and keeps analyse's ``arm`` label
    ("smile-MIP lam=5", "fixed-mst").
    """
    rows = analyse.load_rows(Path(csv_path) if csv_path else DEFAULT_CSV)
    grid = pd.DataFrame(rows)
    grid["lam"] = pd.to_numeric(grid["lambda"], errors="coerce")
    grid["baseline"] = grid["baseline"].fillna("")
    grid["learning"] = grid["learning"].fillna("")
    return grid


def cell_dir(cells_root: Path, row) -> Path:
    """Directory of one cell, named exactly as run_ablation.Cell.directory() names it."""
    if row["baseline"]:
        arm = f"fixed_{row['baseline']}"
    else:
        arm = run_ablation.arm_name(row["learning"], float(row["lam"]))
    return Path(cells_root) / f"s{int(row['scenario'])}" / arm / f"seed{int(row['seed'])}"


def scenario_descriptions(config_path: str | Path | None = None) -> dict[int, str]:
    """Short label per scenario from its non-zero deviation bands in configuration.json.

    e.g. 3 -> "MST links; RT bw+ttw". Derived, so it stays right if a scenario changes.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    short = {"links": "links", "bandwidth_consumption": "bw", "writing_time": "ttw"}
    out: dict[int, str] = {}
    for scenario in range(7):
        moved = {}
        for topology in ("mst", "rt"):
            moved[topology] = "+".join(
                short[m] for m in short
                if any(config.get(f"deviation_scenario_{scenario}_{topology}_{m}", [0, 0])))
        if moved["mst"] and moved["mst"] == moved["rt"]:
            out[scenario] = f"MST & RT {moved['mst']}"
        else:
            parts = [f"{t.upper()} {m}" for t, m in moved.items() if m]
            out[scenario] = "; ".join(parts) if parts else "no deviation"
    return out


# ---------------------------------------------------------------------------
# Per-step series for every cell (time series and violins), cached
# ---------------------------------------------------------------------------

# file -> columns it contributes. MON*Sat give the measured value and the satisfied flag.
SERIES_FILES = {
    "MON1Sat.txt": ["mon1", "mon1flag"],
    "MON2Sat.txt": ["mon2", "mon2flag"],
    "MON3Sat.txt": ["mon3", "mon3flag"],
    "SelectedAction.txt": ["action"],
    "gamma.txt": ["gamma"],
    "surpriseBF.txt": ["surprisebf"],
    "surpriseCC.txt": ["surprisecc"],
    "surpriseMIP.txt": ["surprisemip"],
}


def _read_series(path: Path, columns: list[str], horizon: int) -> dict[str, np.ndarray]:
    """Columns of one per-step file, placed on a 0..horizon-1 timestep grid (NaN = absent)."""
    out = {c: np.full(horizon, np.nan) for c in columns}
    if not path.exists() or path.stat().st_size == 0:
        return out
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None)
    except Exception:
        return out
    if df.shape[1] < len(columns) + 1:
        return out
    t = pd.to_numeric(df[0], errors="coerce").to_numpy()
    keep = np.isfinite(t) & (t >= 0) & (t < horizon)
    idx = t[keep].astype(int)
    for offset, column in enumerate(columns, start=1):
        out[column][idx] = pd.to_numeric(df[offset], errors="coerce").to_numpy()[keep]
    return out


def load_timeseries(grid: pd.DataFrame, cells_root: str | Path | None = None,
                    use_cache: bool = True) -> dict[tuple[int, str, int], pd.DataFrame]:
    """Per-step frame for every cell in ``grid``, keyed by (scenario, arm, seed).

    Reading ~8 files x 980 cells is minutes on this disk, so the result is pickled and
    reused while the set of cells and their runSummary.txt mtimes are unchanged.
    """
    root = Path(cells_root) if cells_root else DEFAULT_CELLS
    directories = {(int(r["scenario"]), r["arm"], int(r["seed"])): (cell_dir(root, r), int(r["timesteps"]))
                   for _, r in grid.iterrows()}
    signature = tuple(sorted(
        (key, (d / "runSummary.txt").stat().st_mtime if (d / "runSummary.txt").exists() else 0.0)
        for key, (d, _) in directories.items()))

    if use_cache and TIMESERIES_CACHE.exists():
        try:
            with TIMESERIES_CACHE.open("rb") as handle:
                cached = pickle.load(handle)
            if cached.get("signature") == signature:
                logger.info("Stage: per-step series loaded from cache ({} cells)", len(cached["data"]))
                return cached["data"]
        except Exception as exc:
            logger.warning("Ignoring unreadable cache {} ({})", TIMESERIES_CACHE, exc)

    logger.info("Stage: reading per-step logs of {} cells (cached afterwards)", len(directories))
    data: dict[tuple[int, str, int], pd.DataFrame] = {}
    for count, (key, (directory, horizon)) in enumerate(sorted(directories.items()), start=1):
        columns: dict[str, np.ndarray] = {}
        for filename, names in SERIES_FILES.items():
            columns.update(_read_series(directory / filename, names, horizon))
        frame = pd.DataFrame(columns)
        frame.index.name = "timestep"
        data[key] = frame
        if count % 100 == 0:
            logger.info("  {}/{} cells read", count, len(directories))

    TIMESERIES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with TIMESERIES_CACHE.open("wb") as handle:
        pickle.dump({"signature": signature, "data": data}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return data
