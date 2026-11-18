#!/usr/bin/env python3
"""Analysis for the RDMSim ablation grid.

Everything here is derived from ``experiments/results.csv`` alone, so analysis never
re-runs the grid and works on a partial grid. ``createCharts.py`` stays what it is - a
single-run inspector; this script is the cross-cell view.

Outputs (to ``experiments/analysis/`` by default):

  per_scenario_<N>.txt   one table per scenario, every arm, mean +/- sd over seeds
  mixture_line.txt       each arm's position on the MST-RT line, with the residual that
                         says whether the run is consistent with the generative model
  lambda_response.txt    MON rates and RT share against lambda, per learning config
  paired_tests.txt       paired comparisons across seeds (valid because CRN pairs arms)
  summary.md             the above, assembled, for pasting into notes

Why the mixture line matters: RDMSim redraws its monitorables from scratch every timestep,
so P(s'|s,a) = P(s'|a) and any policy's satisfaction rate must be a convex combination of
the two per-action rates, weighted by its RT share. An arm that sits off that line is a
bug, not a discovery - which makes the residual column a continuous correctness check on
the whole pipeline.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import statistics
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Analytic per-action NFR satisfaction rates, by scenario, derived from
# Monitorables.generateMonitorables (400k-draw Monte-Carlo, +/-0.002). Verified against the
# simulator by remotemirroring.CRNVerify and by the fixed-policy arms themselves.
# scenario -> action -> (MON1, MON2, MON3, joint)
TRUTH = {
    0: {"mst": (0.464, 0.811, 0.929, 0.254), "rt": (1.000, 0.068, 0.393, 0.045)},
    1: {"mst": (0.374, 0.903, 0.976, 0.266), "rt": (1.000, 0.069, 0.394, 0.046)},
    2: {"mst": (0.464, 0.811, 0.929, 0.256), "rt": (1.000, 0.016, 0.269, 0.009)},
    3: {"mst": (0.374, 0.903, 0.976, 0.266), "rt": (1.000, 0.016, 0.269, 0.009)},
    4: {"mst": (0.374, 0.823, 0.937, 0.178), "rt": (1.000, 0.068, 0.392, 0.045)},
    5: {"mst": (0.464, 0.812, 0.929, 0.256), "rt": (1.000, 0.078, 0.413, 0.053)},
    6: {"mst": (0.373, 0.824, 0.936, 0.177), "rt": (1.000, 0.079, 0.414, 0.055)},
}

METRICS = ["MON1_rate", "MON2_rate", "MON3_rate", "rt_share"]
LEARNING_ORDER = ["static", "bayes", "smile-CC", "smile-BF", "smile-MIP"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found; run run_ablation.py (or --aggregate-only) first")
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status") == "done"]
    if not rows:
        raise SystemExit(f"{csv_path} has no completed rows")
    for row in rows:
        row["scenario"] = int(row["scenario"])
        row["seed"] = int(row["seed"])
        for key in METRICS + ["mean_links", "mean_bw", "mean_ttw", "mean_abs_qgap",
                              "mean_gamma", "state_change_rate", "wallclock_s"]:
            row[key] = to_float(row.get(key))
        row["arm"] = arm_label(row)
    return rows


def to_float(text) -> float | None:
    if text is None or text == "":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def arm_label(row: dict) -> str:
    if row.get("baseline"):
        return f"fixed-{row['baseline']}"
    return f"{row['learning']} lam={row['lambda']}"


def arm_sort_key(label: str) -> tuple:
    if label.startswith("fixed-"):
        return (0, label, 0.0)
    learning, _, lam = label.partition(" lam=")
    index = LEARNING_ORDER.index(learning) if learning in LEARNING_ORDER else 99
    return (1, index, to_float(lam) or 0.0)


# ---------------------------------------------------------------------------
# Small stats helpers
# ---------------------------------------------------------------------------

def mean_sd(values: list[float]) -> tuple[float | None, float | None, int]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None, None, 0
    if len(clean) == 1:
        return clean[0], 0.0, 1
    return statistics.fmean(clean), statistics.stdev(clean), len(clean)


def cell(value: float | None, width: int = 7, digits: int = 3) -> str:
    return "-".rjust(width) if value is None else f"{value:.{digits}f}".rjust(width)


def wilcoxon_signed_rank(differences: list[float]) -> tuple[float | None, str]:
    """Wilcoxon signed-rank statistic with an exact small-sample verdict.

    Paired samples are what CRN buys us: the same environment under two arms. With the 5
    seeds x 7 scenarios the plan calls for, n is small enough that an exact threshold is
    more honest than a normal approximation, so this returns the statistic plus a plain
    verdict rather than a precise p-value we cannot justify.
    """
    nonzero = [d for d in differences if d != 0]
    n = len(nonzero)
    if n < 6:
        return None, f"n={n}: too few non-tied pairs to test"

    ranked = sorted(enumerate(nonzero), key=lambda pair: abs(pair[1]))
    ranks = [0.0] * n
    position = 0
    while position < n:
        end = position
        while end + 1 < n and abs(ranked[end + 1][1]) == abs(ranked[position][1]):
            end += 1
        average = (position + end) / 2 + 1
        for index in range(position, end + 1):
            ranks[ranked[index][0]] = average
        position = end + 1

    positive = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    negative = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    statistic = min(positive, negative)

    # Two-sided critical values for alpha=0.05; reject when statistic <= critical.
    critical = {6: 0, 7: 2, 8: 3, 9: 5, 10: 8, 11: 10, 12: 13, 13: 17, 14: 21, 15: 25,
                16: 29, 17: 34, 18: 40, 19: 46, 20: 52, 21: 58, 22: 65, 23: 73, 24: 81,
                25: 89, 30: 137, 35: 195}
    threshold = critical.get(n)
    if threshold is None:
        # Normal approximation beyond the table.
        expected = n * (n + 1) / 4
        sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (statistic - expected) / sd if sd else 0.0
        verdict = "significant at 0.05" if abs(z) > 1.96 else "not significant at 0.05"
        return statistic, f"n={n}, z={z:.2f}: {verdict}"
    verdict = "significant at 0.05" if statistic <= threshold else "not significant at 0.05"
    return statistic, f"n={n}, W={statistic:.1f} vs crit {threshold}: {verdict}"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def per_scenario_tables(rows: list[dict]) -> dict[int, str]:
    out: dict[int, str] = {}
    by_scenario: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_scenario[row["scenario"]].append(row)

    for scenario in sorted(by_scenario):
        lines = [f"Scenario {scenario} - mean (sd) over seeds", ""]
        lines.append(f"{'arm':22s} {'n':>2s} {'MON1':>15s} {'MON2':>15s} {'MON3':>15s} "
                     f"{'RT share':>15s} {'|Qgap|':>8s} {'gamma':>8s}")
        groups: dict[str, list[dict]] = defaultdict(list)
        for row in by_scenario[scenario]:
            groups[row["arm"]].append(row)

        for arm in sorted(groups, key=arm_sort_key):
            group = groups[arm]
            parts = [f"{arm:22s}", f"{len(group):>2d}"]
            for metric in METRICS:
                mean, sd, _ = mean_sd([r[metric] for r in group])
                parts.append(f"{cell(mean)} ({cell(sd, 5)})")
            qgap, _, _ = mean_sd([r["mean_abs_qgap"] for r in group])
            gamma, _, _ = mean_sd([r["mean_gamma"] for r in group])
            parts.append(cell(qgap, 8, 2))
            parts.append(cell(gamma, 8, 4))
            lines.append(" ".join(parts))

        truth = TRUTH.get(scenario)
        if truth:
            lines.append("")
            lines.append("reference (analytic per-action rates for this scenario):")
            for action in ("mst", "rt"):
                m1, m2, m3, joint = truth[action]
                lines.append(f"  always-{action:3s}  MON1 {m1:.3f}  MON2 {m2:.3f}  "
                             f"MON3 {m3:.3f}  all-three {joint:.3f}")
        out[scenario] = "\n".join(lines)
    return out


def mixture_line_report(rows: list[dict]) -> str:
    """Check each arm against the convex combination its RT share implies.

    residual = observed rate - (p * rate_under_RT + (1-p) * rate_under_MST)

    Residuals should be inside sampling error. A systematic departure means the pipeline is
    not measuring what it claims to - wrong scenario applied, CRN stream misaligned, or the
    satisfaction tally drifting from the monitorables.
    """
    lines = [
        "Position on the MST-RT mixture line",
        "",
        "RDMSim redraws its monitorables from scratch each timestep, so a policy's",
        "satisfaction rate is pinned to a convex combination of the two per-action rates,",
        "weighted by its RT share. Residuals below are observed minus predicted; anything",
        "much beyond sampling error (about 0.015 at 1000 steps) indicates a bug.",
        "",
        f"{'scen':>4s} {'arm':22s} {'RT share':>8s} "
        f"{'MON1 obs/pred':>16s} {'MON2 obs/pred':>16s} {'MON3 obs/pred':>16s} {'worst':>7s}",
    ]

    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["arm"])].append(row)

    worst_overall = 0.0
    worst_where = ""
    for scenario, arm in sorted(groups, key=lambda k: (k[0], arm_sort_key(k[1]))):
        group = groups[(scenario, arm)]
        truth = TRUTH.get(scenario)
        if not truth:
            continue
        share, _, _ = mean_sd([r["rt_share"] for r in group])
        if share is None:
            continue
        parts = [f"{scenario:>4d}", f"{arm:22s}", cell(share, 8)]
        worst = 0.0
        for index, metric in enumerate(METRICS[:3]):
            observed, _, _ = mean_sd([r[metric] for r in group])
            predicted = share * truth["rt"][index] + (1 - share) * truth["mst"][index]
            if observed is None:
                parts.append("-".rjust(16))
                continue
            parts.append(f"{observed:.3f}/{predicted:.3f}".rjust(16))
            worst = max(worst, abs(observed - predicted))
        parts.append(cell(worst, 7))
        lines.append(" ".join(parts))
        if worst > worst_overall:
            worst_overall, worst_where = worst, f"scenario {scenario}, {arm}"

    lines.append("")
    lines.append(f"Largest residual: {worst_overall:.4f} ({worst_where})")
    lines.append("At 1000 timesteps one standard error on a rate near 0.5 is about 0.016,")
    lines.append("so residuals up to roughly 0.03-0.05 are expected from seed noise alone.")
    return "\n".join(lines)


def lambda_response_report(rows: list[dict]) -> str:
    lines = ["Lambda response per learning config", "",
             "Entropy regularisation acts on action selection, so its visible effect is on",
             "RT share; the MON columns follow from that via the mixture line.", ""]

    solver_rows = [r for r in rows if not r.get("baseline")]
    by_scenario: dict[int, list[dict]] = defaultdict(list)
    for row in solver_rows:
        by_scenario[row["scenario"]].append(row)

    for scenario in sorted(by_scenario):
        lines.append(f"-- scenario {scenario} --")
        lambdas = sorted({r["lambda"] for r in by_scenario[scenario]},
                         key=lambda x: to_float(x) or 0.0)
        header = f"{'learning':10s} " + " ".join(f"{'lam=' + str(l):>14s}" for l in lambdas)
        lines.append(header)
        for learning in LEARNING_ORDER:
            subset = [r for r in by_scenario[scenario] if r["learning"] == learning]
            if not subset:
                continue
            cells = []
            for lam in lambdas:
                group = [r for r in subset if r["lambda"] == lam]
                share, _, _ = mean_sd([r["rt_share"] for r in group])
                mon1, _, _ = mean_sd([r["MON1_rate"] for r in group])
                if share is None:
                    cells.append("-".rjust(14))
                else:
                    cells.append(f"RT {share:.2f} M1 {mon1:.2f}".rjust(14))
            lines.append(f"{learning:10s} " + " ".join(cells))
        lines.append("")
    return "\n".join(lines)


def paired_tests_report(rows: list[dict], baseline_arm: str) -> str:
    """Paired comparisons of every solver arm against one reference arm.

    Pairing is by (scenario, seed): under CRN those two runs saw the identical environment,
    so the difference isolates the algorithm. Without CRN this test is not valid and the
    report says so.
    """
    lines = [f"Paired comparison against '{baseline_arm}' (pairing by scenario+seed)", ""]

    unpaired = [r for r in rows if str(r.get("crn", "")).lower() != "true"]
    if unpaired:
        lines.append(f"WARNING: {len(unpaired)} row(s) were run without CRN. Paired tests")
        lines.append("assume both arms saw the same environment; re-run those with --crn.")
        lines.append("")

    reference = {(r["scenario"], r["seed"]): r for r in rows if r["arm"] == baseline_arm}
    if not reference:
        available = sorted({r["arm"] for r in rows}, key=arm_sort_key)
        lines.append(f"Reference arm '{baseline_arm}' not present. Available:")
        lines.extend(f"  {arm}" for arm in available)
        return "\n".join(lines)

    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["arm"]].append(row)

    for metric in METRICS:
        lines.append(f"-- {metric} --")
        lines.append(f"{'arm':22s} {'pairs':>5s} {'mean diff':>10s} {'verdict':s}")
        for arm in sorted(groups, key=arm_sort_key):
            if arm == baseline_arm:
                continue
            differences = []
            for row in groups[arm]:
                other = reference.get((row["scenario"], row["seed"]))
                if other and row[metric] is not None and other[metric] is not None:
                    differences.append(row[metric] - other[metric])
            if not differences:
                continue
            mean = statistics.fmean(differences)
            _, verdict = wilcoxon_signed_rank(differences)
            lines.append(f"{arm:22s} {len(differences):>5d} {mean:>+10.4f} {verdict}")
        lines.append("")
    return "\n".join(lines)


def horizon_warning(rows: list[dict]) -> list[str]:
    """Warn if any arm mixes run lengths.

    Averaging a 200-step cell with a 1000-step one is not meaningful: the satisfaction rate
    is a mean over timesteps, so the short run contributes far noisier numbers at equal
    weight. In a properly run grid every cell shares a horizon, so this should stay silent.
    """
    horizons = defaultdict(set)
    for row in rows:
        horizons[(row["scenario"], row["arm"])].add(row.get("timesteps"))
    mixed = {key: sorted(v, key=lambda x: int(x)) for key, v in horizons.items()
             if len(v) > 1}
    if not mixed:
        return []
    lines = ["", "WARNING: these arms mix run lengths; their means are not comparable:"]
    for (scenario, arm), lengths in sorted(mixed.items()):
        lines.append(f"  scenario {scenario} {arm}: timesteps {lengths}")
    lines.append("Re-run the odd ones out at a single horizon, or filter the CSV first.")
    return lines


def coverage_report(rows: list[dict]) -> str:
    lines = ["Grid coverage", ""]
    by_scenario: dict[int, set] = defaultdict(set)
    for row in rows:
        by_scenario[row["scenario"]].add((row["arm"], row["seed"]))
    arms = {row["arm"] for row in rows}
    seeds = {row["seed"] for row in rows}
    lines.append(f"{len(rows)} completed cell(s); {len(arms)} arm(s); "
                 f"{len(seeds)} seed(s); {len(by_scenario)} scenario(s)")
    for scenario in sorted(by_scenario):
        lines.append(f"  scenario {scenario}: {len(by_scenario[scenario])} cell(s)")
    missing = []
    for scenario in sorted(by_scenario):
        for arm in sorted(arms, key=arm_sort_key):
            present = {seed for a, seed in by_scenario[scenario] if a == arm}
            absent = sorted(seeds - present)
            if absent:
                missing.append(f"  scenario {scenario} {arm}: missing seeds {absent}")
    lines.extend(horizon_warning(rows))
    if missing:
        lines.append("")
        lines.append("Incomplete cells (the grid is partial):")
        lines.extend(missing[:40])
        if len(missing) > 40:
            lines.append(f"  ... and {len(missing) - 40} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build ablation tables and figures from experiments/results.csv.")
    parser.add_argument("--csv", default=str(PROJECT_ROOT / "experiments" / "results.csv"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "experiments" / "analysis"),
                        help="directory for the generated reports")
    parser.add_argument("--baseline-arm", default="static lam=0",
                        help="reference arm for the paired tests (default: 'static lam=0')")
    parser.add_argument("--stdout", action="store_true",
                        help="also print everything to the terminal")
    args = parser.parse_args(argv)

    rows = load_rows(Path(args.csv))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sections: list[tuple[str, str]] = [("coverage", coverage_report(rows))]
    for scenario, text in per_scenario_tables(rows).items():
        sections.append((f"per_scenario_{scenario}", text))
    sections.append(("mixture_line", mixture_line_report(rows)))
    sections.append(("lambda_response", lambda_response_report(rows)))
    sections.append(("paired_tests", paired_tests_report(rows, args.baseline_arm)))

    for name, text in sections:
        (out_dir / f"{name}.txt").write_text(text + "\n", encoding="utf-8")

    summary = io.StringIO()
    summary.write("# RDMSim ablation results\n\n")
    summary.write(f"Source: `{args.csv}` ({len(rows)} completed cells)\n\n")
    for name, text in sections:
        summary.write(f"## {name.replace('_', ' ')}\n\n```\n{text}\n```\n\n")
    (out_dir / "summary.md").write_text(summary.getvalue(), encoding="utf-8")

    print(f"Wrote {len(sections) + 1} report(s) to {out_dir}")
    if args.stdout:
        for name, text in sections:
            print(f"\n===== {name} =====\n{text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
