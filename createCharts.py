"""Charts for RDMSim_TAAS.

Two modes:

  python createCharts.py --output-dir <run dir> [--mon1-threshold ...] [--lookback N]
      Per-run report, <run dir>/charts.html. This is the call RDMSim_TAAS.java makes at
      the end of a run, so its flags are a stable contract.

  python createCharts.py --grid [--families lambda,bars,mixture,paired,timeseries,violins]
      Paper figures for the ablation grid, from experiments/results.csv and the per-cell
      logs, written to figures/ as HTML + PNG with an index.html. Every level plot
      is mean +/- 1 SD across seeds; comparisons are 95% CIs of CRN-paired differences.

NFR thresholds are read from config_log_files/configuration.json (as RDMSim computes them);
the --mon*-threshold flags override that when given.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

from charts import data, single_run  # noqa: E402  (after logger setup)
from charts.data import read_run_dir as getData  # noqa: E402,F401  (old public name)

FAMILIES = ["lambda", "bars", "mixture", "paired", "timeseries", "violins"]


def run(data_dir=None, mon1_threshold=None, mon2_threshold=None, mon3_threshold=None,
        lookback=single_run.DEFAULT_LOOKBACK, config=None, open_browser=True):
    """Per-run report for one RDMSim_TAAS output directory."""
    data_dir = Path(data_dir).resolve() if data_dir else data.PROJECT_ROOT / "output_dir"
    logger.info("Stage: Chart generation pipeline started for {}", data_dir)
    thresholds = data.thresholds_from_config(config, (mon1_threshold, mon2_threshold, mon3_threshold))
    data.check_run_thresholds(data_dir, thresholds)
    frame = data.read_run_dir(data_dir)
    return single_run.createCharts(frame, thresholds, data_dir, warmup_rows=2 * lookback,
                                   open_browser=open_browser)


def run_grid(args) -> Path:
    """Every paper figure family for the ablation grid."""
    from charts import grid_figures, timeseries, violins
    from charts.export import Exporter

    started = time.time()
    families = set(FAMILIES if args.families == "all" else
                   [f.strip() for f in args.families.split(",") if f.strip()])
    unknown = families - set(FAMILIES)
    if unknown:
        raise SystemExit(f"--families: unknown {sorted(unknown)}; choose from {FAMILIES} or all")
    lambdas = ([0.0, 2.0, 5.0, 10.0, 20.0] if args.lambdas == "all" else
               [float(x) for x in args.lambdas.split(",") if x.strip()])

    thresholds = data.thresholds_from_config(
        args.config, (args.mon1_threshold, args.mon2_threshold, args.mon3_threshold))
    logger.info("Thresholds from configuration: MON1 >= {:g}, MON2 <= {:g}, MON3 <= {:g}",
                thresholds.links, thresholds.bandwidth, thresholds.write_time)
    descriptions = data.scenario_descriptions(args.config)
    grid = data.load_grid(args.csv)
    logger.info("Loaded {} completed cell(s) from {}", len(grid), args.csv or data.DEFAULT_CSV)

    # One run per scenario is enough to catch a config edited after the grid ran.
    for scenario in sorted(grid["scenario"].unique()):
        row = grid[grid["scenario"] == scenario].iloc[0]
        data.check_run_thresholds(data.cell_dir(Path(args.cells_root), row), thresholds)

    exporter = Exporter(args.figures_dir, static=not args.html_only)
    grid_figures.build_all(grid, exporter, descriptions, families, thresholds)

    if families & {"timeseries", "violins"}:
        series = data.load_timeseries(grid, args.cells_root, use_cache=not args.no_cache)
        if "timeseries" in families:
            timeseries.build_all(series, exporter, descriptions, thresholds, lambdas, args.window)
        if "violins" in families:
            violins.build_all(series, exporter, descriptions, thresholds, lambdas)

    logger.info("Wrote {} HTML figure(s); rendering static copies", len(exporter.written))
    exporter.flush()
    index = exporter.write_index()
    logger.info("Done in {:.0f}s. Open {}", time.time() - started, index)
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Charts for RDMSim_TAAS: a per-run report (default) or the ablation-grid "
                    "paper figures (--grid).")
    parser.add_argument("--output-dir", type=str, default=None, metavar="PATH",
                        help="per-run mode: directory with the run's output (MON1Sat.txt, "
                             "gamma.txt, ...). Defaults to output_dir/. RDMSim_TAAS.java sets it "
                             "from solver.config's outputDirectory.")
    parser.add_argument("--config", type=str, default=None, metavar="PATH",
                        help="RDMSim configuration.json the NFR thresholds are derived from "
                             "(default: config_log_files/configuration.json)")
    for i, what in ((1, "MON1 (active links)"), (2, "MON2 (bandwidth, Gbps)"),
                    (3, "MON3 (time to write, ms)")):
        parser.add_argument(f"--mon{i}-threshold", type=float, default=None, metavar="VALUE",
                            help=f"override the {what} threshold; default: derived from --config")
    parser.add_argument("--lookback", type=int, default=single_run.DEFAULT_LOOKBACK, metavar="N",
                        help="MIP lookback from solver.config; the first 2*lookback surprise rows "
                             f"are discarded as cold-start. Default {single_run.DEFAULT_LOOKBACK}.")
    parser.add_argument("--no-open", action="store_true",
                        help="per-run mode: do not open the report in a browser")

    grid = parser.add_argument_group("grid mode")
    grid.add_argument("--grid", action="store_true", help="build the ablation-grid paper figures")
    grid.add_argument("--csv", default=str(data.DEFAULT_CSV), help="results.csv from run_ablation.py")
    grid.add_argument("--cells-root", default=str(data.DEFAULT_CELLS), help="per-cell output root")
    grid.add_argument("--figures-dir", default=str(data.PROJECT_ROOT / "figures"),
                      help="where figures are written (default: figures/)")
    grid.add_argument("--families", default="all",
                      help=f"comma list from {','.join(FAMILIES)} (default: all)")
    grid.add_argument("--lambdas", default="all",
                      help="lambda values for the time-series and violin figures (default: all)")
    grid.add_argument("--window", type=int, default=25,
                      help="trailing window for the time-series figures (default: 25)")
    grid.add_argument("--html-only", action="store_true",
                      help="skip PNG export (no kaleido needed)")
    grid.add_argument("--no-cache", action="store_true",
                      help="re-read every cell's per-step logs instead of using the cache")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.grid:
        run_grid(args)
    else:
        run(data_dir=args.output_dir, mon1_threshold=args.mon1_threshold,
            mon2_threshold=args.mon2_threshold, mon3_threshold=args.mon3_threshold,
            lookback=args.lookback, config=args.config, open_browser=not args.no_open)
