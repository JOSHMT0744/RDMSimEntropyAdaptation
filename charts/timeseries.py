"""Per-step series across seeds: one line per learning config at a given lambda.

Each seed's series is first smoothed with a trailing mean over the last ``window`` steps
(min_periods=1, so the first steps average what exists), then the mean +/- 1 SD across the
5 seeds is taken at every timestep, and every timestep is drawn on a linear time axis.
The first ``window`` points average fewer than ``window`` steps, so they are noisier than
the rest; each scenario is stationary in distribution, so the curves settle after a short
learning transient.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import style
from .data import Thresholds
from .stats import seed_band, trailing_mean

# name -> (per-step column, title, y-axis title, kind)
#   kind: "rate" (0-1 with analytic refs), "raw" (monitorable + threshold), "signal"
SERIES = {
    "rt_share": ("action", "RT share", "rolling share of RT", "share"),
    "MON1": ("mon1flag", "MON1 (active links) satisfaction", "rolling satisfaction rate", "rate"),
    "MON2": ("mon2flag", "MON2 (bandwidth) satisfaction", "rolling satisfaction rate", "rate"),
    "MON3": ("mon3flag", "MON3 (time to write) satisfaction", "rolling satisfaction rate", "rate"),
    "links": ("mon1", "Active links (MON1)", "active links", "raw"),
    "bw": ("mon2", "Bandwidth consumption (MON2)", "bandwidth consumption (Gbps)", "raw"),
    "ttw": ("mon3", "Time to write (MON3)", "time to write (ms)", "raw"),
    "gamma": ("gamma", "Learning rate γ", "γ", "signal"),
    "surprise_BF": ("surprisebf", "Bayes-factor surprise", "surprise (BF)", "signal"),
    "surprise_CC": ("surprisecc", "Confidence-corrected surprise", "surprise (CC)", "signal"),
    "surprise_MIP": ("surprisemip", "MIP surprise", "surprise (MIP)", "signal"),
}
RATE_INDEX = {"MON1": 0, "MON2": 1, "MON3": 2}
RAW_INDEX = {"links": 0, "bw": 1, "ttw": 2}


def index_series(timeseries: dict) -> dict[tuple[int, str], list[pd.DataFrame]]:
    """(scenario, arm) -> per-seed frames, seeds in ascending order."""
    grouped: dict[tuple[int, str], list[tuple[int, pd.DataFrame]]] = defaultdict(list)
    for (scenario, arm, seed), frame in timeseries.items():
        grouped[(scenario, arm)].append((seed, frame))
    return {key: [f for _, f in sorted(items, key=lambda p: p[0])] for key, items in grouped.items()}


def series_figure(by_arm: dict, lam: float, name: str, descriptions: dict[int, str],
                  thresholds: Thresholds, window: int) -> go.Figure:
    column, title, y_title, kind = SERIES[name]
    smoothing = f"trailing mean over {window} steps" if window > 1 else "per step"
    heading = f"{title} over time at {style.lam_label(lam)} ({smoothing}; mean ± 1 SD across 5 seeds)"
    nfr_index = RATE_INDEX.get(name, RAW_INDEX.get(name))
    if nfr_index is not None:
        heading = style.with_subtitle(heading, style.goal_text(nfr_index, thresholds))
    fig = style.scenario_grid(heading, "timestep", y_title, descriptions)

    low, high = np.inf, -np.inf
    for s in style.SCENARIOS:
        row, col = style.panel(s)
        for config in style.LEARNING_ORDER:
            frames = by_arm.get((s, f"{config} lam={lam:g}"))
            if not frames:
                continue
            per_seed = [trailing_mean(frame[column].to_numpy(dtype=float), window) for frame in frames]
            if all(np.all(np.isnan(p)) for p in per_seed):
                continue
            mean, sd = seed_band(per_seed)
            style.add_line_band(fig, np.arange(len(mean)), mean, sd, style.CONFIG_COLOURS[config],
                                style.CONFIG_LABELS[config], row, col,
                                showlegend=(s == 0), legendgroup=config, width=1.5)
            band = np.concatenate([mean - sd, mean + sd])
            band = band[np.isfinite(band)]
            if band.size:
                low, high = min(low, band.min()), max(high, band.max())

        if kind == "rate":
            style.add_analytic_refs(fig, s, RATE_INDEX[name], row, col)
        if kind == "raw":
            nfr = style.NFRS[RAW_INDEX[name]]
            # Unlabelled: the subtitle states the threshold; an in-panel label sits on the lines.
            style.add_threshold(fig, thresholds.for_nfr(RAW_INDEX[name]), nfr["op"], row, col,
                                label=False)

    if kind in ("rate", "share"):
        fig.update_yaxes(range=[-0.02, 1.02])
    elif kind == "raw" and np.isfinite(low):
        threshold = thresholds.for_nfr(RAW_INDEX[name])
        lo, hi = min(low, threshold), max(high, threshold)
        pad = 0.06 * (hi - lo or 1.0)
        fig.update_yaxes(range=[lo - pad, hi + pad])
    if kind == "rate":
        style.add_reference_legend(fig)
    if kind == "raw":
        style.add_threshold_legend(fig)
    # The fig-wide y-range above also hit the legend cell's hidden axes; re-hide them.
    fig.update_xaxes(visible=False, row=2, col=4)
    fig.update_yaxes(visible=False, row=2, col=4)
    return fig


def build_all(timeseries: dict, exporter, descriptions: dict[int, str], thresholds: Thresholds,
              lambdas: list[float], window: int) -> None:
    by_arm = index_series(timeseries)
    for lam in lambdas:
        for name in SERIES:
            exporter.save(series_figure(by_arm, lam, name, descriptions, thresholds, window),
                          f"timeseries/lam{lam:g}/{name}")
