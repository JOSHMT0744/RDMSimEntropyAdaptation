"""Raw monitorable distributions with the NFR threshold, per (scenario, lambda).

One figure per scenario and lambda, three panels (active links, bandwidth, time to write),
eight violins per panel: the five learning configs at that lambda plus the three fixed
baselines (which do not depend on lambda, so they repeat across lambda figures as the
reference). A violin pools all 5 seeds x 1000 timesteps of measured values, so it shows the
distribution, not seed spread; the black marker on each is the mean of the five per-seed
means with a +/-1 SD whisker, which is the seed spread.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import style
from .data import Thresholds
from .stats import mean_sd
from .timeseries import index_series


def violin_figure(by_arm: dict, scenario: int, lam: float, descriptions: dict[int, str],
                  thresholds: Thresholds) -> go.Figure:
    titles = [f"{nfr['label']}<br><span style='font-size:11px'>"
              f"{style.OP_SYMBOL[nfr['op']]} {thresholds.for_nfr(i):g} {nfr['unit_short']} "
              f"({thresholds.percent(i):g}% of capacity): {nfr['goal']}</span>"
              for i, nfr in enumerate(style.NFRS)]
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.07)
    goals = " · ".join(
        f"{nfr['key']} {style.OP_SYMBOL[nfr['op']]} {thresholds.for_nfr(i):g} {nfr['unit_short']} "
        f"({nfr['goal']})" for i, nfr in enumerate(style.NFRS))
    style.base_layout(
        fig, style.with_subtitle(
            f"<b>S{scenario}</b> · {descriptions.get(scenario, '')} - monitorable distributions "
            f"at {style.lam_label(lam)} (5 seeds × all timesteps; marker = mean ± 1 SD of seed means)",
            f"Red line = NFR threshold. {goals}"),
        1500, 600)
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=13, color=style.INK)

    arms = ([(style.CONFIG_LABELS[c], f"{c} lam={lam:g}", style.CONFIG_COLOURS[c])
             for c in style.LEARNING_ORDER]
            + [(style.BASELINE_LABELS[b], f"fixed-{b}", style.BASELINE_COLOURS[b])
               for b in ("mst", "rt", "random")])
    categories = [label for label, _, _ in arms]

    for i, nfr in enumerate(style.NFRS):
        col = i + 1
        for label, arm, colour in arms:
            frames = by_arm.get((scenario, arm))
            if not frames:
                continue
            per_seed = [f[nfr["series"]].to_numpy(dtype=float) for f in frames]
            pooled = np.concatenate(per_seed)
            pooled = pooled[np.isfinite(pooled)]
            if pooled.size == 0:
                continue
            fig.add_trace(go.Violin(
                # No x array: on a categorical axis the violin sits at its trace name, which
                # keeps a 5000-copy label list out of every HTML page.
                y=np.round(pooled, 1), name=label,
                line=dict(color=colour, width=1.2), fillcolor=style.rgba(colour, 0.45),
                spanmode="hard", points=False, meanline_visible=False, box_visible=False,
                width=0.85, showlegend=False, hoverinfo="skip"),
                row=1, col=col)
            seed_means = [np.nanmean(p) for p in per_seed]
            mean, sd, n = mean_sd(seed_means)
            fig.add_trace(go.Scatter(
                x=[label], y=[mean], mode="markers", showlegend=False,
                marker=dict(color=style.INK, size=7, symbol="line-ew-open",
                            line=dict(width=2, color=style.INK)),
                error_y=dict(type="data", array=[sd], thickness=1.6, width=6, color=style.INK),
                hovertemplate=f"{label}<br>mean of {n} seed means %{{y:.1f}} ± {sd:.1f} SD"
                              "<extra></extra>"),
                row=1, col=col)
        # Unlabelled: the panel title already states the threshold, and an in-plot label
        # collides with the per-arm mean markers.
        style.add_threshold(fig, thresholds.for_nfr(i), nfr["op"], 1, col, label=False)
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=categories,
                         tickangle=-35, row=1, col=col)
        fig.update_yaxes(title_text=nfr["unit"], row=1, col=col)
    fig.update_layout(violinmode="overlay", margin=dict(t=150, b=110))
    return fig


def build_all(timeseries: dict, exporter, descriptions: dict[int, str], thresholds: Thresholds,
              lambdas: list[float]) -> None:
    by_arm = index_series(timeseries)
    for scenario in style.SCENARIOS:
        for lam in lambdas:
            exporter.save(violin_figure(by_arm, scenario, lam, descriptions, thresholds),
                          f"violins/s{scenario}/lam{lam:g}")
