"""Cross-arm figures built from results.csv alone: lambda response, bars, mixture line,
paired-difference forest plots.

All are seven scenario panels; error bars are +/-1 SD across the 5 seeds of that scenario,
except the forest plots, which show a 95% t-interval on the CRN-paired difference.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

from . import style
from .data import Thresholds, analyse
from .stats import mean_sd, paired_ci, paired_differences, summarise, t_crit

RATE_METRICS = {nfr["rate"]: (i, nfr) for i, nfr in enumerate(style.NFRS)}
METRIC_TITLES = {
    "rt_share": ("RT share", "share of timesteps choosing RT"),
    "MON1_rate": ("MON1 (active links) satisfaction", "MON1 satisfaction rate"),
    "MON2_rate": ("MON2 (bandwidth) satisfaction", "MON2 satisfaction rate"),
    "MON3_rate": ("MON3 (time to write) satisfaction", "MON3 satisfaction rate"),
}
FIGURE_METRICS = ["rt_share", "MON1_rate", "MON2_rate", "MON3_rate"]
SEED_NOTE = "mean ± 1 SD across 5 seeds"


def _short(metric: str) -> str:
    return metric.replace("_rate", "")


def _solver(grid: pd.DataFrame) -> pd.DataFrame:
    return grid[grid["baseline"] == ""]


# ---------------------------------------------------------------------------
# 1. Lambda response
# ---------------------------------------------------------------------------

def _titled(text: str, metric: str, thresholds: Thresholds) -> str:
    """Add the NFR's threshold and min/max goal as a subtitle on per-NFR figures."""
    if metric in RATE_METRICS:
        return style.with_subtitle(text, style.goal_text(RATE_METRICS[metric][0], thresholds))
    return text


def lambda_response(grid: pd.DataFrame, metric: str, descriptions: dict[int, str],
                    thresholds: Thresholds) -> go.Figure:
    """metric against lambda, one line per learning config, +/-1 SD band."""
    title, y_title = METRIC_TITLES[metric]
    fig = style.scenario_grid(
        _titled(f"{title} vs entropy regularisation λ ({SEED_NOTE})", metric, thresholds),
        "λ", y_title, descriptions)
    stats = summarise(_solver(grid), ["scenario", "learning", "lam"], metric)
    x_labels = [f"{lam:g}" for lam in style.LAMBDAS]

    for s in style.SCENARIOS:
        row, col = style.panel(s)
        for config in style.LEARNING_ORDER:
            sub = stats[(stats["scenario"] == s) & (stats["learning"] == config)].set_index("lam")
            sub = sub.reindex(style.LAMBDAS)
            style.add_line_band(fig, x_labels, sub["mean"].to_numpy(), sub["sd"].to_numpy(),
                                style.CONFIG_COLOURS[config], style.CONFIG_LABELS[config], row, col,
                                showlegend=(s == 0), legendgroup=config,
                                symbol=style.CONFIG_SYMBOLS[config], marker=True)
        if metric in RATE_METRICS:
            style.add_analytic_refs(fig, s, RATE_METRICS[metric][0], row, col)
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=x_labels,
                         row=row, col=col)
        fig.update_yaxes(range=[-0.02, 1.02], row=row, col=col)
    if metric in RATE_METRICS:
        style.add_reference_legend(fig)
    return fig


# ---------------------------------------------------------------------------
# 2. Bars
# ---------------------------------------------------------------------------

def nfr_bars(grid: pd.DataFrame, metric: str, descriptions: dict[int, str],
             thresholds: Thresholds) -> go.Figure:
    """Mean measured monitorable per learning config, bars coloured by lambda, +/-1 SD
    whiskers, with the NFR threshold as a red line.

    Each bar is the mean over seeds of the per-run mean (mean_links / mean_bw / mean_ttw in
    results.csv), so it shares units with the threshold and "above/below the red line"
    reads literally. It is a mean, though: a run whose mean sits on the good side of the
    line still violates the NFR on some steps (the satisfaction rate is what the
    lambda-response and mixture figures show).

    Bars are placed on a numeric axis with explicit offsets rather than Plotly's grouped
    categories: the fixed baselines have one bar per category where the configs have five,
    and grouped mode would squeeze each baseline into a fifth of its slot. Configs sit at
    0..4 with five lambda bars each; the three fixed baselines follow at full width, in
    greys, after a gap.
    """
    index, nfr = RATE_METRICS[metric]
    value_col = nfr["mean"]
    threshold = thresholds.for_nfr(index)
    fig = style.scenario_grid(
        style.with_subtitle(
            f"Mean {nfr['quantity']} ({nfr['key']}) by learning config and λ ({SEED_NOTE} of the "
            f"per-run mean)", style.goal_text(index, thresholds)),
        "", nfr["unit"], descriptions)
    stats = summarise(_solver(grid), ["scenario", "learning", "lam"], value_col)
    base = summarise(grid[grid["baseline"] != ""], ["scenario", "baseline"], value_col)
    top = max(threshold, float(np.nanmax(stats["mean"] + stats["sd"])),
              float(np.nanmax(base["mean"] + base["sd"])))
    base_order = ["mst", "rt", "random"]
    bar_w = 0.16
    lam_offsets = {lam: (k - 2) * bar_w for k, lam in enumerate(style.LAMBDAS)}
    config_x = {c: float(i) for i, c in enumerate(style.LEARNING_ORDER)}
    base_x = {b: len(style.LEARNING_ORDER) + 0.25 + 0.55 * j for j, b in enumerate(base_order)}
    tickvals = list(config_x.values()) + list(base_x.values())
    ticktext = [style.CONFIG_SHORT[c] for c in style.LEARNING_ORDER] + ["MST", "RT", "rand"]

    for s in style.SCENARIOS:
        row, col = style.panel(s)
        for lam in style.LAMBDAS:
            sub = stats[(stats["scenario"] == s) & (stats["lam"] == lam)].set_index("learning")
            sub = sub.reindex(style.LEARNING_ORDER)
            labels = [style.CONFIG_LABELS[c] for c in style.LEARNING_ORDER]
            fig.add_trace(go.Bar(
                x=[config_x[c] + lam_offsets[lam] for c in style.LEARNING_ORDER],
                y=sub["mean"], width=bar_w * 0.92, name=style.lam_label(lam),
                marker=dict(color=style.LAMBDA_COLOURS[lam], line=dict(width=0)),
                error_y=dict(type="data", array=sub["sd"], thickness=1, width=1.5,
                             color=style.INK_SECONDARY),
                customdata=labels, legendgroup=f"lam{lam:g}", showlegend=(s == 0),
                hovertemplate=f"%{{customdata}} {style.lam_label(lam)}<br>"
                              "%{y:.1f} ± %{error_y.array:.1f} SD<extra></extra>"),
                row=row, col=col)
        bsub = base[base["scenario"] == s].set_index("baseline").reindex(base_order)
        fig.add_trace(go.Bar(
            x=[base_x[b] for b in base_order], y=bsub["mean"], width=0.4,
            name="fixed policy (MST / RT / random)",
            marker=dict(color=[style.BASELINE_COLOURS[b] for b in base_order], line=dict(width=0)),
            error_y=dict(type="data", array=bsub["sd"], thickness=1, width=1.5,
                         color=style.INK_SECONDARY),
            customdata=[style.BASELINE_LABELS[b] for b in base_order],
            legendgroup="fixed", showlegend=(s == 0),
            hovertemplate="%{customdata}<br>%{y:.1f} ± %{error_y.array:.1f} SD<extra></extra>"),
            row=row, col=col)
        # Unlabelled: the subtitle states the threshold and an in-panel label lands on bars.
        style.add_threshold(fig, threshold, nfr["op"], row, col, label=False)
        fig.update_xaxes(tickmode="array", tickvals=tickvals, ticktext=ticktext, tickangle=0,
                         tickfont=dict(size=11), range=[-0.55, max(base_x.values()) + 0.4],
                         row=row, col=col)
        fig.update_yaxes(range=[0, top * 1.08], row=row, col=col)
    style.add_threshold_legend(fig)
    fig.update_layout(barmode="overlay")
    return fig


# ---------------------------------------------------------------------------
# 3. Mixture line
# ---------------------------------------------------------------------------

def mixture_line(grid: pd.DataFrame, metric: str, descriptions: dict[int, str],
                 thresholds: Thresholds) -> go.Figure:
    """Each arm at (RT share, satisfaction rate) against the analytic MST-RT line.

    RDMSim is memoryless given the action, so every policy must sit on the straight line
    between always-MST (x=0) and always-RT (x=1). Points off the line would mean a bug.
    """
    index, nfr = RATE_METRICS[metric]
    title, y_title = METRIC_TITLES[metric]
    fig = style.scenario_grid(
        _titled(f"{title} vs RT share, against the analytic mixture line "
                f"({SEED_NOTE}, both axes)", metric, thresholds),
        "RT share", y_title, descriptions, shared_y=True)
    groups = grid.groupby(["scenario", "arm"], sort=False)

    for s in style.SCENARIOS:
        row, col = style.panel(s)
        truth = analyse.TRUTH.get(s)
        if truth:
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[truth["mst"][index], truth["rt"][index]], mode="lines",
                line=dict(color=style.REFERENCE_COLOUR, width=1.4),
                name="analytic mixture line", legendgroup="line", showlegend=(s == 0),
                hoverinfo="skip"), row=row, col=col)
        for (scenario, arm), group in groups:
            if scenario != s:
                continue
            x, xsd, _ = mean_sd(group["rt_share"])
            y, ysd, _ = mean_sd(group[metric])
            first = group.iloc[0]
            if first["baseline"]:
                colour = style.BASELINE_COLOURS[first["baseline"]]
                symbol, name = "hexagon", style.BASELINE_LABELS[first["baseline"]]
            else:
                colour = style.CONFIG_COLOURS[first["learning"]]
                symbol = style.LAMBDA_SYMBOLS.get(first["lam"], "circle")
                name = f"{style.CONFIG_LABELS[first['learning']]} {style.lam_label(first['lam'])}"
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers", showlegend=False,
                marker=dict(color=colour, symbol=symbol, size=9, line=dict(width=1, color="white")),
                error_x=dict(type="data", array=[xsd], thickness=1, width=0, color=style.rgba(colour, 0.7)),
                error_y=dict(type="data", array=[ysd], thickness=1, width=0, color=style.rgba(colour, 0.7)),
                name=name,
                hovertemplate=f"{name}<br>RT share %{{x:.3f}} ± {xsd:.3f}<br>"
                              f"rate %{{y:.3f}} ± {ysd:.3f}<extra></extra>"),
                row=row, col=col)
        fig.update_xaxes(range=[-0.03, 1.03], row=row, col=col)
        fig.update_yaxes(range=[-0.02, 1.02], row=row, col=col)

    # Composite legend: colour = learning config, symbol = lambda.
    for config in style.LEARNING_ORDER:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=style.CONFIG_LABELS[config],
                                 marker=dict(color=style.CONFIG_COLOURS[config], symbol="circle", size=9),
                                 legendgroup="colour", legendgrouptitle_text="colour: learning config"))
    for lam in style.LAMBDAS:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=style.lam_label(lam),
                                 marker=dict(color=style.INK_SECONDARY, symbol=style.LAMBDA_SYMBOLS[lam], size=9),
                                 legendgroup="symbol", legendgrouptitle_text="symbol: λ"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="fixed MST / RT / random",
                             marker=dict(color=style.BASELINE_COLOURS["rt"], symbol="hexagon", size=9),
                             legendgroup="fixedkey"))
    fig.update_layout(legend=dict(groupclick="toggleitem"))
    return fig


# ---------------------------------------------------------------------------
# 4. Paired-difference forest plots
# ---------------------------------------------------------------------------

COMPARISONS = {
    # name -> (reference learning config, configs compared against it)
    "vs_static": ("static", ["bayes", "smile-CC", "smile-BF", "smile-MIP"]),
    "vs_bayes": ("bayes", ["smile-CC", "smile-BF", "smile-MIP"]),
}


def _arm(config: str, lam: float) -> str:
    return f"{config} lam={lam:g}"


def paired_forest(grid: pd.DataFrame, comparison: str, metric: str,
                  descriptions: dict[int, str], thresholds: Thresholds) -> go.Figure:
    """Mean CRN-paired difference (arm - reference at the same lambda), 95% t-CI, n=5 seeds."""
    reference, configs = COMPARISONS[comparison]
    title, _ = METRIC_TITLES[metric]
    ref_label = style.CONFIG_LABELS[reference]
    goal = (f"{style.goal_text(RATE_METRICS[metric][0], thresholds)}<br>"
            if metric in RATE_METRICS else "")
    fig = style.scenario_grid(
        f"{title}: difference from {ref_label} at the same λ"
        f"<br><span style='font-size:13px'>{goal}mean of CRN-paired differences (same scenario and "
        f"seed), 95% t-interval over 5 seeds; left of 0 = lower than {ref_label}</span>",
        f"Δ vs {style.CONFIG_SHORT[reference]}", "", descriptions, shared_y=True,
        height=1000 if len(configs) > 3 else 820)
    fig.update_layout(margin=dict(t=135 if goal else 110))

    labels = [f"{style.CONFIG_LABELS[c]} {style.lam_label(lam)}"
              for c in configs for lam in style.LAMBDAS]
    span = 0.0
    for s in style.SCENARIOS:
        row, col = style.panel(s)
        for config in configs:
            for lam in style.LAMBDAS:
                diffs = paired_differences(grid, _arm(config, lam), _arm(reference, lam), metric, s)
                mean, low, high, n = paired_ci(diffs)
                if not math.isfinite(mean):
                    continue
                span = max(span, abs(low) if math.isfinite(low) else 0, abs(high) if math.isfinite(high) else 0)
                label = f"{style.CONFIG_LABELS[config]} {style.lam_label(lam)}"
                colour = style.CONFIG_COLOURS[config]
                fig.add_trace(go.Scatter(
                    x=[mean], y=[label], mode="markers",
                    marker=dict(color=colour, symbol=style.LAMBDA_SYMBOLS[lam], size=8,
                                line=dict(width=1, color="white")),
                    error_x=dict(type="data", symmetric=False, array=[high - mean],
                                 arrayminus=[mean - low], thickness=1.4, width=3, color=colour),
                    name=style.CONFIG_LABELS[config], legendgroup=config,
                    showlegend=(s == 0 and lam == style.LAMBDAS[0]),
                    hovertemplate=f"{label}<br>Δ %{{x:+.3f}}  95% CI [{low:+.3f}, {high:+.3f}]  n={n}"
                                  "<extra></extra>"),
                    row=row, col=col)
        fig.add_vline(x=0, line=dict(color=style.INK_SECONDARY, width=1), row=row, col=col)
        # tickmode=array, or Plotly thins the labels to every other row.
        fig.update_yaxes(type="category", categoryorder="array", categoryarray=labels[::-1],
                         tickmode="array", tickvals=labels, tickfont=dict(size=10),
                         row=row, col=col)
    limit = span * 1.08 if span > 0 else 0.1
    for s in style.SCENARIOS:
        row, col = style.panel(s)
        fig.update_xaxes(range=[-limit, limit], showgrid=True, gridcolor=style.GRID, row=row, col=col)
    for lam in style.LAMBDAS:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=style.lam_label(lam),
                                 marker=dict(color=style.INK_SECONDARY, symbol=style.LAMBDA_SYMBOLS[lam], size=8),
                                 legendgroup="symbol", legendgrouptitle_text="symbol: λ"))
    fig.update_layout(margin=dict(l=150))
    return fig


def pooled_tests(grid: pd.DataFrame, out_path: Path) -> Path:
    """Pooled-over-scenarios paired tests for every comparison, as a CSV.

    Per scenario only 5 pairs exist, and a two-sided Wilcoxon on n=5 cannot go below
    p=0.0625, so per-panel inference uses the t-CI and the rank test runs on all 35 pairs.
    Includes joint_rate (all three NFRs at once), which is tabled rather than plotted.
    """
    metrics = ["rt_share", "MON1_rate", "MON2_rate", "MON3_rate", "joint_rate"]
    rows = []
    for comparison, (reference, configs) in COMPARISONS.items():
        for config in configs:
            for lam in style.LAMBDAS:
                for metric in metrics:
                    if metric not in grid.columns:
                        continue
                    diffs = paired_differences(grid, _arm(config, lam), _arm(reference, lam), metric)
                    if not diffs:
                        continue
                    mean, low, high, n = paired_ci(diffs)
                    statistic, verdict = analyse.wilcoxon_signed_rank(diffs)
                    rows.append({
                        "comparison": comparison, "arm": _arm(config, lam),
                        "reference": _arm(reference, lam), "metric": metric, "pairs": n,
                        "mean_diff": f"{mean:+.4f}", "ci95_low": f"{low:+.4f}", "ci95_high": f"{high:+.4f}",
                        "wilcoxon_W": "" if statistic is None else f"{statistic:.1f}",
                        "verdict": verdict,
                    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["comparison"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote {} pooled paired test(s) to {}", len(rows), out_path)
    return out_path


# ---------------------------------------------------------------------------

def build_all(grid: pd.DataFrame, exporter, descriptions: dict[int, str],
              families: set[str], thresholds: Thresholds) -> None:
    if "lambda" in families:
        for metric in FIGURE_METRICS:
            exporter.save(lambda_response(grid, metric, descriptions, thresholds),
                          f"lambda_response/{_short(metric)}")
    if "bars" in families:
        for metric in FIGURE_METRICS[1:]:
            exporter.save(nfr_bars(grid, metric, descriptions, thresholds),
                          f"bars/{_short(metric)}")
    if "mixture" in families:
        for metric in FIGURE_METRICS[1:]:
            exporter.save(mixture_line(grid, metric, descriptions, thresholds),
                          f"mixture/{_short(metric)}")
    if "paired" in families:
        for comparison in COMPARISONS:
            for metric in FIGURE_METRICS:
                exporter.save(paired_forest(grid, comparison, metric, descriptions, thresholds),
                              f"paired/{comparison}_{_short(metric)}")
        pooled_tests(grid, exporter.root / "paired" / "pooled_tests.csv")
