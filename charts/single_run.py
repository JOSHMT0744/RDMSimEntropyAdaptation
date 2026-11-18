"""The per-run report (``<run>/charts.html``) that RDMSim_TAAS.java requests after a run.

Moved here unchanged from createCharts.py apart from two things: thresholds arrive as a
``Thresholds`` derived from configuration.json (no more placeholder 20.0 defaults), and
the MIP warm-up is a parameter instead of a module global set from __main__.
"""

from __future__ import annotations

import os
import webbrowser

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

from .data import Thresholds

# Window size for moving average smoothing (as in the ACSOS-2026 reference charts).
SMOOTHING_WINDOW = 5
DEFAULT_LOOKBACK = 4


def smooth_series(series, window_size):
    """Centred moving average; the original series if it is shorter than the window."""
    if len(series) < window_size:
        return series
    return series.rolling(window=window_size, center=True).mean()


def satisfactionViolins(df, thresholds: Thresholds):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "MON1 Distribution (Active Links)",
            "MON2 Distribution (Bandwidth, Gbps)",
            "MON3 Distribution (Time to Write, ms)",
        ),
    )
    specs = [("mon1sat", "MON1 (Active Links)", "lightseagreen", thresholds.links),
             ("mon2sat", "MON2 (Bandwidth)", "orange", thresholds.bandwidth),
             ("mon3sat", "MON3 (Time to Write)", "mediumpurple", thresholds.write_time)]
    for col, (column, name, colour, threshold) in enumerate(specs, start=1):
        fig.add_trace(
            go.Violin(y=df[column], name=name, box_visible=True, meanline_visible=True,
                      line_color="black", fillcolor=colour, opacity=0.6),
            row=1, col=col)
        fig.add_hline(y=threshold, line=dict(color="Red"), row=1, col=col,
                      annotation_text=f"threshold {threshold:g}", annotation_position="top left")
    fig.update_layout(yaxis_zeroline=False)
    return fig


def satisfactionPlots(df, thresholds: Thresholds):
    """MON1/MON2/MON3 measured values over time with horizontal threshold lines.

    MON1 = active links (satisfied at >=), MON2 = bandwidth consumption Gbps (<=),
    MON3 = time to write ms (<=). Thresholds come from configuration.json.
    """
    fig = make_subplots(rows=3, cols=1)
    specs = [("mon1sat", "MON1 (Active Links)", "lightseagreen", thresholds.links, "MON1 (Active Links)"),
             ("mon2sat", "MON2 (Bandwidth, Gbps)", "orange", thresholds.bandwidth, "MON2 (Gbps)"),
             ("mon3sat", "MON3 (Time to Write, ms)", "mediumpurple", thresholds.write_time, "MON3 (ms)")]
    for row, (column, name, colour, threshold, y_title) in enumerate(specs, start=1):
        fig.add_trace(go.Scatter(x=df["timestep"], y=df[column], mode="lines",
                                 name=name, line=dict(color=colour)), row=row, col=1)
        fig.add_shape(type="line", x0=df["timestep"].min(), x1=df["timestep"].max(),
                      y0=threshold, y1=threshold, xref=f"x{row}", yref=f"y{row}",
                      line=dict(color="Red"))
        fig.update_yaxes(range=[0, max(df[column].max(), threshold) * 1.2], title_text=y_title,
                         row=row, col=1)
    fig.update_xaxes(title_text="Timestep", row=3, col=1)
    return fig


def _surprise_traces(fig, timestep, series: dict[str, pd.Series]):
    for name, values in series.items():
        fig.add_trace(go.Scatter(x=timestep, y=values, mode="lines", name=name,
                                 line=dict(width=1), opacity=0.5))
    for name, values in series.items():
        fig.add_trace(go.Scatter(x=timestep, y=smooth_series(values, SMOOTHING_WINDOW),
                                 mode="lines", name=f"{name} (Smoothed)", line=dict(width=2)))


def surpriseChart(df):
    fig = go.Figure()
    _surprise_traces(fig, df["timestep"], {
        "Bayes Factor Surprise": df["surprisebf"],
        "Confidence-Corrected Surprise": df["surprisecc"],
        "MIP Surprise": df["surprisemip"],
    })
    fig.update_layout(title="Surprise Over Time", xaxis_title="Timestep",
                      yaxis_title="Surprise", legend_title="Surprise Types")
    return fig


def surpriseChartNormalized(df, warmup_rows: int = 2 * DEFAULT_LOOKBACK):
    """Surprise measures as z-scores for comparison. Skips the first 2*lookback rows so
    MIP cold-start does not skew the standardisation."""
    df = df.iloc[warmup_rows:].reset_index(drop=True)
    if df.empty:
        return go.Figure()

    def zscore(series):
        mu, sigma = series.mean(), series.std()
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sigma

    standardised = {
        "Bayes Factor Surprise": zscore(df["surprisebf"]),
        "Confidence-Corrected Surprise": zscore(df["surprisecc"]),
        "MIP Surprise": zscore(df["surprisemip"]),
    }
    fig = go.Figure()
    _surprise_traces(fig, df["timestep"], standardised)
    y_min = min(s.min() for s in standardised.values())
    y_max = max(s.max() for s in standardised.values())
    fig.update_layout(title="Surprise Over Time (Standardised for Comparison)",
                      xaxis_title="Timestep", yaxis_title="Standardised Surprise (z-score)",
                      legend_title="Surprise Types", yaxis=dict(range=[y_min, y_max]))
    return fig


def gammaChart(df):
    fig = go.Figure(data=go.Scatter(x=df["timestep"], y=df["gamma"], mode="lines",
                                    name="Learning Rate (Gamma)"))
    fig.update_layout(title="Learning Rate (Gamma) Over Time", xaxis_title="Timestep",
                      yaxis_title="Learning Rate (Gamma)")
    return fig


def mipChart(df):
    fig = go.Figure(data=go.Scatter(x=df["timestep"], y=df["surprisemip"], mode="lines", name="MIP"))
    fig.add_trace(go.Scatter(x=df["timestep"], y=df["mip_upper"], mode="lines",
                             name="MIP Upper Bound", line=dict(color="red", width=1)))
    fig.add_trace(go.Scatter(x=df["timestep"], y=df["mip_lower"], mode="lines",
                             name="MIP Lower Bound", line=dict(color="red", width=1)))
    fig.update_layout(title="MIP Over Time with Error Bounds", xaxis_title="Timestep",
                      yaxis_title="MIP")
    return fig


def actionTrace(df):
    """Which topology was selected at each timestep, plus the running share of RT."""
    if "selectedaction" not in df.columns:
        logger.warning("No selectedaction column; skipping action trace chart")
        return go.Figure()
    action = df["selectedaction"]
    running_rt = action.expanding().mean()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Selected topology (0 = MST, 1 = RT)", "Running share of RT"))
    fig.add_trace(go.Scattergl(x=df["timestep"], y=action, mode="markers", name="action",
                               marker=dict(size=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["timestep"], y=running_rt, mode="lines", name="RT share"),
                  row=2, col=1)
    fig.update_yaxes(tickvals=[0, 1], ticktext=["MST", "RT"], row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="timestep", row=2, col=1)
    fig.update_layout(height=520, showlegend=False)
    return fig


def rollingSatisfaction(df, window=50):
    """Rolling per-NFR satisfaction rate from the flags in MON{1,2,3}Sat.txt."""
    flags = [("mon1satflag", "MON1 (active links)"),
             ("mon2satflag", "MON2 (bandwidth)"),
             ("mon3satflag", "MON3 (time to write)")]
    available = [(c, label) for c, label in flags if c in df.columns]
    if not available:
        logger.warning("No satisfaction flag columns; skipping rolling satisfaction chart")
        return go.Figure()
    effective_window = min(window, max(1, len(df) // 2))
    fig = go.Figure()
    for column, label in available:
        rolling = df[column].rolling(window=effective_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=df["timestep"], y=rolling, mode="lines", name=label))
        overall = df[column].mean()
        fig.add_hline(y=overall, line_dash="dot", line_width=1,
                      annotation_text=f"{label} overall {overall:.1%}")
    fig.update_layout(title=f"Rolling per-NFR satisfaction rate (window={effective_window})",
                      xaxis_title="timestep", yaxis_title="satisfaction rate",
                      yaxis=dict(range=[0, 1]), height=460)
    return fig


def buildChartsReport(df, thresholds: Thresholds, warmup_rows: int = 2 * DEFAULT_LOOKBACK) -> str:
    """All per-run charts as one self-contained HTML page (Plotly JS inlined once)."""
    logger.info("Stage: Building per-run charts (thresholds MON1>={}, MON2<={}, MON3<={})",
                thresholds.links, thresholds.bandwidth, thresholds.write_time)
    satisfaction_df = df.filter(items=["timestep", "mon1sat", "mon2sat", "mon3sat"])
    sections = [
        ("MIP Over Time", mipChart(df)),
        ("Learning Rate (Gamma) Over Time", gammaChart(df)),
        ("Surprise Over Time", surpriseChart(df)),
        ("Surprise Over Time (Standardised)", surpriseChartNormalized(df, warmup_rows)),
        ("MON1/MON2/MON3 Measured Values Over Time", satisfactionPlots(satisfaction_df, thresholds)),
        ("MON1/MON2/MON3 Distributions", satisfactionViolins(satisfaction_df, thresholds)),
        ("Rolling Per-NFR Satisfaction Rate", rollingSatisfaction(df)),
        ("Selected Adaptation Action", actionTrace(df)),
    ]
    parts = ["<html><head><title>RDMSim Charts</title></head><body>"]
    for i, (title, fig) in enumerate(sections):
        parts.append(f"<h2>{title}</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs=("inline" if i == 0 else False)))
    parts.append("</body></html>")
    logger.info("Stage: Chart generation complete")
    return "\n".join(parts)


def writeChartsReport(df, thresholds: Thresholds, output_dir, warmup_rows: int) -> str:
    report_path = os.path.abspath(os.path.join(output_dir, "charts.html"))
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(buildChartsReport(df, thresholds, warmup_rows))
    logger.info("Stage: Wrote combined charts report to {}", report_path)
    return report_path


def createCharts(df, thresholds: Thresholds, output_dir, warmup_rows: int,
                 open_browser: bool = True) -> str:
    """Write <output_dir>/charts.html and best-effort open it once."""
    report_path = writeChartsReport(df, thresholds, output_dir, warmup_rows)
    if open_browser:
        try:
            webbrowser.open(f"file://{report_path}")
        except Exception as e:
            logger.warning("Could not auto-open charts report ({}); open it manually: {}", e, report_path)
    return report_path
