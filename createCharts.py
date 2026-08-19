from functools import reduce
import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import webbrowser
from loguru import logger

# Configure loguru: clear default, add stderr with stage-friendly format
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

# Files that RDMSim_TAAS writes but that getData() should not try to merge into the
# main timeseries dataframe: RDM.alpha (solver value-function dump), SelectedAction.txt
# (categorical, not merged), state_transitions.txt and monitorables.txt (wide, own shape),
# charts.html (this script's own output, left in output_dir by a previous run).
SKIP_FILES = {"SelectedAction.txt", "state_transitions.txt", "monitorables.txt", "charts.html"}

# Skip this many initial surprise rows so MIP cold-start (first 2*lookback) does not skew normalisation
MIP_WARMUP_ROWS = 2 * 5  # 2 * lookback (Java default lookback=5)

# Window size for moving average smoothing
SMOOTHING_WINDOW = 5


def smooth_series(series, window_size):
    """Apply centered moving average smoothing to a pandas Series.

    Returns the original series if it is shorter than window_size.
    """
    if len(series) < window_size:
        return series
    return series.rolling(window=window_size, center=True).mean()


def satisfactionViolins(df):
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "MON1 Satisfaction Distribution (Active Links)",
            "MON2 Satisfaction Distribution (Bandwidth, GB/s)",
            "MON3 Satisfaction Distribution (Time to Write, ms)",
        ),
    )

    fig.add_trace(
        go.Violin(
            y=df["mon1sat"], name="MON1 (Active Links)", box_visible=True,
            meanline_visible=True, line_color="black", fillcolor="lightseagreen", opacity=0.6,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Violin(
            y=df["mon2sat"], name="MON2 (Bandwidth)", box_visible=True,
            meanline_visible=True, line_color="black", fillcolor="orange", opacity=0.6,
        ),
        row=1, col=2,
    )

    fig.add_trace(
        go.Violin(
            y=df["mon3sat"], name="MON3 (Time to Write)", box_visible=True,
            meanline_visible=True, line_color="black", fillcolor="mediumpurple", opacity=0.6,
        ),
        row=1, col=3,
    )

    fig.update_layout(yaxis_zeroline=False)
    return fig


def satisfactionPlots(df, mon1_threshold=20.0, mon2_threshold=20.0, mon3_threshold=20.0):
    """
    MON1/MON2/MON3 satisfaction time-series with horizontal threshold lines.
    Thresholds come from solver.config (mon1Threshold, mon2Threshold, mon3Threshold) for the run.
    MON1 = active links (higher is better), MON2 = bandwidth consumption GB/s (lower is better),
    MON3 = time to write ms (lower is better).
    """
    fig = make_subplots(rows=3, cols=1)

    fig.add_trace(
        go.Scatter(
            x=df["timestep"], y=df["mon1sat"], mode="lines",
            name="MON1 Satisfaction (Active Links)", line=dict(color="lightseagreen"),
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestep"], y=df["mon2sat"], mode="lines",
            name="MON2 Satisfaction (Bandwidth, GB/s)", line=dict(color="orange"),
        ),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["timestep"], y=df["mon3sat"], mode="lines",
            name="MON3 Satisfaction (Time to Write, ms)", line=dict(color="mediumpurple"),
        ),
        row=3, col=1,
    )

    fig.add_shape(
        type="line", x0=df["timestep"].min(), x1=df["timestep"].max(),
        y0=mon1_threshold, y1=mon1_threshold, xref="x1", yref="y1", line=dict(color="Red"),
    )
    fig.add_shape(
        type="line", x0=df["timestep"].min(), x1=df["timestep"].max(),
        y0=mon2_threshold, y1=mon2_threshold, xref="x2", yref="y2", line=dict(color="Red"),
    )
    fig.add_shape(
        type="line", x0=df["timestep"].min(), x1=df["timestep"].max(),
        y0=mon3_threshold, y1=mon3_threshold, xref="x3", yref="y3", line=dict(color="Red"),
    )

    fig.update_yaxes(range=[0, max(df["mon1sat"].max(), mon1_threshold) * 1.2], row=1, col=1)
    fig.update_yaxes(range=[0, max(df["mon2sat"].max(), mon2_threshold) * 1.2], row=2, col=1)
    fig.update_yaxes(range=[0, max(df["mon3sat"].max(), mon3_threshold) * 1.2], row=3, col=1)
    fig.update_xaxes(title_text="Timestep", row=3, col=1)
    fig.update_yaxes(title_text="MON1 (Active Links)", row=1, col=1)
    fig.update_yaxes(title_text="MON2 (GB/s)", row=2, col=1)
    fig.update_yaxes(title_text="MON3 (ms)", row=3, col=1)

    return fig


def surpriseChart(df):
    s_bf_smooth = smooth_series(df["surprisebf"], SMOOTHING_WINDOW)
    s_cc_smooth = smooth_series(df["surprisecc"], SMOOTHING_WINDOW)
    s_mip_smooth = smooth_series(df["surprisemip"], SMOOTHING_WINDOW)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["timestep"], y=df["surprisebf"], mode="lines",
                              name="Bayes Factor Surprise", line=dict(width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=df["timestep"], y=df["surprisecc"], mode="lines",
                              name="Confidence-Corrected Surprise", line=dict(width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=df["timestep"], y=df["surprisemip"], mode="lines",
                              name="MIP Surprise", line=dict(width=1), opacity=0.5))

    fig.add_trace(go.Scatter(x=df["timestep"], y=s_bf_smooth, mode="lines",
                              name="Bayes Factor Surprise (Smoothed)", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df["timestep"], y=s_cc_smooth, mode="lines",
                              name="Confidence-Corrected Surprise (Smoothed)", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df["timestep"], y=s_mip_smooth, mode="lines",
                              name="MIP Surprise (Smoothed)", line=dict(width=2)))

    fig.update_layout(
        title="Surprise Over Time",
        xaxis_title="Timestep",
        yaxis_title="Surprise",
        legend_title="Surprise Types",
    )
    return fig


def surpriseChartNormalized(df):
    """Surprise measures standardised (z-score) for comparison: each has mean 0 and std 1.
    Skips the first 2*lookback rows so MIP cold-start does not skew the standardisation."""
    df = df.iloc[MIP_WARMUP_ROWS:].reset_index(drop=True)
    if df.empty:
        return go.Figure()
    s_bf = df["surprisebf"]
    s_cc = df["surprisecc"]
    s_mip = df["surprisemip"]

    def zscore(series):
        mu, sigma = series.mean(), series.std()
        if sigma == 0 or np.isnan(sigma):
            return pd.Series(0.0, index=series.index)
        return (series - mu) / sigma

    std_bf = zscore(s_bf)
    std_cc = zscore(s_cc)
    std_mip = zscore(s_mip)

    std_bf_smooth = smooth_series(std_bf, SMOOTHING_WINDOW)
    std_cc_smooth = smooth_series(std_cc, SMOOTHING_WINDOW)
    std_mip_smooth = smooth_series(std_mip, SMOOTHING_WINDOW)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df["timestep"], y=std_bf, mode="lines",
                              name="Bayes Factor Surprise", line=dict(width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=df["timestep"], y=std_cc, mode="lines",
                              name="Confidence-Corrected Surprise", line=dict(width=1), opacity=0.5))
    fig.add_trace(go.Scatter(x=df["timestep"], y=std_mip, mode="lines",
                              name="MIP Surprise", line=dict(width=1), opacity=0.5))

    fig.add_trace(go.Scatter(x=df["timestep"], y=std_bf_smooth, mode="lines",
                              name="Bayes Factor Surprise (Smoothed)", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df["timestep"], y=std_cc_smooth, mode="lines",
                              name="Confidence-Corrected Surprise (Smoothed)", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=df["timestep"], y=std_mip_smooth, mode="lines",
                              name="MIP Surprise (Smoothed)", line=dict(width=2)))

    y_min = min(std_bf.min(), std_cc.min(), std_mip.min())
    y_max = max(std_bf.max(), std_cc.max(), std_mip.max())
    fig.update_layout(
        title="Surprise Over Time (Standardised for Comparison)",
        xaxis_title="Timestep",
        yaxis_title="Standardised Surprise (z-score)",
        legend_title="Surprise Types",
        yaxis=dict(range=[y_min, y_max]),
    )
    return fig


def gammaChart(df):
    fig = go.Figure(
        data=go.Scatter(x=df["timestep"], y=df["gamma"], mode="lines", name="Learning Rate (Gamma)")
    )
    fig.update_layout(
        title="Learning Rate (Gamma) Over Time",
        xaxis_title="Timestep",
        yaxis_title="Learning Rate (Gamma)",
    )
    return fig


def mipChart(df):
    fig = go.Figure(
        data=go.Scatter(x=df["timestep"], y=df["surprisemip"], mode="lines", name="MIP")
    )
    fig.add_trace(go.Scatter(x=df["timestep"], y=df["mip_upper"], mode="lines",
                              name="MIP Upper Bound", line=dict(color="red", width=1)))
    fig.add_trace(go.Scatter(x=df["timestep"], y=df["mip_lower"], mode="lines",
                              name="MIP Lower Bound", line=dict(color="red", width=1)))
    fig.update_layout(
        title="MIP Over Time with Error Bounds",
        xaxis_title="Timestep",
        yaxis_title="MIP",
    )
    return fig


def buildChartsReport(df, mon1_threshold=20.0, mon2_threshold=20.0, mon3_threshold=20.0):
    """
    Build all charts and combine them into a single self-contained HTML page (Plotly's JS
    inlined once, so the report has no CDN/network dependency and no server to render each
    figure). Thresholds should match the config used for the run (solver.config
    mon1Threshold/mon2Threshold/mon3Threshold).

    :return: the combined report as an HTML string.
    """
    logger.info("Stage: Starting chart generation (MON1/2/3, surprise, gamma, MIP)")

    logger.info("Stage: Building MIP chart (MIP over time with bounds)")
    mip_fig = mipChart(df)

    logger.info("Stage: Building gamma chart (gamma over time)")
    gamma_fig = gammaChart(df)

    logger.info("Stage: Building surprise chart (BF, CC, MIP over time)")
    surprise_fig = surpriseChart(df)

    logger.info("Stage: Building normalized surprise chart")
    surprise_norm_fig = surpriseChartNormalized(df)

    logger.info(
        "Stage: Building MON1/MON2/MON3 satisfaction time-series plots "
        "(mon1Threshold={}, mon2Threshold={}, mon3Threshold={})",
        mon1_threshold, mon2_threshold, mon3_threshold,
    )
    satisfaction_df = df.filter(items=["timestep", "mon1sat", "mon2sat", "mon3sat"])
    satisfaction_fig = satisfactionPlots(
        satisfaction_df,
        mon1_threshold=mon1_threshold,
        mon2_threshold=mon2_threshold,
        mon3_threshold=mon3_threshold,
    )

    logger.info("Stage: Building MON1/MON2/MON3 satisfaction violin distributions")
    violins_fig = satisfactionViolins(satisfaction_df)

    sections = [
        ("MIP Over Time", mip_fig, True),
        ("Learning Rate (Gamma) Over Time", gamma_fig, False),
        ("Surprise Over Time", surprise_fig, False),
        ("Surprise Over Time (Standardised)", surprise_norm_fig, False),
        ("MON1/MON2/MON3 Satisfaction Over Time", satisfaction_fig, False),
        ("MON1/MON2/MON3 Satisfaction Distributions", violins_fig, False),
    ]

    parts = ["<html><head><title>RDMSim Charts</title></head><body>"]
    for title, fig, first in sections:
        parts.append(f"<h2>{title}</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs=("inline" if first else False)))
    parts.append("</body></html>")

    logger.info("Stage: Chart generation complete")
    return "\n".join(parts)


def writeChartsReport(df, mon1_threshold, mon2_threshold, mon3_threshold, output_dir):
    """Build the combined charts report and write it to <output_dir>/charts.html.

    :return: absolute path to the written report.
    """
    html = buildChartsReport(df, mon1_threshold, mon2_threshold, mon3_threshold)
    report_path = os.path.abspath(os.path.join(output_dir, "charts.html"))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Stage: Wrote combined charts report to {}", report_path)
    return report_path


def createCharts(df, mon1_threshold, mon2_threshold, mon3_threshold, output_dir):
    """
    Build all charts, write them to a single combined HTML report in output_dir, and try
    (best-effort, non-fatal) to open it in the browser exactly once. The report is written to
    disk regardless of whether the browser launch succeeds.
    """
    report_path = writeChartsReport(df, mon1_threshold, mon2_threshold, mon3_threshold, output_dir)
    try:
        webbrowser.open(f"file://{report_path}")
    except Exception as e:
        logger.warning("Could not auto-open charts report ({}); open it manually: {}", e, report_path)


def _default_data_dir():
    """Resolve default output directory for chart data (when --output-dir not provided)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "output_dir")


def getData(folder_path=None):
    """
    Reads every monitored-output file in folder_path and merges them into a single
    per-timestep dataframe, inferring each file's role from its column count and filename
    (matching the convention used by RDMSim_TAAS.java's logging): 2-column files are
    "timestep value" and merge directly on timestep; MIPBounds.txt is the one 3-column
    exception ("timestep mip_lower mip_upper").
    """
    if folder_path is None:
        folder_path = _default_data_dir()
    else:
        folder_path = os.path.abspath(folder_path)

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Output directory not found: {folder_path}")

    logger.info("Stage: Scanning {} for monitorable/surprise/gamma output files", folder_path)
    dfs_2 = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            continue
        if filename in SKIP_FILES or not filename.endswith(".txt"):
            continue
        if os.path.getsize(file_path) == 0:
            logger.warning("Skipping empty file: {}", filename)
            continue
        try:
            df = pd.read_csv(file_path, sep=r"\s+", header=None, on_bad_lines="skip", engine="python")
        except pd.errors.EmptyDataError:
            logger.warning("Skipping empty file: {}", filename)
            continue
        except pd.errors.ParserError as e:
            logger.warning("Skipping file {} due to parsing error: {}", filename, e)
            continue
        except Exception as e:
            logger.warning("Skipping file {} due to error: {}", filename, e)
            continue

        if df.empty:
            logger.warning("Skipping file {} - DataFrame is empty after reading", filename)
            continue

        file_col_name = filename.split(".")[0].lower()
        if df.shape[1] == 3 and file_col_name == "mipbounds":
            df.columns = ["timestep", "mip_lower", "mip_upper"]
            logger.info("Read {} ({} rows)", filename, len(df))
            dfs_2.append(df)
        elif df.shape[1] == 2:
            df.columns = ["timestep", file_col_name]
            logger.info("Read {} ({} rows)", filename, len(df))
            dfs_2.append(df)
        else:
            logger.debug("Skipping file {} with unexpected shape {}", filename, df.shape)

    if not dfs_2:
        raise ValueError(f"No usable monitorable output files found in {folder_path}")

    max_rows = max(len(df) for df in dfs_2)
    dfs_all = reduce(lambda left, right: pd.merge(left, right, on="timestep"), dfs_2)
    logger.info("Stage: Merged {} files into a single dataframe ({} rows)", len(dfs_2), len(dfs_all))
    if len(dfs_all) < max_rows:
        logger.warning(
            "Merged dataframe ({} rows) is shorter than the longest input file ({} rows) - "
            "the inner join on 'timestep' dropped rows. Check for a stray/short file in {}",
            len(dfs_all), max_rows, folder_path,
        )
    return dfs_all


def run(data_dir=None, mon1_threshold=20.0, mon2_threshold=20.0, mon3_threshold=20.0):
    """
    Run the full chart generation pipeline.
    :param data_dir: Directory containing solver output (MON1Sat.txt, gamma.txt, etc.). If None,
                     uses output_dir relative to this script. When invoked by RDMSim_TAAS.java,
                     this is set from solver.config's outputDirectory.
    :param mon1_threshold: MON1 (active links) threshold from solver.config.
    :param mon2_threshold: MON2 (bandwidth consumption, GB/s) threshold from solver.config.
    :param mon3_threshold: MON3 (time to write, ms) threshold from solver.config.
    """
    if data_dir is not None:
        data_dir = os.path.abspath(data_dir)
        logger.info("Stage: Using data directory from config: {}", data_dir)
    else:
        data_dir = _default_data_dir()
    logger.info("Stage: Chart generation pipeline started")
    df_all = getData(data_dir)
    createCharts(
        df_all,
        mon1_threshold=mon1_threshold,
        mon2_threshold=mon2_threshold,
        mon3_threshold=mon3_threshold,
        output_dir=data_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate charts from RDMSim_TAAS solver output (MON1/2/3, surprise, gamma).")
    parser.add_argument(
        "--output-dir", type=str, default=None, metavar="PATH",
        help="Directory containing solver output (MON1Sat.txt, MON2Sat.txt, MON3Sat.txt, gamma.txt, etc.). "
             "Defaults to output_dir relative to this script. When called by RDMSim_TAAS.java, this is set "
             "from solver.config's outputDirectory.",
    )
    parser.add_argument(
        "--mon1-threshold", type=float, default=20.0, metavar="VALUE",
        help="MON1 (active links) threshold from solver.config. Default 20.",
    )
    parser.add_argument(
        "--mon2-threshold", type=float, default=20.0, metavar="VALUE",
        help="MON2 (bandwidth consumption, GB/s) threshold from solver.config. Default 20.",
    )
    parser.add_argument(
        "--mon3-threshold", type=float, default=20.0, metavar="VALUE",
        help="MON3 (time to write, ms) threshold from solver.config. Default 20.",
    )
    args = parser.parse_args()
    run(
        data_dir=args.output_dir,
        mon1_threshold=args.mon1_threshold,
        mon2_threshold=args.mon2_threshold,
        mon3_threshold=args.mon3_threshold,
    )
