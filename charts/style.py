"""Shared look for every figure: palette, panel grid, reference lines.

Colour does one job per figure family:
* learning config -> categorical (fixed order, never cycled), CVD-validated set
* lambda          -> sequential single-hue blue ramp, light (lambda=0) to dark (lambda=20)
* fixed baselines -> greys, so they read as reference rather than as competitors
* NFR threshold   -> red line, as in the original single-run charts
Several config hues sit below 3:1 contrast on white, so identity never rests on colour
alone: every figure has a legend, and configs also differ by marker symbol.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import analyse

LEARNING_ORDER = list(analyse.LEARNING_ORDER)   # static, bayes, smile-CC, smile-BF, smile-MIP
LAMBDAS = [0.0, 2.0, 5.0, 10.0, 20.0]
SCENARIOS = list(range(7))

# Validated with the dataviz skill's validate_palette.js (light): CVD worst adjacent
# Delta-E 9.1, normal-vision floor 19.6 -> pass; contrast WARN -> legends + symbols.
CONFIG_COLOURS = {
    "static": "#2a78d6",     # blue
    "bayes": "#eb6834",      # orange
    "smile-CC": "#1baf7a",   # aqua
    "smile-BF": "#eda100",   # yellow
    "smile-MIP": "#e87ba4",  # magenta
}
CONFIG_SYMBOLS = {"static": "circle", "bayes": "square", "smile-CC": "diamond",
                  "smile-BF": "triangle-up", "smile-MIP": "x"}
CONFIG_LABELS = {"static": "static (frozen prior)", "bayes": "Bayes (Dirichlet +1)",
                 "smile-CC": "SMiLe-CC", "smile-BF": "SMiLe-BF", "smile-MIP": "SMiLe-MIP"}
# Axis tick labels where five configs and three baselines share one panel width.
CONFIG_SHORT = {"static": "static", "bayes": "Bayes", "smile-CC": "CC", "smile-BF": "BF",
                "smile-MIP": "MIP"}

# Blue ramp steps 250 -> 650: magnitude of lambda, light to dark.
LAMBDA_COLOURS = {0.0: "#86b6ef", 2.0: "#5598e7", 5.0: "#2a78d6", 10.0: "#1c5cab", 20.0: "#104281"}
LAMBDA_SYMBOLS = {0.0: "circle", 2.0: "square", 5.0: "diamond", 10.0: "triangle-up", 20.0: "star"}

BASELINE_COLOURS = {"mst": "#52514e", "rt": "#898781", "random": "#b9b7b0"}
BASELINE_LABELS = {"mst": "fixed MST", "rt": "fixed RT", "random": "fixed random"}

THRESHOLD_COLOUR = "red"
REFERENCE_COLOUR = "#52514e"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e1"

# NFRs: per-step column, results.csv rate column, label, unit, satisfied direction.
NFRS = [
    {"key": "MON1", "series": "mon1", "rate": "MON1_rate", "mean": "mean_links",
     "label": "MON1 - active links (MR)", "quantity": "active links", "unit_short": "links",
     "unit": "active links", "op": ">=", "goal": "maximise", "nfr_colour": "lightseagreen"},
    {"key": "MON2", "series": "mon2", "rate": "MON2_rate", "mean": "mean_bw",
     "label": "MON2 - bandwidth (MC)", "quantity": "bandwidth consumption", "unit_short": "Gbps",
     "unit": "bandwidth consumption (Gbps)", "op": "<=", "goal": "minimise", "nfr_colour": "orange"},
    {"key": "MON3", "series": "mon3", "rate": "MON3_rate", "mean": "mean_ttw",
     "label": "MON3 - time to write (MP)", "quantity": "time to write", "unit_short": "ms",
     "unit": "time to write (ms)", "op": "<=", "goal": "minimise", "nfr_colour": "mediumpurple"},
]
OP_SYMBOL = {">=": "≥", "<=": "≤"}

FONT = dict(family="Arial, Helvetica, sans-serif", size=13, color=INK)
GRID_WIDTH, GRID_HEIGHT = 1500, 760


def lam_label(lam: float) -> str:
    return f"λ={lam:g}"


def goal_text(index: int, thresholds) -> str:
    """e.g. 'MON1 met when active links ≥ 105 (35% of 300 links): maximise active links'.

    The percentage is what configuration.json specifies; the absolute value is what RDMSim
    scores each timestep against.
    """
    nfr = NFRS[index]
    value = thresholds.for_nfr(index)
    pct = thresholds.percent(index)
    cap = thresholds.capacity[index]
    share = (f" ({pct:g}% of {cap:g} {nfr['unit_short']})"
             if np.isfinite(pct) and np.isfinite(cap) else "")
    return (f"{nfr['key']} met when {nfr['quantity']} {OP_SYMBOL[nfr['op']]} {value:g} "
            f"{nfr['unit_short']}{share}: {nfr['goal']} {nfr['quantity']}")


def with_subtitle(title: str, subtitle: str) -> str:
    """Two-line figure title: the claim, then a smaller line of definitions."""
    return f"{title}<br><span style='font-size:13px'>{subtitle}</span>"


def rgba(hex_colour: str, alpha: float) -> str:
    """'#rrggbb' (or a CSS name we use) -> 'rgba(r,g,b,a)'."""
    named = {"lightseagreen": "#20b2aa", "orange": "#ffa500", "mediumpurple": "#9370db",
             "red": "#ff0000"}
    h = named.get(hex_colour, hex_colour).lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def panel(scenario: int) -> tuple[int, int]:
    """(row, col) of a scenario in the 2x4 grid; the 8th cell holds the legend."""
    return scenario // 4 + 1, scenario % 4 + 1


def base_layout(fig: go.Figure, title: str, width: int, height: int) -> go.Figure:
    fig.update_layout(
        template="plotly_white", font=FONT, width=width, height=height,
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=17)),
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=70, r=30, t=105, b=60),
        legend=dict(font=dict(size=12), bgcolor="rgba(255,255,255,0.9)",
                    tracegroupgap=4),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=INK_SECONDARY, ticks="outside",
                     tickcolor=INK_SECONDARY, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=INK_SECONDARY, zeroline=False)
    return fig


def scenario_grid(title: str, x_title: str, y_title: str, descriptions: dict[int, str],
                  shared_y: bool = True, height: int = GRID_HEIGHT) -> go.Figure:
    """Seven scenario panels in a 2x4 grid; the empty 8th cell holds the legend."""
    titles = [f"<b>S{s}</b> · {descriptions.get(s, '')}" for s in SCENARIOS] + [""]
    fig = make_subplots(rows=2, cols=4, subplot_titles=titles, shared_yaxes=shared_y,
                        horizontal_spacing=0.045, vertical_spacing=0.16)
    base_layout(fig, title, GRID_WIDTH, height)
    for annotation in fig.layout.annotations:
        annotation.font = dict(size=12, color=INK)

    for s in SCENARIOS:
        row, col = panel(s)
        if row == 2 or col == 4:          # bottom panels, and S3 which has none below it
            fig.update_xaxes(title_text=x_title, row=row, col=col)
        if col == 1:
            fig.update_yaxes(title_text=y_title, row=row, col=col)
    fig.update_xaxes(visible=False, row=2, col=4)
    fig.update_yaxes(visible=False, row=2, col=4)

    legend_cell = fig.get_subplot(2, 4)
    fig.update_layout(legend=dict(x=legend_cell.xaxis.domain[0] + 0.01,
                                  y=legend_cell.yaxis.domain[1],
                                  xanchor="left", yanchor="top"))
    return fig


def add_line_band(fig: go.Figure, x, mean, sd, colour: str, name: str, row: int, col: int,
                  showlegend: bool, legendgroup: str, symbol: str | None = None,
                  marker: bool = False, width: float = 2.0, dash: str | None = None) -> None:
    """Mean line with a +/-1 SD band (two invisible bounds filled between)."""
    x = np.asarray(x)
    mean = np.asarray(mean, dtype=float)
    sd = np.asarray(sd, dtype=float)
    keep = np.isfinite(mean)
    x, mean, sd = x[keep], mean[keep], np.nan_to_num(sd[keep])
    if x.size == 0:
        return
    fig.add_trace(go.Scatter(x=x, y=mean + sd, mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False, legendgroup=legendgroup),
                  row=row, col=col)
    fig.add_trace(go.Scatter(x=x, y=mean - sd, mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor=rgba(colour, 0.16),
                             hoverinfo="skip", showlegend=False, legendgroup=legendgroup),
                  row=row, col=col)
    fig.add_trace(go.Scatter(
        x=x, y=mean, mode="lines+markers" if marker else "lines", name=name,
        line=dict(color=colour, width=width, dash=dash),
        marker=dict(symbol=symbol or "circle", size=8, color=colour,
                    line=dict(width=1, color="white")),
        customdata=np.stack([sd], axis=-1),
        hovertemplate=f"{name}<br>x=%{{x}}<br>mean=%{{y:.3f}} ± %{{customdata[0]:.3f}} SD<extra></extra>",
        showlegend=showlegend, legendgroup=legendgroup),
        row=row, col=col)


def add_threshold(fig: go.Figure, value: float, op: str, row: int, col: int,
                  label: bool = True) -> None:
    """Red NFR threshold line, labelled with its satisfied direction."""
    # Annotation kwargs only when labelled: passing annotation_font without text makes
    # Plotly draw its "new text" placeholder.
    annotation = (dict(annotation_text=f"threshold {OP_SYMBOL.get(op, op)} {value:g}",
                       annotation_position="top left",
                       annotation_font=dict(size=10, color=INK_SECONDARY)) if label else {})
    fig.add_hline(y=value, line=dict(color=THRESHOLD_COLOUR, width=1.6), row=row, col=col,
                  **annotation)


def add_analytic_refs(fig: go.Figure, scenario: int, nfr_index: int, row: int, col: int) -> None:
    """Dashed always-MST / dotted always-RT satisfaction rates (analytic, analyse.TRUTH)."""
    truth = analyse.TRUTH.get(scenario)
    if not truth:
        return
    fig.add_hline(y=truth["mst"][nfr_index], line=dict(color=REFERENCE_COLOUR, width=1.2, dash="dash"),
                  row=row, col=col)
    fig.add_hline(y=truth["rt"][nfr_index], line=dict(color=REFERENCE_COLOUR, width=1.2, dash="dot"),
                  row=row, col=col)


def add_reference_legend(fig: go.Figure) -> None:
    """Legend keys for the analytic reference lines (hlines have no legend entry)."""
    for dash, label in (("dash", "always-MST (analytic)"), ("dot", "always-RT (analytic)")):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=label,
                                 line=dict(color=REFERENCE_COLOUR, width=1.2, dash=dash),
                                 showlegend=True, legendgroup="refs"))


def add_threshold_legend(fig: go.Figure) -> None:
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="NFR threshold",
                             line=dict(color=THRESHOLD_COLOUR, width=1.6),
                             showlegend=True, legendgroup="threshold"))
