"""Write figures as HTML (interactive) and PNG (3x).

HTML is written immediately. PNGs are queued and rendered in one batch at ``flush()``:
kaleido drives a headless Chrome, and starting it once for ~100 figures instead of once per
image is the difference between a minute and a quarter of an hour. If kaleido or Chrome is
missing, the HTML still lands and a single warning says how to fix it. (Vector copies can
still be had per figure from the HTML's camera button, which is set to SVG.)

All pages load one shared ``plotly.min.js`` at the figures root rather than inlining 4.6 MB
of JavaScript into every page.
"""

from __future__ import annotations

import html
import os
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from loguru import logger
from plotly.offline import get_plotlyjs

PNG_SCALE = 3


class Exporter:
    def __init__(self, root: str | Path, static: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.static = static
        self.pending: list[tuple[go.Figure, Path]] = []
        self.written: list[Path] = []
        self._plotlyjs = self.root / "plotly.min.js"
        if not self._plotlyjs.exists():
            self._plotlyjs.write_text(get_plotlyjs(), encoding="utf-8")

    def save(self, fig: go.Figure, stem: str) -> Path:
        """Write ``<root>/<stem>.html`` now; queue the .png for ``flush()``."""
        target = self.root / stem
        target.parent.mkdir(parents=True, exist_ok=True)
        html_path = target.with_suffix(".html")
        script = os.path.relpath(self._plotlyjs, html_path.parent).replace(os.sep, "/")
        fig.write_html(html_path, include_plotlyjs=script, full_html=True,
                       config={"displaylogo": False,
                               "toImageButtonOptions": {"format": "svg"}})
        self.written.append(html_path)
        if self.static:
            self.pending.append((fig, target))
        return html_path

    def flush(self) -> None:
        """Render every queued figure to PNG in one Chrome session."""
        if not self.pending:
            return
        figs = [fig for fig, _ in self.pending]
        stems = [stem for _, stem in self.pending]
        logger.info("Stage: rendering {} figure(s) to PNG", len(figs))
        # Batch export does not pick up layout.width/height; without these every figure
        # renders at the 700x500 default and multi-panel layouts collapse.
        widths = [fig.layout.width or 1200 for fig in figs]
        heights = [fig.layout.height or 700 for fig in figs]
        try:
            pio.write_images(figs, [s.with_suffix(".png") for s in stems], scale=PNG_SCALE,
                             width=widths, height=heights)
        except Exception as exc:
            logger.warning(
                "Static export failed ({}). HTML figures were still written. Install the "
                "exporter with `pip install \"kaleido>=1,<2\"` and, if no Chrome is present, "
                "run `plotly_get_chrome` once.", exc)
        self.pending.clear()

    def write_index(self, title: str = "RDMSim ablation figures") -> Path:
        """index.html at the root: every figure, grouped by family, with PNG thumbnails."""
        groups: dict[str, list[Path]] = {}
        for path in sorted(self.root.rglob("*.html")):
            if path.name == "index.html":
                continue
            family = path.relative_to(self.root).parts[0]
            groups.setdefault(family, []).append(path)

        parts = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>{html.escape(title)}</title>",
            "<style>body{font-family:Arial,sans-serif;margin:24px;background:#fcfcfb;color:#0b0b0b}"
            "h2{margin-top:32px;border-bottom:1px solid #ddd}"
            ".grid{display:flex;flex-wrap:wrap;gap:14px}"
            ".card{width:300px;font-size:13px}.card img{width:300px;border:1px solid #ddd;background:#fff}"
            ".card a{color:#1c5cab;text-decoration:none}</style></head><body>",
            f"<h1>{html.escape(title)}</h1>",
            "<p>Each figure: mean ± 1 SD across 5 seeds (SD within scenario), or a 95% CI of "
            "the CRN-paired difference on comparison plots. Click for the interactive version; "
            "a 3× PNG sits next to each HTML file.</p>",
        ]
        for family, paths in groups.items():
            parts.append(f"<h2>{html.escape(family)} ({len(paths)})</h2><div class='grid'>")
            for path in paths:
                rel = path.relative_to(self.root).as_posix()
                png = path.with_suffix(".png")
                thumb = (f"<img loading='lazy' src='{html.escape(png.relative_to(self.root).as_posix())}'>"
                         if png.exists() else "")
                label = rel[len(family) + 1:-5]
                parts.append(f"<div class='card'><a href='{html.escape(rel)}'>{thumb}<br>"
                             f"{html.escape(label)}</a></div>")
            parts.append("</div>")
        parts.append("</body></html>")
        index = self.root / "index.html"
        index.write_text("\n".join(parts), encoding="utf-8")
        return index
