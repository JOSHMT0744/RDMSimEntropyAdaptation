"""Charting for RDMSim_TAAS.

Two consumers share this package:

* ``single_run`` - the per-run report (``<run>/charts.html``) that RDMSim_TAAS.java asks
  ``createCharts.py`` for at the end of every run.
* ``grid_figures``, ``timeseries``, ``violins`` - cross-arm paper figures built from the
  ablation grid (``experiments/results.csv`` plus the per-cell logs), each one mean +/- 1 SD
  across seeds, written as HTML and PNG by ``export``.

``createCharts.py`` at the project root is the CLI over both.
"""
