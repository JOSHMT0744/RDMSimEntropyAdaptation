# Running the ablation studies

This guide covers the RDMSim ablation grid: what the arms are, how to run any slice of
them, and how to tell a correct run from a broken one. It assumes nothing beyond a built
project.

- [1. Concepts](#1-concepts)
- [2. Quickstart](#2-quickstart)
- [3. CLI reference](#3-cli-reference)
- [4. Recipes](#4-recipes)
- [5. Sizing: time and memory](#5-sizing-time-and-memory)
- [6. Output layout and CSV schema](#6-output-layout-and-csv-schema)
- [7. Troubleshooting](#7-troubleshooting)
- [8. Verification checklist](#8-verification-checklist)
- [9. Extending the grid](#9-extending-the-grid)

---

## 1. Concepts

**Scenario** — one of RDMSim's seven uncertainty profiles, S0–S6. A scenario sets six
`[min,max]` deviation bands (links / bandwidth / writing-time, per topology) in
`config_log_files/configuration.json`. Those bands are redrawn every timestep, so the
environment genuinely varies step to step; what stays fixed for a run is the band the draw
comes from.

**Arm** — one algorithm configuration. There are 28 per scenario:

| Arm family | Count | `staticModel` | `useSurpriseUpdating` | `surpriseMeasureForGamma` | `fixedPolicy` |
|---|---|---|---|---|---|
| `static` × λ∈{0,2,5,10,20} | 5 | `true` | — | — | — |
| `bayes` × λ | 5 | `false` | `false` | — | — |
| `smile-CC` × λ | 5 | `false` | `true` | `CC` | — |
| `smile-BF` × λ | 5 | `false` | `true` | `BF` | — |
| `smile-MIP` × λ | 5 | `false` | `true` | `MIP` | — |
| `fixed_mst`, `fixed_rt`, `fixed_random` | 3 | — | — | — | the policy |

**What each arm isolates.** The ladder is nested, not a free factorial:

- `static` vs `bayes` — does the learning loop do any work at all? `static` freezes the
  planning model at the seeded Dirichlet prior. (The belief filter still advances; only the
  *model* is frozen. Freezing the belief too would test something else entirely.)
- `bayes` vs `smile-*` — does surprise-weighting beat plain Dirichlet `+1`?
- `smile-CC` vs `smile-BF` vs `smile-MIP` — which surprise signal? These exist only when
  surprise updating is on: `useSurpriseUpdating=false` short-circuits before γ is computed,
  so a "static + CC" arm is not a thing.
- λ = 0 vs λ > 0 — does entropy regularisation help? λ=0 is a genuine Perseus control:
  `lambda > 0.0` guards both the ER backup (`src/solver/ERPerseus.java:295`) and the policy
  softmax, so at λ=0 the algorithm *is* plain Perseus at both levels.

**Cell** — one run: a (scenario, arm, seed) triple. The full grid is 7 × 28 × 5 = **980
cells**, of which 875 use the solver and 105 are near-free baselines.

**CRN (common random numbers)** — RDMSim draws monitorables with `ThreadLocalRandom`, which
cannot be seeded from outside the jar, so two arms would otherwise face different
environments and any difference between them would be confounded with noise. With
`-Dcrn=true` the environment is pre-drawn for every timestep **and both topologies** from a
`(scenario, seed)` stream, and every arm in that cell consumes the same stream indexed by
`(timestep, topology)`. Arms are then perfectly paired and differ only where their action
paths diverge; drawing both topologies at every step is what stops one arm's choices from
shifting what another arm sees. This also makes runs reproducible, which they are not
otherwise. Implementation: `src/remotemirroring/CRNStream.java`.

**Initial topology.** RDMSim chooses the t=0 topology when it loads its config, from the
JSON's `current_scenario`: MST for S0/S1/S4, RT for S2/S5, and an unseedable coin flip for
S3/S6. A `-Dscenario` override arrives too late to change that, and the JSON is set to S3,
so every cell would otherwise start on a random topology — and since the first observation
seeds the belief and the first action, arms would only be paired about half the time. The
harness therefore re-picks the start from the scenario actually in force whenever
`-Dscenario` or `-Dcrn` is given (`ScenarioOverride.applyInitialTopology`). S3/S6 keep their
coin flip, but seeded by (scenario, seed), so every arm in a cell starts the same way. For
the default seeds:

| | 222 | 223 | 224 | 225 | 226 |
|---|---|---|---|---|---|
| S3 | RT | MST | RT | MST | RT |
| S6 | MST | MST | RT | MST | MST |

### Ground truth

RDMSim is **memoryless given the action**: `generateMonitorables` redraws
`raw_links ~ U[topology.min, topology.max]` from scratch on every call with no dependence on
the previous state, so `P(s'|s,a) = P(s'|a)`. The per-action satisfaction rates are
therefore computable in closed form, and any policy's long-run rate is a convex combination
of two fixed endpoints weighted by its RT share.

Analytic rates (400k-draw Monte-Carlo, ±0.002), with the published thresholds
(MON1 ≥ 105 links, MON2 ≤ 3600 Gbps, MON3 ≤ 2700 ms):

| Scenario | Action | MON1 | MON2 | MON3 | All three |
|---|---|---|---|---|---|
| S0 | MST / RT | 0.464 / **1.000** | **0.811** / 0.068 | **0.929** / 0.393 | **0.254** / 0.045 |
| S1 | MST / RT | 0.374 / **1.000** | **0.903** / 0.069 | **0.976** / 0.394 | **0.266** / 0.046 |
| S2 | MST / RT | 0.464 / **1.000** | **0.811** / 0.016 | **0.929** / 0.269 | **0.256** / 0.009 |
| S3 | MST / RT | 0.374 / **1.000** | **0.903** / 0.016 | **0.976** / 0.269 | **0.266** / 0.009 |
| S4 | MST / RT | 0.374 / **1.000** | **0.823** / 0.068 | **0.937** / 0.392 | **0.178** / 0.045 |
| S5 | MST / RT | 0.464 / **1.000** | **0.812** / 0.078 | **0.929** / 0.413 | **0.256** / 0.053 |
| S6 | MST / RT | 0.373 / **1.000** | **0.824** / 0.079 | **0.936** / 0.414 | **0.177** / 0.055 |

Use this table as the correctness check. Three things follow from it:

1. **A `fixed_mst` or `fixed_rt` arm must reproduce its row** within sampling error. If it
   does not, something upstream is wrong — wrong scenario applied, CRN stream misaligned, or
   the satisfaction tally drifting from the monitorables. `experiments/analyse.py` reports
   this continuously as the mixture-line residual.
2. **RT satisfies MON1 with probability 1.000 in every scenario** (150 links less 12% is
   still above 105), so MON1 carries no decision information under the published thresholds.
3. **The optimal action never flips across S0–S6.** MST wins MON2, MON3 and joint
   satisfaction everywhere; RT wins MON1 everywhere. The scenarios vary severity, not the
   answer. They are seven environmental conditions to compare across, which is how they are
   framed in the paper.

A curiosity worth knowing: S5 makes RT slightly *better* (MON2 0.068 → 0.078). Deviating
links **subtracts** links, which relieves bandwidth and write-time faster than their own
inflation adds to them.

---

## 2. Quickstart

Build first. `javac` needs every source file; the classpath separator is `;` on Windows and
`:` elsewhere.

```bash
javac -d bin -cp "libraries/*" $(find src -name "*.java")
```

Confirm the CRN generator still matches the jar (fast, and the one check that gates
everything else):

```bash
java -cp "bin;libraries/*" remotemirroring.CRNVerify
# expect: PASS: CRNStream reproduces the analytic marginals in all 14 cells.
```

Smoke-test a single cell at a short horizon:

```bash
python experiments/run_ablation.py --scenarios 3 --arms smile-MIP:0 --baselines none --seeds 222 --timesteps 50
```

See what the full grid would cost before committing to it:

```bash
python experiments/run_ablation.py --dry-run
```

Run the scenario-3 pilot — all 28 arms, 5 seeds — and size the rest from it:

```bash
python experiments/run_ablation.py --scenarios 3 --seeds 222-226 --jobs 4 --heap 1g
python experiments/analyse.py
```

Then the full grid, in batches so it can be interrupted:

```bash
python experiments/run_ablation.py --batch-size 100 --jobs 4 --heap 1g   # repeat
python experiments/run_ablation.py --list                               # until nothing pending
python experiments/analyse.py
python createCharts.py --grid                                           # figures -> figures/
```

All commands run from the **project root** — config and domain lookups are relative to the
working directory.

---

## 3. CLI reference

`python experiments/run_ablation.py [options]`. List options accept comma lists and `a-b`
ranges (`--scenarios 0,3,5-6`) plus `all`. The selection is the cross of the given sets,
minus `--exclude`.

### Cell selection

| Flag | Default | Effect |
|---|---|---|
| `--scenarios` | `all` | which of S0–S6 |
| `--learning` | `all` | `static`, `bayes`, `smile-CC`, `smile-BF`, `smile-MIP` |
| `--lambdas` | `0,2,5,10,20` | λ values for the cross (no ranges — the useful values are not evenly spaced) |
| `--baselines` | `all` | `mst`, `rt`, `random`, or `none` |
| `--arms` | — | explicit `learning:lambda` pairs, e.g. `smile-MIP:5,bayes:0`; **overrides** `--learning`/`--lambdas` |
| `--exclude` | — | subtract arms or whole scenarios: `static:20,scenario=0` |
| `--seeds` | `222-226` | seed list or range |
| `--timesteps` | `1000` | horizon per run |
| `--crn` / `--no-crn` | `--crn` | common random numbers on/off |

### Resource bounds

| Flag | Default | Effect |
|---|---|---|
| `--jobs` | half the cores | concurrent JVMs |
| `--heap` | `1g` | per-JVM `-Xmx`. Explicit on purpose: the JVM default is a quarter of system RAM, and N of those at once will thrash |
| `--max-memory` | 75% of RAM | budget for `jobs × heap`; the driver **refuses to start** above it and suggests a `--jobs` that fits |
| `--force-memory` | off | warn instead of refusing |
| `--batch-size` | unlimited | run at most this many cells, then exit resumably |
| `--max-wall` | unlimited | stop *launching* after this long (`6h`, `90m`, `600`); in-flight cells finish |
| `--timeout` | `3600` (s) | per-cell kill; the cell is marked `timeout`, not silently lost |
| `--retries` | `1` | relaunch a failed cell this many times before marking it `failed` |
| `--stop-on-failure` | off | abandon the batch on the first unrecoverable cell |
| `--allow-sleep` | off | don't ask Windows to stay awake while the batch runs (see [§7](#7-troubleshooting)) |

### Inspection and bookkeeping

| Flag | Effect |
|---|---|
| `--list` | print selected cells and their status; run nothing |
| `--dry-run` | as `--list`, plus wall-clock and peak-memory estimates |
| `--verbose` / `-v` | list every cell, not just counts |
| `--force` | re-run cells already marked `done` (default: skip them) |
| `--aggregate-only` | rebuild the CSV from cells on disk; launch nothing |
| `--order` | `interleaved` (default), `sequential`, `random` |
| `--keep-logs` | `all` (default), `summary`, `none` — see [§6](#6-output-layout-and-csv-schema) |
| `--output-root`, `--csv`, `--manifest`, `--template` | paths; defaults under `experiments/` |
| `--java` | JVM to launch. Default: the real JVM behind `java` on PATH, found via its `java.home` (see [§7](#7-troubleshooting)) |

**Interrupting.** Ctrl+C is safe: in-flight cells are killed, recorded as `pending`, and
the CSV is rebuilt from what finished; the driver exits with code 130. Re-run the same
command to resume. Retries are a per-invocation budget, so a cell that failed in an earlier
sitting gets its full `--retries` again.

`--order interleaved` rotates across scenarios, so stopping a batch early leaves every
scenario partially populated rather than S0–S1 complete and S2–S6 empty.

### The manifest

`experiments/manifest.json` records each cell's parameters, status
(`pending`/`running`/`done`/`failed`/`timeout`), wall-clock, peak RSS and the exact command
line. Resume reads the manifest rather than probing for `runSummary.txt`, so a cell killed
mid-write is re-run instead of being silently counted as complete. A stale `running` entry
left by a killed driver is reset to `pending` at startup, with a message.

**Resume is horizon-aware.** A cell completed at a different `--timesteps` does not count as
done and is re-run, with a message saying which and at what horizon it had run. Satisfaction
rates are means over timesteps, so a 50-step smoke result cannot stand in for a 1000-step
cell — without this, an early smoke test would leave silent holes in the real grid.

---

## 4. Recipes

**Pilot one scenario, then add others later.** Nothing is re-run; the manifest remembers.

```bash
python experiments/run_ablation.py --scenarios 3 --seeds 222-226
python experiments/run_ablation.py --scenarios 0,6 --seeds 222-226   # adds, does not redo
```

**Low-memory machine.** Prefer fewer jobs over a smaller heap: starving the JVM makes runs
slower, not smaller.

```bash
python experiments/run_ablation.py --jobs 2 --heap 768m
```

**Walk the grid over several sittings.** Either cap cells or cap time:

```bash
python experiments/run_ablation.py --batch-size 100     # ~100 cells, then exits
python experiments/run_ablation.py --max-wall 6h        # stops launching after 6h
```

**Re-run one suspect cell.** `--force` is needed, or it will be skipped as done.

```bash
python experiments/run_ablation.py --scenarios 4 --arms bayes:10 --seeds 224 --baselines none --force
```

**Add a sixth seed to a completed grid.**

```bash
python experiments/run_ablation.py --seeds 227
```

**Just the λ sweep for one learning config, across all scenarios.**

```bash
python experiments/run_ablation.py --learning smile-MIP --baselines none
```

**Unwrapped confirmation runs** — show the CRN wrapper did not change the exemplar's
behaviour:

```bash
python experiments/run_ablation.py --no-crn --arms smile-MIP:0 --baselines none --seeds 222
```

**Rebuild the CSV after manual cleanup**, without launching anything:

```bash
python experiments/run_ablation.py --aggregate-only
```

**Drop per-step logs to save disk** once a slice is analysed:

```bash
python experiments/run_ablation.py --scenarios 3 --keep-logs summary --aggregate-only
```

---

## 5. Sizing: time and memory

Measured on this machine (8 cores, 31.6 GiB RAM, JDK 25, `--heap 1g`, scenario 3, one job
at a time):

| Arm | 200 steps | 400 steps | 1000 steps | mean \|V\| per solve |
|---|---|---|---|---|
| `smile-MIP` λ=5 | ~9 s | ~21 s | **70 s** | 2.1 |
| `smile-MIP` λ=0 | — | — | **122 s** | 1.4–2.6 |
| `static` λ=0 | — | — | **962 s** | 10.9–11.1 |
| `fixed_mst` baseline | ~3 s | ~3 s | **13 s** | n/a (no solve) |

Two things to plan around.

**Run time is superlinear in the horizon.** Doubling 200 → 400 steps more than doubles the
wall-clock, because the POMDP is re-solved from scratch every timestep. Do not extrapolate
linearly from a short pilot — measure at the horizon you intend to use. `--dry-run` uses
measured per-cell times from the manifest once any cells have completed, and falls back to a
nominal estimate before that.

**The `static` arm dominates grid cost**, by an order of magnitude — 962 s against 70 s for
`smile-MIP` λ=5, a 14x gap. Perseus' cost scales with the size of the value function it
carries, and the `|V|` column explains the gap: a *learned* model sharpens as its Dirichlet
counts concentrate, which collapses the value function to 1–3 alpha vectors, while the
frozen prior stays diffuse and needs ~11. So the learning loop does not merely change the
policy — **it makes planning cheaper**, which is worth a sentence in the paper and is the
reason the bottom rung of the ladder is the expensive one.

**Whole-grid cost.** With the measurements above, `--dry-run --jobs 4 --timesteps 1000`
estimates roughly **70 h serial / 17 h at 4 jobs** for all 980 cells. The 175 `static` cells
are about **two thirds of that** (46 h) on their own. If the grid needs trimming, drop
`static` λ values first — `--exclude static:2,static:10,static:20` saves ~28 h, and λ is the
dimension that matters least where the model never moves.

The estimator prices each cell from measured timings **for its own arm at the same
horizon**, falling back to nominal rates otherwise. Neither shortcut would work: arms differ
by 14x, and run time is superlinear in the horizon, so a global average or a short pilot's
timings would both mislead. It also prints the slowest measured arms, which is usually the
number that decides how to slice the grid.

**Memory.** `-Xmx` bounds the Java heap, not the JVM's total footprint; allow roughly
`heap + 200–300 MB` per process. The driver's guard uses `jobs × heap` against 75% of system
RAM, which is deliberately conservative. Install `psutil` (`pip install psutil`) to have
peak RSS recorded per cell in the CSV — that, not wall-clock, is what sets the real `--jobs`
ceiling:

```bash
pip install psutil
```

Without it the `peak_rss_mb` column is simply left empty.

---

## 6. Output layout and CSV schema

```
experiments/
  run_ablation.py
  analyse.py
  manifest.json              per-cell status; drives --resume
  results.csv                one row per completed cell
  analysis/                  generated reports (analyse.py)
    coverage.txt
    per_scenario_<N>.txt
    mixture_line.txt
    lambda_response.txt
    paired_tests.txt
    summary.md
  cells/
    s3/                      scenario
      smile-MIP_lam5/        arm (baselines: fixed_mst, fixed_rt, fixed_random)
        seed222/
          solver.config      exactly what this cell ran
          run.log            stdout+stderr
          runSummary.txt     the aggregated row's source
          monitorables.txt   timestep active_links bandwidth ttw alpha thresholds
          MON{1,2,3}Sat.txt  timestep measuredValue satisfiedFlag
          MON{1,2,3}SatProb.txt
          SelectedAction.txt timestep action (0=MST, 1=RT)
          state_transitions.txt
          qvalues.txt        timestep Q_MST Q_RT |gap|   (empty for baselines)
          gamma.txt, surprise{CC,BF,MIP}.txt, MIPBounds.txt
          RDM.alpha
```

About 1–2 MB per cell, so 1–2 GB for the full grid. `charts.html` is **not** generated per
cell (`-Dcharts=false`): at ~5 MB each it would add several GB of plots nobody opens.
`--keep-logs` controls retention after a cell completes:

- `all` — keep everything (default)
- `summary` — delete per-step logs; keep `runSummary.txt`, `solver.config`, `run.log`
- `none` — as `summary`, and delete `run.log` too

`runSummary.txt` and `solver.config` are never deleted: the first is what the CSV and resume
logic read, the second is the record of what ran. Only cells marked `done` are pruned, so a
failed cell keeps everything needed to diagnose it.

### `results.csv`

| Column | Meaning |
|---|---|
| `scenario` | 0–6, read back from `runSummary.txt` |
| `learning` | `static`/`bayes`/`smile-CC`/`smile-BF`/`smile-MIP`; empty for baselines |
| `lambda` | entropy temperature; empty for baselines |
| `baseline` | `mst`/`rt`/`random`; empty for solver arms |
| `seed` | `runSeed` |
| `timesteps` | horizon actually run |
| `crn` | whether common random numbers were used |
| `MON1_rate`, `MON2_rate`, `MON3_rate` | per-NFR satisfaction rate over the whole run |
| `joint_rate` | fraction of timesteps at which all three NFRs were satisfied **at the same step**, from the per-step flags in `MON{1,2,3}Sat.txt`; empty if those logs were pruned |
| `MON1_belief`, `MON2_belief`, `MON3_belief` | mean belief marginal for that NFR |
| `rt_share` | fraction of timesteps the RT topology was chosen |
| `mean_links`, `mean_bw`, `mean_ttw` | mean monitorables (links, Gbps, ms) |
| `mean_abs_qgap` | mean \|Q_MST − Q_RT\|; the scale λ must be read against. Empty for baselines (no solve) |
| `mean_gamma` | mean SMiLe γ. `0` for `bayes` (γ is logged as 0 when surprise updating is off); empty for `static` (never computed) |
| `state_change_rate` | fraction of recorded transitions that changed POMDP state |
| `wallclock_s`, `peak_rss_mb` | cost of the run; RSS needs `psutil` |
| `status` | `done` for aggregated rows |

**Joint satisfaction is not the product of the per-NFR rates.** MON2 and MON3 are both
monotone increasing in active links while MON1 wants links high, so within a step the three
flags are negatively correlated: always-MST in S0 has a per-NFR product of 0.464 × 0.811 ×
0.929 ≈ 0.35 but a joint rate of only 0.254. `joint_rate` is the measured co-occurrence, and
it obeys the mixture line too (analytic values in the "All three" column of the ground-truth
table). It is reported in `analyse.py`'s tables, mixture check and paired tests; the figures
plot MON1–MON3 only. Older CSVs gain the column with `run_ablation.py --aggregate-only`.

### Figures

```bash
python createCharts.py --grid                          # everything -> figures/
python createCharts.py --grid --families lambda,bars   # a subset
python createCharts.py --grid --html-only              # no kaleido needed
```

Each figure is written as `.html` (interactive) and `.png` (3×) under `figures/<family>/`,
with `figures/index.html` as a gallery. [`figures/README.md`](../figures/README.md) explains
how to read each family and the order to go through them. PNG export uses `kaleido` (in `requirements.txt`),
which drives a local Chrome; if none is installed, run `plotly_get_chrome` once. The HTML
camera button downloads an SVG if a vector copy of one figure is needed. Every cross-arm
figure has seven scenario panels, and every per-NFR figure states the threshold and the
direction in its title, e.g. "MON1 met when active links ≥ 105 links (35% of 300 links):
maximise active links".

| Family | Files | What it shows |
|---|---|---|
| `lambda_response/` | `rt_share`, `MON1..3` | metric vs λ, one line per learning config |
| `bars/` | `MON1..3` | mean measured value (links / Gbps / ms) per config, coloured by λ, plus the fixed baselines, with the red threshold line |
| `mixture/` | `MON1..3` | every arm at (RT share, rate) against the analytic line: the validity figure |
| `paired/` | `vs_static_*`, `vs_bayes_*`, `pooled_tests.csv` | forest plots of CRN-paired differences at the same λ |
| `timeseries/lam<λ>/` | `rt_share`, `MON1..3`, `links`, `bw`, `ttw`, `gamma`, `surprise_*` | per-step series over the run, every timestep |
| `violins/s<N>/` | `lam<λ>` | raw monitorable distributions with the NFR threshold |

**Reading the error bars.**

- Lines, bars and points are **mean ± 1 SD across the 5 seeds**, with the SD always taken
  within one scenario.
- Forest plots instead show a **95% t-interval of the CRN-paired difference**. Pairing on
  (scenario, seed) removes the environment noise both arms shared, so these intervals are
  far tighter than overlapping SD bars would suggest.
- Per panel, n = 5, and a two-sided Wilcoxon cannot reach p < 0.05 at n = 5 (its minimum p
  is 0.0625). The rank test is therefore run on all 35 pairs pooled over scenarios, in
  `paired/pooled_tests.csv`.

**Smoothing and thresholds.**

- Time series are a trailing mean over the last 25 steps per seed (`--window`), then the
  band across seeds, on a linear time axis. The first 25 points average fewer than 25
  steps, so they are noisier than the rest. Each scenario is stationary in distribution, so
  the curves flatten after a short learning transient.
- Red lines are the NFR thresholds, derived from `config_log_files/configuration.json`
  exactly as RDMSim computes them. If a run's `runSummary.txt` recorded different values,
  the script warns: the config was edited after the grid ran.
- Dashed and dotted grey lines are the analytic always-MST and always-RT rates.

The per-step logs of all cells are cached in `experiments/analysis/timeseries_cache.pkl`
after the first read, and refreshed automatically when any cell is re-run.

---

## 7. Troubleshooting

**`OutOfMemoryError`, or the machine swaps.** Lower `--jobs` before raising `--heap`.
`-Xmx` caps the heap but each JVM also carries 200–300 MB outside it, and the solver's cost
is CPU-bound, not heap-bound, so more concurrency buys less than it costs. Check what the
guard thinks: `--dry-run` prints the memory plan.

**`Output directory could not be created`.** `outputDirectory` must be **relative**:
`configureDirectories()` resolves it against the working directory and rejects absolute
paths. If you pass `--output-root`, keep it inside the project — the driver fails fast with
this explanation if you do not.

**A cell is stuck in `running` and will not re-run.** That is a manifest entry left by a
killed driver. The next invocation resets it to `pending` automatically and says so. If you
want to force the issue, `--force` on that selection.

**Everything says `pending` but the directories have output.** The manifest is the source
of truth for resume, by design — output files alone cannot distinguish a complete run from
one killed mid-write. Use `--aggregate-only` to pull the existing `runSummary.txt` files
into the CSV without re-running, or `--force` to re-run them properly.

**`log.json` contention.** `config_log_files/log.json` is written by `RDMSim.jar` itself and
is shared across concurrent cells. The jar only writes it once, on a run's final timestep,
and that write is guarded by an `IOException` handler, so a collision can at worst print a
stack trace into `run.log` — it cannot fail a run. It is never read for results.

**JVMs still running after Ctrl+C or a timeout.** On Windows, `java` on PATH is usually
Oracle's `javapath\java.exe` launcher shim, which starts the real JDK `java.exe` as a child.
Killing the shim leaves the real JVM running with its heap. The driver avoids this by asking
`java` for its `java.home` and launching `<java.home>\bin\java.exe` directly (it prints the
path as `JVM: ...` when a batch starts), and by killing the whole process tree. If you pass
`--java` yourself, point it at a real JVM, not a shim. To check for strays:

```powershell
Get-CimInstance Win32_Process -Filter "Name='java.exe'" | Where-Object { $_.CommandLine -like '*RDMSim_TAAS*' }
```

**The laptop went to sleep mid-grid.** Two mechanisms handle this. While a batch runs, the
driver asks Windows not to *idle*-sleep (`SetThreadExecutionState`); this does not stop a
lid-close or an explicit Sleep, and it is a no-op on Linux/macOS (wrap the command in
`systemd-inhibit` or `caffeinate` there). If the machine sleeps anyway, the JVMs and the
driver are suspended together. On waking, the driver detects the gap between its own polls
and prints `(system was suspended for ~N min ...)`. That time is then not counted against
`--timeout`, `--max-wall` or the cells' recorded timings. Without that correction, any
sleep over ~45 min would get a healthy 16-minute `static` cell killed as `timeout`. For a
long unattended run, keep the machine plugged in with the lid open, or set the
plugged-in sleep timeout to *Never* for the duration.

**`warning: could not update experiments/manifest.json (file locked?)`.** The repo is in a
OneDrive-synced folder; the sync client can briefly lock files. The driver retries with
backoff and, if that fails, carries on — every save rewrites the whole manifest, so the next
one catches up. If it persists, pause OneDrive syncing for the duration of the grid.

**Linux: `POMDP file not found: .../RDM.POMDP`.** `domains/rdm.POMDP` is tracked with a
lowercase name while the code requests `RDM.POMDP`. This works on NTFS and APFS and breaks
on case-sensitive filesystems. Fix with `git mv domains/rdm.POMDP domains/RDM.POMDP`.

**A cell is marked `timeout`.** It exceeded `--timeout` (default 1 h) and was killed after
`--retries` attempts. Its `run.log` is kept. Either the horizon is longer than you think or
the solver is stuck; check how far `monitorables.txt` got.

**`warning: bin/RDMSim_TAAS.class not found`.** Build first; the driver warns rather than
refusing, because you may be using a different `--output-root` layout.

**Rates nowhere near the ground-truth table.** Check `mixture_line.txt` from
`analyse.py`. Large systematic residuals mean the measurement pipeline is wrong, not that
the algorithm is interesting. Confirm with `CRNVerify` and a `fixed_mst` run.

---

## 8. Verification checklist

Run these in order; each is cheap and each catches a different class of error.

1. **CRN correctness (blocking).**
   `java -cp "bin;libraries/*" remotemirroring.CRNVerify` — must print `PASS` for all 14
   (scenario, topology) cells. This is the acceptance test for having re-implemented the
   jar's generator.
2. **CRN determinism.** Repeated runs of the same cell with `--crn` must be byte-identical
   in `monitorables.txt`, `MON{1,2,3}Sat.txt`, `SelectedAction.txt` and
   `state_transitions.txt` — compare the state stream and the actions, not just the logged
   monitorables, and include S3 or S6, whose start topology is a coin flip. Use λ>0 so the
   policy itself is stochastic. With `--no-crn` they must differ.
3. **Static-model arm.** With `static`, `gamma.txt` and `surpriseMIP.txt` must be empty
   (no model updates) while `qvalues.txt` and `state_transitions.txt` are populated (the
   planner and belief filter still run).
4. **Fixed-policy arms.** `SelectedAction.txt` constant `0` for `fixed_mst` / `1` for
   `fixed_rt`, `qvalues.txt` empty, and the MON rates within sampling error of the
   ground-truth row for that scenario. This validates the pipeline end to end.
5. **Parallel safety.** Run a few cells concurrently; `configuration.json` must be
   unchanged, each `runSummary.txt` complete and on its own scenario, `log.json` still
   valid JSON.
6. **Driver CLI.** `--list` must give 980 cells for the full grid, 140 for `--scenarios 3`,
   and 7 for `--arms smile-MIP:5 --scenarios 0-6 --seeds 222 --baselines none`.
   `--jobs 64 --heap 4g` must be refused with a suggested `--jobs`. `--batch-size 3` must
   run three and leave the rest `pending`; a re-invocation must skip them; `--force` must
   re-run them.
7. **Pilot before the full grid.** Scenario 3, all 28 arms, 5 seeds. Record wall-clock and
   peak RSS, check between-seed variance, and size the grid from the measurement rather than
   an estimate.
8. **Surprise horizon (settle this during the pilot).** Plot `gamma.txt` and
   `surpriseMIP.txt` over the full horizon for `smile-MIP` at λ=0 and λ=10. A scenario's
   deviation band is fixed for the whole run, so there is no changepoint of the kind
   DeltaIoT's link-breaking supplied; the open question is whether γ therefore decays to a
   noise floor (learning effects confined to burn-in, and the retained per-step logs become
   necessary for windowed analysis) or keeps responding because the agent's own drifting
   RT share shifts the observation law it sees. The latter predicts a larger γ response at
   λ=10 than λ=0, since λ>0 mixes actions deliberately. Record which held — it determines
   how the learning ablations are interpreted.
9. **Full grid**, then `analyse.py`; every arm's MON rates should lie on the mixture line
   between its scenario's `fixed_mst` and `fixed_rt` rows.

---

## 9. Extending the grid

**A new learning config.** Add an entry to `LEARNING_CONFIGS` at the top of
`experiments/run_ablation.py`; the dict's keys are written verbatim into each cell's
`solver.config`. Add the same name to `LEARNING_ORDER` in `analyse.py` so it sorts
sensibly. If it needs a config key the harness does not yet read, add that key to
`readConfigFile()` in `src/RDMSim_TAAS.java` and record it in `writeRunSummary()` — the
driver parses the summary back to label CSV rows, so an unrecorded key is invisible in the
results.

**New λ values.** Just pass them: `--lambdas 0,1,3,7,15`. Nothing is hardcoded. Re-derive
the useful range from `qvalues.txt` if the reward table, discount or scenario changes — λ is
a temperature on `softmax(Q/λ)` and is meaningful only relative to the Q-gap, whose measured
mean on this domain is about 6.7.

**A new baseline policy.** Add it to `BASELINE_POLICIES` in the driver and to the
`fixedPolicy` switch in `runCaseRDM()`; the validation regex in `readConfigFile()` needs the
new name too.

**A new scenario.** Add `deviation_scenario_7_*` keys to `configuration.json`, extend the
`0..6` bounds check in `ScenarioOverride` and in RDMSim's own validation (the jar throws on
anything outside 0–6, so a new scenario needs the deviation bounds set through
`UncertaintyScenario`'s setters rather than the scenario index), and add its analytic row to
`TRUTH` in `analyse.py` and `EXPECTED` in `CRNVerify.java`.
