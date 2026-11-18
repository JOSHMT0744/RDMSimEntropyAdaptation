# RDMSim_TAAS

This project tests whether a self-adaptive system makes better decisions when it learns from
surprise. It controls RDMSim, a simulated network of 25 data mirrors. At every timestep it
chooses one of two network topologies by solving a POMDP, then updates its model of the
network from what it observed. An ablation harness runs every variant of the algorithm
across seven uncertainty scenarios and turns the results into paper-ready tables and figures.

In technical terms: entropy-regularised POMDP planning (**ERPerseus**) with
information-driven belief updating (**MIP** / **SMiLe** surprise), applied to **RDMSim**, the
remote data mirroring exemplar for self-adaptive systems. It ports the algorithms from
*Information-Driven Adaptation and Entropy-Regularised Planning under Non-Stationarity*
(originally evaluated on DeltaIoT) onto RDMSim. The harness drives a MAPE-K loop against
`libraries/RDMSim.jar`.

- [The problem](#the-problem)
- [Prerequisites](#prerequisites)
- [Workflow, step by step](#workflow-step-by-step)
- [Definition of done](#definition-of-done)
- [Configuration](#configuration)
- [Output](#output)
- [Layout](#layout)
- [Known issues](#known-issues)
- [References and licence](#references-and-licence)

Further documentation:

| Document | Covers |
|---|---|
| [docs/ABLATION.md](docs/ABLATION.md) | the ablation grid: arms, CLI reference, recipes, ground-truth table, sizing, troubleshooting |
| [docs/NFR-MAPPING.md](docs/NFR-MAPPING.md) | the sourced MON1/MON2/MON3 ↔ MR/MC/MP mapping and threshold definitions |
| [figures/README.md](figures/README.md) | what each paper figure shows and the order to read them in |

---

## The problem

RDMSim simulates a network of 25 data mirrors. The adaptation choice each timestep is one of two
topologies:

- **MST** (minimum spanning tree) — few active links, cheap
- **RT** (redundant topology) — many active links, expensive

Three non-functional requirements are monitored, with thresholds set as percentages of network
capacity in `config_log_files/configuration.json`:

| Monitorable | NFR | Measures | Satisfied when | Baseline |
|---|---|---|---|---|
| MON1 active links | **MR** — Maximisation of Reliability | reliability | `>=` threshold | 35% of 300 = **105 links** |
| MON2 bandwidth consumption | **MC** — Minimisation of Cost | operational cost | `<=` threshold | 40% of 300x30 = **3600** |
| MON3 time to write | **MP** — Maximisation of Performance | performance | `<=` threshold | 45% of 300x20 = **2700 ms** |

Note the numbering does **not** follow the MC/MR/MP letter order: MON1 is MR, MON2 is MC, MON3 is
MP. See **[docs/NFR-MAPPING.md](docs/NFR-MAPPING.md)** for the sourced mapping, the verbatim
definitions, and the two distinct kinds of baseline threshold in this literature (bounds on the
measured values versus bounds on satisfaction probability).

`number_of_links = mirror_number*(mirror_number-1)/2 = 300`. The conversion from percentage to
absolute value is done by RDMSim's `NetworkProperties.loadNetworkSettings()`; nothing in this
repo overrides it. The 35 / 40 / 45 percentages are the default scenario S0 thresholds from the
RDMSim exemplar paper, shipped unmodified.

**Satisfaction is reported per NFR, not jointly.** The simulator generates
`bandwidth = active_links x U{20,30}` and `time_to_write = active_links x U{10,20}`, so MON2 and
MON3 are both monotone *increasing* in the quantity MON1 wants large. The two topologies sit on
opposite sides of that trade-off and no policy can satisfy all three often; joint satisfaction is
not a meaningful headline metric on this exemplar. (The ablation still records it as
`joint_rate`, the fraction of steps with all three met at once, and tests it in
`figures/paired/pooled_tests.csv`, but no figure plots it.)

### POMDP model

- **8 states** — the combinations of (MC, MR, MP) satisfied/violated. Index 0 is all satisfied.
- **2 actions** — MST, RT.
- **27 observations** — 3x3x3 bucketing of the three monitorables.
- Model file: `domains/RDM.POMDP` (discount 0.9). It supplies the reward function and the
  *initial* transition/observation model.

At runtime the planning model is **not** the file's static tables: it is the normalised Dirichlet
posterior (`transitionBeliefCurr` / `observationBelief`), seeded from the file and updated every
timestep by the surprise-weighted (SMiLe) update in `RDMSimConnector`. `RDMTransitionProb` seeds
the prior only.

---

## Prerequisites

| Requirement | Needed for | Notes |
|---|---|---|
| **JDK 21+** | everything | developed against JDK 25. `java` and `javac` on `PATH`. |
| **Python 3** | charts, ablation driver, analysis | developed against 3.12. |
| **`.venv/` at the project root** | the per-run `charts.html` | `RDMSim_TAAS` only looks for `.venv/Scripts/python.exe` or `.venv/bin/python`. Without it the run still completes but skips its charts, with a warning. |
| `pip install -r requirements.txt` | charts and figures | pandas, numpy, plotly, kaleido, loguru, … |
| Chrome (for `kaleido`) | PNG copies of the grid figures | run `plotly_get_chrome` once if none is installed, or use `--html-only` |
| `psutil` (optional) | peak memory per ablation cell | `pip install psutil`. Without it the `peak_rss_mb` column is left empty. |

No environment variables are needed. Every runtime override is a JVM system property (see
[Overrides](#overrides)) or a command-line flag.

**Run every command from the project root.** Config, domain and output paths are resolved
relative to the working directory.

The classpath separator is `;` on Windows and `:` on Linux/macOS. The examples below use `;`.

---

## Workflow, step by step

The steps run in order. Steps 1–4 are enough for a single run. Steps 5–8 produce the
paper results.

### 1. Set up Python

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build

```bash
javac -d bin -cp "libraries/*" $(find src -name "*.java")
```

### 3. Verify the common-random-numbers generator

Do this check first. It confirms that the seeded re-implementation of RDMSim's environment
draws still matches the jar. Every paired comparison in the ablation depends on it.

```bash
java -cp "bin;libraries/*" remotemirroring.CRNVerify
# expect: PASS: CRNStream reproduces the analytic marginals in all 14 cells.
```

### 4. A single run

```bash
java -cp "bin;libraries/*" RDMSim_TAAS
```

This uses `src/solver.config` and `config_log_files/configuration.json` as they stand.
Currently that is ERPerseus at λ = 0, SMiLe-MIP updating, scenario 3 and 500 timesteps. The
results go to `output_dir/` (see [Output](#output)). Start with `output_dir/runSummary.txt`.

In VS Code, **Run and Debug → Launch RDMSIM_TAAS** does the same. `.vscode/` is gitignored,
so a fresh clone does not include that launch configuration. JavaFX `vmArgs` appear in it
because `RDMSimulator.displayResults()` touches JavaFX, but the headless MAPE-K run does not
call it and works from the command line without JavaFX on the module path.

To vary a parameter for one run, copy `src/solver.config`, edit the copy (for example
`lambda` and `outputDirectory`), and pass it with `-DconfigPath=<copy>`. For more than a
handful of runs, use the ablation driver.

### 5. Smoke-test and size the ablation grid

```bash
# one cell, 50 steps: checks the harness end to end in under a minute
python experiments/run_ablation.py --scenarios 3 --arms smile-MIP:0 --baselines none --seeds 222 --timesteps 50

# what the full grid would cost (time and memory), without running anything
python experiments/run_ablation.py --dry-run
```

### 6. Run the grid

The grid is 7 scenarios × 28 arms × 5 seeds = **980 cells** at 1000 timesteps each. On an
8-core machine that is about 17 h at 4 jobs. Run a pilot first, then the rest in resumable
batches:

```bash
# pilot: scenario 3, all 28 arms, 5 seeds (140 cells)
python experiments/run_ablation.py --scenarios 3 --seeds 222-226 --jobs 4 --heap 1g

# full grid; repeat until --list reports nothing pending
python experiments/run_ablation.py --batch-size 100 --jobs 4 --heap 1g
python experiments/run_ablation.py --list
```

Completed cells are skipped on re-invocation, and Ctrl+C is safe. Each cell's status is kept
in `experiments/manifest.json`. Each completed cell adds one row to `experiments/results.csv`.

The 28 arms are `{static, bayes, smile-CC, smile-BF, smile-MIP}` × λ ∈ `{0, 2, 5, 10, 20}`,
plus three fixed-policy baselines (`mst`, `rt`, `random`). Two `solver.config` keys select an
arm. `staticModel` freezes the planning model at the seeded Dirichlet prior, which is the
bottom rung of the ladder. `fixedPolicy` bypasses the solver entirely. The driver writes both
per cell.

**See [docs/ABLATION.md](docs/ABLATION.md)** for why each arm exists, the full CLI reference,
recipes, the per-scenario ground-truth table used to validate runs, and troubleshooting.

### 7. Analyse

```bash
python experiments/analyse.py
```

This reads `experiments/results.csv` only and writes text tables to `experiments/analysis/`:
coverage, per-scenario tables, the mixture-line correctness check, λ-response tables, paired
tests, and `summary.md`, which combines them. It works on a partial grid.

### 8. Build the paper figures

```bash
python createCharts.py --grid                  # all figure families -> figures/
python createCharts.py --grid --html-only      # skip PNGs if kaleido/Chrome is unavailable
```

This reads `results.csv` and the per-cell logs, and writes 108 figures to `figures/` as
interactive HTML plus 3× PNG, with a gallery at `figures/index.html`. Open the gallery, then
follow the reading order in **[figures/README.md](figures/README.md)**. The first step there
is the `mixture/` validity check.

Other grid flags: `--families lambda,bars,mixture,paired,timeseries,violins` builds only some
families, `--lambdas` restricts the time-series and violin figures, `--window N` sets the
time-series smoothing (default 25), and `--no-cache` re-reads per-step logs instead of using
`experiments/analysis/timeseries_cache.pkl`.

---

## Definition of done

| Stage | Done when |
|---|---|
| Build | `bin/RDMSim_TAAS.class` exists and `javac` printed no errors |
| CRN check | `CRNVerify` prints `PASS` for all 14 cells |
| Single run | `output_dir/runSummary.txt` lists per-NFR satisfaction rates and the parameters used. `output_dir/charts.html` exists if `.venv` is set up. |
| Grid | `run_ablation.py --list` shows 980 cells `done` and none `pending`, `failed` or `timeout`. `results.csv` has 980 data rows. |
| Analysis | `experiments/analysis/summary.md` exists, and in `mixture_line.txt` every arm sits on its scenario's MST–RT line within sampling error |
| Figures | `figures/index.html` shows every family. In `figures/mixture/`, every point lies on the analytic line. |

If the mixture-line check fails, the measurement pipeline is wrong, and nothing downstream
should be interpreted. See the troubleshooting section of [docs/ABLATION.md](docs/ABLATION.md).

---

## Configuration

### `src/solver.config` — solver and experiment parameters

| Key | Baseline | Meaning |
|---|---|---|
| `algorithmType` | `erperseus` | `perseus`, `erperseus`, `erpbvi`, `fasterpbvi` |
| `lambda` | `0.0` | entropy-regularisation temperature; **0.0 is the Perseus control arm** |
| `runSeed` | `222` | seeds solver, policy and deviation draws. The ablation uses 222–226. |
| `surpriseMeasureForGamma` | `MIP` | `CC` (confidence-corrected), `BF` (Bayes factor), `MIP` |
| `p_c` | `0.5` | probability of change; `m = p_c/(1-p_c)` in the SMiLe gamma |
| `useSurpriseUpdating` | `true` | `false` = classic Bayesian (Dirichlet +1 only) |
| `lookback` | `4` | window for `MIP = MI[t] - MI[t-lookback]` |
| `staticModel` | `false` | `true` freezes the planning model at the prior (the belief filter still runs) |
| `fixedPolicy` | *(empty)* | `mst`, `rt` or `random` bypasses the solver |
| `priorStrength` | `1.0` | strength of the Dirichlet prior seeded from `RDM.POMDP` |
| `beliefSamplingRuns/Steps` | `10` / `200` | belief-point sampling per solve |
| `timeLimit` | `1000` | seconds per solve |

**Choosing `lambda`.** It is a temperature on `softmax(Q_a / lambda)` and is *not* scale-free — it
only matters relative to the gap between the per-action Q-values, which each run logs to
`output_dir/qvalues.txt`. Measured mean `|Q_MST - Q_RT|` on this domain is **6.67**, so with
`P(better action) = 1/(1+exp(-gap/lambda))` the operative range sits an order of magnitude above
1. Measured over 80 timesteps under scenario 3:

| lambda | P(better) | RT share | MON1 | MON2 | MON3 |
|---|---|---|---|---|---|
| 0.0 | 1.000 | 10% | 46.3% | 85.0% | 92.5% |
| 1.0 | 0.999 | 1% | 35.0% | 90.0% | 95.0% |
| 2.0 | 0.966 | 19% | 52.5% | 72.5% | 82.5% |
| 5.0 | 0.791 | 29% | 61.3% | 61.3% | 77.5% |
| 10.0 | 0.661 | 48% | 63.8% | 45.0% | 61.3% |
| 20.0 | 0.583 | 56% | 67.5% | 40.0% | 55.0% |

`lambda` in (0,1] is effectively still hard-max here; useful mixing starts near 2 and washes
toward uniform past 20. Suggested sweep `{0.0, 2, 5, 10, 20}`, which is the sweep the
ablation uses. Re-derive from `qvalues.txt` if the reward table, discount or scenario changes —
the gap moves with them.

### `config_log_files/configuration.json` — network and uncertainty

`time_steps` (currently 500), `mirror_number`, the three NFR threshold percentages, the
per-topology active-link ranges (as percentages of `number_of_links`), the six deviation bands
per scenario, and `current_scenario` (currently 3).

**`current_scenario` selects the uncertainty scenario (0–6).** A scenario sets `[min,max]`
deviation bands on links, bandwidth and writing time for each topology. Scenario 0 applies
*zero* deviation and serves as the unperturbed control. The baseline uses **scenario 3**,
which deviates MST links and RT bandwidth/writing-time by 9–12%, perturbing each topology on
a different NFR. Every scenario's bands stay fixed for the whole run, so no scenario has a
mid-run changepoint. They differ in severity, not in which action is optimal (see the
ground-truth table in [docs/ABLATION.md](docs/ABLATION.md)).

### Overrides

JVM system properties, given on the `java` command line:

| System property | Effect |
|---|---|
| `-DconfigPath=<file>` | use a different `solver.config` |
| `-DoutputDirectory=<dir>` | write output elsewhere (must be **relative** — see Known issues) |
| `-Dscenario=0..6` | override `current_scenario` without editing `configuration.json` |
| `-Dtimesteps=N` | override `time_steps` likewise |
| `-Dcrn=true` | common random numbers: pre-draw the environment for every timestep and both topologies from a `(scenario, runSeed)` stream, so ablation arms are perfectly paired and runs are reproducible. Default off |
| `-Dcharts=false` | skip `charts.html` (~5 MB a run) |

`-Dscenario` and `-Dtimesteps` exist so that concurrent cells never write to the shared
`config_log_files/configuration.json`, which is what makes parallel execution safe. The
ablation driver sets all of these per cell.

---

## Output

### Single run: `output_dir/`

Git-ignored and rebuilt on every run.

| File | Contents |
|---|---|
| `runSummary.txt` | **start here** — per-NFR satisfaction rates, mean monitorables, action split, state-change rate, and the parameters used |
| `monitorables.txt` | `timestep active_links bandwidth time_to_write alpha thresholds...` |
| `MON{1,2,3}Sat.txt` | `timestep measuredValue satisfiedFlag` |
| `MON{1,2,3}SatProb.txt` | belief marginal for that NFR |
| `SelectedAction.txt` | `timestep action` (0 = MST, 1 = RT) |
| `state_transitions.txt` | `timestep prevState action nextState b[0..7]` |
| `qvalues.txt` | `timestep Q_MST Q_RT abs_gap` — the scale to calibrate `lambda` against |
| `surprise{CC,BF,MIP}.txt`, `gamma.txt`, `MIPBounds.txt` | surprise and SMiLe traces |
| `RDM.alpha` | value function, appended per solve |
| `charts.html` | plotly report (satisfaction, surprise, gamma, MIP, action trace) |

`config_log_files/log.json` is written by `RDMSim.jar` itself. It is tracked in git but listed in
`.gitignore`, so it shows as modified after every run.

### Ablation grid

| Path | Contents | In git |
|---|---|---|
| `experiments/cells/s<N>/<arm>/seed<S>/` | one single-run output directory per cell, plus its `solver.config` and `run.log` | no |
| `experiments/manifest.json` | per-cell status, timings and command line; drives resume | no |
| `experiments/results.csv` | one row per completed cell (schema in [docs/ABLATION.md §6](docs/ABLATION.md#6-output-layout-and-csv-schema)) | no |
| `experiments/analysis/` | text tables and `summary.md` from `analyse.py` | no |
| `figures/` | paper figures from `createCharts.py --grid` ([guide](figures/README.md)) | not yet |

---

## Layout

```
src/RDMSim_TAAS.java              entry point; config parsing and the MAPE-K loop
src/solver.config                 solver and experiment parameters
src/pomdp/POMDP.java              model, belief filter, state/observation classification
src/pomdp/PomdpParser.java        .POMDP parser, Dirichlet prior, distribution validation
src/pomdp/SolverProperties.java   solver settings bean
src/solver/ERPerseus.java         entropy-regularised Perseus (the main solver)
src/solver/ERPolicy.java          action selection: softmax over per-action Q-values
src/solver/Perseus.java           standard Perseus
src/solver/ERPBVI.java            entropy-regularised PBVI (alternative)
src/solver/fastERPBVI.java        faster ERPBVI variant (algorithmType=fasterpbvi)
src/solver/…                      shared solver types (AlphaVector, BeliefPoint, Solver, …)
src/remotemirroring/RDMSimConnector.java    simulator bridge; CC/BF/MIP surprise, SMiLe gamma
src/remotemirroring/RDMTransitionProb.java  per-NFR success probabilities per scenario
src/remotemirroring/OutputLogger.java       per-timestep file logging
src/remotemirroring/ResultsLog.java         static holder for the current step's monitorables and flags
src/remotemirroring/CRNStream.java          common random numbers for paired ablation arms
src/remotemirroring/CRNVerify.java          asserts CRNStream matches the jar's distributions
src/remotemirroring/ScenarioOverride.java   -Dscenario / -Dtimesteps runtime overrides
domains/RDM.POMDP                 POMDP model
config_log_files/configuration.json  RDMSim network, thresholds, uncertainty scenarios
experiments/run_ablation.py       ablation grid driver (selection, resume, resource bounds)
experiments/analyse.py            cross-cell tables and paired tests from results.csv
createCharts.py                   per-run output_dir/charts.html; --grid builds the paper figures
charts/                           charting package behind createCharts.py
figures/                          generated paper figures; see figures/README.md
docs/ABLATION.md                  how to run and validate the ablation studies
docs/NFR-MAPPING.md               sourced MON ↔ NFR mapping and thresholds
requirements.txt                  Python dependencies
```

`libraries/` holds `RDMSim.jar` plus dependencies (log4j2, commons-math3, json-simple, antlr,
libpomdp-parser, jython).

---

## Known issues

- **Runs are not bit-reproducible unless `-Dcrn=true`.** `runSeed` deterministically controls
  the solver, the policy and the deviation draw, but `RDMSim.jar`'s
  `Monitorables.generateMonitorables` uses `ThreadLocalRandom`, which cannot be seeded
  externally. `-Dcrn=true` sidesteps this by pre-drawing the environment from a seeded stream
  (`src/remotemirroring/CRNStream.java`, validated against the jar's own distributions by
  `remotemirroring.CRNVerify`); without it, average over repeats rather than expecting
  identical output. The ablation driver turns CRN on by default.
- **`outputDirectory` must be relative.** `configureDirectories()` resolves it against the working
  directory and rejects an absolute path with "Output directory could not be created".
- **`domains/rdm.POMDP` is tracked with a lowercase name** while the code requests `RDM.POMDP`.
  This works on case-insensitive filesystems (NTFS, APFS) but breaks on Linux; needs a `git mv`.
- **The reward table in `domains/RDM.POMDP` is not monotone in NFR satisfaction.** For MST,
  `R(S5)=43` and `R(S2)=40` both exceed `R(S1)=39` where S1 is all-three-satisfied, and
  `R(S3,MST)=38 > R(S3,RT)=28` rewards staying on the sparse topology precisely when reliability
  is the violated NFR. These 16 values are inherited from prior work and were deliberately left
  unchanged; they bias the policy toward MST.
- **`MIPBounds.txt` is not a usable test.** The bounds depend only on `n`, `m` and `rho`, never on
  the data.
- **RT satisfies MON1 with probability 1.000 in every scenario** (150 links less 12% is still
  above the 105 threshold), so MON1 carries no decision information under the published
  thresholds. The optimal action also never flips across S0-S6: MST wins MON2, MON3 and joint
  satisfaction everywhere. See the ground-truth table in [docs/ABLATION.md](docs/ABLATION.md).
- **Solver time is superlinear in the horizon**, because the POMDP is re-solved from scratch
  every timestep and the alpha-vector set grows as the learned model moves. Do not extrapolate
  grid cost linearly from a short pilot. The `static` arm costs about 14 times as much as a
  learning arm, because its value function never collapses.
- **Per-run charts need a project-local `.venv`.** `RDMSim_TAAS` does not use the `python` on
  `PATH`; see [Prerequisites](#prerequisites).
- Stale artefacts at the repo root (`MAPE_KLoop.class`, `M*RegressionResultsSolvePOMDP.txt`) are
  left over from earlier versions and are superseded by the copies in `output_dir/`.

---

## References and licence

- Samin, Garcia Paucar, Bencomo, Carranza Hurtado, Fredericks. *RDMSim: An Exemplar for Evaluation
  and Comparison of Decision-Making Techniques for Self-Adaptation.* SEAMS 2021.
  [IEEE](https://ieeexplore.ieee.org/document/9462042/) /
  [arXiv:2105.01978](https://arxiv.org/abs/2105.01978) — defines the three monitorables, the
  MC/MR/MP objectives and the default scenario S0 thresholds this repo uses.
- Garcia Paucar. *Requirements-aware Models to Support Better Informed Decision-making for
  Self-adaptation using Partially Observable Markov Decision Processes.*
  PhD thesis, Aston University, December 2019.
  [Aston Publications](https://publications.aston.ac.uk/id/eprint/41929/) — the POMDP formulation
  of NFR satisficement and the SLA thresholds on satisfaction probability.

See [docs/NFR-MAPPING.md](docs/NFR-MAPPING.md) for how both map onto this implementation.

The solver code builds on **SolvePOMDP** (Erwin Walraven, TU Delft), released under the GNU
GPL v3. [LICENSE](LICENSE) carries the full text and the licences of the bundled
libraries.
