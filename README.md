# RDMSim_TAAS

Entropy-regularised POMDP planning (**ERPerseus**) with information-driven belief updating
(**MIP** / **SMiLe** surprise) applied to **RDMSim**, the remote data mirroring exemplar for
self-adaptive systems.

This is a port of the algorithms from *Information-Driven Adaptation and Entropy-Regularised
Planning under Non-Stationarity* (originally evaluated on DeltaIoT) onto RDMSim. The harness
drives a MAPE-K loop against `libraries/RDMSim.jar`, solving a POMDP at every timestep to choose
between the two available network topologies.

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
not a meaningful headline metric on this exemplar.

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

## Layout

```
src/RDMSim_TAAS.java              entry point; config parsing and the MAPE-K loop
src/pomdp/POMDP.java              model, belief filter, state/observation classification
src/pomdp/PomdpParser.java        .POMDP parser, Dirichlet prior, distribution validation
src/pomdp/SolverProperties.java   solver settings bean
src/solver/ERPerseus.java         entropy-regularised Perseus (the main solver)
src/solver/ERPolicy.java          action selection: softmax over per-action Q-values
src/solver/Perseus.java           standard Perseus
src/solver/ERPBVI.java            entropy-regularised PBVI (alternative)
src/remotemirroring/RDMSimConnector.java    simulator bridge; CC/BF/MIP surprise, SMiLe gamma
src/remotemirroring/RDMTransitionProb.java  per-NFR success probabilities per scenario
src/remotemirroring/OutputLogger.java       per-timestep file logging
src/remotemirroring/CRNStream.java          common random numbers for paired ablation arms
src/remotemirroring/CRNVerify.java          asserts CRNStream matches the jar's distributions
src/remotemirroring/ScenarioOverride.java   -Dscenario / -Dtimesteps runtime overrides
experiments/run_ablation.py       ablation grid driver (selection + resource bounds)
experiments/analyse.py            cross-cell tables and paired tests from results.csv
docs/ABLATION.md                  how to run the ablation studies
src/solver.config                 solver and experiment parameters
domains/RDM.POMDP                 POMDP model
config_log_files/configuration.json  RDMSim network, thresholds, uncertainty scenario
createCharts.py                   builds output_dir/charts.html from the run output
```

`libraries/` holds `RDMSim.jar` plus dependencies (log4j2, commons-math3, json-simple, antlr,
libpomdp-parser, jython).

---

## Running

Requires **JDK 21+** (developed against JDK 25) and Python 3 with `requirements.txt` installed
for chart generation. Run from the project root — config and domain lookups are relative to the
working directory.

```bash
# build
javac -d bin -cp "libraries/*" $(find src -name "*.java")

# run (Windows classpath separator; use ':' on Linux/macOS)
java -cp "bin;libraries/*" RDMSim_TAAS
```

In VS Code, use **Run and Debug -> Launch RDMSIM_TAAS**. JavaFX `vmArgs` are present in
`.vscode/launch.json` because `RDMSimulator.displayResults()` touches JavaFX, but the headless
MAPE-K run does not call it and works from the command line without JavaFX on the module path.

### Overrides

| System property | Effect |
|---|---|
| `-DconfigPath=<file>` | use a different `solver.config` |
| `-DoutputDirectory=<dir>` | write output elsewhere (must be **relative** — see Known issues) |
| `-Dscenario=0..6` | override `current_scenario` without editing `configuration.json` |
| `-Dtimesteps=N` | override `time_steps` likewise |
| `-Dcrn=true` | common random numbers; makes runs reproducible and ablation arms paired |
| `-Dcharts=false` | skip `charts.html` (~5 MB a run) |

Useful for sweeps: copy `src/solver.config`, edit `lambda` and `outputDirectory` in the copy, then
point the run at it with `-DconfigPath`. For anything larger than a handful of runs, use
`experiments/run_ablation.py` instead — see [Ablation studies](#ablation-studies).

---

## Configuration

### `src/solver.config` — solver and experiment parameters

| Key | Baseline | Meaning |
|---|---|---|
| `algorithmType` | `erperseus` | `perseus`, `erperseus`, `erpbvi`, `fasterpbvi` |
| `lambda` | `0.0` | entropy-regularisation temperature; **0.0 is the Perseus control arm** |
| `runSeed` | `222` | seeds solver, policy and deviation draws (vary: 222/223/224) |
| `surpriseMeasureForGamma` | `MIP` | `CC` (confidence-corrected), `BF` (Bayes factor), `MIP` |
| `p_c` | `0.5` | probability of change; `m = p_c/(1-p_c)` in the SMiLe gamma |
| `useSurpriseUpdating` | `true` | `false` = classic Bayesian (Dirichlet +1 only) |
| `lookback` | `4` | window for `MIP = MI[t] - MI[t-lookback]` |
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
toward uniform past 20. Suggested sweep `{0.0, 2, 5, 10, 20}`. Re-derive from `qvalues.txt` if the
reward table, discount or scenario changes — the gap moves with them.

### `config_log_files/configuration.json` — network and uncertainty

`time_steps`, `mirror_number`, the three NFR threshold percentages, the per-topology active-link
ranges (as percentages of `number_of_links`), and `current_scenario`.

**`current_scenario` selects the uncertainty scenario (0-6).** Scenario 0 applies *zero* deviation
— the environment is stationary, so there is nothing for MIP/SMiLe to detect; keep it as the
stationary control. The baseline uses **scenario 3**, which deviates MST links and RT
bandwidth/writing-time by 9-12%, perturbing each topology on a different NFR.

---

## Ablation studies

The ablation grid is 7 scenarios x 28 arms x 5 seeds = 980 runs, driven by
`experiments/run_ablation.py`. Every dimension is a filter, so any slice can be run on its
own and resumed later, and `--jobs`/`--heap` are checked against system RAM before anything
launches.

```bash
# check the CRN generator still matches the jar (gates everything else)
java -cp "bin;libraries/*" remotemirroring.CRNVerify

# see what the full grid would cost, without running it
python experiments/run_ablation.py --dry-run

# the scenario-3 pilot: all 28 arms, 5 seeds
python experiments/run_ablation.py --scenarios 3 --seeds 222-226 --jobs 4 --heap 1g

# the full grid, in resumable batches
python experiments/run_ablation.py --batch-size 100 --jobs 4 --heap 1g

# tables and figures from experiments/results.csv
python experiments/analyse.py
```

The 28 arms are `{static, bayes, smile-CC, smile-BF, smile-MIP}` x `lambda in {0,2,5,10,20}`
plus three fixed-policy baselines (`mst`, `rt`, `random`). Two new `solver.config` keys
select an arm: `staticModel` freezes the planning model at the seeded Dirichlet prior
(the bottom rung of the ladder), and `fixedPolicy` bypasses the solver entirely.

**See [docs/ABLATION.md](docs/ABLATION.md)** for the arm rationale, the full CLI reference,
recipes, the per-scenario ground-truth table used to validate runs, and troubleshooting.

### New JVM overrides

| System property | Effect |
|---|---|
| `-Dscenario=0..6` | override `current_scenario` without editing `configuration.json` |
| `-Dtimesteps=N` | override `time_steps` likewise |
| `-Dcrn=true` | common random numbers: pre-draw the environment for every timestep and both topologies from a `(scenario, runSeed)` stream, so ablation arms are perfectly paired and runs are reproducible. Default off |
| `-Dcharts=false` | skip `charts.html` (~5 MB a run) |

`-Dscenario` and `-Dtimesteps` exist so that concurrent cells never write to the shared
`config_log_files/configuration.json`, which is what makes parallel execution safe.

## Output

Written to `output_dir/` (git-ignored) and rebuilt on every run.

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

---

## Known issues

- **Runs are not bit-reproducible unless `-Dcrn=true`.** `runSeed` deterministically controls
  the solver, the policy and the deviation draw, but `RDMSim.jar`'s
  `Monitorables.generateMonitorables` uses `ThreadLocalRandom`, which cannot be seeded
  externally. `-Dcrn=true` sidesteps this by pre-drawing the environment from a seeded stream
  (`src/remotemirroring/CRNStream.java`, validated against the jar's own distributions by
  `remotemirroring.CRNVerify`); without it, average over repeats rather than expecting
  identical output.
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
  grid cost linearly from a short pilot.
- Stale artefacts at the repo root (`MAPE_KLoop.class`, `M*RegressionResultsSolvePOMDP.txt`) are
  left over from earlier versions and are superseded by the copies in `output_dir/`.

---

## References

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
