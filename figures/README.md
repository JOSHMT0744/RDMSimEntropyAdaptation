# Figures

Every figure produced from the RDMSim ablation grid (`experiments/results.csv` plus the
per-step logs under `experiments/cells/`). This page explains what each family of figures
is for, how to read it, and in what order to look at them.

For how the grid itself is run, see [`docs/ABLATION.md`](../docs/ABLATION.md).

- [Quick start](#quick-start)
- [Vocabulary](#vocabulary)
- [Conventions shared by every figure](#conventions-shared-by-every-figure)
- [Suggested reading order](#suggested-reading-order)
- [Figure families](#figure-families)
- [Which figure answers my question?](#which-figure-answers-my-question)
- [Regenerating](#regenerating)

---

## Quick start

1. Open **[`index.html`](index.html)** in a browser. It is a thumbnail gallery of every
   figure, grouped by family. Click a thumbnail to open the interactive version.
2. Each figure exists twice: `.html` (interactive: hover for exact values, click legend
   entries to hide or show series, camera button downloads an SVG) and `.png` (3× resolution,
   for the paper or slides).
3. Every HTML page loads the shared `plotly.min.js` from this folder. Keep the directory
   layout intact. A single HTML file copied somewhere else will open blank.

```
figures/
  index.html              gallery: start here
  mixture/                1. is the pipeline correct?
  lambda_response/        2–3. what does each config do, and how well, as λ changes?
  paired/                 4. which differences are real?
  bars/                   5. how big are the effects in physical units?
  timeseries/lam<λ>/      6. how does behaviour evolve over a run?
  violins/s<N>/           7. what does the full distribution look like?
```

---

## Vocabulary

| Term | Meaning |
|---|---|
| **Scenario** `S0`–`S6` | One of RDMSim's seven uncertainty profiles. Each panel title gives a short label derived from `configuration.json`, e.g. `MST links; RT bw+ttw` (which monitorables deviate under which topology). |
| **Learning config** | `static` (planning model frozen at the prior), `bayes` (plain Dirichlet +1 updates), `smile-CC`, `smile-BF`, `smile-MIP` (surprise-weighted updates, differing in the surprise signal). |
| **λ** | Entropy-regularisation temperature: 0, 2, 5, 10, 20. λ = 0 is plain Perseus. Larger λ makes the policy more random. |
| **Arm** | One learning config at one λ (25 arms), plus three **fixed baselines** that never learn: `fixed MST`, `fixed RT`, `fixed random`. |
| **Seed** | 5 seeds (222–226) per scenario × arm. Seeds use common random numbers (CRN), so every arm at a given (scenario, seed) faces exactly the same environment. |
| **MST / RT** | The two actions (network topologies). **RT share** is the fraction of timesteps on which an arm chose RT. |
| **NFRs** | The three requirements, each a threshold on a monitorable, listed below. |

| NFR | Monitorable | Met when | Goal |
|---|---|---|---|
| **MON1** (MR) | active links | ≥ 105 links | maximise |
| **MON2** (MC) | bandwidth consumption | ≤ 3600 Gbps | minimise |
| **MON3** (MP) | time to write | ≤ 2700 ms | minimise |

The thresholds above are the values at the time of writing. Every per-NFR figure prints the
thresholds actually in force in its subtitle, so trust the figure over this table.

**The trade-off.** RT always satisfies MON1, and MST is better for MON2 and MON3, in every
scenario. RDMSim is memoryless given the action, so every satisfaction rate is a straight
linear function of RT share. **RT share explains most of what you see.** If an arm's MON2
rate moves, check its RT share first.

---

## Conventions shared by every figure

**Layout.** Cross-arm figures use a 2 × 4 grid: S0–S3 on the top row, S4–S6 on the bottom
row, and the legend in the eighth cell. Panels share a y-axis, so heights can be compared
across scenarios.

**Colour and symbol.**
- Colour means learning config, and is the same in every figure.
- Marker symbol means λ: ● 0, ■ 2, ◆ 5, ▲ 10, ★ 20.
- Greys are the fixed baselines (dark = MST, mid = RT, light = random).

**Error bars and bands.**

| Where | What the bar or band is |
|---|---|
| Lines, bars, points (most figures) | mean ± 1 SD **across the 5 seeds**, within one scenario |
| Forest plots (`paired/`) | **95 % t-interval of the CRN-paired difference**, n = 5 |
| Violins | the whole pooled distribution (5 seeds × 1000 steps); the black marker is the mean ± 1 SD of the 5 seed means |

Paired intervals are much tighter than the SD bars elsewhere, because pairing removes the
environment noise that both arms shared. Two SD bars that overlap can still be a clearly
significant paired difference. Use `paired/` to judge significance, not overlapping bars.

**Reference lines.**
- **Red line:** the NFR threshold (raw-value figures).
- **Dashed grey:** the analytic always-MST satisfaction rate.
- **Dotted grey:** the analytic always-RT satisfaction rate.

Every learning arm's satisfaction rate should fall between the dashed and dotted lines.

---

## Suggested reading order

Each step depends on the one before. Do not interpret later steps if an earlier one fails.

| Step | Look at | Question | Move on when… |
|---|---|---|---|
| **1. Validate** | `mixture/MON1..3` | Is the measurement pipeline correct? | every point sits on the analytic line |
| **2. Behaviour** | `lambda_response/rt_share` | What does each config choose, and how does λ change it? | you know which configs lean MST or RT, and where |
| **3. Outcomes** | `lambda_response/MON1..3` | How well does each config satisfy each NFR across λ? | you have candidate "best" arms |
| **4. Significance** | `paired/vs_static_*`, `paired/vs_bayes_*`, `pooled_tests.csv` | Are those differences real? | each claim has a CI that excludes 0 |
| **5. Magnitude** | `bars/MON1..3` | How far from the threshold, in links / Gbps / ms? | you can state effects in physical units |
| **6. Dynamics** | `timeseries/lam<λ>/*` | Is the effect a start-up transient or does it persist? How do γ and surprise behave? | you know whether results are driven by burn-in |
| **7. Distributions** | `violins/s<N>/lam<λ>` | What do individual timesteps look like, including tails? | you have the per-scenario detail |

Steps 1–4 are the core argument of the paper. Steps 5–7 support and explain it.

---

## Figure families

### `mixture/`: validity check

**Files:** `MON1`, `MON2`, `MON3`

**Shows.** Each arm (all 25 learning arms and the 3 baselines) plotted at
(RT share, satisfaction rate), with ± 1 SD on both axes. The grey line runs from the analytic
always-MST rate (x = 0) to the analytic always-RT rate (x = 1).

**Why it matters.** Because the environment is memoryless given the action, **every** policy
must lie on this line. Points on the line mean the simulator, the CRN stream and the
satisfaction tally are working. A point clearly off the line means a pipeline bug, not an
interesting algorithm. In that case stop and see the troubleshooting section of
`docs/ABLATION.md`.

**How to read.** Position along the line is all that varies between arms. This figure also
shows each arm's place on the MST ↔ RT spectrum in a single view. The fixed MST and RT
hexagons should sit on the two ends of the line.

---

### `lambda_response/`: the main result

**Files:** `rt_share`, `MON1`, `MON2`, `MON3`

**Shows.** The metric (y) against λ (x, categorical), one line per learning config, with a
± 1 SD band across seeds. The per-NFR figures add the dashed and dotted analytic MST and RT
references.

**Why it matters.** This is the central ablation view. It shows, in one figure each:
- **learning vs no learning:** `static` against everything else,
- **surprise weighting vs plain Bayes:** `smile-*` against `bayes`,
- **which surprise signal:** CC vs BF vs MIP,
- **the effect of entropy regularisation:** the shape of each line from λ = 0 to λ = 20.

**How to read.**
- Read **`rt_share` first**. The MON2 and MON3 lines are close to mirror images of it, and
  MON1 tracks it directly.
- Higher is always better on the MON panels, since these are satisfaction rates. A line near
  the dashed grey line behaves like always-MST, and a line near the dotted line behaves like
  always-RT.
- As λ grows, the policy is pushed towards 50/50, so lines tend to converge at large λ.

---

### `paired/`: statistical comparisons

**Files:** `vs_static_{rt_share,MON1,MON2,MON3}`, `vs_bayes_{…}`, `pooled_tests.csv`

**Shows.** Forest plots. Each row is one arm, and its point is the mean of the paired
difference **arm − reference** at the same λ, on the same scenario and seed. The whisker is
a 95 % t-interval over the 5 seeds, and the vertical line marks zero.

| File prefix | Reference | Compared arms | Question |
|---|---|---|---|
| `vs_static_` | `static` | bayes, smile-CC, smile-BF, smile-MIP | Does learning help at all? |
| `vs_bayes_` | `bayes` | smile-CC, smile-BF, smile-MIP | Does surprise weighting beat plain Bayes? |

**How to read.**
- A whisker that does not cross zero means a credible difference in that scenario.
- On MON panels, **right of zero means the arm satisfies that NFR more often than the
  reference**. On `rt_share`, right of zero means the arm chooses RT more often, which is
  neither good nor bad in itself.
- The x-axis is symmetric and shared across panels, so effect sizes can be compared across
  scenarios.

**`pooled_tests.csv`** holds the formal tests. With only 5 pairs per panel, a two-sided
Wilcoxon test cannot reach p < 0.05 (its minimum is 0.0625). So the rank test is run on all
35 pairs pooled over the 7 scenarios. Columns:

| Column | Meaning |
|---|---|
| `comparison`, `arm`, `reference` | what is compared with what, at the same λ |
| `metric` | `rt_share`, `MON1..3_rate`, and **`joint_rate`** (all three NFRs met at the same timestep). `joint_rate` is only reported here and is not plotted. |
| `pairs` | number of (scenario, seed) pairs, normally 35 |
| `mean_diff`, `ci95_low`, `ci95_high` | pooled mean paired difference and its 95 % t-interval |
| `wilcoxon_W`, `verdict` | signed-rank statistic and the decision at α = 0.05. The `n` in the verdict can be below 35 because zero differences are dropped. |

Use the CSV for the claims you write down, and the forest plots to see whether an effect is
consistent across scenarios or driven by one of them.

---

### `bars/`: effects in physical units

**Files:** `MON1`, `MON2`, `MON3`

**Shows.** For each learning config, five bars (one per λ, coloured by λ) of the **mean
measured monitorable** (links, Gbps or ms), ± 1 SD across seeds. The three grey baseline bars
follow at the right, and the red line is the NFR threshold.

**Why it matters.** Rates say *how often* an NFR is met. The bars say *by how much*, in units
that operators and reviewers recognise.

**Watch out.** A bar is a **mean**. A run whose mean sits on the good side of the red line
still violates the NFR on some timesteps. For how often an NFR is met, use
`lambda_response/` or `mixture/`, not this figure.

---

### `timeseries/lam<λ>/`: behaviour over a run

**Folders:** `lam0`, `lam2`, `lam5`, `lam10`, `lam20`, one per λ. Each holds 11 figures:

| File | Plots | Reference lines |
|---|---|---|
| `rt_share` | rolling share of steps choosing RT | none |
| `MON1`, `MON2`, `MON3` | rolling satisfaction rate | dashed MST / dotted RT analytic |
| `links`, `bw`, `ttw` | raw monitorable value | red threshold |
| `gamma` | SMiLe learning rate γ | none |
| `surprise_BF`, `surprise_CC`, `surprise_MIP` | per-step surprise signal | none |

**Shows.** One line per learning config at that λ, over all timesteps. Each seed's series is
smoothed with a **trailing 25-step mean**, then shown as mean ± 1 SD across seeds.

**Why it matters.**
- It separates a **learning transient** from steady-state behaviour. Each scenario is
  stationary, so curves should flatten after the early steps. A config that only differs
  during the first ~100 steps has a burn-in effect, not a lasting one.
- The `gamma` and `surprise_*` figures show the learning machinery itself: whether γ
  decays to a noise floor or keeps responding, and whether this differs between λ = 0 and
  larger λ.

**Watch out.**
- The first 25 points average fewer than 25 steps, so they are noisier.
- Signal figures only show configs that compute that signal. `static` has no γ or surprise,
  and `bayes` logs γ as a constant 0. Missing lines are expected.
- To compare λ values, open the same file name in two `lam*` folders side by side.

---

### `violins/s<N>/`: full distributions

**Folders:** `s0`–`s6`, one per scenario. Each holds `lam0`, `lam2`, `lam5`, `lam10` and
`lam20`.

**Shows.** Three panels (active links, bandwidth, time to write) with eight violins each:
the five learning configs at that λ, then the three fixed baselines. Each violin pools
**all 5 seeds × all timesteps** of raw measured values. The black marker is the mean of the
five seed means, ± 1 SD. The red line is the threshold.

**Why it matters.** This is the only view of individual timesteps rather than averages. It
shows spread, skew, bimodality (a policy that switches between MST and RT produces two
humps) and how much mass falls on the wrong side of the threshold.

**Watch out.** The violin width shows the spread of **values**, not of seeds. The seed
spread is the black whisker. The baselines do not depend on λ, so their violins are the same
in every `lam*` file of a scenario. Use them as a fixed reference.

---

## Which figure answers my question?

| Question | Go to |
|---|---|
| Can I trust these results at all? | `mixture/` |
| Does learning beat a frozen model? | `lambda_response/MON*`, then `paired/vs_static_*` |
| Does surprise weighting beat plain Bayes? | `paired/vs_bayes_*`, `pooled_tests.csv` |
| Which surprise signal (CC / BF / MIP) is best? | `lambda_response/MON*` (compare the three smile lines), `paired/vs_bayes_*` |
| What does entropy regularisation (λ) do? | `lambda_response/rt_share`, then `lambda_response/MON*` |
| Is a difference statistically significant? | `paired/pooled_tests.csv` (pooled), forest plots (per scenario) |
| Do the arms satisfy all three NFRs at once? | `paired/pooled_tests.csv`, rows with `metric = joint_rate` |
| How close to the threshold, in real units? | `bars/` |
| Is the effect only a start-up transient? | `timeseries/lam<λ>/MON*` and `rt_share` |
| How does γ or surprise evolve? | `timeseries/lam<λ>/gamma`, `surprise_*` |
| Is one scenario behaving differently? | any 2 × 4 grid figure (compare panels), then `violins/s<N>/` |
| How bad are the worst timesteps? | `violins/s<N>/lam<λ>` |

---

## Regenerating

Run from the project root:

```bash
python createCharts.py --grid                          # all families -> figures/
python createCharts.py --grid --families lambda,bars   # only some families
python createCharts.py --grid --html-only              # skip PNGs (no kaleido/Chrome needed)
```

Figures are overwritten in place and `index.html` is rebuilt. This README is not touched.
If the thresholds in `configuration.json` differ from those recorded in a run's
`runSummary.txt`, the script warns. That means the config was edited after the grid ran, and
the red lines will not match the data.
