# NFR mapping and baseline thresholds

Which monitorable (`MON1`, `MON2`, `MON3`) corresponds to which non-functional requirement
(`MC`, `MR`, `MP`), what direction each is optimised in, and where the baseline thresholds come
from.

**Sources**

- **[S]** Samin, Garcia Paucar, Bencomo, Carranza Hurtado, Fredericks. *RDMSim: An Exemplar for
  Evaluation and Comparison of Decision-Making Techniques for Self-Adaptation.* SEAMS 2021.
  [IEEE Xplore 9462042](https://ieeexplore.ieee.org/document/9462042/) —
  open access: [arXiv:2105.01978](https://arxiv.org/abs/2105.01978)
- **[T]** Garcia Paucar. *Requirements-aware Models to Support Better Informed Decision-making
  for Self-adaptation using Partially Observable Markov Decision Processes.*
  PhD thesis, Aston University, December 2019.
  [publications.aston.ac.uk/id/eprint/41929](https://publications.aston.ac.uk/id/eprint/41929/)

---

## The mapping

| Monitorable | NFR | Full name | Measures | Satisfied when | Unit |
|---|---|---|---|---|---|
| **MON1** — active network links | **MR** | Maximisation of Reliability | reliability | value **>=** threshold | count of links |
| **MON2** — bandwidth consumption | **MC** | Minimisation of Cost | operational cost | value **<=** threshold | GB/s (see note) |
| **MON3** — time to write data | **MP** | Maximisation of Performance | performance | value **<=** threshold | milliseconds |

> **The numbering does not follow the MC/MR/MP letter order.** MON1 is MR, MON2 is MC, MON3 is MP.
> This is the single easiest thing to get wrong in this codebase — see [Reading the code](#reading-the-code).

### Verbatim definitions [S, §IV]

> **Mon1 — Active Network Links**: provides the current active network links to measure the
> **reliability** of the RDM. The RDM will provide a higher level of reliability with a larger
> number of active links.

> **Mon2 — Bandwidth Consumption**: provides the current bandwidth consumption to measure the
> **operational cost** for the RDM in terms of inter-site network traffic. Operational costs will
> be increased for the RDM with a higher amount of bandwidth consumed. Bandwidth Consumption is
> measured in GigaBytes per second.

> **Mon3 — Time to Write Data to mirrors**: measures the **performance** of the network in terms
> of writing time to maintain multiple copies of data on each remote site. A big writing time
> leads to reduction of performance of the RDM. Time to Write Data is measured in milliseconds.

The three NFRs are named in [S, §II] as "Maximization of Reliability (MR), Maximization of
Performance (MP) and Minimization of Cost (MC)", and identically in [T, §4.2]: "Minimization of
Cost (MC), Maximization of Reliability (MR), and Maximization of Performance (MP)".

### Why the directions differ

MR is a *maximisation* goal, so more active links is better and satisfaction is `>=`. MC and MP
are stated as minimisation of the underlying quantity — cost and writing time — so satisfaction
is `<=`. "Maximisation of Performance" is achieved by *minimising* time to write; the NFR name and
the monitorable run in opposite directions, which is why MON3 uses `<=`.

The trade-off between them is the point of the exemplar [S, §II]:

> A RT Topology offers higher levels of reliability than an MST topology. However, the cost of
> maintaining an RT topology may be prohibitive in some contexts, given the additional cost of
> bandwidth consumption required. Conversely, a MST Topology offers higher levels of performance
> with lower levels of cost than an RT topology. However, the reliability of the system can be
> impacted in a negative way when an MST Topology is used.

---

## Baseline thresholds

There are **two different kinds of threshold** in this literature. They are not interchangeable,
and conflating them is a genuine error — one is a bound on a raw measurement, the other is a bound
on a probability.

### 1. Monitorable thresholds — default scenario S0 [S, §V]

Bounds on the measured values, expressed as percentages of network capacity. Verbatim:

> For the RDMSim, the following thresholds for the levels of satisfaction associated with
> reliability, performance and cost are suggested: **bandwidth consumption should be on average
> less than or equal to 40%**. Similarly, the **time to write data should be on average less than
> or equal to 45%**. On the other hand, the **number of active links should be on average greater
> than or equal to 35%** of the total number of links. The initial topology being used is MST
> topology.

| NFR | Monitorable | Baseline | `configuration.json` key |
|---|---|---|---|
| MR | MON1 active links | **>= 35%** of total links | `link_threshold` |
| MC | MON2 bandwidth | **<= 40%** of total bandwidth | `bandwidth_threshold` |
| MP | MON3 time to write | **<= 45%** of total writing time | `writing_times_threshold` |

**The values in `config_log_files/configuration.json` (35 / 40 / 45) are exactly this baseline.**
The repo ships the paper's default scenario S0 thresholds unmodified.

Converted to absolute values by RDMSim's `NetworkProperties.loadNetworkSettings()` for the default
25-mirror network, where `number_of_links = 25*24/2 = 300`:

| NFR | Derivation | Absolute threshold |
|---|---|---|
| MR (MON1) | 35% x 300 | **>= 105 active links** |
| MC (MON2) | 40% x (300 x 30) | **<= 3600** |
| MP (MON3) | 45% x (300 x 20) | **<= 2700 ms** |

The 30 and 20 factors are the per-link maxima given in [S] footnotes: bandwidth per link varies
20–30 per link, and time to write varies 10–20 ms per link. [S] also sets the active-link fraction
`alpha = 1` for its experiments, which matches the `alpha` column in `output_dir/monitorables.txt`.

### 2. Satisficement SLAs [T, §6.3.1]

Bounds on the *probability* that an NFR is satisfied — i.e. on the belief, not the measurement.
Verbatim:

> The identified SLAs for the NFRs Minimization of Cost (MC), Maximization of Reliability (MR) and
> Maximization of Performance (MP) were:
> `P(MC=True) >= 0.7`, `P(MR=True) >= 0.9`, and `P(MP=True) >= 0.75`

and:

> The default values used for the NFRs Minimization of Cost (MC), Maximization of Reliability (MR)
> and Maximization of Performance (MP) are **[0.7, 0.9, 0.75]**.

| NFR | SLA | Compare against |
|---|---|---|
| MC | `P(MC=True) >= 0.70` | `output_dir/MON2SatProb.txt` |
| MR | `P(MR=True) >= 0.90` | `output_dir/MON1SatProb.txt` |
| MP | `P(MP=True) >= 0.75` | `output_dir/MON3SatProb.txt` |

These are directly comparable to the belief marginals this harness already logs — the
`MON*SatProb.txt` files *are* `P(NFR=True)`. In [T] they are the trigger for reward re-estimation
in the ARRoW approach; this repo does not implement ARRoW, so they are useful here as reference
lines on the satisfaction-probability charts and as a calibration check, not as control inputs.

**Do not substitute one set for the other.** 0.7/0.9/0.75 are probabilities; 40/35/45 are
percentages of network capacity. They are unrelated quantities that happen to both be called
"thresholds".

---

## Reading the code

### State encoding

The 8 POMDP states are the combinations of the three NFR satisfaction flags, ordered
**(MC, MR, MP)** with MC most significant — `POMDP.getInitialStateRDM()`:

| State | MC (bandwidth) | MR (links) | MP (time to write) |
|---|---|---|---|
| 0 | satisfied | satisfied | satisfied |
| 1 | satisfied | satisfied | violated |
| 2 | satisfied | violated | satisfied |
| 3 | satisfied | violated | violated |
| 4 | violated | satisfied | satisfied |
| 5 | violated | satisfied | violated |
| 6 | violated | violated | satisfied |
| 7 | violated | violated | violated |

### Belief marginals

`RDMSim_TAAS.runCaseRDM()` derives `P(NFR=True)` by summing the states where that NFR holds:

```java
double mcsatprob = b[0]+b[1]+b[2]+b[3];   // P(MC=True) -> MON2, bandwidth
double mrsatprob = b[0]+b[1]+b[4]+b[5];   // P(MR=True) -> MON1, active links
double mpsatprob = b[0]+b[2]+b[4]+b[6];   // P(MP=True) -> MON3, time to write
```

### The crossover in the output files

Internal variables use the **MC/MR/MP** names; output files use the **MON1/2/3** names. They are
written in a deliberately non-obvious order:

| Written to | Value | NFR |
|---|---|---|
| `MON1Sat.txt` / `MON1SatProb.txt` | `activelinks` / `mrsatprob` | MR |
| `MON2Sat.txt` / `MON2SatProb.txt` | `bandwidthconsumption` / `mcsatprob` | MC |
| `MON3Sat.txt` / `MON3SatProb.txt` | `timetowrite` / `mpsatprob` | MP |

The same applies to the regression files: `MRRegressionResultsSolvePOMDP.txt` holds active links
(MON1), `MCRegressionResultsSolvePOMDP.txt` holds bandwidth (MON2), and
`MPRegressionResultsSolvePOMDP.txt` holds time to write (MON3). `runSummary.txt` reports under the
`MON1/MON2/MON3` names.

---

## Notes and discrepancies

- **Bandwidth units.** [S] states bandwidth consumption is measured in *GigaBytes* per second
  (GBps). Comments in `src/solver.config` and elsewhere in this repo say "Gbps" (gigabits). The
  simulator is unitless — it multiplies active links by a per-link factor of 20–30 — so nothing
  computational depends on this, but the paper's unit is GB/s.
- **Scenario S0 vs the configured scenario.** The 35/40/45 thresholds are specified in [S] for the
  *default scenario S0*, which applies zero deviation. This repo's baseline runs
  `current_scenario: 3`; the thresholds are unchanged, only the environment deviates.
- **Joint satisfaction.** Because the simulator computes bandwidth and time to write as multiples
  of active links, MON2 and MON3 are monotone increasing in the quantity MON1 wants large. At
  35/40/45 the three cannot generally be satisfied simultaneously, which is why this harness
  reports satisfaction per NFR. This is a property of the exemplar's design — the conflicting
  objectives are the stated research problem [S, §I] — not a misconfiguration.
