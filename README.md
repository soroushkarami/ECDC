
# ECDC: Emergency-Centric Dynamic Clustering for VANETs

Reference implementation, simulation harnesses, and analysis scripts for the paper:

> **ECDC: An Emergency-Centric Dynamic Clustering for Reliable Uplink Emergency Message Dissemination in VANETs** (submitted to *IEEE Open Journal of the Communications Society*, OJCOMS-07281-2026).

This repository contains everything needed to reproduce the *numerical* results reported in the paper. It is released under a **private, reviewer-restricted** access policy during peer review.

---

## Scope

### What is included

- **ECDC implementation** — the algorithm itself, plus the two simulation harnesses (mixed highway–intersection, double-intersection) under `ecdc/`.
- **Network XML files** — SUMO network, edges, nodes, routes, vehicle types, and additional files for both topologies under `network/emergency/`.
- **Aggregate per-run results** — one CSV per algorithm × scenario × density, each containing the five paired-seed runs whose means are reported in the paper's tables. Located in `logs/aggregate_csvs/`.
- **Per-packet and per-step event logs for ECDC** — the emergency-delay logs and the per-step cluster-event logs used to derive ECDC's SFR, EMDD, LCD, and CCR results. Located in `logs/mixed/` and `logs/doubleinter/`.
- **Analysis scripts** — the exact scripts that produced every numerical table in the paper, under `analysis/`.

### What is *not* included, and why

- **Baseline source code (MCSC, IMICLiVAN, Zone-based).** These are re-implementations of published work by other authors. To respect the original authors' rights, we do not redistribute their source code. Baseline *results* are fully reported in the aggregate CSVs under `logs/aggregate_csvs/`, which is sufficient for verifying every comparison in the paper.
- **Per-vehicle network logs (`network_data/*.txt`).** These files are hundreds of MB per scenario and are not needed to reproduce any reported metric. The two analysis scripts that depend on them (`active_population.py`, `amb_speed.py`) are placed under `analysis/optional/` with a header comment explaining how to regenerate the input files if required.
- **Figure-generation scripts.** The paper's figures are derived directly from the aggregate CSVs in `logs/aggregate_csvs/`; the plotting code itself is not included.
- **Ablation aggregate CSVs.** Tables 18–20 of the paper report ablation results (`fixedradius`, `nohyst`, `randrelay`, `nogate`, `directbs`, `oldrch`, `naive_ea`, `nojaccard`). These were produced by setting the `ECDC_ABLATION` environment variable and re-running the ECDC harnesses. The aggregated outputs are not distributed here, but the mechanism is preserved in `ecdc/ecdc_clustering.py` and can be reproduced by running the harnesses with the corresponding `ECDC_ABLATION` value.

---

## Repository layout

```
ECDC/
├── ecdc/                          ECDC algorithm + two harnesses
│   ├── ecdc_clustering.py
│   ├── ecdc_emergency.py          Mixed highway–intersection scenario
│   └── ecdc_doubleinter.py        Double-intersection scenario
├── ablations/                     Stripped-down emergency-anchored comparator
│   ├── naive_ea_clustering.py
│   ├── naive_ea_emergency.py
│   └── naive_ea_doubleinter.py
├── analysis/                      Scripts that produce the paper's tables
│   ├── statistical_analysis.py    Table 17 — paired CIs, p-values
│   ├── sfr_decomposition.py       Table 10 — SFR decomposition
│   ├── lcd_reconciliation.py      Table 12 — LCD reconciliation
│   ├── impairment_sensitivity.py  Table 5  — per-hop PHY success
│   ├── mac_layer_check.py         Table 6  — Bianchi validation
│   ├── max_latency.py             Table 10 — max successful latency
│   ├── compute_p95.py             Table 10 — 95th-percentile delay
│   └── optional/                  Require the omitted per-vehicle logs
│       ├── active_population.py
│       └── amb_speed.py
├── network/emergency/             SUMO network files
│   ├── mixed_network_1km.net.xml  Compiled mixed-topology network
│   ├── edge.xml, node.xml         Edge / node definitions
│   ├── route.xml                  Route definitions
│   ├── vtype.add.xml              Vehicle type definitions
│   ├── additional.xml             Additional SUMO config (POIs, RSUs, BS)
│   └── double_inter/
│       ├── double_inter.net.xml   Compiled double-intersection network
│       ├── edge.xml, node.xml
│       ├── route.xml
│       ├── vtype.add.xml
│       └── additional.xml
├── logs/
│   ├── aggregate_csvs/            Per-run summary CSVs (all 4 algorithms)
│   ├── mixed/                     ECDC per-packet + per-step logs (mixed)
│   └── doubleinter/               ECDC per-packet + per-step logs (double)
├── metrics_clustering.py          Shared clustering-metric helpers
├── net_metrics.py                 PHY/MAC model (path loss, Bianchi, energy)
├── requirements.txt
└── README.md
```

---

## Requirements

### Python packages

Install with:

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scipy` (see `requirements.txt`).

### SUMO

SUMO is **not** a pip package. It must be installed separately:

1. Download from https://sumo.dlr.de/docs/Installing/index.html (tested with SUMO 1.26.0).
2. Set the environment variable:

```bash
export SUMO_HOME=/path/to/sumo
# Windows: setx SUMO_HOME "C:\path\to\sumo"
```

3. Verify: `sumo --version` and `echo $SUMO_HOME`.

The `traci` library is provided by SUMO under `$SUMO_HOME/tools/traci` and is *not* a separate pip package. Do not install `traci` from PyPI — it is an unrelated project and will shadow the correct module.

The analysis scripts under `analysis/` (other than the two in `optional/`) do **not** require SUMO. Only the simulation harnesses under `ecdc/` and `ablations/` do.

---

## Running the simulations

All simulation scripts must be run from the repository root. Example for the mixed scenario at density 240, seed 43, five repetitions:

```bash
python ecdc/ecdc_emergency.py \
    --density 240 --seed 43 --reps 5 --amb_depart 150
```

And for the double-intersection scenario at density 240, seed 42:

```bash
python ecdc/ecdc_doubleinter.py \
    --density 240 --seed 42 --reps 5 --amb_depart 150
```

Seeds used in the paper:

- Mixed: 43–47 (five seeds, `--seed 43 --reps 5`)
- Double: 42–46 (five seeds, `--seed 42 --reps 5`)

Output aggregate CSVs are written to `metrics/ecdc_emergency/clustering_data/` and `metrics/ecdc_doubleinter/clustering_data/` (the same naming convention as the files under `logs/aggregate_csvs/`). Per-packet and per-step logs are written to the same directory.

### Ablations

Ablation runs of ECDC are enabled via the `ECDC_ABLATION` environment variable, with one of the following values:

```
fixedradius, nohyst, randrelay, nogate, directbs, oldrch, nojaccard
```

Example:

```bash
ECDC_ABLATION=nogate python ecdc/ecdc_emergency.py \
    --density 240 --seed 43 --reps 5 --amb_depart 150
```

The `ablations/naive_ea_*.py` harnesses implement the stacked emergency-anchored comparator (`naive_ea`) referenced in Section V.E of the paper.

---

## Reproducing the paper's tables

All commands are run from the repository root. The analysis scripts read directly from `logs/aggregate_csvs/` and write their output to `analysis/` (or print it to stdout).

| Paper table | Script |
|---|---|
| Table 5  — Per-hop PHY success by channel model | `python analysis/impairment_sensitivity.py` |
| Table 6  — Bianchi validation | `python analysis/mac_layer_check.py` |
| Table 10 — SFR decomposition and max latency | `python analysis/sfr_decomposition.py --scenario emergency --density 240 --depart 150` |
| Table 10 — 95th-percentile end-to-end delay | `python analysis/compute_p95.py` |
| Table 12 — LCD reconciliation | `python analysis/lcd_reconciliation.py --scenario emergency --density 240 --depart 150` |
| Table 17 — Paired statistical comparison | `python analysis/statistical_analysis.py --scenario emergency --density 240 --depart 150` |

The remaining tables in the paper (Tables 3, 9, 11, 13–16, 18–20) are directly readable from the aggregate CSVs under `logs/aggregate_csvs/` (and, for Tables 18–20, from the ablation runs described above).

For the double-intersection scenario, replace `--scenario emergency` with `--scenario doubleinter` and use `--density {100,240}` as needed.

---

## Reproducibility notes

- All simulation runs use hardcoded seeds (`--seed` above), so the four algorithms see identical mobility traces per repetition.
- The ambulance is configured with a departure speed of 20.0 m/s and a maximum speed of 25.0 m/s (Section IV.A of the paper). At 20.0 m/s, the emergency-cluster radius reaches R_EC = 300 m, exercising the beyond-range pre-registration mechanism.
- The background application load is held constant (20 packets per step) across densities to isolate density effects from application-load effects.
- All hardware timings reported in Section III.D were measured on an AMD A10-9600P APU (2.4 GHz, 8 GB RAM), running Python 3.13 and SUMO 1.26.0.

---

## License and third-party components

This repository contains original code for ECDC and the `naive_ea` comparator, plus network XML files generated with SUMO. The SUMO tools themselves are covered by the Eclipse Public License 2.0 and are not redistributed here.

Baseline implementations (MCSC, IMICLiVAN, Zone-based) are described in the paper's references but are not included in this repository. See the "Scope" section above for the rationale.

---

## Contact

For questions about the code or to request access during double-blind review, please contact the corresponding author.

Corresponding author: Dr. Seyed Amir Asghari (`s.a.asghari@ut.ac.ir`)  
School of Intelligent Systems, University of Tehran
```