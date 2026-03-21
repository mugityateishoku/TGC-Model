# The Thermostatic Gain Control (TGC) Model

**A Stochastic Cusp-Catastrophe Account of Neural Gain Instability**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Under_Review-orange)

**Author:** Aruma Harada  
**Affiliation:** Independent Researcher  
**Target Journal:** Computational Psychiatry  
**Preprint / Submission:** March 2026

---

## Overview

The **Thermostatic Gain Control (TGC) model** is a formally derived stochastic cusp-catastrophe account of abrupt, asymmetric performance collapse under cognitive overload. The model derives a one-dimensional Langevin reduction from a Wilson–Cowan excitatory–inhibitory circuit and shows that the resulting potential possesses a cusp topology controlled by two parameters:

- **Cortical Stability (Ω):** Splitting factor linked to local E/I balance; determines whether the system is monostable or bistable.
- **Input Drive (E):** Normal factor capturing combined cognitive load and sensory prediction error.

### Core Equation

The latent neural gain state β evolves according to:

```
dβ_t = −(β³_t − Ω·β_t − E) dt + σ dW_t        (1)
```

with cusp potential `V(β; Ω, E) = β⁴/4 − Ω·β²/2 − E·β`.

### Four Falsifiable Predictions

| # | Prediction | Diagnostic |
|---|-----------|-----------|
| 1 | **Bistability** | Bimodal performance distributions under intermediate load |
| 2 | **Hysteresis** | Collapse threshold > recovery threshold (loop area A > 0) |
| 3 | **Threshold Asymmetry** | Lower Ω → earlier collapse at lower E |
| 4 | **Overheating** | Pre-collapse autonomic spike (σ-driven, not E-driven transition) |

These predictions are jointly absent from linear models, standard drift-diffusion models, and switching-HMM accounts.

---

## Repository Structure

```
TGC-Model/
├── README.md
├── requirements.txt
├── LICENSE
│
├── simulation/
│   ├── tgc_langevin.py              # Stochastic Langevin simulation (Eq. 1)
│   └── cusp_deterministic.py        # Deterministic bifurcation & hysteresis demo
│
├── analysis/
│   ├── study1_ds003838.py           # Study 1: ds003838 integrated analysis (beh+pupil+EEG)
│   ├── study1_pupil.py              # Study 1: pupil overheating signature
│   ├── study1_eeg.py                # Study 1: EEG 1/f aperiodic slope
│   ├── study2_abide.py              # Study 2: ABIDE resting-state fMRI 1/f (null result)
│   ├── study2_abide_fdr.py          # Study 2: FDR correction for multi-ROI comparison
│   ├── study3_sfari.py              # Study 3: SFARI ASSR 40 Hz (null result)
│   ├── study4_cogbci.py             # Study 4: COG-BCI N-back aperiodic exponent
│   └── study1_supplementary/
│       ├── gmm_bimodality.py        # GMM/BIC bimodality test
│       ├── pupil_single_subject.py  # Event-related pupillometry (single subject)
│       ├── hysteresis_prev_condition.py  # Hysteresis via previous-trial condition
│       ├── rt_iiv_overheating.py    # RT intra-individual variability
│       └── pipeline_v1.py           # Earlier comprehensive pipeline version
│
├── figures/
│   └── (generated figures)
│
└── supplementary/
    ├── S1_wilson_cowan_derivation.md
    ├── S2_simulation_confusion_matrices.md
    └── preregistration_template.md
```

> **Note:** Studies 2–3 (ABIDE, SFARI) yielded null results with known methodological limitations. Study 4 (COG-BCI) showed results consistent with TGC predictions but non-diagnostic. All analyses in Section 3 of the paper are **non-confirmatory boundary checks**, not evidential tests. See the paper and the pre-registered protocol (Section 4) for the planned confirmatory study.

---

## Quick Start

### Installation

```bash
git clone https://github.com/mugityateishoku/TGC-Model.git
cd TGC-Model
pip install -r requirements.txt
```

### Run Simulations

```bash
# Stochastic Langevin dynamics with cusp potential
python simulation/tgc_langevin.py

# Deterministic bifurcation diagram and hysteresis loop
python simulation/cusp_deterministic.py
```

### Run Boundary-Check Analyses

Each analysis script requires the corresponding open dataset to be downloaded separately. See the docstring in each script for dataset source and download instructions.

```bash
# Study 1: ds003838 digit-span (behavioral + pupil + EEG)
python analysis/study1_ds003838.py

# Study 4: COG-BCI N-back aperiodic exponent
python analysis/study4_cogbci.py
```

---

## Identifiability Gate (Pre-Registered)

Before any empirical data collection, the following must be satisfied:

1. **Parameter recovery:** r ≥ 0.80 for Ω, E, σ in hierarchical simulation (n=30 subjects, 600 trials, 50 reps)
2. **Model discrimination:** Confusion-matrix diagonal ≥ 80% across five competing generative models (TGC, DDM, HMM, Hopf, Neural-field)

Simulation code: `simulation/identifiability_gate.py`

---

## Falsification Criteria

The TGC model is **falsified** if, in the pre-registered study:

- (a) Performance distributions remain unimodal at all load levels
- (b) Ascending and descending ramps produce equivalent thresholds
- (c) Collapse threshold does not differ under experimentally manipulated Ω
- (d) Pre-collapse autonomic indices show no elevation

Violation of any two criteria constitutes refutation.

---

## Citation

If you use this model or code, please cite:

```
Harada, A. (2026). A Stochastic Cusp-Catastrophe Account of Neural Gain
Instability: Formal Derivation, Identifiability, and a Pre-Registered
Falsification Protocol. [Under review at Computational Psychiatry]
```

---

## Data Availability

All analyses use openly available datasets:

| Study | Dataset | Source |
|-------|---------|--------|
| 1 | ds003838 | [OpenNeuro](https://openneuro.org/datasets/ds003838) |
| 2 | ABIDE I | [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/) |
| 3 | SFARI ds006780 | [OpenNeuro](https://openneuro.org/datasets/ds006780) |
| 4 | COG-BCI | [Hinss et al. (2023)](https://doi.org/10.1038/s41597-023-01956-1) |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
