# Pre-Registration Template — TGC Model Confirmatory Study

**Reference:** Harada (2026), Section 4 — Pre-Registered Falsification Protocol
**Status:** Template for submission to OSF prior to data collection

---

## 1. Study Information

**Title:** Confirmatory Test of the Thermostatic Gain Control (TGC) Model:
Bistability, Hysteresis, Threshold Asymmetry, and Overheating under Cognitive Load

**Authors:** [redacted for blind review]

**Study type:** Confirmatory, pre-registered

**Intended registry:** OSF Registries (osf.io)

---

## 2. Hypotheses

Based on the TGC model (Eq. 1), four falsifiable predictions are tested:

| Prediction | Operationalisation | Falsification criterion |
|---|---|---|
| P1: Bistability | Bimodal accuracy distribution at Load 9 (intermediate) | Hartigan dip *p* ≥ 0.05 at Load 9 |
| P2: Hysteresis | Collapse threshold > recovery threshold; loop area A > 0 | Circular-shift permutation *p* ≥ 0.05 |
| P3: Threshold Asymmetry | Low-Ω group collapses at lower E than high-Ω group | t-test on collapse threshold *p* ≥ 0.05 |
| P4: Overheating | Pupil dilation peak precedes performance collapse by ≥ 1 trial | Correlation pupil × RT-IIV *p* ≥ 0.05 |

The model is **falsified** if two or more predictions fail.

---

## 3. Design

### Paradigm
- Adaptive N-back digit-span task with load ramp: E increases from 5 to 15 digits,
  then decreases (ascending–descending ramp design)
- Two ramp directions (ascending / descending) counter-balanced within subjects
- Ω manipulation: dual task (high cognitive load on secondary task) vs. single task

### Participants
- N = 60 participants (power calculation based on identifiability gate results)
- Inclusion: age 18–35, normal/corrected-to-normal vision, no neurological history
- Exclusion: ADHD, anxiety disorders, medication affecting attention

### Measures
- **Primary:** Accuracy (% correct recall), trial-by-trial
- **Secondary:** Pupil diameter (LC-NE proxy), RT intra-individual variability (IIV)

### Data collection
- Lab-controlled environment, 60 Hz eye tracking, keyboard response
- Total duration: ~90 min per participant

---

## 4. Identifiability Gate (prerequisite)

**MUST be completed before any data collection.**

Run `simulation/identifiability_gate.py` and verify:

- [ ] r(Ω) ≥ 0.80
- [ ] r(σ) ≥ 0.80
- [ ] Confusion-matrix diagonal ≥ 80% for all five model classes

If any check fails, the measurement design must be revised and the gate re-run.

---

## 5. Analysis Plan

### Pre-processing
1. Exclude trials with RT < 100 ms or RT > 3 × IQR above median
2. Pupil: baseline-correct per trial (−500 ms to 0 ms window), interpolate blinks
3. Compute per-subject load–accuracy curves (smoothed with 3-trial moving average)

### Primary analyses
1. **Bistability (P1):** Hartigan dip test on accuracy distribution at Load 9
2. **Hysteresis (P2):** Calculate loop area A = ∫(β_asc − β_desc) dE;
   circular-shift permutation test (1000 permutations)
3. **Threshold asymmetry (P3):** Estimate collapse threshold per subject per condition;
   paired t-test (low-Ω vs. high-Ω)
4. **Overheating (P4):** Time-lagged cross-correlation of pupil × RT-IIV;
   test peak lag against zero

### Model fitting
- Fit TGC parameters (Ω, E, σ) per subject via analytical OLS (see identifiability_gate.py)
- Fit DDM, HMM, Hopf, Neural-field as null models
- Model comparison via BIC

### Multiple comparisons
- Bonferroni correction across four primary tests: α = 0.05/4 = 0.0125
- All analyses conducted blind to condition assignment

---

## 6. Falsification Criteria

The TGC model is **falsified** if, in this confirmatory study:

- (a) Accuracy distributions remain unimodal at all load levels (P1 fails)
- (b) Ascending and descending ramps produce equivalent thresholds (P2 fails)
- (c) Collapse threshold does not differ under Ω manipulation (P3 fails)
- (d) Pupil/RT-IIV shows no pre-collapse elevation (P4 fails)

**Falsification rule:** Violation of any **two** criteria constitutes refutation.

---

## 7. Deviations from Protocol

Any deviations from this pre-registered protocol will be reported in the
manuscript under "Deviations from Pre-Registration". Exploratory analyses
beyond the pre-registered tests will be clearly labelled as such.

---

*Template version: 1.0 — Harada (2026)*
*To be submitted to OSF prior to data collection.*
