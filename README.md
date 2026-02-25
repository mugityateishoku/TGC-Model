# The Thermostatic Gain Control (TGC) Model 5.0
### A Topologically Constrained Decision-Making Model of the Autism-ADHD Spectrum via Cusp Catastrophe Dynamics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-orange)

**Author:** Aruma Harada (High School Researcher)  
**Target:** Computational Psychiatry / Theoretical Neuroscience

---

## 📌 Overview
The **Thermostatic Gain Control (TGC) Model 5.0** is a unified generative framework integrating Markov Decision Processes (MDP) with **Cusp Catastrophe Theory**. It maps the biological "Exploration-Exploitation Dilemma" to a topological state governed by a potential landscape, providing a strictly falsifiable mathematical foundation for clinical phenotypes.

Unlike standard 1D linear stochastic accumulation models (e.g., Drift Diffusion Models), the TGC model intrinsically captures non-linear behavioral dynamics and structural hysteresis through two operationalized control parameters:
* **Stability Factor ($\Omega$):** Baseline network stability (e.g., tonic LC arousal or GABAergic E/I balance).
* **Input Drive ($E$):** Dynamic combination of external cognitive load and internal reward prediction error.



### 🧠 Clinical Phenotype Mapping
* **ADHD-like (Low-$\Omega$ Topology):** Produces shallow attractor basins, leading to continuous noise-driven switching and quantitative bimodality in behavioral and autonomic outputs.
* **ASD-like (High-$\Omega$ Topology):** Produces deep hysteresis loops ($A > 0$). Demonstrates "Hyper-Systemizing" in static environments but suffers catastrophic phase transitions ("meltdowns") when prediction errors exceed the critical bifurcation set:
  $$|E_{crit}| > \sqrt{\frac{4\Omega^3}{27}}$$

<div align="center">
  <img src="docs/figure1.png" width="80%" alt="Hysteresis Loop Dynamics">
  <br>
  <em>Figure 1: The Hysteresis Loop ($A = \oint \beta \, dE$) under a Catastrophe-Forcing Protocol (CFP). The high-$\Omega$ agent exhibits sudden phase transitions and structural path-dependence that linear models fail to capture.</em>
</div>

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone [https://github.com/mugityateishoku/TGC-Model.git](https://github.com/mugityateishoku/TGC-Model.git)
cd TGC-Model
pip install -r requirements.txt