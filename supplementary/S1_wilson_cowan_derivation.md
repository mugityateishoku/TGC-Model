# Supplementary S1 — Wilson–Cowan Derivation of the TGC Langevin Equation

**Reference:** Harada (2026), Appendix A

---

## Overview

This note derives Equation (1) of the main text from a Wilson–Cowan
excitatory–inhibitory (E/I) circuit. The derivation proceeds in three steps:

1. Write the full 2D Wilson–Cowan system
2. Perform adiabatic elimination of the inhibitory variable
3. Expand around the bifurcation point to obtain the 1D cusp normal form

---

## Step 1: Wilson–Cowan E/I Circuit

Let *r_E* and *r_I* denote the firing rates of excitatory and inhibitory
populations, respectively. The standard Wilson–Cowan equations are:

```
τ_E · ṙ_E = −r_E + F(w_EE · r_E − w_EI · r_I + I_E)
τ_I · ṙ_I = −r_I + G(w_IE · r_E − w_II · r_I + I_I)
```

where *F*, *G* are sigmoidal gain functions, *w_{AB}* are synaptic weights,
and *I_E*, *I_I* are external inputs. The noise term is added as

```
dW_E = ε_E dt,   dW_I = ε_I dt,   ε ~ N(0, σ²)
```

---

## Step 2: Adiabatic Elimination

Assume the inhibitory time constant is fast relative to excitation:
τ_I ≪ τ_E. Then *r_I* equilibrates quasi-statically:

```
r_I* = G(w_IE · r_E + I_I) / (1 + w_II · G')
```

where *G'* is the derivative of *G* at the operating point.
Substituting back into the excitatory equation yields a 1D effective equation:

```
τ_E · ṙ_E = −r_E + F_eff(r_E, I_E, I_I)
```

with effective gain function

```
F_eff(r_E) = F(w_EE · r_E − w_EI · r_I*(r_E) + I_E)
```

---

## Step 3: Cusp Normal Form

Taylor-expand *F_eff* around the bifurcation point (r_E*, I_E*):

Let β = r_E − r_E* (deviation from fixed point), Ω = ∂²F_eff/∂r_E² at the
bifurcation (the splitting factor), E = I_E − I_E* (deviation from critical
input, the normal factor). Keeping terms up to cubic order:

```
τ_E · β̇ = −β³ + Ω·β + E + O(β⁴)
```

This is precisely the cusp normal form. Adding the noise from the excitatory
population (σ dW_t) and rescaling time by τ_E gives Equation (1):

```
dβ_t = −(β³_t − Ω·β_t − E) dt + σ dW_t          (Eq. 1)
```

with cusp potential

```
V(β; Ω, E) = β⁴/4 − Ω·β²/2 − E·β
```

---

## Parameter Identification

| TGC parameter | Wilson–Cowan origin |
|---|---|
| Ω (cortical stability) | E/I balance: ∂²F_eff/∂r_E² — positive when excitation dominates |
| E (input drive) | Combined cognitive load + sensory prediction error |
| σ (noise amplitude) | Neural fluctuations + LC-NE phasic drive |
| β (neural gain) | Deviation of excitatory rate from critical point |

---

## Bifurcation Structure

The cusp catastrophe has a fold set defined by:

```
∂V/∂β = 0   and   ∂²V/∂β² = 0
```

which gives the bifurcation curve:

```
E_crit = ±(2/3) · (Ω/3)^(3/2)   (for Ω > 0)
```

Within the cusp (|E| < E_crit), the system is bistable with two stable
fixed points separated by an unstable saddle. Outside (|E| > E_crit),
the system is monostable.

---

*This derivation follows standard methods in computational neuroscience;
see Wilson & Cowan (1972), Strogatz (1994), and Breakspear (2017).*
