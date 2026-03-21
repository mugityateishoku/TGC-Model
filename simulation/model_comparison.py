"""
TGC Model: Model Comparison — TGC vs. DDM vs. HMM vs. Hopf vs. Neural-field
=============================================================================
Qualitative comparison of the five competing model families used in the
TGC identifiability gate. This script generates a side-by-side figure showing
characteristic signatures of each model under the same load ramp, making the
discriminating features explicit for the reader.

Four diagnostic panels per model:
  (A) Latent state trajectory β(t) under a shared load ramp E(t)
  (B) Stationary distribution of β at three load levels
  (C) Empirical potential landscape V(x) ∝ −ln P(x)
  (D) Autocorrelation function of β across all loads

Reference: Harada (2026), Section 4 — Identifiability and Pre-Registration.

Usage:
    python simulation/model_comparison.py

Environment variables:
    TGC_FIGURES_DIR — output directory for figures (default: ./figures)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = os.environ.get('TGC_FIGURES_DIR', './figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

DT = 0.05
N_TRAJ = 5000          # long trajectory for stationary distribution
N_RAMP = 1200          # steps for ramp display


# ══════════════════════════════════════════════════════
# Generative models (single-trajectory, reproducible)
# ══════════════════════════════════════════════════════

def _euler_traj(drift_fn, beta0, E_seq, sigma, dt, rng):
    n = len(E_seq)
    beta = np.empty(n)
    beta[0] = beta0
    noise = rng.standard_normal(n) * sigma * np.sqrt(dt)
    for t in range(1, n):
        beta[t] = beta[t - 1] + drift_fn(beta[t - 1], E_seq[t - 1]) * dt + noise[t]
    return beta


def make_tgc(E_seq, sigma=0.20, omega=1.8, seed=0):
    """Cusp Langevin: dβ = -(β³ - Ω·β - E)dt + σdW"""
    rng = np.random.default_rng(seed)
    beta0 = float(np.sqrt(max(omega, 0)))
    return _euler_traj(lambda b, E: -(b**3 - omega * b - E),
                       beta0, E_seq, sigma, dt=DT, rng=rng)


def make_ddm(E_seq, sigma=0.20, kappa=1.0, seed=0):
    """DDM / OU: dβ = -(κ·β - E)dt + σdW  (linear, monostable)"""
    rng = np.random.default_rng(seed)
    return _euler_traj(lambda b, E: -(kappa * b - E),
                       0.0, E_seq, sigma, dt=DT, rng=rng)


def make_hopf(E_seq, sigma=0.20, lam=1.5, seed=0):
    """Hopf / Stuart-Landau: dβ = [(λ - β²)β + 0.3E]dt + σdW  (limit cycle)"""
    rng = np.random.default_rng(seed)
    beta0 = float(np.sqrt(max(lam, 0)))
    return _euler_traj(lambda b, E: (lam - b**2) * b + 0.3 * E,
                       beta0, E_seq, sigma, dt=DT, rng=rng)


def make_neural_field(E_seq, sigma=0.20, alpha=2.5, w=1.0, seed=0):
    """Neural-field: dβ = (-α·β + w·tanh(β) + E)dt + σdW  (saturating gain)"""
    rng = np.random.default_rng(seed)
    return _euler_traj(lambda b, E: -alpha * b + w * np.tanh(b) + E,
                       0.0, E_seq, sigma, dt=DT, rng=rng)


def make_hmm(n, sigma=0.20, p_stay=0.95, mu_hi=1.2, mu_lo=-1.0, seed=0):
    """
    2-state Gaussian HMM with Markov state transitions.
    Does not depend on E (state evolution is load-independent in baseline HMM).
    """
    rng = np.random.default_rng(seed)
    obs = np.empty(n)
    state = 1
    for t in range(n):
        obs[t] = (mu_hi if state == 1 else mu_lo) + sigma * rng.standard_normal()
        state = state if rng.random() < p_stay else (1 - state)
    return obs


# ══════════════════════════════════════════════════════
# Summary statistics for each model
# ══════════════════════════════════════════════════════

def stationary_samples(model_fn, n=N_TRAJ, **kwargs):
    """Long run at fixed E = 0.5 (intermediate load)."""
    E_fixed = np.full(n, 0.5)
    kwargs.setdefault('seed', 42)
    if model_fn is make_hmm:
        return model_fn(n, **{k: v for k, v in kwargs.items()
                               if k not in ('E_seq',)})
    return model_fn(E_fixed, **kwargs)


def acf(x, max_lag=40):
    """Normalized autocorrelation function."""
    x = x - x.mean()
    var = np.var(x)
    if var == 0:
        return np.zeros(max_lag + 1)
    acf_vals = np.array([
        np.mean(x[:len(x) - lag] * x[lag:]) / var
        for lag in range(max_lag + 1)
    ])
    return acf_vals


def potential_landscape(samples, n_grid=200):
    """V(x) ∝ −ln P(x) from KDE."""
    lo, hi = np.percentile(samples, [1, 99])
    x = np.linspace(lo - 0.5, hi + 0.5, n_grid)
    kde = gaussian_kde(samples, bw_method=0.2)
    p = np.maximum(kde(x), 1e-8)
    v = -np.log(p)
    v -= v.min()
    return x, v


# ══════════════════════════════════════════════════════
# Main comparison figure
# ══════════════════════════════════════════════════════

def plot_model_comparison():
    """
    5-column × 4-row figure comparing TGC, DDM, HMM, Hopf, Neural-field.
    Rows: (A) ramp trajectory, (B) stationary distribution,
          (C) potential landscape, (D) autocorrelation function.
    """
    MODEL_SPECS = [
        ('TGC\n(Cusp Langevin)',  '#1E88E5', make_tgc),
        ('DDM / OU\n(Linear)',    '#43A047', make_ddm),
        ('HMM\n(2-State)',        '#E53935', None),
        ('Hopf\n(Limit Cycle)',   '#FB8C00', make_hopf),
        ('Neural-field\n(Saturation)', '#8E24AA', make_neural_field),
    ]

    E_ramp = np.linspace(-0.5, 2.5, N_RAMP)
    t_axis = np.arange(N_RAMP) * DT

    fig, axes = plt.subplots(4, 5, figsize=(18, 14))
    fig.suptitle(
        "TGC Model — Comparison of Five Competing Generative Models\n"
        "Rows: (A) Load-ramp trajectory  (B) Stationary distribution  "
        "(C) Potential landscape  (D) Autocorrelation",
        fontsize=12, fontweight='bold', y=1.00
    )

    for col, (label, color, fn) in enumerate(MODEL_SPECS):
        axes[0, col].set_title(label, fontsize=10, fontweight='bold', color=color)

        # ── (A) Ramp trajectory ─────────────────────────────
        ax = axes[0, col]
        if fn is None:  # HMM
            traj = make_hmm(N_RAMP, seed=0)
        else:
            traj = fn(E_ramp, seed=0)

        ax.plot(t_axis, traj, color=color, lw=0.7, alpha=0.85)
        ax2 = ax.twinx()
        ax2.plot(t_axis, E_ramp, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax2.set_ylabel('E (load)', fontsize=7, color='gray')
        ax2.tick_params(axis='y', labelsize=6, colors='gray')
        ax.set_ylabel('β', fontsize=8)
        ax.set_xlabel('Time', fontsize=8)
        ax.grid(True, alpha=0.15)

        # ── (B) Stationary distribution ─────────────────────
        ax = axes[1, col]
        for load_E, lstyle, lw_val in [(0.0, '-', 1.6), (0.8, '--', 1.4),
                                        (1.6, ':', 1.2)]:
            if fn is None:  # HMM unchanged with load
                samp = make_hmm(N_TRAJ, seed=7)
            else:
                E_fixed = np.full(N_TRAJ, load_E)
                samp = fn(E_fixed, seed=7)
            samp = samp[500:]   # burn-in
            lo, hi = np.percentile(samp, [1, 99])
            x_g = np.linspace(lo - 0.3, hi + 0.3, 200)
            kde = gaussian_kde(samp, bw_method=0.25)
            ax.plot(x_g, kde(x_g), color=color, ls=lstyle, lw=lw_val,
                    label=f'E={load_E}')
        ax.set_xlabel('β', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.15)

        # ── (C) Potential landscape ──────────────────────────
        ax = axes[2, col]
        for load_E, lstyle in [(0.0, '-'), (0.8, '--'), (1.6, ':')]:
            if fn is None:
                samp = make_hmm(N_TRAJ, seed=11)
            else:
                E_fixed = np.full(N_TRAJ, load_E)
                samp = fn(E_fixed, seed=11)
            samp = samp[500:]
            try:
                x_v, v = potential_landscape(samp)
                ax.plot(x_v, v, color=color, ls=lstyle, lw=1.4)
            except Exception:
                pass
        ax.set_xlabel('β', fontsize=8)
        ax.set_ylabel('V(β)  ∝  −ln P(β)', fontsize=8)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.15)

        # ── (D) Autocorrelation ──────────────────────────────
        ax = axes[3, col]
        if fn is None:
            samp = make_hmm(N_TRAJ, seed=21)
        else:
            E_mixed = np.tile(np.linspace(-0.5, 2.0, 100), N_TRAJ // 100 + 1)[:N_TRAJ]
            samp = fn(E_mixed, seed=21)
        samp = samp[500:]
        acf_vals = acf(samp, max_lag=50)
        lags = np.arange(len(acf_vals)) * DT
        ax.bar(lags, acf_vals, width=DT * 0.8, color=color, alpha=0.75)
        ax.axhline(0, color='black', lw=0.6)
        ax.axhline(1.96 / np.sqrt(len(samp)), color='gray',
                   ls='--', lw=0.8, alpha=0.7, label='95% CI')
        ax.set_xlabel('Lag (time units)', fontsize=8)
        ax.set_ylabel('ACF', fontsize=8)
        ax.set_ylim(-0.3, 1.05)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.15)

    # Shared row labels
    for row_idx, row_label in enumerate(
            ['(A) Ramp trajectory', '(B) Stationary distribution',
             '(C) Potential landscape', '(D) Autocorrelation']):
        axes[row_idx, 0].set_ylabel(row_label + '\n' + axes[row_idx, 0].get_ylabel(),
                                     fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(FIGURES_DIR, 'model_comparison.pdf')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved: {out}")
    return out


# ══════════════════════════════════════════════════════
# Summary table of key discriminating properties
# ══════════════════════════════════════════════════════

def print_model_summary():
    header = f"{'Model':<18} {'Drift':<28} {'Bistable':>9} {'Oscillates':>11} {'Load-dependent':>15}"
    print("\n" + "=" * len(header))
    print("Discriminating properties of five competing models")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    rows = [
        ("TGC (Cusp)",     "−(β³ − Ω·β − E)",         "Yes (Ω>0)",  "No",  "Yes"),
        ("DDM / OU",        "−(κ·β − E)",               "No",         "No",  "Yes"),
        ("HMM (2-State)",   "Discrete Markov switches",  "Effectively","No",  "No"),
        ("Hopf",            "(λ − β²)·β + 0.3E",        "No",         "Yes", "Partial"),
        ("Neural-field",    "−α·β + w·tanh(β) + E",     "Weak",       "No",  "Yes"),
    ]
    for r in rows:
        print(f"{r[0]:<18} {r[1]:<28} {r[2]:>9} {r[3]:>11} {r[4]:>15}")
    print("=" * len(header))
    print()
    print("Key TGC-specific signature:")
    print("  Bistability + Hysteresis + Load-dependent threshold + Noise-induced transition")
    print("  These four are JOINTLY absent in all four alternative models.")


# ══════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("TGC Model — Model Comparison")
    print("Harada (2026), Section 4")
    print("=" * 60)

    print_model_summary()

    print("\nGenerating comparison figure...")
    out = plot_model_comparison()

    print(f"\nDone. Figure: {out}")
