"""
TGC Model: Identifiability Gate (Pre-Registered)
=================================================
Two mandatory checks before any empirical data collection:

  1. Parameter recovery — r ≥ 0.80 for Ω and σ
     (n=30 subjects × 600 trials × 50 repetitions)

  2. Model discrimination — confusion-matrix diagonal ≥ 80%
     Five competing generative models:
       TGC (cusp Langevin), DDM/OU (linear), HMM (2-state),
       Hopf (Stuart-Landau), Neural-field (saturating linear)

Both gates must pass before the pre-registered confirmatory study proceeds.
Fitting uses an analytical OLS rearrangement of the Euler-Maruyama equation,
which is exact in the limit dt → 0 and fast enough for the required scale.

Reference: Harada (2026), Section 4 — Identifiability and Pre-Registration.

Usage:
    python simulation/identifiability_gate.py

Environment variables:
    TGC_FIGURES_DIR — output directory for figures (default: ./figures)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = os.environ.get('TGC_FIGURES_DIR', './figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# Integration timestep and load ramp (fixed for all simulations)
DT = 0.05
N_TRIALS = 600
E_SEQ = np.linspace(-0.5, 2.0, N_TRIALS)   # known load ramp

# ══════════════════════════════════════════════════════
# 1. Generative models (Euler-Maruyama simulation)
# ══════════════════════════════════════════════════════

def _euler(drift_fn, beta0, E_seq, sigma, dt, rng):
    n = len(E_seq)
    beta = np.empty(n)
    beta[0] = beta0
    noise = rng.standard_normal(n) * sigma * np.sqrt(dt)
    for t in range(1, n):
        beta[t] = beta[t - 1] + drift_fn(beta[t - 1], E_seq[t - 1]) * dt + noise[t]
    return beta


def sim_tgc(omega, sigma, E_seq=E_SEQ, dt=DT, rng=None):
    """TGC cusp Langevin: dβ = -(β³ - Ω·β - E)dt + σdW"""
    if rng is None:
        rng = np.random.default_rng()
    beta0 = float(np.sqrt(max(omega, 0)))
    return _euler(lambda b, E: -(b**3 - omega * b - E), beta0, E_seq, sigma, dt, rng)


def sim_ou(kappa, sigma, E_seq=E_SEQ, dt=DT, rng=None):
    """OU / DDM-like: dβ = -(κ·β - E)dt + σdW  (monostable linear)"""
    if rng is None:
        rng = np.random.default_rng()
    return _euler(lambda b, E: -(kappa * b - E), 0.0, E_seq, sigma, dt, rng)


def sim_hopf(lam, sigma, E_seq=E_SEQ, dt=DT, rng=None):
    """Stuart-Landau / Hopf: dβ = [(λ - β²)β + 0.3E]dt + σdW  (limit cycle)"""
    if rng is None:
        rng = np.random.default_rng()
    beta0 = float(np.sqrt(max(lam, 0))) if lam > 0 else 0.1
    return _euler(lambda b, E: (lam - b**2) * b + 0.3 * E, beta0, E_seq, sigma, dt, rng)


def sim_neural_field(alpha, w, sigma, E_seq=E_SEQ, dt=DT, rng=None):
    """Neural-field: dβ = (-α·β + w·tanh(β) + E)dt + σdW  (saturating gain)"""
    if rng is None:
        rng = np.random.default_rng()
    return _euler(lambda b, E: -alpha * b + w * np.tanh(b) + E, 0.0, E_seq, sigma, dt, rng)


def sim_hmm(p_high, p_low, mu_hi, mu_lo, sigma_obs, n, rng=None):
    """
    2-state Gaussian HMM with discrete jumps.
    p_high = P(stay in high state), p_low = P(stay in low state).
    Returns observation sequence (not Langevin trajectory).
    """
    if rng is None:
        rng = np.random.default_rng()
    obs = np.empty(n)
    state = 1  # start high
    for t in range(n):
        obs[t] = (mu_hi if state == 1 else mu_lo) + sigma_obs * rng.standard_normal()
        if state == 1:
            state = 1 if rng.random() < p_high else 0
        else:
            state = 0 if rng.random() < p_low else 1
    return obs


# ══════════════════════════════════════════════════════
# 2. Fitting via analytical OLS
# ══════════════════════════════════════════════════════

def _sigma_from_residuals(beta_obs, drift_vals, dt):
    residuals = beta_obs[1:] - (beta_obs[:-1] + drift_vals * dt)
    return max(np.std(residuals) / np.sqrt(dt), 1e-6)


def _nll_langevin(beta_obs, drift_vals, sigma, dt):
    """Gaussian transition NLL for Langevin dynamics."""
    residuals = beta_obs[1:] - (beta_obs[:-1] + drift_vals * dt)
    var = sigma**2 * dt
    return 0.5 * float(np.sum(residuals**2 / var + np.log(2 * np.pi * var)))


def fit_tgc(beta_obs, E_seq=E_SEQ, dt=DT):
    """
    OLS fit of Ω (and σ from residuals).
    Rearrangement: Δβ/dt + β³ - E = Ω·β → Ω = (β' y) / (β' β)
    """
    b = beta_obs[:-1]
    E = E_seq[:-1]
    y = (beta_obs[1:] - b) / dt + b**3 - E
    omega = np.dot(b, y) / (np.dot(b, b) + 1e-12)
    omega = max(omega, 0.0)
    drift_vals = -(b**3 - omega * b - E)
    sigma = _sigma_from_residuals(beta_obs, drift_vals, dt)
    nll = _nll_langevin(beta_obs, drift_vals, sigma, dt)
    n_params = 2  # Ω, σ
    bic = 2 * nll + n_params * np.log(len(beta_obs) - 1)
    return omega, sigma, nll, bic


def fit_ou(beta_obs, E_seq=E_SEQ, dt=DT):
    """
    OLS fit of κ.
    Rearrangement: Δβ/dt - E = -κ·β → κ = -(b' (Δβ/dt - E)) / (b' b)
    """
    b = beta_obs[:-1]
    E = E_seq[:-1]
    y = (beta_obs[1:] - b) / dt - E
    kappa = -np.dot(b, y) / (np.dot(b, b) + 1e-12)
    kappa = max(kappa, 0.01)
    drift_vals = -(kappa * b - E)
    sigma = _sigma_from_residuals(beta_obs, drift_vals, dt)
    nll = _nll_langevin(beta_obs, drift_vals, sigma, dt)
    n_params = 2  # κ, σ
    bic = 2 * nll + n_params * np.log(len(beta_obs) - 1)
    return kappa, sigma, nll, bic


def fit_hopf(beta_obs, E_seq=E_SEQ, dt=DT):
    """
    OLS fit of λ.
    Rearrangement: Δβ/dt - 0.3E = (λ - β²)·β = λ·β - β³
    → (y + β³) = λ·β → λ = (b' (y + b³)) / (b' b)
    """
    b = beta_obs[:-1]
    E = E_seq[:-1]
    y = (beta_obs[1:] - b) / dt - 0.3 * E
    lam = np.dot(b, y + b**3) / (np.dot(b, b) + 1e-12)
    drift_vals = (lam - b**2) * b + 0.3 * E
    sigma = _sigma_from_residuals(beta_obs, drift_vals, dt)
    nll = _nll_langevin(beta_obs, drift_vals, sigma, dt)
    n_params = 2  # λ, σ
    bic = 2 * nll + n_params * np.log(len(beta_obs) - 1)
    return lam, sigma, nll, bic


def fit_neural_field(beta_obs, E_seq=E_SEQ, dt=DT):
    """
    OLS fit of [α, w] in drift = -α·β + w·tanh(β) + E.
    Rearrangement: Δβ/dt - E = -α·β + w·tanh(β)  → normal equations.
    """
    b = beta_obs[:-1]
    E = E_seq[:-1]
    y = (beta_obs[1:] - b) / dt - E
    X = np.column_stack([-b, np.tanh(b)])
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, w = float(theta[0]), float(theta[1])
    alpha = max(alpha, 0.01)
    drift_vals = -alpha * b + w * np.tanh(b) + E
    sigma = _sigma_from_residuals(beta_obs, drift_vals, dt)
    nll = _nll_langevin(beta_obs, drift_vals, sigma, dt)
    n_params = 3  # α, w, σ
    bic = 2 * nll + n_params * np.log(len(beta_obs) - 1)
    return alpha, w, sigma, nll, bic


def _hmm_nll(obs, mu_hi, mu_lo, sigma_obs, p_high=0.95, p_low=0.95):
    """Forward-algorithm NLL for 2-state Gaussian HMM."""
    from scipy.stats import norm
    n = len(obs)
    log_B = np.column_stack([
        norm.logpdf(obs, mu_hi, sigma_obs),
        norm.logpdf(obs, mu_lo, sigma_obs),
    ])
    A = np.array([[p_high, 1 - p_high],
                  [1 - p_low, p_low]])
    log_A = np.log(A + 1e-300)

    log_alpha = np.empty((n, 2))
    log_alpha[0] = np.log(0.5) + log_B[0]
    for t in range(1, n):
        for j in range(2):
            log_alpha[t, j] = (
                np.logaddexp(log_alpha[t-1, 0] + log_A[0, j],
                             log_alpha[t-1, 1] + log_A[1, j])
                + log_B[t, j]
            )
    return -float(np.logaddexp(log_alpha[-1, 0], log_alpha[-1, 1]))


def fit_hmm(obs):
    """
    Fit 2-state HMM via simple grid search over (mu_hi, mu_lo, sigma).
    Returns NLL and BIC.
    """
    mu_hi_grid = np.percentile(obs, [75, 85])
    mu_lo_grid = np.percentile(obs, [15, 25])
    sigma_grid = [np.std(obs) * f for f in [0.5, 0.8, 1.0]]
    best_nll = np.inf
    for mu_hi in mu_hi_grid:
        for mu_lo in mu_lo_grid:
            if mu_hi <= mu_lo:
                continue
            for sigma_obs in sigma_grid:
                nll = _hmm_nll(obs, mu_hi, mu_lo, sigma_obs)
                if nll < best_nll:
                    best_nll = nll
    n_params = 5  # mu_hi, mu_lo, sigma, p_high, p_low
    bic = 2 * best_nll + n_params * np.log(len(obs))
    return best_nll, bic


# ══════════════════════════════════════════════════════
# 3. PART 1: Parameter Recovery
# ══════════════════════════════════════════════════════

def run_parameter_recovery(n_subjects=30, n_trials=N_TRIALS,
                            n_reps=50, dt=DT, seed=0):
    """
    Simulate TGC trajectories with known (Ω, σ), recover via OLS,
    report Pearson r. Gate passes if r(Ω) ≥ 0.80 AND r(σ) ≥ 0.80.
    """
    rng = np.random.default_rng(seed)
    E_seq = np.linspace(-0.5, 2.0, n_trials)

    true_omega, true_sigma = [], []
    rec_omega,  rec_sigma  = [], []

    print(f"  Reps={n_reps}, Subjects={n_subjects}, Trials={n_trials}")

    for rep in range(n_reps):
        omegas = rng.uniform(0.5, 3.0, n_subjects)
        sigmas = rng.uniform(0.05, 0.45, n_subjects)
        for s in range(n_subjects):
            traj = sim_tgc(omegas[s], sigmas[s], E_seq, dt, rng=rng)
            omega_fit, sigma_fit, _, _ = fit_tgc(traj, E_seq, dt)
            true_omega.append(omegas[s])
            true_sigma.append(sigmas[s])
            rec_omega.append(omega_fit)
            rec_sigma.append(sigma_fit)
        if (rep + 1) % 10 == 0:
            print(f"    rep {rep + 1}/{n_reps} ✓")

    r_omega, _ = pearsonr(true_omega, rec_omega)
    r_sigma,  _ = pearsonr(true_sigma,  rec_sigma)

    return (np.array(true_omega), np.array(rec_omega),
            np.array(true_sigma),  np.array(rec_sigma),
            r_omega, r_sigma)


# ══════════════════════════════════════════════════════
# 4. PART 2: Model Discrimination (Confusion Matrix)
# ══════════════════════════════════════════════════════

MODEL_NAMES = ['TGC', 'DDM/OU', 'Hopf', 'NeuralField', 'HMM']

def _classify_one(obs, E_seq=E_SEQ):
    """Return index of model with lowest BIC."""
    _, _, _, bic_tgc = fit_tgc(obs, E_seq)
    _, _, _, bic_ou  = fit_ou(obs, E_seq)
    _, _, _, bic_hopf = fit_hopf(obs, E_seq)
    _, _, _, _, bic_nf = fit_neural_field(obs, E_seq)
    _, bic_hmm = fit_hmm(obs)
    bics = [bic_tgc, bic_ou, bic_hopf, bic_nf, bic_hmm]
    return int(np.argmin(bics))


def run_model_discrimination(n_per_model=60, n_trials=N_TRIALS, dt=DT, seed=1):
    """
    Generate n_per_model datasets from each of the 5 models,
    classify each by BIC, return 5×5 confusion matrix.
    Gate passes if every diagonal entry ≥ 0.80.
    """
    rng = np.random.default_rng(seed)
    E_seq = np.linspace(-0.5, 2.0, n_trials)
    n_models = len(MODEL_NAMES)
    conf = np.zeros((n_models, n_models), dtype=int)

    print(f"  {n_per_model} datasets × {n_models} models = "
          f"{n_per_model * n_models} classifications")

    for true_idx in range(n_models):
        name = MODEL_NAMES[true_idx]
        correct = 0
        for k in range(n_per_model):
            # Generate trajectory from the true model
            if true_idx == 0:     # TGC
                omega = rng.uniform(0.8, 2.5)
                sigma = rng.uniform(0.1, 0.35)
                obs = sim_tgc(omega, sigma, E_seq, dt, rng)
            elif true_idx == 1:   # DDM / OU
                kappa = rng.uniform(0.5, 2.0)
                sigma = rng.uniform(0.1, 0.35)
                obs = sim_ou(kappa, sigma, E_seq, dt, rng)
            elif true_idx == 2:   # Hopf
                lam = rng.uniform(0.5, 2.0)
                sigma = rng.uniform(0.1, 0.35)
                obs = sim_hopf(lam, sigma, E_seq, dt, rng)
            elif true_idx == 3:   # Neural-field
                alpha = rng.uniform(1.5, 3.5)
                w     = rng.uniform(0.5, 1.2)
                sigma = rng.uniform(0.1, 0.35)
                obs = sim_neural_field(alpha, w, sigma, E_seq, dt, rng)
            else:                  # HMM
                obs = sim_hmm(
                    p_high=rng.uniform(0.88, 0.97),
                    p_low=rng.uniform(0.88, 0.97),
                    mu_hi=rng.uniform(0.8, 1.5),
                    mu_lo=rng.uniform(-1.5, -0.5),
                    sigma_obs=rng.uniform(0.15, 0.35),
                    n=n_trials, rng=rng
                )

            pred_idx = _classify_one(obs, E_seq)
            conf[true_idx, pred_idx] += 1
            if pred_idx == true_idx:
                correct += 1

        acc = correct / n_per_model
        print(f"    {name:12s}: accuracy = {acc:.0%} "
              f"({'PASS' if acc >= 0.80 else 'FAIL'})")

    return conf


# ══════════════════════════════════════════════════════
# 5. Figures
# ══════════════════════════════════════════════════════

def plot_parameter_recovery(true_omega, rec_omega, true_sigma, rec_sigma,
                             r_omega, r_sigma):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("TGC Identifiability Gate — Part 1: Parameter Recovery\n"
                 "(n=30 subjects × 600 trials × 50 reps)",
                 fontsize=12, fontweight='bold')

    for ax, (true, rec, name, r, color) in zip(axes, [
        (true_omega, rec_omega, 'Ω (Cortical Stability)', r_omega, '#1E88E5'),
        (true_sigma, rec_sigma, 'σ (Noise Amplitude)',    r_sigma, '#E53935'),
    ]):
        ax.scatter(true, rec, alpha=0.3, s=8, color=color)
        lo, hi = min(true.min(), rec.min()), max(true.max(), rec.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.6, label='Identity')
        ax.set_xlabel(f'True {name}', fontsize=10)
        ax.set_ylabel(f'Recovered {name}', fontsize=10)
        gate = 'PASS' if r >= 0.80 else 'FAIL'
        ax.set_title(f'r = {r:.3f}  [{gate}]', fontsize=11,
                     color='green' if r >= 0.80 else 'red')
        ax.annotate(f'threshold r ≥ 0.80', xy=(0.05, 0.92),
                    xycoords='axes fraction', fontsize=9, color='gray')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'identifiability_parameter_recovery.pdf')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")


def plot_confusion_matrix(conf):
    n = len(MODEL_NAMES)
    norm_conf = conf / conf.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm_conf, vmin=0, vmax=1, cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(MODEL_NAMES, fontsize=9)
    ax.set_yticklabels(MODEL_NAMES, fontsize=9)
    ax.set_xlabel('Predicted model', fontsize=10)
    ax.set_ylabel('True model', fontsize=10)

    for i in range(n):
        for j in range(n):
            val = norm_conf[i, j]
            color = 'white' if val > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=9, color=color,
                    fontweight='bold' if i == j else 'normal')

    diag = np.diag(norm_conf)
    gate = 'PASS' if diag.min() >= 0.80 else 'FAIL'
    ax.set_title(
        f"TGC Identifiability Gate — Part 2: Model Discrimination\n"
        f"Diagonal min = {diag.min():.2f}  [{gate}]  (threshold ≥ 0.80)",
        fontsize=11, fontweight='bold',
        color='green' if diag.min() >= 0.80 else 'red'
    )

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'identifiability_confusion_matrix.pdf')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════
# 6. Main
# ══════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("TGC Model — Identifiability Gate")
    print("Harada (2026), Section 4")
    print("=" * 60)

    # ── Part 1: Parameter Recovery ─────────────────
    print("\n[PART 1] Parameter Recovery")
    print("-" * 40)
    (true_omega, rec_omega,
     true_sigma, rec_sigma,
     r_omega, r_sigma) = run_parameter_recovery(
         n_subjects=30, n_trials=N_TRIALS, n_reps=50
    )

    gate1 = r_omega >= 0.80 and r_sigma >= 0.80
    print(f"\n  r(Ω) = {r_omega:.3f}  {'PASS' if r_omega >= 0.80 else 'FAIL'}")
    print(f"  r(σ) = {r_sigma:.3f}  {'PASS' if r_sigma >= 0.80 else 'FAIL'}")
    print(f"  Parameter recovery gate: {'PASSED ✓' if gate1 else 'FAILED ✗'}")

    plot_parameter_recovery(true_omega, rec_omega, true_sigma, rec_sigma,
                            r_omega, r_sigma)

    # ── Part 2: Model Discrimination ───────────────
    print("\n[PART 2] Model Discrimination")
    print("-" * 40)
    conf = run_model_discrimination(n_per_model=60, n_trials=N_TRIALS)

    diag = np.diag(conf / conf.sum(axis=1, keepdims=True))
    gate2 = diag.min() >= 0.80
    print(f"\n  Diagonal: {' '.join(f'{v:.2f}' for v in diag)}")
    print(f"  Min diagonal = {diag.min():.2f}  "
          f"({'PASS' if gate2 else 'FAIL'}, threshold ≥ 0.80)")
    print(f"  Model discrimination gate: {'PASSED ✓' if gate2 else 'FAILED ✗'}")

    plot_confusion_matrix(conf)

    # ── Overall Gate ────────────────────────────────
    print("\n" + "=" * 60)
    overall = gate1 and gate2
    print(f"IDENTIFIABILITY GATE: {'PASSED ✓' if overall else 'FAILED ✗'}")
    if not overall:
        print("  → Do NOT proceed to confirmatory data collection.")
        print("  → Revise measurement design and re-run this check.")
    else:
        print("  → Safe to proceed to pre-registered empirical study.")
    print("=" * 60)
