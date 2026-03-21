"""
TGC Model — Stochastic Langevin Simulation (Equation 1)
========================================================
Harada (2026): Thermostatic Gain Control Model

Simulates the core TGC dynamics:
    dβ_t = −(β³_t − Ω·β_t − E) dt + σ dW_t

Generates:
  - Figure: Langevin trajectories under load ramp for different Ω values
  - Figure: Noise-induced transition (Overheating scenario)
  - Figure: Bistability demonstration (bimodal stationary distribution)

Usage:
    python simulation/tgc_langevin.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ══════════════════════════════════════════════
# Core TGC Dynamics
# ══════════════════════════════════════════════

def cusp_potential(beta, omega, E):
    """V(β; Ω, E) = β⁴/4 − Ω·β²/2 − E·β"""
    return beta**4 / 4 - omega * beta**2 / 2 - E * beta


def cusp_drift(beta, omega, E):
    """Deterministic drift: −dV/dβ = −(β³ − Ω·β − E)"""
    return -(beta**3 - omega * beta - E)


def langevin_step(beta, omega, E, sigma, dt):
    """Single Euler–Maruyama step for the TGC Langevin equation."""
    drift = cusp_drift(beta, omega, E)
    noise = sigma * np.sqrt(dt) * np.random.randn()
    return beta + drift * dt + noise


def simulate_trajectory(omega, E_sequence, sigma, dt=0.01, beta0=None):
    """
    Simulate a full TGC trajectory under a time-varying E sequence.

    Parameters
    ----------
    omega : float
        Stability factor (splitting parameter).
    E_sequence : array-like
        Time series of input drive values.
    sigma : float
        Noise amplitude.
    dt : float
        Integration timestep.
    beta0 : float or None
        Initial gain state. If None, starts at upper stable root for E=0.

    Returns
    -------
    beta_traj : np.ndarray
        Trajectory of latent gain state β.
    """
    if beta0 is None:
        beta0 = np.sqrt(omega) if omega > 0 else 0.0

    n_steps = len(E_sequence)
    beta_traj = np.zeros(n_steps)
    beta_traj[0] = beta0

    for t in range(1, n_steps):
        beta_traj[t] = langevin_step(
            beta_traj[t - 1], omega, E_sequence[t], sigma, dt
        )
    return beta_traj


# ══════════════════════════════════════════════
# Figure 1: Langevin Trajectories Under Load Ramp
# ══════════════════════════════════════════════

def plot_langevin_trajectories():
    """
    Demonstrates TGC predictions 1–3 (bistability, hysteresis, threshold asymmetry)
    by simulating ascending/descending load ramps for three Ω values.
    """
    dt = 0.005
    sigma = 0.15
    n_ramp = 2000

    E_asc = np.linspace(-2.0, 3.0, n_ramp)
    E_desc = np.linspace(3.0, -2.0, n_ramp)
    E_full = np.concatenate([E_asc, E_desc])

    omegas = [0.5, 1.5, 3.0]
    labels = ["Low Ω (0.5)", "Mid Ω (1.5)", "High Ω (3.0)"]
    colors = ["#E53935", "#1E88E5", "#43A047"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        "TGC Langevin Trajectories Under Load Ramp\n"
        "dβ = −(β³ − Ω·β − E)dt + σdW,  σ = 0.15",
        fontsize=13, fontweight="bold",
    )

    for ax, omega, label, color in zip(axes, omegas, labels, colors):
        # Run 5 stochastic replicates
        for rep in range(5):
            np.random.seed(42 + rep)
            traj = simulate_trajectory(omega, E_full, sigma, dt)
            alpha = 0.8 if rep == 0 else 0.25
            ax.plot(E_full, traj, color=color, alpha=alpha, lw=0.6)

        # Mark theoretical bifurcation points
        E_crit = np.sqrt(4 * omega**3 / 27) if omega > 0 else 0
        if E_crit < 3.0:
            ax.axvline(E_crit, color="gray", ls=":", alpha=0.6, label=f"E_crit = {E_crit:.2f}")
            ax.axvline(-E_crit, color="gray", ls=":", alpha=0.6)

        ax.set_xlabel("Input Drive (E)", fontsize=10)
        ax.set_title(label, fontsize=11, color=color)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Neural Gain (β)", fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/fig_langevin_trajectories.png", dpi=300)
    plt.show()
    print("✅ Saved: figures/fig_langevin_trajectories.png")


# ══════════════════════════════════════════════
# Figure 2: Overheating (Noise-Induced Transition)
# ══════════════════════════════════════════════

def plot_overheating_demo():
    """
    Demonstrates Prediction 4: noise-induced transition while E < E_crit.
    σ increases over time (simulating tonic LC-NE escalation).
    """
    dt = 0.005
    omega = 2.0
    E_constant = 0.8  # Below E_crit
    E_crit = np.sqrt(4 * omega**3 / 27)

    n_steps = 6000
    sigma_base = 0.08
    sigma_escalation = np.concatenate([
        np.full(2000, sigma_base),
        np.linspace(sigma_base, 0.5, 2000),
        np.full(2000, 0.5),
    ])

    np.random.seed(123)
    beta = np.sqrt(omega)
    traj = np.zeros(n_steps)
    for t in range(n_steps):
        traj[t] = beta
        beta = langevin_step(beta, omega, E_constant, sigma_escalation[t], dt)

    time = np.arange(n_steps) * dt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(
        f"Overheating Mechanism: Noise-Induced Transition (E = {E_constant} < E_crit = {E_crit:.2f})",
        fontsize=13, fontweight="bold",
    )

    ax1.plot(time, traj, color="#1E88E5", lw=0.8)
    ax1.set_ylabel("Neural Gain (β)", fontsize=10)
    ax1.axhline(0, color="gray", ls="--", alpha=0.3)
    ax1.grid(True, alpha=0.2)

    ax2.plot(time, sigma_escalation, color="#E53935", lw=1.5)
    ax2.set_ylabel("σ (noise)", fontsize=10)
    ax2.set_xlabel("Time", fontsize=10)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("figures/fig_overheating_demo.png", dpi=300)
    plt.show()
    print("✅ Saved: figures/fig_overheating_demo.png")


# ══════════════════════════════════════════════
# Figure 3: Stationary Distribution (Bistability)
# ══════════════════════════════════════════════

def plot_stationary_distribution():
    """
    Long-run simulation at fixed (Ω, E) in the bistable regime.
    Shows bimodal histogram of β (Prediction 1).
    """
    dt = 0.005
    omega = 2.5
    E = 0.3
    sigma = 0.25
    n_steps = 200000

    np.random.seed(77)
    beta = 1.0
    samples = np.zeros(n_steps)
    for t in range(n_steps):
        samples[t] = beta
        beta = langevin_step(beta, omega, E, sigma, dt)

    # Discard burn-in
    samples = samples[10000:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Bistability: Stationary Distribution at Ω={omega}, E={E}, σ={sigma}",
        fontsize=13, fontweight="bold",
    )

    # Histogram
    ax1.hist(samples, bins=120, density=True, color="#1E88E5", alpha=0.7, edgecolor="none")
    ax1.set_xlabel("Neural Gain (β)", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title("Empirical Stationary Distribution", fontsize=11)
    ax1.grid(True, alpha=0.2)

    # Potential landscape
    beta_grid = np.linspace(-2.5, 2.5, 500)
    V = cusp_potential(beta_grid, omega, E)
    ax2.plot(beta_grid, V, color="#E53935", lw=2.5)
    ax2.set_xlabel("Neural Gain (β)", fontsize=10)
    ax2.set_ylabel("V(β; Ω, E)", fontsize=10)
    ax2.set_title("Cusp Potential", fontsize=11)
    ax2.set_ylim(V.min() - 0.5, V.min() + 5)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig("figures/fig_stationary_bistability.png", dpi=300)
    plt.show()
    print("✅ Saved: figures/fig_stationary_bistability.png")


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs("figures", exist_ok=True)

    print("=" * 60)
    print("TGC Model — Stochastic Langevin Simulation")
    print("Harada (2026), Equation 1")
    print("=" * 60)

    plot_langevin_trajectories()
    plot_overheating_demo()
    plot_stationary_distribution()

    print("\n✅ All simulations complete.")
