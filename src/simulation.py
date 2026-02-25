import numpy as np
import matplotlib.pyplot as plt

class TGCAgentV5:
    def __init__(self, omega, name, color):
        """
        Thermostatic Gain Control (TGC) Model 5.0 Agent
        omega (\Omega): Stability Factor (Trait parameter controlling topology)
        """
        self.omega = omega
        self.name = name
        self.color = color
        
        # Initialize latent state x at the stable baseline for E=0
        initial_roots = self._get_stable_roots(E=0)
        self.x = max(initial_roots) if initial_roots else 0.0
        
        self.x_history = []
        self.beta_history = []

    def _get_stable_roots(self, E):
        """
        Solves x^3 - \Omega*x - E = 0 and filters for local minima.
        """
        # Coefficients for x^3 + 0*x^2 - \Omega*x - E = 0
        coeffs = [1.0, 0.0, -self.omega, -E]
        roots = np.roots(coeffs)
        
        # Extract purely real roots
        real_roots = roots[np.isclose(roots.imag, 0)].real
        
        # Filter for local minima: d^2V/dx^2 = 3x^2 - \Omega > 0
        stable_roots = [r for r in real_roots if 3 * r**2 - self.omega > 0]
        return stable_roots

    def update_state(self, E_t):
        """
        Applies the Adiabatic Assumption and Minimum-Distance Selection Rule.
        """
        stable_roots = self._get_stable_roots(E_t)
        
        if len(stable_roots) == 0:
            # Fallback for numerical edge cases
            new_x = self.x 
        else:
            # argmin_x |x - x_{t-1}^*|
            distances = [abs(r - self.x) for r in stable_roots]
            new_x = stable_roots[np.argmin(distances)]
            
        self.x = new_x
        self.x_history.append(self.x)
        
        # Link function to strictly positive decision precision (beta)
        beta_t = np.exp(self.x)
        self.beta_history.append(beta_t)
        
        return beta_t

def run_catastrophe_forcing_protocol():
    """
    Simulates the 'Stress Ramp' to empirically demonstrate Hysteresis (A > 0)
    and catastrophic bifurcations across different topological phenotypes.
    """
    # 1. Define the Agents based on Topological Traits (\Omega)
    agents = [
        # ADHD-like: Low \Omega (Shallow basins, no bistability, noise-driven)
        TGCAgentV5(omega=0.5, name="ADHD-like (Low $\Omega$)", color="red"),
        
        # Neurotypical: Moderate \Omega
        TGCAgentV5(omega=1.5, name="Neurotypical (Mid $\Omega$)", color="blue"),
        
        # ASD-like: High \Omega (Deep hysteresis, hyper-systemizing, sudden meltdowns)
        TGCAgentV5(omega=3.0, name="ASD-like (High $\Omega$)", color="green")
    ]
    
    # 2. Construct the CFP "Stress Ramp" (Input Drive E)
    # Ramping up from -4.0 to 4.0, then ramping back down to -4.0
    E_ascending = np.linspace(-4.0, 4.0, 200)
    E_descending = np.linspace(4.0, -4.0, 200)
    E_sequence = np.concatenate([E_ascending, E_descending])
    
    # 3. Run Simulation
    for agent in agents:
        for E_t in E_sequence:
            agent.update_state(E_t)
            
    # 4. Visualization
    plt.figure(figsize=(12, 7))
    
    for agent in agents:
        # Split history into Ascending and Descending phases for plotting
        beta_asc = agent.beta_history[:len(E_ascending)]
        beta_desc = agent.beta_history[len(E_ascending):]
        
        # Plot ascending path (solid line)
        plt.plot(E_ascending, beta_asc, color=agent.color, linestyle='-', linewidth=2.5, 
                 label=f"{agent.name} - Ascending")
        
        # Plot descending path (dashed line) to reveal Hysteresis Loop area (A > 0)
        plt.plot(E_descending, beta_desc, color=agent.color, linestyle='--', linewidth=2.5, alpha=0.7,
                 label=f"{agent.name} - Descending")
        
        # Mark Theoretical Critical Points if bistable
        if agent.omega > 0:
            E_crit = np.sqrt((4 * agent.omega**3) / 27)
            if E_crit <= 4.0:
                plt.axvline(x=E_crit, color=agent.color, linestyle=':', alpha=0.5)
                plt.axvline(x=-E_crit, color=agent.color, linestyle=':', alpha=0.5)

    plt.title("Figure 1: Catastrophe-Forcing Protocol (CFP)\nDynamical Signature of Structural Hysteresis ($A = \oint \\beta \, dE$)", fontsize=14)
    plt.xlabel("Input Drive ($E_t$)", fontsize=12)
    plt.ylabel("Decision Precision / Gain ($\\beta_t = \exp(x_t^*)$)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save for GitHub README
    plt.savefig("figure1.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    run_catastrophe_forcing_protocol()