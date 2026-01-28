import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class TGCAgent:
    def __init__(self, alpha, tau_init, gamma, theta, beta, decay_rate, name, color, is_static=False):
        self.q = np.array([0.5, 0.5])
        self.tau = tau_init
        self.alpha = alpha
        self.gamma = gamma       # Cooling rate
        self.theta = theta       # Re-heating threshold
        self.beta = beta         # Re-heating amount
        self.decay_rate = decay_rate # Leaky integrator decay
        self.name = name
        self.color = color
        self.is_static = is_static # For baseline agent
        
        self.cum_error = 0.0
        self.entropy_history = []

    def softmax(self, q, tau):
        q_stab = q - np.max(q)
        exp_q = np.exp(q_stab / tau)
        return exp_q / np.sum(exp_q)

    def select_action(self):
        probs = self.softmax(self.q, self.tau)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        self.entropy_history.append(entropy)
        return np.random.choice([0, 1], p=probs)

    def update(self, action, reward):
        # 1. Standard Q-learning update
        prediction_error = reward - self.q[action]
        self.q[action] += self.alpha * prediction_error
        
        if self.is_static:
            return # Static agent does not change tau

        # 2. Thermostatic Gain Control (TGC)
        
        # A. Stabilization (Freezing)
        if reward > 0:
            self.tau = max(0.01, self.tau * self.gamma)

        # B. Destabilization (Melting) with Leaky Integrator
        # 負の予測誤差の絶対値を蓄積するが、同時に時間減衰させる
        if prediction_error < 0:
            self.cum_error = (self.decay_rate * self.cum_error) + abs(prediction_error)
        else:
            # エラーがない時も減衰させる（忘却）
            self.cum_error = self.decay_rate * self.cum_error
        
        # Threshold check
        if self.cum_error > self.theta:
            self.tau += self.beta
            self.cum_error = 0.0 # Reset pressure

def run_simulation_v2():
    n_trials = 100
    reversal_trial = 50
    
    agents = [
        # Baseline: Static Agent (温度固定)
        TGCAgent(alpha=0.3, tau_init=0.5, gamma=1.0, theta=999, beta=0, decay_rate=0, name="Static (Baseline)", color="gray", is_static=True),
        # Healthy: 適切な冷却と、漏れのある積分器
        TGCAgent(alpha=0.3, tau_init=1.0, gamma=0.85, theta=2.5, beta=1.5, decay_rate=0.9, name="Healthy (TGC)", color="blue"),
        # ADHD-like: 冷却不全
        TGCAgent(alpha=0.3, tau_init=1.0, gamma=0.99, theta=2.5, beta=1.5, decay_rate=0.9, name="ADHD-like", color="red"),
        # ASD-like: 加熱不全 (閾値無限大) + 強力な冷却(gamma=0.6)
        TGCAgent(alpha=0.3, tau_init=1.0, gamma=0.6, theta=9999, beta=1.5, decay_rate=0.9, name="ASD-like", color="green")
    ]
    
    probs_phase1 = [0.8, 0.2]
    probs_phase2 = [0.2, 0.8]
    
    plt.figure(figsize=(10, 6))
    
    for agent in agents:
        # Reset for consistency
        np.random.seed(42) 
        for t in range(n_trials):
            current_probs = probs_phase1 if t < reversal_trial else probs_phase2
            action = agent.select_action()
            reward = 1 if np.random.rand() < current_probs[action] else 0
            agent.update(action, reward)
            
        style = '--' if agent.is_static else '-'
        width = 1.5 if agent.is_static else 2.5
        alpha = 0.6 if agent.is_static else 0.9
        plt.plot(agent.entropy_history, label=agent.name, color=agent.color, linestyle=style, linewidth=width, alpha=alpha)

    plt.axvline(x=reversal_trial, color='black', linestyle=':', label='Rule Reversal')
    plt.title("Figure 1: Entropy Dynamics of TGC Model vs Static Baseline")
    plt.xlabel("Trial (t)")
    plt.ylabel("Action Entropy (Uncertainty)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

run_simulation_v2()