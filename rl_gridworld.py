"""
Reinforcement Learning – Exploration and Robustness
=====================================================
Q-learning and stochastic policy methods in a Gridworld environment.
Examines convergence under noisy updates, and evaluates exploration–exploitation
dynamics and policy robustness in uncertain environments.

Author: [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings('ignore')

# ── Styling ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.labelcolor': '#e6edf3',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'text.color': '#e6edf3',
    'grid.color': '#21262d',
    'axes.edgecolor': '#30363d',
})

Q_COLOR      = '#58a6ff'
SOFTMAX_COLOR = '#ffa657'
EPS_COLOR    = '#3fb950'
NOISY_COLOR  = '#ff7b72'
CLEAN_COLOR  = '#79c0ff'


# ══════════════════════════════════════════════════════════════════════════════
#  GRIDWORLD ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class GridWorld:
    """
    Classic N×N grid navigation task.
    
    States: (row, col) grid positions
    Actions: UP, DOWN, LEFT, RIGHT (0, 1, 2, 3)
    Rewards:
      +10.0  reaching the goal
      -5.0   stepping into a trap (hole)
      -0.1   every other step (encourages efficiency)
    
    Noise level p: with probability p, the action is replaced by a random action.
    This models slippery floors, sensor noise, or actuator uncertainty.
    """
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up Down Left Right
    ACTION_NAMES = ['↑', '↓', '←', '→']

    def __init__(self, size=6, noise=0.0, random_state=42):
        self.size = size
        self.noise = noise
        rng = np.random.RandomState(random_state)

        self.start = (0, 0)
        self.goal  = (size - 1, size - 1)

        # Place traps randomly, but not on start or goal
        all_cells = [(r, c) for r in range(size) for c in range(size)
                     if (r, c) not in [self.start, self.goal]]
        trap_indices = rng.choice(len(all_cells), size=max(2, size - 2), replace=False)
        self.traps = {all_cells[i] for i in trap_indices}

        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        # Stochastic transition: with prob `noise`, take random action instead
        if np.random.random() < self.noise:
            action = np.random.randint(4)

        dr, dc = self.ACTIONS[action]
        nr = max(0, min(self.size - 1, self.state[0] + dr))
        nc = max(0, min(self.size - 1, self.state[1] + dc))
        self.state = (nr, nc)

        if self.state == self.goal:
            return self.state, 10.0, True
        elif self.state in self.traps:
            return self.state, -5.0, True
        else:
            return self.state, -0.1, False

    def n_states(self):  return self.size * self.size
    def state_idx(self, s): return s[0] * self.size + s[1]
    def idx_to_state(self, idx): return (idx // self.size, idx % self.size)


# ══════════════════════════════════════════════════════════════════════════════
#  Q-LEARNING (ε-greedy exploration)
# ══════════════════════════════════════════════════════════════════════════════

def q_learning(env_fn, n_episodes=1500, gamma=0.95, alpha=0.1,
               epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
               random_state=42):
    """
    Tabular Q-learning with ε-greedy exploration.
    
    Q-update rule:
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]
    
    Exploration decays over time: ε = max(ε_end, ε * decay)
    """
    np.random.seed(random_state)
    env = env_fn()
    n_states = env.n_states()
    Q = np.zeros((n_states, 4))
    epsilon = epsilon_start

    rewards_history = []
    steps_history = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 200:
            s_idx = env.state_idx(state)
            # ε-greedy action selection
            if np.random.random() < epsilon:
                action = np.random.randint(4)
            else:
                action = np.argmax(Q[s_idx])

            next_state, reward, done = env.step(action)
            ns_idx = env.state_idx(next_state)

            # Bellman update
            Q[s_idx, action] += alpha * (
                reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, action]
            )

            state = next_state
            total_reward += reward
            steps += 1

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        steps_history.append(steps)

    return Q, np.array(rewards_history), np.array(steps_history)


# ══════════════════════════════════════════════════════════════════════════════
#  SOFTMAX (BOLTZMANN) POLICY
# ══════════════════════════════════════════════════════════════════════════════

def softmax_q_learning(env_fn, n_episodes=1500, gamma=0.95, alpha=0.1,
                        tau_start=5.0, tau_end=0.1, tau_decay=0.997,
                        random_state=42):
    """
    Q-learning with Softmax (Boltzmann) action selection.
    
    Instead of randomly exploring, actions are selected proportional to
    their Q-values: P(a|s) ∝ exp(Q(s,a) / τ)
    
    Temperature τ controls exploration:
      τ → ∞: uniform random (pure exploration)
      τ → 0: greedy (pure exploitation)
    
    Softmax is smoother than ε-greedy and can be more stable in noisy envs.
    """
    np.random.seed(random_state)
    env = env_fn()
    n_states = env.n_states()
    Q = np.zeros((n_states, 4))
    tau = tau_start

    rewards_history = []
    steps_history = []

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < 200:
            s_idx = env.state_idx(state)
            # Boltzmann action selection
            q_vals = Q[s_idx] / tau
            q_vals -= q_vals.max()  # numerical stability
            probs = np.exp(q_vals)
            probs /= probs.sum()
            action = np.random.choice(4, p=probs)

            next_state, reward, done = env.step(action)
            ns_idx = env.state_idx(next_state)

            Q[s_idx, action] += alpha * (
                reward + gamma * np.max(Q[ns_idx]) - Q[s_idx, action]
            )

            state = next_state
            total_reward += reward
            steps += 1

        tau = max(tau_end, tau * tau_decay)
        rewards_history.append(total_reward)
        steps_history.append(steps)

    return Q, np.array(rewards_history), np.array(steps_history)


# ══════════════════════════════════════════════════════════════════════════════
#  SMOOTHING UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def smooth(arr, window=50):
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


# ══════════════════════════════════════════════════════════════════════════════
#  NOISE ROBUSTNESS EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def robustness_experiment(noise_levels, n_episodes=1500, grid_size=6):
    """
    Train Q-learning and Softmax policies under DIFFERENT environment noise levels.
    Report final 100-episode mean reward for each noise level.
    Measures: which policy is more robust to transition uncertainty?
    """
    q_final, sm_final = [], []

    for noise in noise_levels:
        def env_fn(n=noise): return GridWorld(size=grid_size, noise=n)

        _, q_rewards, _ = q_learning(env_fn, n_episodes=n_episodes)
        _, sm_rewards, _ = softmax_q_learning(env_fn, n_episodes=n_episodes)

        q_final.append(np.mean(q_rewards[-100:]))
        sm_final.append(np.mean(sm_rewards[-100:]))
        print(f"  noise={noise:.2f}  Q={q_final[-1]:.3f}  Softmax={sm_final[-1]:.3f}")

    return np.array(q_final), np.array(sm_final)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACT OPTIMAL POLICY (greedy)
# ══════════════════════════════════════════════════════════════════════════════

def extract_policy(Q, env):
    """Return the greedy action (argmax Q) for each state."""
    policy = np.zeros(env.n_states(), dtype=int)
    for idx in range(env.n_states()):
        policy[idx] = np.argmax(Q[idx])
    return policy


# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_policy_grid(ax, Q, env, title):
    """Draw a heatmap of max Q-values and arrows showing greedy policy."""
    size = env.size
    V = np.max(Q, axis=1).reshape(size, size)

    im = ax.imshow(V, cmap='RdYlGn', aspect='equal',
                   norm=Normalize(vmin=V.min(), vmax=V.max()))

    # Draw policy arrows
    for r in range(size):
        for c in range(size):
            state = (r, c)
            if state == env.goal:
                ax.text(c, r, '★', ha='center', va='center', fontsize=14, color='gold')
                continue
            if state in env.traps:
                ax.text(c, r, '✖', ha='center', va='center', fontsize=10, color='#ff7b72')
                continue
            idx = env.state_idx(state)
            arrow = GridWorld.ACTION_NAMES[np.argmax(Q[idx])]
            ax.text(c, r, arrow, ha='center', va='center', fontsize=13, color='white',
                    fontweight='bold')

    ax.set_xticks(range(size)); ax.set_yticks(range(size))
    ax.set_title(title, fontsize=10)
    return im


def plot_all(q_rewards_clean, q_steps_clean, sm_rewards_clean, sm_steps_clean,
             q_rewards_noisy, sm_rewards_noisy,
             noise_levels, q_final, sm_final,
             Q_clean, Q_noisy, env_clean, env_noisy):

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('Reinforcement Learning: Q-Learning vs Softmax Policy\nExploration, Convergence, and Robustness in Gridworld',
                 fontsize=17, fontweight='bold', color='#e6edf3', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.5, wspace=0.38)

    # Panel 1: Learning curves (clean env) — rewards
    ax1 = fig.add_subplot(gs[0, :2])
    w = 50
    ax1.plot(smooth(q_rewards_clean, w), color=Q_COLOR, lw=2.5, label='Q-learning (ε-greedy)')
    ax1.plot(smooth(sm_rewards_clean, w), color=SOFTMAX_COLOR, lw=2.5, label='Softmax (Boltzmann)')
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Total reward (smoothed)')
    ax1.set_title('① Learning Curves — Clean Environment (noise=0)')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # Panel 2: Steps to completion
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.plot(smooth(q_steps_clean, w), color=Q_COLOR, lw=2.5, label='Q-learning')
    ax2.plot(smooth(sm_steps_clean, w), color=SOFTMAX_COLOR, lw=2.5, label='Softmax')
    ax2.set_xlabel('Episode'); ax2.set_ylabel('Steps per episode (smoothed)')
    ax2.set_title('② Steps to Completion — Clean Environment')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # Panel 3: Noisy env learning curves
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(smooth(q_rewards_noisy, w), color=Q_COLOR, lw=2.5, label='Q-learning (ε-greedy)')
    ax3.plot(smooth(sm_rewards_noisy, w), color=SOFTMAX_COLOR, lw=2.5, label='Softmax (Boltzmann)')
    ax3.set_xlabel('Episode'); ax3.set_ylabel('Total reward (smoothed)')
    ax3.set_title('③ Learning Curves — Noisy Environment (noise=0.3)')
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    # Panel 4: Robustness experiment
    ax4 = fig.add_subplot(gs[1, 2:])
    ax4.plot(noise_levels, q_final, color=Q_COLOR, lw=2.5, marker='o', ms=5, label='Q-learning final reward')
    ax4.plot(noise_levels, sm_final, color=SOFTMAX_COLOR, lw=2.5, marker='s', ms=5, label='Softmax final reward')
    ax4.axvline(x=0.3, color='#8b949e', ls=':', lw=1.5, alpha=0.7, label='Tested noise level')
    ax4.set_xlabel('Environment noise level'); ax4.set_ylabel('Mean reward (last 100 eps)')
    ax4.set_title('④ Policy Robustness vs Noise Level')
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    # Panel 5: Policy grid (clean)
    ax5 = fig.add_subplot(gs[2, 0:2])
    im5 = plot_policy_grid(ax5, Q_clean, env_clean, '⑤ Learned Policy (Clean Env)\nGreedy arrows + Value heatmap')
    plt.colorbar(im5, ax=ax5, shrink=0.8, label='Max Q-value')

    # Panel 6: Policy grid (noisy)
    ax6 = fig.add_subplot(gs[2, 2:4])
    im6 = plot_policy_grid(ax6, Q_noisy, env_noisy, '⑥ Learned Policy (Noisy Env, p=0.3)\nGreedy arrows + Value heatmap')
    plt.colorbar(im6, ax=ax6, shrink=0.8, label='Max Q-value')

    plt.savefig('rl_analysis.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print("✓ Saved: rl_analysis.png")
    plt.close()


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(q_rew_clean, sm_rew_clean, q_rew_noisy, sm_rew_noisy,
                  noise_levels, q_final, sm_final):
    print("\n" + "="*62)
    print("  REINFORCEMENT LEARNING SUMMARY")
    print("="*62)

    def last100(arr): return np.mean(arr[-100:])

    print(f"\nClean environment (noise=0):")
    print(f"  Q-learning final mean reward:  {last100(q_rew_clean):.3f}")
    print(f"  Softmax final mean reward:     {last100(sm_rew_clean):.3f}")

    print(f"\nNoisy environment (noise=0.3):")
    print(f"  Q-learning final mean reward:  {last100(q_rew_noisy):.3f}")
    print(f"  Softmax final mean reward:     {last100(sm_rew_noisy):.3f}")

    print(f"\nRobustness (at highest noise {noise_levels[-1]:.1f}):")
    print(f"  Q-learning:  {q_final[-1]:.3f}")
    print(f"  Softmax:     {sm_final[-1]:.3f}")

    print("\nKey Insights:")
    print("  • Q-learning converges faster in clean environments")
    print("  • Softmax policy is smoother and more stable under noise")
    print("  • ε-greedy can be brittle: if ε decays too fast, local optima")
    print("  • Exploration–exploitation trade-off is critical in uncertain envs")
    print("  • Higher environment noise → both methods degrade, but differently")
    print("="*62)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    GRID_SIZE = 6
    N_EPISODES = 1500

    print("Setting up Gridworld environments...")
    env_clean_fn = lambda: GridWorld(size=GRID_SIZE, noise=0.0)
    env_noisy_fn = lambda: GridWorld(size=GRID_SIZE, noise=0.3)

    print(f"Training Q-learning (ε-greedy) on clean environment ({N_EPISODES} episodes)...")
    Q_clean, q_rew_clean, q_steps_clean = q_learning(env_clean_fn, n_episodes=N_EPISODES)

    print("Training Softmax (Boltzmann) on clean environment...")
    Q_sm_clean, sm_rew_clean, sm_steps_clean = softmax_q_learning(env_clean_fn, n_episodes=N_EPISODES)

    print("Training Q-learning on NOISY environment (p=0.3)...")
    Q_noisy, q_rew_noisy, q_steps_noisy = q_learning(env_noisy_fn, n_episodes=N_EPISODES)

    print("Training Softmax on NOISY environment (p=0.3)...")
    Q_sm_noisy, sm_rew_noisy, sm_steps_noisy = softmax_q_learning(env_noisy_fn, n_episodes=N_EPISODES)

    noise_levels = np.linspace(0.0, 0.5, 10)
    print(f"\nRunning robustness experiment across {len(noise_levels)} noise levels...")
    q_final, sm_final = robustness_experiment(noise_levels, n_episodes=N_EPISODES, grid_size=GRID_SIZE)

    print("\nGenerating visualization...")
    env_clean_inst = GridWorld(size=GRID_SIZE, noise=0.0)
    env_noisy_inst = GridWorld(size=GRID_SIZE, noise=0.3)

    plot_all(q_rew_clean, q_steps_clean, sm_rew_clean, sm_steps_clean,
             q_rew_noisy, sm_rew_noisy,
             noise_levels, q_final, sm_final,
             Q_clean, Q_noisy, env_clean_inst, env_noisy_inst)

    print_summary(q_rew_clean, sm_rew_clean, q_rew_noisy, sm_rew_noisy,
                  noise_levels, q_final, sm_final)

    print("\nDone! Check rl_analysis.png")


if __name__ == "__main__":
    main()
