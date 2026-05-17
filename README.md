# 🤖 Reinforcement Learning – Exploration and Robustness in Gridworld

> **Tabular reinforcement learning** using Q-learning (ε-greedy) and Softmax (Boltzmann) policies in a stochastic Gridworld. Analyzes convergence under noisy transitions and evaluates exploration–exploitation dynamics in uncertain environments.

---

## 🧠 What This Project Studies

Reinforcement learning agents must learn *which actions lead to high rewards* purely through trial and error — with no labeled training data. This project studies:

1. **Q-learning with ε-greedy** — the classic tabular RL algorithm
2. **Softmax (Boltzmann) policy** — temperature-based probabilistic exploration
3. **Effect of environment noise** on convergence and final policy quality
4. **Exploration–exploitation trade-off** under uncertainty

---

## 🗺️ Environment: Gridworld

A 6×6 grid where an agent must navigate from the top-left to the bottom-right corner.

```
S . . . . .
. . ✖ . . .
. . . . ✖ .
. ✖ . . . .
. . . ✖ . .
. . . . . G
```

| Symbol | Meaning |
|---|---|
| `S` | Start (top-left) |
| `G` | Goal: reward **+10** |
| `✖` | Trap: reward **−5** |
| `.` | Step: reward **−0.1** |

**Stochasticity**: With probability `p` (noise), the agent's chosen action is replaced by a random action. This models actuator noise, slippery floors, or partial observability.

---

## 📐 Algorithms

### Q-Learning (ε-greedy)

$$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s,a) \right]$$

- **Exploration**: with probability ε, take a random action; otherwise be greedy
- ε decays from 1.0 → 0.05 over training (explore → exploit)

### Softmax (Boltzmann) Policy

$$\pi(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}$$

- **Exploration**: actions sampled proportional to their Q-values
- Temperature τ decays from 5.0 → 0.1 (uniform → greedy)
- Smoother than ε-greedy; can be more stable under noise

---

## 🔬 Experiments

### 1. Clean Environment (noise = 0)
- Both algorithms trained for 1500 episodes
- Learning curves smoothed over 50-episode window
- Steps per episode tracked to measure efficiency

### 2. Noisy Environment (noise = 0.3)
- Same training, but 30% of actions are random
- Compares stability of ε-greedy vs Softmax under uncertainty

### 3. Robustness Experiment
- Noise swept from 0.0 to 0.5 in 10 steps
- Final-100-episode mean reward recorded for each method
- Identifies which policy degrades more gracefully

### 4. Learned Policy Visualization
- Value heatmap: color = learned state value (max Q)
- Arrows: greedy action at each state
- Side-by-side: clean vs noisy environment policies

---

## 📈 Key Findings

| Condition | Q-learning | Softmax |
|---|---|---|
| Clean env convergence | ✅ Fast | ✅ Stable |
| Noisy env reward | ⚠️ Degrades | ✅ More robust |
| Policy smoothness | ⚠️ Brittle near ε threshold | ✅ Smooth |
| Exploration efficiency | ✅ Good with decay | ✅ Natural temperature |

- **Softmax is more robust** to environment noise in most settings
- **Q-learning converges faster** when the environment is clean
- **Exploration schedule matters**: aggressive ε-decay → local optima in noisy envs
- Policy grids reveal how noise forces the agent into suboptimal routes

---

## 🚀 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/rl-gridworld
cd rl-gridworld

pip install -r requirements.txt
python rl_gridworld.py
```

Output: `rl_analysis.png` — 6-panel figure with learning curves, policy grids, and robustness results.

---

## 🗂️ Project Structure

```
rl-gridworld/
├── rl_gridworld.py     # Environment + algorithms + experiments
├── requirements.txt
├── rl_analysis.png     # Auto-generated output
└── README.md
```

---

## 📦 Dependencies

```
numpy>=1.24
matplotlib>=3.7
```

No deep learning frameworks required — pure NumPy tabular RL.

---

## 🔗 Related Work

- Watkins & Dayan (1992) — *Q-learning*
- Sutton & Barto (2018) — *Reinforcement Learning: An Introduction* (Chapters 4–6)
- Luce (1959) — *Individual Choice Behavior* (Softmax/Boltzmann selection)
- Mnih et al. (2015) — *Human-level control through deep reinforcement learning* (DQN)
