# 🚦 Smart Traffic RL — Reinforcement Learning Traffic Simulator

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Vanilla JS](https://img.shields.io/badge/JavaScript-ES6+-F7DF1E?logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![HTML5 & CSS3](https://img.shields.io/badge/Frontend-HTML5%20%2F%20CSS3-E34F26?logo=html5&logoColor=white)](index.html)
[![Chart.js](https://img.shields.io/badge/Charts-Chart.js%20v4-FF6384?logo=chartdotjs&logoColor=white)](https://www.chartjs.org/)

An interactive, frontend-only reinforcement learning simulation environment for adaptive traffic signal control. Compare **Q-Learning**, **SARSA**, and **Model-Based Reinforcement Learning (Dyna-Q)** in real-time on a simulated 4-way road intersection.

---

## 🌟 Key Highlights

- **Pure Frontend Execution:** Runs directly in any modern browser without requiring backends, Python runtimes, or external API keys.
- **Side-by-Side Multi-Agent Comparison:** Run **Model-Based RL**, **Q-Learning**, and **SARSA** in parallel synchronized environments with identical vehicle inflow distributions.
- **Deep RL Policy Brain Inspector:** Live telemetry displaying the current environment state $s_t$, action-value estimates $Q(s_t, \text{KEEP})$ vs $Q(s_t, \text{SWITCH})$, exploration rate $\epsilon$, and Bellman Temporal Difference error $\delta$.
- **Physics & Kinematics Engine:** 2D Canvas simulation featuring vehicle acceleration, safe car-following headway distance, dynamic braking lights, stop lines, crosswalks, and 3-aspect traffic signals with lighting glow.
- **Real-Time Telemetry & Charts:** Multi-line live charts plotting **Average Wait Time**, **Cumulative Rewards**, and **Congestion Index**.
- **Interactive Hyperparameter Tuning:** Adjust learning rate $\alpha$, discount factor $\gamma$, exploration $\epsilon$, Dyna-Q planning steps $N$, inflow rates, and reward weights on the fly.

---

## 🧠 Reinforcement Learning Formulation

The intersection is modeled as a discrete **Markov Decision Process (MDP)**:

### 1. State Space ($S$)
At each decision step $t$, the state is represented as:
$$s_t = \langle \text{Phase}, \text{Bucket}(Q_{\text{NS}}), \text{Bucket}(Q_{\text{EW}}), \text{Bucket}(T_{\text{elapsed}}) \rangle$$
- **Phase:** Active green direction (`0: North-South`, `1: East-West`).
- **$Q_{\text{NS}}, Q_{\text{EW}}$:** Discretized vehicle queue lengths waiting before the stop lines.
- **$T_{\text{elapsed}}$:** Time elapsed since the last signal phase change.

### 2. Action Space ($A$)
- **$a = 0$ (`KEEP_PHASE`):** Remain on the current green signal.
- **$a = 1$ (`SWITCH_PHASE`):** Trigger yellow clearance interval ($2.0\text{s}$) and transition green to the opposing approach.

### 3. Reward Function ($R$)
$$R = - w_{\text{wait}} \cdot (Q_{\text{NS}} + Q_{\text{EW}}) - 0.2 \cdot (\bar{W}^2) - P_{\text{switch}} \cdot \mathbb{I}(a=1) + B_{\text{thru}} \cdot C_{\text{cleared}}$$
- **Delay Penalty:** Penalizes waiting vehicles and quadratic queuing delays.
- **Oscillation Penalty ($P_{\text{switch}}$):** Discourages erratic, rapid signal switching.
- **Throughput Bonus ($B_{\text{thru}}$):** Rewards vehicles that successfully clear the intersection.

---

## 🔬 Algorithms Implemented

| Algorithm | Type | Target Update Equation | Key Characteristic |
| :--- | :--- | :--- | :--- |
| **Q-Learning** | Off-Policy TD | $Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$ | Aggressive; optimistically pursues maximum downstream reward. |
| **SARSA** | On-Policy TD | $Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$ | Cautious; accounts for actions taken under exploration policy. |
| **Model-Based RL (Dyna-Q)** | Model-Based Planning | Learns $T(s, a) \to s'$ and $R(s, a) \to r$ + $N$ simulated planning updates | Highly sample-efficient; converges rapidly with minimal gridlock. |
| **Fixed Timer** | Non-Adaptive Baseline | Pre-timed cycle switching after fixed green duration | Standard benchmark to evaluate RL improvements. |

---

## 📁 Repository Structure

```
├── index.html          # Semantic dashboard layout, canvas views, and telemetry modals
├── style.css           # Modern dark navy UI design system with glassmorphism & glows
├── rl_agents.js        # Q-Learning, SARSA, Model-Based (Dyna-Q), and Baseline implementations
├── traffic_sim.js      # 2D Intersection kinematics, queue physics, and canvas renderer
├── app.js              # Simulation lifecycle, Chart.js telemetry, and parameter binding
├── LICENSE             # MIT License
└── README.md           # Documentation and theoretical breakdown
```

---

## 🚀 Quick Start / Local Setup

No build step or Node package installation required!

### Option 1: Direct Browser
Simply double-click [`index.html`](index.html) or open it with any modern web browser (Chrome, Edge, Firefox, Safari).

### Option 2: Local HTTP Server (Recommended)
Using Python:
```bash
python -m http.server 8080
```
Then navigate to `http://localhost:8080` in your browser.

Using Node.js:
```bash
npx serve .
```

---

## 🎮 How to Use the Simulator

1. **Simulation Controls:**
   - Click **Start Simulation** / **Pause Simulation** to toggle real-time execution.
   - Use **Step** for single-frame debugging.
   - Adjust **Speed** from `0.5x` up to `15x Max Turbo` for rapid policy training.
2. **Switching Modes:**
   - **Single Agent Focus:** View the full-scale intersection, live $Q$-value action comparison bars, and state representation.
   - **Side-by-Side Compare (3x):** Watch Model-Based RL, Q-Learning, and SARSA run simultaneously side by side.
3. **Flow Presets:**
   - Test algorithms against `Balanced 50/50 Flow`, `NS Rush Hour`, `EW Rush Hour`, and `Random Spikes & Gridlocks`.
4. **Tuning Parameters:**
   - Open the **Parameters** panel to modify vehicle inflow rates, learning rate $\alpha$, discount factor $\gamma$, exploration $\epsilon$, planning steps $N$, and reward penalty weights.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
