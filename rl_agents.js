/**
 * Smart Traffic RL — Reinforcement Learning Agents
 * Contains authentic implementations of:
 * 1. Q-Learning (Off-policy TD)
 * 2. SARSA (On-policy TD)
 * 3. Model-Based RL (Dyna-Q with environment model planning)
 * 4. Fixed-Timer Baseline
 */

// Helper to discretize continuous values into buckets for tabular RL
function discretize(value, thresholds) {
  for (let i = 0; i < thresholds.length; i++) {
    if (value < thresholds[i]) return i;
  }
  return thresholds.length;
}

class BaseRLAgent {
  constructor(name, id, color) {
    this.name = name;
    this.id = id;
    this.color = color;

    // Hyperparameters
    this.alpha = 0.12;      // Learning rate
    this.gamma = 0.92;      // Discount factor
    this.epsilon = 0.15;    // Exploration probability
    this.minEpsilon = 0.02;
    this.epsilonDecay = 0.9995;

    // Action space: 0 = KEEP_PHASE, 1 = SWITCH_PHASE
    this.actions = [0, 1];

    // Q-Table: Map of stateString -> [Q(s, 0), Q(s, 1)]
    this.qTable = new Map();

    // Diagnostics
    this.lastTDError = 0;
    this.totalSteps = 0;
    this.totalReward = 0;
    this.decisionsCount = 0;
    this.switchesCount = 0;
    this.lastAction = 0;
    this.lastState = null;
    this.lastQValues = [0, 0];
  }

  // Convert raw intersection environment state into discrete state key
  encodeState(envState) {
    // envState: { phase: 0 (NS) or 1 (EW), queueNS: int, queueEW: int, elapsedSec: float }
    // Buckets:
    // Queues: 0 (0-1), 1 (2-4), 2 (5-8), 3 (9-14), 4 (15+)
    const qNSBucket = discretize(envState.queueNS, [2, 5, 9, 15]);
    const qEWBucket = discretize(envState.queueEW, [2, 5, 9, 15]);
    // Elapsed time buckets: 0 (<5s), 1 (5-12s), 2 (12-25s), 3 (25s+)
    const timeBucket = discretize(envState.elapsedSec, [5, 12, 25]);
    const phaseStr = envState.phase === 0 ? "NS" : "EW";
    return `${phaseStr}_Q${qNSBucket}_Q${qEWBucket}_T${timeBucket}`;
  }

  getQ(stateKey) {
    if (!this.qTable.has(stateKey)) {
      // Initialize with optimistic zero/small noise
      this.qTable.set(stateKey, [0.0, 0.0]);
    }
    return this.qTable.get(stateKey);
  }

  selectAction(stateKey, explore = true) {
    const qValues = this.getQ(stateKey);
    this.lastQValues = [...qValues];

    // Epsilon-greedy selection
    if (explore && Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actions.length);
    }

    // Exploit: pick argmax with tie-breaking
    if (qValues[0] === qValues[1]) {
      return 0; // Default to keeping phase to prevent jitter
    }
    return qValues[0] > qValues[1] ? 0 : 1;
  }

  decayEpsilon() {
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
  }

  reset() {
    this.qTable.clear();
    this.totalSteps = 0;
    this.totalReward = 0;
    this.decisionsCount = 0;
    this.switchesCount = 0;
    this.lastTDError = 0;
    this.lastState = null;
    this.lastAction = 0;
    this.lastQValues = [0, 0];
  }
}

/**
 * 1. Q-LEARNING (Off-Policy TD Control)
 * Updates towards greedy action regardless of actual exploration policy:
 * Q(s, a) = Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
 */
class QLearningAgent extends BaseRLAgent {
  constructor() {
    super("Q-Learning", "q_learning", "#22D3EE");
  }

  step(prevStateKey, action, reward, nextStateKey) {
    const prevQ = this.getQ(prevStateKey);
    const nextQ = this.getQ(nextStateKey);

    // Max Q over next actions (Off-policy)
    const maxNextQ = Math.max(nextQ[0], nextQ[1]);
    const target = reward + this.gamma * maxNextQ;
    const tdError = target - prevQ[action];

    // Update
    prevQ[action] += this.alpha * tdError;
    this.lastTDError = Math.abs(tdError);

    this.totalSteps++;
    this.totalReward += reward;
    this.decisionsCount++;
    if (action === 1) this.switchesCount++;
    this.decayEpsilon();

    return tdError;
  }
}

/**
 * 2. SARSA (On-Policy TD Control)
 * Updates towards the actual next action chosen by current epsilon-greedy policy:
 * Q(s, a) = Q(s, a) + alpha * [ r + gamma * Q(s', a') - Q(s, a) ]
 */
class SARSAAgent extends BaseRLAgent {
  constructor() {
    super("SARSA", "sarsa", "#22C55E");
    this.nextAction = null;
  }

  // In SARSA, action for next state is selected beforehand
  chooseAction(stateKey) {
    if (this.nextAction !== null) {
      const act = this.nextAction;
      this.nextAction = null;
      return act;
    }
    return this.selectAction(stateKey, true);
  }

  step(prevStateKey, action, reward, nextStateKey) {
    // Pick next action under current policy
    const nextAction = this.selectAction(nextStateKey, true);
    this.nextAction = nextAction;

    const prevQ = this.getQ(prevStateKey);
    const nextQ = this.getQ(nextStateKey);

    // On-policy target uses Q(s', a')
    const target = reward + this.gamma * nextQ[nextAction];
    const tdError = target - prevQ[action];

    prevQ[action] += this.alpha * tdError;
    this.lastTDError = Math.abs(tdError);

    this.totalSteps++;
    this.totalReward += reward;
    this.decisionsCount++;
    if (action === 1) this.switchesCount++;
    this.decayEpsilon();

    return tdError;
  }
}

/**
 * 3. MODEL-BASED RL (Dyna-Q Architecture)
 * Learns a predictive environment transition and reward model:
 * Model(s, a) -> (s', r)
 * In addition to direct RL, it performs N "hallucinated" planning updates
 * per step, giving high sample efficiency!
 */
class ModelBasedAgent extends BaseRLAgent {
  constructor() {
    super("Model-Based RL (Dyna-Q)", "model_based", "#8B5CF6");
    // World Model: Map of "s_a" -> { nextStateKey, reward, count }
    this.model = new Map();
    this.visitedStates = new Set();
    this.stateActions = new Map(); // state -> Set of actions taken
    this.planningSteps = 5;        // Hallucinated planning steps per real step
  }

  step(prevStateKey, action, reward, nextStateKey) {
    // 1. Direct RL update (Q-Learning step on real world experience)
    const prevQ = this.getQ(prevStateKey);
    const nextQ = this.getQ(nextStateKey);
    const maxNextQ = Math.max(nextQ[0], nextQ[1]);
    const target = reward + this.gamma * maxNextQ;
    const tdError = target - prevQ[action];
    prevQ[action] += this.alpha * tdError;
    this.lastTDError = Math.abs(tdError);

    // 2. Model Learning: Update internal model T(s, a) and R(s, a)
    const saKey = `${prevStateKey}__${action}`;
    this.model.set(saKey, { nextStateKey, reward });
    this.visitedStates.add(prevStateKey);
    
    if (!this.stateActions.has(prevStateKey)) {
      this.stateActions.set(prevStateKey, new Set());
    }
    this.stateActions.get(prevStateKey).add(action);

    // 3. Background Hallucinated Planning (Dyna-Q Steps)
    this.runPlanning();

    this.totalSteps++;
    this.totalReward += reward;
    this.decisionsCount++;
    if (action === 1) this.switchesCount++;
    this.decayEpsilon();

    return tdError;
  }

  runPlanning() {
    const visitedArr = Array.from(this.visitedStates);
    if (visitedArr.length === 0) return;

    for (let i = 0; i < this.planningSteps; i++) {
      // Sample previously visited state
      const randomState = visitedArr[Math.floor(Math.random() * visitedArr.length)];
      const actionsForState = Array.from(this.stateActions.get(randomState) || [0]);
      const randomAction = actionsForState[Math.floor(Math.random() * actionsForState.length)];

      const saKey = `${randomState}__${randomAction}`;
      const modeled = this.model.get(saKey);
      if (!modeled) continue;

      // Update Q-values using the modeled transition
      const simQ = this.getQ(randomState);
      const simNextQ = this.getQ(modeled.nextStateKey);
      const maxSimNextQ = Math.max(simNextQ[0], simNextQ[1]);
      const simTarget = modeled.reward + this.gamma * maxSimNextQ;
      
      simQ[randomAction] += this.alpha * (simTarget - simQ[randomAction]);
    }
  }

  reset() {
    super.reset();
    this.model.clear();
    this.visitedStates.clear();
    this.stateActions.clear();
  }
}

/**
 * 4. FIXED-TIMER BASELINE
 * Traditional non-adaptive traffic light cycle (e.g. 18s NS, 18s EW)
 */
class FixedTimerAgent {
  constructor() {
    this.name = "Fixed Timer";
    this.id = "fixed_timer";
    this.color = "#64748B";
    this.fixedGreenDuration = 18.0; // seconds
    this.totalSteps = 0;
    this.totalReward = 0;
    this.decisionsCount = 0;
    this.switchesCount = 0;
    this.lastQValues = [0, 0];
    this.lastTDError = 0;
  }

  encodeState(envState) {
    return `FixedTimer_T${Math.floor(envState.elapsedSec)}`;
  }

  selectAction(stateKey, explore = false, envState = null) {
    if (!envState) return 0;
    // Switch only after fixed green time has expired
    return envState.elapsedSec >= this.fixedGreenDuration ? 1 : 0;
  }

  chooseAction(stateKey, envState) {
    return this.selectAction(stateKey, false, envState);
  }

  step(prevStateKey, action, reward, nextStateKey) {
    this.totalSteps++;
    this.totalReward += reward;
    this.decisionsCount++;
    if (action === 1) this.switchesCount++;
    return 0;
  }

  reset() {
    this.totalSteps = 0;
    this.totalReward = 0;
    this.decisionsCount = 0;
    this.switchesCount = 0;
  }
}

window.RLAgents = {
  QLearningAgent,
  SARSAAgent,
  ModelBasedAgent,
  FixedTimerAgent
};
