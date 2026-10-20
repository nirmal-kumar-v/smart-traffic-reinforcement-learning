/**
 * Smart Traffic RL — Intersection Physics & Rendering Simulation Engine
 * Handles vehicle kinematics, lane queues, collision avoidance, traffic light state machine,
 * and canvas rendering with modern aesthetics (headlights, brake lights, road markings).
 */

class Vehicle {
  constructor(id, direction, x, y, speed, maxSpeed, color) {
    this.id = id;
    this.direction = direction; // 'N', 'S', 'E', 'W' (direction moving towards)
    this.x = x;
    this.y = y;
    this.speed = speed;
    this.maxSpeed = maxSpeed;
    this.color = color;
    this.length = 22;
    this.width = 12;

    // Wait time & kinematics
    this.waitTime = 0;       // Seconds spent at speed < 0.2
    this.hasPassed = false;   // Passed stop line
    this.isBraking = false;
    this.creationTime = performance.now();
  }

  update(dt, speedLimit) {
    this.maxSpeed = speedLimit;
    if (this.speed < 0.2) {
      this.waitTime += dt;
      this.isBraking = true;
    } else {
      this.isBraking = false;
    }

    // Move forward according to direction
    switch (this.direction) {
      case 'S': // Moving South (North -> South)
        this.y += this.speed;
        break;
      case 'N': // Moving North (South -> North)
        this.y -= this.speed;
        break;
      case 'E': // Moving East (West -> East)
        this.x += this.speed;
        break;
      case 'W': // Moving West (East -> West)
        this.x -= this.speed;
        break;
    }
  }

  draw(ctx, scale = 1) {
    ctx.save();
    ctx.translate(this.x * scale, this.y * scale);

    // Rotate according to heading
    let angle = 0;
    if (this.direction === 'S') angle = Math.PI / 2;
    else if (this.direction === 'N') angle = -Math.PI / 2;
    else if (this.direction === 'E') angle = 0;
    else if (this.direction === 'W') angle = Math.PI;

    ctx.rotate(angle);

    const l = this.length * scale;
    const w = this.width * scale;

    // Vehicle Shadow
    ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
    ctx.beginPath();
    ctx.roundRect(-l / 2 + 2, -w / 2 + 3, l, w, 4 * scale);
    ctx.fill();

    // Car Body
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.roundRect(-l / 2, -w / 2, l, w, 3.5 * scale);
    ctx.fill();

    // Windshield / Cabin
    ctx.fillStyle = "#0F172A";
    ctx.beginPath();
    ctx.roundRect(-l * 0.15, -w * 0.38, l * 0.45, w * 0.76, 2 * scale);
    ctx.fill();

    // Headlights (Front glow)
    ctx.fillStyle = "#FEF08A";
    ctx.fillRect(l / 2 - 2 * scale, -w / 2 + 1.5 * scale, 2.5 * scale, 2.5 * scale);
    ctx.fillRect(l / 2 - 2 * scale, w / 2 - 4 * scale, 2.5 * scale, 2.5 * scale);

    // Brake Lights (Rear)
    ctx.fillStyle = this.isBraking ? "#EF4444" : "#7F1D1D";
    ctx.fillRect(-l / 2, -w / 2 + 1.5 * scale, 2 * scale, 2.5 * scale);
    ctx.fillRect(-l / 2, w / 2 - 4 * scale, 2 * scale, 2.5 * scale);

    // Brake glow when stopping
    if (this.isBraking) {
      ctx.shadowColor = "#EF4444";
      ctx.shadowBlur = 8 * scale;
      ctx.fillStyle = "#EF4444";
      ctx.fillRect(-l / 2 - 1 * scale, -w / 2 + 1 * scale, 2 * scale, 3 * scale);
      ctx.fillRect(-l / 2 - 1 * scale, w / 2 - 4 * scale, 2 * scale, 3 * scale);
      ctx.shadowBlur = 0;
    }

    ctx.restore();
  }
}

class TrafficSimulation {
  constructor(canvasId, agent, isMini = false) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.agent = agent;
    this.isMini = isMini;

    // Simulation dimensions (standard virtual coord space is 600x600)
    this.virtualWidth = 600;
    this.virtualHeight = 600;
    this.center = 300;
    this.roadWidth = 130; // 2 lanes each direction

    // Traffic Light Phases
    // 0: NS_GREEN, 1: NS_YELLOW, 2: EW_GREEN, 3: EW_YELLOW
    this.phase = 0;
    this.phaseTime = 0; // elapsed seconds in current phase
    this.yellowDuration = 2.0; // seconds
    this.minGreenDuration = 3.0; // minimum seconds before agent can switch
    this.targetPhase = 0;

    // Vehicle collections by approach
    this.vehicles = [];
    this.nextVehicleId = 1;

    // Configuration
    this.inflowNS = 0.7; // vehicles per second per direction
    this.inflowEW = 0.7;
    this.maxSpeed = 3.5;
    this.penaltyWait = 1.0;
    this.penaltySwitch = 5.0;
    this.bonusThroughput = 10.0;

    // Spawn timers
    this.spawnTimerN = 0;
    this.spawnTimerS = 0;
    this.spawnTimerE = 0;
    this.spawnTimerW = 0;

    // Stop Lines (Virtual coordinates)
    this.stopLines = {
      N: this.center - this.roadWidth / 2, // 235
      S: this.center + this.roadWidth / 2, // 365
      W: this.center - this.roadWidth / 2, // 235
      E: this.center + this.roadWidth / 2  // 365
    };

    // Telemetry & Metrics
    this.totalThroughput = 0;
    this.cumulativeReward = 0;
    this.currentWaitTotal = 0;
    this.avgWaitTime = 0;
    this.queueCountNS = 0;
    this.queueCountEW = 0;
    this.congestionPct = 0;
    this.episode = 1;
    this.decisionTimer = 0;
    this.decisionInterval = 1.8; // agent makes a decision every 1.8s

    // Colors for vehicles
    this.carColors = [
      '#22D3EE', '#38BDF8', '#818CF8', '#A78BFA', '#F472B6',
      '#FBBF24', '#34D399', '#E2E8F0', '#94A3B8'
    ];
  }

  reset() {
    this.vehicles = [];
    this.phase = 0;
    this.phaseTime = 0;
    this.totalThroughput = 0;
    this.cumulativeReward = 0;
    this.currentWaitTotal = 0;
    this.avgWaitTime = 0;
    this.queueCountNS = 0;
    this.queueCountEW = 0;
    this.congestionPct = 0;
    this.decisionTimer = 0;
    this.agent.reset();
  }

  // Set flow rates based on presets
  setFlowPreset(preset) {
    if (preset === 'balanced') {
      this.inflowNS = 0.7;
      this.inflowEW = 0.7;
    } else if (preset === 'rush_ns') {
      this.inflowNS = 1.3;
      this.inflowEW = 0.35;
    } else if (preset === 'rush_ew') {
      this.inflowNS = 0.35;
      this.inflowEW = 1.3;
    } else if (preset === 'bursts') {
      this.inflowNS = 1.1;
      this.inflowEW = 1.1;
    }
  }

  // Spawn new vehicle if headway permits
  spawnVehicle(direction) {
    let x, y, speed = this.maxSpeed;
    const color = this.carColors[Math.floor(Math.random() * this.carColors.length)];

    // Check if spawn zone is clear (prevent overlapping cars at boundary)
    const spawnClearDistance = 35;

    if (direction === 'S') { // Spawns at top (North), heading South (lane right of center)
      x = this.center + this.roadWidth / 4;
      y = 0;
      const blocking = this.vehicles.find(v => v.direction === 'S' && v.y < spawnClearDistance);
      if (blocking) return;
    } else if (direction === 'N') { // Spawns at bottom (South), heading North (lane right of center)
      x = this.center - this.roadWidth / 4;
      y = this.virtualHeight;
      const blocking = this.vehicles.find(v => v.direction === 'N' && v.y > this.virtualHeight - spawnClearDistance);
      if (blocking) return;
    } else if (direction === 'E') { // Spawns at left (West), heading East (lane right of center)
      x = 0;
      y = this.center + this.roadWidth / 4;
      const blocking = this.vehicles.find(v => v.direction === 'E' && v.x < spawnClearDistance);
      if (blocking) return;
    } else if (direction === 'W') { // Spawns at right (East), heading West (lane right of center)
      x = this.virtualWidth;
      y = this.center - this.roadWidth / 4;
      const blocking = this.vehicles.find(v => v.direction === 'W' && v.x > this.virtualWidth - spawnClearDistance);
      if (blocking) return;
    }

    const vehicle = new Vehicle(this.nextVehicleId++, direction, x, y, speed, this.maxSpeed, color);
    this.vehicles.push(vehicle);
  }

  // Step simulation forward by dt seconds
  update(dt) {
    this.phaseTime += dt;
    this.decisionTimer += dt;

    // 1. Handle Phase Transitions (Yellow clearance handling)
    if (this.phase === 1) { // NS Yellow
      if (this.phaseTime >= this.yellowDuration) {
        this.phase = 2; // Transition to EW Green
        this.phaseTime = 0;
      }
    } else if (this.phase === 3) { // EW Yellow
      if (this.phaseTime >= this.yellowDuration) {
        this.phase = 0; // Transition to NS Green
        this.phaseTime = 0;
      }
    }

    // 2. Periodic Reinforcement Learning Agent Decision
    let currentAction = 0;
    if (this.decisionTimer >= this.decisionInterval) {
      this.decisionTimer = 0;
      currentAction = this.executeRLStep();
    }

    // 3. Spawning Vehicles
    this.spawnTimerN += dt;
    this.spawnTimerS += dt;
    this.spawnTimerE += dt;
    this.spawnTimerW += dt;

    if (this.spawnTimerN >= (1 / this.inflowNS)) {
      this.spawnVehicle('S');
      this.spawnTimerN = 0;
    }
    if (this.spawnTimerS >= (1 / this.inflowNS)) {
      this.spawnVehicle('N');
      this.spawnTimerS = 0;
    }
    if (this.spawnTimerE >= (1 / this.inflowEW)) {
      this.spawnVehicle('E');
      this.spawnTimerE = 0;
    }
    if (this.spawnTimerW >= (1 / this.inflowEW)) {
      this.spawnVehicle('W');
      this.spawnTimerW = 0;
    }

    // 4. Update Vehicle Physics, Queuing & Collision Avoidance
    this.updateVehicles(dt);

    return currentAction;
  }

  // Query RL Agent to evaluate state, receive reward, and execute action
  executeRLStep() {
    // Current Environment State
    const activePhaseCode = (this.phase === 0 || this.phase === 1) ? 0 : 1; // 0: NS, 1: EW
    const envState = {
      phase: activePhaseCode,
      queueNS: this.queueCountNS,
      queueEW: this.queueCountEW,
      elapsedSec: this.phaseTime
    };

    const stateKey = this.agent.encodeState(envState);

    // Compute Reward from previous step
    // Penalty for queue count & delay, plus penalty if switching, plus bonus for cleared cars
    const waitingCars = this.queueCountNS + this.queueCountEW;
    const avgWaitSq = Math.min(25, (this.avgWaitTime * this.avgWaitTime) * 0.15);
    
    // Reward formulation
    const stepReward = -(this.penaltyWait * waitingCars) 
                     - (avgWaitSq * 0.5) 
                     + (this.bonusThroughput * (this.lastStepCleared || 0));

    // RL Agent Action Selection
    let action = 0;
    if (typeof this.agent.chooseAction === 'function') {
      action = this.agent.chooseAction(stateKey, envState);
    } else {
      action = this.agent.selectAction(stateKey, true);
    }

    // Learn from experience if we have a previous state
    if (this.agent.lastState) {
      const switchPenalty = (this.agent.lastAction === 1) ? -this.penaltySwitch : 0;
      const totalReward = stepReward + switchPenalty;
      this.agent.step(this.agent.lastState, this.agent.lastAction, totalReward, stateKey);
      this.cumulativeReward += totalReward;
    }

    this.agent.lastState = stateKey;
    this.agent.lastAction = action;
    this.lastStepCleared = 0;

    // Apply action: 1 = SWITCH_PHASE (only if min green duration has elapsed and not already in yellow)
    if (action === 1 && this.phaseTime >= this.minGreenDuration) {
      if (this.phase === 0) { // NS Green -> trigger NS Yellow
        this.phase = 1;
        this.phaseTime = 0;
      } else if (this.phase === 2) { // EW Green -> trigger EW Yellow
        this.phase = 3;
        this.phaseTime = 0;
      }
    }

    return action;
  }

  updateVehicles(dt) {
    let waitingNS = 0;
    let waitingEW = 0;
    let totalWaitTime = 0;
    let activeVehiclesCount = 0;
    const safeDistance = 28; // headway gap

    // Sort vehicles in each direction by progress to update from front to back
    const dirs = ['N', 'S', 'E', 'W'];
    dirs.forEach(d => {
      const dirVehicles = this.vehicles.filter(v => v.direction === d);
      
      // Sort front-first
      if (d === 'S') dirVehicles.sort((a, b) => b.y - a.y);
      if (d === 'N') dirVehicles.sort((a, b) => a.y - b.y);
      if (d === 'E') dirVehicles.sort((a, b) => b.x - a.x);
      if (d === 'W') dirVehicles.sort((a, b) => a.x - b.x);

      for (let i = 0; i < dirVehicles.length; i++) {
        const v = dirVehicles[i];
        activeVehiclesCount++;
        totalWaitTime += v.waitTime;

        // Determine if light allows passage
        const isNS = (d === 'N' || d === 'S');
        const isGreen = isNS ? (this.phase === 0) : (this.phase === 2);
        const isYellow = isNS ? (this.phase === 1) : (this.phase === 3);

        // Check if passed stop line
        let hasCrossedStop = false;
        let distanceToStop = 999;

        if (d === 'S') {
          distanceToStop = this.stopLines.N - v.y;
          hasCrossedStop = v.y > this.stopLines.N + 5;
        } else if (d === 'N') {
          distanceToStop = v.y - this.stopLines.S;
          hasCrossedStop = v.y < this.stopLines.S - 5;
        } else if (d === 'E') {
          distanceToStop = this.stopLines.W - v.x;
          hasCrossedStop = v.x > this.stopLines.W + 5;
        } else if (d === 'W') {
          distanceToStop = v.x - this.stopLines.E;
          hasCrossedStop = v.x < this.stopLines.E - 5;
        }

        if (hasCrossedStop) {
          v.hasPassed = true;
        }

        // Target speed calculation
        let targetSpeed = this.maxSpeed;

        // 1. Vehicle ahead obstacle check
        if (i > 0) {
          const lead = dirVehicles[i - 1];
          let distToLead = 999;
          if (d === 'S') distToLead = lead.y - v.y;
          else if (d === 'N') distToLead = v.y - lead.y;
          else if (d === 'E') distToLead = lead.x - v.x;
          else if (d === 'W') distToLead = v.x - lead.x;

          if (distToLead < safeDistance) {
            targetSpeed = 0;
          } else if (distToLead < safeDistance * 1.8) {
            targetSpeed = Math.min(lead.speed, this.maxSpeed * 0.4);
          }
        }

        // 2. Traffic Light check (if not yet crossed stop line)
        if (!v.hasPassed && distanceToStop > 0 && distanceToStop < 90) {
          if (!isGreen) {
            if (isYellow) {
              // Yellow light dilemma: stop if far enough, continue if already close to stop line
              if (distanceToStop > 25) {
                targetSpeed = 0;
              }
            } else {
              // Red Light
              targetSpeed = 0;
            }
          }
        }

        // Smooth speed adjustment (acceleration / braking physics)
        if (v.speed < targetSpeed) {
          v.speed = Math.min(targetSpeed, v.speed + 0.15);
        } else if (v.speed > targetSpeed) {
          v.speed = Math.max(targetSpeed, v.speed - 0.28);
        }

        v.update(dt, this.maxSpeed);

        // Queue counts (vehicles stopped before stop line)
        if (!v.hasPassed && v.speed < 0.3) {
          if (isNS) waitingNS++;
          else waitingEW++;
        }
      }
    });

    // Remove vehicles that left simulation canvas & credit throughput
    const remaining = [];
    for (let v of this.vehicles) {
      const outOfBounds = (v.x < -40 || v.x > this.virtualWidth + 40 ||
                           v.y < -40 || v.y > this.virtualHeight + 40);
      if (outOfBounds) {
        this.totalThroughput++;
        this.lastStepCleared = (this.lastStepCleared || 0) + 1;
      } else {
        remaining.push(v);
      }
    }
    this.vehicles = remaining;

    this.queueCountNS = waitingNS;
    this.queueCountEW = waitingEW;
    this.currentWaitTotal = totalWaitTime;
    this.avgWaitTime = activeVehiclesCount > 0 ? (totalWaitTime / activeVehiclesCount) : 0;
    
    // Congestion index: 0 to 100% (based on queue capacity threshold of 20 cars)
    const totalQueued = waitingNS + waitingEW;
    this.congestionPct = Math.min(100, Math.round((totalQueued / 20) * 100));
  }

  // Draw full intersection canvas
  draw() {
    const ctx = this.ctx;
    const w = this.canvas.width;
    const h = this.canvas.height;
    const scale = w / this.virtualWidth;

    ctx.clearRect(0, 0, w, h);

    // 1. Terrain Grass Background
    ctx.fillStyle = "#0A0F1D";
    ctx.fillRect(0, 0, w, h);

    // 2. Asphalt Roads
    ctx.fillStyle = "#161E2E";
    // Vertical Road (North-South)
    ctx.fillRect((this.center - this.roadWidth / 2) * scale, 0, this.roadWidth * scale, h);
    // Horizontal Road (East-West)
    ctx.fillRect(0, (this.center - this.roadWidth / 2) * scale, w, this.roadWidth * scale);

    // 3. Road Markings & Zebra Crossings
    this.drawRoadMarkings(ctx, scale);

    // 4. Vehicles
    for (let v of this.vehicles) {
      v.draw(ctx, scale);
    }

    // 5. Traffic Lights
    this.drawTrafficSignals(ctx, scale);
  }

  drawRoadMarkings(ctx, scale) {
    const c = this.center * scale;
    const rw = this.roadWidth * scale;
    const half = rw / 2;

    ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
    ctx.lineWidth = 1.5 * scale;
    ctx.setLineDash([8 * scale, 8 * scale]);

    // Dashed center divider lanes
    // North lane divider
    ctx.beginPath();
    ctx.moveTo(c, 0);
    ctx.lineTo(c, c - half - 15 * scale);
    ctx.stroke();

    // South lane divider
    ctx.beginPath();
    ctx.moveTo(c, c + half + 15 * scale);
    ctx.lineTo(c, this.virtualHeight * scale);
    ctx.stroke();

    // West lane divider
    ctx.beginPath();
    ctx.moveTo(0, c);
    ctx.lineTo(c - half - 15 * scale, c);
    ctx.stroke();

    // East lane divider
    ctx.beginPath();
    ctx.moveTo(c + half + 15 * scale, c);
    ctx.lineTo(this.virtualWidth * scale, c);
    ctx.stroke();

    ctx.setLineDash([]); // Reset line dash

    // Solid White Stop Lines
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 3 * scale;

    // North Stop Line (right lane only)
    ctx.beginPath();
    ctx.moveTo(c, c - half);
    ctx.lineTo(c + half, c - half);
    ctx.stroke();

    // South Stop Line (left lane only)
    ctx.beginPath();
    ctx.moveTo(c - half, c + half);
    ctx.lineTo(c, c + half);
    ctx.stroke();

    // West Stop Line (bottom lane only)
    ctx.beginPath();
    ctx.moveTo(c - half, c);
    ctx.lineTo(c - half, c + half);
    ctx.stroke();

    // East Stop Line (top lane only)
    ctx.beginPath();
    ctx.moveTo(c + half, c - half);
    ctx.lineTo(c + half, c);
    ctx.stroke();

    // Zebra Crosswalk Lines
    this.drawCrosswalk(ctx, c - half, c - half - 14 * scale, rw, 10 * scale, true, scale);
    this.drawCrosswalk(ctx, c - half, c + half + 4 * scale, rw, 10 * scale, true, scale);
    this.drawCrosswalk(ctx, c - half - 14 * scale, c - half, 10 * scale, rw, false, scale);
    this.drawCrosswalk(ctx, c + half + 4 * scale, c - half, 10 * scale, rw, false, scale);
  }

  drawCrosswalk(ctx, x, y, width, height, isHorizontal, scale) {
    ctx.fillStyle = "rgba(255, 255, 255, 0.18)";
    const stripeCount = 6;
    if (isHorizontal) {
      const stripeWidth = width / (stripeCount * 2);
      for (let i = 0; i < stripeCount; i++) {
        ctx.fillRect(x + i * 2 * stripeWidth, y, stripeWidth, height);
      }
    } else {
      const stripeHeight = height / (stripeCount * 2);
      for (let i = 0; i < stripeCount; i++) {
        ctx.fillRect(x, y + i * 2 * stripeHeight, width, stripeHeight);
      }
    }
  }

  // Draw 4 Traffic Light Signal Units with realistic lamp glow
  drawTrafficSignals(ctx, scale) {
    const c = this.center * scale;
    const rw = this.roadWidth * scale;
    const half = rw / 2;

    const nsGreen = (this.phase === 0);
    const nsYellow = (this.phase === 1);
    const ewGreen = (this.phase === 2);
    const ewYellow = (this.phase === 3);

    // Northbound Light (at South stop line)
    this.drawSignalBox(ctx, c - half - 18 * scale, c + half + 6 * scale, nsGreen, nsYellow, !nsGreen && !nsYellow, scale);
    // Southbound Light (at North stop line)
    this.drawSignalBox(ctx, c + half + 6 * scale, c - half - 36 * scale, nsGreen, nsYellow, !nsGreen && !nsYellow, scale);
    // Eastbound Light (at West stop line)
    this.drawSignalBox(ctx, c - half - 36 * scale, c - half - 18 * scale, ewGreen, ewYellow, !ewGreen && !ewYellow, scale);
    // Westbound Light (at East stop line)
    this.drawSignalBox(ctx, c + half + 6 * scale, c + half + 6 * scale, ewGreen, ewYellow, !ewGreen && !ewYellow, scale);
  }

  drawSignalBox(ctx, x, y, greenOn, yellowOn, redOn, scale) {
    const boxW = 12 * scale;
    const boxH = 30 * scale;
    const r = 3 * scale;

    // Housing
    ctx.fillStyle = "#0F172A";
    ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
    ctx.lineWidth = 1 * scale;
    ctx.beginPath();
    ctx.roundRect(x, y, boxW, boxH, 3 * scale);
    ctx.fill();
    ctx.stroke();

    const lampR = 2.5 * scale;
    const lampX = x + boxW / 2;

    // Red Lamp
    ctx.fillStyle = redOn ? "#EF4444" : "#450A0A";
    ctx.beginPath();
    ctx.arc(lampX, y + 6 * scale, lampR, 0, Math.PI * 2);
    ctx.fill();
    if (redOn) {
      ctx.shadowColor = "#EF4444";
      ctx.shadowBlur = 8 * scale;
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Yellow Lamp
    ctx.fillStyle = yellowOn ? "#F59E0B" : "#451A03";
    ctx.beginPath();
    ctx.arc(lampX, y + 15 * scale, lampR, 0, Math.PI * 2);
    ctx.fill();
    if (yellowOn) {
      ctx.shadowColor = "#F59E0B";
      ctx.shadowBlur = 8 * scale;
      ctx.fill();
      ctx.shadowBlur = 0;
    }

    // Green Lamp
    ctx.fillStyle = greenOn ? "#22C55E" : "#052E16";
    ctx.beginPath();
    ctx.arc(lampX, y + 24 * scale, lampR, 0, Math.PI * 2);
    ctx.fill();
    if (greenOn) {
      ctx.shadowColor = "#22C55E";
      ctx.shadowBlur = 8 * scale;
      ctx.fill();
      ctx.shadowBlur = 0;
    }
  }
}

window.TrafficSimulation = TrafficSimulation;
