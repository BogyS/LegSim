(() => {
  "use strict";

  const FPS = 60;
  const T = 1.0;
  const NUM_STEPS = 3;
  const TOTAL_TIME = NUM_STEPS * T;
  const N = Math.floor(TOTAL_TIME * FPS);
  const T_ARR = new Array(N);
  for (let i = 0; i < N; i += 1) {
    T_ARR[i] = (i / FPS);
  }

  const BASE_HEIGHT = 1.8;
  const BASE_L1 = 0.45;
  const BASE_L2 = 0.43;
  const BASE_FOOT_TOTAL = 0.265;
  const BASE_HEEL_BACK = 0.06;
  const BASE_MTP_FWD = 0.105;
  const BASE_TORSO_LEN = 0.55;
  const WORLD_MIN_X = -0.4;
  const WORLD_MAX_X = 2.5;
  const DEFAULTS = {
    humanHeight: 1.8,
    speed: 1.0,
  };

  const DATA = {};

  const elements = {
    walkCanvas: document.getElementById("sim-walk-canvas"),
    angleCanvas: document.getElementById("sim-angle-canvas"),
    pauseBtn: document.getElementById("sim-pause"),
    resetBtn: document.getElementById("sim-reset"),
    directionBtn: document.getElementById("sim-direction"),
    modeBtn: document.getElementById("sim-mode"),
    nextPhaseBtn: document.getElementById("sim-next-phase"),
    interpMode: document.getElementById("sim-interp-mode"),
    humanHeight: document.getElementById("sim-human-height"),
    speed: document.getElementById("sim-speed"),
    phaseStatus: document.getElementById("sim-phase-status"),
    humanHeightVal: document.getElementById("sim-human-height-val"),
    hipHeightVal: document.getElementById("sim-hip-height-val"),
    stepLenVal: document.getElementById("sim-step-len-val"),
    l1Val: document.getElementById("sim-l1-val"),
    l2Val: document.getElementById("sim-l2-val"),
    speedVal: document.getElementById("sim-speed-val"),
  };

  let paused = false;
  let frame = 0;
  let lastTick = 0;
  let moveForward = true;
  let phaseMode = false;
  let phaseIndex = 0;
  let interpMode = "smoothstep";
  let currentStep = 0;
  let currentPhase = 0;
  const STATE = { ...DEFAULTS };

  const STANCE_RATIO = 0.6;

  function smoothstep(s) {
    const x = Math.max(0, Math.min(1, s));
    return x * x * (3 - 2 * x);
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function mapPhaseToCanonical(phase) {
    const stanceRef = 0.6;
    if (phase <= STANCE_RATIO) {
      const s = STANCE_RATIO === 0 ? 0 : phase / STANCE_RATIO;
      return s * stanceRef;
    }
    const swingPhase = (phase - STANCE_RATIO) / Math.max(1e-6, 1 - STANCE_RATIO);
    return stanceRef + swingPhase * (1 - stanceRef);
  }

  const GAIT_KEYS_FORWARD = [
    { p: 0.0, hip: 20, knee: -0, ankle: 0 },
    { p: 0.1, hip: 15, knee: -15, ankle: 5 },
    { p: 0.3, hip: 5, knee: -5, ankle: -5 },
    { p: 0.5, hip: -10, knee: -5, ankle: -10 },
    { p: 0.6, hip: -10, knee: -30, ankle: 20 },
    { p: 0.73, hip: 20, knee: -60, ankle: 10 },
    { p: 0.87, hip: 30, knee: -30, ankle: 0 },
    { p: 1.0, hip: 30, knee: -0, ankle: 0 },
  ];

  const GAIT_KEYS_BACKWARD = [
    { p: 0.0, hip: -20, knee: -0, ankle: -0 },
    { p: 0.1, hip: -15, knee: -15, ankle: -5 },
    { p: 0.3, hip: -5, knee: -5, ankle: 5 },
    { p: 0.5, hip: 10, knee: -5, ankle: 10 },
    { p: 0.6, hip: 10, knee: -30, ankle: -20 },
    { p: 0.73, hip: -20, knee: -60, ankle: -10 },
    { p: 0.87, hip: -30, knee: -30, ankle: 0 },
    { p: 1.0, hip: -30, knee: -0, ankle: 0 },
  ];

  const GAIT_PHASES = GAIT_KEYS_FORWARD.map((k) => k.p);
  const PHASE_INFO = [
    "Initial Contact: heel strikes the ground and prepares weight acceptance.",
    "Loading Response: body weight transfers onto the left leg, knee starts flexing.",
    "Mid Stance: body progresses over the supporting leg with postural control.",
    "Terminal Stance: heel rises and propulsion preparation begins.",
    "Pre Swing (Toe-off): final push-off as the foot leaves the ground.",
    "Initial Swing: thigh advances and knee flexes for toe clearance.",
    "Mid Swing: shank advances as the leg passes the opposite side.",
    "Terminal Swing: extension prepares the next initial contact.",
  ];

  function normFrame(x) {
    const n = Math.floor(x) % N;
    return n < 0 ? n + N : n;
  }

  function nearestPhaseIndex(phase) {
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < GAIT_PHASES.length; i += 1) {
      const d = Math.abs(GAIT_PHASES[i] - phase);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    return best;
  }

  function stepPhaseToFrame(step, phase) {
    const s = ((step % NUM_STEPS) + NUM_STEPS) % NUM_STEPS;
    const p = Math.max(0, Math.min(1, phase));
    const base = s * FPS;
    return (base + Math.round(p * (FPS - 1))) % N;
  }

  function syncStepPhaseFromFrame() {
    const fi = normFrame(frame);
    currentStep = Math.floor(fi / FPS);
    currentPhase = (fi - (currentStep * FPS)) / (FPS - 1);
  }

  function gaitAngles(phase) {
    const canonical = mapPhaseToCanonical(phase);
    const keys = moveForward ? GAIT_KEYS_FORWARD : GAIT_KEYS_BACKWARD;
    let i = 0;
    while (i < keys.length - 1 && keys[i + 1].p < canonical) {
      i += 1;
    }
    const k0 = keys[i];
    const k1 = keys[Math.min(i + 1, keys.length - 1)];
    const span = Math.max(1e-6, k1.p - k0.p);
    const rawT = Math.max(0, Math.min(1, (canonical - k0.p) / span));
    const t = interpMode === "step" ? 0 : (interpMode === "linear" ? rawT : smoothstep(rawT));

    let hip = lerp(k0.hip, k1.hip, t);
    let knee = lerp(k0.knee, k1.knee, t);
    let ankle = lerp(k0.ankle, k1.ankle, t);

    return { hip, knee, ankle };
  }

  function buildGeometry() {
    const scale = STATE.humanHeight / BASE_HEIGHT;
    const l1 = BASE_L1 * scale;
    const l2 = BASE_L2 * scale;
    const footTotal = BASE_FOOT_TOTAL * scale;
    const heelBack = BASE_HEEL_BACK * scale;
    const mtpFwd = BASE_MTP_FWD * scale;
    return {
      l1,
      l2,
      mtpFwd,
      toeTipFwd: Math.max(0.02, footTotal - mtpFwd),
      heelBack,
      torsoLen: BASE_TORSO_LEN * scale,
      hipHeight: l1 + l2,
    };
  }

  function deriveStepLength(geom) {
    const toX = (phase) => {
      const a = gaitAngles(phase);
      const q1 = (a.hip - 90) * (Math.PI / 180);
      const q2 = a.knee * (Math.PI / 180);
      return geom.l1 * Math.cos(q1) + geom.l2 * Math.cos(q1 + q2);
    };
    const step = Math.abs(toX(0.0) - toX(0.5));
    return Math.max(0.2, step);
  }

  function clamp01(x) {
    return Math.max(0, Math.min(1, x));
  }

  function legMinRelativeY(phase, angles, geom) {
    const q1 = (angles.hip - 90) * (Math.PI / 180);
    const q2 = angles.knee * (Math.PI / 180);
    const footTheta = angles.ankle * (Math.PI / 180);
    const q3 = footTheta - (q1 + q2);
    const footAng = q1 + q2 + q3;

    const kneeY = geom.l1 * Math.sin(q1);
    const ankleY = kneeY + geom.l2 * Math.sin(q1 + q2);
    const heelY = ankleY - geom.heelBack * Math.sin(footAng);
    const mtpY = ankleY + geom.mtpFwd * Math.sin(footAng);
    let toeY = mtpY + geom.toeTipFwd * Math.sin(footAng);

    if (phase >= 0.3 && phase <= 0.5) {
      toeY = mtpY;
    }

    return Math.min(0, kneeY, ankleY, heelY, mtpY, toeY);
  }

  function recomputeAll() {
    const geom = buildGeometry();
    const stepLen = deriveStepLength(geom);
    const scale = geom.l1 / BASE_L1;

    const hipX = new Array(N);
    const hipY = new Array(N);
    const topHipY = geom.hipHeight - (0.02 * scale);
    const bottomHipY = topHipY - (0.05 * scale);
    const groundMargin = 0.002 * scale;
    // Speed factor should change only playback rate, not spatial step size.
    const v = (stepLen / T);
    for (let i = 0; i < N; i += 1) {
      hipX[i] = moveForward ? (v * T_ARR[i]) : (v * (TOTAL_TIME - T_ARR[i]));
      const phaseL = ((T_ARR[i] / T) + 0.0) % 1.0;
      const phaseR = ((T_ARR[i] / T) + 0.5) % 1.0;
      const anglesL = gaitAngles(phaseL);
      const anglesR = gaitAngles(phaseR);

      const spreadNorm = clamp01(Math.abs(anglesL.hip - anglesR.hip) / 70.0);
      const desiredHipY = topHipY - ((topHipY - bottomHipY) * spreadNorm);

      const reqL = -legMinRelativeY(phaseL, anglesL, geom);
      const reqR = -legMinRelativeY(phaseR, anglesR, geom);
      const requiredHipY = Math.max(reqL, reqR) + groundMargin;

      hipY[i] = Math.max(desiredHipY, requiredHipY);
    }

    const left = computeLegSeries(0.0, hipX, hipY, geom);
    const right = computeLegSeries(0.5, hipX, hipY, geom);

    DATA.hipX = hipX;
    DATA.hipY = hipY;
    DATA.left = left;
    DATA.right = right;

    DATA.minX = WORLD_MIN_X;
    DATA.maxX = WORLD_MAX_X;
    DATA.minY = -0.05;
    DATA.maxY = Math.max(...hipY) + geom.torsoLen + 0.15;

    DATA.Lq1 = left.q1deg;
    DATA.Lq2 = left.q2deg;
    DATA.Lq3 = left.q3deg;
    DATA.stepLen = stepLen;
    DATA.hipHeight = hipY.reduce((acc, y) => acc + y, 0) / hipY.length;
    DATA.geom = geom;
  }

  function computeLegSeries(offset, hipX, hipY, geom) {
    const q1 = new Array(N);
    const q2 = new Array(N);
    const q3 = new Array(N);
    const q1deg = new Array(N);
    const q2deg = new Array(N);
    const q3deg = new Array(N);
    const kx = new Array(N);
    const ky = new Array(N);
    const ax = new Array(N);
    const ay = new Array(N);
    const mx = new Array(N);
    const my = new Array(N);
    const hx = new Array(N);
    const hy = new Array(N);
    const tx = new Array(N);
    const ty = new Array(N);

    for (let i = 0; i < N; i += 1) {
      const phase = ((T_ARR[i] / T) + offset) % 1.0;
      const angles = gaitAngles(phase);
      const hipFlex = angles.hip;
      const kneeFlex = angles.knee;
      const ankleRel = angles.ankle;
      

      const q1i = (hipFlex - 90) * (Math.PI / 180);
      const q2i = kneeFlex * (Math.PI / 180);
      // Treat ankle keyframe as target foot orientation vs horizontal.
      const footTheta = ankleRel * (Math.PI / 180);
      const q3i = footTheta - (q1i + q2i);

      q1[i] = q1i;
      q2[i] = q2i;
      q3[i] = q3i;
      q1deg[i] = hipFlex;
      q2deg[i] = kneeFlex;
      q3deg[i] = ankleRel;

      const kneeX = hipX[i] + geom.l1 * Math.cos(q1i);
      const kneeY = hipY[i] + geom.l1 * Math.sin(q1i);
      const ankleX = kneeX + geom.l2 * Math.cos(q1i + q2i);
      const ankleY = kneeY + geom.l2 * Math.sin(q1i + q2i);
      const footAng = q1i + q2i + q3i;

      kx[i] = kneeX;
      ky[i] = kneeY;
      ax[i] = ankleX;
      ay[i] = ankleY;
      mx[i] = ankleX + geom.mtpFwd * Math.cos(footAng);
      my[i] = ankleY + geom.mtpFwd * Math.sin(footAng);
      if (phase >= 0.3 && phase <= 0.5) {
        // Keep toe segment horizontal in local mid-to-terminal stance.
        const toeDir = Math.sign(Math.cos(footAng)) || 1;
        tx[i] = mx[i] + (geom.toeTipFwd * toeDir);
        ty[i] = my[i];
      } else {
        tx[i] = mx[i] + geom.toeTipFwd * Math.cos(footAng);
        ty[i] = my[i] + geom.toeTipFwd * Math.sin(footAng);
      }
      hx[i] = ankleX - geom.heelBack * Math.cos(footAng);
      hy[i] = ankleY - geom.heelBack * Math.sin(footAng);
    }

    return { q1, q2, q3, q1deg, q2deg, q3deg, kx, ky, ax, ay, mx, my, hx, hy, tx, ty };
  }

  function normalizeCanvas(canvas, ctx) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function drawWalk() {
    const ctx = elements.walkCanvas.getContext("2d");
    normalizeCanvas(elements.walkCanvas, ctx);
    const rect = elements.walkCanvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#fbfaf8";
    ctx.fillRect(0, 0, w, h);

    if (!DATA.hipX) {
      return;
    }

    const i = normFrame(frame);
    const mapX = (x) => (x - DATA.minX) * (w / (DATA.maxX - DATA.minX));
    const mapY = (y) => h - (y - DATA.minY) * (h / (DATA.maxY - DATA.minY));

    ctx.strokeStyle = "#d9cbb7";
    ctx.lineWidth = 2;
    const groundY = mapY(0);
    ctx.beginPath();
    ctx.moveTo(20, groundY);
    ctx.lineTo(w - 20, groundY);
    ctx.stroke();

    const hx = DATA.hipX[i];
    const hy = DATA.hipY[i];

    const left = DATA.left;
    const right = DATA.right;

    ctx.strokeStyle = "rgba(26, 26, 26, 0.45)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let j = 0; j < DATA.hipX.length; j += 1) {
      const x = mapX(DATA.hipX[j]);
      const y = mapY(DATA.hipY[j]);
      if (j === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    ctx.lineWidth = 4;
    ctx.strokeStyle = "#1564a6";
    ctx.beginPath();
    ctx.moveTo(mapX(hx), mapY(hy));
    ctx.lineTo(mapX(left.kx[i]), mapY(left.ky[i]));
    ctx.stroke();

    ctx.strokeStyle = "#c06030";
    ctx.beginPath();
    ctx.moveTo(mapX(left.kx[i]), mapY(left.ky[i]));
    ctx.lineTo(mapX(left.ax[i]), mapY(left.ay[i]));
    ctx.stroke();

    ctx.strokeStyle = "#1a1a1a";
    ctx.beginPath();
    ctx.moveTo(mapX(hx), mapY(hy));
    ctx.lineTo(mapX(right.kx[i]), mapY(right.ky[i]));
    ctx.lineTo(mapX(right.ax[i]), mapY(right.ay[i]));
    ctx.stroke();

    ctx.lineWidth = 3;
    ctx.strokeStyle = "#2b7a4b";
    ctx.beginPath();
    ctx.moveTo(mapX(left.hx[i]), mapY(left.hy[i]));
    ctx.lineTo(mapX(left.mx[i]), mapY(left.my[i]));
    ctx.stroke();

    ctx.strokeStyle = "#1f9f60";
    ctx.beginPath();
    ctx.moveTo(mapX(left.mx[i]), mapY(left.my[i]));
    ctx.lineTo(mapX(left.tx[i]), mapY(left.ty[i]));
    ctx.stroke();

    ctx.strokeStyle = "#1a1a1a";
    ctx.beginPath();
    ctx.moveTo(mapX(right.hx[i]), mapY(right.hy[i]));
    ctx.lineTo(mapX(right.mx[i]), mapY(right.my[i]));
    ctx.lineTo(mapX(right.tx[i]), mapY(right.ty[i]));
    ctx.stroke();

    ctx.fillStyle = "#1a1a1a";
    const dot = (x, y) => {
      ctx.beginPath();
      ctx.arc(mapX(x), mapY(y), 4, 0, Math.PI * 2);
      ctx.fill();
    };
    dot(hx, hy);
    ctx.fillStyle = "#c06030";
    dot(left.kx[i], left.ky[i]);
    ctx.fillStyle = "#2b7a4b";
    dot(left.ax[i], left.ay[i]);
    ctx.fillStyle = "#1a1a1a";
    dot(right.kx[i], right.ky[i]);
    dot(right.ax[i], right.ay[i]);

    ctx.strokeStyle = "#1a1a1a";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(mapX(hx), mapY(hy));
    ctx.lineTo(mapX(hx), mapY(hy + DATA.geom.torsoLen));
    ctx.stroke();
  }

  function drawAnglePanel(ctx, data, panel, color, label) {
    const pad = 24;
    const panelHeight = (panel.height - pad * 2) / 3;
    const top = pad + panelHeight * panel.index;
    const left = pad;
    const width = panel.width - pad * 2;
    const height = panelHeight - 10;

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;
    const padY = range * 0.1;
    const lo = min - padY;
    const hi = max + padY;

    const mapX = (i) => left + (i / (N - 1)) * width;
    const mapY = (v) => top + height - ((v - lo) / (hi - lo)) * height;

    ctx.strokeStyle = "#d9cbb7";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(left, top, width, height);
    ctx.stroke();

    const zeroY = mapY(0);
    ctx.strokeStyle = "#b7a58e";
    ctx.setLineDash([5, 6]);
    ctx.beginPath();
    ctx.moveTo(left, zeroY);
    ctx.lineTo(left + width, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < N; i += 1) {
      const x = mapX(i);
      const y = mapY(data[i]);
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    ctx.fillStyle = "#1a1a1a";
    ctx.font = "12px Manrope, sans-serif";
    ctx.fillText(label, left + 6, top + 14);

    const fi = normFrame(frame);
    const cursorX = mapX(fi);
    ctx.strokeStyle = "#1a1a1a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cursorX, top);
    ctx.lineTo(cursorX, top + height);
    ctx.stroke();

    ctx.fillStyle = "#1a1a1a";
    ctx.beginPath();
    ctx.arc(cursorX, mapY(data[fi]), 3, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawAngles() {
    const ctx = elements.angleCanvas.getContext("2d");
    normalizeCanvas(elements.angleCanvas, ctx);
    const rect = elements.angleCanvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fbfaf8";
    ctx.fillRect(0, 0, width, height);

    if (!DATA.Lq1) {
      return;
    }

    const panel = { width, height };
    drawAnglePanel(ctx, DATA.Lq1, { ...panel, index: 0 }, "#1564a6", "Left hip (deg)");
    drawAnglePanel(ctx, DATA.Lq2, { ...panel, index: 1 }, "#c06030", "Left knee (deg)");
    drawAnglePanel(ctx, DATA.Lq3, { ...panel, index: 2 }, "#2b7a4b", "Left ankle (deg)");
  }

  function updatePhaseStatus() {
    const fi = normFrame(frame);
    const step = Math.floor(fi / FPS);
    const phase = (fi - (step * FPS)) / (FPS - 1);
    const phaseSlot = nearestPhaseIndex(phase);
    elements.phaseStatus.textContent =
      `Left step ${step + 1}/${NUM_STEPS} | Phase ${phase.toFixed(2)} (${phaseSlot + 1}/${GAIT_PHASES.length}) - ${PHASE_INFO[phaseSlot]}`;
  }

  function render() {
    drawWalk();
    drawAngles();
    updatePhaseStatus();
  }

  function tick(ts) {
    if (!lastTick) {
      lastTick = ts;
    }
    const delta = ts - lastTick;
    if (!phaseMode && !paused && delta >= 1000 / FPS) {
      frame = (frame + STATE.speed) % N;
      syncStepPhaseFromFrame();
      lastTick = ts;
    }
    render();
    window.requestAnimationFrame(tick);
  }

  function updateLabels() {
    const geom = DATA.geom ?? buildGeometry();
    elements.humanHeightVal.textContent = Number(STATE.humanHeight).toFixed(2);
    elements.hipHeightVal.textContent = Number(DATA.hipHeight ?? geom.hipHeight).toFixed(2);
    elements.stepLenVal.textContent = Number(DATA.stepLen ?? deriveStepLength(geom)).toFixed(2);
    elements.l1Val.textContent = Number(geom.l1).toFixed(3);
    elements.l2Val.textContent = Number(geom.l2).toFixed(3);
    elements.speedVal.textContent = Number(STATE.speed).toFixed(1);
  }

  function resetDefaults() {
    STATE.humanHeight = DEFAULTS.humanHeight;
    STATE.speed = DEFAULTS.speed;
    elements.humanHeight.value = String(STATE.humanHeight);
    elements.speed.value = String(STATE.speed);
    recomputeAll();
    updateLabels();
  }

  elements.pauseBtn.addEventListener("click", () => {
    paused = !paused;
    elements.pauseBtn.textContent = paused ? "Resume" : "Pause";
  });

  elements.resetBtn.addEventListener("click", () => {
    paused = false;
    elements.pauseBtn.textContent = "Pause";
    frame = 0;
    currentStep = 0;
    currentPhase = 0;
    phaseIndex = 0;
    resetDefaults();
  });

  elements.directionBtn.addEventListener("click", () => {
    moveForward = !moveForward;
    elements.directionBtn.textContent = moveForward ? "Backwards" : "Forwards";
    recomputeAll();
    syncStepPhaseFromFrame();
    updateLabels();
    render();
  });

  elements.interpMode.addEventListener("change", () => {
    interpMode = elements.interpMode.value;
    recomputeAll();
    syncStepPhaseFromFrame();
    updateLabels();
    render();
  });

  elements.humanHeight.addEventListener("input", () => {
    STATE.humanHeight = Number(elements.humanHeight.value);
    recomputeAll();
    syncStepPhaseFromFrame();
    updateLabels();
    render();
  });

  elements.speed.addEventListener("input", () => {
    STATE.speed = Number(elements.speed.value);
    recomputeAll();
    syncStepPhaseFromFrame();
    updateLabels();
    render();
  });

  elements.modeBtn.addEventListener("click", () => {
    phaseMode = !phaseMode;
    elements.modeBtn.textContent = phaseMode ? "Continuous" : "Phase Mode";
    elements.nextPhaseBtn.disabled = !phaseMode;
    if (!phaseMode) {
      lastTick = 0;
    } else {
      syncStepPhaseFromFrame();
      phaseIndex = nearestPhaseIndex(currentPhase);
      currentPhase = GAIT_PHASES[phaseIndex];
      frame = stepPhaseToFrame(currentStep, currentPhase);
    }
    render();
  });

  elements.nextPhaseBtn.addEventListener("click", () => {
    if (!phaseMode) {
      return;
    }
    phaseIndex = (phaseIndex + 1) % GAIT_PHASES.length;
    if (phaseIndex === 0) {
      currentStep = (currentStep + 1) % NUM_STEPS;
    }
    currentPhase = GAIT_PHASES[phaseIndex];
    frame = stepPhaseToFrame(currentStep, currentPhase);
    render();
  });

  moveForward = !(elements.directionBtn.textContent.trim().toLowerCase() === "forwards");
  interpMode = elements.interpMode.value;
  //phaseMode = elements.modeBtn.textContent.trim().toLowerCase().includes("phase");
  elements.nextPhaseBtn.disabled = !phaseMode;
  if (phaseMode) {
    frame = stepPhaseToFrame(currentStep, GAIT_PHASES[phaseIndex]);
  }
  resetDefaults();
  syncStepPhaseFromFrame();
  updateLabels();
  render();
  window.addEventListener("resize", render);
  window.requestAnimationFrame(tick);
})();
