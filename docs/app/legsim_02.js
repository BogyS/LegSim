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

  const L1 = 0.45;
  const L2 = 0.43;
  const FOOT_TOTAL = 0.265;
  const HEEL_BACK = 0.06;
  const TOE_FWD = FOOT_TOTAL - HEEL_BACK;
  const TORSO_LEN = 0.55;

  const DEFAULTS = {
    hip_height: 0.92,
    step_len: 0.7,
  };

  const STATE = { ...DEFAULTS };

  const DATA = {};

  const elements = {
    walkCanvas: document.getElementById("sim-walk-canvas"),
    angleCanvas: document.getElementById("sim-angle-canvas"),
    pauseBtn: document.getElementById("sim-pause"),
    resetBtn: document.getElementById("sim-reset"),
    directionBtn: document.getElementById("sim-direction"),
    modeBtn: document.getElementById("sim-mode"),
    nextPhaseBtn: document.getElementById("sim-next-phase"),
    hipHeight: document.getElementById("sim-hip-height"),
    stepLen: document.getElementById("sim-step-len"),
    hipHeightVal: document.getElementById("sim-hip-height-val"),
    stepLenVal: document.getElementById("sim-step-len-val"),
  };

  let paused = false;
  let frame = 0;
  let lastTick = 0;
  let moveForward = true;
  let phaseMode = false;
  let phaseIndex = 0;

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
    { p: 0.5, hip: -10, knee: -5, ankle: 0 },
    { p: 0.6, hip: -10, knee: -30, ankle: 20 },
    { p: 0.73, hip: 20, knee: -60, ankle: 10 },
    { p: 0.87, hip: 30, knee: -30, ankle: 0 },
    { p: 1.0, hip: 30, knee: -0, ankle: 0 },
  ];

  const GAIT_KEYS_BACKWARD = [
    { p: 0.0, hip: -20, knee: -0, ankle: -0 },
    { p: 0.1, hip: -15, knee: -15, ankle: -5 },
    { p: 0.3, hip: -5, knee: -5, ankle: 5 },
    { p: 0.5, hip: 10, knee: -5, ankle: 0 },
    { p: 0.6, hip: 10, knee: -30, ankle: -20 },
    { p: 0.73, hip: -20, knee: -60, ankle: -10 },
    { p: 0.87, hip: -30, knee: -30, ankle: 0 },
    { p: 1.0, hip: -30, knee: -0, ankle: 0 },
  ];

  const GAIT_PHASES = GAIT_KEYS_FORWARD.map((k) => k.p);

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
    const t = smoothstep((canonical - k0.p) / span);

    let hip = lerp(k0.hip, k1.hip, t);
    let knee = lerp(k0.knee, k1.knee, t);
    let ankle = lerp(k0.ankle, k1.ankle, t);

    return { hip, knee, ankle };
  }

  function recomputeAll() {
    const hipHeight = Number(STATE.hip_height);
    const stepLen = Number(STATE.step_len);

    const hipX = new Array(N);
    const hipY = new Array(N);
    const v = stepLen / T;
    for (let i = 0; i < N; i += 1) {
      hipX[i] = moveForward ? (v * T_ARR[i]) : (v * (TOTAL_TIME - T_ARR[i]));
      hipY[i] = hipHeight;
    }

    const left = computeLegSeries(0.0, hipX, hipY, true);
    const right = computeLegSeries(0.5, hipX, hipY, false);

    DATA.hipX = hipX;
    DATA.hipY = hipY;
    DATA.left = left;
    DATA.right = right;

    DATA.minX = Math.min(...hipX) - 0.4;
    DATA.maxX = Math.max(...hipX) + 0.4;
    DATA.minY = -0.05;
    DATA.maxY = hipHeight + TORSO_LEN + 0.15;

    const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const q1m = mean(left.q1deg);
    const q2m = mean(left.q2deg);
    const q3m = mean(left.q3deg);
    DATA.Lq1c = left.q1deg.map((v) => v - q1m);
    DATA.Lq2c = left.q2deg.map((v) => v - q2m);
    DATA.Lq3c = left.q3deg.map((v) => v - q3m);
  }

  function computeLegSeries(offset, hipX, hipY, isLeft) {
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
    const hx = new Array(N);
    const hy = new Array(N);
    const tx = new Array(N);
    const ty = new Array(N);

    let footOffset = 0;
    for (let i = 0; i < N; i += 1) {
      const phase = ((T_ARR[i] / T) + offset) % 1.0;
      const angles = gaitAngles(phase);
      const hipFlex = angles.hip;
      const kneeFlex = angles.knee;
      const ankleRel = angles.ankle;
      

      const q1i = (hipFlex - 90) * (Math.PI / 180);
      const q2i = kneeFlex * (Math.PI / 180);
      let q3i = ankleRel * (Math.PI / 180);

      if (i === 0) {
        footOffset = -(q1i + q2i + q3i);
      }
      q3i += footOffset;

      q1[i] = q1i;
      q2[i] = q2i;
      q3[i] = q3i;
      q1deg[i] = hipFlex;
      q2deg[i] = kneeFlex;
      q3deg[i] = ankleRel + (footOffset * 180 / Math.PI);

      const kneeX = hipX[i] + L1 * Math.cos(q1i);
      const kneeY = hipY[i] + L1 * Math.sin(q1i);
      const ankleX = kneeX + L2 * Math.cos(q1i + q2i);
      const ankleY = kneeY + L2 * Math.sin(q1i + q2i);
      const footAng = q1i + q2i + q3i;

      kx[i] = kneeX;
      ky[i] = kneeY;
      ax[i] = ankleX;
      ay[i] = ankleY;
      tx[i] = ankleX + TOE_FWD * Math.cos(footAng);
      ty[i] = ankleY + TOE_FWD * Math.sin(footAng);
      hx[i] = ankleX - HEEL_BACK * Math.cos(footAng);
      hy[i] = ankleY - HEEL_BACK * Math.sin(footAng);
    }

    return { q1, q2, q3, q1deg, q2deg, q3deg, kx, ky, ax, ay, hx, hy, tx, ty };
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

    const i = frame;
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
    ctx.lineTo(mapX(left.tx[i]), mapY(left.ty[i]));
    ctx.stroke();

    ctx.strokeStyle = "#1a1a1a";
    ctx.beginPath();
    ctx.moveTo(mapX(right.hx[i]), mapY(right.hy[i]));
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
    ctx.lineTo(mapX(hx), mapY(hy + TORSO_LEN));
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

    const cursorX = mapX(frame);
    ctx.strokeStyle = "#1a1a1a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cursorX, top);
    ctx.lineTo(cursorX, top + height);
    ctx.stroke();

    ctx.fillStyle = "#1a1a1a";
    ctx.beginPath();
    ctx.arc(cursorX, mapY(data[frame]), 3, 0, Math.PI * 2);
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

    if (!DATA.Lq1c) {
      return;
    }

    const panel = { width, height };
    drawAnglePanel(ctx, DATA.Lq1c, { ...panel, index: 0 }, "#1564a6", "Left hip (deg, centered)");
    drawAnglePanel(ctx, DATA.Lq2c, { ...panel, index: 1 }, "#c06030", "Left knee (deg, centered)");
    drawAnglePanel(ctx, DATA.Lq3c, { ...panel, index: 2 }, "#2b7a4b", "Left ankle (deg, centered)");
  }

  function render() {
    drawWalk();
    drawAngles();
  }

  function tick(ts) {
    if (!lastTick) {
      lastTick = ts;
    }
    const delta = ts - lastTick;
    if (!phaseMode && !paused && delta >= 1000 / FPS) {
      frame = (frame + 1) % N;
      lastTick = ts;
    }
    render();
    window.requestAnimationFrame(tick);
  }

  function updateLabels() {
    elements.hipHeightVal.textContent = Number(STATE.hip_height).toFixed(2);
    elements.stepLenVal.textContent = Number(STATE.step_len).toFixed(2);
  }

  function bindSlider(input, key) {
    input.addEventListener("input", () => {
      STATE[key] = Number(input.value);
      updateLabels();
      recomputeAll();
    });
    input.value = STATE[key];
  }

  function resetDefaults() {
    Object.assign(STATE, DEFAULTS);
    elements.hipHeight.value = STATE.hip_height;
    elements.stepLen.value = STATE.step_len;
    updateLabels();
    recomputeAll();
  }

  elements.pauseBtn.addEventListener("click", () => {
    paused = !paused;
    elements.pauseBtn.textContent = paused ? "Resume" : "Pause";
  });

  elements.resetBtn.addEventListener("click", () => {
    paused = false;
    elements.pauseBtn.textContent = "Pause";
    frame = 0;
    phaseIndex = 0;
    resetDefaults();
  });

  elements.directionBtn.addEventListener("click", () => {
    moveForward = !moveForward;
    elements.directionBtn.textContent = moveForward ? "Backwards" : "Forwards";
    recomputeAll();
    render();
  });

  elements.modeBtn.addEventListener("click", () => {
    phaseMode = !phaseMode;
    elements.modeBtn.textContent = phaseMode ? "Phase mode" : "Continuous";
    if (!phaseMode) {
      lastTick = 0;
    }
    render();
  });

  elements.nextPhaseBtn.addEventListener("click", () => {
    if (!phaseMode) {
      phaseMode = true;
      elements.modeBtn.textContent = "Phase mode";
    }
    phaseIndex = (phaseIndex + 1) % GAIT_PHASES.length;
    frame = Math.round(GAIT_PHASES[phaseIndex] * (N - 1));
    render();
  });

  bindSlider(elements.hipHeight, "hip_height");
  bindSlider(elements.stepLen, "step_len");

  moveForward = !(elements.directionBtn.textContent.trim().toLowerCase() === "forwards");
  phaseMode = elements.modeBtn.textContent.trim().toLowerCase().includes("phase");
  if (phaseMode) {
    frame = Math.round(GAIT_PHASES[phaseIndex] * (N - 1));
  }
  resetDefaults();
  render();
  window.addEventListener("resize", render);
  window.requestAnimationFrame(tick);
})();
