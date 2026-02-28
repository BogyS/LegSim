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
    clearance: 0.08,
    stance_ratio: 0.62,
    toe_up_deg: 12.0,
    push_off_deg: 12.0,
    knee_branch: -1.0,
  };

  const STATE = { ...DEFAULTS };

  const DATA = {};

  const elements = {
    walkCanvas: document.getElementById("sim-walk-canvas"),
    angleCanvas: document.getElementById("sim-angle-canvas"),
    pauseBtn: document.getElementById("sim-pause"),
    resetBtn: document.getElementById("sim-reset"),
    hipHeight: document.getElementById("sim-hip-height"),
    stepLen: document.getElementById("sim-step-len"),
    clearance: document.getElementById("sim-clearance"),
    stanceRatio: document.getElementById("sim-stance-ratio"),
    toeUp: document.getElementById("sim-toe-up"),
    pushOff: document.getElementById("sim-push-off"),
    hipHeightVal: document.getElementById("sim-hip-height-val"),
    stepLenVal: document.getElementById("sim-step-len-val"),
    clearanceVal: document.getElementById("sim-clearance-val"),
    stanceRatioVal: document.getElementById("sim-stance-ratio-val"),
    toeUpVal: document.getElementById("sim-toe-up-val"),
    pushOffVal: document.getElementById("sim-push-off-val"),
  };

  let paused = false;
  let frame = 0;
  let lastTick = 0;

  function smoothstep(s) {
    const x = Math.max(0, Math.min(1, s));
    return x * x * (3 - 2 * x);
  }

  function clamp(x, lo, hi) {
    return Math.min(Math.max(x, lo), hi);
  }

  function legPhase(time, offset) {
    return ((time / T) + offset) % 1.0;
  }

  function buildLegTrajectory(offset, hipX, stanceRatio, stepLen, clearance, toeUp, pushOff) {
    const ankX = new Array(N).fill(0);
    const ankY = new Array(N).fill(0);
    const theta = new Array(N).fill(0);

    let plantedX = hipX[0];
    let prevInStance = null;

    for (let i = 0; i < N; i += 1) {
      const ph = legPhase(T_ARR[i], offset);
      const inStance = ph < stanceRatio;

      if (prevInStance === null) {
        prevInStance = inStance;
      }

      if (inStance && !prevInStance) {
        plantedX = hipX[i] + 0.25 * stepLen;
      }
      if (!inStance && prevInStance) {
        plantedX = hipX[i] - 0.25 * stepLen;
      }
      prevInStance = inStance;

      if (inStance) {
        const s = ph / stanceRatio;
        ankX[i] = plantedX;
        ankY[i] = 0.0;
        const push = smoothstep((s - 0.75) / 0.25);
        theta[i] = -pushOff * push;
      } else {
        const s = (ph - stanceRatio) / (1.0 - stanceRatio);
        const xTakeoff = plantedX;
        const xLand = hipX[i] + 0.25 * stepLen;
        ankX[i] = xTakeoff + (xLand - xTakeoff) * smoothstep(s);
        ankY[i] = clearance * Math.sin(Math.PI * s);
        theta[i] = toeUp * Math.sin(Math.PI * s);
      }
    }
    return { ankX, ankY, theta };
  }

  function solveLegIK(ankX, ankY, theta, hipX, hipY, kneeBranch) {
    const q1 = new Array(N);
    const q2 = new Array(N);
    const q3 = new Array(N);
    const kx = new Array(N);
    const ky = new Array(N);
    const ax = new Array(N);
    const ay = new Array(N);
    const hx = new Array(N);
    const hy = new Array(N);
    const tx = new Array(N);
    const ty = new Array(N);

    for (let i = 0; i < N; i += 1) {
      const xRel = ankX[i] - hipX[i];
      const yRel = ankY[i] - hipY[i];
      const r2 = xRel * xRel + yRel * yRel;
      let c2 = (r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2);
      c2 = clamp(c2, -1, 1);
      const s2 = kneeBranch * Math.sqrt(Math.max(0, 1 - c2 * c2));
      const q2i = Math.atan2(s2, c2);
      const q1i = Math.atan2(yRel, xRel) - Math.atan2(L2 * Math.sin(q2i), L1 + L2 * Math.cos(q2i));
      const q3i = theta[i] - (q1i + q2i);

      q1[i] = q1i;
      q2[i] = q2i;
      q3[i] = q3i;

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

    return { q1, q2, q3, kx, ky, ax, ay, hx, hy, tx, ty };
  }

  function recomputeAll() {
    const hipHeight = Number(STATE.hip_height);
    const stepLen = Number(STATE.step_len);
    const clearance = Number(STATE.clearance);
    const stanceRatio = Number(STATE.stance_ratio);
    const toeUp = Number(STATE.toe_up_deg) * (Math.PI / 180);
    const pushOff = Number(STATE.push_off_deg) * (Math.PI / 180);
    const kneeBranch = Number(STATE.knee_branch);

    const hipX = new Array(N);
    const hipY = new Array(N);
    const v = stepLen / T;
    for (let i = 0; i < N; i += 1) {
      hipX[i] = v * T_ARR[i];
      hipY[i] = hipHeight;
    }

    const leftTraj = buildLegTrajectory(0.0, hipX, stanceRatio, stepLen, clearance, toeUp, pushOff);
    const rightTraj = buildLegTrajectory(0.5, hipX, stanceRatio, stepLen, clearance, toeUp, pushOff);

    const left = solveLegIK(leftTraj.ankX, leftTraj.ankY, leftTraj.theta, hipX, hipY, kneeBranch);
    const right = solveLegIK(rightTraj.ankX, rightTraj.ankY, rightTraj.theta, hipX, hipY, kneeBranch);

    DATA.hipX = hipX;
    DATA.hipY = hipY;
    DATA.ankLx = leftTraj.ankX;
    DATA.ankLy = leftTraj.ankY;
    DATA.ankRx = rightTraj.ankX;
    DATA.ankRy = rightTraj.ankY;
    DATA.left = left;
    DATA.right = right;

    DATA.minX = Math.min(...hipX) - 0.4;
    DATA.maxX = Math.max(...hipX) + 0.4;
    DATA.minY = -0.05;
    DATA.maxY = hipHeight + TORSO_LEN + 0.15;

    const toDeg = (arr) => arr.map((v) => v * (180 / Math.PI));
    const Lq1 = toDeg(left.q1);
    const Lq2 = toDeg(left.q2);
    const Lq3 = toDeg(left.q3);
    const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const q1m = mean(Lq1);
    const q2m = mean(Lq2);
    const q3m = mean(Lq3);
    DATA.Lq1c = Lq1.map((v) => v - q1m);
    DATA.Lq2c = Lq2.map((v) => v - q2m);
    DATA.Lq3c = Lq3.map((v) => v - q3m);
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

    ctx.strokeStyle = "#1564a6";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(mapX(hx), mapY(hy));
    ctx.lineTo(mapX(left.kx[i]), mapY(left.ky[i]));
    ctx.lineTo(mapX(left.ax[i]), mapY(left.ay[i]));
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(mapX(hx), mapY(hy));
    ctx.lineTo(mapX(right.kx[i]), mapY(right.ky[i]));
    ctx.lineTo(mapX(right.ax[i]), mapY(right.ay[i]));
    ctx.stroke();

    ctx.strokeStyle = "#0f4a7a";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(mapX(left.hx[i]), mapY(left.hy[i]));
    ctx.lineTo(mapX(left.tx[i]), mapY(left.ty[i]));
    ctx.stroke();

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
    dot(left.kx[i], left.ky[i]);
    dot(left.ax[i], left.ay[i]);
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
    if (!paused && delta >= 1000 / FPS) {
      frame = (frame + 1) % N;
      lastTick = ts;
    }
    render();
    window.requestAnimationFrame(tick);
  }

  function updateLabels() {
    elements.hipHeightVal.textContent = Number(STATE.hip_height).toFixed(2);
    elements.stepLenVal.textContent = Number(STATE.step_len).toFixed(2);
    elements.clearanceVal.textContent = Number(STATE.clearance).toFixed(2);
    elements.stanceRatioVal.textContent = Number(STATE.stance_ratio).toFixed(3);
    elements.toeUpVal.textContent = Number(STATE.toe_up_deg).toFixed(1);
    elements.pushOffVal.textContent = Number(STATE.push_off_deg).toFixed(1);
  }

  function bindSlider(input, key, formatter) {
    input.addEventListener("input", () => {
      STATE[key] = Number(input.value);
      updateLabels();
      recomputeAll();
    });
    if (formatter) {
      input.value = formatter(STATE[key]);
    } else {
      input.value = STATE[key];
    }
  }

  function resetDefaults() {
    Object.assign(STATE, DEFAULTS);
    elements.hipHeight.value = STATE.hip_height;
    elements.stepLen.value = STATE.step_len;
    elements.clearance.value = STATE.clearance;
    elements.stanceRatio.value = STATE.stance_ratio;
    elements.toeUp.value = STATE.toe_up_deg;
    elements.pushOff.value = STATE.push_off_deg;
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
    resetDefaults();
  });

  bindSlider(elements.hipHeight, "hip_height");
  bindSlider(elements.stepLen, "step_len");
  bindSlider(elements.clearance, "clearance");
  bindSlider(elements.stanceRatio, "stance_ratio");
  bindSlider(elements.toeUp, "toe_up_deg");
  bindSlider(elements.pushOff, "push_off_deg");

  resetDefaults();
  render();
  window.addEventListener("resize", render);
  window.requestAnimationFrame(tick);
})();
