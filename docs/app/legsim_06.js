(() => {
  "use strict";

  const FPS = 60;
  const T = 1.0;
  const N = Math.floor(T * FPS);
  const T_ARR = new Array(N);
  for (let i = 0; i < N; i += 1) {
    T_ARR[i] = i / FPS;
  }

  const BASE_HEIGHT = 1.8;
  const BASE_L1 = 0.45;
  const BASE_L2 = 0.43;
  const BASE_FOOT = 0.18;
  const BED_MARGIN = 0.002;
  const WORLD_MIN_X = -0.15;
  const WORLD_MAX_X = 1.2;

  const DEFAULTS = {
    humanHeight: 1.8,
    hipMaxDeg: 35,
    kneeMaxDeg: 110,
    speed: 1.0,
    cycle: 0.0,
  };

  const elements = {
    bedCanvas: document.getElementById("sim-bed-canvas"),
    angleCanvas: document.getElementById("sim-angle-canvas"),
    pauseBtn: document.getElementById("sim-pause"),
    resetBtn: document.getElementById("sim-reset"),
    humanHeight: document.getElementById("sim-human-height"),
    hipMax: document.getElementById("sim-hip-max"),
    kneeMax: document.getElementById("sim-knee-max"),
    speed: document.getElementById("sim-speed"),
    cycle: document.getElementById("sim-cycle"),
    status: document.getElementById("sim-status"),
    humanHeightVal: document.getElementById("sim-human-height-val"),
    hipMaxVal: document.getElementById("sim-hip-max-val"),
    kneeMaxVal: document.getElementById("sim-knee-max-val"),
    speedVal: document.getElementById("sim-speed-val"),
    cycleVal: document.getElementById("sim-cycle-val"),
    l1Val: document.getElementById("sim-l1-val"),
    l2Val: document.getElementById("sim-l2-val"),
  };

  const DATA = {};
  const STATE = { ...DEFAULTS };

  let paused = false;
  let cycle = DEFAULTS.cycle;
  let lastTick = 0;

  function clamp01(x) {
    return Math.max(0, Math.min(1, x));
  }

  function smoothstep(s) {
    const x = clamp01(s);
    return x * x * (3 - 2 * x);
  }

  function mirrorCycle(t) {
    return t < 0.5 ? (t * 2) : (2 - t * 2);
  }

  function cycleEnvelope(t) {
    return smoothstep(mirrorCycle(t));
  }

  function normalizeCanvas(canvas, ctx) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function buildGeometry() {
    const scale = STATE.humanHeight / BASE_HEIGHT;
    return {
      l1: BASE_L1 * scale,
      l2: BASE_L2 * scale,
      foot: BASE_FOOT * scale,
      groundMargin: BED_MARGIN * scale,
    };
  }

  function computeAngles(cyclePos) {
    const e = cycleEnvelope(cyclePos);
    const hip = STATE.hipMaxDeg * e;
    const kneeGate = smoothstep(clamp01((e - 0.12) / 0.88));
    const knee = STATE.kneeMaxDeg * kneeGate;
    return { hip, knee };
  }

  function computePose(cyclePos, geom) {
    const angles = computeAngles(cyclePos);
    const hipRad = angles.hip * (Math.PI / 180);
    const kneeRad = angles.knee * (Math.PI / 180);
    const thighTheta = hipRad;
    const shankTheta = hipRad + kneeRad;
    const hipX = 0.08;
    const hipY = geom.groundMargin;
    const kneeX = hipX + geom.l1 * Math.cos(thighTheta);
    const kneeY = hipY + geom.l1 * Math.sin(thighTheta);
    const ankleX = kneeX + geom.l2 * Math.cos(shankTheta);
    const ankleY = kneeY + geom.l2 * Math.sin(shankTheta);
    const toeX = ankleX + geom.foot * Math.cos(shankTheta);
    const toeY = ankleY + geom.foot * Math.sin(shankTheta);
    const clearance = Math.min(hipY, kneeY, ankleY, toeY);

    return {
      cyclePos,
      hipDeg: angles.hip,
      kneeDeg: angles.knee,
      hipX,
      hipY,
      kneeX,
      kneeY,
      ankleX,
      ankleY,
      toeX,
      toeY,
      clearance,
    };
  }

  function recomputeAll() {
    const geom = buildGeometry();
    const hipSeries = new Array(N);
    const kneeSeries = new Array(N);
    for (let i = 0; i < N; i += 1) {
      const pose = computePose(T_ARR[i], geom);
      hipSeries[i] = pose.hipDeg;
      kneeSeries[i] = pose.kneeDeg;
    }

    DATA.geom = geom;
    DATA.hipSeries = hipSeries;
    DATA.kneeSeries = kneeSeries;
    DATA.pose = computePose(cycle, geom);
    DATA.minY = -0.03;
    DATA.maxY = Math.max(geom.l1 + geom.l2 + geom.foot + 0.12, 0.75);
  }

  function drawBedView() {
    const ctx = elements.bedCanvas.getContext("2d");
    normalizeCanvas(elements.bedCanvas, ctx);
    const rect = elements.bedCanvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#fbfaf8";
    ctx.fillRect(0, 0, w, h);

    const mapX = (x) => (x - WORLD_MIN_X) * (w / (WORLD_MAX_X - WORLD_MIN_X));
    const groundY = h - 30;
    const topPad = 16;
    const pxPerMeter = (groundY - topPad) / Math.max(0.2, DATA.maxY);
    const mapY = (y) => groundY - (y * pxPerMeter);

    ctx.strokeStyle = "#d9cbb7";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(20, groundY);
    ctx.lineTo(w - 20, groundY);
    ctx.stroke();

    ctx.fillStyle = "#5c5c5c";
    ctx.font = "12px Manrope, sans-serif";
    ctx.fillText("Bed line", 24, groundY - 8);

    const pose = DATA.pose;
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#1564a6";
    ctx.beginPath();
    ctx.moveTo(mapX(pose.hipX), mapY(pose.hipY));
    ctx.lineTo(mapX(pose.kneeX), mapY(pose.kneeY));
    ctx.stroke();

    ctx.strokeStyle = "#c06030";
    ctx.beginPath();
    ctx.moveTo(mapX(pose.kneeX), mapY(pose.kneeY));
    ctx.lineTo(mapX(pose.ankleX), mapY(pose.ankleY));
    ctx.stroke();

    ctx.strokeStyle = "#2b7a4b";
    ctx.beginPath();
    ctx.moveTo(mapX(pose.ankleX), mapY(pose.ankleY));
    ctx.lineTo(mapX(pose.toeX), mapY(pose.toeY));
    ctx.stroke();

    ctx.fillStyle = "#1a1a1a";
    const dot = (x, y) => {
      ctx.beginPath();
      ctx.arc(mapX(x), mapY(y), 4, 0, Math.PI * 2);
      ctx.fill();
    };
    dot(pose.hipX, pose.hipY);
    dot(pose.kneeX, pose.kneeY);
    dot(pose.ankleX, pose.ankleY);
  }

  function drawAnglePanel(ctx, data, panel, color, label) {
    const pad = 24;
    const panelHeight = (panel.height - pad * 2) / 2;
    const top = pad + panelHeight * panel.index;
    const left = pad;
    const width = panel.width - pad * 2;
    const height = panelHeight - 12;
    const lo = 0;
    const hi = Math.max(...data, 10) + 8;

    const mapX = (i) => left + (i / (N - 1)) * width;
    const mapY = (v) => top + height - ((v - lo) / (hi - lo)) * height;

    ctx.strokeStyle = "#d9cbb7";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(left, top, width, height);
    ctx.stroke();

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

    const fi = Math.max(0, Math.min(N - 1, Math.round(cycle * (N - 1))));
    const cursorX = mapX(fi);
    ctx.strokeStyle = "#1a1a1a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cursorX, top);
    ctx.lineTo(cursorX, top + height);
    ctx.stroke();

    ctx.fillStyle = "#1a1a1a";
    ctx.font = "12px Manrope, sans-serif";
    ctx.fillText(label, left + 6, top + 14);
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

    drawAnglePanel(ctx, DATA.hipSeries, { width, height, index: 0 }, "#1564a6", "Hip flexion (deg)");
    drawAnglePanel(ctx, DATA.kneeSeries, { width, height, index: 1 }, "#c06030", "Knee flexion (deg)");
  }

  function updateStatus() {
    const pose = DATA.pose;
    elements.status.textContent =
      `Cycle ${pose.cyclePos.toFixed(2)} | Hip ${pose.hipDeg.toFixed(1)} deg | ` +
      `Knee ${pose.kneeDeg.toFixed(1)} deg | Clearance ${pose.clearance.toFixed(3)} m`;
  }

  function render() {
    DATA.pose = computePose(cycle, DATA.geom);
    drawBedView();
    drawAngles();
    updateStatus();
  }

  function updateLabels() {
    elements.humanHeightVal.textContent = STATE.humanHeight.toFixed(2);
    elements.hipMaxVal.textContent = `${STATE.hipMaxDeg.toFixed(0)} deg`;
    elements.kneeMaxVal.textContent = `${STATE.kneeMaxDeg.toFixed(0)} deg`;
    elements.speedVal.textContent = STATE.speed.toFixed(1);
    elements.cycleVal.textContent = cycle.toFixed(2);
    elements.l1Val.textContent = `L1 ${DATA.geom.l1.toFixed(3)}`;
    elements.l2Val.textContent = `L2 ${DATA.geom.l2.toFixed(3)}`;
  }

  function syncCycleInput() {
    elements.cycle.value = String(cycle);
  }

  function resetDefaults() {
    Object.assign(STATE, DEFAULTS);
    cycle = DEFAULTS.cycle;
    elements.humanHeight.value = String(STATE.humanHeight);
    elements.hipMax.value = String(STATE.hipMaxDeg);
    elements.kneeMax.value = String(STATE.kneeMaxDeg);
    elements.speed.value = String(STATE.speed);
    syncCycleInput();
    recomputeAll();
    updateLabels();
    render();
  }

  function handleParamChange() {
    recomputeAll();
    updateLabels();
    render();
  }

  function tick(ts) {
    if (!lastTick) {
      lastTick = ts;
    }
    const delta = ts - lastTick;
    if (!paused && delta >= 1000 / FPS) {
      cycle = (cycle + ((STATE.speed / FPS) / T)) % 1;
      syncCycleInput();
      updateLabels();
      render();
      lastTick = ts;
    }
    window.requestAnimationFrame(tick);
  }

  elements.pauseBtn.addEventListener("click", () => {
    paused = !paused;
    elements.pauseBtn.textContent = paused ? "Resume" : "Pause";
  });

  elements.resetBtn.addEventListener("click", () => {
    paused = false;
    elements.pauseBtn.textContent = "Pause";
    lastTick = 0;
    resetDefaults();
  });

  elements.humanHeight.addEventListener("input", () => {
    STATE.humanHeight = Number(elements.humanHeight.value);
    handleParamChange();
  });

  elements.hipMax.addEventListener("input", () => {
    STATE.hipMaxDeg = Number(elements.hipMax.value);
    handleParamChange();
  });

  elements.kneeMax.addEventListener("input", () => {
    STATE.kneeMaxDeg = Number(elements.kneeMax.value);
    handleParamChange();
  });

  elements.speed.addEventListener("input", () => {
    STATE.speed = Number(elements.speed.value);
    updateLabels();
  });

  elements.cycle.addEventListener("input", () => {
    cycle = Number(elements.cycle.value);
    paused = true;
    elements.pauseBtn.textContent = "Resume";
    updateLabels();
    render();
  });

  resetDefaults();
  window.addEventListener("resize", render);
  window.requestAnimationFrame(tick);
})();
