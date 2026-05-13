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
  const BASE_FOOT_TOTAL = 0.265;
  const BASE_HEEL_BACK = 0.06;
  const BASE_LEG_RATIO = (BASE_L1 + BASE_L2) / BASE_HEIGHT;
  const HUMAN_HEIGHT_MIN = 1.18;
  const HUMAN_HEIGHT_MAX = 2.2;

  const DEFAULTS = {
    humanHeight: 1.8,
    l1: BASE_L1,
    l2: BASE_L2,
    kneeMaxDeg: 75,
    seatHeight: 0.52,
    speed: 1.0,
    cycle: 0.0,
  };

  const elements = {
    chairCanvas: document.getElementById("sim-chair-canvas"),
    angleCanvas: document.getElementById("sim-angle-canvas"),
    pauseBtn: document.getElementById("sim-pause"),
    resetBtn: document.getElementById("sim-reset"),
    humanHeight: document.getElementById("sim-human-height"),
    humanHeightEdit: document.getElementById("sim-human-height-edit"),
    l1: document.getElementById("sim-l1"),
    l1Edit: document.getElementById("sim-l1-edit"),
    l2: document.getElementById("sim-l2"),
    l2Edit: document.getElementById("sim-l2-edit"),
    kneeMax: document.getElementById("sim-knee-max"),
    seatHeight: document.getElementById("sim-seat-height"),
    speed: document.getElementById("sim-speed"),
    cycle: document.getElementById("sim-cycle"),
    status: document.getElementById("sim-status"),
    humanHeightVal: document.getElementById("sim-human-height-val"),
    kneeMaxVal: document.getElementById("sim-knee-max-val"),
    seatHeightVal: document.getElementById("sim-seat-height-val"),
    speedVal: document.getElementById("sim-speed-val"),
    cycleVal: document.getElementById("sim-cycle-val"),
    l1Val: document.getElementById("sim-l1-val"),
    l2Val: document.getElementById("sim-l2-val"),
    l1PctVal: document.getElementById("sim-l1-pct"),
    l2PctVal: document.getElementById("sim-l2-pct"),
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

  function syncSegmentsFromHeight(nextHeight) {
    const clampedHeight = Math.max(HUMAN_HEIGHT_MIN, Math.min(HUMAN_HEIGHT_MAX, nextHeight));
    const l1Ratio = STATE.l1 / Math.max(1e-6, STATE.humanHeight);
    const l2Ratio = STATE.l2 / Math.max(1e-6, STATE.humanHeight);
    STATE.humanHeight = clampedHeight;
    STATE.l1 = l1Ratio * clampedHeight;
    STATE.l2 = l2Ratio * clampedHeight;
  }

  function syncHeightFromSegments() {
    const estimatedHeight = (STATE.l1 + STATE.l2) / BASE_LEG_RATIO;
    STATE.humanHeight = Math.max(HUMAN_HEIGHT_MIN, Math.min(HUMAN_HEIGHT_MAX, estimatedHeight));
  }

  function buildGeometry() {
    const scale = STATE.humanHeight / BASE_HEIGHT;
    return {
      l1: STATE.l1,
      l2: STATE.l2,
      heelBack: BASE_HEEL_BACK * scale,
      toeFwd: (BASE_FOOT_TOTAL - BASE_HEEL_BACK) * scale,
    };
  }

  function computeAngles(cyclePos) {
    const e = cycleEnvelope(cyclePos);
    return { knee: STATE.kneeMaxDeg * e };
  }

  function computePose(cyclePos, geom) {
    const angles = computeAngles(cyclePos);
    const hipX = 0;
    const hipY = STATE.seatHeight;
    const thighTheta = 0;
    const kneeX = hipX + geom.l1 * Math.cos(thighTheta);
    const kneeY = hipY + geom.l1 * Math.sin(thighTheta);

    const shankTheta = -Math.PI * 0.5 + (angles.knee * Math.PI / 180);
    const footTheta = shankTheta + Math.PI * 0.5;
    const ankleX = kneeX + geom.l2 * Math.cos(shankTheta);
    const ankleY = kneeY + geom.l2 * Math.sin(shankTheta);
    const heelX = ankleX - geom.heelBack * Math.cos(footTheta);
    const heelY = ankleY - geom.heelBack * Math.sin(footTheta);
    const toeX = ankleX + geom.toeFwd * Math.cos(footTheta);
    const toeY = ankleY + geom.toeFwd * Math.sin(footTheta);
    const footHeight = Math.max(heelY, toeY);

    return {
      cyclePos,
      kneeDeg: angles.knee,
      hipX,
      hipY,
      kneeX,
      kneeY,
      ankleX,
      ankleY,
      heelX,
      heelY,
      toeX,
      toeY,
      footHeight,
    };
  }

  function recomputeAll() {
    const geom = buildGeometry();
    const kneeSeries = new Array(N);
    const poses = new Array(N);
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = 0;
    let maxY = STATE.seatHeight + 0.2;

    for (let i = 0; i < N; i += 1) {
      const pose = computePose(T_ARR[i], geom);
      poses[i] = pose;
      kneeSeries[i] = pose.kneeDeg;
      for (const x of [pose.hipX, pose.kneeX, pose.ankleX, pose.heelX, pose.toeX]) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
      }
      for (const y of [pose.hipY, pose.kneeY, pose.ankleY, pose.heelY, pose.toeY]) {
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }

    const spanX = Math.max(0.7, maxX - minX);
    const spanY = Math.max(0.7, maxY - minY);
    DATA.geom = geom;
    DATA.kneeSeries = kneeSeries;
    DATA.pose = computePose(cycle, geom);
    DATA.minX = minX - spanX * 0.12;
    DATA.maxX = maxX + spanX * 0.12;
    DATA.minY = minY - spanY * 0.12;
    DATA.maxY = maxY + spanY * 0.12;
  }

  function drawChairView() {
    const ctx = elements.chairCanvas.getContext("2d");
    normalizeCanvas(elements.chairCanvas, ctx);
    const rect = elements.chairCanvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#fbfaf8";
    ctx.fillRect(0, 0, w, h);

    const leftPad = 24;
    const rightPad = 24;
    const topPad = 16;
    const bottomPad = 28;
    const scaleX = (w - leftPad - rightPad) / (DATA.maxX - DATA.minX);
    const scaleY = (h - topPad - bottomPad) / (DATA.maxY - DATA.minY);
    const pxPerMeter = Math.min(scaleX, scaleY);
    const drawingWidth = (DATA.maxX - DATA.minX) * pxPerMeter;
    const drawingHeight = (DATA.maxY - DATA.minY) * pxPerMeter;
    const offsetX = (w - drawingWidth) * 0.5;
    const offsetY = topPad + ((h - topPad - bottomPad - drawingHeight) * 0.5);
    const mapX = (x) => offsetX + ((x - DATA.minX) * pxPerMeter);
    const mapY = (y) => offsetY + drawingHeight - ((y - DATA.minY) * pxPerMeter);

    const pose = DATA.pose;
    const seatY = mapY(STATE.seatHeight);
    const backX = mapX(pose.hipX - 0.06);
    const frontX = mapX(pose.kneeX + 0.08);

    ctx.strokeStyle = "#b7a58e";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(backX, seatY);
    ctx.lineTo(frontX, seatY);
    ctx.moveTo(backX, seatY);
    ctx.lineTo(backX, mapY(STATE.seatHeight + 0.45));
    ctx.stroke();

    ctx.strokeStyle = "#1564a6";
    ctx.lineWidth = 4;
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
    ctx.moveTo(mapX(pose.heelX), mapY(pose.heelY));
    ctx.lineTo(mapX(pose.ankleX), mapY(pose.ankleY));
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

  function drawAnglePanel(ctx) {
    const pad = 26;
    const left = pad + 18;
    const top = pad;
    const width = ctx.canvas.getBoundingClientRect().width - left - pad;
    const height = ctx.canvas.getBoundingClientRect().height - pad * 2;
    const lo = 0;
    const hi = 90;
    const mapX = (i) => left + (i / (N - 1)) * width;
    const mapY = (v) => top + height - ((v - lo) / (hi - lo)) * height;

    ctx.strokeStyle = "#d9cbb7";
    ctx.lineWidth = 1;
    ctx.strokeRect(left, top, width, height);

    for (let deg = 0; deg <= 90; deg += 15) {
      const gy = mapY(deg);
      ctx.beginPath();
      ctx.moveTo(left, gy);
      ctx.lineTo(left + width, gy);
      ctx.stroke();
      ctx.fillStyle = "#5c5c5c";
      ctx.font = "11px Manrope, sans-serif";
      ctx.fillText(`${deg}`, 5, gy + 4);
    }

    ctx.strokeStyle = "#c06030";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < N; i += 1) {
      const x = mapX(i);
      const y = mapY(DATA.kneeSeries[i]);
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
    ctx.fillText("Knee extension (deg)", left + 6, top + 14);
    ctx.fillText("(deg)", left + width - 38, top + 14);
  }

  function drawAngles() {
    const ctx = elements.angleCanvas.getContext("2d");
    normalizeCanvas(elements.angleCanvas, ctx);
    const rect = elements.angleCanvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#fbfaf8";
    ctx.fillRect(0, 0, rect.width, rect.height);
    drawAnglePanel(ctx);
  }

  function updateStatus() {
    const pose = DATA.pose;
    elements.status.textContent =
      `Cycle ${pose.cyclePos.toFixed(2)} | Knee ${pose.kneeDeg.toFixed(1)} deg | ` +
      `Foot height ${pose.footHeight.toFixed(3)} m`;
  }

  function render() {
    DATA.pose = computePose(cycle, DATA.geom);
    drawChairView();
    drawAngles();
    updateStatus();
  }

  function updateLabels() {
    elements.humanHeightVal.textContent = STATE.humanHeight.toFixed(2);
    elements.kneeMaxVal.textContent = `${STATE.kneeMaxDeg.toFixed(0)} deg`;
    elements.seatHeightVal.textContent = STATE.seatHeight.toFixed(2);
    elements.speedVal.textContent = STATE.speed.toFixed(1);
    elements.cycleVal.textContent = cycle.toFixed(2);
    elements.l1Val.textContent = STATE.l1.toFixed(3);
    elements.l2Val.textContent = STATE.l2.toFixed(3);
    elements.l1PctVal.textContent = `${((STATE.l1 / STATE.humanHeight) * 100).toFixed(1)}%`;
    elements.l2PctVal.textContent = `${((STATE.l2 / STATE.humanHeight) * 100).toFixed(1)}%`;
  }

  function syncCycleInput() {
    elements.cycle.value = String(cycle);
  }

  function syncAnthropometryInputs() {
    elements.humanHeight.value = String(STATE.humanHeight);
    elements.humanHeightEdit.value = STATE.humanHeight.toFixed(2);
    elements.l1.value = String(STATE.l1);
    elements.l1Edit.value = STATE.l1.toFixed(3);
    elements.l2.value = String(STATE.l2);
    elements.l2Edit.value = STATE.l2.toFixed(3);
  }

  function clampToInputRange(input, value) {
    const min = Number(input.min);
    const max = Number(input.max);
    return Math.max(min, Math.min(max, value));
  }

  function resetDefaults() {
    Object.assign(STATE, DEFAULTS);
    cycle = DEFAULTS.cycle;
    syncAnthropometryInputs();
    elements.kneeMax.value = String(STATE.kneeMaxDeg);
    elements.seatHeight.value = String(STATE.seatHeight);
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
    syncSegmentsFromHeight(Number(elements.humanHeight.value));
    syncAnthropometryInputs();
    handleParamChange();
  });

  elements.humanHeightEdit.addEventListener("change", () => {
    syncSegmentsFromHeight(clampToInputRange(elements.humanHeightEdit, Number(elements.humanHeightEdit.value)));
    syncAnthropometryInputs();
    handleParamChange();
  });

  elements.l1.addEventListener("input", () => {
    STATE.l1 = Number(elements.l1.value);
    syncHeightFromSegments();
    syncAnthropometryInputs();
    handleParamChange();
  });

  elements.l1Edit.addEventListener("change", () => {
    STATE.l1 = clampToInputRange(elements.l1Edit, Number(elements.l1Edit.value));
    syncHeightFromSegments();
    syncAnthropometryInputs();
    handleParamChange();
  });

  elements.l2.addEventListener("input", () => {
    STATE.l2 = Number(elements.l2.value);
    syncHeightFromSegments();
    syncAnthropometryInputs();
    handleParamChange();
  });

  elements.l2Edit.addEventListener("change", () => {
    STATE.l2 = clampToInputRange(elements.l2Edit, Number(elements.l2Edit.value));
    syncHeightFromSegments();
    syncAnthropometryInputs();
    handleParamChange();
  });

  elements.kneeMax.addEventListener("input", () => {
    STATE.kneeMaxDeg = Number(elements.kneeMax.value);
    handleParamChange();
  });

  elements.seatHeight.addEventListener("input", () => {
    STATE.seatHeight = Number(elements.seatHeight.value);
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
