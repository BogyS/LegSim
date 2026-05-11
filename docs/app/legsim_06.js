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
  const BED_MARGIN = 0.002;
  const HUMAN_HEIGHT_MIN = 1.18;
  const HUMAN_HEIGHT_MAX = 2.2;
  const WORLD_MIN_X = -0.15;
  const WORLD_MAX_X = 1.2;

  const DEFAULTS = {
    humanHeight: 1.8,
    l1: BASE_L1,
    l2: BASE_L2,
    hipMaxDeg: 35,
    kneeFactor: -2,
    speed: 1.0,
    cycle: 0.0,
  };

  const elements = {
    bedCanvas: document.getElementById("sim-bed-canvas"),
    angleCanvas: document.getElementById("sim-angle-canvas"),
    pauseBtn: document.getElementById("sim-pause"),
    resetBtn: document.getElementById("sim-reset"),
    humanHeight: document.getElementById("sim-human-height"),
    humanHeightEdit: document.getElementById("sim-human-height-edit"),
    l1: document.getElementById("sim-l1"),
    l1Edit: document.getElementById("sim-l1-edit"),
    l2: document.getElementById("sim-l2"),
    l2Edit: document.getElementById("sim-l2-edit"),
    hipMax: document.getElementById("sim-hip-max"),
    kneeFactor: document.getElementById("sim-knee-factor"),
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
      groundMargin: BED_MARGIN * scale,
    };
  }

  function computeAngles(cyclePos) {
    const e = cycleEnvelope(cyclePos);
    const hip = STATE.hipMaxDeg * e;
    const knee = hip * STATE.kneeFactor;
    return { hip, knee };
  }

  function computeRelativePose(cyclePos, geom) {
    const angles = computeAngles(cyclePos);
    const hipRad = angles.hip * (Math.PI / 180);
    const kneeRad = angles.knee * (Math.PI / 180);
    const thighTheta = hipRad;
    const shankTheta = hipRad + kneeRad;
    const footTheta = shankTheta + (Math.PI * 0.5);
    const hipX = 0.08;
    const hipYRel = 0;
    const kneeX = hipX + geom.l1 * Math.cos(thighTheta);
    const kneeYRel = hipYRel + geom.l1 * Math.sin(thighTheta);
    const ankleX = kneeX + geom.l2 * Math.cos(shankTheta);
    const ankleYRel = kneeYRel + geom.l2 * Math.sin(shankTheta);
    const heelX = ankleX - geom.heelBack * Math.cos(footTheta);
    const heelYRel = ankleYRel - geom.heelBack * Math.sin(footTheta);
    const toeX = ankleX + geom.toeFwd * Math.cos(footTheta);
    const toeYRel = ankleYRel + geom.toeFwd * Math.sin(footTheta);

    return {
      cyclePos,
      hipDeg: angles.hip,
      kneeDeg: angles.knee,
      footDeg: footTheta * (180 / Math.PI),
      hipX,
      hipYRel,
      kneeX,
      kneeYRel,
      ankleX,
      ankleYRel,
      heelX,
      heelYRel,
      toeX,
      toeYRel,
    };
  }

  function poseMinRelativeY(pose) {
    return Math.min(pose.hipYRel, pose.kneeYRel, pose.ankleYRel, pose.heelYRel, pose.toeYRel);
  }

  function withFixedHipHeight(relativePose, hipY) {
    const kneeY = relativePose.kneeYRel + hipY;
    const ankleY = relativePose.ankleYRel + hipY;
    const heelY = relativePose.heelYRel + hipY;
    const toeY = relativePose.toeYRel + hipY;
    const clearance = Math.min(hipY, kneeY, ankleY, heelY, toeY);

    return {
      cyclePos: relativePose.cyclePos,
      hipDeg: relativePose.hipDeg,
      kneeDeg: relativePose.kneeDeg,
      footDeg: relativePose.footDeg,
      hipX: relativePose.hipX,
      hipY,
      kneeX: relativePose.kneeX,
      kneeY,
      ankleX: relativePose.ankleX,
      ankleY,
      heelX: relativePose.heelX,
      heelY,
      toeX: relativePose.toeX,
      toeY,
      clearance,
    };
  }

  function recomputeAll() {
    const geom = buildGeometry();
    const hipSeries = new Array(N);
    const kneeSeries = new Array(N);
    const relativePoses = new Array(N);
    let fixedHipY = geom.groundMargin;
    for (let i = 0; i < N; i += 1) {
      const pose = computeRelativePose(T_ARR[i], geom);
      relativePoses[i] = pose;
      hipSeries[i] = pose.hipDeg;
      kneeSeries[i] = pose.kneeDeg;
      fixedHipY = Math.max(fixedHipY, geom.groundMargin - poseMinRelativeY(pose));
    }

    DATA.geom = geom;
    DATA.hipSeries = hipSeries;
    DATA.kneeSeries = kneeSeries;
    DATA.fixedHipY = fixedHipY;
    DATA.pose = withFixedHipHeight(computeRelativePose(cycle, geom), fixedHipY);
    DATA.minY = -0.03;
    DATA.maxY = Math.max(fixedHipY + geom.l1 + geom.l2 + geom.toeFwd + 0.12, 0.75);
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

  function drawAnglePanel(ctx, data, panel, color, label) {
    const pad = 24;
    const left = pad + 18;
    const width = panel.width - left - pad;
    const axis = panel.index === 0
      ? { lo: -10, hi: 90, ticks: [-10, 10, 30, 50, 70, 90] }
      : { lo: -150, hi: 10, ticks: [10, -10, -30, -50, -70, -90, -110, -130, -150] };
    const top = panel.top;
    const height = panel.height;
    const lo = axis.lo;
    const hi = axis.hi;

    const mapX = (i) => left + (i / (N - 1)) * width;
    const mapY = (v) => top + height - ((v - lo) / (hi - lo)) * height;

    ctx.strokeStyle = "#d9cbb7";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(left, top, width, height);
    ctx.stroke();

    ctx.strokeStyle = "#d9cbb7";
    ctx.lineWidth = 1;
    for (const deg of axis.ticks) {
      const gy = mapY(deg);
      ctx.beginPath();
      ctx.moveTo(left, gy);
      ctx.lineTo(left + width, gy);
      ctx.stroke();
      ctx.fillStyle = "#5c5c5c";
      ctx.font = "11px Manrope, sans-serif";
      ctx.fillText(`${deg}`, 4, gy + 4);
    }

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
    ctx.fillText("(deg)", left + width - 38, top + 14);
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

    const pad = 24;
    const gap = 12;
    const availableHeight = height - (pad * 2) - gap;
    const hipRange = 100;
    const kneeRange = 160;
    const totalRange = hipRange + kneeRange;
    const hipHeight = (availableHeight * hipRange / totalRange);
    const kneeHeight = (availableHeight * kneeRange / totalRange);
    const hipPanel = { width, index: 0, top: pad, height: hipHeight };
    const kneePanel = { width, index: 1, top: pad + hipHeight + gap, height: kneeHeight };

    drawAnglePanel(ctx, DATA.hipSeries, hipPanel, "#1564a6", "Hip flexion (deg)");
    drawAnglePanel(ctx, DATA.kneeSeries, kneePanel, "#c06030", "Knee flexion (deg)");
  }

  function updateStatus() {
    const pose = DATA.pose;
    elements.status.textContent =
      `Cycle ${pose.cyclePos.toFixed(2)} | Hip ${pose.hipDeg.toFixed(1)} deg | ` +
      `Knee ${pose.kneeDeg.toFixed(1)} deg | Clearance ${pose.clearance.toFixed(3)} m`;
  }

  function render() {
    DATA.pose = withFixedHipHeight(computeRelativePose(cycle, DATA.geom), DATA.fixedHipY);
    drawBedView();
    drawAngles();
    updateStatus();
  }

  function updateLabels() {
    elements.humanHeightVal.textContent = STATE.humanHeight.toFixed(2);
    elements.hipMaxVal.textContent = `${STATE.hipMaxDeg.toFixed(0)} deg`;
    elements.kneeMaxVal.textContent = `${(STATE.hipMaxDeg * STATE.kneeFactor).toFixed(0)} deg max`;
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
    elements.hipMax.value = String(STATE.hipMaxDeg);
    elements.kneeFactor.value = String(STATE.kneeFactor);
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

  elements.hipMax.addEventListener("input", () => {
    STATE.hipMaxDeg = Number(elements.hipMax.value);
    handleParamChange();
  });

  elements.speed.addEventListener("input", () => {
    STATE.speed = Number(elements.speed.value);
    updateLabels();
  });

  elements.kneeFactor.addEventListener("change", () => {
    STATE.kneeFactor = Number(elements.kneeFactor.value);
    handleParamChange();
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
