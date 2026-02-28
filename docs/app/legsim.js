(() => {
  "use strict";

  const WALK_FPS = 60;
  const TRAVEL_M = 1.2;

  const L1 = 0.45;
  const L2 = 0.43;
  const FOOT_TOTAL = 0.265;
  const HEEL_BACK = 0.06;
  const TOE_FWD = FOOT_TOTAL - HEEL_BACK;
  const TORSO_LEN = 0.55;

  const HIP_OFFSET = -Math.PI / 2;
  const KNEE_OFFSET = 0.0;
  const ANKLE_OFFSET = Math.PI / 2;

  const elements = {
    dataset: document.getElementById("dataset"),
    loadDataset: document.getElementById("load-dataset"),
    fileInput: document.getElementById("file-input"),
    loadFile: document.getElementById("load-file"),
    pauseBtn: document.getElementById("pause-btn"),
    resetBtn: document.getElementById("reset-btn"),
    flipHip: document.getElementById("flip-hip"),
    flipKnee: document.getElementById("flip-knee"),
    flipAnkle: document.getElementById("flip-ankle"),
    status: document.getElementById("status"),
    walkCanvas: document.getElementById("walk-canvas"),
    angleCanvas: document.getElementById("angle-canvas"),
  };

  const state = {
    t: null,
    hipW: null,
    kneeW: null,
    ankleW: null,
    theta1: null,
    theta2: null,
    theta3: null,
    q1: null,
    q2: null,
    q3: null,
    q1c: null,
    q2c: null,
    q3c: null,
    paused: false,
    frame: 0,
    frameIndices: null,
    sign: {
      hip: 1.0,
      knee: 1.0,
      ankle: 1.0,
    },
    warning: "",
  };

  function setStatus(message, tone = "normal") {
    elements.status.textContent = message;
    elements.status.dataset.tone = tone;
  }

  function parseNumericTokens(line) {
    const parts = line.trim().split(/[,\t;]+/g);
    const nums = [];
    for (const part of parts) {
      if (!part) {
        continue;
      }
      const token = part.trim();
      if (/^[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?$/.test(token)) {
        nums.push(Number(token));
      }
    }
    return nums;
  }

  function loadCsvText(text) {
    const rows = [];
    const lines = text.split(/\r?\n/);
    for (const line of lines) {
      const nums = parseNumericTokens(line);
      if (nums.length >= 6) {
        rows.push(nums.slice(0, 6));
      }
    }
    if (!rows.length) {
      throw new Error("No numeric rows found with at least 6 values.");
    }
    const columns = rows[0].map((_, i) => rows.map((row) => row[i]));
    return {
      hipT: columns[0],
      hipW: columns[1],
      kneeT: columns[2],
      kneeW: columns[3],
      ankleT: columns[4],
      ankleW: columns[5],
    };
  }

  function dedupeAndSort(t, y) {
    const pairs = t.map((ti, i) => [ti, y[i]]);
    pairs.sort((a, b) => a[0] - b[0]);
    const t2 = [];
    const y2 = [];
    for (let i = 0; i < pairs.length; i += 1) {
      if (i > 0 && pairs[i][0] === pairs[i - 1][0]) {
        continue;
      }
      t2.push(pairs[i][0]);
      y2.push(pairs[i][1]);
    }
    return { t: t2, y: y2 };
  }

  function interp1(x, xp, fp) {
    const out = new Array(x.length);
    let j = 0;
    for (let i = 0; i < x.length; i += 1) {
      const xi = x[i];
      while (j < xp.length - 2 && xp[j + 1] < xi) {
        j += 1;
      }
      const x0 = xp[j];
      const x1 = xp[j + 1];
      const y0 = fp[j];
      const y1 = fp[j + 1];
      const t = x1 === x0 ? 0 : (xi - x0) / (x1 - x0);
      out[i] = y0 + t * (y1 - y0);
    }
    return out;
  }

  function resampleToCommonTime(hipT, hipW, kneeT, kneeW, ankleT, ankleW) {
    const hip = dedupeAndSort(hipT, hipW);
    const knee = dedupeAndSort(kneeT, kneeW);
    const ankle = dedupeAndSort(ankleT, ankleW);
    const t = hip.t;
    const kneeWInterp = interp1(t, knee.t, knee.y);
    const ankleWInterp = interp1(t, ankle.t, ankle.y);
    return { t, hipW: hip.y, kneeW: kneeWInterp, ankleW: ankleWInterp };
  }

  function integrateVelocity(t, w) {
    const theta = new Array(w.length).fill(0);
    for (let i = 1; i < w.length; i += 1) {
      const dt = t[i] - t[i - 1];
      theta[i] = theta[i - 1] + 0.5 * (w[i] + w[i - 1]) * dt;
    }
    return theta;
  }

  function ankleVelocityLooksInvalid(t, ankleW) {
    if (t.length < 10) {
      return { invalid: false, reason: "Not enough samples to validate ankle." };
    }
    const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const meanT = mean(t);
    const meanW = mean(ankleW);
    const tw = t.map((v) => v - meanT);
    const aw = ankleW.map((v) => v - meanW);
    const dot = tw.reduce((acc, v, i) => acc + v * aw[i], 0);
    const normT = Math.sqrt(tw.reduce((acc, v) => acc + v * v, 0));
    const normW = Math.sqrt(aw.reduce((acc, v) => acc + v * v, 0));
    const corr = dot / (normT * normW + 1e-12);
    const rmse = Math.sqrt(
      ankleW.reduce((acc, v, i) => acc + (v - t[i]) ** 2, 0) / ankleW.length
    );
    const range = Math.max(...t) - Math.min(...t) + 1e-12;
    const nrmse = rmse / range;

    if (corr > 0.999 && nrmse < 0.02) {
      return { invalid: true, reason: "Ankle velocity appears to match time (likely wrong column)." };
    }

    const maxAbs = Math.max(...ankleW.map((v) => Math.abs(v)));
    const meanAbs = mean(ankleW.map((v) => Math.abs(v)));
    if (maxAbs > 15.0 || meanAbs > 6.0) {
      return {
        invalid: true,
        reason: `Ankle velocity magnitude looks unrealistic (max|ω|=${maxAbs.toFixed(2)} rad/s).`,
      };
    }
    return { invalid: false, reason: "Ankle data looks plausible." };
  }

  function recomputeAngles() {
    const { t, hipW, kneeW, ankleW } = state;
    if (!t) {
      return;
    }
    const theta1 = integrateVelocity(t, hipW);
    const theta2 = integrateVelocity(t, kneeW);
    const theta3 = integrateVelocity(t, ankleW);
    state.theta1 = theta1;
    state.theta2 = theta2;
    state.theta3 = theta3;

    state.q1 = theta1.map((v) => HIP_OFFSET + state.sign.hip * v);
    state.q2 = theta2.map((v) => KNEE_OFFSET + state.sign.knee * v);
    state.q3 = theta3.map((v) => ANKLE_OFFSET + state.sign.ankle * v);

    const mean = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
    const q1m = mean(state.q1);
    const q2m = mean(state.q2);
    const q3m = mean(state.q3);
    state.q1c = state.q1.map((v) => (v - q1m) * (180 / Math.PI));
    state.q2c = state.q2.map((v) => (v - q2m) * (180 / Math.PI));
    state.q3c = state.q3.map((v) => (v - q3m) * (180 / Math.PI));
  }

  function buildFrameIndices() {
    if (!state.t) {
      state.frameIndices = null;
      return;
    }
    const t = state.t;
    const diffs = t.slice(1).map((v, i) => v - t[i]);
    const baseDt = diffs.length ? diffs.sort((a, b) => a - b)[Math.floor(diffs.length / 2)] : 1 / WALK_FPS;
    const step = Math.max(1, Math.round((1 / WALK_FPS) / Math.max(baseDt, 1e-9)));
    state.frameIndices = [];
    for (let i = 0; i < t.length; i += step) {
      state.frameIndices.push(i);
    }
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

    ctx.strokeStyle = "#d9cbb7";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(20, h - 60);
    ctx.lineTo(w - 20, h - 60);
    ctx.stroke();

    if (!state.t || !state.frameIndices || !state.q1) {
      ctx.fillStyle = "#9a8f82";
      ctx.font = "16px Manrope, sans-serif";
      ctx.fillText("Load data to see motion.", 24, 40);
      return;
    }

    const idx = state.frameIndices[state.frame % state.frameIndices.length];
    const t = state.t;
    const q1 = state.q1;
    const q2 = state.q2;
    const q3 = state.q3;

    const duration = t[t.length - 1] - t[0] || 1;
    const hx = ((t[idx] - t[0]) / duration) * TRAVEL_M;
    const hy = 0.92;

    const kx = hx + L1 * Math.cos(q1[idx]);
    const ky = hy + L1 * Math.sin(q1[idx]);

    const ax = kx + L2 * Math.cos(q1[idx] + q2[idx]);
    const ay = ky + L2 * Math.sin(q1[idx] + q2[idx]);

    const footAng = q1[idx] + q2[idx] + q3[idx];
    const tx = ax + TOE_FWD * Math.cos(footAng);
    const ty = ay + TOE_FWD * Math.sin(footAng);
    const hx2 = ax - HEEL_BACK * Math.cos(footAng);
    const hy2 = ay - HEEL_BACK * Math.sin(footAng);

    const worldMinX = -0.2;
    const worldMaxX = TRAVEL_M + 0.6;
    const worldMinY = -0.05;
    const worldMaxY = 0.92 + TORSO_LEN + 0.25;
    const scaleX = w / (worldMaxX - worldMinX);
    const scaleY = h / (worldMaxY - worldMinY);

    const mapX = (x) => (x - worldMinX) * scaleX;
    const mapY = (y) => h - (y - worldMinY) * scaleY;

    ctx.lineWidth = 4;
    ctx.strokeStyle = "#1564a6";
    ctx.beginPath();
    ctx.moveTo(mapX(hx), mapY(hy));
    ctx.lineTo(mapX(kx), mapY(ky));
    ctx.lineTo(mapX(ax), mapY(ay));
    ctx.stroke();

    ctx.strokeStyle = "#0f4a7a";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(mapX(hx2), mapY(hy2));
    ctx.lineTo(mapX(tx), mapY(ty));
    ctx.stroke();

    ctx.fillStyle = "#1a1a1a";
    const dot = (x, y) => {
      ctx.beginPath();
      ctx.arc(mapX(x), mapY(y), 4, 0, Math.PI * 2);
      ctx.fill();
    };
    dot(hx, hy);
    dot(kx, ky);
    dot(ax, ay);

    ctx.strokeStyle = "#1a1a1a";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(mapX(hx), mapY(hy));
    ctx.lineTo(mapX(hx), mapY(hy + TORSO_LEN));
    ctx.stroke();
  }

  function drawAngles() {
    const ctx = elements.angleCanvas.getContext("2d");
    normalizeCanvas(elements.angleCanvas, ctx);
    const rect = elements.angleCanvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#fbfaf8";
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = "#d9cbb7";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(10, 10, w - 20, h - 20);
    ctx.stroke();

    if (!state.t || !state.q1c) {
      ctx.fillStyle = "#9a8f82";
      ctx.font = "16px Manrope, sans-serif";
      ctx.fillText("Angles will appear here.", 24, 40);
      return;
    }

    const pad = 30;
    const plotW = w - pad * 2;
    const plotH = h - pad * 2;

    const yOffsets = [40, 0, -40];
    const maxRange = 90;

    const mapX = (i) => pad + (i / (state.t.length - 1)) * plotW;
    const mapY = (val, offset) => pad + plotH / 2 - ((val + offset) / (maxRange * 2)) * plotH;

    ctx.strokeStyle = "#b7a58e";
    ctx.setLineDash([6, 6]);
    yOffsets.forEach((offset) => {
      const y = mapY(0, offset);
      ctx.beginPath();
      ctx.moveTo(pad, y);
      ctx.lineTo(pad + plotW, y);
      ctx.stroke();
    });
    ctx.setLineDash([]);

    const drawLine = (data, color, offset) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      data.forEach((v, i) => {
        const x = mapX(i);
        const y = mapY(v, offset);
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();
    };

    drawLine(state.q1c, "#1564a6", yOffsets[0]);
    drawLine(state.q2c, "#c06030", yOffsets[1]);
    drawLine(state.q3c, "#2b7a4b", yOffsets[2]);

    const idx = state.frameIndices[state.frame % state.frameIndices.length];
    const cursorX = mapX(idx);
    ctx.strokeStyle = "#1a1a1a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cursorX, pad);
    ctx.lineTo(cursorX, pad + plotH);
    ctx.stroke();
  }

  function render() {
    drawWalk();
    drawAngles();
  }

  function animate() {
    if (!state.paused && state.frameIndices) {
      state.frame = (state.frame + 1) % state.frameIndices.length;
    }
    render();
    window.requestAnimationFrame(animate);
  }

  async function loadDataset(path) {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error("Failed to load dataset.");
    }
    const text = await response.text();
    return loadCsvText(text);
  }

  function applyData(data, label) {
    const resampled = resampleToCommonTime(
      data.hipT,
      data.hipW,
      data.kneeT,
      data.kneeW,
      data.ankleT,
      data.ankleW
    );
    state.t = resampled.t;
    state.hipW = resampled.hipW;
    state.kneeW = resampled.kneeW;
    state.ankleW = resampled.ankleW;

    const validation = ankleVelocityLooksInvalid(state.t, state.ankleW);
    state.warning = validation.reason;
    if (validation.invalid) {
      state.ankleW = state.ankleW.map(() => 0);
    }

    recomputeAngles();
    buildFrameIndices();
    state.frame = 0;
    setStatus(`${label}. ${validation.invalid ? "Warning: " + validation.reason : validation.reason}`);
  }

  elements.loadDataset.addEventListener("click", async () => {
    const path = elements.dataset.value;
    if (!path) {
      setStatus("Please choose a dataset first.");
      return;
    }
    try {
      setStatus("Loading dataset...");
      const data = await loadDataset(path);
      applyData(data, `Loaded dataset: ${path.split("/").pop()}`);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  });

  elements.loadFile.addEventListener("click", async () => {
    const file = elements.fileInput.files[0];
    if (!file) {
      setStatus("Pick a local file first.");
      return;
    }
    try {
      const text = await file.text();
      const data = loadCsvText(text);
      applyData(data, `Loaded file: ${file.name}`);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  });

  elements.pauseBtn.addEventListener("click", () => {
    state.paused = !state.paused;
    elements.pauseBtn.textContent = state.paused ? "Resume" : "Pause";
  });

  elements.resetBtn.addEventListener("click", () => {
    state.frame = 0;
    state.paused = false;
    elements.pauseBtn.textContent = "Pause";
    render();
  });

  elements.flipHip.addEventListener("change", () => {
    state.sign.hip = elements.flipHip.checked ? -1.0 : 1.0;
    recomputeAngles();
  });
  elements.flipKnee.addEventListener("change", () => {
    state.sign.knee = elements.flipKnee.checked ? -1.0 : 1.0;
    recomputeAngles();
  });
  elements.flipAnkle.addEventListener("change", () => {
    state.sign.ankle = elements.flipAnkle.checked ? -1.0 : 1.0;
    recomputeAngles();
  });

  window.addEventListener("resize", render);

  render();
  animate();
})();
