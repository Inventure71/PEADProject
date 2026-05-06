const canvas = document.getElementById("demoCanvas");
const ctx = canvas.getContext("2d");

const loadingOverlay = document.getElementById("loadingOverlay");
const playgroundPanel = document.getElementById("playgroundPanel");
const sceneTitle = document.getElementById("sceneTitle");
const sceneKicker = document.getElementById("sceneKicker");
const sceneSummary = document.getElementById("sceneSummary");
const modeSwitch = document.getElementById("modeSwitch");
const controlStrip = document.querySelector(".control-strip");
const stepSlider = document.getElementById("stepSlider");
const stepReadout = document.getElementById("stepReadout");
const playButton = document.getElementById("playButton");
const prevButton = document.getElementById("prevButton");
const nextButton = document.getElementById("nextButton");
const resetButton = document.getElementById("resetButton");
const stepTitle = document.getElementById("stepTitle");
const stepNarrative = document.getElementById("stepNarrative");
const formulaBox = document.getElementById("formulaBox");
const valueGrid = document.getElementById("valueGrid");

const colors = {
  ink: "#20201d",
  muted: "#68645c",
  hairline: "#ddd4c2",
  teal: "#157a72",
  tealSoft: "#d9f0ec",
  berry: "#b53b5d",
  amber: "#c5872e",
  violet: "#5b5aa6",
  paper: "#fffaf0",
  grid: "rgba(32, 32, 29, 0.08)",
  blue: "#2176d2",
};

const state = {
  scene: "fruit",
  fruitMode: "hebbian",
  stepIndex: 0,
  playing: false,
  playTimer: null,
  data: null,
  width: 0,
  height: 0,
  playgroundInput: null,
  selectedPresetIndex: 0,
};

const sceneCopy = {
  fruit: {
    kicker: "Local association vs prediction error",
    title: "Fruit Weights",
    summary:
      "A small neuron diagram shows how active fruit features change class connections under Hebbian learning and backpropagation.",
  },
  oja: {
    kicker: "Normalized Hebbian learning",
    title: "Oja's Rule as Online PCA",
    summary:
      "Pure Hebbian and Oja learn almost the same direction, so the important difference is stability: Oja keeps the weight size bounded while pure Hebbian explodes.",
  },
  forgetting: {
    kicker: "Sequential backpropagation vs EWC",
    title: "Catastrophic Forgetting",
    summary:
      "EWC means Elastic Weight Consolidation: standard backprop trains only the current task, while EWC adds a memory penalty for weights that mattered to the old task.",
  },
  playground: {
    kicker: "Verification playground",
    title: "Compare Model Performance",
    summary:
      "Toggle fruit features and compare what the trained Hebbian and Backprop models predict from the same input vector.",
  },
};

function formatNumber(value, digits = 3) {
  if (typeof value !== "number") return value;
  return Number(value).toFixed(digits).replace(/\.?0+$/, "");
}

function formatPercent(value, digits = 1) {
  return `${formatNumber(value * 100, digits)}%`;
}

function formatMagnitude(value) {
  if (typeof value !== "number") return value;
  const absValue = Math.abs(value);
  if (absValue >= 10000 || (absValue > 0 && absValue < 0.001)) {
    return value.toExponential(2).replace("e+", "e");
  }
  return formatNumber(value, 3);
}

function forgettingTakeaway(payload) {
  const standard = payload.summary.standard;
  const ewc = payload.summary.ewc;
  return `Standard backprop trains only the new task (${formatPercent(
    standard.task_b_after_b,
  )}) and old-task accuracy falls to ${formatPercent(standard.task_a_after_b)}. EWC adds a memory penalty, preserving more old knowledge (${formatPercent(
    ewc.task_a_after_b,
  )}) while still learning the new task (${formatPercent(ewc.task_b_after_b)}).`;
}

function ojaTakeaway(payload) {
  const initialAngle = payload.steps?.[0]?.angle_degrees;
  return `This is the teacher's Hebbian-network requirement: one Oja neuron learns from unlabeled points one at a time. The angle to PCA starts at ${formatNumber(
    initialAngle,
    1,
  )} degrees and ends at ${formatNumber(payload.final_angle_degrees, 2)} degrees, while pure Hebbian weight size grows to ${formatMagnitude(
    payload.pure_hebbian_final_norm,
  )}.`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function currentSteps() {
  if (!state.data) return [];
  if (state.scene === "fruit") {
    return fruitDisplaySteps();
  }
  if (state.scene === "oja") return state.data.oja.steps;
  if (state.scene === "playground") return [];
  return state.data.forgetting.steps;
}

function fruitDisplaySteps() {
  const sourceSteps = state.data.fruit[state.fruitMode].steps;
  const repeatTotals = new Map();
  const repeatSeen = new Map();

  sourceSteps.forEach((step) => {
    const key = fruitRepeatKey(step);
    repeatTotals.set(key, (repeatTotals.get(key) || 0) + 1);
  });

  return sourceSteps.flatMap((step, rawIndex) => {
    const key = fruitRepeatKey(step);
    const repeatIndex = (repeatSeen.get(key) || 0) + 1;
    repeatSeen.set(key, repeatIndex);
    const repeatTotal = repeatTotals.get(key) || 1;
    const enrichedStep = {
      ...step,
      raw_step_index: rawIndex,
      repeat_index: repeatIndex,
      repeat_total: repeatTotal,
      repeat_label: repeatTotal > 1 ? `repeat ${repeatIndex}/${repeatTotal}` : "unique example",
    };

    return [
    {
      ...enrichedStep,
      visual_phase: "pass",
    },
    {
      ...enrichedStep,
      visual_phase: "update",
    },
    ];
  });
}

function fruitRepeatKey(step) {
  return `${step.epoch}:${step.label}:${step.input.join(",")}`;
}

function currentStep() {
  const steps = currentSteps();
  if (steps.length === 0) return null;
  return steps[clamp(state.stepIndex, 0, steps.length - 1)];
}

function setScene(scene) {
  state.scene = scene;
  state.stepIndex = 0;
  stopPlayback();
  if (scene === "playground") ensurePlaygroundInput();

  document.querySelectorAll(".scene-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.scene === scene);
  });

  modeSwitch.classList.toggle("hidden", scene !== "fruit");
  controlStrip.classList.toggle("hidden", scene === "playground");
  updateChrome();
  render();
}

function setFruitMode(mode) {
  state.fruitMode = mode;
  state.stepIndex = 0;
  document.querySelectorAll(".mode-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  updateChrome();
  render();
}

function updateChrome() {
  const copy = sceneCopy[state.scene];
  sceneTitle.textContent = copy.title;
  sceneKicker.textContent = copy.kicker;
  if (state.scene === "forgetting" && state.data?.forgetting) {
    sceneSummary.textContent = forgettingTakeaway(state.data.forgetting);
  } else if (state.scene === "oja" && state.data?.oja) {
    sceneSummary.textContent = ojaTakeaway(state.data.oja);
  } else {
    sceneSummary.textContent = copy.summary;
  }
  controlStrip.classList.toggle("hidden", state.scene === "playground");

  const steps = currentSteps();
  stepSlider.max = Math.max(0, steps.length - 1);
  stepSlider.value = state.stepIndex;
  if (state.scene === "fruit" && steps.length) {
    const step = steps[clamp(state.stepIndex, 0, steps.length - 1)];
    const rawCount = state.data.fruit[state.fruitMode].steps.length;
    stepReadout.textContent = `sample ${step.raw_step_index + 1}/${rawCount} · ${step.visual_phase}`;
  } else {
    stepReadout.textContent = steps.length ? `${state.stepIndex + 1} / ${steps.length}` : "0 / 0";
  }
}

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  state.width = Math.max(320, rect.width);
  state.height = Math.max(320, rect.height);
  canvas.width = Math.floor(state.width * dpr);
  canvas.height = Math.floor(state.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  render();
}

function clearCanvas() {
  ctx.clearRect(0, 0, state.width, state.height);
}

function drawPanelTitle(title, subtitle) {
  ctx.save();
  ctx.fillStyle = colors.ink;
  ctx.font = "700 18px Inter, system-ui, sans-serif";
  ctx.fillText(fitText(title, state.width - 56), 28, 34);
  if (subtitle) {
    ctx.fillStyle = colors.muted;
    ctx.font = "13px Inter, system-ui, sans-serif";
    ctx.fillText(fitText(subtitle, state.width - 56), 28, 56);
  }
  ctx.restore();
}

function drawNode(x, y, radius, label, options = {}) {
  const active = options.active || false;
  const fill = options.fill || (active ? colors.teal : "#ffffff");
  const stroke = options.stroke || (active ? colors.teal : colors.hairline);
  const textColor = options.textColor || (active ? "#ffffff" : colors.ink);

  ctx.save();
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = active ? 2.4 : 1.2;
  ctx.strokeStyle = stroke;
  ctx.stroke();
  ctx.fillStyle = textColor;
  ctx.font = "700 11px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x, y);
  ctx.restore();
}

function drawArrow(fromX, fromY, toX, toY, color, width = 3) {
  const angle = Math.atan2(toY - fromY, toX - fromX);
  const headLength = 12;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(
    toX - headLength * Math.cos(angle - Math.PI / 6),
    toY - headLength * Math.sin(angle - Math.PI / 6),
  );
  ctx.lineTo(
    toX - headLength * Math.cos(angle + Math.PI / 6),
    toY - headLength * Math.sin(angle + Math.PI / 6),
  );
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function weightColor(weight) {
  if (weight > 0.001) return colors.teal;
  if (weight < -0.001) return colors.berry;
  return "rgba(104, 100, 92, 0.22)";
}

function deltaColor(delta) {
  if (delta > 0.0001) return colors.teal;
  if (delta < -0.0001) return colors.berry;
  return "rgba(104, 100, 92, 0.24)";
}

function shortDelta(value) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${formatNumber(value, 3)}`;
}

function drawBadge(x, y, text, fill, options = {}) {
  const paddingX = options.paddingX || 8;
  const height = options.height || 22;

  ctx.save();
  ctx.font = options.font || "700 12px Inter, system-ui, sans-serif";
  const width = ctx.measureText(text).width + paddingX * 2;
  ctx.fillStyle = fill;
  ctx.strokeStyle = options.stroke || "rgba(255, 255, 255, 0.88)";
  ctx.lineWidth = 2;
  roundRect(x - width / 2, y - height / 2, width, height, height / 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = options.textColor || "#ffffff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y + 0.5);
  ctx.restore();
}

function fruitConnectionPath(feature, classNode) {
  ctx.beginPath();
  ctx.moveTo(feature.x + 26, feature.y);
  ctx.bezierCurveTo(
    lerp(feature.x, classNode.x, 0.38),
    feature.y,
    lerp(feature.x, classNode.x, 0.62),
    classNode.y,
    classNode.x - 26,
    classNode.y,
  );
}

function fruitConnectionPoint(feature, classNode, t) {
  const x0 = feature.x + 26;
  const y0 = feature.y;
  const x1 = lerp(feature.x, classNode.x, 0.38);
  const y1 = feature.y;
  const x2 = lerp(feature.x, classNode.x, 0.62);
  const y2 = classNode.y;
  const x3 = classNode.x - 26;
  const y3 = classNode.y;
  const mt = 1 - t;
  return {
    x: mt ** 3 * x0 + 3 * mt ** 2 * t * x1 + 3 * mt * t ** 2 * x2 + t ** 3 * x3,
    y: mt ** 3 * y0 + 3 * mt ** 2 * t * y1 + 3 * mt * t ** 2 * y2 + t ** 3 * y3,
  };
}

function drawFruit() {
  const fruit = state.data.fruit;
  const step = currentStep();
  if (!step) return;
  const isPassPhase = step.visual_phase === "pass";
  const modeName = state.fruitMode === "hebbian" ? "Hebbian" : "Backpropagation";
  const phaseTitle = isPassPhase ? `${modeName} pass-through` : `${modeName} weight update`;

  drawPanelTitle(
    phaseTitle,
    `Epoch ${step.epoch}, example ${step.sample_index}: ${step.label} (${step.repeat_label})`,
  );

  const leftX = state.width * 0.22;
  const rightX = state.width * 0.76;
  const top = 150;
  const bottom = state.height - 76;
  const featureGap = (bottom - top) / (fruit.features.length - 1);
  const classGap = (bottom - top) / (fruit.classes.length - 1);

  const featurePositions = fruit.features.map((feature, index) => ({
    label: feature,
    x: leftX,
    y: top + index * featureGap,
  }));
  const classPositions = fruit.classes.map((className, index) => ({
    label: className,
    x: rightX,
    y: top + index * classGap,
  }));

  const weights = isPassPhase ? step.old_weights : step.new_weights;
  const oldWeights = step.old_weights;
  const delta = step.weight_delta;
  const activeFeatures = new Set(step.active_features);
  const activatedConnections = [];
  const changedConnections = [];

  featurePositions.forEach((feature, featureIndex) => {
    classPositions.forEach((classNode, classIndex) => {
      const weight = weights[featureIndex][classIndex];
      const activated =
        activeFeatures.has(feature.label) &&
        (state.fruitMode === "backprop" || classNode.label === step.label);
      const changed = Math.abs(delta[featureIndex][classIndex]) > 0.0001;
      if (activated) {
        activatedConnections.push({
          feature,
          classNode,
          weight,
        });
      }
      if (changed) {
        changedConnections.push({
          feature,
          classNode,
          delta: delta[featureIndex][classIndex],
          oldWeight: oldWeights[featureIndex][classIndex],
          newWeight: weight,
        });
      }
      ctx.save();
      ctx.globalAlpha = changed && !isPassPhase ? 0.2 : 0.34;
      ctx.strokeStyle = weightColor(weight);
      ctx.lineWidth = 0.9 + Math.min(4, Math.abs(weight) * 4);
      fruitConnectionPath(feature, classNode);
      ctx.stroke();
      ctx.restore();
    });
  });

  activatedConnections.forEach((connection) => {
    ctx.save();
    ctx.shadowColor = colors.blue;
    ctx.shadowBlur = isPassPhase ? 12 : 2;
    ctx.strokeStyle = colors.blue;
    ctx.lineWidth = (isPassPhase ? 3.2 : 1.5) + Math.min(3.2, Math.abs(connection.weight) * 3);
    ctx.globalAlpha = isPassPhase ? 0.72 : 0.16;
    ctx.setLineDash([7, 7]);
    ctx.lineCap = "round";
    fruitConnectionPath(connection.feature, connection.classNode);
    ctx.stroke();
    ctx.restore();
  });

  if (!isPassPhase) {
    changedConnections.forEach((connection) => {
      const color = deltaColor(connection.delta);
      ctx.save();
      ctx.shadowColor = color;
      ctx.shadowBlur = 16;
      ctx.strokeStyle = color;
      ctx.lineWidth = 4.2 + Math.min(6.5, Math.abs(connection.delta) * 26 + Math.abs(connection.newWeight) * 3);
      ctx.globalAlpha = 0.96;
      ctx.lineCap = "round";
      fruitConnectionPath(connection.feature, connection.classNode);
      ctx.stroke();
      ctx.restore();
    });

    labelledFruitConnections(changedConnections).forEach((connection, index) => {
      const color = deltaColor(connection.delta);
      const labelT =
        state.fruitMode === "hebbian"
          ? Math.min(0.56, 0.34 + index * 0.07)
          : 0.56 + (index % 2) * 0.1;
      const labelPoint = fruitConnectionPoint(connection.feature, connection.classNode, labelT);
      drawBadge(labelPoint.x, labelPoint.y, shortDelta(connection.delta), color);
    });
  }

  featurePositions.forEach((feature) => {
    drawNode(feature.x, feature.y, 25, feature.label.slice(0, 3), {
      active: activeFeatures.has(feature.label),
      fill: activeFeatures.has(feature.label) ? colors.teal : "#ffffff",
    });
    ctx.fillStyle = colors.muted;
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(feature.label, feature.x - 36, feature.y + 4);
  });

  classPositions.forEach((classNode) => {
    const active = classNode.label === step.label;
    drawNode(classNode.x, classNode.y, 30, classNode.label.slice(0, 3), {
      active,
      fill: active ? colors.amber : "#ffffff",
      stroke: active ? colors.amber : colors.hairline,
    });
    ctx.fillStyle = colors.muted;
    ctx.font = "12px Inter, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(classNode.label, classNode.x + 40, classNode.y + 4);
  });

  drawFruitStepRail(step, activatedConnections, changedConnections);
  drawFruitLegend();

  ctx.save();
  ctx.fillStyle = "rgba(255, 250, 240, 0.86)";
  ctx.strokeStyle = colors.hairline;
  roundRect(28, state.height - 58, state.width - 56, 34, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = colors.ink;
  ctx.font = "700 13px Inter, system-ui, sans-serif";
  ctx.fillText(`Active features: ${step.active_features.join(", ")}`, 44, state.height - 36);
  ctx.textAlign = "right";
  ctx.fillText(
    isPassPhase
      ? `Pass-through: ${activatedConnections.length} weights read`
      : `${step.repeat_label}; prediction: ${step.prediction}`,
    state.width - 44,
    state.height - 36,
  );
  ctx.restore();
}

function labelledFruitConnections(changedConnections) {
  if (state.fruitMode === "hebbian") return changedConnections;

  const positive = changedConnections
    .filter((connection) => connection.delta > 0)
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));
  const negative = changedConnections
    .filter((connection) => connection.delta < 0)
    .sort((a, b) => Math.abs(b.delta) - Math.abs(a.delta));

  return [...positive.slice(0, 2), ...negative.slice(0, 2)];
}

function drawFruitStepRail(step, activatedConnections, changedConnections) {
  const isPassPhase = step.visual_phase === "pass";
  const x = 28;
  const y = 70;
  const gap = 6;
  const railWidth = state.width - 56;
  const chipWidth = (railWidth - gap * 3) / 4;
  const firstDelta = changedConnections[0]?.delta || 0;
  const changeWord =
    state.fruitMode === "hebbian"
      ? `${changedConnections.length} rays strengthen`
      : `${changedConnections.filter((item) => item.delta > 0).length} up, ${changedConnections.filter((item) => item.delta < 0).length} down`;

  drawStepChip(x, y, chipWidth, "1", "input fires", colors.teal, isPassPhase);
  drawStepChip(
    x + chipWidth + gap,
    y,
    chipWidth,
    "2",
    `${activatedConnections.length} active rays`,
    colors.blue,
    isPassPhase,
  );
  drawStepChip(
    x + (chipWidth + gap) * 2,
    y,
    chipWidth,
    "3",
    state.fruitMode === "hebbian" ? `${step.label} fires` : "error is computed",
    state.fruitMode === "hebbian" ? colors.amber : colors.berry,
    !isPassPhase,
  );
  drawStepChip(
    x + (chipWidth + gap) * 3,
    y,
    chipWidth,
    "4",
    `${changedConnections.length} change, ${shortDelta(firstDelta)}`,
    state.fruitMode === "hebbian" ? colors.teal : colors.violet,
    !isPassPhase,
  );
}

function drawFruitLegend() {
  const x = 28;
  const y = 105;
  ctx.save();
  ctx.fillStyle = "rgba(255, 250, 240, 0.88)";
  ctx.strokeStyle = "rgba(221, 212, 194, 0.9)";
  roundRect(x, y, 350, 26, 8);
  ctx.fill();
  ctx.stroke();

  ctx.strokeStyle = colors.blue;
  ctx.setLineDash([6, 5]);
  ctx.lineWidth = 2.4;
  ctx.beginPath();
  ctx.moveTo(x + 12, y + 13);
  ctx.lineTo(x + 42, y + 13);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = colors.muted;
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.fillText("activated/read", x + 50, y + 17);

  ctx.strokeStyle = colors.teal;
  ctx.lineWidth = 3.4;
  ctx.beginPath();
  ctx.moveTo(x + 145, y + 13);
  ctx.lineTo(x + 175, y + 13);
  ctx.stroke();
  ctx.fillText("increase", x + 183, y + 17);

  ctx.strokeStyle = colors.berry;
  ctx.beginPath();
  ctx.moveTo(x + 250, y + 13);
  ctx.lineTo(x + 280, y + 13);
  ctx.stroke();
  ctx.fillText("decrease", x + 288, y + 17);
  ctx.restore();
}

function drawStepChip(x, y, width, number, text, accent, active = false) {
  ctx.save();
  ctx.fillStyle = active ? "rgba(255, 255, 255, 0.98)" : "rgba(255, 250, 240, 0.82)";
  ctx.strokeStyle = active ? accent : "rgba(221, 212, 194, 0.92)";
  ctx.lineWidth = active ? 1.8 : 1;
  roundRect(x, y, width, 28, 8);
  ctx.fill();
  ctx.stroke();
  drawBadge(x + 17, y + 14, number, accent, { height: 20, paddingX: 7, font: "800 11px Inter, system-ui, sans-serif" });
  ctx.fillStyle = colors.ink;
  ctx.font = "700 12px Inter, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(fitText(text, width - 44), x + 36, y + 14);
  ctx.restore();
}

function fitText(text, maxWidth) {
  if (ctx.measureText(text).width <= maxWidth) return text;
  let clipped = text;
  while (clipped.length > 4 && ctx.measureText(`${clipped}...`).width > maxWidth) {
    clipped = clipped.slice(0, -1);
  }
  return `${clipped.trim()}...`;
}

function plotMapper(points, box) {
  const maxAbs = Math.max(
    1,
    ...points.flat().map((value) => Math.abs(value)),
  );
  const scale = Math.min(box.width, box.height) * 0.42 / maxAbs;
  return {
    x: (value) => box.x + box.width / 2 + value * scale,
    y: (value) => box.y + box.height / 2 - value * scale,
    scale,
  };
}

function drawOja() {
  const oja = state.data.oja;
  const step = currentStep();
  if (!step) return;

  drawPanelTitle("Teacher's Hebbian network", `Step ${step.step}: Oja is normalized Hebbian learning, not fruit classification`);
  const compactLegend = state.width < 640;
  const legendHeight = compactLegend ? 84 : 58;
  drawOjaLegend(28, 69, compactLegend);

  const box = {
    x: 58,
    y: 69 + legendHeight + 15,
    width: state.width - 116,
    height: state.height - (69 + legendHeight + 15) - 88,
  };
  const map = plotMapper(oja.points, box);
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;

  ctx.save();
  ctx.strokeStyle = colors.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(box.x, cy);
  ctx.lineTo(box.x + box.width, cy);
  ctx.moveTo(cx, box.y);
  ctx.lineTo(cx, box.y + box.height);
  ctx.stroke();

  oja.points.forEach((point) => {
    ctx.beginPath();
    ctx.fillStyle = "rgba(91, 90, 166, 0.22)";
    ctx.arc(map.x(point[0]), map.y(point[1]), 3.2, 0, Math.PI * 2);
    ctx.fill();
  });

  const input = step.input;
  ctx.beginPath();
  ctx.fillStyle = colors.amber;
  ctx.arc(map.x(input[0]), map.y(input[1]), 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = colors.ink;
  ctx.font = "700 12px Inter, system-ui, sans-serif";
  ctx.fillText("current input x", map.x(input[0]) + 12, map.y(input[1]) - 10);

  const pca = oja.pca_vector;
  const pcaLength = Math.min(box.width, box.height) * 0.44;
  const pureWeight = step.pure_weight_unit;
  ctx.strokeStyle = "rgba(91, 90, 166, 0.72)";
  ctx.lineWidth = 3.2;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(cx - pca[0] * pcaLength, cy + pca[1] * pcaLength);
  ctx.lineTo(cx + pca[0] * pcaLength, cy - pca[1] * pcaLength);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = colors.violet;
  ctx.font = "800 12px Inter, system-ui, sans-serif";
  ctx.fillText(
    "batch PCA direction",
    clamp(cx + pca[0] * pcaLength + 10, box.x + 8, box.x + box.width - 138),
    clamp(cy - pca[1] * pcaLength - 8, box.y + 14, box.y + box.height - 10),
  );

  drawArrow(
    cx,
    cy,
    cx + pureWeight[0] * pcaLength * 1.08,
    cy - pureWeight[1] * pcaLength * 1.08,
    colors.berry,
    3,
  );
  ctx.fillStyle = colors.berry;
  ctx.font = "800 12px Inter, system-ui, sans-serif";
  ctx.fillText(
    `pure Hebbian: same direction, ||w||=${formatMagnitude(step.pure_weight_norm)}`,
    clamp(cx + pureWeight[0] * pcaLength * 1.08 + 12, box.x + 8, box.x + box.width - 260),
    clamp(cy - pureWeight[1] * pcaLength * 1.08 + 18, box.y + 14, box.y + box.height - 10),
  );

  const weight = step.new_weight_unit;
  drawArrow(cx, cy, cx + weight[0] * pcaLength * 0.82, cy - weight[1] * pcaLength * 0.82, colors.teal, 5);
  ctx.fillStyle = colors.teal;
  ctx.font = "800 12px Inter, system-ui, sans-serif";
  ctx.fillText(
    "Oja learned weight",
    clamp(cx + weight[0] * pcaLength * 0.82 + 12, box.x + 8, box.x + box.width - 130),
    clamp(cy - weight[1] * pcaLength * 0.82 + 14, box.y + 14, box.y + box.height - 10),
  );

  ctx.fillStyle = colors.ink;
  ctx.font = "700 14px Inter, system-ui, sans-serif";
  const angleText = `Same direction: Oja ${formatNumber(step.angle_degrees, 2)} deg from PCA; pure ${formatNumber(
    step.pure_angle_degrees,
    2,
  )} deg. Different size: Oja ${formatNumber(step.weight_norm, 3)}, pure ${formatMagnitude(step.pure_weight_norm)}.`;
  if (compactLegend) {
    ctx.fillText(fitText(angleText, box.width), box.x, box.y + box.height + 26);
    drawNormTrace(oja.steps, state.stepIndex, box.x, box.y + box.height + 44, box.width, 34);
  } else {
    const traceWidth = Math.min(300, box.width * 0.5);
    ctx.fillText(fitText(angleText, box.width - traceWidth - 18), box.x, box.y + box.height + 42);
    drawNormTrace(oja.steps, state.stepIndex, box.x + box.width - traceWidth, box.y + box.height + 14, traceWidth, 56);
  }
  ctx.restore();
}

function drawOjaLegend(x, y, compact = false) {
  const width = Math.min(state.width - 56, compact ? 420 : 560);
  const height = compact ? 84 : 58;
  const rowY = compact ? [y + 15, y + 42, y + 69] : [y + 15, y + 42, y + 42];
  ctx.save();
  ctx.fillStyle = "rgba(255, 250, 240, 0.9)";
  ctx.strokeStyle = "rgba(221, 212, 194, 0.9)";
  roundRect(x, y, width, height, 8);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "rgba(91, 90, 166, 0.42)";
  ctx.beginPath();
  ctx.arc(x + 18, rowY[0], 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = colors.muted;
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.fillText("unlabeled data", x + 30, rowY[0] + 4);

  ctx.fillStyle = colors.amber;
  ctx.beginPath();
  ctx.arc(x + 144, rowY[0], 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = colors.muted;
  ctx.fillText("current input", x + 156, rowY[0] + 4);

  ctx.strokeStyle = colors.violet;
  ctx.lineWidth = 2.4;
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.moveTo(x + 18, rowY[1]);
  ctx.lineTo(x + 50, rowY[1]);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = colors.muted;
  ctx.fillText("PCA from full dataset", x + 58, rowY[1] + 4);

  ctx.strokeStyle = colors.teal;
  ctx.lineWidth = 3.2;
  ctx.beginPath();
  ctx.moveTo(x + (compact ? 220 : 240), rowY[1]);
  ctx.lineTo(x + (compact ? 256 : 276), rowY[1]);
  ctx.stroke();
  ctx.fillText("Oja stable", x + (compact ? 264 : 284), rowY[1] + 4);

  ctx.strokeStyle = colors.berry;
  ctx.lineWidth = 2.6;
  ctx.beginPath();
  ctx.moveTo(x + (compact ? 18 : 388), rowY[2]);
  ctx.lineTo(x + (compact ? 50 : 424), rowY[2]);
  ctx.stroke();
  ctx.fillText("pure norm grows", x + (compact ? 58 : 432), rowY[2] + 4);
  ctx.restore();
}

function drawNormTrace(steps, currentIndex, x, y, width, height) {
  const visible = steps.slice(0, currentIndex + 1);
  const maxLogNorm = Math.max(1, ...steps.map((step) => step.pure_weight_log10_norm || 0));
  const yForNorm = (norm) => {
    const logNorm = Math.max(0, Math.log10(Math.max(norm, 1e-12)));
    return y + height - (logNorm / maxLogNorm) * height;
  };

  ctx.save();
  ctx.strokeStyle = colors.hairline;
  ctx.strokeRect(x, y, width, height);

  ctx.beginPath();
  visible.forEach((step, index) => {
    const px = x + (index / Math.max(1, steps.length - 1)) * width;
    const py = yForNorm(step.weight_norm);
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = colors.teal;
  ctx.lineWidth = 2.4;
  ctx.stroke();

  ctx.beginPath();
  visible.forEach((step, index) => {
    const px = x + (index / Math.max(1, steps.length - 1)) * width;
    const py = yForNorm(step.pure_weight_norm);
    if (index === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.strokeStyle = colors.berry;
  ctx.lineWidth = 2.4;
  ctx.stroke();

  ctx.fillStyle = colors.muted;
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.fillText("weight size, log scale", x, y - 6);
  ctx.textAlign = "right";
  ctx.fillText(`1e${formatNumber(maxLogNorm, 0)}`, x + width - 4, y + 12);
  ctx.fillText("1", x + width - 4, y + height - 4);
  ctx.restore();
}

function drawForgetting() {
  const payload = state.data.forgetting;
  const standardSteps = payload.series?.standard || payload.steps;
  const ewcSteps = payload.series?.ewc || [];
  const step = standardSteps[clamp(state.stepIndex, 0, standardSteps.length - 1)];
  const ewcStep = ewcSteps[clamp(state.stepIndex, 0, ewcSteps.length - 1)];
  if (!step) return;
  const previousStep = state.stepIndex > 0 ? standardSteps[state.stepIndex - 1] : null;

  drawPanelTitle("Backprop vs EWC", `${step.phase}, epoch ${step.epoch}`);

  const isNarrow = state.width < 620;
  const networkBox = isNarrow
    ? {
        x: 38,
        y: 92,
        width: state.width - 76,
        height: 240,
      }
    : {
        x: 32,
        y: 92,
        width: Math.min(360, state.width * 0.38),
        height: state.height - 150,
      };
  drawTinyNetwork(step.weights, networkBox, previousStep?.weights || null);

  const chartBox = isNarrow
    ? {
        x: 66,
        y: networkBox.y + networkBox.height + 78,
        width: state.width - 104,
        height: Math.max(190, state.height - (networkBox.y + networkBox.height + 228)),
      }
    : {
        x: networkBox.x + networkBox.width + 46,
        y: 96,
        width: state.width - (networkBox.x + networkBox.width + 84),
        height: state.height - 260,
      };
  drawAccuracyChart(standardSteps, state.stepIndex, chartBox, ewcSteps);

  drawForgettingExplanation(
    chartBox.x,
    chartBox.y + chartBox.height + (isNarrow ? 76 : 74),
    chartBox.width,
    step,
    ewcStep,
  );
}

function drawTinyNetwork(weights, box, previousWeights = null) {
  const inputHidden = weights.input_hidden_1 || weights.input_hidden;
  const hiddenHidden = weights.hidden_1_hidden_2 || null;
  const hiddenOutput = weights.hidden_2_output || weights.hidden_output;
  const hiddenBias1 = weights.hidden_1_bias || weights.hidden_bias || [];
  const hiddenBias2 = weights.hidden_2_bias || [];
  const previousInputHidden = previousWeights?.input_hidden_1 || previousWeights?.input_hidden || null;
  const previousHiddenHidden = previousWeights?.hidden_1_hidden_2 || null;
  const previousHiddenOutput = previousWeights?.hidden_2_output || previousWeights?.hidden_output || null;
  const hiddenCount = inputHidden[0].length;
  const hiddenRadius = hiddenCount > 12 ? 9 : 16;

  const inputNodes = [
    { x: box.x + 28, y: box.y + box.height * 0.36, label: "x" },
    { x: box.x + 28, y: box.y + box.height * 0.64, label: "y" },
  ];
  const hiddenNodes1 = Array.from({ length: hiddenCount }, (_, index) => ({
    x: box.x + box.width * (hiddenHidden ? 0.35 : 0.52),
    y: box.y + 28 + index * ((box.height - 58) / Math.max(1, hiddenCount - 1)),
    label: `h${index + 1}`,
  }));
  const hiddenNodes2 = hiddenHidden
    ? Array.from({ length: hiddenHidden[0].length }, (_, index) => ({
        x: box.x + box.width * 0.64,
        y: box.y + 28 + index * ((box.height - 58) / Math.max(1, hiddenHidden[0].length - 1)),
        label: `g${index + 1}`,
      }))
    : null;
  const outputNodes = [
    { x: box.x + box.width - 30, y: box.y + box.height * 0.38, label: "0" },
    { x: box.x + box.width - 30, y: box.y + box.height * 0.62, label: "1" },
  ];

  const inputHiddenHighlights = strongestChanges(inputHidden, previousInputHidden, 8);
  const hiddenHiddenHighlights = hiddenHidden ? strongestChanges(hiddenHidden, previousHiddenHidden, 12) : new Set();
  const hiddenOutputHighlights = strongestChanges(hiddenOutput, previousHiddenOutput, 7);

  inputHidden.forEach((fromWeights, inputIndex) => {
    fromWeights.forEach((weight, hiddenIndex) => {
      const delta = weightDelta(weight, previousInputHidden, inputIndex, hiddenIndex);
      const highlighted = inputHiddenHighlights.has(`${inputIndex}:${hiddenIndex}`);
      drawWeightLine(inputNodes[inputIndex], hiddenNodes1[hiddenIndex], weight, 0.24, delta, highlighted);
    });
  });

  if (hiddenHidden && hiddenNodes2) {
    hiddenHidden.forEach((fromWeights, hiddenIndex) => {
      fromWeights.forEach((weight, nextHiddenIndex) => {
        const delta = weightDelta(weight, previousHiddenHidden, hiddenIndex, nextHiddenIndex);
        const highlighted = hiddenHiddenHighlights.has(`${hiddenIndex}:${nextHiddenIndex}`);
        drawWeightLine(hiddenNodes1[hiddenIndex], hiddenNodes2[nextHiddenIndex], weight, 0.12, delta, highlighted);
      });
    });
  }

  const outputSourceNodes = hiddenNodes2 || hiddenNodes1;
  hiddenOutput.forEach((fromWeights, hiddenIndex) => {
    fromWeights.forEach((weight, outputIndex) => {
      const delta = weightDelta(weight, previousHiddenOutput, hiddenIndex, outputIndex);
      const highlighted = hiddenOutputHighlights.has(`${hiddenIndex}:${outputIndex}`);
      drawWeightLine(outputSourceNodes[hiddenIndex], outputNodes[outputIndex], weight, 0.24, delta, highlighted);
    });
  });

  inputNodes.forEach((node) => drawNode(node.x, node.y, 22, node.label, { fill: "#ffffff" }));
  hiddenNodes1.forEach((node, index) => {
    const bias = hiddenBias1[index] || 0;
    drawNode(node.x, node.y, hiddenRadius, "", {
      fill: bias > 0 ? "rgba(21, 122, 114, 0.18)" : "rgba(181, 59, 93, 0.14)",
      stroke: colors.hairline,
    });
  });
  if (hiddenNodes2) {
    hiddenNodes2.forEach((node, index) => {
      const bias = hiddenBias2[index] || 0;
      drawNode(node.x, node.y, hiddenRadius, "", {
        fill: bias > 0 ? "rgba(21, 122, 114, 0.18)" : "rgba(181, 59, 93, 0.14)",
        stroke: colors.hairline,
      });
    });
  }
  outputNodes.forEach((node) => drawNode(node.x, node.y, 24, node.label, { fill: "#ffffff" }));

  ctx.fillStyle = colors.muted;
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.textAlign = "left";
  const compact = box.width < 310;
  ctx.fillText("Input", box.x, box.y - 14);
  ctx.fillText(compact ? "H1" : "Hidden 1", box.x + box.width * (hiddenHidden ? 0.28 : 0.42), box.y - 14);
  if (hiddenHidden) ctx.fillText(compact ? "H2" : "Hidden 2", box.x + box.width * 0.58, box.y - 14);
  ctx.fillText(compact ? "Out" : "Output", box.x + box.width - 72, box.y - 14);

  drawNetworkLegend(box, previousWeights !== null);
}

function weightDelta(weight, previousMatrix, rowIndex, columnIndex) {
  if (!previousMatrix) return 0;
  return weight - previousMatrix[rowIndex][columnIndex];
}

function strongestChanges(matrix, previousMatrix, limit) {
  if (!previousMatrix) return new Set();
  const changes = [];
  matrix.forEach((row, rowIndex) => {
    row.forEach((value, columnIndex) => {
      changes.push({
        key: `${rowIndex}:${columnIndex}`,
        magnitude: Math.abs(value - previousMatrix[rowIndex][columnIndex]),
      });
    });
  });
  return new Set(
    changes
      .sort((a, b) => b.magnitude - a.magnitude)
      .slice(0, limit)
      .filter((item) => item.magnitude > 0.0008)
      .map((item) => item.key),
  );
}

function drawWeightLine(from, to, weight, alpha, delta = 0, highlighted = false) {
  const changed = highlighted && Math.abs(delta) > 0.0008;
  ctx.save();
  ctx.globalAlpha = changed ? 0.92 : alpha + Math.min(0.26, Math.abs(weight) * 0.16);
  ctx.strokeStyle = changed ? deltaColor(delta) : weightColor(weight);
  ctx.lineWidth = changed ? 2.4 + Math.min(4.2, Math.abs(delta) * 180) : 0.8 + Math.min(3.8, Math.abs(weight) * 1.5);
  if (changed) {
    ctx.shadowColor = deltaColor(delta);
    ctx.shadowBlur = 9;
  }
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
  ctx.restore();
}

function drawNetworkLegend(box, hasPrevious) {
  const x = box.x + 4;
  const y = box.y + box.height - 6;
  const message = hasPrevious ? "network: teal up, red down" : "network: thicker line = stronger weight";

  ctx.save();
  ctx.fillStyle = "rgba(255, 250, 240, 0.9)";
  ctx.strokeStyle = "rgba(221, 212, 194, 0.85)";
  roundRect(x, y - 28, Math.min(285, box.width - 8), 28, 8);
  ctx.fill();
  ctx.stroke();
  ctx.strokeStyle = colors.teal;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(x + 10, y - 14);
  ctx.lineTo(x + 34, y - 14);
  ctx.stroke();
  ctx.strokeStyle = colors.berry;
  ctx.beginPath();
  ctx.moveTo(x + 42, y - 14);
  ctx.lineTo(x + 66, y - 14);
  ctx.stroke();
  ctx.fillStyle = colors.muted;
  ctx.font = "11px Inter, system-ui, sans-serif";
  ctx.fillText(message, x + 76, y - 10);
  ctx.restore();
}

function drawAccuracyChart(steps, currentIndex, box, ewcSteps = []) {
  const visible = steps.slice(0, currentIndex + 1);
  const visibleEwc = ewcSteps.slice(0, currentIndex + 1);
  const splitIndex = steps.findIndex((step) => step.phase === "Train Task B");

  ctx.save();
  ctx.strokeStyle = colors.hairline;
  ctx.lineWidth = 1.4;
  ctx.strokeRect(box.x, box.y, box.width, box.height);

  for (let i = 0; i <= 4; i += 1) {
    const y = box.y + box.height - (i / 4) * box.height;
    ctx.strokeStyle = "rgba(32, 32, 29, 0.08)";
    ctx.beginPath();
    ctx.moveTo(box.x, y);
    ctx.lineTo(box.x + box.width, y);
    ctx.stroke();
    ctx.fillStyle = colors.muted;
    ctx.font = "11px Inter, system-ui, sans-serif";
    ctx.fillText(`${i * 25}%`, box.x - 36, y + 4);
  }

  if (splitIndex > 0) {
    const splitX = box.x + (splitIndex / Math.max(1, steps.length - 1)) * box.width;
    ctx.strokeStyle = colors.amber;
    ctx.setLineDash([6, 6]);
    ctx.beginPath();
    ctx.moveTo(splitX, box.y);
    ctx.lineTo(splitX, box.y + box.height);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = colors.amber;
    ctx.font = "700 12px Inter, system-ui, sans-serif";
    ctx.fillText("Task B begins", splitX + 8, box.y + 18);
  }

  drawAccuracyLine(visible, steps.length, box, "task_a_accuracy", colors.teal);
  drawAccuracyLine(visible, steps.length, box, "task_b_accuracy", colors.berry);
  if (visibleEwc.length) {
    drawAccuracyLine(visibleEwc, ewcSteps.length, box, "task_a_accuracy", colors.violet);
    drawAccuracyLine(visibleEwc, ewcSteps.length, box, "task_b_accuracy", colors.amber);
  }

  ctx.fillStyle = colors.ink;
  ctx.font = "700 14px Inter, system-ui, sans-serif";
  ctx.fillText("Accuracy over time", box.x, box.y - 18);
  drawForgettingColorKey(box.x, box.y + box.height + 20, box.width, visibleEwc.length > 0);
  ctx.restore();
}

function drawForgettingColorKey(x, y, width, showEwc) {
  const items = [
    { color: colors.teal, label: "standard old task" },
    { color: colors.berry, label: "standard new task" },
    ...(showEwc
      ? [
          { color: colors.violet, label: "EWC old task" },
          { color: colors.amber, label: "EWC new task" },
        ]
      : []),
  ];
  const columnWidth = Math.max(132, width / 2);

  ctx.save();
  ctx.font = "700 11px Inter, system-ui, sans-serif";
  items.forEach((item, index) => {
    const column = index % 2;
    const row = Math.floor(index / 2);
    const itemX = x + column * columnWidth;
    const itemY = y + row * 20;
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(itemX, itemY);
    ctx.lineTo(itemX + 24, itemY);
    ctx.stroke();
    ctx.fillStyle = item.color;
    ctx.fillText(item.label, itemX + 32, itemY + 4);
  });
  ctx.restore();
}

function drawForgettingExplanation(x, y, width, step, ewcStep) {
  const isTaskB = step.phase === "Train Task B";
  const lines = isTaskB
    ? [
        "Four dots = current accuracies.",
        "2 methods: standard backprop and EWC.",
        "2 tests: old Task A and new Task B.",
      ]
    : [
        "First, both methods learn Task A.",
        "EWC saves important weights after Task A.",
        "When Task B begins, the old-task curves reveal forgetting.",
      ];

  ctx.save();
  ctx.fillStyle = "rgba(255, 250, 240, 0.92)";
  ctx.strokeStyle = "rgba(221, 212, 194, 0.92)";
  roundRect(x, y - 14, width, 104, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = colors.ink;
  ctx.font = "800 12px Inter, system-ui, sans-serif";
  ctx.fillText("What this chart is doing", x + 12, y + 5);
  ctx.fillStyle = colors.muted;
  ctx.font = "12px Inter, system-ui, sans-serif";
  lines.forEach((line, index) => {
    ctx.fillText(fitText(line, width - 24), x + 12, y + 25 + index * 17);
  });
  if (ewcStep && isTaskB) {
    ctx.fillStyle = colors.violet;
    ctx.font = "800 12px Inter, system-ui, sans-serif";
    ctx.fillText(
      fitText(
        `EWC now: old ${formatPercent(ewcStep.task_a_accuracy)}, new ${formatPercent(ewcStep.task_b_accuracy)}`,
        width - 24,
      ),
      x + 12,
      y + 76,
    );
  }
  ctx.restore();
}

function drawAccuracyLine(visible, totalLength, box, field, color) {
  if (visible.length === 0) return;
  ctx.save();
  ctx.beginPath();
  visible.forEach((step, index) => {
    const x = box.x + (index / Math.max(1, totalLength - 1)) * box.width;
    const y = box.y + box.height - step[field] * box.height;
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.stroke();
  const last = visible[visible.length - 1];
  const lastX = box.x + ((visible.length - 1) / Math.max(1, totalLength - 1)) * box.width;
  const lastY = box.y + box.height - last[field] * box.height;
  ctx.beginPath();
  ctx.arc(lastX, lastY, 5, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
}

function roundRect(x, y, width, height, radius) {
  ctx.beginPath();
  ctx.moveTo(x + radius, y);
  ctx.arcTo(x + width, y, x + width, y + height, radius);
  ctx.arcTo(x + width, y + height, x, y + height, radius);
  ctx.arcTo(x, y + height, x, y, radius);
  ctx.arcTo(x, y, x + width, y, radius);
  ctx.closePath();
}

function updateInspector() {
  if (state.scene === "playground") {
    updatePlaygroundInspector();
    return;
  }

  const step = currentStep();
  if (!step) return;

  if (state.scene === "fruit") {
    const isPassPhase = step.visual_phase === "pass";
    const modeName = state.fruitMode === "hebbian" ? "Hebbian" : "Backprop";
    const activatedCount =
      state.fruitMode === "hebbian"
        ? step.active_features.length
        : step.active_features.length * state.data.fruit.classes.length;
    const changedCount = step.weight_delta.flat().filter((value) => Math.abs(value) > 0.0001).length;
    stepTitle.textContent = `${modeName} ${isPassPhase ? "pass-through" : "weight update"}: ${step.label}`;
    stepNarrative.textContent = isPassPhase
      ? state.fruitMode === "hebbian"
        ? "The example activates its feature neurons and the observed class neuron. This frame shows which existing connections are being read before any weight is changed."
        : "The example activates its feature neurons, sends scores to every class, and computes the prediction. This frame shows the forward pass before learning changes the weights."
      : step.explanation;
    if (state.fruitMode === "hebbian") {
      formulaBox.innerHTML = formulaLines([
        isPassPhase ? "pass-through: active input features -> observed class" : "output[class] = 1 for the observed fruit",
        isPassPhase ? "no weights change in this frame" : "Delta W = learning_rate * outer(input, output)",
        isPassPhase ? "blue dashed rays show connections being read" : "teal rays show active connections that increased",
      ]);
    } else {
      formulaBox.innerHTML = formulaLines([
        isPassPhase ? "scores = input dot W" : "probabilities = softmax(input dot W)",
        isPassPhase ? "probabilities = softmax(scores)" : "error = probabilities - target",
        isPassPhase ? "no weights change in this frame" : "Delta W = -learning_rate * outer(input, error)",
      ]);
    }
    valueGrid.innerHTML = valueRows([
      ["phase", isPassPhase ? "pass-through" : "weight update"],
      ["epoch", step.epoch],
      ["sample", step.sample_index],
      ["repeat", step.repeat_label],
      ["active", step.active_features.join(", ")],
      ["rays read", activatedCount],
      ["rays changed", isPassPhase ? 0 : changedCount],
      ["target", step.label],
      ["prediction", isPassPhase ? "not updated yet" : step.prediction],
      ...(isPassPhase && state.fruitMode === "hebbian" ? [] : [["scores", vectorText(step.scores)]]),
      ...(step.probabilities ? [["probabilities", vectorText(step.probabilities)]] : []),
      ...(step.loss ? [["loss", step.loss]] : []),
    ]);
  }

  if (state.scene === "oja") {
    stepTitle.textContent = `Oja update ${step.step}`;
    stepNarrative.textContent =
      "The teal and red arrows point almost the same way because both methods find the high-variance PCA direction. The difference is length: pure Hebbian keeps strengthening the same weight until ||w|| explodes, while Oja keeps it bounded.";
    formulaBox.innerHTML = formulaLines([
      "Pure Hebbian: dw = eta * y * x",
      "Oja: dw = eta * (y*x - y^2*w)",
      "-y^2*w is the brake: it prevents weights from growing forever.",
      "Main plot: same direction. Bottom chart: different weight size.",
    ]);
    valueGrid.innerHTML = valueRows([
      ["what it learns", "first principal component"],
      ["epoch", step.epoch],
      ["initial angle", `${formatNumber(state.data.oja.steps[0].angle_degrees, 2)} deg`],
      ["input x", vectorText(step.input)],
      ["old w", vectorText(step.old_weight)],
      ["y", step.output],
      ["Delta w", vectorText(step.weight_delta)],
      ["Oja norm", formatNumber(step.weight_norm, 4)],
      ["Oja angle", `${formatNumber(step.angle_degrees, 2)} deg`],
      ["pure norm", formatMagnitude(step.pure_weight_norm)],
      ["pure angle", `${formatNumber(step.pure_angle_degrees, 2)} deg`],
      ["pure final norm", formatMagnitude(state.data.oja.pure_hebbian_final_norm)],
    ]);
  }

  if (state.scene === "forgetting") {
    const payload = state.data.forgetting;
    const standardSteps = payload.series?.standard || payload.steps;
    const ewcSteps = payload.series?.ewc || [];
    const standardStep = standardSteps[clamp(state.stepIndex, 0, standardSteps.length - 1)] || step;
    const ewcStep = ewcSteps[clamp(state.stepIndex, 0, ewcSteps.length - 1)];
    const standardSummary = payload.summary.standard || payload.summary;
    const ewcSummary = payload.summary.ewc;
    stepTitle.textContent = `${step.phase}, epoch ${step.epoch}`;
    stepNarrative.textContent =
      step.phase === "Train Task A"
        ? "Both runs first learn Task A. EWC then stores the learned weights and estimates which weights were important, so it has something to protect later."
        : "Standard backprop updates weights only to improve Task B. EWC still learns Task B, but adds a penalty when Task-A-important weights move too far from their saved values.";
    formulaBox.innerHTML = formulaLines([
      "Color key: teal/red = standard old/new task; purple/orange = EWC old/new task.",
      step.phase === "Train Task A"
        ? "Standard backprop: learn the current task by reducing prediction error."
        : "Standard backprop: L_total = L_B, so only Task B matters now.",
      step.phase === "Train Task A"
        ? "EWC: after Task A, save theta* and estimate Fisher importance F_i."
        : "EWC: L_total = L_B + lambda * sum(F_i * (theta_i - theta*_i)^2).",
      step.phase === "Train Task A"
        ? "That saved state is the memory EWC protects during Task B."
        : "Difference: EWC punishes moving important old-task weights; standard backprop does not.",
    ]);
    valueGrid.innerHTML = valueRows([
      ["phase", step.phase],
      ["epoch", step.epoch],
      ["standard old", formatPercent(standardStep.task_a_accuracy)],
      ["standard new", formatPercent(standardStep.task_b_accuracy)],
      ...(ewcStep
        ? [
            ["EWC old", formatPercent(ewcStep.task_a_accuracy)],
            ["EWC new", formatPercent(ewcStep.task_b_accuracy)],
            ["EWC penalty", ewcStep.ewc_penalty],
          ]
        : []),
      ["std old final", formatPercent(standardSummary.task_a_after_b)],
      ...(ewcSummary ? [["EWC old final", formatPercent(ewcSummary.task_a_after_b)]] : []),
    ]);
  }
}

function formulaLines(lines) {
  return lines.map((line) => `<div class="formula-line">${line}</div>`).join("");
}

function valueRows(rows) {
  return rows
    .map(
      ([label, value]) =>
        `<div class="value-row"><div class="value-label">${label}</div><div class="value-number">${value}</div></div>`,
    )
    .join("");
}

function vectorText(values) {
  if (!Array.isArray(values)) return values;
  return `[${values.map((value) => formatNumber(value, 3)).join(", ")}]`;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function ensurePlaygroundInput() {
  if (!state.data?.playground) return;
  const featureCount = state.data.playground.features.length;
  if (Array.isArray(state.playgroundInput) && state.playgroundInput.length === featureCount) return;
  state.playgroundInput = state.data.playground.presets[0].input.slice();
  state.selectedPresetIndex = 0;
}

function setPlaygroundPreset(index) {
  const preset = state.data.playground.presets[index];
  if (!preset) return;
  state.selectedPresetIndex = index;
  state.playgroundInput = preset.input.slice();
  render();
}

function togglePlaygroundFeature(index) {
  ensurePlaygroundInput();
  state.selectedPresetIndex = null;
  state.playgroundInput[index] = state.playgroundInput[index] ? 0 : 1;
  render();
}

function clearPlaygroundInput() {
  ensurePlaygroundInput();
  state.selectedPresetIndex = null;
  state.playgroundInput = state.playgroundInput.map(() => 0);
  render();
}

function playgroundActiveFeatures() {
  ensurePlaygroundInput();
  return state.data.playground.features.filter((feature, index) => state.playgroundInput[index] === 1);
}

function dotScores(input, weights) {
  return weights[0].map((_, classIndex) =>
    input.reduce((sum, value, featureIndex) => sum + value * weights[featureIndex][classIndex], 0),
  );
}

function softmaxValues(values) {
  const max = Math.max(...values);
  const exp = values.map((value) => Math.exp(value - max));
  const sum = exp.reduce((total, value) => total + value, 0);
  return exp.map((value) => value / sum);
}

function positiveStrengths(values) {
  const positives = values.map((value) => Math.max(0, value));
  const total = positives.reduce((sum, value) => sum + value, 0);
  if (total <= 0) return values.map(() => 0);
  return positives.map((value) => value / total);
}

function scorePlaygroundModel(modelKey) {
  ensurePlaygroundInput();
  const payload = state.data.playground;
  const scores = dotScores(state.playgroundInput, payload.models[modelKey].weights);
  const predictionIndex = scores.indexOf(Math.max(...scores));
  const distribution = modelKey === "backprop" ? softmaxValues(scores) : positiveStrengths(scores);
  return {
    scores,
    distribution,
    prediction: payload.classes[predictionIndex],
    confidence: distribution[predictionIndex] || 0,
  };
}

function selectedPlaygroundPreset() {
  if (state.selectedPresetIndex === null) return null;
  return state.data.playground.presets[state.selectedPresetIndex] || null;
}

function renderPlayground() {
  ensurePlaygroundInput();
  const payload = state.data.playground;
  const preset = selectedPlaygroundPreset();
  const hebbian = scorePlaygroundModel("hebbian");
  const backprop = scorePlaygroundModel("backprop");
  const activeFeatures = playgroundActiveFeatures();

  playgroundPanel.innerHTML = `
    <div class="playground-content">
      <div class="playground-header">
        <div>
          <p class="eyebrow">Live learned-weight comparison</p>
          <h3>Try an input, then compare the two trained models</h3>
          <p>${escapeHtml(preset?.note || "Custom input: the models are reading the feature vector you selected below.")}</p>
        </div>
        <div class="accuracy-summary" aria-label="Verification accuracy">
          ${modelAccuracyHTML("Hebbian", payload.models.hebbian.verification_accuracy)}
          ${modelAccuracyHTML("Backprop", payload.models.backprop.verification_accuracy)}
        </div>
      </div>

      <div class="preset-row" aria-label="Verification presets">
        ${payload.presets
          .map(
            (item, index) => `
              <button class="preset-button ${state.selectedPresetIndex === index ? "active" : ""}" type="button" data-preset-index="${index}">
                <span>${String(index + 1).padStart(2, "0")}</span>
                ${escapeHtml(item.name)}
              </button>
            `,
          )
          .join("")}
      </div>

      <div class="playground-grid">
        <section class="feature-board" aria-label="Fruit feature vector">
          <div class="board-heading">
            <div>
              <p class="eyebrow">Input vector</p>
              <h4>${activeFeatures.length ? escapeHtml(activeFeatures.join(" + ")) : "No active features"}</h4>
            </div>
            <button class="small-button" type="button" id="clearPlaygroundInput">Clear</button>
          </div>
          <div class="feature-toggle-grid">
            ${payload.features
              .map(
                (feature, index) => `
                  <button
                    class="feature-toggle ${state.playgroundInput[index] ? "active" : ""}"
                    type="button"
                    aria-pressed="${state.playgroundInput[index] ? "true" : "false"}"
                    data-feature-index="${index}">
                    <span>${state.playgroundInput[index] ? "1" : "0"}</span>
                    ${escapeHtml(feature)}
                  </button>
                `,
              )
              .join("")}
          </div>
          <div class="vector-readout">x = [${state.playgroundInput.join(", ")}]</div>
        </section>

        <section class="model-comparison" aria-label="Model prediction comparison">
          ${modelResultHTML("hebbian", "Hebbian", "Associative score", hebbian, preset)}
          ${modelResultHTML("backprop", "Backprop", "Softmax probability", backprop, preset)}
        </section>
      </div>
    </div>
  `;

  playgroundPanel.querySelectorAll("[data-preset-index]").forEach((button) => {
    button.addEventListener("click", () => setPlaygroundPreset(Number(button.dataset.presetIndex)));
  });
  playgroundPanel.querySelectorAll("[data-feature-index]").forEach((button) => {
    button.addEventListener("click", () => togglePlaygroundFeature(Number(button.dataset.featureIndex)));
  });
  playgroundPanel.querySelector("#clearPlaygroundInput")?.addEventListener("click", clearPlaygroundInput);
}

function modelAccuracyHTML(label, accuracy) {
  return `
    <div class="accuracy-pill">
      <span>${escapeHtml(label)}</span>
      <strong>${formatNumber(accuracy * 100, 1)}%</strong>
    </div>
  `;
}

function modelResultHTML(modelKey, title, scoreLabel, result, preset) {
  const expected = preset?.expected || null;
  const correctClass = expected && result.prediction === expected ? "correct" : "";
  const mismatchClass = expected && result.prediction !== expected ? "mismatch" : "";
  return `
    <article class="model-result ${modelKey}">
      <div class="model-result-head">
        <div>
          <p class="eyebrow">${escapeHtml(scoreLabel)}</p>
          <h4>${escapeHtml(title)}</h4>
        </div>
        <div class="prediction-badge ${correctClass} ${mismatchClass}">
          ${escapeHtml(result.prediction)}
        </div>
      </div>
      <div class="score-bars">
        ${state.data.playground.classes
          .map((className, index) =>
            scoreBarHTML(
              className,
              result.scores[index],
              result.distribution[index],
              className === result.prediction,
              expected === className,
            ),
          )
          .join("")}
      </div>
      <p class="model-note">
        ${
          expected
            ? `Expected: ${escapeHtml(expected)}. ${result.prediction === expected ? "This model matches the label." : "This model misses this label."}`
            : "No expected label here. Use this to see how each learned model generalizes."
        }
      </p>
    </article>
  `;
}

function scoreBarHTML(className, score, amount, predicted, expected) {
  const percent = clamp(amount * 100, 0, 100);
  return `
    <div class="score-row ${predicted ? "predicted" : ""} ${expected ? "expected" : ""}">
      <div class="score-label">
        <span>${escapeHtml(className)}</span>
        <small>${formatNumber(score, 3)}</small>
      </div>
      <div class="score-track" aria-hidden="true">
        <div class="score-fill" style="transform: scaleX(${percent / 100})"></div>
      </div>
      <strong>${formatNumber(percent, 1)}%</strong>
    </div>
  `;
}

function updatePlaygroundInspector() {
  const preset = selectedPlaygroundPreset();
  const hebbian = scorePlaygroundModel("hebbian");
  const backprop = scorePlaygroundModel("backprop");
  const activeFeatures = playgroundActiveFeatures();

  stepTitle.textContent = preset ? `Preset: ${preset.name}` : "Custom fruit vector";
  stepNarrative.textContent =
    "The playground uses the final trained weights from the fruit demo. Changing a feature immediately recomputes each model's scores from the same input vector.";
  formulaBox.innerHTML = formulaLines([
    "x = selected fruit feature vector",
    "Hebbian scores = x dot W_hebbian",
    "Backprop probabilities = softmax(x dot W_backprop)",
  ]);
  valueGrid.innerHTML = valueRows([
    ["expected", preset?.expected || "custom / unlabeled"],
    ["active", activeFeatures.length ? activeFeatures.join(", ") : "none"],
    ["input x", vectorText(state.playgroundInput)],
    ["Hebbian", `${hebbian.prediction} (${formatNumber(hebbian.confidence * 100, 1)}%)`],
    ["Backprop", `${backprop.prediction} (${formatNumber(backprop.confidence * 100, 1)}%)`],
  ]);
}

function render() {
  if (!state.data || !ctx) return;
  updateChrome();
  const isPlayground = state.scene === "playground";
  canvas.classList.toggle("hidden", isPlayground);
  playgroundPanel.classList.toggle("hidden", !isPlayground);

  if (isPlayground) {
    renderPlayground();
  } else {
    clearCanvas();
    if (state.scene === "fruit") drawFruit();
    if (state.scene === "oja") drawOja();
    if (state.scene === "forgetting") drawForgetting();
  }
  updateInspector();
}

function nextStep() {
  const maxIndex = currentSteps().length - 1;
  if (state.stepIndex >= maxIndex) {
    stopPlayback();
    return;
  }
  state.stepIndex += 1;
  render();
}

function previousStep() {
  state.stepIndex = Math.max(0, state.stepIndex - 1);
  render();
}

function resetStep() {
  state.stepIndex = 0;
  stopPlayback();
  render();
}

function togglePlayback() {
  if (state.playing) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

function startPlayback() {
  state.playing = true;
  playButton.textContent = "Pause";
  state.playTimer = window.setInterval(nextStep, state.scene === "oja" ? 28 : 420);
}

function stopPlayback() {
  state.playing = false;
  playButton.textContent = "Play";
  if (state.playTimer) {
    window.clearInterval(state.playTimer);
    state.playTimer = null;
  }
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} failed with ${response.status}`);
  return response.json();
}

async function loadData() {
  const [fruit, oja, forgetting, playground] = await Promise.all([
    fetchJson("/api/fruit"),
    fetchJson("/api/oja"),
    fetchJson("/api/forgetting"),
    fetchJson("/api/playground"),
  ]);
  state.data = { fruit, oja, forgetting, playground };
  ensurePlaygroundInput();
  loadingOverlay.classList.add("hidden");
  updateChrome();
  resizeCanvas();
  render();
}

document.querySelectorAll(".scene-tab").forEach((button) => {
  button.addEventListener("click", () => setScene(button.dataset.scene));
});

document.querySelectorAll(".mode-button").forEach((button) => {
  button.addEventListener("click", () => setFruitMode(button.dataset.mode));
});

playButton.addEventListener("click", togglePlayback);
prevButton.addEventListener("click", previousStep);
nextButton.addEventListener("click", nextStep);
resetButton.addEventListener("click", resetStep);
stepSlider.addEventListener("input", (event) => {
  state.stepIndex = Number(event.target.value);
  stopPlayback();
  render();
});

window.addEventListener("resize", resizeCanvas);

loadData().catch((error) => {
  loadingOverlay.textContent = `Could not load demo data: ${error.message}`;
});
