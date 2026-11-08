/**
 * natHacks '26 Robotic Sorting Arm - Dashboard Client Application
 */

// Application State
const state = {
  ws: null,
  wsConnected: false,
  cameraMode: "simulated", // "simulated" or "webcam"
  webcamStream: null,
  detections: [],
  selectedTargetId: "obj_1",
  activeFilter: "ALL",
  events: [],
  arm: {
    joints: {
      deg: { base: 90, shoulder: 90, elbow: 90, wrist: 90, gripper: 90 },
      rad: { base: 0, shoulder: 0, elbow: 0, wrist: 0, gripper: 0.5 },
    },
    task_step: "IDLE",
    packet_counter: 1042,
    watchdog_state: "HEALTHY",
    is_moving: false,
    is_gripping: false,
  },
  stats: {},
};

// DOM Elements
const elements = {
  linkState: document.getElementById("link-state"),
  watchdogState: document.getElementById("watchdog-state"),
  btnStop: document.getElementById("btn-stop"),
  btnReset: document.getElementById("btn-reset"),
  btnModeToggle: document.getElementById("btn-mode-toggle"),
  activeModeText: document.getElementById("active-mode-text"),
  safetyBanner: document.getElementById("safety-banner"),
  safetyBannerTitle: document.getElementById("safety-banner-title"),
  safetyBannerMsg: document.getElementById("safety-banner-msg"),
  cmdInput: document.getElementById("cmd-input"),
  cmdStatusBadge: document.getElementById("cmd-status-badge"),
  cmdFeedbackBox: document.getElementById("cmd-feedback-box"),
  feedbackIcon: document.getElementById("feedback-icon"),
  feedbackText: document.getElementById("feedback-text"),
  jsonDisplay: document.getElementById("json-display"),
  webcamVideo: document.getElementById("webcam-video"),
  simulatedFeed: document.getElementById("simulated-feed"),
  visionCanvas: document.getElementById("vision-canvas"),
  hudSource: document.getElementById("hud-source"),
  hudFps: document.getElementById("hud-fps"),
  hudCount: document.getElementById("hud-count"),
  hudTargetName: document.getElementById("hud-target-name"),
  targetsChipsContainer: document.getElementById("targets-chips-container"),
  packetCounterBadge: document.getElementById("packet-counter-badge"),
  currentStepLabel: document.getElementById("current-step-label"),
  stepDescription: document.getElementById("step-description"),
  valBase: document.getElementById("val-base"),
  radBase: document.getElementById("rad-base"),
  gaugeBase: document.getElementById("gauge-base"),
  valShoulder: document.getElementById("val-shoulder"),
  radShoulder: document.getElementById("rad-shoulder"),
  gaugeShoulder: document.getElementById("gauge-shoulder"),
  valElbow: document.getElementById("val-elbow"),
  radElbow: document.getElementById("rad-elbow"),
  gaugeElbow: document.getElementById("gauge-elbow"),
  valWrist: document.getElementById("val-wrist"),
  radWrist: document.getElementById("rad-wrist"),
  gaugeWrist: document.getElementById("gauge-wrist"),
  valGripper: document.getElementById("val-gripper"),
  radGripper: document.getElementById("rad-gripper"),
  gaugeGripper: document.getElementById("gauge-gripper"),
  armSchematicCanvas: document.getElementById("arm-schematic-canvas"),
  statSorted: document.getElementById("stat-sorted"),
  statRejects: document.getElementById("stat-rejects"),
  statSuccessRate: document.getElementById("stat-success-rate"),
  statCycleTime: document.getElementById("stat-cycle-time"),
  statLastCycle: document.getElementById("stat-last-cycle"),
  inaCurrent: document.getElementById("ina-current"),
  inaVolt: document.getElementById("ina-volt"),
  inaPower: document.getElementById("ina-power"),
  olAttempts: document.getElementById("ol-attempts"),
  olSuccess: document.getElementById("ol-success"),
  olRejects: document.getElementById("ol-rejects"),
  olAccuracy: document.getElementById("ol-accuracy"),
  olCycle: document.getElementById("ol-cycle"),
  cgAttempts: document.getElementById("cg-attempts"),
  cgSuccess: document.getElementById("cg-success"),
  cgRejects: document.getElementById("cg-rejects"),
  cgAccuracy: document.getElementById("cg-accuracy"),
  cgCycle: document.getElementById("cg-cycle"),
  eventStream: document.getElementById("event-stream"),
  docsModal: document.getElementById("docs-modal"),
};

// -------------------------------------------------------- WEBSOCKET
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  state.ws = new WebSocket(wsUrl);

  state.ws.onopen = () => {
    state.wsConnected = true;
    elements.linkState.textContent = "SIMULATED (CONNECTED)";
    elements.linkState.className = "status-val live";
    console.log("WebSocket connected to", wsUrl);
  };

  state.ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleServerMessage(msg);
    } catch (e) {
      console.error("WS parse error:", e);
    }
  };

  state.ws.onclose = () => {
    state.wsConnected = false;
    elements.linkState.textContent = "DISCONNECTED";
    elements.linkState.className = "status-val danger";
    setTimeout(connectWebSocket, 2000);
  };

  state.ws.onerror = (err) => {
    console.error("WS error:", err);
  };
}

function handleServerMessage(msg) {
  if (msg.type === "init") {
    if (msg.arm) updateArmState(msg.arm);
    if (msg.stats) updateStats(msg.stats);
    if (msg.vision && msg.vision.detections) {
      state.detections = msg.vision.detections;
      updateTargetChips();
    }
    if (msg.recent_events) {
      state.events = msg.recent_events;
      renderEvents();
    }
  } else if (msg.type === "event") {
    const ev = msg.data;
    handleDomainEvent(ev);
  } else if (msg.type === "command_result") {
    displayCommandResult(msg.data);
  }
}

function handleDomainEvent(ev) {
  state.events.push(ev);
  if (state.events.length > 300) state.events.shift();
  renderEvents();

  const name = ev.event;
  const p = ev.payload || {};

  if (name === "arm:status_update" || name === "planner:step_changed") {
    updateArmState(p);
  } else if (name === "stats:production_update") {
    updateStats(p);
  } else if (name === "sensors:ina219_read") {
    updateIna219(p);
  } else if (name === "safety:rejected") {
    showSafetyBanner("SAFETY REJECTION", p.reason || "Safety check failed.");
  } else if (name === "safety:estop") {
    showSafetyBanner("EMERGENCY STOPPED", p.reason || "Stop commanded. Physical kill switch remains real safety.");
    elements.btnStop.classList.add("danger-active");
  } else if (name === "safety:reset") {
    hideSafetyBanner();
    elements.btnStop.classList.remove("danger-active");
  } else if (name === "vision:target_selected") {
    if (p.target) {
      state.selectedTargetId = p.target.id;
      elements.hudTargetName.textContent = p.target.label.toUpperCase();
      updateTargetChips();
    }
  } else if (name === "system:watchdog_tick") {
    updateWatchdog(p);
  }
}

// -------------------------------------------------------- COMMAND DISPATCH
window.sendCommand = function () {
  const text = elements.cmdInput.value.trim();
  if (!text) return;

  if (state.wsConnected && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ action: "command", text: text }));
  } else {
    // REST Fallback
    fetch("/api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command: text }),
    })
      .then((r) => r.json())
      .then(displayCommandResult)
      .catch((err) => console.error(err));
  }
};

window.setCommand = function (text) {
  elements.cmdInput.value = text;
  sendCommand();
};

function displayCommandResult(res) {
  // Update badge
  const status = res.status; // "accepted" | "rejected" | "needs_clarification"
  elements.cmdStatusBadge.className = "badge-status";

  if (status === "accepted") {
    elements.cmdStatusBadge.textContent = "ACCEPTED";
    elements.cmdStatusBadge.classList.add("badge-accepted");
    elements.cmdFeedbackBox.className = "feedback-box hidden";
  } else if (status === "rejected") {
    elements.cmdStatusBadge.textContent = "REJECTED";
    elements.cmdStatusBadge.classList.add("badge-rejected");
    elements.cmdFeedbackBox.className = "feedback-box feedback-rejected";
    elements.feedbackIcon.textContent = "🛑";
    elements.feedbackText.textContent = res.reason || "Safety layer rejected command.";
    showSafetyBanner("COMMAND REJECTED", res.reason || "Unsafe command rejected.");
  } else if (status === "needs_clarification") {
    elements.cmdStatusBadge.textContent = "NEEDS CLARIFICATION";
    elements.cmdStatusBadge.classList.add("badge-clarify");
    elements.cmdFeedbackBox.className = "feedback-box feedback-clarify";
    elements.feedbackIcon.textContent = "❓";
    elements.feedbackText.textContent = res.prompt || "Could not resolve unambiguous target.";
  }

  // Display JSON
  elements.jsonDisplay.textContent = JSON.stringify(res.parsed_json, null, 2);
}

window.copyJson = function () {
  navigator.clipboard.writeText(elements.jsonDisplay.textContent).then(() => {
    alert("Parsed JSON copied to clipboard!");
  });
};

// -------------------------------------------------------- E-STOP & RESET
elements.btnStop.onclick = () => {
  if (state.wsConnected) {
    state.ws.send(JSON.stringify({ action: "stop" }));
  } else {
    fetch("/api/stop", { method: "POST" });
  }
};

elements.btnReset.onclick = () => {
  if (state.wsConnected) {
    state.ws.send(JSON.stringify({ action: "reset" }));
  } else {
    fetch("/api/reset", { method: "POST" });
  }
};

function showSafetyBanner(title, msg) {
  elements.safetyBanner.classList.remove("hidden");
  elements.safetyBannerTitle.textContent = title;
  elements.safetyBannerMsg.textContent = msg;
}

function hideSafetyBanner() {
  elements.safetyBanner.classList.add("hidden");
}

// -------------------------------------------------------- MODE TOGGLE
elements.btnModeToggle.onclick = () => {
  const current = elements.activeModeText.textContent;
  const next = current.includes("CAM") ? "Open-Loop" : "Camera-Guided";
  setOperatingMode(next);
};

function setOperatingMode(mode) {
  if (state.wsConnected) {
    state.ws.send(JSON.stringify({ action: "set_mode", mode: mode }));
  } else {
    fetch("/api/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: mode }),
    });
  }

  if (mode.includes("Cam")) {
    elements.btnModeToggle.className = "mode-pill camera-active";
    elements.activeModeText.textContent = "CAMERA-GUIDED";
    document.getElementById("row-camera-guided").classList.add("highlight-row");
    document.getElementById("row-open-loop").classList.remove("highlight-row");
  } else {
    elements.btnModeToggle.className = "mode-pill openloop-active";
    elements.activeModeText.textContent = "OPEN-LOOP";
    document.getElementById("row-open-loop").classList.add("highlight-row");
    document.getElementById("row-camera-guided").classList.remove("highlight-row");
  }
}

// -------------------------------------------------------- ARM STATUS & TELEMETRY
function updateArmState(data) {
  if (data.packet_counter !== undefined) {
    elements.packetCounterBadge.textContent = `PKT #${data.packet_counter}`;
  }

  const step = data.task_step || data.step;
  if (step) {
    elements.currentStepLabel.textContent = step;
    elements.stepDescription.textContent = data.step_message || data.message || `Step: ${step}`;
    updateStepTrail(step);
  }

  if (data.joints) {
    const deg = data.joints.deg;
    const rad = data.joints.rad;

    if (deg) {
      elements.valBase.textContent = `${deg.base.toFixed(1)}°`;
      elements.valShoulder.textContent = `${deg.shoulder.toFixed(1)}°`;
      elements.valElbow.textContent = `${deg.elbow.toFixed(1)}°`;
      elements.valWrist.textContent = `${deg.wrist.toFixed(1)}°`;
      elements.valGripper.textContent = `${deg.gripper.toFixed(1)}°`;

      elements.gaugeBase.style.width = `${Math.min(100, Math.max(0, (deg.base - 30) / 1.2))}%`;
      elements.gaugeShoulder.style.width = `${Math.min(100, Math.max(0, (deg.shoulder - 60) / 0.6))}%`;
      elements.gaugeElbow.style.width = `${Math.min(100, Math.max(0, (deg.elbow - 45) / 0.9))}%`;
      elements.gaugeWrist.style.width = `${Math.min(100, Math.max(0, deg.wrist / 1.8))}%`;
      elements.gaugeGripper.style.width = `${Math.min(100, Math.max(0, (deg.gripper - 70) / 0.4))}%`;

      state.arm.joints = data.joints;
      drawArmSchematic(deg.shoulder, deg.elbow, deg.gripper);
    }

    if (rad) {
      elements.radBase.textContent = `(${rad.base.toFixed(2)} rad)`;
      elements.radShoulder.textContent = `(${rad.shoulder.toFixed(2)} rad)`;
      elements.radElbow.textContent = `(${rad.elbow.toFixed(2)} rad)`;
      elements.radWrist.textContent = `(${rad.wrist.toFixed(2)} rad)`;
      elements.radGripper.textContent = deg.gripper > 90 ? "OPEN" : "CLOSED";
    }
  }
}

function updateStepTrail(activeStep) {
  const nodes = document.querySelectorAll(".trail-node");
  nodes.forEach((node) => {
    const s = node.getAttribute("data-step");
    if (s === activeStep) {
      node.className = "trail-node active";
    } else {
      node.className = "trail-node";
    }
  });
}

function updateWatchdog(data) {
  const remaining = Math.max(0, data.timeout_ms - data.elapsed_ms);
  const status = data.status || "HEALTHY";

  if (status === "HEALTHY") {
    elements.watchdogState.textContent = `HEALTHY (${remaining}ms)`;
    elements.watchdogState.className = "status-val safe";
  } else if (status === "WARNING") {
    elements.watchdogState.textContent = `WARN (${remaining}ms)`;
    elements.watchdogState.className = "status-val warn";
  } else {
    elements.watchdogState.textContent = `TIMEOUT (${data.elapsed_ms}ms)`;
    elements.watchdogState.className = "status-val danger";
  }
}

// -------------------------------------------------------- ARM 2D SCHEMATIC PREVIEW
function drawArmSchematic(shoulderDeg, elbowDeg, gripperDeg) {
  const canvas = elements.armSchematicCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Table baseline
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(10, h - 15);
  ctx.lineTo(w - 10, h - 15);
  ctx.stroke();

  // Arm kinematics parameters
  const originX = 60;
  const originY = h - 15;
  const L1 = 30; // base riser
  const L2 = 65; // upper arm
  const L3 = 60; // forearm
  const L4 = 25; // tool/gripper

  // Angles relative to horizontal
  const th1 = ((shoulderDeg - 90) * Math.PI) / 180 + Math.PI / 2.5;
  const th2 = th1 + ((elbowDeg - 90) * Math.PI) / 180 - Math.PI / 3;

  const j0X = originX;
  const j0Y = originY - L1;

  const j1X = j0X + L2 * Math.cos(th1);
  const j1Y = j0Y - L2 * Math.sin(th1);

  const j2X = j1X + L3 * Math.cos(th2);
  const j2Y = j1Y - L3 * Math.sin(th2);

  const tipX = j2X + L4;
  const tipY = j2Y;

  // Base mount
  ctx.fillStyle = "#64748b";
  ctx.fillRect(originX - 14, originY - 10, 28, 10);

  // Link 1
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 8;
  ctx.beginPath();
  ctx.moveTo(originX, originY);
  ctx.lineTo(j0X, j0Y);
  ctx.stroke();

  // Link 2 (Upper arm)
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.moveTo(j0X, j0Y);
  ctx.lineTo(j1X, j1Y);
  ctx.stroke();

  // Link 3 (Forearm)
  ctx.strokeStyle = "#0ea5e9";
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(j1X, j1Y);
  ctx.lineTo(j2X, j2Y);
  ctx.stroke();

  // Gripper
  ctx.strokeStyle = gripperDeg > 90 ? "#a855f7" : "#ef4444";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(j2X, j2Y);
  ctx.lineTo(tipX, tipY);
  ctx.stroke();

  // Pivot joints
  const joints = [
    [originX, originY],
    [j0X, j0Y],
    [j1X, j1Y],
    [j2X, j2Y],
    [tipX, tipY],
  ];
  joints.forEach(([x, y]) => {
    ctx.fillStyle = "#f8fafc";
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  });
}

// -------------------------------------------------------- PRODUCTION METRICS
function updateStats(data) {
  if (data.parts_sorted !== undefined) elements.statSorted.textContent = data.parts_sorted;
  if (data.rejects !== undefined) elements.statRejects.textContent = data.rejects;
  if (data.pick_success_rate !== undefined) elements.statSuccessRate.textContent = `${data.pick_success_rate.toFixed(1)}%`;
  if (data.avg_cycle_time_s !== undefined) elements.statCycleTime.textContent = `${data.avg_cycle_time_s.toFixed(2)}s`;
  if (data.last_cycle_time_s !== undefined) elements.statLastCycle.textContent = `${data.last_cycle_time_s.toFixed(2)}s`;

  if (data.open_loop) {
    const ol = data.open_loop;
    elements.olAttempts.textContent = ol.attempts;
    elements.olSuccess.textContent = ol.successes;
    elements.olRejects.textContent = ol.rejects;
    elements.olAccuracy.textContent = `${ol.accuracy_pct.toFixed(1)}%`;
    elements.olCycle.textContent = `${ol.avg_cycle_time_s.toFixed(2)}s`;
  }

  if (data.camera_guided) {
    const cg = data.camera_guided;
    elements.cgAttempts.textContent = cg.attempts;
    elements.cgSuccess.textContent = cg.successes;
    elements.cgRejects.textContent = cg.rejects;
    elements.cgAccuracy.textContent = `${cg.accuracy_pct.toFixed(1)}%`;
    elements.cgCycle.textContent = `${cg.avg_cycle_time_s.toFixed(2)}s`;
  }

  if (data.ina219) {
    updateIna219(data.ina219);
  }
}

function updateIna219(data) {
  if (data.current_ma !== undefined) elements.inaCurrent.textContent = `${data.current_ma.toFixed(1)} mA`;
  if (data.voltage_v !== undefined) elements.inaVolt.textContent = `${data.voltage_v.toFixed(2)}V`;
  if (data.power_mw !== undefined) elements.inaPower.textContent = `${data.power_mw.toFixed(0)} mW`;

  if (data.overcurrent) {
    elements.inaCurrent.classList.add("text-danger");
    elements.inaCurrent.classList.remove("text-warning");
  } else {
    elements.inaCurrent.classList.remove("text-danger");
    elements.inaCurrent.classList.add("text-warning");
  }
}

// -------------------------------------------------------- VISION & TARGETING
window.toggleCameraSource = function (mode) {
  state.cameraMode = mode;
  if (mode === "webcam") {
    elements.simulatedFeed.classList.add("hidden");
    elements.webcamVideo.classList.remove("hidden");
    elements.hudSource.textContent = "LIVE WEBCAM (LOCAL)";
    startWebcam();
  } else {
    elements.webcamVideo.classList.add("hidden");
    elements.simulatedFeed.classList.remove("hidden");
    elements.hudSource.textContent = "SIMULATED BENCH";
    stopWebcam();
  }
};

function startWebcam() {
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 640, height: 480 } })
      .then((stream) => {
        state.webcamStream = stream;
        elements.webcamVideo.srcObject = stream;
      })
      .catch((err) => {
        console.warn("Could not access browser webcam:", err);
        alert("Webcam permission denied or camera unavailable. Switched to simulated bench feed.");
        toggleCameraSource("simulated");
        document.getElementById("camera-source-select").value = "simulated";
      });
  }
}

function stopWebcam() {
  if (state.webcamStream) {
    state.webcamStream.getTracks().forEach((track) => track.stop());
    state.webcamStream = null;
  }
}

function updateTargetChips() {
  const container = elements.targetsChipsContainer;
  container.innerHTML = "";

  state.detections.forEach((obj) => {
    const chip = document.createElement("button");
    const isSel = obj.id === state.selectedTargetId || obj.selected;
    chip.className = `target-chip ${isSel ? "selected" : ""}`;
    chip.innerHTML = `
      <span class="chip-color-dot" style="background:${obj.color}"></span>
      <span>${obj.label}</span>
      <span style="opacity:0.6">(${obj.confidence.toFixed(2)})</span>
    `;
    chip.onclick = () => selectTarget(obj.id);
    container.appendChild(chip);

    if (isSel) {
      elements.hudTargetName.textContent = obj.label.toUpperCase();
    }
  });

  elements.hudCount.textContent = `${state.detections.length} OBJECTS`;
}

function selectTarget(targetId) {
  state.selectedTargetId = targetId;
  if (state.wsConnected) {
    state.ws.send(JSON.stringify({ action: "select_target", target_id: targetId }));
  } else {
    fetch("/api/vision/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ target_id: targetId }),
    });
  }
  updateTargetChips();
}

// Canvas detection overlay click
elements.visionCanvas.addEventListener("click", (evt) => {
  const rect = elements.visionCanvas.getBoundingClientRect();
  const scaleX = 640 / rect.width;
  const scaleY = 480 / rect.height;
  const clickX = (evt.clientX - rect.left) * scaleX;
  const clickY = (evt.clientY - rect.top) * scaleY;

  for (const obj of state.detections) {
    const [x1, y1, x2, y2] = obj.bbox;
    if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
      selectTarget(obj.id);
      break;
    }
  }
});

// Render dynamic canvas overlay loop
function renderVisionOverlay() {
  const canvas = elements.visionCanvas;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  state.detections.forEach((obj) => {
    const [x1, y1, x2, y2] = obj.bbox;
    const isSelected = obj.id === state.selectedTargetId || obj.selected;

    // Box
    ctx.lineWidth = isSelected ? 3 : 2;
    ctx.strokeStyle = isSelected ? "#00f2fe" : "rgba(255, 255, 255, 0.7)";
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Label tag
    ctx.font = isSelected ? "bold 13px 'JetBrains Mono'" : "11px 'JetBrains Mono'";
    ctx.fillStyle = isSelected ? "#00f2fe" : "#ffffff";
    const tag = `${obj.label} ${obj.confidence.toFixed(2)}`;
    ctx.fillText(tag, x1, Math.max(16, y1 - 6));

    // Target reticle
    if (isSelected) {
      const cx = (x1 + x2) / 2;
      const cy = (y1 + y2) / 2;

      ctx.strokeStyle = "#00f2fe";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 18, 0, Math.PI * 2);
      ctx.moveTo(cx - 24, cy);
      ctx.lineTo(cx + 24, cy);
      ctx.moveTo(cx, cy - 24);
      ctx.lineTo(cx, cy + 24);
      ctx.stroke();

      ctx.fillStyle = "#00f2fe";
      ctx.fillText(`TARGET LOCKED`, x1, y2 + 16);
    }
  });

  requestAnimationFrame(renderVisionOverlay);
}

// -------------------------------------------------------- EVENT LOG
window.filterLogs = function (cat) {
  state.activeFilter = cat;
  document.querySelectorAll(".filter-btn").forEach((btn) => {
    if (btn.getAttribute("data-filter") === cat) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });
  renderEvents();
};

function renderEvents() {
  const container = elements.eventStream;
  if (!container) return;

  const filter = state.activeFilter;
  const filtered = state.events.filter((ev) => {
    if (filter === "ALL") return true;
    if (filter === "REJECTIONS") return ev.event === "safety:rejected" || (ev.payload && ev.payload.status === "rejected");
    return ev.event.startsWith(filter);
  });

  container.innerHTML = "";
  filtered.slice(-40).forEach((ev) => {
    const row = document.createElement("div");
    const isRejection = ev.event === "safety:rejected" || (ev.payload && ev.payload.status === "rejected");
    const isEstop = ev.event === "safety:estop" || ev.event === "arm:stop";

    let css = "event-row";
    if (isRejection) css += " log-rejection";
    else if (isEstop) css += " log-estop";
    else if (ev.event.startsWith("safety:")) css += " log-safety";
    else if (ev.event.startsWith("arm:") || ev.event.startsWith("planner:")) css += " log-arm";

    row.className = css;

    const timeStr = ev.timestamp ? ev.timestamp.substring(11, 23) : new Date().toISOString().substring(11, 23);
    const p = ev.payload || {};
    let msg = "";

    if (ev.event === "safety:rejected") {
      msg = `REJECTED: ${p.reason || "Safety check failed"}`;
    } else if (ev.event === "safety:estop") {
      msg = `EMERGENCY STOP COMMANDED: ${p.reason || "Operator triggered"}`;
    } else if (ev.event === "planner:step_changed") {
      msg = `Step ${p.index || 0}/${p.total || 0} [${p.step}]: ${p.message || ""}`;
    } else if (ev.event === "nlp:parsed") {
      msg = `Parsed: status=${p.status} | "${p.text}"`;
    } else {
      msg = JSON.stringify(p);
    }

    row.innerHTML = `
      <span class="log-time">${timeStr}</span>
      <span class="log-tag">${ev.event}</span>
      <span class="log-msg">${msg}</span>
    `;

    container.appendChild(row);
  });

  const parent = elements.eventStream.parentElement;
  parent.scrollTop = parent.scrollHeight;
}

// -------------------------------------------------------- EXPORTS & DOCS
window.exportJsonl = function () {
  window.location.href = "/api/logs/download";
};

window.openDocsModal = function () {
  elements.docsModal.classList.remove("hidden");
};

window.closeDocsModal = function () {
  elements.docsModal.classList.add("hidden");
};

// -------------------------------------------------------- INITIALIZATION
window.addEventListener("DOMContentLoaded", () => {
  connectWebSocket();
  renderVisionOverlay();
  drawArmSchematic(90, 90, 90);

  // Initial detections setup
  fetch("/api/status")
    .then((r) => r.json())
    .then((d) => {
      if (d.arm) updateArmState(d.arm);
      if (d.stats) updateStats(d.stats);
      if (d.vision && d.vision.detections) {
        state.detections = d.vision.detections;
        updateTargetChips();
      }
    })
    .catch((err) => console.log("Initial status fetch error:", err));
});
