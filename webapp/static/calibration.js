(() => {
  "use strict";

  // Calibration page: assign the line camera, show its live preview with the
  // saved ROI on top, capture a frame and draw/edit the ROI polygon.
  // Based on powder_line_vision's calibration.js (same flow, one camera,
  // polygon instead of rectangles). Polygon corners are kept as fractions of
  // the frame (0..1), the same format roi.json and monitor.py use.

  const root = document.querySelector('.page[data-page="calibration"]');
  if (!root) return;
  const q = (r) => root.querySelector(`[data-role="${r}"]`);

  const camLabel = q("calib-cam-label");
  const reassignBtn = q("reassign-btn");
  const assignPanel = q("device-assign-panel");
  const assignMsg = q("device-assign-message");
  const detectBtn = q("detect-cameras-btn");
  const cancelAssignBtn = q("cancel-assign-btn");
  const deviceGrid = q("device-grid");
  const mjpeg = q("calib-mjpeg-src");
  const canvas = q("calib-canvas");
  const placeholder = q("calib-placeholder");
  const modeBadge = q("calib-mode-badge");
  const modeText = q("calib-mode-text");
  const captureBtn = q("capture-btn");
  const undoBtn = q("undo-btn");
  const clearBtn = q("clear-btn");
  const cancelDrawBtn = q("cancel-draw-btn");
  const saveBtn = q("save-roi-btn");
  const stepStatus = q("calib-step-status");
  const errorBox = q("calib-error");
  const ctx = canvas.getContext("2d");

  const HANDLE_PX = 7; // corner handle radius, in screen pixels
  const HIT_PX = 12; // how close a click must be to grab a corner / edge

  const state = {
    visible: false,
    cameraIndex: null,
    sessionOpen: false,
    mode: "none", // none | live | drawing
    raf: null,
    frozen: null, // captured HTMLImageElement
    points: [], // [[x, y], ...] fractions, being drawn
    saved: null, // saved polygon (fractions) or null
    drag: null, // index of the corner being dragged
  };

  async function api(path, options) {
    const resp = await fetch(path, options);
    let data = {};
    try {
      data = await resp.json();
    } catch (_) {
      /* empty body (204) */
    }
    if (!resp.ok) throw new Error(data.error || `${path} failed (${resp.status})`);
    return data;
  }

  function showError(msg) {
    errorBox.textContent = msg || "";
    errorBox.hidden = !msg;
  }

  function notifyConfigChanged() {
    document.dispatchEvent(new CustomEvent("pad:config-changed"));
  }

  // ============================== Saved ROI panel ==============================
  async function loadSavedRoi() {
    try {
      const roi = await api("/api/calibration/roi");
      state.saved = roi.exists ? roi.polygon : null;
      q("roi-status").textContent = roi.exists ? "Saved" : roi.error ? "Unreadable" : "Not drawn yet";
      q("roi-corners").textContent = roi.exists ? roi.polygon.length : "—";
      q("roi-saved").textContent = roi.saved_at || (roi.exists ? "before the web UI" : "—");
      q("roi-camera").textContent =
        roi.camera_index !== undefined && roi.camera_index !== null ? `index ${roi.camera_index}` : "—";
      if (roi.error) showError(roi.error);
    } catch (err) {
      showError(err.message);
    }
  }

  // ============================== Camera assignment ==============================
  async function refreshAssignment() {
    const dev = await api("/api/calibration/device");
    state.cameraIndex = dev.camera_index;
    camLabel.textContent = dev.test_mode
      ? "TEST MODE: a video file stands in for the camera"
      : dev.camera_index === null ? "not assigned" : `camera index ${dev.camera_index}`;
    if (dev.camera_index === null) {
      reassignBtn.hidden = true;
      cancelAssignBtn.hidden = true;
      assignMsg.textContent = "No camera assigned yet. Detect the cameras on this PC and assign the one that sees the belt.";
      assignPanel.hidden = false;
      placeholder.querySelector("span").textContent = "Assign a camera to begin.";
      await detectCameras();
    } else {
      assignPanel.hidden = true;
      reassignBtn.hidden = false;
      await openSession();
    }
  }

  async function detectCameras() {
    showError("");
    deviceGrid.textContent = "";
    deviceGrid.appendChild(Object.assign(document.createElement("p"), { textContent: "Detecting cameras… (a few seconds)" }));
    detectBtn.disabled = true;
    try {
      const data = await api("/api/calibration/cameras");
      deviceGrid.textContent = "";
      if (!data.cameras.length) {
        deviceGrid.appendChild(Object.assign(document.createElement("p"), {
          textContent: "No cameras found. Check the USB cable and that no other program (e.g. the Camera app) is using it.",
        }));
      }
      data.cameras.forEach((cam) => {
        const card = document.createElement("div");
        card.className = "device-card" + (cam.index === state.cameraIndex ? " device-card-current" : "");
        const img = document.createElement("img");
        img.alt = cam.label;
        if (cam.thumbnail_b64) img.src = `data:image/jpeg;base64,${cam.thumbnail_b64}`;
        const info = document.createElement("div");
        info.textContent = `${cam.label} · ${cam.width}×${cam.height}`;
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "btn btn-primary";
        btn.textContent = cam.index === state.cameraIndex ? "Keep this camera" : "Use this camera";
        btn.addEventListener("click", () => assignCamera(cam.index));
        card.append(img, info, btn);
        deviceGrid.appendChild(card);
      });
    } catch (err) {
      deviceGrid.textContent = "";
      showError(err.message);
    } finally {
      detectBtn.disabled = false;
    }
  }

  async function assignCamera(index) {
    showError("");
    try {
      await api("/api/calibration/device", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ camera_index: index }),
      });
      notifyConfigChanged();
      await refreshAssignment();
    } catch (err) {
      showError(err.message);
    }
  }

  detectBtn.addEventListener("click", detectCameras);
  reassignBtn.addEventListener("click", async () => {
    await teardown();
    assignMsg.textContent = "Pick the camera that sees the belt.";
    assignPanel.hidden = false;
    cancelAssignBtn.hidden = false;
    reassignBtn.hidden = true;
    placeholder.querySelector("span").textContent = "Choosing a camera…";
    await detectCameras();
  });
  cancelAssignBtn.addEventListener("click", () => refreshAssignment().catch((e) => showError(e.message)));

  // ============================== Live preview ==============================
  async function openSession() {
    showError("");
    try {
      const info = await api("/api/calibration/session/open", { method: "POST" });
      state.sessionOpen = true;
      // canvas pixel size = frame aspect, capped so drawing stays light
      const scale = Math.min(1, 1280 / info.width);
      canvas.width = Math.round(info.width * scale);
      canvas.height = Math.round(info.height * scale);
      canvas.hidden = false;
      placeholder.hidden = true;
      captureBtn.disabled = false;
      startLive();
    } catch (err) {
      placeholder.hidden = false;
      placeholder.querySelector("span").textContent = "Camera preview not available.";
      showError(err.message);
    }
  }

  async function startLive() {
    // make sure the camera session is (still) open: it may have been released
    // in the meantime (idle timeout, monitoring started from another tab)
    try {
      await api("/api/calibration/session/open", { method: "POST" });
    } catch (err) {
      showError(err.message);
      return;
    }
    state.mode = "live";
    setDrawingUi(false);
    modeText.textContent = "LIVE PREVIEW";
    modeBadge.hidden = false;
    mjpeg.src = `/api/calibration/preview?t=${Date.now()}`;
    stepStatus.textContent = state.saved
      ? "Live preview. The saved ROI is shown in yellow. Capture a frame to redraw it."
      : "Live preview. Capture a frame to draw the ROI.";
    const loop = () => {
      if (state.mode !== "live") return;
      if (mjpeg.complete && mjpeg.naturalWidth > 0) {
        ctx.drawImage(mjpeg, 0, 0, canvas.width, canvas.height);
        if (state.saved) drawPolygon(state.saved, { color: "#facc15", fill: "rgba(250,204,21,0.12)", handles: false, closed: true });
      }
      state.raf = requestAnimationFrame(loop);
    };
    state.raf = requestAnimationFrame(loop);
  }

  function stopLive() {
    if (state.raf) cancelAnimationFrame(state.raf);
    state.raf = null;
    mjpeg.removeAttribute("src");
  }

  async function teardown() {
    stopLive();
    state.mode = "none";
    setDrawingUi(false);
    modeBadge.hidden = true;
    if (state.sessionOpen) {
      state.sessionOpen = false;
      try {
        await api("/api/calibration/session/close", { method: "POST" });
      } catch (_) {
        /* best effort */
      }
    }
    canvas.hidden = true;
    placeholder.hidden = false;
    captureBtn.disabled = true;
  }

  // ============================== Capture + polygon editing ==============================
  function setDrawingUi(on) {
    captureBtn.hidden = on;
    [undoBtn, clearBtn, cancelDrawBtn, saveBtn].forEach((b) => (b.hidden = !on));
    canvas.classList.toggle("drawing", on);
  }

  captureBtn.addEventListener("click", async () => {
    showError("");
    try {
      const resp = await fetch("/api/calibration/capture", { method: "POST" });
      if (!resp.ok) {
        const data = await resp.json().catch(() => ({}));
        throw new Error(data.error || "capture failed");
      }
      const blob = await resp.blob();
      const img = new Image();
      img.onload = () => {
        stopLive();
        state.frozen = img;
        state.mode = "drawing";
        // start from the saved ROI so small corrections are easy; Clear starts over
        state.points = state.saved ? state.saved.map((p) => [...p]) : [];
        modeText.textContent = "CAPTURED FRAME";
        setDrawingUi(true);
        redraw();
      };
      img.src = URL.createObjectURL(blob);
    } catch (err) {
      showError(err.message);
    }
  });

  function toCanvas([x, y]) {
    return [x * canvas.width, y * canvas.height];
  }

  function pxPerCanvasUnit() {
    // screen pixels per canvas pixel (canvas is scaled by CSS)
    return canvas.getBoundingClientRect().width / canvas.width || 1;
  }

  function drawPolygon(points, { color, fill, handles, closed }) {
    if (!points.length) return;
    const k = 1 / pxPerCanvasUnit();
    ctx.save();
    ctx.lineWidth = 2.5 * k;
    ctx.strokeStyle = color;
    ctx.beginPath();
    points.forEach((p, i) => {
      const [cx, cy] = toCanvas(p);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    if (closed && points.length >= 3) {
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
    }
    ctx.stroke();
    if (handles) {
      points.forEach((p, i) => {
        const [cx, cy] = toCanvas(p);
        ctx.beginPath();
        ctx.arc(cx, cy, HANDLE_PX * k, 0, Math.PI * 2);
        ctx.fillStyle = i === state.drag ? "#f59e0b" : "#ffffff";
        ctx.fill();
        ctx.lineWidth = 2 * k;
        ctx.strokeStyle = color;
        ctx.stroke();
      });
    }
    ctx.restore();
  }

  function redraw() {
    if (state.mode !== "drawing" || !state.frozen) return;
    ctx.drawImage(state.frozen, 0, 0, canvas.width, canvas.height);
    drawPolygon(state.points, { color: "#22c55e", fill: "rgba(34,197,94,0.15)", handles: true, closed: true });
    const n = state.points.length;
    saveBtn.disabled = n < 3;
    undoBtn.disabled = n === 0;
    clearBtn.disabled = n === 0;
    stepStatus.textContent =
      n < 3
        ? `Click to place corners around the belt area (${n} placed, at least 3 needed).`
        : `${n} corners. Drag a corner to move it, right-click to remove it, click on an edge to add one. Then Save ROI.`;
  }

  function eventFraction(evt) {
    const r = canvas.getBoundingClientRect();
    const x = Math.min(Math.max((evt.clientX - r.left) / r.width, 0), 1);
    const y = Math.min(Math.max((evt.clientY - r.top) / r.height, 0), 1);
    return [x, y];
  }

  function screenDist(a, b) {
    const r = canvas.getBoundingClientRect();
    return Math.hypot((a[0] - b[0]) * r.width, (a[1] - b[1]) * r.height);
  }

  function nearestCorner(p) {
    let best = -1;
    let bestD = HIT_PX;
    state.points.forEach((c, i) => {
      const d = screenDist(c, p);
      if (d <= bestD) {
        best = i;
        bestD = d;
      }
    });
    return best;
  }

  function nearestEdge(p) {
    // index to insert at if p is within HIT_PX of an edge of the closed polygon
    const n = state.points.length;
    if (n < 3) return -1;
    const r = canvas.getBoundingClientRect();
    const P = [p[0] * r.width, p[1] * r.height];
    for (let i = 0; i < n; i++) {
      const a = state.points[i];
      const b = state.points[(i + 1) % n];
      const A = [a[0] * r.width, a[1] * r.height];
      const B = [b[0] * r.width, b[1] * r.height];
      const len2 = (B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2;
      if (!len2) continue;
      const t = Math.max(0, Math.min(1, ((P[0] - A[0]) * (B[0] - A[0]) + (P[1] - A[1]) * (B[1] - A[1])) / len2));
      const d = Math.hypot(P[0] - (A[0] + t * (B[0] - A[0])), P[1] - (A[1] + t * (B[1] - A[1])));
      if (d <= HIT_PX && t > 0.05 && t < 0.95) return i + 1;
    }
    return -1;
  }

  canvas.addEventListener("mousedown", (evt) => {
    if (state.mode !== "drawing" || evt.button !== 0) return;
    const p = eventFraction(evt);
    const corner = nearestCorner(p);
    if (corner >= 0) {
      state.drag = corner;
    } else {
      const edge = nearestEdge(p);
      if (edge >= 0) {
        state.points.splice(edge, 0, p);
        state.drag = edge;
      } else {
        state.points.push(p);
      }
    }
    redraw();
  });

  canvas.addEventListener("mousemove", (evt) => {
    if (state.mode !== "drawing") return;
    if (state.drag !== null) {
      state.points[state.drag] = eventFraction(evt);
      redraw();
    } else {
      canvas.style.cursor = nearestCorner(eventFraction(evt)) >= 0 ? "grab" : "crosshair";
    }
  });

  window.addEventListener("mouseup", () => {
    if (state.drag !== null) {
      state.drag = null;
      redraw();
    }
  });

  canvas.addEventListener("contextmenu", (evt) => {
    if (state.mode !== "drawing") return;
    evt.preventDefault();
    const corner = nearestCorner(eventFraction(evt));
    if (corner >= 0) {
      state.points.splice(corner, 1);
      redraw();
    }
  });

  undoBtn.addEventListener("click", () => {
    state.points.pop();
    redraw();
  });
  clearBtn.addEventListener("click", () => {
    state.points = [];
    redraw();
  });
  cancelDrawBtn.addEventListener("click", () => {
    state.frozen = null;
    startLive();
  });

  saveBtn.addEventListener("click", async () => {
    showError("");
    saveBtn.disabled = true;
    try {
      const polygon = state.points.map(([x, y]) => [Number(x.toFixed(4)), Number(y.toFixed(4))]);
      await api("/api/calibration/roi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ polygon }),
      });
      await loadSavedRoi();
      notifyConfigChanged();
      state.frozen = null;
      startLive();
      stepStatus.textContent = "ROI saved. The next Start on Live Monitoring uses it.";
    } catch (err) {
      saveBtn.disabled = false;
      showError(err.message);
    }
  });

  // ============================== Page lifecycle ==============================
  async function onShow() {
    state.visible = true;
    showError("");
    await loadSavedRoi();
    try {
      const status = await api("/api/monitor/status");
      if (["running", "starting", "reconnecting"].includes(status.state)) {
        const dev = await api("/api/calibration/device");
        camLabel.textContent = dev.camera_index === null ? "not assigned" : `camera index ${dev.camera_index}`;
        assignPanel.hidden = true;
        reassignBtn.hidden = true;
        placeholder.hidden = false;
        placeholder.querySelector("span").textContent = "Monitoring is using the camera.";
        stepStatus.textContent = "Stop monitoring on the Live Monitoring page to change the camera or the ROI.";
        return;
      }
      await refreshAssignment();
    } catch (err) {
      showError(err.message);
    }
  }

  async function onHide() {
    state.visible = false;
    await teardown();
  }

  // free the camera if the tab is closed while on this page
  window.addEventListener("pagehide", () => {
    if (state.sessionOpen) navigator.sendBeacon("/api/calibration/session/close");
  });

  window.PadCalibration = { onShow, onHide };
})();
