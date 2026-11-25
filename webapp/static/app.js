(() => {
  "use strict";

  const STATUS_POLL_MS = 1000;
  const OVERVIEW_REFRESH_MS = 10000;
  const $ = (sel, root = document) => root.querySelector(sel);
  const role = (name) => $(`[data-role="${name}"]`);

  async function getJSON(url, options) {
    const resp = await fetch(url, options);
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || `${url} failed (${resp.status})`);
    return data;
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  // ============================== Navigation ==============================
  const navButtons = document.querySelectorAll(".nav-item[data-page]");
  const pages = document.querySelectorAll(".page[data-page]");
  let currentPage = "overview";

  function showPage(name) {
    if (currentPage === "calibration" && name !== "calibration" && window.PadCalibration) {
      window.PadCalibration.onHide(); // release the camera preview
    }
    navButtons.forEach((b) => b.classList.toggle("active", b.dataset.page === name));
    pages.forEach((p) => (p.hidden = p.dataset.page !== name));
    currentPage = name;
    if (name === "overview") loadOverview();
    if (name === "analytics") loadAnalytics();
    if (name === "live") loadConfig();
    if (name === "calibration" && window.PadCalibration) window.PadCalibration.onShow();
  }
  navButtons.forEach((b) => b.addEventListener("click", () => showPage(b.dataset.page)));

  // ============================== Shared: batch table ==============================
  function resultChip(result) {
    return el("span", `result-chip ${result === "NC" ? "result-nc" : "result-ok"}`, result);
  }

  function snapshotCell(row) {
    const td = el("td");
    if (row.snapshot) {
      const a = el("a");
      a.href = `/snapshots/${encodeURIComponent(row.snapshot)}`;
      a.target = "_blank";
      a.rel = "noopener";
      const img = el("img", "snap-thumb");
      img.src = a.href;
      img.alt = `NC snapshot, batch ${row.batch_no}`;
      img.loading = "lazy";
      a.appendChild(img);
      td.appendChild(a);
    } else {
      td.textContent = "—";
    }
    return td;
  }

  function renderBatchTable(container, rows, emptyText, compact) {
    container.textContent = "";
    if (!rows.length) {
      container.appendChild(el("div", "table-empty", emptyText));
      return;
    }
    const wrap = el("div", "table-scroll");
    const table = el("table", "data-table");
    const head = el("tr");
    const cols = compact
      ? ["TIME", "SOURCE", "BATCH", "PADS", "LEFT", "REASON", "SNAPSHOT"]
      : ["TIME", "SOURCE", "BATCH", "PADS", "PULLED", "LEFT", "RESULT", "REASON", "SNAPSHOT"];
    cols.forEach((c) => head.appendChild(el("th", null, c)));
    const thead = el("thead");
    thead.appendChild(head);
    table.appendChild(thead);
    const tbody = el("tbody");
    rows.forEach((r) => {
      const tr = el("tr");
      tr.appendChild(el("td", "num", r.ts));
      tr.appendChild(el("td", null, r.source || "—"));
      tr.appendChild(el("td", "num", r.batch_no));
      tr.appendChild(el("td", "num", r.pads_in));
      if (!compact) tr.appendChild(el("td", "num", r.pulled));
      tr.appendChild(el("td", "num", r.leftover));
      if (!compact) {
        const td = el("td");
        td.appendChild(resultChip(r.result));
        tr.appendChild(td);
      }
      tr.appendChild(el("td", null, r.reason || "—"));
      tr.appendChild(snapshotCell(r));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    container.appendChild(wrap);
  }

  function fillStats(prefix, t) {
    role(`${prefix}-batches`).textContent = t.batches;
    role(`${prefix}-pads`).textContent = t.pads;
    role(`${prefix}-pulled`).textContent = t.pulled;
    const nc = role(`${prefix}-nc`);
    nc.textContent = t.nc;
    nc.classList.toggle("has-nc", t.nc > 0);
    role(`${prefix}-batches-sub`).textContent = t.batches ? `${t.ok} OK · ${t.nc} NC` : "";
    role(`${prefix}-pulled-sub`).textContent = t.left ? `${t.left} left behind` : t.pads ? "none left behind" : "";
    role(`${prefix}-nc-sub`).textContent = t.batches ? `${t.nc_rate}% of batches` : "";
  }

  function drawCharts(prefix, data) {
    PadCharts.renderBarChart(role(`${prefix}-chart-batches`), {
      labels: data.labels, a: data.batches.ok, b: data.batches.nc, names: ["OK", "NC"],
    });
    PadCharts.renderBarChart(role(`${prefix}-chart-pads`), {
      labels: data.labels, a: data.pads.pulled, b: data.pads.left, names: ["Pulled", "Left behind"],
      empty: "No pads recorded yet for this period.",
    });
  }

  // ============================== Overview ==============================
  async function loadOverview() {
    try {
      const data = await getJSON("/api/overview");
      fillStats("ov", data.totals);
      drawCharts("ov", data);
      renderBatchTable(role("ov-nc-table"), data.recent_nc, "No NC batches recorded. 👍", true);
    } catch (err) {
      console.error("overview load failed", err); // keep last values on screen
    }
  }
  setInterval(() => currentPage === "overview" && loadOverview(), OVERVIEW_REFRESH_MS);

  // ============================== Live Monitoring ==============================
  const startBtn = role("start-btn");
  const stopBtn = role("stop-btn");
  const processText = role("process-text");
  const liveError = role("live-error");
  const stream = role("stream");
  const placeholder = role("placeholder");
  const liveBadge = role("live-badge");
  const statusBadge = role("status-badge");

  // What Start will use (read-only here; changed on the Calibration page)
  let setupReady = false;

  function setSetup(roleName, text, ok) {
    const item = role(roleName);
    item.querySelector("span").textContent = text;
    item.classList.toggle("setup-missing", !ok);
  }

  async function loadConfig() {
    try {
      const cfg = await getJSON("/api/config");
      const camOk = cfg.camera_index !== null && cfg.camera_index !== undefined;
      setSetup("setup-camera",
        cfg.test_video ? "TEST VIDEO" : camOk ? `index ${cfg.camera_index}` : "not assigned", camOk || !!cfg.test_video);
      setSetup("setup-roi",
        cfg.roi.exists ? `${cfg.roi.polygon.length} corners${cfg.roi.saved_at ? " · saved " + cfg.roi.saved_at : ""}` : "not drawn",
        cfg.roi.exists);
      setSetup("setup-model", cfg.model || "no trained model", !!cfg.model);
      setupReady = (camOk || !!cfg.test_video) && cfg.roi.exists && !!cfg.model;
      renderStatus(status);
    } catch (err) {
      showLiveError(err.message);
    }
  }
  document.addEventListener("pad:config-changed", loadConfig);
  role("goto-calibration").addEventListener("click", () => showPage("calibration"));

  function showLiveError(msg) {
    liveError.textContent = msg || "";
    liveError.hidden = !msg;
  }

  let streaming = false;
  let busy = false;
  let lastBatches = -1;
  let status = { state: "idle", events: [], batches: 0, pads: 0, pulled: 0, nc: 0, last_nc: null };

  function setStream(on) {
    if (on === streaming) return;
    streaming = on;
    if (on) {
      stream.src = `/api/watch?t=${Date.now()}`;
    } else {
      stream.removeAttribute("src");
    }
    stream.hidden = !on;
    placeholder.hidden = on;
    liveBadge.hidden = !on;
  }

  const STATE_LABEL = {
    idle: ["Idle", ""], starting: ["Starting…", "processing"], running: ["Live", "processing"],
    reconnecting: ["Reconnecting…", "error"], stopped: ["Stopped", ""], error: ["Error", "error"],
  };

  function renderStatus(s) {
    const running = ["running", "starting", "reconnecting"].includes(s.state);
    const [label, cls] = STATE_LABEL[s.state] || [s.state, ""];
    statusBadge.textContent = label;
    statusBadge.className = "status-badge" + (cls ? " " + cls : "");

    const pill = role("feed-pill");
    pill.classList.toggle("live", s.state === "running");
    pill.classList.toggle("error", s.state === "error" || s.state === "reconnecting");
    role("feed-text").textContent =
      s.state === "running" ? `Monitoring camera ${s.camera_index}`
      : s.state === "reconnecting" ? "Camera lost - reconnecting"
      : s.state === "starting" ? "Starting…"
      : s.state === "error" ? "Monitor error" : "Monitor idle";
    role("nav-live-dot").hidden = !running;

    setStream(s.state === "running" || (s.state === "reconnecting" && streaming));
    startBtn.disabled = busy || running || !setupReady;
    stopBtn.disabled = busy || !running;

    if (!busy) {
      if (s.state === "running") processText.textContent = s.notice
        ? `⚠ ${s.notice}`
        : `Camera ${s.camera_index} · model ${s.model} · since ${(s.started_at || "").slice(11)}`;
      else if (s.state === "starting") processText.textContent = "Loading the model and opening the camera… (the first start can take a minute or two)";
      else if (s.state === "reconnecting") processText.textContent = s.notice || "Reconnecting to the camera…";
      else if (s.state === "stopped") processText.textContent = "Stopped.";
      else if (!setupReady) processText.textContent = "Finish the setup in Calibration before starting.";
      else processText.textContent = "";
    }
    if (s.state === "error") showLiveError(s.error);

    const chip = role("mon-state");
    const ms = s.monitor_state || "—";
    chip.textContent = ms;
    chip.className = "state-chip" + (ms.startsWith("WAIT") ? " s-approach" : ms === "PULLING" ? " s-pulling" : ms.startsWith("POST") ? " s-verify" : "");
    role("mon-pads-roi").textContent = running ? s.pads_in_roi : "—";
    role("mon-puller").textContent = running ? s.puller_score.toFixed(2) : "—";
    role("mon-fps").textContent = running && s.fps ? `${s.fps} fps` : "—";
    role("run-batches").textContent = s.batches;
    role("run-pads").textContent = s.pads;
    role("run-pulled").textContent = s.pulled;
    const nc = role("run-nc");
    nc.textContent = s.nc;
    nc.classList.toggle("has-nc", s.nc > 0);

    const log = role("event-log");
    log.textContent = "";
    if (!s.events.length) log.appendChild(el("li", "event-empty", "No events yet."));
    s.events.forEach((e) => {
      const li = el("li");
      li.appendChild(el("span", "ev-time", e.t));
      const cls = / NC /.test(e.text) ? "ev-nc" : / OK /.test(e.text) ? "ev-ok" : null;
      li.appendChild(el("span", cls, e.text.replace(/^\s*[\d.]+s\s+/, "")));
      log.appendChild(li);
    });

    updateNcModal(s);

    // a new batch was judged -> refresh whichever dashboard is open
    if (s.batches !== lastBatches) {
      if (lastBatches !== -1 && currentPage === "overview") loadOverview();
      if (lastBatches !== -1 && currentPage === "analytics") loadAnalytics();
      lastBatches = s.batches;
    }
  }

  async function pollStatus() {
    try {
      status = await getJSON("/api/monitor/status");
      renderStatus(status);
    } catch (err) {
      console.error("status poll failed", err);
    }
  }

  async function control(action) {
    busy = true;
    showLiveError("");
    processText.textContent = action === "start" ? "Starting…" : "Stopping…";
    startBtn.disabled = stopBtn.disabled = true;
    try {
      await getJSON(`/api/monitor/${action}`, { method: "POST" });
    } catch (err) {
      showLiveError(err.message);
    } finally {
      busy = false;
      pollStatus();
    }
  }
  startBtn.addEventListener("click", () => control("start"));
  stopBtn.addEventListener("click", () => control("stop"));

  // ============================== NC alert ==============================
  const ncModal = role("nc-modal");
  let shownNcKey = null;

  function updateNcModal(s) {
    const nc = s.last_nc;
    if (!nc || nc.acknowledged) {
      ncModal.hidden = true;
      return;
    }
    const key = `${s.started_at}/${nc.batch}`;
    if (key === shownNcKey && !ncModal.hidden) return;
    shownNcKey = key;
    role("nc-reason").textContent = `Batch ${nc.batch} at ${nc.time}: ${nc.reason}.`;
    role("nc-detail").textContent = `Pads in batch: ${nc.pads_in} · left behind: ${nc.leftover}`;
    const img = role("nc-snapshot");
    if (nc.snapshot) {
      // the snapshot is written in the background, so retry briefly if it isn't on disk yet
      let tries = 0;
      img.onerror = () => {
        if (++tries <= 5) setTimeout(() => (img.src = `/snapshots/${encodeURIComponent(nc.snapshot)}?r=${tries}`), 800);
      };
      img.src = `/snapshots/${encodeURIComponent(nc.snapshot)}`;
      img.hidden = false;
    } else {
      img.hidden = true;
    }
    role("nc-error").hidden = true;
    ncModal.hidden = false;
    role("nc-ack-btn").focus();
  }

  role("nc-ack-btn").addEventListener("click", async () => {
    const nc = status.last_nc;
    if (!nc) return (ncModal.hidden = true);
    try {
      await getJSON("/api/monitor/ack", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ batch: nc.batch }),
      });
      ncModal.hidden = true;
      pollStatus();
    } catch (err) {
      const e = role("nc-error");
      e.textContent = err.message;
      e.hidden = false;
    }
  });

  // ============================== Analytics ==============================
  function todayISO() {
    const now = new Date();
    return new Date(now - now.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
  }

  const an = { range: "day", mode: "daily" };
  const rangeButtons = document.querySelectorAll('[data-role="range-select"] .segmented-btn');
  const modeButtons = document.querySelectorAll('[data-role="mode-select"] .segmented-btn');
  const inputs = {
    date: role("input-date"), month: role("input-month"), from: role("input-from"), to: role("input-to"),
  };
  inputs.date.value = inputs.from.value = inputs.to.value = todayISO();
  inputs.month.value = todayISO().slice(0, 7);

  function updateFields() {
    role("field-date").hidden = !(an.range === "day" || an.range === "week");
    role("field-month").hidden = an.range !== "month";
    role("field-from").hidden = role("field-to").hidden = an.range !== "custom";
    role("mode-select").hidden = !(an.range === "week" || an.range === "month");
  }

  rangeButtons.forEach((btn) =>
    btn.addEventListener("click", () => {
      rangeButtons.forEach((b) => b.classList.toggle("active", b === btn));
      an.range = btn.dataset.range;
      loadAnalytics();
    })
  );
  modeButtons.forEach((btn) =>
    btn.addEventListener("click", () => {
      modeButtons.forEach((b) => b.classList.toggle("active", b === btn));
      an.mode = btn.dataset.mode;
      loadAnalytics();
    })
  );
  let debounce = null;
  Object.values(inputs).forEach((i) =>
    i.addEventListener("change", () => {
      clearTimeout(debounce);
      debounce = setTimeout(loadAnalytics, 150);
    })
  );

  function analyticsQuery() {
    const p = new URLSearchParams({ range: an.range });
    if (an.range === "day" || an.range === "week") p.set("date", inputs.date.value || todayISO());
    if (an.range === "month") p.set("date", `${inputs.month.value || todayISO().slice(0, 7)}-01`);
    if (an.range === "week" || an.range === "month") p.set("mode", an.mode);
    if (an.range === "custom") {
      p.set("from", inputs.from.value || todayISO());
      p.set("to", inputs.to.value || todayISO());
    }
    return p.toString();
  }

  let anSeq = 0;
  async function loadAnalytics() {
    updateFields();
    const err = role("analytics-error");
    err.hidden = true;
    const q = analyticsQuery();
    role("csv-link").href = `/api/analytics.csv?${q}`;
    const id = ++anSeq;
    try {
      const data = await getJSON(`/api/analytics?${q}`);
      if (id !== anSeq) return; // a newer filter change owns the screen
      fillStats("an", data.totals);
      drawCharts("an", data);
      role("an-table-sub").textContent =
        `${data.from}${data.to !== data.from ? " → " + data.to : ""}` +
        (data.rows_total > data.rows.length ? ` · showing latest ${data.rows.length} of ${data.rows_total}` : "");
      renderBatchTable(role("an-table"), data.rows, "No batches recorded for this period.", false);
    } catch (e) {
      err.textContent = e.message;
      err.hidden = false;
    }
  }

  // ============================== Start ==============================
  updateFields();
  loadConfig();
  loadOverview();
  pollStatus();
  setInterval(pollStatus, STATUS_POLL_MS);
})();
