(() => {
  "use strict";

  // Grouped two-series bar chart built from plain DOM/CSS (no charting library,
  // no CDN). Adapted from powder_line_vision's charts.js; the only change is
  // that the two series can be named, e.g. OK / NC or Pulled / Left behind.
  const TICK_COUNT = 4;

  function niceMax(rawMax) {
    if (rawMax <= 0) return 4;
    const exp = Math.floor(Math.log10(rawMax));
    const base = Math.pow(10, exp);
    const f = rawMax / base;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * base;
  }

  function formatTick(v) {
    return Number(v.toFixed(2)).toString();
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  /**
   * @param {HTMLElement} container
   * @param {{labels: string[], a: number[], b: number[], names?: [string, string], empty?: string}} data
   *   a = good series (green), b = bad series (red)
   */
  function renderBarChart(container, data) {
    container.textContent = "";
    const { labels, a, b } = data;
    const [nameA, nameB] = data.names || ["OK", "NC"];

    const total = a.reduce((s, v) => s + v, 0) + b.reduce((s, v) => s + v, 0);
    if (!labels.length || total === 0) {
      container.appendChild(el("div", "chart-empty", data.empty || "No batches recorded yet for this period."));
      return;
    }

    const axisMax = niceMax(Math.max(1, ...a, ...b));
    const plot = el("div", "chart-plot");

    const yAxis = el("div", "chart-y-axis");
    for (let i = TICK_COUNT; i >= 0; i--) {
      yAxis.appendChild(el("span", "chart-y-tick", formatTick((axisMax * i) / TICK_COUNT)));
    }
    plot.appendChild(yAxis);

    const scrollArea = el("div", "chart-scroll");
    const gridWrap = el("div", "chart-grid-wrap");
    const gridlines = el("div", "chart-gridlines");
    for (let i = 0; i <= TICK_COUNT; i++) gridlines.appendChild(el("div", "chart-gridline"));
    gridWrap.appendChild(gridlines);

    const dense = labels.some((l) => l.includes("/"));
    const bars = el("div", dense ? "chart-bars chart-bars-dense" : "chart-bars");
    labels.forEach((label, i) => {
      const group = el("div", "bar-group");
      const row = el("div", "bar-group-bars");

      const barA = el("div", "bar bar-success");
      barA.style.height = `${(a[i] / axisMax) * 100}%`;
      barA.title = `${label} · ${nameA}: ${a[i]}`;
      row.appendChild(barA);

      const barB = el("div", "bar bar-fail");
      barB.style.height = `${(b[i] / axisMax) * 100}%`;
      barB.title = `${label} · ${nameB}: ${b[i]}`;
      row.appendChild(barB);

      group.appendChild(row);
      group.appendChild(el("span", "bar-cat-label", label));
      bars.appendChild(group);
    });
    gridWrap.appendChild(bars);
    scrollArea.appendChild(gridWrap);
    plot.appendChild(scrollArea);
    container.appendChild(plot);
  }

  window.PadCharts = { renderBarChart };
})();
