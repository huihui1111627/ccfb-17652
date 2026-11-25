/* Tiny dependency-free SVG chart helpers (no D3, no external requests). */
"use strict";

const SVGNS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}, parent) {
  const node = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function text(svg, x, y, str, cls = "", anchor = "middle") {
  const t = el("text", { x, y, "text-anchor": anchor, class: cls }, svg);
  t.textContent = str;
  return t;
}

function legend(svg, items, x0, y0) {
  let x = x0;
  for (const { color, label } of items) {
    el("rect", { x, y: y0 - 9, width: 11, height: 11, rx: 2, fill: color },
       svg);
    const t = text(svg, x + 16, y0, label, "barlabel", "start");
    t.setAttribute("fill", "#8a98a8");
    x += 16 + label.length * 7.5 + 22;
  }
}

/* Grouped vertical bars.
 * data: {labels:[], series:[{name,color,values:[]}]} */
function groupedBars(container, data, { yLabel = "", fmt = v => v.toFixed(3),
  zeroLine = false } = {}) {
  container.innerHTML = "";
  const n = data.labels.length, ns = data.series.length;
  const W = Math.max(520, n * 78 + 80), H = 320;
  const ml = 56, mr = 20, mt = 34, mb = 64;
  const iw = W - ml - mr, ih = H - mt - mb;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%",
                          style: "max-width:" + W + "px" }, container);
  let vmin = Infinity, vmax = -Infinity;
  data.series.forEach(s => s.values.forEach(v => {
    vmin = Math.min(vmin, v); vmax = Math.max(vmax, v);
  }));
  if (zeroLine) vmin = Math.min(vmin, 0);
  vmax = Math.max(vmax, vmax * 1.15, 0.001);
  vmin = vmin > 0 ? 0 : vmin * 1.15;
  const y = v => mt + ih * (1 - (v - vmin) / (vmax - vmin));
  // gridlines + y ticks
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const val = vmin + (vmax - vmin) * i / ticks;
    const yy = y(val);
    el("line", { x1: ml, x2: W - mr, y1: yy, y2: yy, class: "gridline" }, svg);
    text(svg, ml - 8, yy + 4, fmt(val), "barlabel", "end");
  }
  if (zeroLine && vmin < 0) {
    const zy = y(0);
    el("line", { x1: ml, x2: W - mr, y1: zy, y2: zy,
                 stroke: "#dce3ec", "stroke-width": 1 }, svg);
  }
  const groupW = iw / n, barW = Math.min(20, (groupW - 14) / ns);
  data.labels.forEach((lab, i) => {
    const gx = ml + i * groupW + groupW / 2;
    data.series.forEach((s, j) => {
      const v = s.values[i];
      const bx = gx + (j - (ns - 1) / 2) * (barW + 2) - barW / 2;
      const by = v >= 0 ? y(v) : y(0);
      const bh = Math.abs(y(v) - y(0));
      el("rect", { x: bx, y: by, width: barW, height: Math.max(1, bh),
                   rx: 2, fill: s.color }, svg);
    });
    const t = text(svg, gx, H - mb + 16, lab, "barlabel");
    t.setAttribute("fill", "#8a98a8");
  });
  legend(svg, data.series.map(s => ({ color: s.color, label: s.name })),
         ml, 18);
  if (yLabel) text(svg, 16, mt + ih / 2, yLabel, "barlabel");
}

/* Horizontal bars for ranked layer sensitivity.
 * rows: [{label, value, color}] */
function hBars(container, rows, { fmt = v => v.toFixed(3),
  width = 760 } = {}) {
  container.innerHTML = "";
  const rowH = 34, H = rows.length * rowH + 30, W = width;
  const ml = 150, mr = 70, mt = 12;
  const iw = W - ml - mr;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%",
                          style: `max-width:${W}px` }, container);
  const vmax = Math.max(...rows.map(r => Math.abs(r.value)), 0.001);
  const x0 = ml + iw / 2;
  el("line", { x1: x0, x2: x0, y1: mt, y2: H - 14,
               stroke: "#dce3ec", "stroke-width": 1 }, svg);
  rows.forEach((r, i) => {
    const yy = mt + i * rowH + rowH / 2;
    const bw = Math.abs(r.value) / vmax * (iw / 2 - 10);
    const bx = r.value >= 0 ? x0 : x0 - bw;
    el("rect", { x: bx, y: yy - 11, width: Math.max(1, bw), height: 22,
                 rx: 3, fill: r.color }, svg);
    const t = text(svg, ml - 10, yy + 4, r.label, "barlabel", "end");
    t.setAttribute("fill", "#dce3ec");
    text(svg, (r.value >= 0 ? x0 + bw : x0 - bw) +
         (r.value >= 0 ? 6 : -6), yy + 4, fmt(r.value),
         "barlabel", r.value >= 0 ? "start" : "end");
  });
}

/* Scatter plot for cost trade-off.
 * points: [{x, y, label, color, r}] */
function scatter(container, points, { xLabel, yLabel,
  width = 760, height = 380 } = {}) {
  container.innerHTML = "";
  const ml = 70, mr = 24, mt = 24, mb = 56;
  const iw = width - ml - mr, ih = height - mt - mb;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%",
                          style: `max-width:${width}px` }, container);
  const xmin = Math.min(...points.map(p => p.x)) * 0.96;
  const xmax = Math.max(...points.map(p => p.x)) * 1.03;
  const ymin = Math.min(...points.map(p => p.y)) * 0.96;
  const ymax = Math.max(...points.map(p => p.y)) * 1.03;
  const X = v => ml + (v - xmin) / (xmax - xmin) * iw;
  const Y = v => mt + ih * (1 - (v - ymin) / (ymax - ymin));
  for (let i = 0; i <= 5; i++) {
    const xv = xmin + (xmax - xmin) * i / 5;
    el("line", { x1: X(xv), x2: X(xv), y1: mt, y2: mt + ih,
                 class: "gridline" }, svg);
    text(svg, X(xv), mt + ih + 18, Math.round(xv).toLocaleString(),
         "barlabel");
    const yv = ymin + (ymax - ymin) * i / 5;
    text(svg, ml - 8, Y(yv) + 4, Math.round(yv).toLocaleString(),
         "barlabel", "end");
  }
  el("line", { x1: ml, x2: ml, y1: mt, y2: mt + ih,
               stroke: "#2d3a4a" }, svg);
  el("line", { x1: ml, x2: ml + iw, y1: mt + ih, y2: mt + ih,
               stroke: "#2d3a4a" }, svg);
  points.forEach(p => {
    el("circle", { cx: X(p.x), cy: Y(p.y), r: p.r || 6,
                   fill: p.color, "fill-opacity": 0.85,
                   stroke: "#0f1419", "stroke-width": 1.5 }, svg);
    if (p.label) text(svg, X(p.x), Y(p.y) - 11, p.label, "barlabel");
  });
  text(svg, ml + iw / 2, height - 12, xLabel, "barlabel");
  const yl = text(svg, 18, mt + ih / 2, yLabel, "barlabel");
  yl.setAttribute("transform", `rotate(-90 18 ${mt + ih / 2})`);
}

/* Paired line chart for one layer's float vs quant outputs. */
function pairedLines(container, a, b, { nameA = "FP32",
  nameB = "INT", width = 900, height = 320 } = {}) {
  container.innerHTML = "";
  const ml = 50, mr = 20, mt = 28, mb = 48;
  const iw = width - ml - mr, ih = height - mt - mb;
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%",
                          style: `max-width:${width}px` }, container);
  const all = a.concat(b);
  const vmin = Math.min(...all), vmax = Math.max(...all);
  const pad = (vmax - vmin) * 0.1 || 1;
  const Y = v => mt + ih * (1 - (v - vmin + pad) /
                            (vmax - vmin + 2 * pad));
  const X = i => ml + (a.length === 1 ? iw / 2 :
                       i * iw / (a.length - 1));
  for (let i = 0; i <= 4; i++) {
    const val = vmin - pad + (vmax - vmin + 2 * pad) * i / 4;
    el("line", { x1: ml, x2: width - mr, y1: Y(val), y2: Y(val),
                 class: "gridline" }, svg);
    text(svg, ml - 8, Y(val) + 4, val.toFixed(2), "barlabel", "end");
  }
  const path = vals => vals.map((v, i) =>
    `${i === 0 ? "M" : "L"}${X(i)},${Y(v)}`).join(" ");
  el("path", { d: path(a), fill: "none", stroke: "#9b8cff",
               "stroke-width": 2 }, svg);
  el("path", { d: path(b), fill: "none", stroke: "#4da3ff",
               "stroke-width": 2, "stroke-dasharray": "5 3" }, svg);
  legend(svg, [{ color: "#9b8cff", label: nameA },
               { color: "#4da3ff", label: nameB }], ml, 16);
  text(svg, ml + iw / 2, height - 12, "输出分量索引", "barlabel");
}
