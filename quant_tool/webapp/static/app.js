"use strict";

let DATA = null, MODEL = null;
const COLORS = { fp: "#9b8cff", q8: "#4da3ff", q4: "#ff9f43",
                 mix: "#3ecf8e", warn: "#f0b429", bad: "#ff6b6b",
                 good: "#3ecf8e", muted: "#8a98a8" };

const pct = v => (v * 100).toFixed(1) + "%";
const signedPct = v => (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + "pt";
const dropClass = v => v < -0.005 ? "bad" : (v > 0.005 ? "good" : "");
const bitsLabel = b => b === 32 ? "FP32" : "INT" + b;
const bitsClass = b => "bits-" + b;

async function load() {
  const res = await fetch("/api/results");
  if (!res.ok) {
    document.querySelector("main").innerHTML =
      "<p style='color:#ff6b6b'>results.json 缺失，请先运行 " +
      "experiments/02_run_analysis.py</p>";
    return;
  }
  DATA = await res.json();
  const sel = document.getElementById("modelSelect");
  Object.keys(DATA).forEach(m => {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    sel.appendChild(o);
  });
  sel.addEventListener("change", () => render(sel.value));
  document.querySelectorAll("#tabs button").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#tabs button").forEach(b =>
        b.classList.remove("active"));
      document.querySelectorAll(".tab").forEach(t =>
        t.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).classList.add(
        "active");
    });
  });
  render(Object.keys(DATA)[0]);
}

function render(model) {
  MODEL = model;
  const r = DATA[model];
  document.getElementById("datasetTag").textContent =
    "数据集: " + r.meta.dataset;
  const kname = r.meta.class_names[r.meta.key_class];
  document.getElementById("keyTag").textContent =
    "关键类别: " + kname + " (class " + r.meta.key_class + ")";
  renderOverview(r);
  renderSensitivity(r);
  renderMixed(r);
  renderCalib(r);
  renderTrace(r);
}

/* ---------------- Tab 1: overview ---------------- */
function renderOverview(r) {
  const fp = r.fp, s8 = r.schemes.all_int8, s4 = r.schemes.all_int4;
  const mix = r.mixed;
  const cards = [
    ["FP32 准确率", pct(fp.accuracy), "关键召回 " + pct(fp.key_recall),
     COLORS.fp],
    ["全 INT8", signedPct(s8.acc_drop), "关键召回 " + signedPct(
      s8.key_recall_drop), COLORS.q8],
    ["全 INT4", signedPct(s4.acc_drop), "关键召回 " + signedPct(
      s4.key_recall_drop), COLORS.q4],
  ];
  if (mix) cards.push([
    "混合方案", signedPct(mix.acc_drop),
    "关键召回 " + signedPct(mix.key_recall_drop), COLORS.mix]);
  document.getElementById("schemeCards").innerHTML = cards.map(c =>
    `<div class="card"><div class="k">${c[0]}</div>
     <div class="v" style="color:${c[3]}">${c[1]}</div>
     <div class="d flat">${c[2]}</div></div>`).join("");

  const rows = [
    ["FP32（原精度）", "—", pct(fp.accuracy), "—", pct(fp.key_recall),
     "—", "—", "—"],
  ];
  const push = (lab, s) => rows.push([
    lab, Object.values(s.weight_bits).map(bitsLabel).join("/"),
    pct(s.accuracy), signedPct(s.acc_drop),
    pct(s.key_recall), signedPct(s.key_recall_drop),
    Math.round(s.total_bytes).toLocaleString(),
    Math.round(s.estimated_cycles).toLocaleString()]);
  push("全 INT8 (per-tensor)", s8);
  push("全 INT4 (per-tensor)", s4);
  push("INT8 per-channel", r.schemes.per_channel_int8);
  if (mix) push("工具选择混合方案", mix);
  document.getElementById("schemeTable").innerHTML = table(
    ["方案", "层精度", "准确率", "准确率变化", "关键类召回", "召回变化",
     "模型字节", "整数周期(估)"], rows);

  // per-class recall chart
  const classes = r.meta.class_names;
  const series = [{
    name: "FP32", color: COLORS.fp,
    values: fp.per_class.map(c => c.recall)
  }, {
    name: "全INT8", color: COLORS.q8,
    values: s8.per_class.map(c => c.recall)
  }, {
    name: "全INT4", color: COLORS.q4,
    values: s4.per_class.map(c => c.recall)
  }];
  if (mix) series.push({
    name: "混合", color: COLORS.mix,
    values: mix.per_class.map(c => c.recall)
  });
  groupedBars(document.getElementById("perClassChart"),
              { labels: classes.map((c, i) =>
                i === r.meta.key_class ? c + " ★" : c), series },
              { fmt: v => v.toFixed(2), yLabel: "recall" });
}

/* ---------------- Tab 2: sensitivity ---------------- */
function renderSensitivity(r) {
  const sq = r.layer_sensitivity.sqnr;
  hBars(document.getElementById("sqnrChart"),
    sq.map(d => ({
      label: `L${d.layer} ${d.name}`,
      value: d.sqnr_db,
      color: d.sqnr_db < 15 ? COLORS.bad :
             d.sqnr_db < 25 ? COLORS.warn : COLORS.good
    })), { fmt: v => v.toFixed(1) + " dB", width: 720 });

  const abl = r.layer_sensitivity.ablation;
  // two-series horizontal bars via two stacked charts sharing labels
  document.getElementById("ablationChart").innerHTML =
    "<div class='legend'><span><i style='background:" + COLORS.q4 +
    "'></i>准确率掉点</span><span><i style='background:" + COLORS.bad +
    "'></i>关键类召回掉点</span></div>";
  const wrap = document.createElement("div");
  document.getElementById("ablationChart").appendChild(wrap);
  const labels = abl.map(a => `L${a.layer} ${a.name}`);
  const tbl = document.createElement("div");
  tbl.className = "table-wrap";
  tbl.style.border = "none";
  tbl.innerHTML = table(
    ["层", "仅该层 INT4 准确率变化", "仅该层 INT4 关键召回变化",
     "量化后准确率", "量化后关键召回"],
    abl.map(a => [
      labels[abl.indexOf(a)],
      `<span class="${dropClass(a.acc_drop)}">${signedPct(a.acc_drop)}</span>`,
      `<span class="${dropClass(a.key_recall_drop)}">${signedPct(
        a.key_recall_drop)}</span>`,
      pct(a.acc), pct(a.key_recall)]));
  wrap.appendChild(tbl);
}

/* ---------------- Tab 3: mixed precision ---------------- */
function renderMixed(r) {
  const ms = r.mixed_search, mix = r.mixed;
  const sum = document.getElementById("mixedSummary");
  const cards = [
    ["评估方案数", ms.n_evaluated, `可行 ${ms.n_feasible} 个`, COLORS.accent],
    ["准确率上限", "−" + (ms.acc_budget * 100).toFixed(0) + "pt",
     "关键召回上限 −" + (ms.key_budget * 100).toFixed(0) + "pt", COLORS.warn],
  ];
  if (mix) cards.push(
    ["选中方案准确率变化", signedPct(mix.acc_drop),
     "关键召回 " + signedPct(mix.key_recall_drop), COLORS.mix],
    ["选中方案周期", Math.round(mix.estimated_cycles).toLocaleString(),
     Math.round(mix.total_bytes).toLocaleString() + " 字节", COLORS.good]);
  sum.innerHTML = cards.map(c =>
    `<div class="card"><div class="k">${c[0]}</div>
     <div class="v">${c[1]}</div><div class="d flat">${c[2]}</div></div>`
  ).join("");
  document.getElementById("budgetHint").textContent =
    `约束：准确率下降不超过 ${(ms.acc_budget * 100).toFixed(0)}pt，` +
    `关键类召回下降不超过 ${(ms.key_budget * 100).toFixed(0)}pt；` +
    "在全部满足约束的方案中取整数周期最低者。";

  if (mix) {
    document.getElementById("mixedLayerTable").innerHTML = table(
      ["层", "类型", "选中精度", "该层 MAC", "该层字节", "该层周期"],
      mix.per_layer_cost.map(l => [
        "L" + l.layer, l.name,
        `<span class="${bitsClass(l.bits)}">${bitsLabel(l.bits)}</span>`,
        l.macs.toLocaleString(), Math.round(l.bytes).toLocaleString(),
        Math.round(l.cycles).toLocaleString()]));
  }

  // scatter: all feasible alternatives + cheapest highlighted
  const pts = ms.top_feasible.map((p, i) => ({
    x: p.estimated_cycles, y: p.total_bytes,
    label: i === 0 ? "选中(最省)" : "",
    color: i === 0 ? COLORS.mix : COLORS.q8,
    r: i === 0 ? 8 : 5
  }));
  ms.pareto.forEach(p => {
    if (!pts.some(q => q.x === p.estimated_cycles &&
        q.y === p.total_bytes))
      pts.push({ x: p.estimated_cycles, y: p.total_bytes,
                 label: "Pareto", color: COLORS.warn, r: 6 });
  });
  if (pts.length) scatter(
    document.getElementById("paretoChart"), pts,
    { xLabel: "估计整数周期 →", yLabel: "模型字节 →" });
  else document.getElementById("paretoChart").textContent =
      "没有满足双约束的可行方案。";

  // comparison table of different feasible selections
  const listHost = document.getElementById("feasibleList");
  if (ms.top_feasible.length > 1) {
    listHost.innerHTML =
      "<h2>同一上限下的不同可行选法对比</h2><div class='table-wrap'>" +
      table(
      ["可行方案（同上限下的不同选法）", "层精度", "准确率变化",
       "关键召回变化", "字节", "周期"],
      ms.top_feasible.map((p, i) => [
        i === 0 ? "★ 选中（周期最低）" : "备选 " + i,
        Object.entries(p.weight_bits)
          .map(([k, v]) => `L${k}:${bitsLabel(v)}`).join(" "),
        `<span class="${dropClass(p.acc_drop)}">${signedPct(
          p.acc_drop)}</span>`,
        `<span class="${dropClass(p.key_recall_drop)}">${signedPct(
          p.key_recall_drop)}</span>`,
        Math.round(p.total_bytes).toLocaleString(),
        Math.round(p.estimated_cycles).toLocaleString()])) +
      "</div>";
  } else listHost.innerHTML = "";
}

/* ---------------- Tab 4: calibration + outliers ---------------- */
function renderCalib(r) {
  const obsNames = {
    minmax: "MinMax（完整极值，不截断）", "pct99.9": "分位 99.9% 截断",
    "pct99.5": "分位 99.5% 截断", "pct99.0": "分位 99.0% 截断"
  };
  const obsRows = Object.entries(r.observer_compare || {}).map(([k, v]) => [
    obsNames[k] || k, pct(v.accuracy),
    `<span class="${dropClass(v.acc_drop)}">${signedPct(v.acc_drop)}</span>`,
    pct(v.key_recall),
    `<span class="${dropClass(v.key_recall_drop)}">${signedPct(
      v.key_recall_drop)}</span>`
  ]);
  document.getElementById("observerTable").innerHTML = table(
    ["定范围方法", "准确率", "准确率变化", "关键类召回", "召回变化"],
    obsRows);

  const names = {
    stratified: "分层抽样（推荐）", head: "顺序前 128 条",
    key_heavy: "关键类占一半", key_free: "完全不含关键类",
    tiny: "仅 16 条（小样本）"
  };
  const rows = Object.entries(r.calibration_compare).map(([k, v]) => [
    names[k] || k, v.n_calib, pct(v.accuracy),
    `<span class="${dropClass(v.acc_drop)}">${signedPct(v.acc_drop)}</span>`,
    pct(v.key_recall),
    `<span class="${dropClass(v.key_recall_drop)}">${signedPct(
      v.key_recall_drop)}</span>`
  ]);
  document.getElementById("calibTable").innerHTML = table(
    ["校准集构造", "样本数", "准确率", "准确率变化", "关键类召回",
     "召回变化"], rows);

  const outRows = r.outliers.map(o => [
    `L${o.layer} ${o.name}`, "ch" + o.channel,
    o.w_absmax.toFixed(2),
    `<span style="color:${o.w_ratio_vs_median > 3 ? COLORS.bad :
       COLORS.muted}">${o.w_ratio_vs_median.toFixed(1)}×</span>`,
    o.act_absmax.toFixed(2),
    `<span style="color:${o.act_ratio_vs_median > 3 ? COLORS.bad :
       COLORS.muted}">${o.act_ratio_vs_median.toFixed(1)}×</span>`,
    o.outlier ? "<b style='color:#ff6b6b'>异常</b>" : "正常"
  ]);
  document.getElementById("outlierTable").innerHTML = table(
    ["层", "输出通道", "权重绝对值最大", "权重/层中位数",
     "激活绝对值最大", "激活/层中位数", "判定"], outRows);
}

/* ---------------- Tab 5: sample trace ---------------- */
function renderTrace(r) {
  const sel = document.getElementById("sampleSelect");
  sel.innerHTML = "";
  r.sample_traces.forEach((t, i) => {
    const o = document.createElement("option");
    o.value = i;
    o.textContent = `样本 #${t.sample_id} ` +
      `(${t.is_key ? "关键类★" : "普通类"} 真实=${r.meta.class_names[
        t.label]} FP预测=${r.meta.class_names[t.pred_fp]} ` +
      `INT预测=${r.meta.class_names[t.pred_q]})`;
    sel.appendChild(o);
  });
  const draw = () => {
    const t = r.sample_traces[+sel.value];
    document.getElementById("sampleMeta").textContent =
      `FP预测=${r.meta.class_names[t.pred_fp]} INT预测=${
       r.meta.class_names[t.pred_q]} 平均logit偏差=${t.logit_gap.toFixed(3)}`;
    document.getElementById("traceTable").innerHTML = table(
      ["层", "SQNR (dB)", "余弦相似度", "平均绝对偏差", "最大绝对偏差"],
      t.trace.map(d => [
        `L${d.layer} ${d.name}`,
        `<span style="color:${d.sqnr_db < 15 ? COLORS.bad :
          d.sqnr_db < 25 ? COLORS.warn : COLORS.good}">${
          d.sqnr_db.toFixed(1)}</span>`,
        d.cosine.toFixed(4), d.mean_abs_gap.toFixed(4),
        d.max_abs_gap.toFixed(4)]));
    const lsel = document.getElementById("traceLayerSelect");
    const prev = lsel.value;
    lsel.innerHTML = "";
    t.trace.forEach((d, i) => {
      const o = document.createElement("option");
      o.value = i;
      o.textContent = `L${d.layer} ${d.name}`;
      lsel.appendChild(o);
    });
    if (prev) lsel.value = prev;
    const layer = t.trace[+lsel.value];
    pairedLines(document.getElementById("traceChart"),
      layer.fp_flat, layer.q_flat,
      { nameA: "FP32 输出", nameB: "整数量化输出" });
  };
  sel.onchange = draw;
  document.getElementById("traceLayerSelect").onchange = draw;
  draw();
}

/* ---------------- table helper ---------------- */
function table(headers, rows) {
  return "<table><thead><tr>" +
    headers.map(h => `<th>${h}</th>`).join("") +
    "</tr></thead><tbody>" + rows.map(r =>
      "<tr>" + r.map(c => `<td>${c}</td>`).join("") + "</tr>").join("") +
    "</tbody></table>";
}

load();
