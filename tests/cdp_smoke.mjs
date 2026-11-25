/* Minimal CDP smoke test: load the dashboard, switch every tab and model,
 * capture console errors and assert key DOM nodes render. No external deps;
 * speaks the DevTools Protocol over a websocket using the built-in ws from
 * Node 26 (global WebSocket).
 */
const BASE = "http://127.0.0.1:8765/";

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function main() {
  // attach to an existing page target (avoids /json/new verb restrictions)
  const listResp = await fetch("http://127.0.0.1:9222/json");
  const targets = await listResp.json();
  const tab = targets.find(t => t.type === "page");
  const ws = new WebSocket(tab.webSocketDebuggerUrl);
  let id = 0;
  const pending = new Map();
  const errors = [];
  const logs = [];
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      pending.get(msg.id)(msg.result);
      pending.delete(msg.id);
    }
    if (msg.method === "Runtime.consoleAPICalled" &&
        msg.params.type === "error")
      errors.push(msg.params.args.map(a => a.value || a.description).join(" "));
    if (msg.method === "Runtime.exceptionThrown")
      errors.push("EXCEPTION: " +
        (msg.params.exceptionDetails.exception?.description ||
         msg.params.exceptionDetails.text));
    if (msg.method === "Runtime.consoleAPICalled")
      logs.push(msg.params.type + ": " +
        msg.params.args.map(a => a.value ?? a.description ?? "").join(" "));
  };
  await new Promise(r => ws.onopen = r);
  const send = (method, params = {}) => new Promise(res => {
    const mid = ++id; pending.set(mid, res);
    ws.send(JSON.stringify({ id: mid, method, params }));
  });
  await send("Runtime.enable");
  await send("Page.enable");
  await send("Page.navigate", { url: BASE });
  await sleep(2500);

  const evalJs = async (expr) => {
    const r = await send("Runtime.evaluate", {
      expression: expr, returnByValue: true });
    if (r.exceptionDetails)
      errors.push("EVAL EXCEPTION: " +
        (r.exceptionDetails.exception?.description ||
         r.exceptionDetails.text));
    return r.result?.value;
  };

  const checks = [];
  const assert = (name, cond) => checks.push([name, !!cond]);

  // initial render
  assert("overview cards rendered",
    await evalJs("document.querySelectorAll('#schemeCards .card').length") >= 3);
  assert("scheme table rows",
    await evalJs("document.querySelectorAll('#schemeTable tbody tr').length") >= 4);
  assert("per-class svg",
    await evalJs("document.querySelector('#perClassChart svg') !== null"));

  // switch models
  const models = await evalJs(
    "Array.from(document.getElementById('modelSelect').options).map(o=>o.value)");
  for (const m of models) {
    await evalJs(`(()=>{const s=document.getElementById('modelSelect');s.value='${m}';s.dispatchEvent(new Event('change'));})();`);
    await sleep(300);
    const nan = await evalJs(
      "document.body.innerText.includes('NaN') || document.body.innerText.includes('undefined')");
    assert(`model ${m} no NaN/undefined`, !nan);
  }

  // switch every tab on the sensitive model
  await evalJs(`(()=>{const s=document.getElementById('modelSelect');s.value='mlp_sensitive';s.dispatchEvent(new Event('change'));})();`);
  await sleep(300);
  for (const t of ["sensitivity", "mixed", "calib", "trace"]) {
    await evalJs(`document.querySelector('[data-tab="${t}"]').click()`);
    await sleep(200);
    const active = await evalJs(
      `document.getElementById('tab-${t}').classList.contains('active')`);
    assert(`tab ${t} activates`, active);
  }
  // sensitivity content
  assert("SQNR chart",
    await evalJs("document.querySelector('#sqnrChart svg') !== null"));
  assert("ablation table",
    await evalJs("document.querySelectorAll('#ablationChart table tr').length") >= 3);
  // mixed content
  assert("mixed layer table",
    await evalJs("document.querySelectorAll('#mixedLayerTable tbody tr').length") >= 3);
  assert("pareto scatter",
    await evalJs("document.querySelector('#paretoChart svg') !== null"));
  assert("feasible comparison rows",
    await evalJs("document.querySelectorAll('#feasibleList table tbody tr').length") >= 1);
  // calib content
  assert("observer table 4 methods",
    await evalJs("document.querySelectorAll('#observerTable tbody tr').length") === 4);
  assert("calib table 5 variants",
    await evalJs("document.querySelectorAll('#calibTable tbody tr').length") === 5);
  assert("outlier table",
    await evalJs("document.querySelectorAll('#outlierTable tbody tr').length") > 0);
  // trace: switch sample and layer, expect paired-line svg with 2 paths
  assert("trace sample options",
    await evalJs("document.getElementById('sampleSelect').options.length") >= 2);
  await evalJs(`(()=>{const ss=document.getElementById('sampleSelect');ss.value=String(ss.options.length-1);ss.dispatchEvent(new Event('change'));})();`);
  await sleep(200);
  assert("trace table rows",
    await evalJs("document.querySelectorAll('#traceTable tbody tr').length") > 0);
  assert("trace paired lines (2 paths)",
    await evalJs("document.querySelectorAll('#traceChart svg path').length") === 2);
  await evalJs(`(()=>{const ls=document.getElementById('traceLayerSelect');ls.value=String(Math.min(2,ls.options.length-1));ls.dispatchEvent(new Event('change'));})();`);
  await sleep(200);
  assert("trace still 2 paths after layer switch",
    await evalJs("document.querySelectorAll('#traceChart svg path').length") === 2);
  const traceNans = await evalJs(
    "document.getElementById('traceTable').innerText.includes('NaN')");
  assert("trace table no NaN", !traceNans);

  // total svg count sanity across tabs
  const totalSvg = await evalJs("document.querySelectorAll('svg').length");
  assert("multiple svg charts rendered", totalSvg >= 4);

  console.log("\n=== CDP interaction checks ===");
  let pass = 0;
  for (const [name, ok] of checks) {
    console.log((ok ? "[OK] " : "[FAIL] ") + name);
    if (ok) pass++;
  }
  console.log(`\n${pass}/${checks.length} checks passed`);
  console.log("\n=== console errors (" + errors.length + ") ===");
  errors.forEach(e => console.log("  " + e));
  console.log("\n=== console logs (sample) ===");
  logs.slice(0, 5).forEach(l => console.log("  " + l));

  await ws.close();
  if (pass !== checks.length || errors.length) process.exit(1);
}

main().catch(e => { console.error(e); process.exit(1); });
