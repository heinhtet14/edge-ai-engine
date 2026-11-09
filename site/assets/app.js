// edge-ai-engine site — plain JS, no build step. Data lives in ../data/*.json.

const STAGES = [
  { id: "cam", name: "Wrist camera", tag: "input",
    what: "A single USB camera on the follower arm's wrist captures RGB frames.",
    cost: "Capture + resize + normalize on CPU.",
    plan: "Measure capture latency separately so it isn't blamed on the model." },
  { id: "ve", name: "Vision encoder", tag: "once / chunk",
    what: "Turns each image into visual tokens for the language model.",
    cost: "Dense convolution/attention — a strong INT8 candidate.",
    plan: "Separate TensorRT engine; INT8 with calibration from the held-out split." },
  { id: "lm", name: "LM prefix", tag: "once / chunk",
    what: "Reads visual tokens + the task instruction and builds a KV cache.",
    cost: "Most of SmolVLA's parameters live here; memory-bandwidth bound.",
    plan: "TensorRT FP16 → INT8; INT4 weight-only explored for pi0." },
  { id: "ae", name: "Action expert", tag: "× N steps",
    what: "Flow-matching head that denoises a whole chunk of future actions over ~10 steps.",
    cost: "Small, but runs N times per chunk — so per-step latency multiplies.",
    plan: "Kept in FP16 (precision sensitive); sweep N steps and chunk size." },
  { id: "safe", name: "Safety layer", tag: "every tick",
    what: "Clamps joint targets, enforces velocity/torque limits, watchdog, e-stop, dry-run.",
    cost: "Negligible compute; non-negotiable.",
    plan: "Never optimized away. Its own latency is logged." },
  { id: "arm", name: "SO-ARM101", tag: "output",
    what: "Six Feetech STS3215 servos execute the queued actions at the control rate.",
    cost: "Bus write latency bounds the max control frequency.",
    plan: "Async inference keeps the action queue full while the next chunk computes." },
];

const LADDER = [
  { what: "PyTorch FP32", why: "Reference for accuracy and the speed baseline.", tag: "baseline" },
  { what: "PyTorch FP16 / BF16", why: "Halves memory traffic; check numerical range issues.", tag: "precision" },
  { what: "TensorRT FP16", why: "Kernel fusion + tuned kernels; three per-stage engines.", tag: "TensorRT" },
  { what: "TensorRT INT8 (mixed)", why: "INT8 vision/LM, FP16 action expert; calibration required.", tag: "TensorRT" },
  { what: "INT4 weight-only", why: "If practical on Orin — key for fitting pi0 into 8 GB.", tag: "stretch" },
  { what: "ONNX Runtime + TensorRT EP", why: "Comparison point and fallback for unsupported ops.", tag: "comparison" },
  { what: "Chunk size × denoising steps", why: "Trade reaction time against control frequency.", tag: "sweep" },
];

const RESULT_COLUMNS = [
  { key: "model", label: "Model" },
  { key: "config", label: "Config" },
  { key: "latency_p50_ms", label: "p50 (ms)", num: true },
  { key: "latency_p95_ms", label: "p95 (ms)", num: true },
  { key: "control_hz", label: "Control Hz", num: true },
  { key: "peak_mem_gb", label: "Peak mem (GB)", num: true },
  { key: "power_w", label: "Power (W)", num: true },
  { key: "action_err_vs_fp32", label: "Err vs FP32", num: true },
  { key: "real_success", label: "Real success", num: true },
];

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v);
  }
  for (const c of children) n.append(c);
  return n;
};

function safeStorage(op, key, value) {
  try { return op === "get" ? localStorage.getItem(key) : localStorage.setItem(key, value); }
  catch { return null; }
}

// Theme
(function initTheme() {
  const saved = safeStorage("get", "theme");
  if (saved) document.documentElement.dataset.theme = saved;
  $("#themeToggle").addEventListener("click", () => {
    const current = document.documentElement.dataset.theme ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    safeStorage("set", "theme", next);
  });
})();

// Pipeline explorer
function renderPipeline() {
  const row = $("#pipelineRow");
  STAGES.forEach((s, i) => {
    if (i > 0) row.append(el("span", { class: "arrow", "aria-hidden": "true" }, "→"));
    row.append(el("button", { class: "stage", role: "tab", "data-id": s.id, "aria-selected": "false",
      onclick: () => selectStage(s.id) }, s.name, el("small", {}, s.tag)));
  });
  selectStage("ve");
}
function selectStage(id) {
  const s = STAGES.find((x) => x.id === id);
  document.querySelectorAll(".stage").forEach((b) => b.setAttribute("aria-selected", b.dataset.id === id));
  const d = $("#stageDetail");
  d.replaceChildren(
    el("h3", {}, s.name),
    el("p", {}, s.what),
    el("dl", {}, el("dt", {}, "Cost"), el("dd", {}, s.cost), el("dt", {}, "Plan"), el("dd", {}, s.plan)),
  );
}

// Ladder
function renderLadder() {
  const list = $("#ladderList");
  LADDER.forEach((r) => list.append(el("li", {},
    el("div", {}, el("div", { class: "what" }, r.what), el("div", { class: "why" }, r.why)),
    el("span", { class: "tag" }, r.tag))));
}

// Roadmap
const STATUS_LABEL = { planned: "planned", in_progress: "in progress", done: "done" };
function renderRoadmap(progress) {
  $("#roadmapUpdated").textContent = `Last updated ${progress.updated}.`;
  const filters = $("#phaseFilters");
  const timeline = $("#timeline");
  let active = "all";
  const draw = () => {
    timeline.replaceChildren();
    for (const phase of progress.phases) {
      if (active !== "all" && active !== phase.id) continue;
      for (const m of phase.milestones) {
        timeline.append(el("div", { class: "ms", "data-status": m.status },
          el("div", { class: "id" }, m.id),
          el("div", {},
            el("div", { class: "title" }, m.title),
            el("p", {}, m.summary),
            el("div", { class: "meta" },
              el("span", { class: `tag status-${m.status}` }, STATUS_LABEL[m.status] || m.status),
              el("span", { class: "tag" }, `Phase ${phase.id} · ${phase.name}`),
              ...m.machine.map((x) => el("span", { class: "tag" }, `[${x}]`))))));
      }
    }
  };
  const options = [{ id: "all", name: "All" }, ...progress.phases.map((p) => ({ id: p.id, name: `${p.id} · ${p.name}` }))];
  options.forEach((o) => filters.append(el("button", { "aria-pressed": String(o.id === active), onclick: (e) => {
    active = o.id;
    filters.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === e.currentTarget)));
    draw();
  } }, o.name)));
  draw();

  const all = progress.phases.flatMap((p) => p.milestones);
  const done = all.filter((m) => m.status === "done").length;
  const current = all.find((m) => m.status === "in_progress");
  $("#heroStats").replaceChildren(
    el("div", { class: "stat" }, el("b", {}, `${done}/${all.length}`), el("span", {}, "milestones done")),
    el("div", { class: "stat" }, el("b", {}, current ? current.id : "—"), el("span", {}, current ? current.title : "nothing in progress")),
    el("div", { class: "stat" }, el("b", {}, "8 GB"), el("span", {}, "shared CPU/GPU memory")),
    el("div", { class: "stat" }, el("b", {}, "25 W"), el("span", {}, "power budget")),
  );
}

// Results
function renderResults(results) {
  const table = $("#resultsTable");
  const rows = results.rows || [];
  if (!rows.length) { table.hidden = true; $("#resultsEmpty").hidden = false; return; }
  let sortKey = null, dir = 1;
  const head = table.querySelector("thead tr");
  RESULT_COLUMNS.forEach((c) => head.append(el("th", { scope: "col", onclick: () => {
    dir = sortKey === c.key ? -dir : 1; sortKey = c.key;
    head.querySelectorAll("th").forEach((th) => th.removeAttribute("aria-sort"));
    head.children[RESULT_COLUMNS.indexOf(c)].setAttribute("aria-sort", dir > 0 ? "ascending" : "descending");
    draw();
  } }, c.label)));
  const draw = () => {
    const sorted = sortKey ? [...rows].sort((a, b) => (a[sortKey] > b[sortKey] ? 1 : a[sortKey] < b[sortKey] ? -1 : 0) * dir) : rows;
    table.querySelector("tbody").replaceChildren(...sorted.map((r) => el("tr", {},
      ...RESULT_COLUMNS.map((c) => el("td", c.num ? { class: "num" } : {}, r[c.key] ?? "—")))));
  };
  draw();
}

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

renderPipeline();
renderLadder();
loadJSON("data/progress.json").then(renderRoadmap).catch((e) => {
  $("#timeline").textContent = "Could not load roadmap. Serve the site over HTTP (see README).";
  console.error(e);
});
loadJSON("data/results.json").then(renderResults).catch(() => renderResults({ rows: [] }));
