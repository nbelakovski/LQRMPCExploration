// Page wiring: sliders -> Pyodide worker -> uPlot charts -> the 3D arm.
//
// Python does the physics and returns arrays; nothing here knows what an LQR is. The
// charts and the CAD view are both driven off the same returned run, so pointing at a
// moment in either chart moves the other chart's cursor and the arm with it.

import { createArmViewer, ARM } from "./arm3d.js";

/* ---------------------------------------------------------------- worker ---- */

const worker = new Worker("worker.js", { type: "module" });
const pending = new Map();
let nextId = 0;
let sources = {};

const ready = new Promise((resolve, reject) => {
  worker.onmessage = ({ data }) => {
    if (data.type === "status") return setStatus(data.text);
    if (data.type === "ready") { sources = data.sources; showAllCode(); return resolve(); }
    if (data.type === "fatal") { setStatus(`failed: ${data.message}`, true); return reject(); }
    const entry = pending.get(data.id);
    if (!entry) return;
    pending.delete(data.id);
    data.type === "error" ? entry.reject(new Error(data.message)) : entry.resolve(data.result);
  };
});

// A worker that fails to start would otherwise leave the page saying "starting…"
// forever, with the reason only visible in the devtools console.
worker.onerror = (e) => setStatus(
  `the Python runtime could not start: ${e.message ?? "worker failed to load"}`, true);
worker.onmessageerror = () => setStatus("the Python runtime sent something unreadable", true);

function runSimulation(controller, params) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, controller, params });
  });
}

// Set while a lab is mid-run, so that worker progress (notably the one-off
// python-control download) shows up on that lab's overlay. A reader sitting in the MPC
// chapter never sees the status line at the top of the page.
let statusSink = null;

function setStatus(text, isError = false) {
  if (statusSink && text) statusSink(text);
  const el = document.getElementById("status");
  el.textContent = text;
  el.classList.toggle("error", isError);
  el.classList.toggle("done", text === "");
}

/* ------------------------------------------------------------------ code ---- */

// Pull one function out of a source file so a chapter can show just the part it is
// talking about, rather than the whole module.
function extractDef(source, name) {
  const lines = source.split("\n");
  const start = lines.findIndex((l) => l.startsWith(`def ${name}(`));
  if (start < 0) return source;
  let end = start + 1;
  while (end < lines.length && !/^\S/.test(lines[end])) end++;
  return lines.slice(start, end).join("\n").replace(/\s+$/, "");
}

// One pass, one alternation. Highlighting in several passes looks simpler but is wrong:
// a later pass sees the markup the earlier ones inserted, and `class` is itself a Python
// keyword, so the keyword rule cheerfully mangles every <span class="..."> already there.
const PY_TOKENS = new RegExp([
  String.raw`('''[\s\S]*?'''|"""[\s\S]*?"""|'[^'\n]*'|"[^"\n]*")`,
  String.raw`(#[^\n]*)`,
  String.raw`\b(def|return|if|else|elif|for|while|import|from|as|lambda|None|True|False|and|or|not|in|is|assert|nonlocal|global|class|with|raise)\b`,
].join("|"), "g");

function highlight(code) {
  return code
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(PY_TOKENS, (m, str, comment, keyword) =>
      str ? `<span class="s">${str}</span>`
        : comment ? `<span class="c">${comment}</span>`
          : `<span class="k">${keyword}</span>`);
}

function showAllCode() {
  document.querySelectorAll("[data-src]").forEach((el) => {
    const [file, fn] = el.dataset.src.split(":");
    const text = sources[file] ?? `(${file} not loaded)`;
    el.innerHTML = highlight(fn ? extractDef(text, fn) : text);
  });
}

/* ----------------------------------------------------------------- charts ---- */

const AXIS = { stroke: "#9aa4b2", grid: { stroke: "#9aa4b222" }, ticks: { stroke: "#9aa4b244" } };

function makeCharts(host, syncKey, onHover) {
  // The two charts in a lab share a cursor, so pointing at a moment on one of them marks
  // the same moment on the other. With the slider gone this cursor is the only scrub.
  const sync = uPlot.sync(syncKey);
  const common = (title, series, extra = {}) => ({
    title, width: host.clientWidth, height: 190,
    cursor: { y: false, sync: { key: sync.key } },
    legend: { show: false },
    scales: { x: { time: false } },
    axes: [{ ...AXIS, label: "time (s)" }, { ...AXIS, size: 60 }],
    series: [{}, ...series],
    hooks: { setCursor: [(u) => onHover(u.cursor.idx)] },
    ...extra,
  });

  // Every series has to exist before the first setData, and the initial data needs one
  // array per series, or uPlot throws while drawing the still-empty chart. uPlot mutates
  // the series objects it is given, so each dashed line gets its own.
  const dashed = () => ({ stroke: "#9aa4b2", dash: [6, 4], width: 1 });
  const angle = new uPlot(
    common("arm angle (deg)", [{ stroke: "#4c8dff", width: 2 }, dashed()]),
    [[0], [0], [null]], host);
  const torque = new uPlot(
    common("control effort (Nm)", [{ stroke: "#e0569a", width: 2 }, dashed(), dashed()]),
    [[0], [0], [null], [null]], host);

  new ResizeObserver(() => {
    for (const u of [angle, torque]) u.setSize({ width: host.clientWidth, height: 190 });
  }).observe(host);

  return { angle, torque };
}

// Dashed reference lines (the setpoint, the motor's torque limit) drawn as flat series,
// which is the cheapest way to get them out of uPlot without a plugin.
function flat(t, value) { return t.map(() => value); }

/* -------------------------------------------------------------- 3D viewer ---- */

/* ------------------------------------------------------------------- labs ---- */

const CONTROLS = {
  // The range runs past vertical because the target does, and the step is 1 so that
  // every angle is reachable. Keep `value` in step with TARGET_ANGLE_DEG in main.py.
  x_ref_deg: { label: "target angle", min: -90, max: 180, step: 1, value: 102, unit: "°" },
  tf_s: { label: "duration", min: 0.5, max: 20, step: 0.5, value: 1, unit: " s" },
  Kp: { label: "Kp", min: 0, max: 600, step: 10, value: 300 },
  Kd: { label: "Kd", min: 0, max: 200, step: 5, value: 75 },
  Kg: { label: "Kg", min: 0, max: 30, step: 1, value: 12 },
  Q11: { label: "Q11 (angle error)", min: 1, max: 250, step: 1, value: 100 },
  Q22: { label: "Q22 (rate error)", min: 0, max: 50, step: 1, value: 1 },
  R_lqr: { key: "R", label: "R (effort)", min: -3, max: 1, step: 0.1, value: -0.3, log: true },
  R_mpc: { key: "R", label: "R (effort)", min: -4, max: 0, step: 0.1, value: -3, log: true },
  // Solve time is driven by the number of points far more than by how far ahead they
  // reach, so the point count is the one to treat carefully. Keep these in step with the
  // defaults in mpc_control.make_controller.
  horizon_s: { label: "horizon", min: 0.05, max: 1, step: 0.05, value: 0.2, unit: " s" },
  horizon_points: { label: "horizon points", min: 3, max: 8, step: 1, value: 3 },
};

const LABS = [
  { id: "lab-none", controller: "none", controls: ["tf_s"], friction: true, auto: true,
    defaults: { tf_s: { value: 8 } } },
  { id: "lab-pid", controller: "pid", controls: ["x_ref_deg", "Kp", "Kd", "Kg"], auto: true },
  { id: "lab-lqr", controller: "lqr", controls: ["x_ref_deg", "Q11", "Q22", "R_lqr"], auto: true },
  { id: "lab-mpc", controller: "mpc", auto: false,
    controls: ["x_ref_deg", "Q11", "Q22", "R_mpc", "horizon_s", "horizon_points"] },
];

function buildLab(spec) {
  const host = document.getElementById(spec.id);
  const state = {};
  const inputs = [];

  // --- controls ---
  const panel = host.querySelector(".controls");
  for (const name of spec.controls) {
    const cfg = { ...CONTROLS[name], ...(spec.defaults?.[name] ?? {}) };
    const key = cfg.key ?? name;
    const row = document.createElement("label");
    row.className = "control";
    const out = document.createElement("output");
    const input = Object.assign(document.createElement("input"), {
      type: "range", min: cfg.min, max: cfg.max, step: cfg.step, value: cfg.value,
    });
    const read = () => (cfg.log ? Number(10 ** input.value).toPrecision(2) : Number(input.value));
    const sync = () => { state[key] = Number(read()); out.textContent = `${read()}${cfg.unit ?? ""}`; };
    // The readout follows the drag, but the simulation only runs on the value actually
    // settled on: "input" fires continuously while dragging, "change" once on release
    // (and once per step when the slider is driven from the keyboard).
    input.addEventListener("input", sync);
    if (spec.auto) input.addEventListener("change", () => schedule(true));
    inputs.push({ input, cfg, sync });
    sync();
    row.append(Object.assign(document.createElement("span"), { textContent: cfg.label }), input, out);
    panel.appendChild(row);
  }

  if (spec.friction) {
    const row = document.createElement("label");
    row.className = "control toggle";
    const box = Object.assign(document.createElement("input"), { type: "checkbox", checked: true });
    box.addEventListener("change", () => { state.friction = box.checked; schedule(); });
    state.friction = true;
    row.append(box, Object.assign(document.createElement("span"), { textContent: "friction" }));
    panel.appendChild(row);
  }

  const button = document.createElement("button");
  button.textContent = spec.auto ? "reset" : "run";
  button.className = spec.auto ? "ghost" : "primary";
  button.addEventListener("click", () => {
    if (spec.auto) {
      for (const i of inputs) { i.input.value = i.cfg.value; i.sync(); }
      // "reset" means the whole lab, so put the camera back where it started too.
      viewer?.resetView();
    }
    schedule(true);
  });
  panel.appendChild(button);

  // --- output ---
  // Built here rather than in the markup so there is one copy to keep right.
  const overlay = document.createElement("div");
  overlay.className = "overlay";
  const overlayText = document.createElement("span");
  overlay.append(document.createElement("i"), overlayText);
  host.appendChild(overlay);

  const charts = makeCharts(host.querySelector(".charts"), spec.id, (idx) => showAt(idx));
  const readout = host.querySelector(".readout");
  const info = host.querySelector(".info");
  let run = null;
  let viewer = null;

  // Every chapter keeps its own view of the arm, so it is on screen wherever you are on
  // the page. The model behind them is fetched and parsed once; see arm3d.js.
  createArmViewer(host.querySelector(".stage"))
    .then((v) => { viewer = v; showAt(0); })
    .catch((err) => {
      host.querySelector(".stage").innerHTML =
        `<p>could not load the CAD: ${err.message}</p>`;
    });

  // Pointing at either chart is what moves the arm now. uPlot reports a null index when
  // the pointer leaves, and we simply hold the last pose rather than snapping back.
  function showAt(idx) {
    if (!run || idx == null) return;
    readout.textContent =
      `t ${run.t[idx].toFixed(3)} s   θ ${run.angle_deg[idx].toFixed(1)}°   ` +
      `θ̇ ${run.rate_degps[idx].toFixed(0)}°/s   u ${run.torque_Nm[idx].toFixed(2)} Nm`;
    viewer?.setAngle(run.angle_deg[idx]);
  }

  let timer = null;
  function schedule(immediate = false) {
    clearTimeout(timer);
    timer = setTimeout(execute, immediate ? 0 : 120);
  }

  async function execute() {
    await ready;
    // Only announce a run slow enough to be worth announcing. PID and LQR finish in well
    // under a tenth of a second, and an overlay on every slider tick would just strobe.
    overlayText.textContent = "running\u2026";
    const announce = setTimeout(() => host.classList.add("running"), 250);
    statusSink = (text) => { overlayText.textContent = text; };
    try {
      run = await runSimulation(spec.controller, state);
      const { t, angle_deg, torque_Nm, x_ref_deg, torque_limit_Nm } = run;
      charts.angle.setData([t, angle_deg,
        x_ref_deg == null ? t.map(() => null) : flat(t, x_ref_deg)]);
      charts.torque.setData([t, torque_Nm, flat(t, torque_limit_Nm), flat(t, -torque_limit_Nm)]);
      showAt(0);
      info.textContent = run.K
        ? `K = [${run.K.map((v) => v.toFixed(2)).join(", ")}]   ` +
          `closed-loop poles ${run.eigenvalues.map((v) => v.toFixed(1)).join(", ")}`
        : "";
    } catch (err) {
      info.textContent = `failed: ${err.message}`;
    } finally {
      statusSink = null;
      clearTimeout(announce);
      host.classList.remove("running");
    }
  }

  // With no slider under the charts, the way to scrub is not self-evident, so say it.
  host.querySelector(".run-hint").textContent =
    (spec.auto ? "re-runs when you let go of a slider" : "a few seconds per run")
    + " \u00b7 point at either chart to move the arm to that moment";

  if (spec.auto) ready.then(() => schedule(true));
  return { execute };
}

ready.then(() => setStatus(""));
LABS.forEach(buildLab);
