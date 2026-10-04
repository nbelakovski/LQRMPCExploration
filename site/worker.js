// Pyodide lives in here, off the main thread, so that an MPC solve taking several
// seconds never freezes the page or the 3D view.
//
// Packages load in two stages. numpy and scipy are enough for the model, the PID and the
// LQR, which is about 28 MB. python-control is only needed for the MPC chapter and drags
// matplotlib in behind it for another 10 MB, so it waits until someone asks for it.

// The ES module build, not the classic one: importScripts() refuses to load pyodide.js
// from the CDN, while importing pyodide.mjs works. This is a module worker for that
// reason, so app.js has to create it with { type: "module" }.
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

const PYTHON_SOURCES = [
  "constants.py", "dynamics.py", "simpid.py", "lqr.py", "mpc_control.py",
  "main.py", "websim.py",
];

let pyodide = null;
let controlLoaded = false;

const say = (text) => postMessage({ type: "status", text });

async function boot() {
  say("downloading the Python runtime…");
  pyodide = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/",
  });

  say("downloading numpy and scipy…");
  await pyodide.loadPackage(["numpy", "scipy", "micropip"], { messageCallback: () => {} });
  await pyodide.pyimport("micropip").install("pint");

  say("loading the controllers…");
  // Fetched rather than inlined, so that the page can display the very same text.
  //
  // cache: "no-store" matters more than it looks. These files carry no Cache-Control and
  // no ETag, so a browser is free to apply heuristic freshness and serve a stale copy
  // without revalidating. That would quietly break the one promise this page makes --
  // that the code on screen is the code that ran -- and it would show up as edits to a
  // controller simply not appearing after a rebuild. They are a few KB; always refetch.
  const sources = await Promise.all(
    PYTHON_SOURCES.map(async (name) => {
      const res = await fetch(`py/${name}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`py/${name}: HTTP ${res.status}`);
      return [name, await res.text()];
    })
  );
  for (const [name, text] of sources) pyodide.FS.writeFile(`/home/pyodide/${name}`, text);
  pyodide.runPython("import sys; sys.path.insert(0, '/home/pyodide')");

  postMessage({ type: "ready", sources: Object.fromEntries(sources) });
}

async function ensureControl() {
  if (controlLoaded) return;
  say("downloading python-control for the MPC…");
  await pyodide.pyimport("micropip").install("control");
  controlLoaded = true;
}

onmessage = async ({ data }) => {
  const { id, controller, params } = data;
  try {
    if (controller === "mpc") await ensureControl();
    // Round-trip through JSON rather than proxying a dict, so nothing needs destroying
    // on this side and the main thread gets a plain object.
    const run = pyodide.runPython(`
import json, websim
lambda kwargs: json.dumps(websim.run(**json.loads(kwargs)))
`);
    const result = JSON.parse(run(JSON.stringify({ controller, ...params })));
    run.destroy();
    postMessage({ type: "result", id, result });
  } catch (err) {
    postMessage({ type: "error", id, message: String(err) });
  }
};

boot().catch((err) => postMessage({ type: "fatal", message: String(err) }));
