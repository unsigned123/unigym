/* app.js -- hash routing + the one shared toolbox.
 *
 * Two static pages, one app module:
 *   /            -> index.html  -> views.portal
 *   /view/<id>   -> run.html    -> views.runPage
 *
 * Views live in their own modules (portal.js / runpage.js + the run-page
 * panels stageview.js / runpanels.js / consolecard.js / freecam.js /
 * telemetry.js -- single responsibility, <=300 lines each) and ONLY talk
 * to api.js.  Shared DOM/format helpers live here -- exactly one
 * formatter per concern (§9: no duplicated implementations anywhere).
 */

import * as portal from "./portal.js";
import * as runPage from "./runpage.js";

/* ---- tiny DOM helper: el("div", {class: "card"}, children) --------------- */
export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (k === "class") node.className = v;
    // lowercase: attrs like onSubmit must map to the DOM event "submit"
    // (addEventListener is case-sensitive; "Submit" never fires and the
    // form falls through to a native GET navigation -- spawn-form bug)
    else if (k.startsWith("on")) node.addEventListener(k.slice(2).toLowerCase(), v);
    else node.setAttribute(k, v);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(child));
  }
  return node;
}

/* ---- the single frontend formatters --------------------------------------- */
export function fmtAge(startTs, now) {
  /* seconds-since-epoch -> "1h02m" / "3m05s" / "-" */
  if (!startTs) return "-";
  const s = Math.max(0, Math.floor((now ?? Date.now() / 1000) - startTs));
  if (s >= 3600) return `${Math.floor(s / 3600)}h${String(Math.floor(s % 3600 / 60)).padStart(2, "0")}m`;
  return `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s`;
}

export function fmtNum(v, digits = 1) {
  /* number -> fixed-decimal string; null/undefined -> "-" */
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  if (typeof v === "boolean") return v ? "yes" : "no";
  return Number(v).toFixed(digits);
}

export const phasePill = (phase, stale) => {
  if (stale) return el("span", {class: "pill bad"}, "STALE");
  // UNKNOWN: gray -- the segment exists but cannot be read.
  // STARTING: gray -- the spawn window before the worker's first
  // heartbeat (ledger exists, segment not yet built).
  const cls = {RUNNING: "run", PAUSED: "pause", DONE: "done",
               STARTING: "unknown", UNKNOWN: "unknown"}[phase] || "bad";
  return el("span", {class: `pill ${cls}`}, phase || "?");
};

/* task-face badges (contract §1.20): benchmark = green, collect =
 * gray; mode "both" shows both.  tf = a /api/runs row's `task_face`
 * ({mode, tasks} | null) -- old runs without a meta.json carry null and
 * render no badge at all. */
export const taskFacePills = (tf) => {
  if (!tf) return [];
  const pills = [];
  if (tf.mode === "benchmark" || tf.mode === "both")
    pills.push(el("span", {class: "pill bench", title: "benchmark run"},
        "bench"));
  if (tf.mode === "collect" || tf.mode === "both")
    pills.push(el("span", {class: "pill collect", title: "collect run"},
        "collect"));
  return pills;
};

/* ---- THE unified error area (§9): stacked, deduped, dismissable -----------
 * identical messages collapse into one row with an xN counter;
 * rows auto-drop after 8 s and carry a manual x.  The poller's good-cycle
 * clear now drops ONLY the poller's own entry (it used to wipe the whole
 * banner -- transient freecam timeouts flashed and vanished).
 */
const _errEntries = new Map();         // key -> {row, badge, n, timer}
let _pollKey = null;                   // the poller's own last entry

const _errBox = () => {
  const box = document.getElementById("err");
  if (!box) return null;
  if (!box.dataset.stack) {
    box.className = "errstack";
    box.dataset.stack = "1";
    box.textContent = "";
  }
  return box;
};

const _dropErr = (key) => {
  const e = _errEntries.get(key);
  if (!e) return;
  clearTimeout(e.timer);
  e.row.remove();
  _errEntries.delete(key);
  if (_pollKey === key) _pollKey = null;
};

const _errKey = (msg) => (msg.length > 90 ? msg.slice(0, 90) : msg);

export function showError(msg) {
  const box = _errBox();
  if (!box) return;
  if (!msg) {                          // bare "clear" -- poller only
    if (_pollKey) _dropErr(_pollKey);
    return;
  }
  const key = _errKey(msg);
  let e = _errEntries.get(key);
  if (!e) {
    const row = el("div", {class: "errrow"});
    row.append(el("span", {class: "errtext"}, `error: ${msg}`));
    const badge = el("span", {class: "errcount"});
    row.append(badge,
        el("button", {class: "errx", title: "dismiss",
                      onclick: () => _dropErr(key)}, "x"));
    box.append(row);
    box.style.display = "block";
    e = {row, badge, n: 0, timer: null};
    _errEntries.set(key, e);
  }
  e.n += 1;
  e.badge.textContent = e.n > 1 ? `x${e.n}` : "";
  clearTimeout(e.timer);
  e.timer = setTimeout(() => _dropErr(key), 8000);   // quiet auto-drop
}

export const fail = (err) => {
  const msg = String(err.message || err);
  showError(msg);
  _pollKey = _errKey(msg);             // poll noise self-clears on a good cycle
};

/* ---- poller: run fn immediately then every period; stops on leave -------- */
export function poll(fn, periodMs) {
  let timer = null;
  const tick = async () => {
    try {
      await fn();
      showError("");                 // a good cycle clears poll noise
    } catch (e) { fail(e); }         // (one transient blip must not
  };                                 //  mar the page forever)
  tick();
  timer = setInterval(tick, periodMs);
  window.addEventListener("pagehide", () => clearInterval(timer));
  return () => clearInterval(timer);
}

/* ---- trailing debounce (event-driven refreshes coalesce) ----------- */
export function debounce(fn, ms) {
  let timer = null;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

/* ---- routing ---------------------------------------------------------------- */
function route() {
  const view = /\/view\/(.+)$/.exec(location.pathname);
  if (view) runPage.mount(decodeURIComponent(view[1]));
  else portal.mount();
}

route();
