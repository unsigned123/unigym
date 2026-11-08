/* runpage.js -- the /view/<run_id> detail page view (cockpit;
 * module split: the live-view stage moved to stageview.js, the read-only
 * side cards to runpanels.js, the console panel to consolecard.js --
 * this file is the status line + assembly shell).
 *
 * Live status line (phase pill / step / frame / stale badge), the top
 * bar (camera switcher / mode / terminate / back) and the two-column
 * grid wiring.  Talks ONLY to api.js and the panel modules.
 */

import {api, events, media, errMsg} from "./api.js";
import {debounce, el, fail, phasePill, poll, showError, taskFacePills}
    from "./app.js";
import {startTelemetry} from "./telemetry.js";
import {mountFreeCam} from "./freecam.js";
import {mountStageView} from "./stageview.js";
import {consoleCard} from "./consolecard.js";
import {analysisCard, benchmarkCard, logCard, metricsCard} from "./runpanels.js";
import {sceneCard} from "./scenecard.js";
import {watchCard} from "./watchcard.js";
import {inspectCard} from "./inspectcard.js";

let runId = "";
let stage = null;                 // stageview handle (mount-time)

/* ---- status line ----------------------------------------------------------------- */
async function refreshStatus() {
  const res = await media.status(runId);
  const st = res.doc;
  if (!st || res.status !== 200) {     // run gone: mark it once (the old
    const phase = document.getElementById("phase");    // code kept the
    if (res.status === 404 && phase && !phase.dataset.gone) {  // stale pill)
      phase.dataset.gone = "1";
      phase.replaceChildren(phasePill("GONE"),
          el("span", {class: "muted"},
              " run is gone (finished or terminated) -- result cards "
              + "keep rendering from its output_dir artifacts"));
    }
    return;
  }
  const phase = document.getElementById("phase");
  phase.replaceChildren(
      phasePill(st.phase, st.stale), " ",
      el("span", {class: "muted"},
          "step ", el("span", {class: "num"}, String(st.step ?? "-")),
          " · frame ", el("span", {class: "num"}, String(st.frame_id ?? "-"))),
      st.stale ? el("span", {class: "pill bad"},
          "engine not advancing (SIGSTOP/crash?)") : "");
  await stage.refresh(st);             // camera sync / first-bind self-heal
}

/* ---- mount ---------------------------------------------------------------------------------- */
export function mount(id) {
  runId = id;
  stage = mountStageView(runId);
  const tel = startTelemetry(runId);   // live-view telemetry line (1s)
  const term = el("button", {class: "danger", onclick: async () => {
    if (!confirm(`shutdown ${runId}?`)) return;
    const r = await api.terminate(runId);
    if (r.doc && r.doc.ok === false) showError(errMsg(r));
  }}, "terminate");
  const back = el("a", {href: "/" + location.search}, "← portal");

  const face = el("span", {id: "taskface"});
  document.querySelector("header h1").after(face,
      el("span", {class: "muted"}, runId));
  // task-face badge: ONE GET /api/runs/<id> -- the task face is a
  // start-time constant, so this is a one-shot, not a second status poll.
  // a finished run misses the live face -> the ledger row
  // (/api/runs?history=1) still carries its task_face (post-mortem).
  api.run(runId).then((res) => {
    const tf = res.doc?.run?.task_face;
    if (tf) face.replaceChildren(...taskFacePills(tf), " ");
    else return api.runsHistory(200);
  }).then((res) => {                   // ledger fallback (gone runs)
    if (!res) return;
    const row = (res.doc?.runs || [])
        .find((r) => r.run_id === runId);
    if (row && row.task_face)
      face.replaceChildren(...taskFacePills(row.task_face), " ");
  }).catch(() => {});                  // gone/dead run: no badge, not an error
  const benchCard = benchmarkCard(runId);
  const inspect = inspectCard(runId);
  document.getElementById("app").replaceChildren(
      el("div", {class: "bar"},
          el("span", {id: "phase"}, "-"),
          el("span", {class: "muted"}, "camera"), stage.camSel,
          el("span", {class: "muted"}, "mode"), stage.modeBox,
          el("span", {class: "spacer"}), term, back),
      el("div", {class: "grid two"},
          el("div", {}, stage.card, tel, consoleCard(runId)),
      el("div", {},
          metricsCard(runId),
          benchCard,
          inspect.card,
          sceneCard(runId),
          watchCard(runId),
          analysisCard(runId),
          logCard(runId))));

  poll(refreshStatus, 2000);
  stage.firstBind().catch(fail);       // dead-run notice / first MJPEG bind
  mountFreeCam(id);
  // SSE nudge for THIS run (server-side run_id filter; contract
  // §1.22): run.* bumps the status line, bench.draft bumps the benchmark
  // card, reset re-pulls both.  Same discipline as the portal: the
  // pollers above are the correctness floor and never go away.
  const bumpStatus = debounce(
      () => { refreshStatus().catch(() => {}); }, 500);
  const bumpBench = debounce(() => { benchCard.refresh?.(); }, 500);
  events((ev) => {
    if (ev.type === "reset") { bumpStatus(); bumpBench(); return; }
    if (ev.type === "bench.draft") bumpBench();
    else if (ev.type.startsWith("run.")) bumpStatus();
  }, {types: ["run", "bench"], runId});
}
