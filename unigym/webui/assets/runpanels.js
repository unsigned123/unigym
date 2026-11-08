/* runpanels.js -- the run page's read-only side cards (module split of
 * runpage.js): metric tiles + rewards sparkline (typed metrics query,
 * 2s poll), log tail (3s poll, bottom-stick) and the behavior report
 * card (toolkit C batch artifact -- one-shot fetch, no poll).  Each
 * factory owns its card DOM and starts its own poller (telemetry.js
 * precedent); the module talks ONLY to api.js.
 */

import {api, artifactUrl} from "./api.js";
import {el, fmtNum, poll} from "./app.js";
import {sparkline} from "./charts.js";

function tile(label, value) {
  return el("div", {class: "stat"},
      el("div", {class: "label"}, label),
      el("div", {class: "value"}, value));
}

export function metricsCard(runId) {
  const box = el("div", {id: "stats"});
  poll(async () => {
    const res = await api.metrics(runId);
    const m = res.doc?.metrics;
    if (!m || res.status === 202) return;
    const ep = m.episode || {};
    const timing = m.timing || {};
    box.replaceChildren(
        el("div", {class: "stats"},
            tile("step", String(timing.step ?? "-")),
            tile("episodes", String(timing.episodes ?? "-")),
            tile("fps", fmtNum(timing.fps_policy)),
            tile("fps sim", fmtNum(timing.fps_sim)),
            tile("sim time", `${fmtNum(timing.sim_time, 1)}s`),
            tile("last success", fmtNum(ep.last?.success)),
            tile("last reward", fmtNum(ep.last?.reward)),
            tile("last steps", String(ep.last?.steps ?? "-"))),
        el("div", {class: "muted", style: "margin:10px 0 2px"},
            `rewards (last ${ep.rewards?.length ?? 0})` +
            (m.task ? ` · task ${m.task.pack ? m.task.pack + "/" : ""}` +
                m.task.name : "") +
            (m.progress ? ` · progress ${m.progress.episodes}/` +
                `${m.progress.min_episodes}` : "")),
        sparkline(ep.rewards || []));
  }, 2000);
  return el("div", {class: "card"}, el("h2", {}, "metrics"), box);
}

export function logCard(runId) {
  const pre = el("pre", {class: "pane", id: "log"});
  poll(async () => {
    const res = await api.logTail(runId, 4000);
    if (res.status !== 200) return;
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 4;
    pre.textContent = res.doc.tail;
    if (atBottom) pre.scrollTop = pre.scrollHeight;   // stick to the tail
  }, 3000);
  return el("div", {class: "card", style: "margin-top:14px"},
      el("h2", {}, "log tail"), pre);
}

export function analysisCard(runId) {
  const box = el("div", {id: "analysis"},
      el("span", {class: "muted"}, "loading..."));
  (async () => {
    const res = await fetch(artifactUrl(
        {run: runId, name: "behavior_report.json"}));
    if (res.status !== 200) {
      box.replaceChildren(el("span", {class: "muted"},
          "no behavior report yet -- run: unigym analyze <run_dir>"));
      return;
    }
    let doc;
    try { doc = await res.json(); } catch { return; }
    const rep = doc.report || {};
    const eps = rep.episodes || [];
    const fell = eps.filter((e) => e.fall_tick !== null).length;
    const touched = eps.filter((e) =>
        (e.verdict || "").startsWith("touched")).length;
    box.replaceChildren(
        el("div", {class: "muted"},
            `${rep.run?.ticks ?? "-"} ticks · ${eps.length} episodes · `
            + `fell ${fell} · touched ${touched}`),
        el("a", {href: artifactUrl(
            {run: runId, name: "behavior_report.md"})},
            "open behavior_report.md"),
        " ",
        el("a", {href: artifactUrl(
            {run: runId, name: "behavior_report.json"})}, "json"));
  })();
  return el("div", {class: "card", style: "margin-top:14px"},
      el("h2", {}, "behavior report"), box);
}

/* benchmark report card (contract §1.20): per-task verdicts from
 * TWO faces of the same document.  While the run is alive the typed
 * report query (§1.6) returns BenchmarkSink's LIVE DRAFT (updated per
 * episode); the artifact face serves the FINAL benchmark_report.json
 * (BenchmarkSink lands it at run END -- a millisecond window before the
 * run leaves the live view, so it wins when caught).  A collect-only
 * run (draft null + no benchmark tasks) says so explicitly -- that
 * fallback is information, NOT an error, so it never touches the shared
 * error banner. */
export function benchmarkCard(runId) {
  const box = el("div", {id: "benchmark"},
      el("span", {class: "muted"}, "loading..."));
  let face = null;                     // one-shot task_face (start-time)
  let settled = false;                 // final artifact rendered
  api.run(runId).then((res) => {
    face = res.doc?.run?.task_face || null;
  }).catch(() => {});
  const rateClass = (rate) =>
      (rate === null || rate === undefined) ? "na"
          : rate >= 0.999 ? "good" : rate <= 0 ? "bad" : "mid";
  const rateText = (rate) =>
      (rate === null || rate === undefined) ? "n/a"
          : `${Math.round(rate * 100)}%`;
  const taskRow = ([name, t]) => el("div", {class: "bench-row"},
      el("span", {class: "bench-name"}, name),
      el("span", {class: "muted num"}, `n=${t.n_episodes ?? "-"}`),
      el("span", {class: `bench-rate ${rateClass(t.success_rate)}`},
          rateText(t.success_rate)));
  const render = (doc, live) => {
    const bits = [`episodes ${doc.episodes_total ?? "-"}`,
                  `steps ${doc.steps_total ?? "-"}`];
    if (typeof doc.wall_time_s === "number")
      bits.push(`${doc.wall_time_s.toFixed(1)}s`);
    if (doc.directed) bits.push("directed");
    const rows = Object.entries(doc.tasks || {}).map(taskRow);
    // replaceChildren takes nodes, not arrays/nullables -- build the kid
    // list explicitly (an array argument stringifies to [object ...])
    const kids = [el("div", {class: "muted", style: "margin-bottom:6px"},
        bits.join(" · ") + (live ? " · live (judged per episode)" : ""))];
    if (rows.length) kids.push(...rows);
    else kids.push(el("span", {class: "muted"}, "no episodes scored yet"));
    if (!live) {
      kids.push(el("div", {style: "margin-top:8px"},
          el("a", {href: artifactUrl({run: runId,
              name: "benchmark_report.json"})}, "benchmark_report.json")));
    }
    box.replaceChildren(...kids);
  };
  const pending = () => {
    if (face && face.mode === "collect")     // no benchmark tasks at all
      return el("span", {class: "muted"},
          "collect-only run -- no benchmark tasks, nothing to judge");
    const tasks = face?.tasks?.length ? ` (tasks: ${face.tasks.join(", ")})`
        : "";
    return el("span", {class: "muted"},
        `collect-only / judged at end${tasks} -- benchmark_report.json `
        + "lands when the run finishes");
  };
  const tick = async () => {
    if (!settled) {                          // final artifact face first
      try {
        const res = await fetch(artifactUrl(
            {run: runId, name: "benchmark_report.json"}));
        if (res.status === 200) {
          const doc = await res.json().catch(() => null);
          if (doc) {
            settled = true;
            return render(doc, false);
          }                                  // torn read: retry next cycle
        }
      } catch { /* network blip: the poller owns the surface */ }
    }
    // live draft face (contract §1.6): same document, updated per episode
    try {
      const res = await api.report(runId);
      const rep = res.doc?.report;
      if (res.status === 200 && rep && !settled) return render(rep, true);
      if (res.status === 200 && !settled && !box.dataset.pending) {
        box.dataset.pending = "1";           // collect-only / no episodes yet
        box.replaceChildren(pending());
      }
    } catch {                                // gone/wedged: keep last content
    }
  };
  poll(tick, 2000);
  const card = el("div", {class: "card", style: "margin-top:14px"},
      el("h2", {}, "benchmark report"), box);
  card.refresh = tick;         // SSE nudge entry (bench.draft/reset)
  return card;
}
