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
            `rewards (last ${ep.rewards?.length ?? 0})`),
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
