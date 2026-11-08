/* watchcard.js -- the watchers card on the run page.
 *
 * List + trigger history + badges: polls the run status document
 * (st.watchers -- the per-tick status_brief plus the owner-side full
 * face) and the timeline tail (GET /api/runs/<id>/events, watcher
 * kinds).  Talks ONLY to api.js; follows runpanels.js conventions.
 */

import {api} from "./api.js";
import {el, fmtNum, poll} from "./app.js";

const WATCH_KINDS = "watcher";

function row(w) {
  const state = w.state === "disabled" ? "bad" : "run";
  return el("div", {class: "muted", style: "font-size:12px"},
      el("span", {class: `pill ${state}`}, w.name),
      " ",
      `${w.kind} · tier ${w.tier} · every ${w.every} · ` +
      `fired ${w.fired ?? 0}`,
      w.state === "disabled"
          ? el("div", {class: "muted"}, w.disabled_reason || "disabled")
          : "",
      w.last_message
          ? el("div", {class: "muted"}, `« ${w.last_message} »`)
          : "");
}

export function watchCard(runId) {
  const box = el("div", {id: "watchers"},
      el("span", {class: "muted"}, "loading..."));
  const hist = el("div", {class: "muted", style: "font-size:12px"}, "");
  let hadWatchers = false;

  poll(async () => {
    const res = await api.runStatus(runId);
    const st = res.doc;
    if (!st || res.status !== 200) return;
    const w = st.watchers;
    if (!w) {
      if (!hadWatchers) {
        box.replaceChildren(el("span", {class: "muted"},
            "no watchers on this run -- arm with: unigym watch <run> add "
            + "... (or unigym run CONFIG --watch pack.yaml)"));
      }
      return;
    }
    hadWatchers = true;
    const fired = w.fired_total ?? 0;
    const badge = (w.disabled?.length)
        ? el("span", {class: "pill bad"},
             `watcher disabled ×${w.disabled.length}`)
        : (fired ? el("span", {class: "pill pause"}, `fired ${fired}`)
                 : el("span", {class: "pill run"}, "armed"));
    box.replaceChildren(
        el("div", {class: "stats"},
            el("div", {class: "stat"},
                el("div", {class: "label"}, "watchers"),
                el("div", {class: "value"}, String(w.armed ?? 0))),
            el("div", {class: "stat"},
                el("div", {class: "label"}, "fired"),
                el("div", {class: "value"}, String(fired))),
            el("div", {class: "stat"},
                el("div", {class: "label"}, "captures"),
                el("div", {class: "value"}, String(w.captures ?? 0))),
            el("div", {class: "stat"},
                el("div", {class: "label"}, "overhead µs/tick"),
                el("div", {class: "value"},
                    fmtNum(w.overhead_us ?? null)))),
        el("div", {style: "margin:6px 0"}, badge),
        w.last ? el("div", {class: "muted"},
            `last: ${w.last.name} @t${w.last.tick} — ${w.last.message}`)
               : "");
    const ev = await api.events(runId, 12);
    const rows = (ev.doc && ev.doc.events) || [];
    hist.replaceChildren(...(rows.length
        ? rows.map((e) => el("div", {},
            `t=${e.tick} [${e.kind}${e.level ? "/" + e.level : ""}] ` +
            `${e.name} ${((e.data || {}).message) || ""}`))
        : [el("span", {}, "no trigger events yet")]));
  }, 2000);
  return el("div", {class: "card", style: "margin-top:14px"},
      el("h2", {}, "watchers"), box,
      el("div", {class: "muted", style: "margin:10px 0 2px"},
          "trigger history (events.jsonl)"), hist);
}
