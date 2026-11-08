/* telemetry.js -- the live-view telemetry line (design §4).
 *
 * Polls GET /api/runs/<id>/status at 1s (contract §1.15: each status
 * poll samples the frame path once) and renders
 *   "{publish_rate} fps · dropped {gap_total} · stale yes|no"
 * with the design-system pill / monospace-number styling.  Talks ONLY
 * to api.js; DOM/format helpers come from app.js (no duplicates).
 *
 * The rate line is UNCONDITIONAL -- every phase renders the live
 * readout.  During PAUSED the number is the parked loop's own cadence
 * (25Hz design ceiling): the scene is frozen, the feed is not (a free
 * camera can fly during a pause and needs the live number to tell
 * smooth from stalled) -- hiding the readout would mask a real stall.
 * A PAUSED pill prefixes the same line so the phase stays visible.
 */

import {api} from "./api.js";
import {el, fmtNum, poll} from "./app.js";

function render(node, st) {
  if (!st) {                             // run gone / not answering
    node.replaceChildren("-");
    return;
  }
  const media = st.media || {};
  const paused = st.phase === "PAUSED";
  node.replaceChildren(...
      [paused ? el("span", {class: "pill pause"}, "PAUSED") : null,
       paused ? el("span", {class: "muted"}, " · ") : null,
       el("span", {class: "num"}, fmtNum(media.publish_rate)), " fps",
       el("span", {class: "muted"}, " · dropped "),
       el("span", {class: "num"}, String(media.gap_total ?? "-")),
       el("span", {class: "muted"}, " · stale "),
       media.stale ? el("span", {class: "pill bad"}, "yes")
                   : el("span", {class: "pill run"}, "no"),
      ].filter(Boolean));    // replaceChildren coerces null -> "null"
}

/* Mount the telemetry line into the live-view card; returns the node. */
export function startTelemetry(runId) {
  const node = el("div", {id: "media-telemetry", class: "muted",
                          style: "margin-top:8px"}, "-");
  poll(async () => {
    const r = await api.runStatus(runId);
    render(node, r.status === 200 ? r.doc : null);
  }, 1000);
  return node;
}
