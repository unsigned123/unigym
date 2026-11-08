/* consolecard.js -- the run page console panel (module split of
 * runpage.js): eval / step through the op face, plus the pause/resume
 * gate buttons.  Echoes every exchange into the output pane; failures
 * land in the unified #err area.  Talks ONLY to api.js.
 */

import {api, errMsg} from "./api.js";
import {el, showError} from "./app.js";

export function consoleCard(runId) {
  const out = el("pre", {class: "pane", id: "console-out"},
      "console output lands here (eval via the op face; step auto-pauses).");
  const expr = el("input", {type: "text", style: "flex:1",
                            placeholder: "python expr, e.g. env.num_envs + 0"});
  const stepN = el("input", {type: "number", value: "1", min: "1",
                             style: "width:90px"});
  const echo = (line) => {
    out.textContent += `\n> ${line}`;
    out.scrollTop = out.scrollHeight;
  };
  const evalBtn = el("button", {onclick: async () => {
    const e = expr.value.trim();
    if (!e) return;
    echo(`eval ${e}`);
    const r = await api.eval(runId, e);
    echo(JSON.stringify(r.doc?.result ?? r.doc, null, 1));
    if (r.doc && r.doc.ok === false) showError(errMsg(r));
  }}, "eval");
  const stepBtn = el("button", {onclick: async () => {
    echo(`step ${stepN.value || 1}`);
    const r = await api.step(runId, Number(stepN.value || 1));
    echo(JSON.stringify(r.doc ?? r, null, 1));
    if (r.doc && r.doc.ok === false) showError(errMsg(r));
  }}, "step");
  const gateBtn = (label, op) => el("button", {onclick: async () => {
    echo(label);
    const r = op === "pause" ? await api.pause(runId)
                             : await api.resume(runId);
    echo(JSON.stringify(r.doc ?? r, null, 1));
    if (r.doc && r.doc.ok === false) showError(errMsg(r));
  }}, label);
  expr.addEventListener("keydown", (e) => {
    if (e.key === "Enter") evalBtn.click();
  });
  return el("div", {class: "card"}, el("h2", {}, "console"),
      el("div", {class: "bar"}, expr, evalBtn,
          el("span", {class: "muted"}, "step"), stepN, stepBtn),
      el("div", {class: "bar"}, el("span", {class: "muted"}, "gate"),
          gateBtn("pause", "pause"), gateBtn("resume", "resume"),
          el("span", {class: "muted"}, "step auto-pauses; resume lets "
              + "the engine continue")),
      out);
}
