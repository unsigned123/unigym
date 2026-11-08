/* portal.js -- the index page view: cockpit cards + spawn wizard.
 *
 * Live run cards (phase pill / GPU / uptime / cameras / entry buttons),
 * the recent-runs ledger card (history + live in one card, with
 * task-face badges / exit verdicts / result links), and the spawn wizard
 * (config dropdown from /api/configs, backend, GPU, steps, instruction,
 * label, preflight result area).
 * Talks ONLY to api.js.
 */

import {api, artifactUrl, errMsg, events} from "./api.js";
import {debounce, el, fmtAge, phasePill, poll, showError, taskFacePills}
    from "./app.js";

/* ---- live run cards ---------------------------------------------------------- */
function errorBlock(err) {
  /* an UNKNOWN row's read error -- inline, selectable, and a
   * click-to-copy attempt on top (clipboard needs a secure context;
   * manual selection always works, which is the honest floor). */
  if (!err) return null;
  const pre = el("pre", {title: "read error (select to copy; click tries "
      + "the clipboard)", style: "cursor:pointer;max-width:420px",
      onclick: () => {
        try { navigator.clipboard.writeText(err); } catch { /* selection */ }
      }}, err);
  return el("details", {class: "logtail", style: "margin-top:4px"},
      el("summary", {}, "read error"), pre);
}

function runCard(r) {
  const cams = r.cameras || [];
  const view = el("a", {class: "primary", href: `/view/${
      encodeURIComponent(r.run_id)}${location.search}`}, "watch");
  const kill = el("button", {class: "danger", onclick: async () => {
    if (!confirm(`shutdown ${r.run_id}?`)) return;
    const res = await api.terminate(r.run_id);
    if (res.doc && res.doc.ok === false) showError(errMsg(res));
  }}, "terminate");
  return el("div", {class: "card"},
      el("h2", {}, phasePill(r.phase, r.stale), " ",
          taskFacePills(r.task_face), " ",
          el("span", {class: "pill gpu"}, `gpu ${r.gpu ?? "-"}`), " ",
          el("span", {class: "muted num"}, fmtAge(r.start_ts))),
      el("div", {class: "run-link"}, r.run_id),
      el("div", {class: "muted", style: "margin:6px 0"},
          `camera ${r.camera || "-"} · ${cams.length} cam`
          + (r.task_face?.tasks?.length
              ? ` · tasks ${r.task_face.tasks.join(", ")}` : "")),
      r.phase === "UNKNOWN" ? errorBlock(r.error) : null,
      el("div", {class: "bar"}, view, kill,
          el("span", {class: "muted"},
              (r.config || "").split("/").pop())));
}

function runsGrid(runs) {
  const alive = runs.filter((r) => r.phase !== "DONE" || r.alive);
  if (!alive.length) {
    return el("div", {class: "card"},
        el("h2", {}, "no active runs"),
        el("p", {class: "muted"}, "spawn one below, or from any shell: ",
            el("code", {}, "unigym debug CONFIG [-b mujoco]")),
        el("p", {class: "muted"},
            "runs live in their own processes -- this portal may restart ",
            "without touching them."));
  }
  return el("div", {class: "grid cards"},
      alive.map((r) => runCard(r)));
}

/* ---- spawn wizard -------------------------------------------------------------- */

/* adapter-params face: declared adapters render one row per param
 * (doc tooltip + the declared default as the placeholder -- empty means
 * "leave the yaml value alone", so prefills can never clobber the
 * config); undeclared adapters fall back to a raw-JSON box.  The
 * product is ONLY body.adapter_params -- the spawn channel's red line
 * (adapter section, nothing else) is enforced server-side. */
function adapterParamRow(pname, spec) {
  const star = spec.required ? " *" : "";
  const def = spec.default === undefined || spec.default === null
      ? null : spec.default;
  let input;
  if (spec.type === "boolean") {
    input = el("select", {},
        el("option", {value: ""},
           def === null ? "(unset)" : `(default: ${def})`),
        el("option", {value: "true"}, "true"),
        el("option", {value: "false"}, "false"));
  } else if (spec.type === "number" || spec.type === "integer") {
    input = el("input", {type: "number", step: "any",
                         placeholder: def === null ? "number"
                             : `default: ${def}`});
  } else if (spec.type === "array") {
    input = el("input", {type: "text",
                         placeholder: def === null
                             ? 'JSON array, e.g. ["a", "b"]'
                             : `default: ${JSON.stringify(def)}`});
  } else {
    input = el("input", {type: "text",
                         placeholder: def === null ? "text"
                             : `default: ${def}`});
  }
  input.title = spec.doc || "";
  const label = el("label", {title: input.title},
      `${pname}${star}`, input);
  return {pname, spec, input};
}

function typedAdapterParam(row) {
  const {pname, spec} = row;
  const v = (row.input.value || "").trim();
  if (v === "") return undefined;          // omit -> yaml value survives
  if (spec.type === "number" || spec.type === "integer") {
    const n = Number(v);
    if (Number.isNaN(n)) throw new Error(
        `adapter param '${pname}': not a number: ${v}`);
    return [pname, n];
  }
  if (spec.type === "boolean") return [pname, v === "true"];
  if (spec.type === "array") {
    let arr;
    try { arr = JSON.parse(v); } catch {
      throw new Error(`adapter param '${pname}': not a JSON array: ${v}`);
    }
    return [pname, arr];
  }
  return [pname, v];
}

function spawnWizard() {
  const cfgSel = el("select", {name: "config"});
  const backend = el("select", {},
      el("option", {value: ""}, "(from yaml)"),
      el("option", {value: "mujoco"}, "mujoco"),
      el("option", {value: "isaaclab"}, "isaaclab"));
  const gpu = el("input", {type: "text",
                           placeholder: "GPU # / auto / empty = inherit"});
  const steps = el("input", {type: "number", min: "1",
                             placeholder: "collect.steps"});
  const instr = el("input", {type: "text",
                             placeholder: "adapter.instruction (language)"});
  const label = el("input", {type: "text", maxlength: "80",
                             placeholder: "label (display metadata only)"});
  const paused = el("input", {type: "checkbox"});
  const direct = el("input", {type: "checkbox"});   // director mode
  const adapterName = el("span", {class: "pill"}, "-");
  const adapterRows = el("div");
  const adapterRaw = el("textarea", {rows: "3", style: "width:100%",
      placeholder: "raw adapter section JSON for undeclared adapters"});
  let adapterRowList = [];
  let adapterSchemas = {};
  const pre = el("pre", {class: "pane", style: "max-height:140px"},
      "pick a config and spawn -- the result lands here.");
  const submit = el("button", {class: "primary", type: "submit"}, "spawn");

  function renderAdapterFace() {
    const opt = cfgSel.selectedOptions[0];
    const name = opt ? (opt.dataset.adapter || "") : "";
    adapterName.textContent = name || "-";
    // the label placeholder previews the display name this config
    // would produce (yaml run_name / file stem -- server-resolved in
    // /api/configs): empty input = that default shows in the ledger
    label.placeholder = opt && opt.dataset.run_name
        ? `default display name: ${opt.dataset.run_name}`
        : "label (display metadata only)";
    adapterRowList = [];
    adapterRows.replaceChildren();
    const schema = name ? adapterSchemas[name] : null;
    if (schema && schema.declared) {
      adapterRaw.style.display = "none";
      adapterRowList = Object.entries(schema.params)
          .map(([pname, spec]) => adapterParamRow(pname, spec));
      adapterRows.append(...adapterRowList.map((r) => r.input.parentNode));
    } else {
      adapterRaw.style.display = "";
      adapterRaw.value = "";
    }
  }

  function collectAdapterParams() {
    if (adapterRowList.length) {
      const out = {};
      for (const row of adapterRowList) {
        const kv = typedAdapterParam(row);
        if (kv) out[kv[0]] = kv[1];
      }
      return out;
    }
    const t = adapterRaw.value.trim();
    if (!t) return {};
    let doc;
    try { doc = JSON.parse(t); } catch (e) {
      throw new Error(`adapter params are not valid JSON: ${e.message}`);
    }
    if (typeof doc !== "object" || doc === null || Array.isArray(doc)) {
      throw new Error("adapter params must be a JSON object");
    }
    return doc;
  }

  const form = el("form", {onSubmit: async (e) => {
    e.preventDefault();
    showError("");
    pre.textContent = "spawning (server preflight + wait) ...";
    submit.disabled = true;
    try {
      let adapterParams;
      try { adapterParams = collectAdapterParams(); }
      catch (err) {
        pre.textContent = `failed: ${err.message}`;
        return;
      }
      const overrides = {backend: backend.value || null,
                         gpu: gpu.value.trim() || null};
      if (steps.value) overrides.steps = Number(steps.value);
      if (instr.value) overrides.instruction = instr.value;
      if (label.value.trim()) overrides.label = label.value.trim();
      const body = {config: cfgSel.value, overrides, debug: true,
                    wait_timeout: 120};
      if (Object.keys(adapterParams).length) {
        body.adapter_params = adapterParams;
      }
      if (paused.checked) body.paused = true;
      if (direct.checked) body.direct = true;
      const res = await api.start(body);
      if (res.doc && res.doc.ok) {
        const info = res.doc.info || {};
        pre.textContent = `ok -- run ${info.run_id || "?"} is up ` +
            `(pid ${res.doc.handle.pid})\nview: /view/${info.run_id}` +
            (res.doc.warnings?.length
                ? `\nwarnings: ${res.doc.warnings.join("; ")}` : "");
        const link = el("a",
            {href: `/view/${info.run_id}${location.search}`}, "open view →");
        pre.append(el("div", {}, link));
        refresh();          // instant feedback: the new run must show up
        refreshDone();      // in BOTH faces without waiting for a tick
      } else {
        pre.textContent = `failed: ${errMsg(res)}\n` +
            (res.doc?.hint ? `hint: ${res.doc.hint}` : "");
      }
    } finally {
      submit.disabled = false;
    }
  }},
  el("label", {}, "config (whitelisted)"), cfgSel,
  el("div", {class: "grid two"},
      el("div", {}, el("label", {}, "backend"), backend,
          el("label", {}, "steps override"), steps),
      el("div", {}, el("label", {}, "gpu"), gpu,
          el("label", {}, "instruction"), instr)),
  el("label", {title: "display-only name for the ledger (label || "
      + "run_name || run_id); <= 80 chars, never a path component"},
      "label (display)", label),
  el("label", {style: "display:flex;align-items:center;gap:6px",
               title: "spawn overlay of the yaml adapter section: one "
                   + "row per declared param (empty = the yaml value "
                   + "survives); undeclared adapters take raw JSON"},
     adapterName, " adapter params"),
  adapterRows,
  adapterRaw,
  el("label", {style: "display:flex;align-items:center;gap:6px"},
     paused, "start paused (born-paused: `unigym resume latest` releases)"),
  el("label", {style: "display:flex;align-items:center;gap:6px"},
     direct, "director mode (--direct: pin/follow/release/push on the "
     + "scene card; else rejected)"),
  el("div", {style: "margin-top:10px"}, submit),
  el("label", {}, "preflight result"), pre);
  cfgSel.addEventListener("change", renderAdapterFace);
  api.configs().then((res) => {
    const cfgs = res.doc?.configs || [];
    cfgSel.append(...cfgs.map((c) => {
      const bits = [c.backend || "?", c.adapter || "no adapter",
                    c.cameras?.length ? `${c.cameras.length} cam` : "no cam"];
      const opt = el("option", {value: c.path},
          `${c.relative}  (${bits.join(", ")})`);
      opt.dataset.adapter = c.adapter || "";
      opt.dataset.run_name = c.run_name || "";
      return opt;
    }));
    renderAdapterFace();
    if (res.doc?.ok === false) showError(errMsg(res));
  });
  api.adapterParams().then((res) => {
    adapterSchemas = res.doc?.adapters || {};
    renderAdapterFace();
  });
  return el("div", {class: "card"}, el("h2", {}, "spawn a run"), form,
      el("p", {class: "muted"},
          "spawned runs are setsid-detached SHM workers (no ports); ",
          "this portal serves their media & console."));
}

/* ---- recent runs (ledger card, design §3): the output_dir
 * book -- history (run.json) and live runs in ONE card.  Badge = task
 * face, verdict = exit code + wall time, direct artifact links via the
 * run= whitelist face (post-mortem: the ledger keeps output_dir
 * resolvable after the run leaves the live view).  Display name =
 * label || run_name || run_id (the label is display metadata written by
 * the run process into its own ledger; run_id never disappears -- it
 * degrades to the sub-line). ---------------------------------------- */

async function runsCard() {
  let doc;
  try { doc = (await api.runsHistory(12)).doc; } catch { return el("span"); }
  const rows = [];
  for (const r of doc?.runs || []) {
    const name = r.label || r.run_name || r.run_id || "?";
    // identity is never lost: run_id (and the run_name a label
    // displaced) degrade to the muted sub-line
    const sub = [r.run_id && r.run_id !== name ? r.run_id : null,
                 r.label && r.run_name && r.run_name !== r.label
                     ? r.run_name : null].filter(Boolean);
    const idLine = sub.length
        ? el("div", {class: "muted num", style: "font-size:11px"},
             sub.join(" · "))
        : null;
    const stale = r.state === "RUNNING" && r.stale;
    // UNKNOWN = gray pill (exists but unreadable), error inline
    const unknown = r.state === "UNKNOWN";
    const pill = unknown
        ? el("span", {class: "pill unknown",
                      title: r.error
                          || "segment exists but cannot be read"}, "UNKNOWN")
        : (stale
            ? el("span", {class: "pill bad",
                          title: "no exit record and the pid is gone "
                               + "(killed? daemon died?) -- no reaper, no "
                               + "verdict (no reaper, no verdict)"}, "RUNNING(stale)")
            : phasePill(r.state));
    // exit cell: code + wall time (absent = not reaped yet: live,
    // starting (pre-first-heartbeat spawn window), or RUNNING-stale when
    // the reaper is gone)
    const ex = r.exit;
    const exitCell = ex
        ? `${ex.code === null ? "sig" : `exit ${ex.code}`}` +
          (typeof ex.wall_time_s === "number"
              ? ` · ${Math.round(ex.wall_time_s)}s` : "")
        : (stale || unknown || r.state === "STARTING" ? "?" : "alive");
    const exitTitle = ex && ex.reason ? `${ex.reason}` +
        (ex.log_tail?.length ? "\n--- log tail ---\n"
            + ex.log_tail.join("\n") : "") : null;
    const startedTxt = new Date((r.started_ts || 0) * 1000)
        .toLocaleString(undefined,
            {month: "2-digit", day: "2-digit", hour: "2-digit",
             minute: "2-digit", second: "2-digit"});
    // results: per-task success rates (the light run.json summary),
    // linked to the full benchmark_report.json artifact
    const results = [];
    for (const [task, t] of Object.entries(r.results || {})) {
      const rate = (t.success_rate === null || t.success_rate === undefined)
          ? "n/a" : `${Math.round(t.success_rate * 100)}%`;
      results.push(el("div", {},
          el("a", {href: r.run_id
              ? artifactUrl({run: r.run_id, name: "benchmark_report.json"})
              : "#", title: "benchmark_report.json"},
              `${task} ${rate}`)));
    }
    const view = r.run_id
        ? el("a", {href: `/view/${encodeURIComponent(r.run_id)}${
            location.search}`}, "view") : null;
    rows.push(el("tr", {},
        el("td", {}, pill, " ", ...taskFacePills(r.task_face)),
        el("td", {}, el("span", {class: "run-link", title: r.label
                   ? `label: ${r.label}` : undefined}, name),
           idLine, r.paused
               ? el("div", {class: "muted", style: "font-size:11px"},
                    "born-paused") : null,
           unknown ? errorBlock(r.error) : null),
        el("td", {class: "num", title: exitTitle || undefined},
            exitCell,
            ex && ex.reason
                ? el("div", {class: "reason-line"}, ex.reason) : null,
            ex && ex.log_tail?.length
                ? el("details", {class: "logtail"},
                     el("summary", {}, `log tail (${ex.log_tail.length})`),
                     el("pre", {}, ex.log_tail.join("\n"))) : null),
        el("td", {}, results.length ? results : "-"),
        el("td", {class: "num"}, startedTxt),
        el("td", {}, view || "-"),
        el("td", {}, el("button", {class: "danger",
            title: "delete this run record and its data dir",
            onclick: async () => {
              if (!confirm(`delete run ${name}?\nthis removes the whole `
                  + `data dir:\n${r.output_dir}`)) return;
              const res = await api.deleteRun(r.output_dir);
              if (res.doc && res.doc.ok) {
                refresh();
                refreshDone();
              } else {
                showError(errMsg(res));
              }
            }}, "delete"))));
  }
  const body = rows.length
      ? el("table", {}, el("tr", {}, el("th", {}, "state"),
          el("th", {}, "run"), el("th", {}, "exit"),
          el("th", {}, "results"), el("th", {}, "started"),
          el("th", {}, ""), el("th", {}, "")), rows)
      : el("p", {class: "muted"},
          "no runs in the ledger yet -- every run records itself in its "
          + "output_dir (run.json); spawn one below.");
  return el("div", {class: "card"},
      el("h2", {}, "recent runs"),
      body,
      el("p", {class: "muted", style: "margin-top:6px"},
          "ledger = ", el("code", {}, "<output_dir>/run.json"),
          ` (scan root: ${doc?.output_root ?? "?"})`));
}

/* ---- mount -------------------------------------------------------------------------- */
export function mount() {
  document.getElementById("app").replaceChildren(
      el("div", {class: "grid two"},
          el("div", {id: "live"}),
          spawnWizard()),
      el("div", {id: "done", style: "margin-top:14px"}));
  poll(refresh, 3000);
  const ledgerLoop = () => refreshDone()
      .catch((e) => console.warn("ledger refresh failed:", e));
  ledgerLoop();                          // first paint must not be able to
  setInterval(ledgerLoop, 3000);         // kill the loop (was .then(setInterval))
  // SSE connection lamp: the stream is a latency win, the pollers
  // above are the correctness floor (contract §1.22) -- when the
  // stream drops, the page must SAY so instead of silently degrading
  // to polling.  Same 3s cadence, so the text is the truth.
  const lampDot = el("span", {class: "dot warn"});
  const lampTxt = el("span", {}, "poll only 3s");
  const header = document.querySelector("header");
  if (header) header.append(el("span", {class: "muted",
          style: "font-size:12px;gap:6px;display:inline-flex;"
              + "align-items:center",
          title: "SSE event stream (/api/events): green = push live, "
              + "yellow = the 3s poll floor is in charge"},
      lampDot, lampTxt));
  const setLamp = (up) => {
    lampDot.className = up ? "dot" : "dot warn";
    lampTxt.textContent = up ? "push live" : "poll only 3s";
  };
  // SSE nudge (contract §1.22): events say "changed" -- the SAME
  // refresh faces pull the truth (push notifies, pull confirms).  The
  // pollers above stay untouched: they are the correctness floor, the
  // stream only cuts the latency; onerror just leaves them in charge.
  const bump = debounce(() => {
    refresh().catch(() => {});
    ledgerLoop();
  }, 500);
  events((ev) => {
    if (ev.type === "reset") {           // ring overflow: re-pull right now
      refresh().catch(() => {});
      ledgerLoop();
      return;
    }
    bump();
  }, {types: ["run", "bench"], onState: setLamp});
}

async function refresh() {
  const res = await api.runs();
  if (res.doc?.ok) {
    const box = document.getElementById("live");
    if (box) box.replaceChildren(runsGrid(res.doc.runs || []));
  } else {
    showError(errMsg(res));
  }
}

async function refreshDone() {
  const card = await runsCard();
  const box = document.getElementById("done");
  if (box) box.replaceChildren(card);
}
