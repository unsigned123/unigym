/* scenecard.js -- the scene card on the run page (scene face;
 * director console: node-contextual actions; scene
 * editor: direct edit-mode toggle + per-value numeric inputs; joints
 * joint limits: the joints editor carries the tree.robot.joints model
 * ranges as native min/max + interval display + out-of-range blocks).
 *
 * The curated scene face (typed query cmd 16 via GET /api/runs/<id>/
 * scene): static tree as collapsible groups with live per-node summaries
 * from the state half, refreshed on the 2s status cadence.  Clicking a
 * node opens its detail pane: the full state JSON in a copyable <pre>.
 *
 * edit mode: the scene card carries a "direct edit" toggle
 * (default OFF) -- off means ZERO action buttons are rendered (not
 * hidden: not built); on, every prop/robot/contact row gets its
 * director actions, and double-clicking a row is shorthand for opening
 * that node's edit form (robot -> joints editor, prop -> teleport).
 *
 * Per-value input policy (hard rule): every vector parameter is
 * SEPARATE numeric boxes -- teleport pos x/y/z + quat w/x/y/z (quat may
 * stay blank), push lin x/y/z + ang x/y/z + mode dropdown, joints one
 * box per joint (group-collapsible, prefilled with the current scene
 * state).  Each box validates as it types and on submit (NaN/empty
 * required -> red border, no request).  There is deliberately NO
 * comma-string parsing anywhere in this file.
 *
 * joints boxes whose joint carries `limited: true` + a range
 * in tree.robot.joints get min/max attributes (native stepper/browser
 * clamping) and render `[lo, hi]`; out-of-range values turn the border
 * red as they type and BLOCK the submit (server side enforces the same
 * model limits -- director contract §3.11/§6).
 *
 * State badges (pinned / following -> host) come from
 * /api/op/<run>/direct/status -- the SAME path the CLI uses (dual-face
 * parity); a run without --direct keeps the read-only card.  Polls only
 * while on screen (IntersectionObserver).  Talks ONLY to api.js.
 */

import {api, errMsg} from "./api.js";
import {el, fmtNum, showError} from "./app.js";

const v3 = (a) => (Array.isArray(a) ? a.map((x) => fmtNum(x, 3)).join(", ")
                                    : "-");

export function sceneCard(runId) {
  const box = el("div", {id: "scene"},
      el("span", {class: "muted"}, "loading..."));
  const detail = el("pre", {class: "pane scene-pre", style: "display:none"});
  const note = el("span", {class: "muted"});
  let face = null;                   // last good {tree, state}
  let dir = null;                    // director status doc (null: no --direct)
  let dirTried = false;
  let noFace = false;                // scene-less worker: hint, not error
  let picked = null;                 // "robot"/node name, or an open form:
                                     // "tp:<prop>" | "jt:robot" | "pf:robot"
  let edit = false;                  // direct edit mode (default OFF)
  let visible = true;
  let poller = null;
  let busy = false;

  const note_ = (msg) => { note.textContent = msg || ""; };

  /* the ONE director action path (contract §5: same as the CLI) */
  const act = async (op, body) => {
    if (busy) return;
    busy = true;
    note_(`${op}...`);
    try {
      const res = await api.direct(runId, op, body);
      if (res.doc && res.doc.ok === false) {
        note_("");
        showError(errMsg(res) + (res.doc?.hint
            ? ` (hint: ${res.doc.hint})` : ""));
        return;
      }
      note_(`${op} ok`);
      picked = null;                 // applied: close any open form
      dirTried = false;              // badges re-probe after a mutation
      await tick();
    } catch (e) {
      note_("");
      showError(String(e.message || e));
    } finally {
      busy = false;
    }
  };

  const pick = (name) => () => {
    picked = name;
    render();
  };

  const detailText = (st) => {
    if (picked === "robot") return JSON.stringify(st.robot, null, 1);
    if (picked !== null && st.props && picked in st.props)
      return JSON.stringify({[picked]: st.props[picked]}, null, 1);
    if (picked !== null && st.contacts && picked in st.contacts)
      return JSON.stringify({[picked]: st.contacts[picked]}, null, 1);
    return null;
  };

  /* ---- per-value input helpers (NO comma parsing, by design) ------ */
  const btn = (label, onclick, cls = "") =>
      el("button", {class: `mini ${cls}`.trim(),
                    onclick: (e) => {
                      e.stopPropagation();
                      onclick();
                    }}, label);

  /* "" -> null (blank optional slot); finite -> number; else NaN (bad) */
  const parseNum = (s) => {
    const t = String(s ?? "").trim();
    if (t === "") return null;
    const v = Number(t);
    return Number.isFinite(v) ? v : NaN;
  };

  /* numBox: optional joint-limit face {min, max} -> native min/max
   * attributes + live out-of-range red border */
  const numBox = (key, val, ph, lim = null) => {
    const b = el("input", {class: "scene-num", "data-k": key,
                           inputmode: "decimal", placeholder: ph});
    if (lim) {
      b.min = String(lim.min);
      b.max = String(lim.max);
    }
    b.value = val === null || val === undefined ? "" : String(val);
    b.addEventListener("input", () => {        // live re-validation
      const v = parseNum(b.value);
      if (v === null) {
        b.classList.remove("bad");             // blank optional slot
      } else if (Number.isNaN(v)) {
        /* keep the red border (parse error) */
      } else {
        b.classList.remove("bad");
        if (lim && (v < lim.min || v > lim.max)) b.classList.add("bad");
      }
    });
    return b;
  };

  const boxVal = (b) => parseNum(b.value);
  const markBad = (b, bad) => b.classList.toggle("bad", bad);

  /* one vector group: {vals, bad, n} -- complete = n == boxes.length */
  const groupVals = (boxes) => {
    const vals = boxes.map(boxVal);
    return {vals, bad: vals.some((v) => v !== null && Number.isNaN(v)),
            n: vals.filter((v) => v !== null).length};
  };

  /* validate a possibly-optional vector group; red-borders violations */
  const checkGroup = (boxes, required) => {
    const g = groupVals(boxes);
    const need = required || (g.n > 0 && g.n < boxes.length);
    boxes.forEach((b, i) => markBad(b, Number.isNaN(g.vals[i])
        || (need && g.vals[i] === null)));
    return g;
  };

  /* ---- director console forms ------------------------------------------- */
  const formWrap = (...kids) =>
      el("span", {class: "scene-actions"}, ...kids,
         btn("x", () => {
           picked = null;
           render();
         }));

  const teleportForm = (name, p) => {
    const cur = p || {};
    const pos = [0, 1, 2].map((i) =>
        numBox(`tp:${name}:p${i}`, cur.pos?.[i], "xyz"[i]));
    const quat = [0, 1, 2, 3].map((i) =>
        numBox(`tp:${name}:q${i}`, cur.quat?.[i], "wxyz"[i]));
    const apply = btn("apply", () => {
      const gp = checkGroup(pos, true);
      const gq = checkGroup(quat, false);
      if (gp.bad || gp.n !== 3 || gq.bad || (gq.n > 0 && gq.n !== 4)) {
        showError("teleport: pos x/y/z required; quat w/x/y/z optional "
                  + "(all four or blank)");
        return;
      }
      const body = {name, pos: gp.vals};
      if (gq.n === 4) body.quat = gq.vals;
      act("teleport", body);
    }, "primary");
    return formWrap(el("span", {class: "grp"}, "pos ", ...pos),
                    el("span", {class: "grp"}, "quat ", ...quat), apply);
  };

  const pushForm = () => {
    const lin = [0, 1, 2].map((i) => numBox(`push:l${i}`, null, "xyz"[i]));
    const ang = [0, 1, 2].map((i) => numBox(`push:a${i}`, null, "xyz"[i]));
    const mode = el("select", {},
        el("option", {value: "add"}, "add"),
        el("option", {value: "set"}, "set"));
    const go = btn("push", () => {
      const gl = checkGroup(lin, false);
      const ga = checkGroup(ang, false);
      if (gl.bad || ga.bad) {
        showError("push: lin/ang need numbers in every filled group");
        return;
      }
      if (gl.n !== 3 && ga.n !== 3) {
        showError("push: fill all of lin x/y/z and/or ang x/y/z");
        return;
      }
      const body = {mode: mode.value};
      if (gl.n === 3) body.lin = gl.vals;
      if (ga.n === 3) body.ang = ga.vals;
      act("push", body);
    }, "danger");
    return formWrap(el("span", {class: "grp"}, "lin ", ...lin),
                    el("span", {class: "grp"}, "ang ", ...ang), mode, go);
  };

  /* tree.robot.joints -> {name: {min, max}} for limited joints */
  const jointLimits = () => {
    const lim = {};
    for (const j of face?.tree?.robot?.joints || [])
      if (j?.limited && Array.isArray(j.range) && j.range.length === 2)
        lim[j.name] = {min: j.range[0], max: j.range[1]};
    return lim;
  };

  const jointsPanel = () => {
    const groups = face?.tree?.robot?.joint_groups || {};
    const cur = face?.state?.robot?.joints || {};
    const lim = jointLimits();
    const boxes = [];                        // [jointName, input]
    const groupEls = Object.entries(groups).map(([g, names]) => {
      const rows = names.map((jn) => {
        const b = numBox(`jt:${jn}`, cur[jn]?.pos, "rad", lim[jn] || null);
        boxes.push([jn, b]);
        const kids = [jn, b];
        if (lim[jn])
          kids.push(el("span", {class: "muted", title: "model range"},
              `[${fmtNum(lim[jn].min, 2)}, ${fmtNum(lim[jn].max, 2)}]`));
        return el("label", {class: "box"}, ...kids);
      });
      return el("details", {class: "jt-group", open: ""},
          el("summary", {}, `${g} (${names.length})`),
          el("div", {class: "jt-grid"}, ...rows));
    });
    const zv = el("input", {type: "checkbox", id: "scene-jt-zv"});
    const zeroAll = btn("zero all",
        () => { for (const [, b] of boxes) b.value = "0"; });
    const apply = btn("apply", () => {
      let bad = false;
      let rangeMsg = "";
      const joints = {};
      for (const [jn, b] of boxes) {
        const v = boxVal(b);
        if (v === null || Number.isNaN(v)) {
          markBad(b, true);
          bad = true;
          continue;
        }
        const L = lim[jn];
        if (L && (v < L.min || v > L.max)) {
          markBad(b, true);
          bad = true;
          rangeMsg = rangeMsg
            || `${jn} outside model range [${fmtNum(L.min, 3)}, `
               + `${fmtNum(L.max, 3)}]`;
          continue;
        }
        joints[jn] = v;
      }
      if (bad) {
        showError(rangeMsg
                  || "joints: every box needs a number (rad)");
        return;
      }
      const body = {joints};
      if (zv.checked) body.zero_vel = true;
      act("joints", body);
    }, "primary");
    return el("div", {class: "scene-form block"},
        el("span", {class: "muted"},
            "robot joint writes (rad; prefill = current pose"
            + (Object.keys(lim).length
                ? `; min/max = model limits on ${Object.keys(lim).length}`
                : "") + ")"),
        ...groupEls,
        el("span", {class: "grp"},
            zv, el("label", {class: "inline", for: "scene-jt-zv"},
                   "zero_vel"),
            zeroAll, apply));
  };

  /* ---- node action buttons (rendered ONLY in edit mode) ------------------ */
  const followHosts = (self) => {
    const opts = [];
    if (dir?.gripper) opts.push(dir.gripper.name);   // vgrip (when declared)
    opts.push("hand");                               // robot hand alias
    for (const name of Object.keys(face?.tree?.props || {}))
      if (name !== self) opts.push(`prop:${name}`);
    return opts;
  };

  const propActions = (name, p) => {
    const kids = [btn("teleport", () => {
                    picked = `tp:${name}`;
                    render();
                  }),
                  btn("pin", () => act("pin", {name})),
                  btn("release", () => act("release", {name}))];
    const sel = el("select", {},
        ...followHosts(name).map((h) => el("option", {value: h}, h)));
    kids.push(btn("follow", () => act("follow",
                                      {obj: name, host: sel.value})),
              sel);
    return kids;
  };

  const badges = (name) => {
    const out = [];
    if (!dir) return out;
    if ((dir.pinned || []).includes(name))
      out.push(el("span", {class: "pill pause", title: "director pin"},
                  "pinned"));
    if (dir.following && name in dir.following)
      out.push(el("span", {class: "pill run",
                           title: "director follow"}, `→${dir.following[name]}`));
    return out;
  };

  const actions = (name, p) => {
    const kids = [...badges(name)];
    if (dir && edit) {
      if (name === "robot")
        kids.push(btn("push", () => {
                      picked = "pf:robot";
                      render();
                    }),
                   btn("joints", () => {
                     picked = "jt:robot";
                     render();
                   }));
      else if (name in (face?.tree?.props || {}))
        kids.push(...propActions(name, p));
      else if (name in (face?.tree?.contacts || {}))
        kids.push(btn(p.state?.on ? "force off" : "force on",
                      () => act("contact",
                                {group: name,
                                 state: p.state?.on ? "off" : "on"})));
    }
    return kids.length
        ? el("span", {class: "scene-actions"}, ...kids) : null;
  };

  /* ---- rows -------------------------------------------------------------- */
  const summaryLine = (name, detailTxt, onPick, extra, onDbl) => {
    const row = el("div", {class: "scene-row", onclick: onPick},
        el("span", {class: "scene-name"}, name),
        el("span", {class: "muted num scene-detail"}, detailTxt));
    if (extra) row.append(extra);
    if (onDbl) {
      row.addEventListener("dblclick", (e) => {
        e.stopPropagation();
        onDbl();
      });
    }
    return row;
  };

  const nodeRows = (sc) => {
    const t = sc.tree || {};
    const st = sc.state || {};
    const r = st.robot || {};
    const canEdit = Boolean(dir && edit);
    const rows = [];
    rows.push(summaryLine(
        `robot ${t.robot?.name ?? ""}`,
        `base ${t.robot?.base ?? "-"} · ${t.robot?.n_joints ?? "-"} joints · `
        + `pos ${v3(r.base_pos_w)}`, pick("robot"),
        actions("robot", {state: r}),
        canEdit ? () => {
          picked = "jt:robot";               // dblclick = joints editor
          render();
        } : null));
    for (const [name, info] of Object.entries(t.props || {})) {
      const p = (st.props || {})[name] || {};
      const joints = p.joints ? Object.keys(p.joints).length : 0;
      rows.push(summaryLine(
          `prop ${name}`,
          `${info.type ?? "?"} · pos ${v3(p.pos)}`
          + (joints ? ` · ${joints} joint(s)` : ""),
          pick(name),
          actions(name, {state: p}),
          canEdit ? () => {
            picked = `tp:${name}`;           // dblclick = teleport form
            render();
          } : null));
    }
    for (const [name, info] of Object.entries(t.contacts || {})) {
      const c = (st.contacts || {})[name] || {};
      rows.push(summaryLine(
          `contact ${name}`,
          `${(info.members || []).length} members · `
          + `${c.on ? "ON" : "off"} · fz ${fmtNum(c.force_z, 2)}N`,
          pick(name),
          actions(name, {state: c})));
    }
    rows.push(summaryLine(
        "cameras", (t.cameras || []).join(", ") || "-", null));
    return rows;
  };

  /* the open-form states: "tp:<prop>" | "jt:robot" | "pf:robot" */
  const formOpen = () => picked !== null
      && (String(picked).startsWith("tp:") || picked === "jt:robot"
          || picked === "pf:robot");

  let lastSig = null;
  const render = (open) => {
    if (!face) return;
    if (open) picked = open;
    const kids = nodeRows(face);
    if (picked === "jt:robot") kids.push(jointsPanel());
    else if (picked === "pf:robot") kids.push(pushForm());
    else if (picked !== null && String(picked).startsWith("tp:")) {
      const nm = String(picked).slice(3);
      kids.push(teleportForm(nm, (face.state || {}).props?.[nm]));
    }
    kids.push(detail);
    if (picked !== null && !formOpen()) {
      const txt = detailText(face.state || {});
      if (txt !== null) {
        detail.style.display = "block";
        detail.textContent = `${picked}\n${txt}`;
      } else {
        detail.style.display = "none";
      }
    } else {
      detail.style.display = "none";
    }
    // edit toggle lives in the card head (outside the swapped box)
    editBtn.style.display = dir ? "" : "none";
    editBtn.textContent = `direct edit: ${edit ? "on" : "off"}`;
    editBtn.classList.toggle("on", edit);
    // content signature: skip the DOM swap when nothing changed -- the
    // 2s poll must NOT rebuild buttons under the user's cursor.  While a
    // form is open the LIVE state half is frozen out of the signature
    // (an edit session owns its prefill snapshot; applying re-syncs).
    const sig = JSON.stringify([face.tree, dir, picked, note.textContent,
                                edit, formOpen() ? null : face.state]);
    if (sig === lastSig) return;
    lastSig = sig;
    box.replaceChildren(...kids, el("div", {}, note));
  };

  const tick = async () => {
    if (!visible) return;              // off-screen: no polling (§2)
    try {
      const res = await api.scene(runId);
      if (res.status === 200 && res.doc?.scene) {
        face = res.doc.scene;
      } else if (res.status !== 200 && !face && !noFace) {
        noFace = true;                 // old worker / gone run: hint only
        box.replaceChildren(el("span", {class: "muted"},
            "no scene face on this run (a scene-enabled worker is required) -- "
            + "card stays idle"));
        return;
      }
      // director status: a run without --direct rejects with the
      // explicit REJECT_ERROR -> the read-only card.  Probed once
      // per visibility and again after every director action (the badge
      // data source must track mutations -- dirTried is the refresh
      // request flag).  A transient failure (tunnel blip / 404 on a
      // dying run / ack timeout under load) keeps the last known face
      // and retries on the next tick -- ONLY the explicit "not enabled"
      // rejection drops the card to read-only (the edit-mode UI
      // rides on dir, so any other 400 must NOT kill it).
      if (!dirTried) {
        dirTried = true;
        try {
          const ds = await api.directorStatus(runId);
          if (ds.status === 200 && ds.doc?.enabled) dir = ds.doc;
          else if (ds.status === 400
                   && String(ds.doc?.error || "").includes("not enabled"))
            dir = null;                  // the explicit rejection
          else dirTried = false;         // transient: retry next tick
        } catch {
          dirTried = false;              // network blip: retry next tick
        }
      }
      if (face) render();
    } catch {                          // blip: keep last content
    }
  };

  const start = () => {
    if (poller === null) poller = setInterval(tick, 2000);
  };
  const stop = () => {
    if (poller !== null) {
      clearInterval(poller);
      poller = null;
    }
  };

  /* direct edit toggle (card head; default OFF = zero buttons) */
  const editBtn = el("button", {class: "mini", title: "scene editor: "
      + "action buttons + edit forms (default off; dblclick a row = "
      + "open its form)", onclick: () => {
        edit = !edit;
        if (!edit) picked = null;      // leaving edit mode closes forms
        lastSig = null;
        render();
      }}, "direct edit: off");

  const card = el("div", {class: "card", style: "margin-top:14px"},
      el("div", {class: "card-head"},
          el("h2", {}, "scene"), editBtn), box);
  if (typeof IntersectionObserver === "function") {
    new IntersectionObserver((entries) => {
      visible = entries.some((e) => e.isIntersecting);
      if (visible) {
        dirTried = false;              // re-arm the director probe
        tick();
        start();
      } else {
        stop();
      }
    }, {threshold: 0.05}).observe(card);
  }
  start();
  return card;
}
