/* inspectcard.js -- the camera Inspect panel (v054 RenderSpec).
 *
 * The v031 camera-inspect skeleton, rebuilt on the RenderSpec capture
 * face (camera_contract §14): view dropdown (registered views) + geom
 * radio + contacts/seg checkbox + camera dropdown (RunInfo.cameras) +
 * a single-frame button + a continuous (1Hz) checkbox; the result area
 * holds rgb/seg tabs, the legend color list and a GIF player (orbit
 * views answer gif_path).  Every capture sends a RenderSpec -- the
 * dropdown sends {"view": name} merged with the seg override
 * (buffers is complementary-safe against every builtin view); the
 * geom/contacts controls build an inline fragment and are DISABLED
 * while a view that already owns scene/decor is selected (the merge
 * compiler fail-fasts double claims -- the panel never asks for one).
 * Frames go to POST /api/op/<run>/camera with
 * out=<output_dir>/capture/inspect_<ts>_<cam>.png; the artifacts come
 * back through the /artifact whitelist face (capture/<basename>).
 *
 * isaaclab runs auto-disable (capture_backend != mujoco: no capture
 * face -- honest, not hidden).  annotate is NOT here (adapter-side
 * /render; this panel is the engine capture face).  Talks ONLY to
 * api.js.
 */

import {api, artifactUrl, errMsg, media} from "./api.js";
import {el, showError} from "./app.js";

export function inspectCard(runId) {
  let cam = "";
  let camsLoaded = false;
  let outputDir = null;
  let noCapture = false;
  let timer = null;
  let seq = 0;
  let lastSeg = null;              // artifact name of the latest seg PNG

  const camSel = el("select", {onchange: (e) => { cam = e.target.value; }});

  /* registered views (camera_contract §14; "visual" = the plain frame).
   * scene/decor-owning views disable the geom/contacts widgets. */
  const VIEWS = [
    {name: "visual", label: "visual", owns: []},
    {name: "clean_collision", label: "clean_collision", owns: ["scene"]},
    {name: "palm_pose_occl", label: "palm_pose_occl",
     owns: ["scene", "decor"]},
    {name: "palm_pose_xray", label: "palm_pose_xray",
     owns: ["scene", "decor"]},
  ];
  const viewSel = el("select", {onchange: (e) => {
    syncViewLocks();
    if (autoShot.checked) captureOnce();
  }}, ...VIEWS.map((v) => el("option", {value: v.name}, v.label)));
  viewSel.value = "visual";
  const viewOwns = () => (VIEWS.find((v) => v.name === viewSel.value)
                          || VIEWS[0]).owns;

  const geomVis = el("input", {type: "radio", name: "insgeom"});
  const geomCol = el("input", {type: "radio", name: "insgeom"});
  geomVis.checked = true;          // the designed default
  const contactsChk = el("input", {type: "checkbox"});

  const segChk = el("input", {type: "checkbox"});

  /* orbit(GIF) checkbox: composes the camera+output aspects into the
   * capture spec (complementary against visual/clean_collision, which
   * own neither) -- the GIF player below shows the returned artifact. */
  const orbitChk = el("input", {type: "checkbox"});

  /* the capture spec: {"view": name} + the complementary overrides.
   * buffers rides with every view (nobody declares it); scene/decor
   * fragments only for views that do not own them; camera/output only
   * for views that do not declare them. */
  function spec() {
    const s = {};
    if (viewSel.value !== "visual") s.view = viewSel.value;
    if (!viewOwns().includes("scene") && !geomVis.checked) {
      s.scene = {geom_class: "collision"};
    }
    if (!viewOwns().includes("decor") && contactsChk.checked) {
      s.decor = [{gen: "contact_mjvis", target: "*"}];
    }
    if (orbitChk.checked) {
      s.camera = {mode: "orbit", d: 0.20, el: -32.0,
                  orbit: {n: 24, step_deg: 15.0, ms: 125}};
      s.output = {gif: true, gif_ms: 125};
    }
    if (segChk.checked) s.buffers = ["rgb", "seg"];
    return s;
  }

  function syncViewLocks() {
    const owns = viewOwns();
    const geomLocked = owns.includes("scene");
    const contactsLocked = owns.includes("decor");
    const orbitLocked = owns.includes("camera") || owns.includes("output");
    geomVis.disabled = geomLocked || noCapture;
    geomCol.disabled = geomLocked || noCapture;
    contactsChk.disabled = contactsLocked || noCapture;
    orbitChk.disabled = orbitLocked || noCapture;
    const t = viewSel.value === "visual" ? null
        : `${viewSel.value} owns ${(owns || ["the plain render"]).join("+")}`
            + " -- conflicting controls locked";
    viewSel.title = t || "registered RenderSpec view";
  }

  /* result area: rgb/seg tabs + legend list + gif player */
  const imgRgb = el("img", {alt: "rgb capture", class: "inspect-img",
                            style: "display:none"});
  const imgSeg = el("img", {alt: "segmentation capture", class: "inspect-img",
                            style: "display:none"});
  const imgGif = el("img", {alt: "orbit gif capture", class: "inspect-img",
                            style: "display:none"});
  const tabRgb = el("button", {class: "tab active", onclick: () => {
    showTab("rgb");
  }}, "rgb");
  const tabSeg = el("button", {class: "tab", onclick: () => {
    showTab("seg");
  }}, "seg");
  const tabs = el("div", {class: "inspect-tabs"}, tabRgb, tabSeg);
  const legendList = el("div", {class: "inspect-legend muted"}, "-");
  const status = el("div", {class: "muted"}, "idle");

  function showTab(which) {
    tabRgb.classList.toggle("active", which === "rgb");
    tabSeg.classList.toggle("active", which === "seg");
    imgRgb.style.display = which === "rgb" ? "" : "none";
    imgSeg.style.display = which === "seg" && lastSeg ? "" : "none";
  }

  function renderLegend(legend) {
    const objs = (legend && legend.objects) || [];
    if (!objs.length) {
      legendList.replaceChildren("(no legend objects hit)");
      return;
    }
    legendList.replaceChildren(...objs.map((o) => {
      const sw = el("span", {class: "swatch"});
      sw.style.background =
          `rgb(${o.color[0]},${o.color[1]},${o.color[2]})`;
      return el("div", {class: "legend-row"}, sw,
          el("span", {class: "mono"}, o.name),
          el("span", {class: "muted"},
              ` (${o.type}${o.body ? ", " + o.body : ""}) · `
              + `${o.pixels}px`));
    }));
  }

  function setEnabled(on) {
    for (const w of [camSel, viewSel, segChk, orbitChk, shotBtn]) {
      w.disabled = !on;
    }
    contChk.disabled = !on;
    syncViewLocks();
  }

  async function captureOnce() {
    if (!cam || noCapture) return;
    seq += 1;
    const name = `${String(Date.now())}_inspect${seq}_${cam}.png`;
    const out = outputDir
        ? `${outputDir.replace(/\/$/, "")}/capture/${name}` : null;
    status.replaceChildren("capturing ", el("span", {class: "num"}, cam),
        " …");
    try {
      const body = {camera_op: "get", name: cam, render: spec()};
      if (out) body.out = out;
      const r = await api.cameraOp(runId, body);
      if (!r.doc || r.doc.ok === false) {
        status.replaceChildren(el("span", {class: "bad"}, errMsg(r)));
        showError(errMsg(r, "capture failed"));
        return;
      }
      const doc = r.doc;
      imgRgb.src = artifactUrl({run: runId, name: `capture/${name}`});
      imgRgb.style.display = "";
      lastSeg = doc.seg_path
          ? `capture/${doc.seg_path.split("/").pop()}` : null;
      if (lastSeg) {
        imgSeg.src = artifactUrl({run: runId, name: lastSeg});
        tabSeg.classList.remove("off");
      } else {
        imgSeg.removeAttribute("src");
        tabSeg.classList.add("off");
        if (tabSeg.classList.contains("active")) showTab("rgb");
      }
      if (doc.gif_path) {
        imgGif.src = artifactUrl({run: runId,
            name: `capture/${doc.gif_path.split("/").pop()}`});
        imgGif.style.display = "";
      } else {
        imgGif.removeAttribute("src");
        imgGif.style.display = "none";
      }
      renderLegend(doc.legend);
      const src = doc.render_source
          ? ` · render ${doc.render_source}` : "";
      const extras = (doc.gif_path ? " · gif" : "")
          + (doc.frame_paths
              ? ` · +${Object.keys(doc.frame_paths).length} views` : "");
      status.replaceChildren("frame ",
          el("span", {class: "num"}, String(doc.frame_id)), src, extras);
    } catch (e) {
      status.replaceChildren(el("span", {class: "bad"},
          String(e && e.message || e)));
    }
  }

  const shotBtn = el("button", {onclick: () => captureOnce()},
      "\u5355\u5e27");
  const autoShot = el("input", {type: "checkbox", onchange: (e) => {
    if (e.target.checked) {
      captureOnce();
      timer = setInterval(captureOnce, 1000);
    } else if (timer) {
      clearInterval(timer);
      timer = null;
    }
  }});
  const contChk = autoShot;

  const card = el("div", {class: "card"},
      el("h2", {}, "camera inspect"),
      el("div", {class: "bar wrap"},
          el("span", {class: "muted"}, "camera"), camSel,
          el("span", {class: "muted"}, "view"), viewSel,
          el("span", {class: "muted"}, "geom"),
          el("label", {class: "mode-opt"}, geomVis, " visual"),
          el("label", {class: "mode-opt"}, geomCol, " collision"),
          el("label", {class: "mode-opt"}, contactsChk, " contacts"),
          el("label", {class: "mode-opt"}, segChk, " seg"),
          el("label", {class: "mode-opt"}, orbitChk, " orbit(GIF)"),
          shotBtn,
          el("label", {class: "mode-opt"}, contChk,
              " \u8fde\u7eed(1Hz)")),
      imgGif, tabs, imgRgb, imgSeg, legendList, status);

  setEnabled(false);

  /* mount-time + refresh(): camera options / backend gate / output dir */
  async function refresh() {
    try {
      const info = await media.info(runId);
      const doc = info.doc || {};
      outputDir = doc.output_dir || outputDir;
      const inst = doc.cameras?.instantiated || [];
      if (inst.length && camSel.dataset.cur !== JSON.stringify(inst)) {
        camSel.replaceChildren(...inst.map((c) => el("option", {}, c)));
        camSel.dataset.cur = JSON.stringify(inst);
        if (!cam || !inst.includes(cam)) cam = inst[0];
      }
      const backend = doc.capture_backend;
      if (backend !== "mujoco" && !noCapture) {
        noCapture = true;
        setEnabled(false);
        status.replaceChildren(el("span", {class: "muted"},
            `the render capture face is mujoco-only (this run: `
            + (backend || "no capture face") + `)`));
        return;
      }
      if (!noCapture && cam) setEnabled(true);
    } catch {                       // gone/dead run: stay disabled
    }
  }
  refresh();
  // the portal pollers re-pull status anyway; a slow refresh keeps the
  // camera list / backend gate current without a second subscription
  setInterval(refresh, 4000);

  return {card, refresh};
}
