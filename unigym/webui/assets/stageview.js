/* stageview.js -- the live-view panel (module split of runpage.js;
 * WebRTC is the DEFAULT route -- designs/v048-stream-qos §4-d).
 *
 * Owns everything about THE PICTURE: the stage with the camera switcher
 * (stream_bind), the mode radio group (Auto / WebRTC / MJPEG) and the
 * dead-run notice.  Posture: on first bind the page starts BOTH
 * routes -- the MJPEG <img> shows life immediately while the WebRTC
 * handshake runs; when the <video> actually decodes its first frame the
 * MJPEG pump is released (src dropped) and WebRTC takes the stage.
 * Any failure -- signalling error, ICE failed/closed, or no first frame
 * within RTC_CONNECT_TIMEOUT_S -- falls back to MJPEG silently (the
 * fallback is the correctness floor; measured anchors: signalling RTT
 * p50 187ms, live at 23.3fps within ~1-2s over the ssh tunnel, so 8s
 * is ~4x worst case -- never pre-judge the ICE leg before first frame).
 * The radios carry the user's CHOICE (Auto = the designed WebRTC-first
 * + MJPEG-safety-net behavior); the status line under the stage shows
 * the EFFECTIVE mode highlighted, with connection state / fps /
 * resolution sampled at 1s (fps = totalVideoFrames delta for WebRTC,
 * onload delta for MJPEG -- the v048 measurement recipe).  runpage.js
 * mounts it, renders the bar pieces (camera select + mode radios) and
 * feeds each status snapshot via refresh(st); the module talks ONLY to
 * api.js.
 */

import {api, media, errMsg, query} from "./api.js";
import {el, fmtNum, showError} from "./app.js";

const RTC_CONNECT_TIMEOUT_MS = 8000;

const tokenSuffix = () => query("token")
    ? `token=${encodeURIComponent(query("token"))}` : "";

export function mountStageView(runId) {
  let pc = null;                    // RTCPeerConnection while WebRTC is on
  let mode = "auto";                // user choice: auto | rtc | mjpg
  let wantRTC = false;              // the WebRTC leg is currently wanted
  let rtcLive = false;              // first video frame actually decoded
  let rtcWatch = null;
  let mjpegRetries = 0;
  let mjpegFrames = 0;              // img.onload counter (fps sampling)
  let lastSample = null;            // {t, rtc, mjpg} fps-difference base

  const mjpegSrc = (camera) => {
    const base = `/run/${encodeURIComponent(runId)}/cam/` +
        `${encodeURIComponent(camera)}.mjpg`;
    return base + (tokenSuffix() ? (base.includes("?") ? "&" : "?")
        + tokenSuffix() : "");
  };

  function setMjpegSource(camera) {
    if (!camera) return;                 // no camera known yet: stay unbound
    const img = document.getElementById("mjpg");
    img.dataset.bound = camera;          // refresh() binds only once
    img.style.display = "";
    mjpegRetries = 0;                    // explicit (re)bind: fresh budget
    img.onload = () => { mjpegRetries = 0; mjpegFrames += 1; };
    img.onerror = () => {                // stream died (daemon restart, drop,
      if (rtcLive || img.dataset.bound !== camera) return;   // network blip)
      if (mjpegRetries >= 5) return;    // bounded self-heal: re-kick
      mjpegRetries += 1;
      setTimeout(() => {
        if (!rtcLive && img.dataset.bound === camera) {
          img.src = mjpegSrc(camera);
        }
      }, 2000);
    };
    img.src = mjpegSrc(camera);
  }

  /* MJPEG becomes (or stays) the live view -- the radio choice and
   * every WebRTC failure path land here (the fallback is the floor). */
  function toMjpeg() {
    wantRTC = false;
    rtcLive = false;
    if (rtcWatch) { clearTimeout(rtcWatch); rtcWatch = null; }
    if (pc) { try { pc.close(); } catch {} pc = null; }
    document.getElementById("rtc").style.display = "none";
    const img = document.getElementById("mjpg");
    img.style.display = "";
    if (!img.dataset.bound) {            // was released for a live WebRTC
      setMjpegSource(document.getElementById("camsel").value);
    }
  }

  async function startRTC(auto = false) {
    wantRTC = true;
    const v = document.getElementById("rtc");
    pc = new RTCPeerConnection();
    const thisPc = pc;
    pc.addTransceiver("video", {direction: "recvonly"});
    pc.createDataChannel("control");
    pc.ontrack = (ev) => {
      v.srcObject = ev.streams[0] || new MediaStream([ev.track]);
    };
    // the swap waits for a DECODABLE frame, not the track attach: the
    // ICE leg must never be pre-judged before it had a chance
    v.onloadeddata = () => {
      if (!wantRTC || pc !== thisPc) return;
      rtcLive = true;
      if (rtcWatch) { clearTimeout(rtcWatch); rtcWatch = null; }
      v.style.display = "";
      const img = document.getElementById("mjpg");
      img.style.display = "none";
      img.removeAttribute("src");        // release the MJPEG pump
      img.dataset.bound = "";            // toMjpeg() rebinds on fallback
    };
    pc.onconnectionstatechange = () => {
      if (["failed", "closed"].includes(pc.connectionState) && wantRTC) {
        toMjpeg();
      }
    };
    rtcWatch = setTimeout(() => {
      if (wantRTC && !rtcLive) toMjpeg();
    }, RTC_CONNECT_TIMEOUT_MS);
    try {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const r = await media.offer(runId, pc.localDescription.sdp);
      if (!r.doc || r.doc.sdp === undefined) {
        throw new Error(errMsg(r, "WebRTC signalling failed"));
      }
      await pc.setRemoteDescription({type: "answer", sdp: r.doc.sdp});
    } catch (e) {
      toMjpeg();
      if (!auto) showError(String(e && e.message || e));
    }
  }

  /* Radio intent -> routes.  Auto = the designed WebRTC-first behavior
   * (silent fallbacks); WebRTC = explicit attempt (failure toasts, the
   * MJPEG floor still catches it); MJPEG = no WebRTC leg at all.
   * Applies immediately -- a switch takes effect on the live stream. */
  async function applyMode(m) {
    mode = m;
    if (m === "mjpg") { toMjpeg(); return; }
    await startRTC(m !== "auto");
  }

  const radio = (val, label) => {
    const r = el("input", {type: "radio", name: "stagemode", value: val,
                           onchange: () => applyMode(val)});
    r.checked = val === "auto";          // Auto is the designed default
    return el("label", {class: "mode-opt", title: val === "auto"
        ? "WebRTC first, silent MJPEG fallback (v048 QoS)"
        : val === "rtc" ? "force WebRTC (MJPEG still catches failures)"
        : "force MJPEG (~10fps, no WebRTC leg)"}, r, " " + label);
  };
  const modeBox = el("span", {id: "modebox"}, radio("auto", "Auto"),
      radio("rtc", "WebRTC"), radio("mjpg", "MJPEG"));

  /* Effective-mode status line: highlighted mode pill + connection
   * state / fps / resolution, sampled at 1s.  fps = the v048 recipe:
   * totalVideoFrames delta for WebRTC, onload delta for MJPEG. */
  const modeStat = el("div", {id: "modestat", class: "muted"}, "-");
  let lastMedia = null;             // the status poll's media face (MJPEG
                                    // fps face: multipart imgs don't fire
                                    // onload per frame in Chrome, so the
                                    // client cannot count decodes -- show
                                    // the server publish rate instead)

  function sampleModeStat() {
    const now = performance.now();
    const v = document.getElementById("rtc");
    const img = document.getElementById("mjpg");
    let rtcN = 0;
    if (rtcLive && v.getVideoPlaybackQuality) {
      rtcN = v.getVideoPlaybackQuality().totalVideoFrames;
    }
    let dt = null;
    let dRtc = null;
    if (lastSample) {
      dt = (now - lastSample.t) / 1000;
      dRtc = rtcN - lastSample.rtc;
    }
    lastSample = {t: now, rtc: rtcN};
    // MJPEG: multipart <img> never fires onload per frame in Chrome --
    // the client-side counter cannot move, so the fps face shows the
    // server's publish rate (the honest number the stream carries)
    const fps = rtcLive
        ? (dRtc !== null && dt > 0 ? dRtc / dt : null)
        : (lastMedia && typeof lastMedia.publish_rate === "number"
            ? lastMedia.publish_rate : null);
    const fpsTitle = rtcLive ? "decoded fps (client-side)"
        : "publish rate (server-side)";
    const res = rtcLive
        ? (v.videoWidth ? `${v.videoWidth}\u00d7${v.videoHeight}` : null)
        : (img.naturalWidth ? `${img.naturalWidth}\u00d7${img.naturalHeight}`
                            : null);
    const conn = rtcLive
        ? (pc ? pc.connectionState : null)
        : (img.dataset.bound
            ? (lastMedia && lastMedia.stale ? "stalled" : "streaming")
            : null);
    const active = rtcLive ? ["WebRTC", "run"]
        : img.dataset.bound ? ["MJPEG", "collect"] : null;
    modeStat.replaceChildren(...
        [el("span", {class: "muted"}, "active: "),
         active ? el("span", {class: `pill ${active[1]}`}, active[0])
                : el("span", {class: "pill unknown"}, "\u2014"),
         el("span", {class: "muted"}, " · "),
         el("span", {class: "num", title: fpsTitle}, fmtNum(fps)), " fps",
         el("span", {class: "muted"}, " · "),
         el("span", {class: "num"}, res || "\u2014"),
         conn ? el("span", {class: "muted"}, " · " + conn) : null,
        ].filter(Boolean));      // replaceChildren coerces null -> "null"
  }
  setInterval(sampleModeStat, 1000);

  const camSel = el("select", {id: "camsel", onchange: async (e) => {
    const cam = e.target.value;
    if (!cam) return;
    const r = await api.streamBind(runId, cam);
    if (r.doc && r.doc.ok === false) { showError(errMsg(r)); return; }
    if (!rtcLive) setMjpegSource(cam);   // WebRTC follows the pump itself
  }});

  /* ---- live style selector (camera unification; §5) ----
   * ONE dropdown per camera = that camera's live render STYLE (the
   * publish loop's render state, written via camera_set_properties
   * render= -- the same state unigym_camera(style) writes; the stream
   * re-renders on the spot).  rgb = the plain path (bitwise default),
   * collision / contacts / cage+contacts = registered style views.
   * LAST ACTION WINS: the dropdown sends {"view": name}; any advanced
   * widget change sends the inline spec built from the controls (the
   * widgets compose on top of the style; all-default = rgb).  The
   * badge + source label read the server state (status render_source).
   * isaaclab runs (capture_backend != mujoco) disable the whole row. */
  const STYLE_VIEWS = [["visual", "rgb"],
                       ["clean_collision", "collision"],
                       ["contacts", "contacts"],
                       ["cage_contacts", "cage+contacts"]];
  let renderCam = "";               // the camera the state was last read on
  let renderBusy = false;
  async function applyRenderSpec(spec) {
    if (renderBusy || !renderCam) return;
    renderBusy = true;
    try {
      const r = await api.cameraOp(runId,
          {camera_op: "set", name: renderCam, params: {render: spec}});
      if (r.doc && r.doc.ok === false) showError(errMsg(r));
    } catch (e) { showError(String(e && e.message || e)); }
    finally { renderBusy = false; }
  }
  const styleSel = el("select", {id: "stylesel", title: "per-camera live "
      + "render style (the publish loop re-renders the stream on the "
      + "spot; recordings follow -- WYSIWYG)"},
      ...STYLE_VIEWS.map(([v, label]) => el("option", {value: v}, label)));
  styleSel.value = "visual";             // the designed default
  const controlsSpec = () => {
    const spec = {};
    if (!geomVis.checked) spec.scene = {geom_class: "collision"};
    const decor = [];
    if (contactsChk.checked) decor.push({gen: "contact_mjvis", target: "*"});
    if (decor.length) spec.decor = decor;
    if (segChk.checked) spec.buffers = ["rgb", "seg"];
    return Object.keys(spec).length ? spec : {view: "visual"};
  };
  styleSel.onchange = () =>
      applyRenderSpec({view: styleSel.value});
  const geomVis = el("input", {type: "radio", name: "geommode",
                               onchange: () => applyRenderSpec(controlsSpec())});
  const geomCol = el("input", {type: "radio", name: "geommode",
                               onchange: () => applyRenderSpec(controlsSpec())});
  geomVis.checked = true;                 // the designed default
  const contactsChk = el("input", {type: "checkbox",
                                   onchange: () => applyRenderSpec(controlsSpec())});
  const segChk = el("input", {type: "checkbox",
                              onchange: () => applyRenderSpec(controlsSpec())});
  const renderBadge = el("span", {id: "renderbadge", class: "pill flag-active",
                                  title: "the live stream is rendered under "
                                      + "a non-default render spec "
                                      + "(WYSIWYG: recordings follow)",
                                  style: "display:none"},
      "render active");
  const renderSrc = el("span", {class: "muted", id: "rendersrc"}, "");
  const flagsRow = el("span", {id: "renderrow", title: "live render state "
      + "(publish loop state; capture stays stateless)"},
      el("span", {class: "muted"}, "style"), styleSel, " ",
      el("span", {class: "muted"}, "geom"),
      el("label", {class: "mode-opt"}, geomVis, " visual"),
      el("label", {class: "mode-opt"}, geomCol, " collision"),
      el("label", {class: "mode-opt"}, contactsChk, " contacts"),
      el("label", {class: "mode-opt"}, segChk, " seg"),
      " ", renderBadge, " ", renderSrc);

  function resetRenderControls() {
    styleSel.value = "visual";
    geomVis.checked = true; geomCol.checked = false;
    contactsChk.checked = false; segChk.checked = false;
  }

  function syncRenderFromStatus(st) {
    // the authoritative state lives server-side: the badge + source
    // label reflect it; the widgets stay the user's PENDING choice
    // (a spec state cannot be reverse-mapped onto the widgets
    // honestly -- the applied source label is the truth)
    if (st.camera && st.camera !== renderCam) {
      renderCam = st.camera;             // camera switch: fresh defaults
      resetRenderControls();
    }
    const src = st.render_source || null;
    renderBadge.style.display = src ? "" : "none";
    renderSrc.textContent = src ? `live: ${src}` : "";
  }

  /* ---- expand-to-fill: overlay the page, stream untouched ---- */
  const card = el("div", {class: "card"}, el("h2", {}, "live view"),
      el("div", {id: "stage"},
          el("img", {id: "mjpg", alt: "live stream",
                     style: "display:none"}),   // shown on first bind
          el("video", {id: "rtc", autoplay: "", playsinline: "",
                       muted: "", style: "display:none"}),
          el("div", {id: "pausebadge", class: "pause-badge",
                     style: "display:none"},
              el("span", {class: "pill pause"}, "PAUSED"),
              el("span", {class: "muted"},
                  " · scene frozen · cam live"))),
      el("div", {class: "stage-flags"}, flagsRow), modeStat);
  const zoomBtn = el("button", {id: "zoombtn", title: "fill the page "
          + "(Esc to collapse)"}, "\u2922");
  const setExpanded = (on) => {
    card.classList.toggle("stage-expanded", on);
    zoomBtn.textContent = on ? "\u2921" : "\u2922";
    zoomBtn.title = on ? "collapse (Esc)" : "fill the page (Esc to collapse)";
  };
  zoomBtn.onclick = () => setExpanded(!card.classList.contains("stage-expanded"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && card.classList.contains("stage-expanded")) {
      setExpanded(false);
    }
  });
  const h2row = el("div", {class: "card-head"},
      el("h2", {}, "live view"), zoomBtn);
  card.replaceChild(h2row, card.firstChild);

  /* Status-fed refresh (the 2s poller in runpage.js passes each
   * snapshot): sync the camera options, honor the publishing camera,
   * self-heal the very first bind if the initial /status raced. */
  async function refresh(st) {
    const img = document.getElementById("mjpg");
    // the media face lives on the /api mirror status (the portal
    // /run/<id>/status face does not carry it) -- it feeds the MJPEG
    // fps/conn readout (client-side decode counting is impossible for
    // a multipart <img>)
    try {
      const full = await api.runStatus(runId);
      lastMedia = full.doc?.media || lastMedia;
    } catch {}                     // status face is best-effort here
    // paused = the SCENE freezes, the feed does not (the parked loop
    // keeps publishing at its own 25Hz cadence, so a free camera can
    // fly and the rate readout stays live) -- the badge just names the
    // phase; hiding telemetry here would mask a real stall
    const badge = document.getElementById("pausebadge");
    if (badge) badge.style.display = st.phase === "PAUSED" ? "" : "none";
    // live render toolbar: reflect the server state + auto-disable on
    // a backend without the capture face (isaaclab / unknown: the
    // field is always carried, null = no capture face)
    const noCapture = st.capture_backend !== "mujoco";
    flagsRow.classList.toggle("disabled", !!noCapture);
    for (const w of [styleSel, geomVis, geomCol, contactsChk, segChk]) {
      w.disabled = !!noCapture;
    }
    if (noCapture) {
      renderBadge.style.display = "none";
      renderSrc.textContent = "";
      flagsRow.title = "the render face is mujoco-only (this run: "
          + (st.capture_backend || "no capture face") + ")";
    } else {
      syncRenderFromStatus(st);
    }
    const info = await media.info(runId);
    const inst = info.doc?.cameras?.instantiated || [];
    if (camSel.dataset.cur !== JSON.stringify(inst)) {
      camSel.replaceChildren(...inst.map((c) => el("option", {value: c}, c)));
      camSel.dataset.cur = JSON.stringify(inst);
    }
    if (st.camera && camSel.value !== st.camera) camSel.value = st.camera;
    if (!img.dataset.bound && !rtcLive && (st.camera || camSel.value)) {
      setMjpegSource(st.camera || camSel.value);
    }
  }

  /* First contact at mount time: bind the stage or explain a dead run.
   * WebRTC is the default route -- MJPEG binds as the visible
   * safety net, WebRTC takes over on its first decoded frame. */
  async function firstBind() {
    const res = await media.status(runId);
    if (res.status !== 200) {          // dead/unknown run: say so instead
      const stage = document.getElementById("stage");   // of a silent blank
      if (stage) stage.append(el("p", {class: "muted"},
          "no live view -- the run is not answering (finished, removed "
          + "or the URL is stale). Artifacts stay in the run's "
          + "output_dir; the result cards beside render post-mortem."));
      return;
    }
    const cam = res.doc?.camera || res.doc?.cameras?.instantiated?.[0] || "";
    setMjpegSource(cam);
    if (mode === "mjpg") return;         // explicit MJPEG choice: no RTC leg
    await startRTC(true);
  }

  return {card, camSel, modeBox, refresh, firstBind};
}
