/* freecam.js -- interactive free-fly control for the "free" camera
 * (standalone module; runpage.js only mounts it).
 *
 * Enabled while the watched camera is named "free" (the debug fly rig:
 * a FIXED camera by convention, console contract §free -- name-based
 * enablement is the minimal face, see daemon_api_contract §1.14).
 *
 * Click the stage to pointer-lock, then:
 *   mousemove    yaw/pitch, 0.002 rad/px, pitch clamped to ~+-89 deg
 *   W/A/S/D      world-horizontal move, heading = yaw (Minecraft creative)
 *   Space/Shift  world up / down
 *   wheel        movement speed gear (x0.25 .. x8, default 1 m/s)
 *
 * Mechanics: the pose integrates locally at rAF rate and pushes
 * THROTTLED (>=100 ms and only with input) through the camera op face
 *   POST /api/op/<run>/camera  {camera_op: "set", name, pos, lookat}
 * with pos/lookat in the WORLD frame (snapshot semantics; lookat =
 * pos + unit view direction).  The initial pose comes from
 * camera_op:"get_pose".  Esc / a second click / tab-blur exits the
 * lock; keys are never hijacked while unfocused; every failure lands
 * in the unified #err area.
 */

import {api, errMsg} from "./api.js";
import {el, showError} from "./app.js";

const SENS = 0.002;          // rad per pointer pixel
const PUSH_MS = 100;         // contract: >=50 ms between pushes
const PITCH_LIM = 1.55;      // ~89 deg (no gimbal flip)
const BASE_SPEED = 1.0;      // m/s at gear x1

export function mountFreeCam(runId) {
  const stage = document.getElementById("stage");
  if (!stage) return;

  let active = false;          // watched camera is "free"
  let locked = false;          // pointer lock engaged
  let pos = null;              // [x, y, z] world frame
  let yaw = 0, pitch = 0;      // rad; yaw about +Z, pitch = elevation
  let speed = BASE_SPEED;      // m/s (wheel gear)
  let dirty = false;           // pose changed since last push
  const keys = new Set();
  let pushTimer = 0, raf = 0, lastT = 0, pollTimer = 0;

  const hint = el("div", {class: "fc-hint"});
  hint.style.display = "none";
  stage.append(hint);

  const forward = () => {
    const cp = Math.cos(pitch);
    return [cp * Math.cos(yaw), cp * Math.sin(yaw), Math.sin(pitch)];
  };

  const updateHint = () => {
    hint.textContent = locked
        ? `freecam FOCUSED · WASD horizontal · Space/Shift up/down · ` +
          `speed x${speed.toFixed(2)} · Esc exit`
        : "点击画面聚焦 · WASD 移动 · Space/Shift 升降 · Esc 退出";
  };

  /* ---- pose readback (entry) ------------------------------------------- */
  const ensurePose = async () => {
    if (pos) return;
    const r = await api.cameraOp(runId,
        {camera_op: "get_pose", name: "free"});
    const d = r.doc;
    if (!d || d.ok === false || !Array.isArray(d.pos)) {
      throw new Error(errMsg(r, "freecam: pose readback failed"));
    }
    pos = d.pos.slice();
    const dir = (d.lookat || [pos[0] + 1, pos[1], pos[2]])
        .map((v, i) => v - pos[i]);
    const n = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    yaw = Math.atan2(dir[1], dir[0]);
    pitch = Math.asin(Math.max(-1, Math.min(1, dir[2] / n)));
  };

  /* ---- throttled push (>=PUSH_MS, only with input) ----------------------- */
  let pushing = false;                 // ONE set_props in flight: the
                                       // bridge command slot is single --
                                       // an overlapping push overwrites the
                                       // slot and orphans the first req_id
                                       // ("no ack within ... set_props").
  const push = async () => {
    if (pushing || !dirty || !pos) return;
    pushing = true;
    dirty = false;                     // input during flight re-arms dirty:
    const f = forward();               // the next tick coalesces the latest
    try {                              // pose -- nothing is lost
      const r = await api.cameraOp(runId,
          {camera_op: "set", name: "free", pos: pos.slice(),
           lookat: [pos[0] + f[0], pos[1] + f[1], pos[2] + f[2]]});
      if (r.doc && r.doc.ok === false) showError(errMsg(r));
    } catch (e) {
      dirty = true;                    // transient: retry on the next tick
      showError(String(e.message || e));
    } finally {
      pushing = false;
    }
  };

  /* ---- local integration (60 fps while focused) -------------------------- */
  const integrate = (dt) => {
    if (!keys.size) return;
    /* ---- Minecraft-creative movement: WASD travels the world-horizontal
     * plane (heading = yaw only -- pitch never tilts travel); Space/Shift
     * own the world Z axis exclusively.  Gimbal lock: none reachable --
     * yaw/pitch has no roll, pitch is clamped to +-89 deg, and horizontal
     * travel depends on yaw alone (the strafe vector is also the exact
     * screen-right direction's horizontal projection at any pitch). */
    const heading = [Math.cos(yaw), Math.sin(yaw), 0];
    const strafe = [Math.sin(yaw), -Math.cos(yaw), 0];  // screen-right, horizontal
    const v = speed * dt;
    const move = (d, k) => { pos[0] += d[0] * k; pos[1] += d[1] * k;
                             pos[2] += d[2] * k; };
    if (keys.has("KeyW")) move(heading, v);
    if (keys.has("KeyS")) move(heading, -v);
    if (keys.has("KeyD")) move(strafe, v);
    if (keys.has("KeyA")) move(strafe, -v);
    if (keys.has("Space")) pos[2] += v;
    if (keys.has("ShiftLeft") || keys.has("ShiftRight")) pos[2] -= v;
    dirty = true;
  };

  const frame = (t) => {
    if (!locked) return;
    if (lastT) integrate(Math.min((t - lastT) / 1000, 0.1));
    lastT = t;
    raf = requestAnimationFrame(frame);
  };

  const stopLoop = () => {
    if (pushTimer) { clearInterval(pushTimer); pushTimer = 0; }
    if (raf) { cancelAnimationFrame(raf); raf = 0; }
    lastT = 0;
  };

  /* ---- pointer lock lifecycle -------------------------------------------- */
  document.addEventListener("pointerlockchange", () => {
    locked = document.pointerLockElement === stage;
    hint.classList.toggle("on", locked);
    keys.clear();
    if (locked) {
      updateHint();
      pushTimer = setInterval(push, PUSH_MS);
      raf = requestAnimationFrame(frame);
    } else {
      stopLoop();
      push();                          // flush the final pose
      updateHint();
    }
  });

  stage.addEventListener("click", () => {
    if (!active || locked) {
      if (locked) document.exitPointerLock();   // second click = exit
      return;
    }
    ensurePose().then(() => {
      const p = stage.requestPointerLock();
      if (p && p.catch) {
        p.catch(() => showError("freecam: pointer lock rejected "
            + "(iframe/sandbox? use a top-level tab)"));
      }
    }).catch((e) => showError(e.message));
  });

  document.addEventListener("mousemove", (e) => {
    if (!locked) return;
    yaw -= e.movementX * SENS;
    pitch = Math.max(-PITCH_LIM,
        Math.min(PITCH_LIM, pitch - e.movementY * SENS));
    dirty = true;
  });

  const FLY_KEYS = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "Space",
                            "ShiftLeft", "ShiftRight"]);
  window.addEventListener("keydown", (e) => {
    if (!locked) return;
    if (e.key === "Escape") {          // UA normally consumes Esc itself;
      document.exitPointerLock();      // this covers embedded contexts
      return;
    }
    if (!FLY_KEYS.has(e.code)) return;   // unfocused: no hijack
    e.preventDefault();
    keys.add(e.code);
  });
  window.addEventListener("keyup", (e) => { keys.delete(e.code); });

  stage.addEventListener("wheel", (e) => {
    if (!locked) return;
    e.preventDefault();
    speed = Math.min(8, Math.max(0.25, speed * (e.deltaY < 0 ? 1.25 : 0.8)));
    updateHint();
  }, {passive: false});

  document.addEventListener("visibilitychange", () => {
    if (document.hidden && locked) document.exitPointerLock();
  });

  /* ---- activation watcher: watched camera == "free" ---------------------- */
  const sync = () => {
    const sel = document.getElementById("camsel");
    const isFree = !!sel && sel.value === "free";
    if (isFree === active) return;
    active = isFree;
    hint.style.display = active ? "" : "none";
    if (!active && locked) document.exitPointerLock();
    updateHint();
  };
  pollTimer = setInterval(sync, 1000);
  sync();
  window.addEventListener("pagehide", () => {
    clearInterval(pollTimer);
    stopLoop();
  });
}
