"""core.session -- attached run operations .

docs/designs/v026-unigymd.md §2: the bodies of the client.py SHM-family
commands (pause/resume/step/inspect/snapshot/capture/camera) extracted
behind explicit-argument methods.  A Session is a STATELESS-ATTACH
object: one temporary SHM attach (design §1(4)), one command at a time
(single-controller discipline, runtime_contract §3), closed as soon as
the operation returns.  The daemon's /api/op/<run>/<op> handlers call
these; during the CLI migration the CLI calls them directly (SCAFFOLDING
-- removed in the final step when the CLI switches to daemon HTTP only).

Output FORMATTING stays in the CLI layer (stdout shapes are the stable
contract); Session returns plain data and raises typed errors whose
messages are the exact pre-daemon CLI strings (minus the "[unigym <cmd>]"
prefix, which is CLI formatting).
"""
from __future__ import annotations

import time
from pathlib import Path

from unigym.runtime import bridge as _bridge
from unigym.runtime.bridge import ShmClient

_REPO_ROOT = Path(__file__).resolve().parents[2]


class SessionError(RuntimeError):
    """Attach/resolution failure (message = pre-daemon CLI text minus
    prefix)."""


class SessionOpError(SessionError):
    """A command failed (ack status=1, timeout, or missing frame)."""

    hint: str


def resolve_run_id(run: str, state_dir: str | None = None) -> str:
    """`latest` -> newest live run id (else the id unchanged)."""
    if run != "latest":
        return run
    ids = _bridge.active_run_ids()
    if not ids:
        raise SessionError(
            "no live bridge run (start with: unigym run CONFIG --serve)")
    return ids[-1]


def attach(run: str, timeout_s: float = 10.0,
           state_dir: str | None = None) -> Session:
    """Attach to a live run (temporary SHM attach; closes via close())."""
    run_id = resolve_run_id(run, state_dir)
    try:
        client = ShmClient(run_id)
    except (_bridge.BridgeError, FileNotFoundError) as e:
        raise SessionError(f"cannot attach to '{run}': {e}") from e
    return Session(client, timeout_s=timeout_s, state_dir=state_dir)


class Session:
    """One temporary attach; explicit-argument operations (§2 discipline)."""

    def __init__(self, client: ShmClient, timeout_s: float = 10.0,
                 state_dir: str | None = None):
        self.client = client
        self.timeout_s = float(timeout_s)
        self.state_dir = state_dir

    # -- pass-through identity/status -----------------------------------
    @property
    def run_id(self) -> str:
        return self.client.run_id

    @property
    def phase(self) -> str:
        return self.client.phase

    @property
    def paused(self) -> bool:
        return self.client.paused

    @property
    def port(self) -> int:
        return self.client.port

    @property
    def camera_name(self) -> str:
        return self.client.camera_name

    def status(self) -> dict:
        """Full SHM status snapshot (read-only)."""
        return self.client.read_status()

    def close(self) -> None:
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- commands ---------------------------------------------------------
    def _send(self, cmd: str, payload: dict | None = None,
              timeout_s: float | None = None) -> dict:
        timeout = self.timeout_s if timeout_s is None else timeout_s
        try:
            status, res = self.client.send(cmd, payload, timeout=timeout)
        except _bridge.BridgeTimeout as e:
            raise SessionOpError(str(e)) from e
        except _bridge.BridgeError as e:
            # 'run finished before ack ...' races (run dies between the
            # attach and the ack poll) used to escape every daemon-level
            # ``except SessionError`` clause and crash the HTTP handler
            # with an empty response -- normalize like BridgeTimeout.
            raise SessionOpError(str(e)) from e
        if status != 0:
            err = res.get("error", res) if isinstance(res, dict) else str(res)
            exc = SessionOpError(err)
            # run-side handlers may attach a next-action hint (director
            # contract §6); carried for the daemon's {error, hint} face
            if isinstance(res, dict) and res.get("hint"):
                exc.hint = res["hint"]
            raise exc
        return res

    def pause(self, timeout_s: float | None = None) -> dict:
        """Freeze at the step boundary -> {"phase", "paused"} post-state."""
        self._send("pause", timeout_s=timeout_s)
        return {"phase": self.phase, "paused": self.paused}

    def resume(self, timeout_s: float | None = None) -> dict:
        self._send("resume", timeout_s=timeout_s)
        return {"phase": self.phase, "paused": self.paused}

    def shutdown(self, timeout_s: float | None = None) -> dict:
        self._send("shutdown", timeout_s=timeout_s)
        return {"phase": self.phase, "paused": self.paused}

    def step(self, n: int = 1, timeout_s: float | None = None) -> dict:
        """Auto-pause then queue n policy steps (stays paused).

        Returns {"auto_paused", "stepping", "hint"} -- exactly the keys
        of the `unigym step --json` document (minus the constant
        ok:true, which is CLI formatting).
        """
        auto_paused = False
        if not self.paused:
            try:
                self._send("pause", timeout_s=timeout_s)
            except SessionOpError as e:
                raise SessionOpError(f"pause failed: {e}") from e
            auto_paused = True
        res = self._send("step", {"n": int(n)}, timeout_s=timeout_s)
        return {"auto_paused": auto_paused,
                "stepping": res.get("stepping", int(n)),
                "hint": res.get("hint")}

    def inspect(self, expr: str | None = None, target: str | None = None,
                key: str | None = None,
                timeout_s: float | None = None) -> dict:
        """Debug console: eval expr / fixed query / summary -> ack result."""
        if expr:
            payload = {"target": "eval", "expr": expr}
        elif target:
            payload = {"target": target, "key": key}
        else:
            payload = {"target": "summary"}
        return self._send("inspect", payload, timeout_s=timeout_s)

    def query(self, kind: str, timeout_s: float | None = None,
              params: dict | None = None):
        """Typed structured query (runtime_contract §4 cmd 10-12,16).

        ``metrics`` -> the typed document itself; ``config`` -> the
        effective-config snapshot (or None); ``report`` -> the benchmark
        draft (or None without a benchmark); ``scene`` -> the curated
        scene face (``params={"full": True}`` = the raw escape hatch).
        """
        res = self._send(kind, params, timeout_s=timeout_s)
        if kind == "metrics":
            return res
        return res.get(kind)

    def direct(self, op: str, payload: dict | None = None,
               timeout_s: float | None = None) -> dict:
        """Director command (cmd 14; director_contract.md §1).

        ``op`` in teleport|setvel|contact|attach|detach|pin|follow|
        release|push|gripper|status; the payload is the command body (the
        ``op`` key is injected here).  Ack errors raise SessionOpError
        with the run-side message; the contract ``hint`` rides the ack
        document on the daemon face.
        """
        body = dict(payload or {})
        body["op"] = op
        return self._send("direct", body, timeout_s=timeout_s)

    def watch(self, sub: str, payload: dict | None = None,
              timeout_s: float | None = None) -> dict:
        """Watcher command (cmd 15; watchers_contract.md §7).

        ``sub`` in add|list|clear|update|test; add/test carry the spec
        under ``spec``.  Ack errors (bad predicate with candidate keys,
        budget full) raise SessionOpError with the run-side message.
        """
        body = dict(payload or {})
        body["watch_op"] = sub
        return self._send("watch", body, timeout_s=timeout_s)

    # -- rewind face (runtime_contract §4 cmd 17-19) ----------------------
    def rewind_snapshot(self, label: str | None = None,
                        timeout_s: float | None = None) -> dict:
        """Capture an engine snapshot -> {"snap_id", "tick", ...}.

        Two-level gated run-side (backend SNAPSHOT trait + adapter
        rewindable); slot cap 16 LRU (checkpoint ring) with the
        eviction surfaced in the ack.
        """
        payload = {"label": label} if label else {}
        return self._send("snapshot", payload, timeout_s=timeout_s)

    def rewind(self, snap_id: int, timeout_s: float | None = None) -> dict:
        """Restore a snapshot -> {"restored_tick", "snap_id", "paused"}.

        The world auto-pauses on the ack (owner ruling §5-③); resume or
        step continues on the forked timeline."""
        return self._send("rewind", {"snap_id": int(snap_id)},
                          timeout_s=timeout_s)

    # -- frames -----------------------------------------------------------
    def _get_frame(self, name: str | None, env: int | None,
                   timeout_s: float | None,
                   render: dict | None = None,
                   pos: list | None = None,
                   lookat: list | None = None) -> tuple[dict, dict]:
        """get_view ack + published frame (the common two-step dance).

        ``pos`` + ``lookat`` (given TOGETHER, world frame) render this
        one frame from a TRANSIENT world pose (ux-7; the capture face
        composes it with ``render``)."""
        payload: dict = {"camera": name, "env_id": env}
        if render is not None:
            # RenderSpec capture (camera_contract §14): rgb always RAW
            # (no jpeg encode on this path); the seg/gif/still artifacts
            # + effective_render ride the ack when requested
            payload["render"] = render
        else:
            payload["jpeg"] = True
        if pos is not None and lookat is not None:
            payload["pos"] = pos
            payload["lookat"] = lookat
        res = self._send("get_view", payload, timeout_s=timeout_s)
        frame = self.client.read_frame(want_fid=res["frame_id"],
                                       timeout=5.0)
        if frame is None:
            raise SessionOpError("frame not published in time")
        return res, frame

    def snapshot(self, out: str | Path | None = None, camera: str | None = None,
                 env: int | None = None, pos: list | None = None,
                 lookat: list | None = None,
                 timeout_s: float | None = None) -> dict:
        """One JPEG frame (ux-7 transient pose optional) -> {"path", ...}.

        ``out`` default = `<run_id>-<camera>.jpg` in the cwd .
        ``pos``/``lookat`` are validated triples (CLI parses them).
        """
        posed = pos is not None
        name = camera or ("free" if posed else (self.camera_name or None))
        payload: dict = {"camera": name, "jpeg": True, "env_id": env}
        if posed:
            payload["pos"] = pos                # transient one-shot override
            payload["lookat"] = lookat          # (restored after the frame)
        res = self._send("get_view", payload, timeout_s=timeout_s)
        frame = self.client.read_frame(want_fid=res["frame_id"], timeout=5.0)
        if frame is None:
            raise SessionOpError("frame not published in time")
        cam = name or self.camera_name
        out_path = Path(out if out is not None
                        else f"{self.run_id}-{cam}.jpg")
        out_path.write_bytes(frame["data"])
        return {"path": str(out_path), "camera": cam,
                "frame_id": frame["frame_id"],          # displayed first
                "ack_frame_id": res["frame_id"], "posed": posed}

    def camera_get(self, out: str | Path | None = None,
                   name: str | None = None, env: int | None = None,
                   render: dict | None = None,
                   pos: list | None = None, lookat: list | None = None,
                   timeout_s: float | None = None) -> dict:
        """RenderSpec capture-face frame fetch -> {"path",
        "effective_render", "render_source", ...}.

        ``pos`` + ``lookat`` (world frame, both together) render the one
        frame from a TRANSIENT world pose (camera aim; world rigs
        only -- the rig pose restores before the ack).

        ``render`` is REQUIRED -- ``{"view": name}`` (optionally +
        complementary overrides) or a full inline spec (camera_contract
        §14; this session face is an INNER layer, the daemon API does
        not guess; direct API callers included).  With ``buffers:
        ["rgb","seg"]`` the response also carries ``seg_path``
        (colorized PNG written run-side under ``<out_dir>/capture/``)
        and ``legend`` (the id->name table); ``scene.geom_class:
        "collision"`` swaps the render to the collision-class geometry;
        orbit views add ``gif_path`` + ``frame_paths`` (run-side
        artifacts).  mujoco-only (an isaaclab run answers with the
        honest capture-face error).  A plain flagless JPEG frame is the
        ``snapshot`` op's job.
        """
        cam = name or self.camera_name
        if not cam:
            raise SessionOpError("no camera specified/available")
        if render is None:
            raise SessionOpError(
                "camera get requires render={'view': name} or a full "
                "RenderSpec dict (camera_contract §14; direct API "
                "callers included -- a plain frame is the snapshot "
                "op's job)")
        res, frame = self._get_frame(cam, env, timeout_s,
                                     render=render, pos=pos,
                                     lookat=lookat)
        out_path = Path(out if out is not None else f"{self.run_id}-{cam}.jpg")
        data = frame["data"]
        if out_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            # the render face publishes RAW rgb; re-encode so the
            # artifact matches its extension (PIL lazy, capture-caller
            # side)
            import io
            import numpy as np
            from PIL import Image

            arr = np.frombuffer(data, dtype=np.uint8).reshape(frame["shape"])
            buf = io.BytesIO()
            fmt = "PNG" if out_path.suffix.lower() == ".png" else "JPEG"
            Image.fromarray(arr).save(buf, format=fmt,
                                      **({"quality": 92} if fmt == "JPEG"
                                         else {}))
            data = buf.getvalue()
        out_path.write_bytes(data)
        doc: dict = {"name": cam, "frame_id": res["frame_id"],
                     "path": str(out_path)}
        for key in ("effective_render", "render_source", "seg_path",
                    "legend", "gif_path", "frame_paths", "intrinsics",
                    "seg_error", "gif_error"):
            if key in res:
                doc[key] = res[key]
        return doc

    def camera_set(self, name: str, params: dict, env: int | None = None,
                   timeout_s: float | None = None) -> dict:
        """Apply live camera properties -> {"name", "applied"}."""
        res = self._send("set_props", {"camera": name, "params": params,
                                       "env_id": env}, timeout_s=timeout_s)
        return {"name": name, "applied": params, "ack": res}

    def camera_get_pose(self, name: str | None = None,
                        env: int | None = None,
                        timeout_s: float | None = None) -> dict:
        """Read back one camera's current world pose (freecam
        face; fixed rigs only) -> {"camera", "rig", "pos", "quat_wxyz",
        "lookat"}."""
        cam = name or self.camera_name
        if not cam:
            raise SessionOpError("no camera specified/available")
        res = self._send("get_world_pose", {"camera": cam, "env_id": env},
                         timeout_s=timeout_s)
        return res

    # -- capture (ux-5) -----------------------------------------------------
    def _registry_entry(self) -> dict:
        from unigym.core import registry
        return registry.read_entry(self.run_id, self.state_dir)

    @staticmethod
    def _resolve_server_path(raw: str) -> Path:
        """Absolute path of a server-produced (possibly relative) artifact.

        Relative paths resolve against the collect process cwd (the repo
        root by spawn contract) first, then the caller's cwd.
        """
        p = Path(raw)
        if p.is_absolute():
            return p
        for cand in (_REPO_ROOT / p, Path.cwd() / p):
            if cand.is_file():
                return cand
        return _REPO_ROOT / p

    def _default_capture_out(self, camera: str) -> Path | None:
        od = self._registry_entry().get("output_dir")
        if not od:
            return None
        p = Path(od)
        if not p.is_absolute():
            p = _REPO_ROOT / p
        return p / "capture" / \
            f"{time.strftime('%Y%m%d_%H%M%S')}_{camera}.jpg"

    def capture(self, out: str | Path | None = None, npz: str | None = None,
                camera: str | None = None, env: int | None = None,
                timeout_s: float | None = None) -> dict:
        """One artifact, one absolute path (ux-5).

        Default: one JPEG frame (broadcast camera unless camera/env).
        ``npz`` in {obs, extra, all}: a console save_npz dump instead.
        Returns {"kind", "path", ...} WITHOUT the ok key -- ok/JSON
        wrapping is CLI formatting (stdout contract).
        """
        if npz:
            try:
                res = self._send("save_npz", {"source": npz},
                                 timeout_s=timeout_s)
            except SessionOpError as e:
                raise SessionOpError(f"save_npz failed: {e}") from e
            raw = res.get("npz_path")
            if not raw:
                raise SessionOpError(f"save_npz ack without npz_path: {res}")
            path = self._resolve_server_path(raw)
            if not path.is_file():
                raise SessionOpError(f"npz not found at {path} (ack: {raw})")
            return {"kind": "npz", "path": str(path),
                    "keys": res.get("keys", [])}
        name = camera or self.camera_name
        try:
            res = self._send("get_view", {"camera": name, "jpeg": True,
                                          "env_id": env},
                             timeout_s=timeout_s)
        except SessionOpError as e:
            raise SessionOpError(f"get_view failed: {e}") from e
        frame = self.client.read_frame(want_fid=res["frame_id"], timeout=5.0)
        if frame is None:
            raise SessionOpError("frame not published in time")
        if out:
            out_path = Path(out)
        else:
            out_path = self._default_capture_out(name)
            if out_path is None:
                raise SessionOpError(
                    f"no output_dir in the serve registry of "
                    f"{self.run_id}; pass --out explicitly")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(frame["data"])
        return {"kind": "jpg", "path": str(out_path.resolve()),
                "camera": name, "frame_id": res["frame_id"]}
