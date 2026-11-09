"""Director mode: the directable state space.

A Director mounted on a ``--direct`` run turns the scene into a settable
fixture: props teleport, velocities overwrite, contact sensors override,
a virtual gripper walks around carrying things -- and with the console the
KINEMATIC primitives are the universal grasp face: ``pin`` freezes an
object in place, ``follow`` carries it host-relative (ghost rigid body:
it pushes the robot "hard", the robot pushes it "soft"), ``release``
hands it back to dynamics.  ``attach``/``detach`` remain as alternate
aliases (attach defaults to kinematic; ``--mode weld`` is the
capability-gated physical mode for declared props), and ``push`` writes
the robot root velocity.  Every command marks the current episode
``directed`` -- a directed success proves fixture wiring, never policy
capability (contract §0).

Wiring (collect_cli): ``unigym run CONFIG --direct`` constructs the
Director after the env and registers it

* as a bridge command handler (cmd 14 ``direct`` -- the ``direct.*`` op
  family), and
* as a SINK (per-tick kinematic follower application + episode-end
  directed flags).

Runs started WITHOUT ``--direct`` never construct a Director; the bridge
serves cmd 14 with a fail-fast rejection + restart hint (contract §6).
The ``--direct`` flag ALSO gates spec-side model injection on backends
that inject director bodies (gripper + weld slots) -- without it the
model is byte-identical (regression red line).

Backend mechanics (teleport/setvel/attach/host-pose addressing) live in
per-backend :class:`director.ops.DirectorPropOps` implementations.  The
ops face is INJECTED by the run layer through ``BackendInfo.director_ops``
and the former backend-name branches collapsed
into one dispatch point, and the weld/kinematic split is a registry trait
check (``Trait.DIRECTOR_ATTACH_WELD`` / ``..._SLAVE``), not a string
compare.

Module-level imports stay stdlib+numpy only; both engines are lazy
(backend_contract §10.2 -- ``director/`` is agnostic).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from .ops import build_ops, pose_compose, pose_rel, strip_prop_prefix

if TYPE_CHECKING:
    from .gripper import Gripper

__all__ = ["Director", "DirectorError", "OPS", "REJECT_ERROR", "REJECT_HINT", "Rejector"]

OPS = (
    "teleport",
    "setvel",
    "contact",
    "attach",
    "detach",
    "pin",
    "follow",
    "release",
    "push",
    "joints",
    "gripper",
    "status",
)

# attach-mode spellings: the canonical kinematic face + the follow alias
KIN_MODES = ("kin", "slave")

REJECT_ERROR = "director is not enabled on this run (started without --direct)"
REJECT_HINT = "restart with: unigym run CONFIG --direct"


class DirectorError(ValueError):
    """A rejected command; ``.hint`` is the next-action string."""

    def __init__(self, msg: str, hint: str = ""):
        super().__init__(msg)
        self.hint = hint or "(none)"


def _fail(msg: str, hint: str = "") -> DirectorError:
    return DirectorError(msg, hint)


# --------------------------------------------------------------------------- #
# the rejecting face (runs without --direct; still logs the attempt)
# --------------------------------------------------------------------------- #
class Rejector:
    """Bridge cmd-14 handler for runs started without ``--direct``.

    Rejects every command with the contract §6 pair; the attempt still
    lands in the shared command log (``rejected: true``) so the audit
    trail shows guard-gate hits.
    """

    def __init__(self, log_path: Path | None):
        self.log_path = log_path

    def handle(self, rid: int, params: dict) -> dict:
        self._log(params, rid)
        raise _fail(REJECT_ERROR, REJECT_HINT)

    def _log(self, params: dict, rid: int) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "rid": rid,
                            "op": (params or {}).get("op"),
                            "params": params or {},
                            "ok": False,
                            "rejected": True,
                            "error": REJECT_ERROR,
                        }
                    )
                    + "\n"
                )
        except OSError:  # best effort: the command log must never break dispatch
            pass


# --------------------------------------------------------------------------- #
# the Director
# --------------------------------------------------------------------------- #
class Director:
    """Run-side executor of the ``direct.*`` op family.

    Args:
        env: the wired env backend (mujoco or isaaclab).
        contact_manager: optional ContactManager (contact op target).
        cfg: the yaml ``director:`` section (gripper declaration).
        out_dir: run output dir (the command log + status.json root).
        status_fn: zero-arg callable returning the current status dict
            (the bridge's read_status -- used to stamp ``tick`` into the
            command log); optional.
        director_ops: the ``BackendInfo.director_ops`` zero-argument factory
            injected by the run layer.  Required: the neutral
            Director never inspects ``env`` to discover a backend.
    """

    def __init__(
        self,
        env,
        contact_manager=None,
        cfg: dict | None = None,
        out_dir: str | Path | None = None,
        status_fn=None,
        *,
        director_ops=None,
    ):
        self.env = env
        self.contact_manager = contact_manager
        self.cfg = dict(cfg or {})
        self.status_fn = status_fn
        # per-backend mechanics seam injected by the run layer
        self._ops = build_ops(env, director_ops)
        self.backend = self._ops.backend
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self.log_path = (
            self.out_dir / "director" / "commands.jsonl" if self.out_dir is not None else None
        )
        self.enabled = True
        # per-env directed marks (contract §7: cleared at episode end)
        self._directed: set[int] = set()
        self._ep_flags: list[dict] = []
        self._n_commands = 0
        # attachments: prop -> {"host", "mode": kin|weld,
        #                       "kind": pin|follow (kin), pose|rel}
        self._attachments: dict[str, dict] = {}
        # resolved at bind_env
        self._gripper: Gripper | None = None
        self._bind()

    # ------------------------------------------------------------------ #
    # binding
    # ------------------------------------------------------------------ #
    def _bind(self) -> None:
        from .gripper import normalize_gripper_cfg

        gcfg = normalize_gripper_cfg(self.cfg.get("gripper"))
        self._gripper = cast("Gripper | None", self._ops.bind(gcfg))

    # ------------------------------------------------------------------ #
    # sink protocol (kinematic followers + directed flags)
    # ------------------------------------------------------------------ #
    def on_step(self, t, bundle) -> None:
        """Per-tick kinematic follower application (contract §3.4/§3.8).

        The director state application point (pre-step for the NEXT
        physics tick -- the same hook slave following used):
        pinned objects are written back at their pinned pose, followers
        at host x locked-offset; both carry zero root velocity (the
        ghost discipline -- release resumes dynamics at rest).
        """
        if not self._attachments:
            return
        for obj, att in self._attachments.items():
            if att["mode"] != "kin":
                continue
            if att["kind"] == "pin":
                pos, quat = att["pose"]
            else:
                hpos, hquat = self._host_pose(att["host"])
                pos, quat = pose_compose(hpos, hquat, *att["rel"])
            self._write_slave_pose(obj, pos, quat)
            self._ops.halt(obj)

    def on_episode_end(self, env_ids, bundle) -> None:
        """Snapshot + clear per-env directed marks (contract §7)."""
        extra = bundle.get("extra") if isinstance(bundle, dict) else None
        for ei in np.asarray(env_ids).reshape(-1).tolist():
            ei = int(ei)
            self._ep_flags.append(
                {
                    "env_id": ei,
                    "episode_id": (
                        int(extra["episode_id"][ei])
                        if isinstance(extra, dict) and "episode_id" in extra
                        else None
                    ),
                    "directed": ei in self._directed,
                }
            )
            self._directed.discard(ei)

    def is_episode_directed(self, env_id: int) -> bool:
        """Live per-env mark (BenchmarkSink asks BEFORE the snapshot)."""
        return int(env_id) in self._directed

    def on_step_end(self, t, bundle) -> None:
        pass

    def on_run_end(self) -> dict:
        doc = self.status_doc()
        doc["episodes"] = list(self._ep_flags)
        if self.out_dir is not None:
            try:
                p = self.out_dir / "director" / "status.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(doc, indent=1, default=str))
            except OSError:  # best effort: status sidecar is telemetry only
                pass
        return doc

    # ------------------------------------------------------------------ #
    # command dispatch (bridge cmd-14 handler face)
    # ------------------------------------------------------------------ #
    def handle(self, rid: int, params: dict) -> dict:
        params = dict(params or {})
        op = params.pop("op", None)
        if op is None:
            raise _fail("direct payload needs an 'op' key", f"valid ops: {', '.join(OPS)}")
        if op not in OPS:
            raise _fail(f"unknown direct op {op!r}", f"valid ops: {', '.join(OPS)}")
        mutating = not (op == "status" or (op == "gripper" and params.get("action") == "where"))
        try:
            result = getattr(self, f"_op_{op}")(params)
        except DirectorError as e:
            self._log({**params, "op": op}, ok=False, error=str(e))
            raise
        if mutating:
            self._mark()
        self._log({**params, "op": op}, ok=True, result=result)
        return result

    def _mark(self) -> None:
        """Any mutating command marks the (single) env's episode directed."""
        self._directed.add(0)

    def _log(
        self, params: dict, ok: bool, result: dict | None = None, error: str | None = None
    ) -> None:
        """Append the command log (contract §7).

        Audit semantics: EVERY processed command counts (``commands`` in
        the status face), read-only and failed ones included.
        """
        self._n_commands += 1
        st = None
        if self.status_fn is not None:
            try:
                st = self.status_fn()
            except Exception:  # noqa: BLE001 (logging)
                st = None
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tick": int(st.get("step", -1)) if st else None,
            "op": params.get("op"),
            "params": params,
            "ok": bool(ok),
            "directed_envs": sorted(self._directed),
        }
        if error is not None:
            entry["error"] = error
        if result is not None:
            entry["result"] = result
        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(entry, default=str) + "\n")
            except OSError:  # best effort: log sidecar is telemetry only
                pass

    # ------------------------------------------------------------------ #
    # ops
    # ------------------------------------------------------------------ #
    def _op_teleport(self, p: dict) -> dict:
        name = self._need(p, "name")
        pos = self._opt_vec(p, "pos", 3)
        quat = self._opt_vec(p, "quat", 4)
        reset_vel = bool(p.get("reset_vel", True))
        joints = p.get("joints") or {}
        if not isinstance(joints, dict):
            raise _fail("teleport 'joints' must be a {joint: rad} mapping")
        if pos is None and quat is None and not joints:
            raise _fail("teleport needs pos and/or quat and/or joints")
        return self._ops.teleport(name, pos, quat, reset_vel, joints)

    def _op_setvel(self, p: dict) -> dict:
        name = self._need(p, "name")
        lin = self._need_vec(p, "lin", 3)
        ang = self._opt_vec(p, "ang", 3)
        return self._ops.setvel(name, lin, ang)

    def _op_contact(self, p: dict) -> dict:
        group = self._need(p, "group")
        state = self._need(p, "state")
        if state not in ("on", "off", "clear"):
            raise _fail(f"contact state must be on|off|clear, got {state!r}")
        if self.contact_manager is None:
            raise _fail(
                "this run declares no contacts: section",
                "add a contacts: group to the config to override it",
            )
        available = [s.name for s in self.contact_manager.specs if s.format == "contact"]
        try:
            if state == "clear":
                self.contact_manager.clear_override(group)
            else:
                self.contact_manager.set_override(group, state)
        except KeyError:
            raise _fail(
                f"unknown contact group {group!r}",
                f"declared groups: " f"{[s.name for s in self.contact_manager.specs]}",
            )
        except ValueError as e:
            raise _fail(str(e), f"overridable (format: contact) groups: " f"{available or []}")
        return {"group": group, "state": state}

    def _op_attach(self, p: dict) -> dict:
        """attach = the follow alias (kinematic is the default;
        ``--mode weld`` is the capability-gated physical mode)."""
        obj = self._need(p, "obj")
        host = self._need(p, "host")
        mode = p.get("mode") or self._ops.default_attach_mode()
        if mode in KIN_MODES:
            mode = "kin"
            return self._follow(obj, host)
        if mode != "weld":
            raise _fail(
                f"attach mode must be kin|weld, got {mode!r}",
                "kin is the default; weld is the capability-gated " "physical mode",
            )
        if not self._ops.has_weld():
            raise _fail(
                "attach mode 'weld' is unavailable: this backend does not "
                "declare the DIRECTOR_ATTACH_WELD capability",
                "use mode: kin (the universal attach face)",
            )
        self._resolve_obj(obj)
        weld_host = self._ops.resolve_weld_host(host)  # opaque handle
        if obj in self._attachments:
            self._detach(obj)
        self._attachments[obj] = self._ops.attach_weld(obj, weld_host)
        return {"attached": obj, "host": host, "mode": mode}

    def _op_detach(self, p: dict) -> dict:
        obj = self._need(p, "obj")
        self._resolve_obj(obj)
        was = obj in self._attachments
        self._detach(obj)
        return {"detached": obj, "was": was}

    def _op_pin(self, p: dict) -> dict:
        """pin: freeze one prop at its CURRENT pose (ghost rigid body)."""
        name = self._need(p, "name")
        self._resolve_obj(name)
        pose = self._ops.obj_pose(name)
        if name in self._attachments:
            self._detach(name)
        self._attachments[name] = {"host": None, "mode": "kin", "kind": "pin", "pose": pose}
        self._ops.halt(name)
        return {"pinned": name}

    def _op_follow(self, p: dict) -> dict:
        obj = self._need(p, "obj")
        host = self._need(p, "host")
        return self._follow(obj, host)

    def _follow(self, obj: str, host: str) -> dict:
        """The kinematic follow: lock the host-relative offset NOW, then
        per-tick obj = host x rel."""
        if strip_prop_prefix(str(obj)) == strip_prop_prefix(str(host)):
            raise _fail(
                f"follow host == obj ({obj!r})",
                "follow carries an object WITH a host; pin it " "instead",
            )
        self._resolve_obj(obj)
        opos, oquat = self._ops.obj_pose(obj)
        hpos, hquat = self._host_pose(host)  # resolves + fail-fast
        rel = pose_rel(opos, oquat, hpos, hquat)
        if obj in self._attachments:
            self._detach(obj)
        self._attachments[obj] = {"host": host, "mode": "kin", "kind": "follow", "rel": rel}
        self._ops.halt(obj)
        return {"following": obj, "host": host, "mode": "kin"}

    def _op_release(self, p: dict) -> dict:
        name = p.get("name") or p.get("obj")
        if name is None:
            raise _fail(
                "release needs 'name' (or 'obj')", "unigym direct release <run> --name PROP"
            )
        self._resolve_obj(name)
        was = name in self._attachments
        self._detach(name)
        return {"released": name, "was": was}

    def _op_push(self, p: dict) -> dict:
        """push: robot root velocity write (set|add)."""
        name = p.get("name") or "robot"
        if name != "robot":
            raise _fail(
                f"push targets the robot root (name='robot'), " f"got {name!r}", "props use setvel"
            )
        lin = self._opt_vec(p, "lin", 3)
        ang = self._opt_vec(p, "ang", 3)
        if lin is None and ang is None:
            raise _fail(
                "push needs --lin and/or --ang",
                "unigym direct push <run> --lin X,Y,Z " "[--ang X,Y,Z] [--mode set|add]",
            )
        mode = p.get("mode") or "add"
        if mode not in ("set", "add"):
            raise _fail(
                f"push mode must be set|add, got {mode!r}",
                "add = impulse on top of the current velocity "
                "(default); set = overwrite (the DR setvel "
                "semantics)",
            )
        return self._ops.push_root(lin, ang, mode)

    def _op_joints(self, p: dict) -> dict:
        """joints: robot joint-angle write (contract §3.11 -- the pose
        teleport: PAUSED = final pose; RUNNING = the position controller
        pulls back toward the action targets, honestly)."""
        joints = p.get("joints")
        if not isinstance(joints, dict) or not joints:
            raise _fail(
                "joints needs a non-empty {joint: rad} mapping",
                "unigym direct joints <run> --joint name=0.6 " "[--joint ...] [--zero-vel]",
            )
        bad = {
            k: v
            for k, v in joints.items()
            if not isinstance(v, (int, float)) or isinstance(v, bool)
        }
        if bad:
            raise _fail(
                f"joint values must be numbers, got {bad!r}",
                "angles are radians (scene face " "state.robot.joints)",
            )
        return self._ops.write_joints(
            {str(k): float(v) for k, v in joints.items()}, bool(p.get("zero_vel", False))
        )

    def _op_gripper(self, p: dict) -> dict:
        action = self._need(p, "action")
        if self._gripper is None:
            raise _fail(
                "no gripper on this run (declare director.gripper)",
                "yaml director.gripper section (e.g. director: "
                "{gripper: {pos: [...]}}) + start with --direct",
            )
        if action == "where":
            pos, quat = self._gripper.get_pose(self.env)
            return {
                "gripper": self._gripper.name,
                "pos": [float(v) for v in pos],
                "quat": [float(v) for v in quat],
            }
        if action == "move":
            pose = p.get("pose") or {}
            if not isinstance(pose, dict):
                raise _fail("gripper move 'pose' must be {pos?, quat?}")
            rel = bool(p.get("rel", False))
            pos = self._opt_vec(pose, "pos", 3)
            quat = self._opt_vec(pose, "quat", 4)
            if pos is not None and rel:
                cur, _ = self._gripper.get_pose(self.env)
                pos = [float(cur[i]) + pos[i] for i in range(3)]
            if pos is None and quat is None:
                raise _fail("gripper move needs pose.pos and/or pose.quat")
            self._gripper.set_pose(self.env, pos=pos, quat=quat)
            self._ops.after_gripper_move()
            pos, quat = self._gripper.get_pose(self.env)
            return {
                "gripper": self._gripper.name,
                "pos": [float(v) for v in pos],
                "quat": [float(v) for v in quat],
            }
        if action == "grasp":
            obj = self._need(p, "obj")
            mode = p.get("mode") or self._ops.default_attach_mode()
            return self._op_attach({"obj": obj, "host": self._gripper.name, "mode": mode})
        if action == "release":
            held = [o for o, a in self._attachments.items() if a["host"] == self._gripper.name]
            if not held:
                return {"released": None, "held": []}
            for o in held:
                self._detach(o)
            return {"released": held, "held": []}
        raise _fail(f"unknown gripper action {action!r}", "actions: move | grasp | release | where")

    def _op_status(self, p: dict) -> dict:
        return self.status_doc()

    # ------------------------------------------------------------------ #
    # shared resolution / following helpers (backend mechanics in _ops)
    # ------------------------------------------------------------------ #
    def _prop_names(self) -> list[str]:
        return self._ops.prop_names()

    def _resolve_obj(self, name: str):
        return self._ops.resolve_obj(name)

    def _host_pose(self, host: str) -> tuple[np.ndarray, np.ndarray]:
        return self._ops.host_pose(host)

    def _write_slave_pose(self, obj, pos, quat) -> None:
        self._ops.write_slave_pose(obj, pos, quat)

    def _detach(self, obj) -> None:
        att = self._attachments.pop(obj, None)
        if att is not None:
            self._ops.detach(att)

    def _need(self, p: dict, key: str):
        val = p.get(key)
        if val is None:
            raise _fail(
                f"op {p.get('op')!r} needs {key!r}", "see docs/contracts/director_contract.md §3"
            )
        return val

    def _need_vec(self, p: dict, key: str, n: int) -> list[float]:
        val = self._need(p, key)
        return self._as_vec(key, val, n)

    def _opt_vec(self, p: dict, key: str, n: int) -> list[float] | None:
        val = p.get(key)
        if val is None:
            return None
        return self._as_vec(key, val, n)

    @staticmethod
    def _as_vec(key: str, val, n: int) -> list[float]:
        if (
            not isinstance(val, (list, tuple))
            or len(val) != n
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in val)
        ):
            raise _fail(f"{key} must be a {n}-vector of numbers, got {val!r}")
        return [float(v) for v in val]

    # ------------------------------------------------------------------ #
    # status face
    # ------------------------------------------------------------------ #
    def status_doc(self) -> dict:
        att = {
            o: {
                "host": a.get("host"),
                "mode": a["mode"],
                **({"kind": a["kind"]} if a["mode"] == "kin" else {}),
            }
            for o, a in self._attachments.items()
        }
        doc = {
            "enabled": self.enabled,
            "backend": self.backend,
            "commands": self._n_commands,
            "directed": sorted(self._directed),
            "log": str(self.log_path) if self.log_path else None,
            "attachments": att,
            # scene-card face: the pinned/following ledger
            "pinned": [
                o for o, a in self._attachments.items() if a["mode"] == "kin" and a["kind"] == "pin"
            ],
            "following": {
                o: a["host"]
                for o, a in self._attachments.items()
                if a["mode"] == "kin" and a["kind"] == "follow"
            },
        }
        if self._gripper is not None:
            pos, quat = self._gripper.get_pose(self.env)
            doc["gripper"] = {
                "name": self._gripper.name,
                "pos": [float(v) for v in pos],
                "quat": [float(v) for v in quat],
            }
        else:
            doc["gripper"] = None
        return doc


# --------------------------------------------------------------------------- #
# bridge wiring (collect_cli face)
# --------------------------------------------------------------------------- #
def attach(bridge, director: Director) -> None:
    """Register the Director as the bridge cmd-14 handler.

    (The Director is additionally mounted as a SINK by collect_cli --
    ``FanOutSink(base, director)`` before the bridge -- so per-tick
    kinematic follower application and episode-end directed flags run on
    the engine thread.)
    """
    bridge.register_command("direct", director.handle)
