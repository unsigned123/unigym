"""IsaacLab scene mechanics for the ``scene`` typed query.

The ``IsaacLabSceneOps`` half of the SceneOps seam
(``unigym/runtime/scene.py``): curated tree + live state over the
IsaaclabEnv's existing objects (``robot`` articulation data, prop
handler assets, contact manager) -- all READ-ONLY (the zero-engine
boundary: data reads and ``.cpu()`` copies only, no writes, no ``sim``
pokes).  isaaclab/torch stay lazy function-body imports.

Curated depth and field semantics mirror the mujoco face
(daemon_api_contract §1.23 -- the parity table lives there): same
schema, honest differences declared.  The declared ones:

* ``tree.robot.joints`` / prop ``joints`` limits -- sourced
  from the articulation's sim-provided limits (``joint_limits_sim``
  3.x name, then 2.3.2's ``joint_pos_limits``; contract §1.23 chain);
  sourceless joints face honest nulls;
* ``tree.contacts.members`` -- resolved sensor prim body names plus the
  DECLARED filter expressions (prim-path form; the mujoco face resolves
  the filter to concrete MJ body names);
* ``scene.full`` (?full=1) -- a DATA-LEVEL inventory (robot/prop links
  and joints) instead of a USD stage prim walk: the prim recursion
  payload is not boundable, the UI never consumes this face, and the
  escape hatch still has to stay honest (contract §1.23).
"""

from __future__ import annotations

from unigym.props.base import _prop_shape
from unigym.runtime.scene import require_robot_name

# PhysX/USD bucket unbounded joints with huge finite values; anything
# past this magnitude faces as "not limited" (contract §1.23 -- no
# ±1e9-class "range" is ever served as a real limit).
_UNBOUNDED = 1e8


def _asset_joint_names(env):
    """The policy controller's full asset joint names (no primary binding)."""
    from unigym.runtime.controllers import single_articulation_runtime

    return list(single_articulation_runtime(env).joint_names)


def _profile_joint_sets(env):
    """The articulation profile's named joint sets (empty without a profile)."""
    from unigym.runtime.controllers import single_articulation_runtime

    profile = single_articulation_runtime(env).profile
    return dict(getattr(profile, "joint_sets", None) or {})


def sim_joint_limits(data, joint_names) -> dict[str, tuple[float, float]] | None:
    """An articulation's sim-provided joint limits by name.

    Static: parsed from the USD/PhysX schema at articulation init.
    Honest source chain (contract §1.23 -- nothing guessed):

    1. ``joint_limits_sim`` -- the Isaac Lab 3.x API name;
    2. ``joint_pos_limits`` -- the same 2.3.2 face ("joint position
       limits provided to the simulation", (num_envs, num_joints, 2),
       filled from ``root_physx_view.get_dof_limits()`` at init);

    else ``None`` (callers keep the honest null face -- every joint
    ``limited: False, range: null``).  Both attrs are class-annotated
    (default ``None``) in 2.3.2, so the test is a VALUE test.
    """
    if data is None:
        return None
    for attr in ("joint_limits_sim", "joint_pos_limits"):
        tens = getattr(data, attr, None)
        if tens is None:
            continue
        try:
            import torch

            t = torch.as_tensor(tens).detach().cpu().reshape(-1, 2)
            names = [str(n) for n in joint_names]
            if t.shape[0] != len(names):
                return None  # face mismatch: honest null
            out: dict = {}
            for n, lo, hi in zip(names, t[:, 0].tolist(), t[:, 1].tolist()):
                out[n] = None if (abs(lo) > _UNBOUNDED or abs(hi) > _UNBOUNDED) else (lo, hi)
            return out
        except (TypeError, ValueError, RuntimeError):
            return None
    return None


class IsaacLabSceneOps:
    """Curated scene face over the isaaclab IsaaclabEnv (N=1)."""

    backend = "isaaclab"

    def __init__(self, env, robot_name: str, camera_names=None):
        self.env = env
        self.robot_name = require_robot_name(robot_name)
        self.camera_names = list(camera_names or [])

    # ------------------------------------------------------------------ #
    # curated tree (lazy-cached by the face)
    # ------------------------------------------------------------------ #
    def tree(self) -> dict:
        """Curated read-only scene tree per daemon_api_contract §1.23.

        Lazy-cached by the face; limit rows over all actuated joints,
        prop joint limits via ``sim_joint_limits``.
        """
        from unigym.runtime.scene import joint_limit_rows, robot_joint_groups

        env = self.env
        robot = env.robot
        base = str(robot.data.body_names[0])  # root link
        joints = list(_asset_joint_names(env))
        # limit rows over ALL actuated joints
        # (the policy controller's asset_joint_names == the articulation
        # table, hands included)
        # -- the sim-provided limits cover the same full (N, J, 2) face;
        # a degraded/sourceless face keeps the honest nulls (contract
        # §1.23 chain).
        limits = sim_joint_limits(robot.data, [str(n) for n in robot.joint_names]) or {}
        joint_sets = _profile_joint_sets(env)
        return {
            "robot": {
                "name": self.robot_name,
                "base": base,
                "n_joints": len(joints),
                "joint_groups": robot_joint_groups(joints, joint_sets),
                "joints": joint_limit_rows(joints, {n: limits.get(n) for n in joints}, joint_sets),
            },
            "props": self._props_tree(),
            "contacts": self._contacts_tree(),
            "cameras": list(self.camera_names),
        }

    def _handlers(self) -> list:
        pm = getattr(self.env, "prop_manager", None)
        return list(getattr(pm, "_handlers", []) or []) if pm else []

    def _props_tree(self) -> dict:
        out: dict = {}
        for h in self._handlers():
            spec = h.spec
            jnames = sorted(spec.joints) if getattr(spec, "joints", None) else []
            out[spec.name] = {
                "type": spec.type,
                "joints": self._prop_joint_rows(h, jnames),
                "shape": _prop_shape(spec),
            }
        return out

    @staticmethod
    def _prop_joint_rows(handler, jnames) -> list[dict]:
        """Articulation prop joints as limit rows.

        The asset's sim-provided face first, then the PropSpec declared
        ``range``
        (the props/base.py tier: "the isaaclab side keeps the asset
        limits"), else the honest null row.
        """
        if not jnames:
            return []
        spec = handler.spec
        asset = getattr(handler, "_asset", None)
        limits = (
            sim_joint_limits(
                getattr(asset, "data", None),
                [str(n) for n in getattr(asset, "joint_names", []) or []],
            )
            or {}
        )
        rows: list[dict] = []
        for jn in jnames:
            lo_hi = limits.get(jn)
            if lo_hi is None:
                decl = (spec.joints.get(jn) or {}).get("range")
                lo_hi = (float(decl[0]), float(decl[1])) if decl and len(decl) == 2 else None
            rows.append(
                {
                    "name": jn,
                    "range": None if lo_hi is None else [lo_hi[0], lo_hi[1]],
                    "limited": lo_hi is not None,
                }
            )
        return rows

    def _contacts_tree(self) -> dict:
        cm = getattr(self.env, "contact_manager", None)
        if cm is None:
            return {}
        out: dict = {}
        resolved = self._resolved_sensor_names(cm)
        for spec in cm.specs:
            sensors = resolved.get(spec.name)
            members = list(sensors) if sensors else list(spec.sensor_bodies)
            members += list(spec.filter_bodies or [])
            out[spec.name] = {"members": members}
        return out

    def _resolved_sensor_names(self, cm) -> dict[str, list[str]]:
        """Resolved sensor body names per group (post bind_env).

        Read from the contact backend's own tables; unbound -> {}
        (callers fall back to the declared entries).
        """
        backend = getattr(cm, "backend", None)
        sensors = getattr(backend, "_sensors", None) or {}
        perms = getattr(backend, "_perm", None) or {}
        robot = getattr(self.env, "robot", None)
        if not (sensors and perms and robot is not None):
            return {}
        body_names = [str(n) for n in robot.data.body_names]
        out: dict[str, list[str]] = {}
        for name, pieces in sensors.items():
            names: list[str] = []
            for piece, perm in zip(pieces["sensors"], perms.get(name, [])):
                for idx in perm:
                    try:
                        names.append(piece.body_names[idx])
                    except (IndexError, TypeError):
                        pass
            # sensor body names are robot-qualified frames; keep the
            # robot-relative tail (the curated face names robot bodies
            # the way the declarations do)
            out[name] = [self._robot_relative(n, body_names) for n in names]
        return out

    @staticmethod
    def _robot_relative(name: str, body_names: list[str]) -> str:
        for b in body_names:
            if str(name).endswith(b):
                return b
        return str(name)

    # ------------------------------------------------------------------ #
    # live state (fresh per query; data reads only)
    # ------------------------------------------------------------------ #
    def state(self) -> dict:
        """Live scene state -- fresh per query, data reads only.

        Robot base pose (world; m / wxyz) + per-joint pos/vel (rad,
        rad/s), prop poses/joint state, contact-group state (contract
        §1.23 live-state schema).
        """
        env = self.env
        robot = env.robot
        device = env.device
        import torch

        def f(v):
            return [float(x) for x in torch.as_tensor(v, device=device).reshape(-1).tolist()]

        joints = {
            name: {"pos": float(p), "vel": float(v)}
            for name, p, v in zip(
                _asset_joint_names(env),
                robot.data.joint_pos[0].detach().cpu(),
                robot.data.joint_vel[0].detach().cpu(),
            )
        }
        return {
            "robot": {
                "base_pos_w": f(robot.data.root_pos_w[0]),
                "base_quat_w": f(robot.data.root_quat_w[0]),
                "joints": joints,
            },
            "props": self._props_state(torch),
            "contacts": self._contacts_state(),
        }

    def _props_state(self, torch) -> dict:
        out: dict = {}

        def f(v):
            return [float(x) for x in torch.as_tensor(v).reshape(-1).detach().cpu().tolist()]

        for h in self._handlers():
            asset = getattr(h, "_asset", None)
            if asset is None or not hasattr(asset, "data"):
                continue
            entry: dict = {"pos": f(asset.data.root_pos_w[0]), "quat": f(asset.data.root_quat_w[0])}
            jnames = list(getattr(asset, "joint_names", []) or [])
            if jnames:
                jp = asset.data.joint_pos[0].detach().cpu().tolist()
                jv = asset.data.joint_vel[0].detach().cpu().tolist()
                entry["joints"] = {
                    n: {"pos": float(p), "vel": float(v)} for n, p, v in zip(jnames, jp, jv)
                }
            out[h.spec.name] = entry
        return out

    def _contacts_state(self) -> dict:
        from unigym.runtime.scene import contact_groups_state

        return contact_groups_state(getattr(self.env, "contact_manager", None))

    # ------------------------------------------------------------------ #
    # ?full=1 raw face: data-level inventory (contract §1.23 declared diff)
    # ------------------------------------------------------------------ #
    def full_tree(self) -> dict:
        """Data-level inventory for the ``?full=1`` face.

        Robot/prop links and joints (the USD stage prim walk is not
        boundable here -- daemon_api_contract §1.23 declared diff); live
        poses live in :meth:`state`.
        """
        robot = self.env.robot
        doc: dict = {
            "robot": {
                "base": str(robot.data.body_names[0]),
                "links": [str(n) for n in robot.data.body_names],
                "joints": [str(n) for n in _asset_joint_names(self.env)],
            },
            "props": {},
            "note": "isaaclab face is a data-level inventory (no USD "
            "stage prim walk -- payload unboundable; "
            "daemon_api_contract §1.23); live poses in state",
        }
        for h in self._handlers():
            asset = getattr(h, "_asset", None)
            links: list[str] = []
            if asset is not None:
                if hasattr(asset, "body_names"):
                    links = [str(n) for n in asset.body_names]
                elif hasattr(getattr(asset, "data", None), "body_names"):
                    links = [str(n) for n in asset.data.body_names]
            doc["props"][h.spec.name] = {
                "type": h.spec.type,
                "links": links,
                "joints": [str(n) for n in getattr(asset, "joint_names", []) or []],
            }
        return doc
