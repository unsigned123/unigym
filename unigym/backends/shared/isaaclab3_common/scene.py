"""IsaacLab 3 scene mechanics for the ``scene`` typed query.

The ``IsaacLabSceneOps`` half of the SceneOps seam
(``unigym/runtime/scene.py``): curated tree + live state over the
IsaacLab 3 env's existing objects (``robot`` articulation data, prop
handler assets, contact manager) -- all READ-ONLY (the zero-engine
boundary: data reads and ``.cpu()`` copies only, no writes, no ``sim``
pokes).  isaaclab/torch stay lazy function-body imports.

IsaacLab 3 data views are ``ProxyArray`` objects, so every tensor read goes
through the explicit ``.torch`` payload; root/body quaternions are reordered
``xyzw -> wxyz`` at this boundary (the repo-internal convention is ``wxyz``).

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

from typing import TYPE_CHECKING

from unigym.props.base import _prop_shape
from unigym.runtime.scene import require_robot_name

if TYPE_CHECKING:
    import torch

# PhysX/USD bucket unbounded joints with huge finite values; anything
# past this magnitude faces as "not limited" (contract §1.23 -- no
# ±1e9-class "range" is ever served as a real limit).
_UNBOUNDED = 1e8


def _torch_view(value) -> "torch.Tensor":
    """Explicit ``.torch`` payload of an IsaacLab 3 ``ProxyArray`` (else *value*)."""
    import torch

    payload = getattr(value, "torch", None)
    if payload is None:
        payload = value
    assert isinstance(payload, torch.Tensor)
    return payload


def _asset_joint_names(env):
    """The policy controller's full asset joint names (no primary binding)."""
    from unigym.runtime.controllers import single_articulation_runtime

    return list(single_articulation_runtime(env).joint_names)


def _profile_joint_sets(env):
    """The articulation profile's named joint sets (empty without a profile)."""
    from unigym.runtime.controllers import single_articulation_runtime

    profile = single_articulation_runtime(env).profile
    return dict(getattr(profile, "joint_sets", None) or {})


def _runtime_base_body(env, robot_name: str) -> str:
    """The articulation's runtime ``base_body`` binding (backend-native name).

    The scene tree never infers the base from ``body_names[0]``: the run
    layer's effective profile binding is the authority.
    """
    runtime = (getattr(env, "articulations", None) or {}).get(robot_name)
    base = str(getattr(runtime, "base_body", "") or "")
    if base:
        return base
    robot = getattr(env, "robot", None)
    bodies = [str(name) for name in getattr(getattr(robot, "data", None), "body_names", [])]
    raise ValueError(
        f"scene tree: articulation {robot_name!r} has no runtime base_body "
        f"binding; declared body names: {bodies}"
    )


def sim_joint_limits(data, joint_names) -> dict[str, tuple[float, float]] | None:
    """An articulation's sim-provided joint limits by name.

    Static: parsed from the USD/PhysX schema at articulation init.
    Honest source chain (contract §1.23 -- nothing guessed):

    1. ``joint_limits_sim`` -- the Isaac Lab 3.x API name;
    2. ``joint_pos_limits`` -- the older 2.3.2 face ("joint position
       limits provided to the simulation", ``(num_envs, num_joints, 2)``,
       filled from the physics view's DOF limits at init);

    else ``None`` (callers keep the honest null face -- every joint
    ``limited: False, range: null``).  Both attrs are class-annotated
    (default ``None``), so the test is a VALUE test.
    """
    if data is None:
        return None
    for attr in ("joint_limits_sim", "joint_pos_limits"):
        tens = getattr(data, attr, None)
        if tens is None:
            continue
        try:
            import torch

            t = torch.as_tensor(_torch_view(tens)).detach().cpu().reshape(-1, 2)
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
    """Curated scene face over the IsaacLab 3 env (N=1).

    ``backend`` stays empty in the shared library; the backend identity
    subclass sets it to ``"isaaclab3_physx"`` / ``"isaaclab3_newton"``.
    """

    backend = ""

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
        base = _runtime_base_body(env, self.robot_name)
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
        (the props/base.py tier: "the IsaacLab side keeps the asset
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
        backend = getattr(cm, "backend", None)
        inventory = getattr(backend, "inventory", None)
        out: dict = {}
        for spec in cm.specs:
            if inventory is not None:
                members = [row.native for ref in spec.sensor_refs for row in inventory.resolve(ref)]
                members += [
                    row.native for ref in spec.filter_refs for row in inventory.resolve(ref)
                ]
            else:
                members = [ref.label() for ref in (*spec.sensor_refs, *spec.filter_refs)]
            out[spec.name] = {"members": members}
        return out

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

        from unigym.core.quat import to_wxyz

        def f(v):
            return [
                float(x)
                for x in torch.as_tensor(_torch_view(v), device=device).reshape(-1).tolist()
            ]

        joints = {
            name: {"pos": float(p), "vel": float(v)}
            for name, p, v in zip(
                _asset_joint_names(env),
                _torch_view(robot.data.joint_pos)[0].detach().cpu(),
                _torch_view(robot.data.joint_vel)[0].detach().cpu(),
            )
        }
        return {
            "robot": {
                "base_pos_w": f(_torch_view(robot.data.root_pos_w)[0]),
                # il3 data is xyzw; the schema is repo-internal wxyz.
                "base_quat_w": f(to_wxyz(_torch_view(robot.data.root_quat_w)[0])),
                "joints": joints,
            },
            "props": self._props_state(torch),
            "contacts": self._contacts_state(),
        }

    def _props_state(self, torch) -> dict:
        from unigym.core.quat import to_wxyz

        out: dict = {}

        def f(v):
            t = torch.as_tensor(_torch_view(v)).reshape(-1).detach().cpu()
            return [float(x) for x in t.tolist()]

        for h in self._handlers():
            asset = getattr(h, "_asset", None)
            if asset is None or not hasattr(asset, "data"):
                continue
            entry: dict = {
                "pos": f(_torch_view(asset.data.root_pos_w)[0]),
                "quat": f(to_wxyz(_torch_view(asset.data.root_quat_w)[0])),
            }
            jnames = list(getattr(asset, "joint_names", []) or [])
            if jnames:
                jp = _torch_view(asset.data.joint_pos)[0].detach().cpu().tolist()
                jv = _torch_view(asset.data.joint_vel)[0].detach().cpu().tolist()
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
                "base": _runtime_base_body(self.env, self.robot_name),
                "links": [str(n) for n in robot.data.body_names],
                "joints": [str(n) for n in _asset_joint_names(self.env)],
            },
            "props": {},
            "note": "IsaacLab 3 face is a data-level inventory (no USD "
            "stage prim walk -- payload unboundable; "
            "daemon_api_contract §1.23); live poses in state",
        }
        for h in self._handlers():
            asset = getattr(h, "_asset", None)
            links: list[str] = []
            if asset is not None:
                raw = getattr(getattr(asset, "data", None), "body_names", None)
                if raw is not None:
                    links = [str(n) for n in raw]
            doc["props"][h.spec.name] = {
                "type": h.spec.type,
                "links": links,
                "joints": [str(n) for n in getattr(asset, "joint_names", []) or []],
            }
        return doc
