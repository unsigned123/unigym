"""Mujoco scene mechanics for the ``scene`` typed query.

The ``MujocoSceneOps`` half of the SceneOps seam
(``unigym/runtime/scene.py``): curated tree + live state + raw full
tree, all READ-ONLY over the existing ``mj_model``/``mj_data`` (the
zero-engine boundary -- this module never writes to the model, the
data, or any manager).  mujoco stays a lazy function-body import.

Curated depth (design v045-scene-face §1): robot (base + joints folded
into G1 groups) / props (type + declared joints + shape family) /
contact groups (resolved member bodies) / cameras (the bridge's
instantiated rig names).  Addressing reuses the existing accessors --
``prop_backend.free_addresses`` / ``articulations`` and the contact
backend's resolved body ids -- the same tables director uses, so the
scene vocabulary and the director vocabulary can never drift.
"""
from __future__ import annotations

from unigym.backends.shared.mjfamily.model.joint_limits import (  # noqa: F401
    _jnt_limit,
    model_joint_limits,
)
from unigym.props.base import _prop_shape

class MujocoSceneOps:
    """Curated scene face over ``MjModel``/``MjData`` (mujoco, N=1)."""

    backend = "mujoco"

    def __init__(self, env, robot_name: str | None = None,
                 camera_names=None):
        self.env = env
        self.robot_name = robot_name or "g1"
        self.camera_names = list(camera_names or [])
        # static per-run joint addressing (model-level, never changes)
        self._jadr: dict[str, tuple[int, int]] | None = None

    # ------------------------------------------------------------------ #
    # curated tree (lazy-cached by the face; this method itself is pure)
    # ------------------------------------------------------------------ #
    def tree(self) -> dict:
        """Curated read-only scene tree per daemon_api_contract §1.23.

        Lazy-cached by the face; robot joints = the policy controller's
        ``asset_joint_names`` minus the ``Prop_*`` injected freejoints; joint
        limit rows over all
        actuated joints.
        """
        import mujoco

        from unigym.runtime.scene import joint_limit_rows, robot_joint_groups

        env, m = self.env, self.env.mj_model
        base_bid = int(env._base_body_id)
        base = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, base_bid)
        joints = self._robot_joints()
        backend_joints = self._backend_joints()
        self._jadr = {
            j: (int(m.jnt_qposadr[m.joint(b).id]),
                int(m.jnt_dofadr[m.joint(b).id]))
            for j, b in zip(joints, backend_joints)}
        # Limit rows over ALL actuated robot
        # joints -- the model is the source (``jnt_range`` by name; the
        # 29-joint policy table stays an action-face concern), computed
        # in the same lazy batch as the joint addressing (model-static,
        # cached by the face).
        backend_limits = model_joint_limits(m, backend_joints)
        limits = {j: backend_limits[b]
                  for j, b in zip(joints, backend_joints)}
        tree: dict = {
            "robot": {"name": self.robot_name, "base": base,
                      "n_joints": len(joints),
                      "joint_groups": robot_joint_groups(joints),
                      "joints": joint_limit_rows(joints, limits)},
            "props": self._props_tree(),
            "contacts": self._contacts_tree(),
            "cameras": list(self.camera_names),
        }
        return tree

    def _robot_joints(self) -> list[str]:
        """The ROBOT's neutral ``unigym_name`` joints (no ``Prop_*``)."""
        from unigym.runtime.controllers import single_articulation_runtime

        names = single_articulation_runtime(self.env).joint_names
        return [j for j in names if not j.startswith("Prop_")]

    def _backend_joints(self) -> list[str]:
        """The compiled backend names aligned with :meth:`_robot_joints`."""
        from unigym.runtime.controllers import single_articulation_runtime

        art = single_articulation_runtime(self.env)
        binding = (getattr(self.env, "_mj_articulation_bindings", {})
                   or {}).get(art.name)
        names = list(binding.backend_joint_names if binding is not None
                     else art.joint_names)
        return [j for j in names if not j.startswith("Prop_")]

    def _props_tree(self) -> dict:
        pb = getattr(self.env, "prop_backend", None)
        if pb is None:
            return {}
        specs = {s.name: s for s in pb.prop_manager.specs}
        m = self.env.mj_model
        out: dict = {}
        for name in list(pb.free_prop_names()) + sorted(pb.articulations()):
            spec = specs.get(name)
            art = pb.articulations().get(name)
            if art is not None:
                # object rows with the model limit per joint
                # (the internal MJ joint id is the articulation's own
                # resolution -- the serializer adds no new queries)
                joints = [{"name": jn,
                           "range": None if (lr := _jnt_limit(
                               m, art.joints[jn]["jid"])) is None
                           else [lr[0], lr[1]],
                           "limited": lr is not None}
                          for jn in sorted(art.joints)]
            else:
                joints = []
            out[name] = {
                "type": spec.type if spec is not None
                else ("articulation" if art is not None else "rigid_object"),
                "joints": joints,
                "shape": _prop_shape(spec) if spec is not None else None,
            }
        return out

    def _contacts_tree(self) -> dict:
        cm = getattr(self.env, "contact_manager", None)
        if cm is None:
            return {}
        backend = getattr(cm, "backend", None)
        resolved = getattr(backend, "_resolved", None) or {}
        out: dict = {}
        for spec in cm.specs:
            members = resolved.get(spec.name)
            if members is not None:
                import mujoco

                m = self.env.mj_model
                names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
                         for bid in members.get("sensor_ids", [])]
                names += [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
                          for bid in members.get("filter_ids", [])]
                out[spec.name] = {"members": [n for n in names if n]}
            else:
                # unbound (or non-mujoco backend object): the resolved
                # declarations are the honest best face
                out[spec.name] = {
                    "members": list(spec.sensor_bodies)
                    + list(spec.filter_bodies or [])}
        return out

    # ------------------------------------------------------------------ #
    # live state (fresh per query; reads mj_data only)
    # ------------------------------------------------------------------ #
    def state(self) -> dict:
        """Live scene state -- fresh per query, reads mj_data only.

        Robot base pose (world) + per-joint qpos/qvel (rad, rad/s),
        prop poses/joint state, contact-group state (daemon_api_contract
        §1.23 live-state schema).
        """
        import numpy as np

        env, d = self.env, self.env.mj_data
        base_bid = int(env._base_body_id)
        if self._jadr is None:             # tree not queried yet: state-only
            m = env.mj_model
            self._jadr = {
                j: (int(m.jnt_qposadr[m.joint(b).id]),
                    int(m.jnt_dofadr[m.joint(b).id]))
                for j, b in zip(self._robot_joints(),
                                self._backend_joints())}
        joints = {name: {"pos": float(d.qpos[qadr]),
                         "vel": float(d.qvel[dadr])}
                  for name, (qadr, dadr) in self._jadr.items()}
        return {
            "robot": {
                "base_pos_w": [float(v) for v in d.xpos[base_bid]],
                "base_quat_w": [float(v) for v in d.xquat[base_bid]],
                "joints": joints},
            "props": self._props_state(np),
            "contacts": self._contacts_state(),
        }

    def _props_state(self, np) -> dict:
        pb = getattr(self.env, "prop_backend", None)
        if pb is None:
            return {}
        d = self.env.mj_data
        out: dict = {}
        arts = pb.articulations()
        for name in pb.free_prop_names():
            _, qadr, _ = pb.free_addresses(name)
            out[name] = {"pos": [float(v) for v in d.qpos[qadr:qadr + 3]],
                         "quat": [float(v) for v in d.qpos[qadr + 3:qadr + 7]]}
        for name, art in arts.items():
            out[name] = {
                "pos": [float(v) for v in d.xpos[art.root_bid]],
                "quat": [float(v) for v in d.xquat[art.root_bid]],
                "joints": {jname: {"pos": float(d.qpos[j["qadr"]]),
                                   "vel": float(d.qvel[j["dadr"]])}
                           for jname, j in art.joints.items()},
            }
        return out

    def _contacts_state(self) -> dict:
        from unigym.runtime.scene import contact_groups_state

        return contact_groups_state(
            getattr(self.env, "contact_manager", None))

    # ------------------------------------------------------------------ #
    # ?full=1 raw face: the MJ body recursion (UI never consumes this)
    # ------------------------------------------------------------------ #
    def full_tree(self) -> dict:
        """Raw MJ body/joint recursion for the ``?full=1`` face.

        The UI never consumes this; poses are model-level (qpos0), live
        world poses live in :meth:`state`.
        """
        import mujoco

        m = self.env.mj_model
        bodies = []
        for bid in range(1, m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
            bodies.append({
                "name": name or f"body_{bid}",
                "parent": mujoco.mj_id2name(
                    m, mujoco.mjtObj.mjOBJ_BODY, int(m.body_parentid[bid]))
                or "world",
                "pos": [round(float(v), 6) for v in m.body_pos[bid]],
                "is_prop": name.startswith("Prop_") if name else False})
        joints = []
        for jid in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
            joints.append({"name": name or f"joint_{jid}",
                           "type": int(m.jnt_type[jid])})
        return {"bodies": bodies, "joints": joints,
                "note": "mujoco body recursion; poses are model-level "
                        "(qpos0), live world poses live in state"}
