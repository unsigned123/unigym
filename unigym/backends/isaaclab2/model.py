"""IsaacLab model provider: declared ``build`` and live ``describe``.

The two faces are split explicitly:

* :meth:`IsaacLabModelProvider.build` reads a scene/articulation ``cfg``
  BEFORE the sim exists and emits the declared IR (``compiled=False``).
  Only cfg-declared facts are emitted: joint names from
  ``init_state.joint_pos``, actuator groups/gains, prop names/prim paths,
  camera prim paths/resolutions.  Everything the sim owns (limits, mass,
  inertia, body poses, joint addresses) stays ``None``.
* :meth:`IsaacLabModelProvider.describe` reads a live ``Articulation``
  AFTER the sim exists and fills the compiled subset (``compiled=True``).
  IsaacLab has no MuJoCo address model, so ``qpos_adr``/``dof_adr`` and
  element ids remain ``None``; USD prim paths and the remaining declared
  extras live in :class:`~unigym.model.ir.UsdExt` (namespace ``usd``).

The module imports no isaaclab/pxr symbol at import time: cfg/data reads
are duck-typed and the optional USD lookup imports ``pxr``/``omni.usd``
inside the call.  The provider therefore runs in the base lane too.
"""
from __future__ import annotations

from collections.abc import Mapping

from unigym.model.ir import (
    Actuator,
    Body,
    Camera,
    Joint,
    ModelIR,
    UsdExt,
)
from unigym.model.table import Table

__all__ = ["IsaacLabModelProvider"]

_UNKNOWN_JOINT = "unknown"
_USD_JOINT_TYPES = {
    "PhysicsRevoluteJoint": "hinge",
    "PhysicsPrismaticJoint": "slide",
    "PhysicsSphericalJoint": "ball",
    "PhysicsFixedJoint": "fixed",
}


# ---------------------------------------------------------------------------
# small duck-typed readers (no torch/isaaclab import)
# ---------------------------------------------------------------------------
def _as_sequence(value):
    """Return *value* as a plain list, or ``None`` when it has no sequence."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _first_env(value):
    """Return env 0 of an ``(N, ·)`` tensor/list, else the value itself."""
    seq = _as_sequence(value)
    if not seq:
        return None
    first = seq[0]
    if isinstance(first, (list, tuple)):
        return first
    return seq


def _names(value):
    """Return *value* as a list of names (empty when ``None``)."""
    seq = _as_sequence(value)
    return [] if seq is None else [str(name) for name in seq]


def _vec(values, size, default):
    """Return a float tuple of *size* from *values*, else *default*."""
    seq = _as_sequence(values)
    if seq is None or len(seq) < size:
        return default
    return tuple(float(item) for item in seq[:size])


def _fixed_vec3(values, default: tuple[float, float, float]
                ) -> tuple[float, float, float]:
    """Return an exact 3-tuple from *values*, else *default* (validated)."""
    seq = _as_sequence(values)
    if seq is None or len(seq) < 3:
        return default
    return (float(seq[0]), float(seq[1]), float(seq[2]))


def _fixed_vec4(values, default: tuple[float, float, float, float]
                ) -> tuple[float, float, float, float]:
    """Return an exact 4-tuple from *values*, else *default* (validated)."""
    seq = _as_sequence(values)
    if seq is None or len(seq) < 4:
        return default
    return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))


def _prim_path(cfg):
    """Return a cfg's declared ``prim_path`` (``None`` when absent)."""
    path = getattr(cfg, "prim_path", None)
    return path if isinstance(path, str) and path else None


def _tail(path):
    """Return the last component of a prim path (``None`` for empty)."""
    if not path:
        return None
    return path.rstrip("/").split("/")[-1] or None


def _usd_path(cfg):
    """Return a cfg's declared USD asset path (``None`` when absent)."""
    spawn = getattr(cfg, "spawn", None)
    path = getattr(spawn, "usd_path", None)
    return path if isinstance(path, str) and path else None


def _declared_joint_names(cfg):
    """Return joint names declared by an articulation cfg's init state."""
    state = getattr(cfg, "init_state", None)
    joint_pos = getattr(state, "joint_pos", None)
    if isinstance(joint_pos, Mapping):
        return [str(name) for name in joint_pos]
    return _names(joint_pos)


def _named_values(value, prefix):
    """Normalize a scene-field mapping/list into ``{name: cfg}``."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(name): item for name, item in value.items()}
    out = {}
    for index, item in enumerate(value):
        name = getattr(item, "name", None) or f"{prefix}{index}"
        out[str(name)] = item
    return out


def _scene_parts(cfg):
    """Return ``(robot, props, cameras)`` from a scene cfg or mapping."""
    if isinstance(cfg, Mapping):
        return (
            cfg.get("robot"),
            _named_values(cfg.get("props"), "prop_"),
            _named_values(cfg.get("cameras"), "cam_"),
        )
    robot = getattr(cfg, "robot", None)
    props = getattr(cfg, "props", None)
    cameras = getattr(cfg, "cameras", None)
    try:
        fields = vars(cfg)
    except TypeError:
        fields = {}
    if props is None:
        props = {key[5:]: value for key, value in fields.items()
                 if key.startswith("prop_")}
    else:
        props = _named_values(props, "prop_")
    if cameras is None:
        cameras = {key[4:]: value for key, value in fields.items()
                   if key.startswith("cam_")}
    else:
        cameras = _named_values(cameras, "cam_")
    return robot, props, cameras


# ---------------------------------------------------------------------------
# declared (pre-sim) assembly
# ---------------------------------------------------------------------------
def _declared_joints(cfg, body_name):
    """Build the declared joint rows of one articulation cfg."""
    rows = []
    for name in _declared_joint_names(cfg):
        rows.append(Joint(
            name=name, type=_UNKNOWN_JOINT, body=body_name,
            path=(body_name, name),
            qualname=f"{body_name}/{name}"))
    return rows


def _declared_actuators(cfg, body_name, attrs):
    """Build the declared actuator rows (one per actuator group)."""
    rows = []
    groups = getattr(cfg, "actuators", None) or {}
    for name, group in groups.items():
        expr = _names(getattr(group, "joint_names_expr", None))
        rows.append(Actuator(
            name=str(name), kind="robot", owner=body_name,
            qualname=str(name)))
        attrs[f"actuator/{name}"] = {
            "joint_names_expr": expr,
            "stiffness": getattr(group, "stiffness", None),
            "damping": getattr(group, "damping", None),
            "effort_limit_sim": getattr(group, "effort_limit_sim", None),
        }
    return rows


def _declared_camera(name, cfg, body_name, attrs, prim_paths):
    """Build one declared camera row plus its USD extras."""
    path = _prim_path(cfg)
    offset = getattr(cfg, "offset", None)
    pos = _fixed_vec3(getattr(offset, "pos", None), (0.0, 0.0, 0.0))
    quat = _fixed_vec4(getattr(offset, "rot", None),
                       (1.0, 0.0, 0.0, 0.0))
    if path:
        prim_paths[str(name)] = path
    attrs[f"camera/{name}"] = {
        "prim_path": path,
        "width": getattr(cfg, "width", None),
        "height": getattr(cfg, "height", None),
    }
    return Camera(name=str(name), body=body_name, pos=pos, quat=quat)


def _camera_host(bodies, robot_body):
    """Return the declared body a camera is attached to."""
    if robot_body is not None:
        return robot_body
    return bodies[0].name if bodies else "world"


# ---------------------------------------------------------------------------
# live (post-sim) readers
# ---------------------------------------------------------------------------
def _mass_and_inertia(articulation, body_names):
    """Return per-body ``(masses, inertias)`` from the PhysX view if exposed."""
    view = getattr(articulation, "root_physx_view", None)
    if view is None:
        view = getattr(articulation, "_root_physx_view", None)
    count = len(body_names)
    masses = _physx_values(view, "get_masses", count)
    raw_inertias = _physx_rows(view, "get_inertias", count)
    inertias = [_inertia_diag(value) for value in raw_inertias]
    return masses, inertias


def _physx_rows(view, method, count):
    """Call one PhysX view getter and return *count* raw rows."""
    if view is None:
        return [None] * count
    getter = getattr(view, method, None)
    if getter is None:
        return [None] * count
    try:
        values = _first_env(getter())
    except (RuntimeError, TypeError, ValueError):
        return [None] * count
    if not isinstance(values, (list, tuple)) or len(values) != count:
        return [None] * count
    return list(values)


def _physx_values(view, method, count):
    """Call one PhysX view getter and return *count* finite floats."""
    return [_finite(item)
            for item in _physx_rows(view, method, count)]


def _finite(value):
    """Return a finite float, else ``None`` (never fake a value)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _inertia_diag(value):
    """Return a diagonal inertia triple from a 3/9/3x3 array, else ``None``."""
    seq = _as_sequence(value)
    if seq is None:
        return None
    if len(seq) == 3 and all(isinstance(item, (int, float)) for item in seq):
        return tuple(_finite(item) for item in seq)
    if len(seq) == 9:
        return (_finite(seq[0]), _finite(seq[4]), _finite(seq[8]))
    if len(seq) == 3 and all(isinstance(item, (list, tuple)) for item in seq):
        return tuple(_finite(seq[i][i]) for i in range(3))
    return None


def _row_at(values, index, size):
    """Return row *index* of a list-of-lists as a float tuple (or default)."""
    if not isinstance(values, (list, tuple)) or index >= len(values):
        return None
    return _vec(values[index], size, None)


def _resolve_root_prim(articulation):
    """Return the articulation asset's root prim path (env 0).

    IsaacLab's ``root_physx_view.prim_paths[0]`` names the ROOT LINK
    (``.../Robot/pelvis``), not the articulation asset prim, so the cfg's
    declared (env-regex) prim path is the authoritative source; the view
    path is only a fallback when no cfg is present.
    """
    cfg = getattr(articulation, "cfg", None)
    path = _prim_path(cfg) or _prim_path(articulation)
    if path:
        # InteractiveScene rewrites the cfg regex to ``/World/envs/env_.*``;
        # the raw ``{ENV_REGEX_NS}`` form appears before construction.
        return (path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
                .replace("/World/envs/env_.*", "/World/envs/env_0"))
    view = getattr(articulation, "root_physx_view", None)
    if view is None:
        view = getattr(articulation, "_root_physx_view", None)
    paths = getattr(view, "prim_paths", None)
    if paths:
        try:
            return str(paths[0])
        except (IndexError, TypeError):
            pass
    return None


def _usd_stage():
    """Return the live USD stage, or ``None`` outside a Kit process."""
    try:
        import omni.usd
    except Exception:  # pragma: no cover - base lane / no Kit
        return None
    context = omni.usd.get_context()
    return context.get_stage() if context is not None else None


def _relationship_target(prim, name):
    """Return the first target name of one USD relationship."""
    rel = prim.GetRelationship(name)
    if rel is None:
        return None
    targets = rel.GetTargets()
    return targets[0].name if targets else None


def _usd_facts(root_prim, joint_names, body_names):
    """Collect prim paths, joint types and body parents from the USD stage."""
    facts = {
        "prim_paths": {},
        "joint_paths": {},
        "joint_type": {},
        "joint_body": {},
        "parents": {},
    }
    stage = _usd_stage()
    if stage is None or not root_prim:
        return facts
    from pxr import Usd

    prim = stage.GetPrimAtPath(root_prim)
    if not prim.IsValid():
        return facts
    body_set = set(body_names)
    joint_set = set(joint_names)
    for node in Usd.PrimRange(prim):
        name = node.GetName()
        if name in body_set:
            facts["prim_paths"][name] = str(node.GetPath())
        if name not in joint_set:
            continue
        joint_type = _USD_JOINT_TYPES.get(str(node.GetTypeName()))
        if joint_type is None:
            continue
        facts["joint_paths"][name] = str(node.GetPath())
        facts["joint_type"][name] = joint_type
        parent = _relationship_target(node, "physics:body0")
        child = _relationship_target(node, "physics:body1")
        if child in body_set:
            facts["joint_body"][name] = child
            if parent in body_set:
                facts["parents"][child] = parent
    return facts


def _body_path(name, parents):
    """Return the root-to-body name path from the joint parent map."""
    chain = []
    seen = set()
    current = name
    while current and current not in seen:
        seen.add(current)
        chain.append(current)
        current = parents.get(current)
    return tuple(reversed(chain))


def _body_rows(body_names, data, masses, inertias, facts):
    """Build the compiled body rows (pose, inertia, USD path/parent)."""
    positions = _first_env(getattr(data, "body_pos_w", None))
    quats = _first_env(getattr(data, "body_quat_w", None))
    rows = []
    for index, name in enumerate(body_names):
        parent = facts["parents"].get(name)
        path = _body_path(name, facts["parents"])
        rows.append(Body(
            name=name, parent=parent, path=path,
            qualname="/".join(path), pos=_row_at(positions, index, 3)
            or (0.0, 0.0, 0.0),
            quat=_row_at(quats, index, 4) or (1.0, 0.0, 0.0, 0.0),
            mass=masses[index], inertia=inertias[index]))
    return rows


def _limit_pair(values, index):
    """Return one joint's ``(lo, hi)`` limit pair, or ``None``."""
    row = _row_at(values, index, 2)
    if row is None or row[0] > row[1]:
        return None
    return row


def _scalar_at(values, index):
    """Return one per-joint scalar from a first-env row, or ``None``."""
    if not isinstance(values, (list, tuple)) or index >= len(values):
        return None
    value = values[index]
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return _finite(value)


def _joint_rows(joint_names, data, root_body, facts, attrs):
    """Build the compiled joint rows and their USD extras."""
    limits = _first_env(getattr(data, "joint_pos_limits", None))
    vel_limits = _first_env(getattr(data, "joint_vel_limits", None))
    defaults = _first_env(getattr(data, "default_joint_pos", None))
    rows = []
    for index, name in enumerate(joint_names):
        joint_range = _limit_pair(limits, index)
        body = facts["joint_body"].get(name, root_body)
        path = _body_path(body, facts["parents"]) + (name,)
        rows.append(Joint(
            name=name, type=facts["joint_type"].get(name, _UNKNOWN_JOINT),
            body=body, range=joint_range, limited=joint_range is not None,
            path=path, qualname="/".join(path)))
        attrs[f"joint/{name}"] = {
            "vel_limit": _scalar_at(vel_limits, index),
            "default_joint_pos": _scalar_at(defaults, index),
        }
    return rows


def _live_actuator_rows(articulation, root_body, attrs):
    """Build actuator rows from the live articulation's actuator groups."""
    groups = getattr(articulation, "actuators", None)
    if not isinstance(groups, Mapping):
        groups = getattr(articulation, "_actuators", None)
    if not isinstance(groups, Mapping):
        return []
    rows = []
    for name, group in groups.items():
        rows.append(Actuator(
            name=str(name), kind="robot", owner=root_body,
            qualname=str(name)))
        attrs[f"actuator/{name}"] = {
            "joint_names": _names(getattr(group, "joint_names", None)),
            "class": type(group).__name__,
        }
    return rows


# ---------------------------------------------------------------------------
# provider
# ---------------------------------------------------------------------------
class IsaacLabModelProvider:
    """IsaacLab provider: declared cfg assembly + live articulation read."""

    backend = "isaaclab2"

    def build(self, cfg):
        """Assemble the declared model IR from a scene/robot cfg (pre-sim).

        Args:
            cfg: An IsaacLab scene cfg (``.robot`` plus ``prop_*``/
                ``cam_*`` fields) or a mapping with ``robot``/``props``/
                ``cameras`` keys.

        Returns:
            A validated ``ModelIR`` with ``compiled=False``; sim-owned
            values (limits, mass/inertia, poses, addresses) stay ``None``.
        """
        robot, props, cameras = _scene_parts(cfg)
        attrs = {}
        prim_paths = {}
        bodies = []
        joints = []
        robot_prim = _prim_path(robot)
        robot_body = None
        if robot is not None:
            robot_body = _tail(robot_prim) or "robot"
            bodies.append(Body(name=robot_body, path=(robot_body,),
                               qualname=robot_body))
            if robot_prim:
                prim_paths[robot_body] = robot_prim
            attrs["robot"] = {"prim_path": robot_prim,
                              "usd_path": _usd_path(robot)}
            joints.extend(_declared_joints(robot, robot_body))
        for name, prop in props.items():
            path = _prim_path(prop)
            bodies.append(Body(name=str(name), path=(str(name),),
                               qualname=str(name)))
            if path:
                prim_paths[str(name)] = path
            attrs[f"prop/{name}"] = {
                "prim_path": path, "usd_path": _usd_path(prop)}
            joints.extend(_declared_joints(prop, str(name)))
        host = _camera_host(bodies, robot_body)
        if cameras and not bodies:
            bodies.append(Body(name=host, path=(host,), qualname=host))
        camera_rows = [
            _declared_camera(name, camera, host, attrs, prim_paths)
            for name, camera in cameras.items()]
        actuators = (_declared_actuators(robot, robot_body, attrs)
                     if robot is not None else [])
        return ModelIR(
            name=_scene_name(cfg),
            backend=self.backend,
            compiled=False,
            bodies=Table(bodies),
            joints=Table(joints),
            cameras=Table(camera_rows),
            actuators=Table(actuators),
            ext={"usd": UsdExt(
                prim_paths=prim_paths, articulation=robot_prim,
                attrs=attrs)},
        ).validate()

    def describe(self, articulation):
        """Fill the compiled model IR from a live IsaacLab ``Articulation``.

        Args:
            articulation: The constructed ``Articulation`` (env 0 values
                are read; the IR describes the shared asset, not a clone).

        Returns:
            A validated ``ModelIR`` with ``compiled=True``.  IsaacLab-only
            facts (prim paths, joint types, velocity limits, defaults)
            live in ``ext["usd"]``; MuJoCo address fields stay ``None``.
        """
        data = getattr(articulation, "data", None)
        joint_names = _names(getattr(articulation, "joint_names", None))
        if not joint_names:
            joint_names = _names(getattr(data, "joint_names", None))
        body_names = _names(getattr(articulation, "body_names", None))
        if not body_names:
            body_names = _names(getattr(data, "body_names", None))
        root_body = body_names[0] if body_names else "root"
        root_prim = _resolve_root_prim(articulation)
        facts = _usd_facts(root_prim, joint_names, body_names)
        masses, inertias = _mass_and_inertia(articulation, body_names)
        attrs = {"robot": {"prim_path": root_prim,
                           "root_body": root_body}}
        bodies = _body_rows(
            body_names, data, masses, inertias, facts)
        joints = _joint_rows(joint_names, data, root_body, facts, attrs)
        actuators = _live_actuator_rows(articulation, root_body, attrs)
        prim_paths = dict(facts["prim_paths"])
        prim_paths.update(facts["joint_paths"])
        if root_prim:
            prim_paths.setdefault(root_body, root_prim)
        return ModelIR(
            name=_scene_name(articulation),
            backend=self.backend,
            compiled=True,
            bodies=Table(bodies),
            joints=Table(joints),
            actuators=Table(actuators),
            ext={"usd": UsdExt(
                prim_paths=prim_paths, articulation=root_prim,
                attrs=attrs)},
        ).validate()


def _scene_name(cfg):
    """Return a stable model name for a cfg/articulation object."""
    name = getattr(cfg, "name", None)
    if isinstance(name, str) and name:
        return name
    path = _prim_path(getattr(cfg, "cfg", None) or cfg)
    return _tail(path) or "isaaclab_model"
