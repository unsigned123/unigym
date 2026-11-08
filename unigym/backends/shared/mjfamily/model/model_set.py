"""Assemble a per-articulation ``ModelSet`` for the MuJoCo family (v103.4).

The compiled ``MjModel`` IR is the whole-scene ``merged`` view.  Each policy
articulation's IR is a ``ModelIR.subset`` filtered to that runtime's
compiled backend names.  v105.3 translates every published row back to the
neutral ``unigym_name`` through the backend-private :class:`MjNameMap`; the
internal raw IR stays backend-native.

A single-policy run uses the whole compiled IR for its one articulation (no
filtering needed); ``merged`` stays the same object.
"""
from __future__ import annotations

from dataclasses import replace

from unigym.model import MjExt, ModelIR, ModelSet, Table

__all__ = ["articulation_model_ir", "build_model_set", "translate_model_ir"]


def _backend_joint_names(name_map, runtime) -> tuple[str, ...]:
    names = name_map.backend_joint_names(runtime.name)
    if names:
        return names
    return tuple(runtime.robot_joint_names or runtime.joint_names)


def articulation_model_ir(merged, runtime, name_map=None) -> ModelIR:
    """Return one articulation runtime's IR as a filtered, translated view."""
    from ..name_map import MjNameMap

    name_map = name_map or MjNameMap.identity()
    joints = set(_backend_joint_names(name_map, runtime))
    prefix = name_map.prefix(runtime.name)
    base_body = name_map.backend_body(runtime.name, runtime.base_body)
    bodies = set()
    if prefix:
        bodies.update(
            body.name for body in merged.bodies
            if body.name.startswith(prefix))
    for joint in merged.joints:
        if joint.name in joints:
            bodies.add(joint.body)
    if base_body:
        bodies.add(base_body)
    body_names = set(bodies)
    geoms = {geom.name for geom in merged.geoms if geom.body in body_names}
    sites = {site.name for site in merged.sites if site.body in body_names}
    cameras = {
        camera.name for camera in merged.cameras
        if camera.body in body_names}
    actuators = {
        actuator.name for actuator in merged.actuators
        if actuator.target in joints
        or actuator.target in body_names
        or actuator.target in sites}
    subset = merged.subset(
        joints=joints, bodies=bodies, geoms=geoms, sites=sites,
        cameras=cameras, actuators=actuators)
    return translate_model_ir(subset, name_map)


def _translate_mj_ext(mj: MjExt, nm) -> MjExt:
    """Translate the backend-native ``mj`` extension tables to unigym names."""
    return replace(
        mj,
        geom_groups=Table(
            replace(row, geoms=tuple(nm(g) for g in row.geoms))
            for row in mj.geom_groups),
        eq_slots=Table(
            replace(row,
                    obj1=(nm(row.obj1) if row.obj1 else row.obj1),
                    obj2=(nm(row.obj2) if row.obj2 else row.obj2))
            for row in mj.eq_slots),
        mocap_bodies=Table(
            replace(row, name=nm(row.name),
                    parent=(nm(row.parent) if row.parent else row.parent))
            for row in mj.mocap_bodies),
        textures=Table(
            replace(row, name=nm(row.name)) for row in mj.textures),
        geom_contact=Table(
            replace(row, name=nm(row.name)) for row in mj.geom_contact),
    )


def translate_model_ir(ir: ModelIR, name_map) -> ModelIR:
    """Return *ir* with articulation-owned backend names translated.

    Non-articulation scene entities (world base, props) keep their compiled
    asset names: the v105 contract translates the *articulation* vocabulary,
    not arbitrary scene geometry.  Empty/identity maps are a no-op.
    """
    from ..name_map import MjNameMap

    if not isinstance(name_map, MjNameMap) or not any(name_map.prefixes):
        return ir

    def nm(name):
        return name_map.unigym_name(name) if name else name

    bodies = Table(
        replace(row, name=nm(row.name),
                parent=(nm(row.parent) if row.parent else row.parent),
                path=tuple(nm(part) for part in row.path))
        for row in ir.bodies)
    body_names = {row.name for row in ir.bodies}

    def body_ref(name):
        return nm(name) if name in body_names else name

    joints = Table(
        replace(row, name=nm(row.name), body=body_ref(row.body),
                actuator=(nm(row.actuator) if row.actuator else row.actuator),
                path=tuple(nm(part) for part in row.path))
        for row in ir.joints)
    geoms = Table(
        replace(row, name=nm(row.name), body=body_ref(row.body),
                path=tuple(nm(part) for part in row.path))
        for row in ir.geoms)
    sites = Table(
        replace(row, name=nm(row.name), body=body_ref(row.body),
                path=tuple(nm(part) for part in row.path))
        for row in ir.sites)
    cameras = Table(
        replace(row, name=nm(row.name), body=body_ref(row.body),
                targetbody=body_ref(row.targetbody)
                if row.targetbody else row.targetbody,
                path=tuple(nm(part) for part in row.path))
        for row in ir.cameras)
    actuators = Table(
        replace(row, name=nm(row.name), target=nm(row.target),
                owner=(nm(row.owner) if row.owner else row.owner),
                path=tuple(nm(part) for part in row.path))
        for row in ir.actuators)
    materials = Table(replace(row, name=nm(row.name)) for row in ir.materials)
    ext = dict(ir.ext)
    mj = ext.get("mj")
    if isinstance(mj, MjExt):
        ext["mj"] = _translate_mj_ext(mj, nm)
    return replace(
        ir, bodies=bodies, joints=joints, geoms=geoms, sites=sites,
        cameras=cameras, actuators=actuators, materials=materials, ext=ext)


def build_model_set(merged, *, runtimes=(), single_name=None,
                    meta=None, name_map=None,
                    allow_merge=True) -> tuple[ModelSet, "ModelIR | None"]:
    """Build ``(ModelSet, translated_merged_ir)`` from the compiled IR.

    Args:
        merged: The backend's compiled whole-scene ``ModelIR`` (backend
            names).
        runtimes: The per-articulation runtimes; empty for the single-name
            path.
        single_name: The single articulation name (single path).
        meta: Optional meta entries merged into the IR(s).
        name_map: The backend-private composition name map (identity when
            None).
        allow_merge: When false the published ``merged`` IR is ``None``
            (prefixed multi-articulation runs would collide in the neutral
            joint namespace; read the per-articulation views instead).
    """
    from ..name_map import MjNameMap

    name_map = name_map or MjNameMap.identity()
    if meta:
        merged = replace(merged, meta={**merged.meta, **meta})
    if runtimes:
        irs = {runtime.name: articulation_model_ir(merged, runtime, name_map)
               for runtime in runtimes}
        published_merged = (translate_model_ir(merged, name_map)
                            if allow_merge else None)
    elif single_name is not None:
        published_merged = translate_model_ir(merged, name_map)
        irs = {single_name: published_merged}
    else:
        published_merged = translate_model_ir(merged, name_map)
        irs = {}
    return ModelSet(irs, merged=published_merged), published_merged
