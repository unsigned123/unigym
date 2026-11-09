"""mujoco generated-asset renderer: AssetDefinition IR -> MjSpec elements.

The stage-2 render half: a prop whose resolution
produced a neutral :class:`~unigym.props.asset_ir.AssetDefinition` payload
is BUILT and ATTACHED here instead of loaded from a fragment file.  The
attach face is REUSED from :mod:`unigym.backends.shared.mjfamily.props.props_articulation`
(the sole authority): the payload tree is built into a dedicated child
spec (an explicit ``<compiler angle="radian"/>`` pins the IR's radian
convention regardless of the main spec's text units), then attached with
the ``Prop_<prop>_`` prefix at a ``Prop_<prop>_attach`` worldbody frame
-- the prefixing itself is the attach machinery, not hand-spelled names.
Config-override semantics for declared joint damping/range run
pre-attach (the authority order); actuated joints get the
``Prop_<prop>_<j>_act`` position servo on the main spec post-attach; the
effective-range table (``handler._asset_ranges``) feeds the shared
runtime view's ``init`` re-check.

This module is the SOLE interpreter of the payload's ``solver_hints``
block (ruling §12.4-1): neutral names in, engine contact/limit parameters
out.  A material reference is logical (a name, never a path): the payload
declares its meshes (:class:`~unigym.props.asset_ir.MeshDef`, S2.2),
textures (:class:`~unigym.props.asset_ir.TextureDef`, S2.1)
and materials (:class:`~unigym.props.asset_ir.MaterialDef`) INTO the child
spec -- the attach pass namespaces them (and every geom's logical
material/mesh reference) with the same ``Prop_<prop>_`` prefix as the
tree.  A mesh or texture ``file`` may carry ``${TOKEN}`` placeholders;
they are expanded here (child-spec build time) through the sanctioned
:func:`unigym.core.asset_root.resolve_asset_path` ladder (placeholder
expansion + repo-first -> pack fallback for the tracked generator tree),
because a generated payload has no file anchor to resolve relative
paths against.

Runtime is the REUSED ``MujocoPropArticulation``: the compiled ``Prop_``
subtree this renderer produces is exactly what the shared bind/reset/
state faces address, so no new runtime lifecycle exists.

The FREE face (:func:`add_ir_free_body`) covers the robocasa data
objects: jointless free bodies -- the payload carries no IR joint and
the renderer INJECTS the root freejoint (the prop's live dof) with the
official sampling-path damping; runtime is the reused
``MujocoPropFreeArticulation``.  The BALL-joint face (the composite
chains' quaternion dofs with their mass-spring ``springdamper``
parametrization) and the material ``specular`` appearance attribute
are also rendered.
"""
from __future__ import annotations

import functools

import numpy as np

from unigym.core.asset_root import resolve_asset_path
from unigym.props.asset_ir import (
    AssetDefinition,
    ContactHints,
    JointHints,
    asset_roots,
    iter_asset_bodies,
    validate_asset_definition,
)

_CHILD_MODEL = (
    '<mujoco model="unigym_generated_prop">\n'
    '  <compiler angle="radian"/>\n'
    "  <worldbody/>\n"
    "</mujoco>\n"
)


@functools.lru_cache(maxsize=1)
def material_texture_width() -> int:
    """The engine-declared ``mjsMaterial.textures`` width.

    The width is a fixed engine constant -- the own setter refuses any
    other length ("material.textures must have exactly 10 elements",
    mujoco 3.11 through 3.17).  It is probed instead of spelled so a
    future width change cannot silently resurrect the short-list copy
    below.
    """
    import mujoco

    probe = mujoco.MjSpec.from_string(
        '<mujoco model="probe"><asset><material name="m"/></asset>'
        '<worldbody><body><geom type="box" size="0.1 0.1 0.1"'
        ' material="m"/></body></worldbody></mujoco>')
    return len(list(probe.materials[0].textures))


def material_textures(bound: list[str]) -> list[str]:
    """A full-width ``add_material(textures=...)`` binding list.

    ``bound`` is the declared slot prefix (the IR material texture
    reference spells ``["", <name>]`` because slot 1 is the slot the
    MJCF ``texture=`` attribute binds; the USD box tier repeats the
    name across the leading slots).  The tail is padded with empty
    names to :func:`material_texture_width`.

    Why full width is mandatory: mujoco 3.17.0's ``add_material
    (textures=)`` pybind path copies SHORT lists unsafely -- heap
    corruption whose next native call dies.  Observed shapes: attach
    raises "Caught an unknown exception!", deleting spec bodies trips
    a glibc double-free abort, and the GHA runner took a plain
    SIGSEGV two native calls later (the fast suite's exit 139);
    mujoco 3.11 tolerated any length.  Full width is byte-identical
    on 3.11 (the engine left the tail slots empty) and safe on 3.17.
    """
    width = material_texture_width()
    if len(bound) > width:
        raise ValueError(
            f"material texture binding carries {len(bound)} slots -- "
            f"the engine declares {width}")
    return list(bound) + [""] * (width - len(bound))


def add_ir_body(mjspec, handler, asset: AssetDefinition) -> None:
    """Render one generated prop payload into the main MjSpec (pre-compile).

    *handler* is the wired articulation handler (its declared joint table
    is the validated config face); *asset* is the neutral IR payload.
    """
    if asset.forest:
        raise ValueError(
            f"prop {handler.spec.name!r}: a forest payload (worldbody-level "
            f"roots) attaches only through the scene composer's verbatim "
            f"face (scene_render) -- the prop runtime mounts anchored "
            f"single-root payloads")
    child, joints = _build_child(handler, asset)
    _apply_declared_overrides(handler, joints)
    ranges = _effective_ranges(handler, asset)
    handler._asset_ranges = ranges
    spec = handler.spec
    frame = mjspec.worldbody.add_frame(name=f"Prop_{spec.name}_attach")
    mjspec.attach(child, prefix=f"Prop_{spec.name}_", suffix="", frame=frame)
    _add_servos(mjspec, handler, ranges)


#: the official freejoint damping of the robocasa data-object sampling
#: path (robocasa/models/objects/objects.py MJCFObject:
#: ``joints=[dict(type="free", damping="0.0005")]`` -- every sampled
#: kitchen object ships this value; the dump-verbatim fragments carry
#: the same ``obj_joint0 damping="0.0005"``).
_OFFICIAL_FREE_DAMPING = 0.0005


def add_ir_free_body(mjspec, handler, asset: AssetDefinition) -> None:
    """Render a JOINTLESS generated payload with an INJECTED root freejoint.

    The free-body face (the robocasa data objects): the IR keeps
    its no-freejoint guard (a provider can never spell an engine joint),
    so the freejoint is ATTACH MACHINERY here -- added mechanically to
    the payload's root body with the official sampling-path damping,
    like the ``Prop_`` prefix itself.  The resolve-time gate
    (``props.base._resolve_generated_spec``) already refused any
    internal IR joint; the check repeats here as the compile-side
    mirror (defense in depth, same message contract).
    """
    import mujoco

    if asset.forest:
        raise ValueError(
            f"free_articulation prop {handler.spec.name!r}: a forest "
            f"payload (worldbody-level roots) attaches only through the "
            f"scene composer's verbatim face (scene_render)")
    child, joints = _build_child(handler, asset)
    passive = [j for j in joints.values()
               if j.type in (mujoco.mjtJoint.mjJNT_HINGE,
                             mujoco.mjtJoint.mjJNT_SLIDE)]
    if len(passive) != len(joints):
        raise ValueError(
            f"free_articulation prop {handler.spec.name!r}: a generated "
            f"payload carries passive hinge/slide internal joints only "
            f"in v1 (the injected freejoint is the prop's live dof); "
            f"got {sorted(joints)}")
    if len(child.worldbody.bodies) > 1 \
            and child.worldbody.bodies[0].joints:
        raise ValueError(
            f"free_articulation prop {handler.spec.name!r}: the payload "
            f"ROOT body carries an internal joint -- the injected "
            f"freejoint owns the root's dofs; internal joints live on "
            f"child bodies (the laptop lid face)")
    root = child.worldbody.bodies[0]
    free = root.add_joint(name="free", type=mujoco.mjtJoint.mjJNT_FREE)
    free.damping = [_OFFICIAL_FREE_DAMPING, 0.0, 0.0]
    spec = handler.spec
    frame = mjspec.worldbody.add_frame(name=f"Prop_{spec.name}_attach")
    mjspec.attach(child, prefix=f"Prop_{spec.name}_", suffix="", frame=frame)


def _build_child(handler, asset: AssetDefinition):
    """Build the payload's child MjSpec; returns (child, {jname: MjsJoint}).

    The shared construction of both attach forms (anchored
    :func:`add_ir_body` / free-root :func:`add_ir_free_body`):
    identity gate, declaration tables, body tree.
    """
    import mujoco

    validate_asset_definition(asset)
    spec = handler.spec
    if asset.name != spec.name:
        raise ValueError(
            f"articulation prop {spec.name!r}: the asset definition names "
            f"{asset.name!r} -- a payload must carry its own prop identity")
    child = mujoco.MjSpec.from_string(_CHILD_MODEL)
    _add_declarations(mujoco, child, asset)
    joints: dict = {}
    for root in asset_roots(asset):
        _add_body(mujoco, child.worldbody, root, asset.solver_hints, joints)
    return child, joints


def _add_declarations(mujoco, child, asset: AssetDefinition) -> None:
    """Render the payload's mesh/texture/material tables into the child spec.

    Meshes first (S2.2), then textures (materials reference them), then
    materials; declaration order is preserved because the compiled asset
    arrays follow it.  A mesh ``file`` is placeholder-expanded exactly
    like a texture ``file``.  The material texture reference is declared
    as ``["", <name>]``: the engine binds
    the reference at the same material slot the MJCF ``texture=``
    attribute lands in, regardless of the texture's projection kind
    (binding a name at index 0 would land in a different slot and break
    compiled equality with a reference authored as MJCF).
    """
    kind_enum = {
        "2d": mujoco.mjtTexture.mjTEXTURE_2D,
        "cube": mujoco.mjtTexture.mjTEXTURE_CUBE,
    }
    for mesh in asset.meshes:
        kwargs = {"name": mesh.name,
                  "file": resolve_asset_path(mesh.file)}
        if mesh.scale is not None:
            kwargs["scale"] = [float(v) for v in mesh.scale]
        obj = child.add_mesh(**kwargs)
        if mesh.refquat is not None:
            # post-add assignment: the mesh orientation fact lands on
            # the declaration exactly as the MJCF ``refquat`` attribute
            # would (the engine applies it to the loaded vertices)
            obj.refquat = [float(v) for v in mesh.refquat]
    for tex in asset.textures:
        child.add_texture(name=tex.name, type=kind_enum[tex.kind],
                          file=resolve_asset_path(tex.file))
    for mat in asset.materials:
        kwargs = {}
        if mat.rgba is not None:
            kwargs["rgba"] = [float(v) for v in mat.rgba]
        if mat.shininess is not None:
            kwargs["shininess"] = float(mat.shininess)
        if mat.reflectance is not None:
            kwargs["reflectance"] = float(mat.reflectance)
        if mat.specular is not None:
            kwargs["specular"] = float(mat.specular)
        if mat.texuniform is not None:
            kwargs["texuniform"] = bool(mat.texuniform)
        if mat.texrepeat is not None:
            kwargs["texrepeat"] = [float(v) for v in mat.texrepeat]
        if mat.texture is not None:
            kwargs["textures"] = material_textures(["", mat.texture.name])
        child.add_material(name=mat.name, **kwargs)


def _add_body(mujoco, parent, body, hints, joints) -> None:
    """Add one IR body (+ subtree) under *parent*.

    Raw names: the attach prefix lands later, exactly as for a fragment
    file.
    """
    mjs_body = parent.add_body(name=body.name,
                               pos=list(body.pos), quat=list(body.quat))
    if body.inertial is not None:
        # the explicit-inertial face: the dump's full-precision
        # row verbatim; the engine flags the body explicitinertial and
        # skips the geom mass aggregation, exactly like the XML spelling
        mjs_body.ipos = list(body.inertial.pos)
        mjs_body.iquat = list(body.inertial.quat)
        mjs_body.mass = float(body.inertial.mass)
        mjs_body.inertia = [float(v) for v in body.inertial.diaginertia]
    if body.gravcomp:
        mjs_body.gravcomp = body.gravcomp
    for geom in body.geoms:
        _add_geom(mujoco, mjs_body, geom, hints.contact)
    for joint in body.joints:
        joints[joint.name] = _add_joint(mujoco, mjs_body, joint,
                                        hints.joint)
    for site in body.sites:
        _add_site(mjs_body, site)
    for child in body.children:
        _add_body(mujoco, mjs_body, child, hints, joints)


def _geom_type(mujoco, type_name: str):
    """Map a closed IR geom type onto the engine enum (coupling point)."""
    return {
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
        "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
        "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        "mesh": mujoco.mjtGeom.mjGEOM_MESH,
    }[type_name]


#: the engine-default impedance tail (MJCF element-wise padding): an
#: authored prefix of the 5-value curve keeps these defaults for the
#: slots the author left silent (verified: XML ``solimp="0.998 0.998"``
#: compiles to ``[0.998, 0.998, 0.001, 0.5, 2.0]``).
_SOLIMP_DEFAULT_TAIL = (0.001, 0.5, 2.0)


def _add_geom(mujoco, mjs_body, geom, contact) -> None:
    """Add one IR geom (name/pos/quat/material/mass + contact hints)."""
    kwargs = {
        "type": _geom_type(mujoco, geom.type),
        "pos": list(geom.pos),
        "quat": list(geom.quat),
        "group": int(geom.group),
    }
    if geom.type != "mesh":
        kwargs["size"] = list(geom.size)
    else:
        kwargs["meshname"] = geom.mesh
    if not geom.collidable:
        # the pure-visual twin: both contact-activation bitmasks zeroed
        kwargs["contype"] = 0
        kwargs["conaffinity"] = 0
    if geom.name is not None:
        kwargs["name"] = geom.name
    if geom.material is not None:
        kwargs["material"] = geom.material.name
    if geom.mass is not None:
        kwargs["mass"] = geom.mass
    if geom.density is not None:
        kwargs["density"] = geom.density
    if geom.rgba is not None:
        kwargs["rgba"] = list(geom.rgba)
    if contact is not None:
        kwargs.update(_contact_kwargs(contact))
    if geom.hints is not None:
        kwargs.update(_contact_kwargs(geom.hints))
    mjs_body.add_geom(**kwargs)


def _add_site(mjs_body, site) -> None:
    """Add one IR site (pos + optional size / rgba / type).

    The size is a scalar in the IR: it is assigned to the FIRST
    component only, post-add, so the trailing components keep the
    engine's default vector (the compiled-residue semantics of a
    scalar-authored reference; an ``add_site(size=[s])`` kwarg would
    zero-pad them instead).  A declared non-sphere type (the composite
    cylinder markers) is assigned post-add the same way -- the element
    identity, never a payload free-form string.
    """
    import mujoco

    obj = mjs_body.add_site(name=site.name, pos=list(site.pos))
    if site.size_vector is not None:
        obj.size = [*site.size_vector, *obj.size[len(site.size_vector):]]
    elif site.size is not None:
        obj.size[0] = float(site.size)
    if site.rgba is not None:
        obj.rgba = [float(v) for v in site.rgba]
    if site.type is not None:
        obj.type = {
            "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
            "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
            "box": mujoco.mjtGeom.mjGEOM_BOX,
        }[site.type]


def _contact_kwargs(contact: ContactHints) -> dict:
    """Interpret the neutral contact hints into engine geom parameters.

    A prefix-form impedance curve (2 or 3 authored values) is padded
    with the engine-default tail (element-wise, the MJCF convention) --
    the spec API requires the full 5-vector and does not pad.
    """
    kwargs = {}
    if contact.friction is not None:
        kwargs["friction"] = list(contact.friction)
    if contact.condim is not None:
        kwargs["condim"] = int(contact.condim)
    if contact.priority is not None:
        kwargs["priority"] = int(contact.priority)
    if contact.solver_ref is not None:
        kwargs["solref"] = list(contact.solver_ref)
    if contact.solver_impedance is not None:
        curve = list(contact.solver_impedance)
        if len(curve) < 5:
            curve.extend(_SOLIMP_DEFAULT_TAIL[len(curve) - 2:])
        kwargs["solimp"] = curve
    if contact.margin is not None:
        kwargs["margin"] = float(contact.margin)
    if contact.gap is not None:
        kwargs["gap"] = float(contact.gap)
    return kwargs


def _add_joint(mujoco, mjs_body, joint, hints):
    """Add one IR joint; returns the MjsJoint (for declaration overrides).

    A ball joint (the composite chains) carries no
    axis (quaternion dofs) and its ``springdamper`` parametrization --
    set AFTER the explicit damping default so the engine derives the
    compiled stiffness/damping from the parametrization (the official
    MJCF attribute semantics), not from the IR's silent zero.
    """
    jtype = {
        "hinge": mujoco.mjtJoint.mjJNT_HINGE,
        "slide": mujoco.mjtJoint.mjJNT_SLIDE,
        "ball": mujoco.mjtJoint.mjJNT_BALL,
    }[joint.type]
    kwargs = {"name": joint.name, "type": jtype, "pos": list(joint.pos)}
    if joint.type != "ball":
        kwargs["axis"] = list(joint.axis)
    jobj = mjs_body.add_joint(**kwargs)
    if joint.range is not None:
        jobj.range = [float(joint.range[0]), float(joint.range[1])]
    jobj.damping = [float(joint.damping), 0.0, 0.0]
    jobj.armature = float(joint.armature)
    jobj.frictionloss = float(joint.frictionloss)
    if joint.springdamper is not None:
        jobj.springdamper = [float(v) for v in joint.springdamper]
    if joint.stiffness is not None:
        jobj.stiffness = [float(joint.stiffness), 0.0, 0.0]
    if hints is not None:
        _apply_joint_hints(jobj, hints)
    return jobj


#: the engine-default joint-limit solver pair (MJCF element-wise
#: padding): a single authored ``solreflimit`` value keeps 1 for the
#: damping-ratio slot; the impedance prefix keeps the
#: :data:`_SOLIMP_DEFAULT_TAIL`.
_SOLREFLIMIT_DEFAULT_TAIL = (1.0,)


def _apply_joint_hints(jobj, hints: JointHints) -> None:
    """Interpret the neutral joint limit hints into engine parameters.

    MJCF prefix-form padding: authored prefix + engine-default
    tail -- the same element-wise convention the geom solimp face
    speaks.
    """
    if hints.limit_solver_ref is not None:
        values = tuple(hints.limit_solver_ref)
        values += _SOLREFLIMIT_DEFAULT_TAIL[len(values):]
        jobj.solref_limit = list(values)
    if hints.limit_solver_impedance is not None:
        values = tuple(hints.limit_solver_impedance)
        values += _SOLIMP_DEFAULT_TAIL[len(values) - 2:]
        jobj.solimp_limit = list(values)


def _apply_declared_overrides(handler, joints: dict) -> None:
    """Config-override semantics (fragment-path parity).

    A DECLARED damping/range replaces the IR default; a silent
    declaration keeps the IR value (the raw spec.joints mapping carries
    the presence information -- the handler's validated table coerces
    absent values to numeric defaults).
    """
    declared_raw = handler.spec.joints or {}
    for jname, jobj in joints.items():
        raw = declared_raw.get(jname) or {}
        decl = (handler.joints or {}).get(jname) or {}
        if raw.get("damping") is not None:
            jobj.damping = [float(decl["damping"]), 0.0, 0.0]
        if raw.get("range") is not None:
            jobj.range = [float(v) for v in decl["range"]]


def _effective_ranges(handler, asset: AssetDefinition) -> dict:
    """``{jname: [lo, hi]}`` per declared joint with a range at any level.

    The fragment-path parity table: the shared runtime view re-checks a
    declared ``init`` (whose declaration omitted a range) against these
    at bind time.  Declared ranges win; IR ranges fill the silence.
    """
    declared_raw = handler.spec.joints or {}
    declared = (handler.joints or {})
    ranges: dict = {}
    for body in iter_asset_bodies(asset):
        for joint in body.joints:
            raw = declared_raw.get(joint.name) or {}
            if raw.get("range") is not None:
                ranges[joint.name] = [float(v) for v in
                                      declared[joint.name]["range"]]
            elif joint.range is not None:
                ranges[joint.name] = [float(joint.range[0]),
                                      float(joint.range[1])]
    return ranges


def _add_servos(mjspec, handler, ranges: dict) -> None:
    """Position servo per actuated joint (the authority form)."""
    import mujoco

    spec = handler.spec
    for jname, p in (handler.joints or {}).items():
        if p["mode"] != "actuated":
            continue
        kp, kd = p["stiffness"], p["damping"]
        effective = (p["range"] if p["range"] is not None
                     else ranges.get(jname))
        if effective is None:
            lo, hi, limited = 0.0, 0.0, False
        else:
            lo, hi = effective
            limited = p["range"] is not None or lo != hi or lo != 0.0
        mjspec.add_actuator(
            name=f"Prop_{spec.name}_{jname}_act",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=f"Prop_{spec.name}_{jname}",
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=np.array([kp] + [0.0] * 9),
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            # position servo: force = kp*ctrl - kp*q - kd*dq  (biasprm[1]
            # carries the NEGATED stiffness -- the mjcf <position> form)
            biasprm=np.array([0.0, -kp, -kd] + [0.0] * 7),
            ctrllimited=limited, ctrlrange=[lo, hi])


def verify_prop_subtree(m, prop_name: str, declared_joints) -> dict:
    """Compile gate: one ``Prop_`` root body + every declared joint present.

    The post-compile mirror of the runtime bind's fail-fasts (exactly one
    root body under the prefix; each declared internal joint resolves in
    the compiled model).  Returns the resolved addresses for tests and
    runtime diagnostics.
    """
    import mujoco

    prefix = f"Prop_{prop_name}_"
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
             for b in range(m.nbody)]
    cands = [b for b, n in enumerate(names) if n and n.startswith(prefix)]
    roots = [b for b in cands
             if not (names[m.body_parentid[b]] or "").startswith(prefix)]
    if len(roots) != 1:
        raise ValueError(
            f"articulation prop {prop_name!r}: expected exactly one root "
            f"body under prefix {prefix!r}, found {roots}")
    joints = {}
    for jname in declared_joints:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{prefix}{jname}")
        if jid < 0:
            raise ValueError(
                f"articulation prop {prop_name!r}: joint {jname!r} "
                f"({prefix + jname!r}) not in the compiled model")
        joints[jname] = jid
    return {"root_bid": roots[0], "joints": joints}
