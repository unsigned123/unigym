"""Post-compile MuJoCo ``MjModel`` readers for the neutral model IR.

This is the ``describe`` half: declared geometry/pose stays as compiled,
and the compiler-owned values (addresses, mass/inertia, element ids) are
filled in because they now genuinely exist.
"""
from __future__ import annotations

from .spec_walk import (
    _CAM_MODES,
    _EQ_TYPES,
    _GEOM_TYPES,
    _JOINT_TYPES,
    _TRN_TYPES,
    _TEXTURE_TYPES,
    _actuator_kind,
    _basename,
    _finite,
    _option_dict,
    _optional_int,
    _safe_name,
    _vec,
    _vec2,
    _vec3,
    _vec4,
)
from unigym.model.ir import (
    Actuator,
    Body,
    Camera,
    EqSlot,
    Geom,
    GeomContact,
    GeomGroup,
    Joint,
    Material,
    MjExt,
    ModelIR,
    Site,
    Texture,
)
from unigym.model.table import Table

_EQ_OBJ_TYPES = {1: "body", 3: "joint", 5: "geom", 6: "site", 18: "tendon"}


def _model_name(model):
    """Return the first NUL-terminated model name, or ``"model"``."""
    names = getattr(model, "names", b"")
    if isinstance(names, (bytes, bytearray)):
        head = bytes(names).split(b"\0", 1)[0].decode("utf-8", "replace")
        return head or "model"
    return "model"


def _body_name(model, index):
    """Return one model body's IR name."""
    if index == 0:
        return model.body(0).name or "world"
    return _safe_name("body", index, model.body(index).name)


def _body_paths(model):
    """Return ``{body id: path tuple}`` from the parent-id chain."""
    cache = {}

    def path(index):
        if index in cache:
            return cache[index]
        if index == 0:
            result = ()
        else:
            parent = int(model.body_parentid[index])
            result = path(parent) + (_body_name(model, index),)
        cache[index] = result
        return result

    for index in range(model.nbody):
        path(index)
    return cache


def _body_row(model, index, paths, explicit_inertial):
    """Build one :class:`Body` row from a compiled model body."""
    name = _body_name(model, index)
    path_value = paths.get(index, ())
    parent = None if index == 0 else _body_name(
        model, int(model.body_parentid[index]))
    return Body(
        name=name,
        parent=parent,
        path=path_value,
        qualname="/".join(path_value) or name,
        pos=_vec3(model.body_pos[index]),
        quat=_vec4(model.body_quat[index]),
        mocap=int(model.body_mocapid[index]) >= 0,
        explicitinertial=bool(explicit_inertial[index]),
        body_id=index,
        mass=float(model.body_mass[index]),
        inertia=_vec3(model.body_inertia[index]),
        ipos=_vec3(model.body_ipos[index]),
        iquat=_vec4(model.body_iquat[index]),
    )


def _material_name(model, index):
    """Return a material name for an id, or ``None`` when unset."""
    if index < 0 or index >= model.nmat:
        return None
    return _safe_name("material", index, model.material(index).name)


def _target_name(model, trntype, target_id):
    """Return the native name an actuator target id points at."""
    if target_id < 0:
        return ""
    kind = trntype if isinstance(trntype, str) else _TRN_TYPES.get(
        trntype, "")
    if kind in ("joint", "jointinparent"):
        return _safe_name("joint", target_id, model.joint(target_id).name)
    if kind == "site":
        return _safe_name("site", target_id, model.site(target_id).name)
    if kind == "body":
        return _body_name(model, target_id)
    return ""


def _actuator_joint_map(model):
    """Return ``{joint id: actuator name}`` for joint actuators."""
    result = {}
    for index in range(model.nu):
        trntype = int(model.actuator_trntype[index])
        if trntype in (0, 1):
            joint_id = int(model.actuator_trnid[index][0])
            result[joint_id] = _safe_name(
                "actuator", index, model.actuator(index).name)
    return result


def _actuator_row(model, index):
    """Build one compiled :class:`Actuator` row."""
    trntype = _TRN_TYPES.get(
        int(model.actuator_trntype[index]),
        str(int(model.actuator_trntype[index])),
    )
    target = _target_name(
        model, trntype, int(model.actuator_trnid[index][0]))
    name = _safe_name("actuator", index, model.actuator(index).name)
    kind, owner = _actuator_kind(name, target, trntype)
    ctrlrange = (
        _vec2(model.actuator_ctrlrange[index])
        if int(model.actuator_ctrllimited[index]) else None
    )
    return Actuator(
        name=name,
        target=target,
        trntype=trntype,
        kind=kind,
        owner=owner,
        ctrlrange=ctrlrange,
        gear=_vec(model.actuator_gear[index]),
        gainprm=_vec(model.actuator_gainprm[index]),
        biasprm=_vec(model.actuator_biasprm[index]),
        actuator_id=index,
        qualname=name,
    )


def _joint_row(model, index, actuator_by_joint):
    """Build one compiled :class:`Joint` row."""
    name = _safe_name("joint", index, model.joint(index).name)
    body = _body_name(model, int(model.jnt_bodyid[index]))
    raw_range = _vec2(model.jnt_range[index])
    limited = bool(int(model.jnt_limited[index])) and raw_range != (0.0, 0.0)
    dof = int(model.jnt_dofadr[index])
    return Joint(
        name=name,
        type=_JOINT_TYPES.get(
            int(model.jnt_type[index]), str(int(model.jnt_type[index]))),
        body=body,
        axis=_vec3(model.jnt_axis[index]),
        range=raw_range if limited else None,
        limited=limited,
        pos=_vec3(model.jnt_pos[index]),
        damping=float(model.dof_damping[dof]),
        armature=float(model.dof_armature[dof]),
        actuator=actuator_by_joint.get(index),
        qpos_adr=int(model.jnt_qposadr[index]),
        dof_adr=dof,
        qualname=f"{body}/{name}",
    )


def _geom_row(model, index):
    """Build one compiled :class:`Geom` row."""
    name = _safe_name("geom", index, model.geom(index).name)
    return Geom(
        name=name,
        type=_GEOM_TYPES.get(
            int(model.geom_type[index]), str(int(model.geom_type[index]))),
        body=_body_name(model, int(model.geom_bodyid[index])),
        pos=_vec3(model.geom_pos[index]),
        quat=_vec4(model.geom_quat[index]),
        size=_vec3(model.geom_size[index]),
        material=_material_name(model, int(model.geom_matid[index])),
        group=int(model.geom_group[index]),
        contype=int(model.geom_contype[index]),
        conaffinity=int(model.geom_conaffinity[index]),
        geom_id=index,
        qualname=f"{_body_name(model, int(model.geom_bodyid[index]))}/{name}",
    )


def _site_row(model, index):
    """Build one compiled :class:`Site` row."""
    name = _safe_name("site", index, model.site(index).name)
    return Site(
        name=name,
        type=_GEOM_TYPES.get(
            int(model.site_type[index]), str(int(model.site_type[index]))),
        body=_body_name(model, int(model.site_bodyid[index])),
        pos=_vec3(model.site_pos[index]),
        quat=_vec4(model.site_quat[index]),
        size=_vec3(model.site_size[index]),
        material=_material_name(model, int(model.site_matid[index])),
        group=int(model.site_group[index]),
        site_id=index,
        qualname=f"{_body_name(model, int(model.site_bodyid[index]))}/{name}",
    )


def _camera_row(model, index):
    """Build one compiled :class:`Camera` row."""
    name = _safe_name("camera", index, model.camera(index).name)
    target_id = int(model.cam_targetbodyid[index])
    return Camera(
        name=name,
        body=_body_name(model, int(model.cam_bodyid[index])),
        pos=_vec3(model.cam_pos[index]),
        quat=_vec4(model.cam_quat[index]),
        fovy=float(model.cam_fovy[index]),
        mode=_CAM_MODES.get(
            int(model.cam_mode[index]), str(int(model.cam_mode[index]))),
        targetbody=_body_name(model, target_id) if target_id >= 0 else None,
        camera_id=index,
        qualname=f"{_body_name(model, int(model.cam_bodyid[index]))}/{name}",
    )


def _material_textures(model, index):
    """Return the named texture bindings of one material."""
    names = []
    for texture_id in model.mat_texid[index]:
        texture_id = int(texture_id)
        if 0 <= texture_id < model.ntex:
            name = model.texture(texture_id).name
            if name:
                names.append(name)
    return tuple(names)


def _material_row(model, index):
    """Build one compiled :class:`Material` row."""
    return Material(
        name=_safe_name("material", index, model.material(index).name),
        rgba=_vec4(model.mat_rgba[index]),
        textures=_material_textures(model, index),
        reflectance=_finite(model.mat_reflectance[index]),
        roughness=_finite(model.mat_roughness[index]),
        metallic=_finite(model.mat_metallic[index]),
        emission=_finite(model.mat_emission[index]),
        specular=_finite(model.mat_specular[index]),
        shininess=_finite(model.mat_shininess[index]),
        texrepeat=_vec2(model.mat_texrepeat[index]),
        texuniform=bool(model.mat_texuniform[index]),
    )


def _eq_obj_name(model, objtype, objid):
    """Return the name of one equality operand."""
    kind = _EQ_OBJ_TYPES.get(objtype)
    if kind is None or objid < 0:
        return None
    if kind == "body":
        return _body_name(model, objid)
    accessor = getattr(model, kind, None)
    if accessor is None:
        return None
    return _safe_name(kind, objid, accessor(objid).name)


def _eq_row(model, index):
    """Build one compiled :class:`EqSlot` row."""
    objtype = int(model.eq_objtype[index])
    return EqSlot(
        name=_safe_name("eq", index, model.eq(index).name),
        type=_EQ_TYPES.get(
            int(model.eq_type[index]), str(int(model.eq_type[index]))),
        obj1=_eq_obj_name(model, objtype, int(model.eq_obj1id[index])),
        obj2=_eq_obj_name(model, objtype, int(model.eq_obj2id[index])),
        active=bool(model.eq_active0[index]),
    )


def _geom_groups(geom_rows):
    """Group compiled geoms by their render group."""
    groups = {}
    for geom in geom_rows:
        groups.setdefault(geom.group, []).append(geom.name)
    return [
        GeomGroup(name=f"group{group}", group=group, geoms=tuple(names))
        for group, names in sorted(groups.items())
    ]


def _model_texture_file(model, index):
    """Return one compiled texture's file basename, when it has one."""
    pathadr = getattr(model, "tex_pathadr", None)
    names = getattr(model, "names", b"")
    if pathadr is None or not isinstance(names, (bytes, bytearray)):
        return None
    address = int(pathadr[index])
    if address < 0:
        return None
    text = bytes(names)[address:].split(b"\0", 1)[0].decode(
        "utf-8", "replace")
    return _basename(text)


def _texture_row_from_model(model, index):
    """Build one :class:`Texture` row from a compiled model texture."""
    name = _safe_name("texture", index, model.texture(index).name)
    return Texture(
        name=name,
        type=_TEXTURE_TYPES.get(
            int(model.tex_type[index]), str(int(model.tex_type[index]))),
        file=_model_texture_file(model, index),
        width=_optional_int(model.tex_width[index], minimum=1),
        height=_optional_int(model.tex_height[index], minimum=1),
        nchannel=_optional_int(model.tex_nchannel[index], minimum=1),
    )


def _geom_contact_row_from_model(model, index, name):
    """Build one :class:`GeomContact` row from a compiled model geom."""
    return GeomContact(
        name=name,
        friction=_vec(model.geom_friction[index]),
        solref=_vec(model.geom_solref[index]),
        solimp=_vec(model.geom_solimp[index]),
        margin=_finite(model.geom_margin[index]),
        gap=_finite(model.geom_gap[index]),
        priority=int(model.geom_priority[index]),
        condim=int(model.geom_condim[index]),
    )


def _ext(model, body_rows, geom_rows):
    """Build the ``mj`` extension from a compiled model."""
    mocap = [row for row in body_rows if row.mocap]
    eq_slots = [_eq_row(model, index) for index in range(model.neq)]
    textures = [
        _texture_row_from_model(model, index)
        for index in range(model.ntex)
    ]
    geom_contact = [
        _geom_contact_row_from_model(model, index, geom_rows[index].name)
        for index in range(model.ngeom)
    ]
    return MjExt(
        opt=_option_dict(model.opt),
        geom_groups=Table(_geom_groups(geom_rows)),
        eq_slots=Table(eq_slots),
        mocap_bodies=Table(mocap),
        textures=Table(textures),
        geom_contact=Table(geom_contact),
    )


def describe_model(model, backend="mujoco"):
    """Read a compiled ``MjModel`` and return the filled model IR."""
    explicit = getattr(model, "body_explicitinertial", None)
    if explicit is None:
        explicit = [False] * model.nbody
    paths = _body_paths(model)
    body_rows = [
        _body_row(model, index, paths, explicit)
        for index in range(model.nbody)
    ]
    actuator_by_joint = _actuator_joint_map(model)
    joint_rows = [
        _joint_row(model, index, actuator_by_joint)
        for index in range(model.njnt)
    ]
    geom_rows = [_geom_row(model, index) for index in range(model.ngeom)]
    site_rows = [_site_row(model, index) for index in range(model.nsite)]
    camera_rows = [
        _camera_row(model, index) for index in range(model.ncam)
    ]
    actuator_rows = [
        _actuator_row(model, index) for index in range(model.nu)
    ]
    material_rows = [
        _material_row(model, index) for index in range(model.nmat)
    ]
    ir = ModelIR(
        name=_model_name(model),
        backend=backend,
        compiled=True,
        bodies=Table(body_rows),
        joints=Table(joint_rows),
        geoms=Table(geom_rows),
        sites=Table(site_rows),
        cameras=Table(camera_rows),
        actuators=Table(actuator_rows),
        materials=Table(material_rows),
        meta={
            "source": None,
            "scene": _model_name(model),
            "builder": None,
        },
        ext={"mj": _ext(model, body_rows, geom_rows)},
    )
    return ir.validate()
