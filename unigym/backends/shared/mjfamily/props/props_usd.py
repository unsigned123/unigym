"""mujoco USD props: the three DECLARED representation tiers .

The USD-facing half of ``mujoco_backend.py`` (this package's mujoco-facing
modules are lazily imported by the env backend only; split by the
file-size discipline).  Two pieces:

* ``_add_material`` -- ``spawn.texture`` -> an MJCF 2d texture + material
  (cosmetic albedo for box-family props and the ``box_texture`` tier).
* ``_add_usd_body`` -- the mujoco representation of a declared-
  representation prop: ``type: usd`` and ``type: rigid_object``
  carrying a ``mujoco:`` block; ``mjcf_file`` (fragment, ATTACHED with
  the ``Prop_`` prefix), ``box_texture`` / ``box`` (declared envelope-box
  tiers).
"""

from __future__ import annotations


def add_material(mjspec, spec) -> str | None:
    """spawn.texture = {file, name?} -> an MJCF 2d texture + material.

    Honoured by box-family props (cosmetic albedo) and the usd
    ``box_texture`` tier -- the mj-side stand-in for USD materials
    (mujoco has no USD loader).  Returns the material name or None.
    """
    import mujoco

    tex = spec.spawn.get("texture") if spec.type != "usd" else (spec.mujoco or {}).get("texture")
    if not tex:
        return None
    file = tex.get("file") if isinstance(tex, dict) else str(tex)
    if not file:
        return None
    name = (tex.get("name") if isinstance(tex, dict) else None) or f"tex_{spec.name}"
    tex_name = f"{name}_tex"
    mjspec.add_texture(
        name=tex_name,
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        file=str(file),
    )
    mat_name = f"mat_{spec.name}"
    from ..scene.asset_render import material_textures

    mjspec.add_material(name=mat_name, textures=material_textures([tex_name] * 6))
    return mat_name


def add_usd_body(mjspec, spec, body) -> None:
    """Append the mujoco representation of a declared-representation prop.

    Reached for ``type: usd`` props and for ``type: rigid_object`` props
    carrying a ``mujoco:`` block (URDF-less mesh declaration -- the
    fragment carries its own mass/inertia).  Three DECLARED tiers
    (PropSpec.mujoco; the declaration is mandatory for the usd kind, a
    missing/unknown one fails fast listing the options):

    * ``mjcf_file``  -- high-fidelity converted fragment (the equivalent of
      an online USD->MJCF conversion; see scripts/usd_to_mjcf.py).  The
      fragment's worldbody must hold EXACTLY ONE body named after the prop
      carrying one freejoint; it is ATTACHED into this spec with the
      ``Prop_`` prefix (MjSpec.attach), so its body lands as
      ``Prop_<name>`` (the address the runtime side binds).
    * ``box_texture`` -- the declared ``size`` envelope box + texture
      (box_layout single-box layout; approximation tier).
    * ``box`` -- the plain envelope box (geometry-only tier).
    """
    import mujoco

    mj = spec.mujoco or {}
    rep = mj.get("representation")
    if rep is None:
        raise ValueError(
            f"usd prop {spec.name!r}: the mujoco backend has no USD loader "
            f"-- declare its representation explicitly: mujoco: "
            f"{{representation: mjcf_file | box_texture | box, ...}} "
            f"(mjcf_file = converted fragment via scripts/usd_to_mjcf.py; "
            f"box_texture = envelope box + texture; box = plain envelope)"
        )
    if rep == "mjcf_file":
        from ..scene.mjcf_io import load_spec

        child = load_spec(mj["mjcf_file"])
        bodies = list(child.worldbody.bodies)
        if len(bodies) != 1 or bodies[0].name != spec.name:
            raise ValueError(
                f"usd prop {spec.name!r}: mjcf_file {mj['mjcf_file']!r} "
                f"must declare exactly ONE worldbody body named "
                f"{spec.name!r} (found: {[b.name for b in bodies]})"
            )
        joints = list(bodies[0].joints)
        if len(joints) != 1 or int(joints[0].type) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(
                f"usd prop {spec.name!r}: mjcf_file body {spec.name!r} "
                f"must carry exactly one freejoint"
            )
        if not joints[0].name:
            # attach does NOT prefix unnamed joints -- an empty joint name
            # would leak into the env's hand-DOF table; name it explicitly
            # so the Prop_ prefix lands (Prop_<name>_free)
            joints[0].name = f"{spec.name}_free"
        frame = mjspec.worldbody.add_frame(name=f"Prop_{spec.name}_attach")
        mjspec.attach(child, prefix="Prop_", suffix="", frame=frame)
        return
    # box_texture / box: envelope-box tiers
    if not spec.size:
        raise ValueError(
            f"usd prop {spec.name!r}: representation {rep!r} requires the "
            f"declared size: [sx, sy, sz] envelope"
        )
    from unigym.props.container import box_layout

    body = mjspec.worldbody.add_body(name=f"Prop_{spec.name}")
    body.add_joint(name=f"Prop_{spec.name}_free", type=mujoco.mjtJoint.mjJNT_FREE)
    mat_name = add_material(mjspec, spec)
    boxes = box_layout(spec)
    mass = float(mj.get("mass", spec.spawn.get("mass", 1.0)))
    friction = mj.get("friction", spec.spawn.get("friction"))
    rgba = spec.spawn.get("rgba")
    per = mass / len(boxes)
    for i, (center, half) in enumerate(boxes):
        kwargs = {}
        if friction is not None:
            f = float(friction)
            kwargs["friction"] = [f, f, f]
        if rgba is not None:
            kwargs["rgba"] = [float(v) for v in rgba]
        if mat_name is not None:
            kwargs["material"] = mat_name
        body.add_geom(
            name=f"Prop_{spec.name}_geom{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[float(v) for v in center],
            size=[float(v) for v in half],
            mass=per,
            **kwargs,
        )
