"""Shared MuJoCo-family director spec injection (mocap hand + weld slots).

``mujoco`` and ``mjwarp`` speak the same pre-compile ``MjSpec``, so the
director's spec-side injection lives here instead of in either backend
package (v100 design §3 ``shared/mjfamily/director``).  Engine access stays
function-local; the public runtime hands live in the backend packages.
"""
from __future__ import annotations

from unigym.director.gripper import normalize_gripper_cfg


def add_director_bodies(mjspec, director_cfg: dict | None,
                        prop_manager=None) -> dict:
    """``--direct`` spec-side injection (mjfamily, pre-compile).

    1. the gripper mocap body (when declared), and
    2. one DISABLED weld equality per rigid prop (attach runtime slots).

    Returns ``{"gripper": name | None, "welds": {prop: eq_name}}``.
    No ``--direct`` -> never called (the model stays byte-identical,
    contract §0.3).
    """
    import mujoco

    cfg = director_cfg or {}
    gripper = normalize_gripper_cfg(cfg.get("gripper"))
    welds: dict[str, str] = {}
    if prop_manager is not None:
        for handler in prop_manager._handlers:
            hspec = handler.spec
            if hspec.type in ("articulation", "usd_articulation"):
                continue              # anchored subtrees: no freejoint body
            if hspec.type == "usd" and hspec.spawn.get("static"):
                continue              # stage decoration: no body
            welds[hspec.name] = f"director_weld_{hspec.name}"
            mjspec.add_equality(name=welds[hspec.name],
                                type=int(mujoco.mjtEq.mjEQ_WELD),
                                objtype=int(mujoco.mjtObj.mjOBJ_BODY),
                                name1=f"Prop_{hspec.name}", name2="world",
                                active=0)
    if gripper is not None:
        _add_gripper_body(mujoco, mjspec, gripper)
    return {"gripper": gripper["name"] if gripper else None, "welds": welds}


def _add_gripper_body(mujoco, mjspec, g: dict) -> None:
    """The mocap hand: non-colliding box + grasp site + camera element."""
    name = g["name"]
    _fail_on_collision(mujoco, mjspec, [
        (name, mujoco.mjtObj.mjOBJ_BODY),
        (f"{name}_grasp", mujoco.mjtObj.mjOBJ_SITE),
        (f"{name}_{g['camera']['name']}", mujoco.mjtObj.mjOBJ_CAMERA),
    ])
    # the declared pos seeds d.mocap_pos (mj initializes mocap_pos from the
    # model body_pos) -- the contract SS4.1 "initial world pose" face.
    # shipped without it on the mujoco side (parks at origin; the
    # isaaclab builder honoured the same key), masked in early director
    # flows by explicit absolute move commands before any grasp.
    body = mjspec.worldbody.add_body(name=name, mocap=True,
                                     pos=[float(v) for v in g["pos"]])
    body.add_geom(name=f"{name}_geom",
                  type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[float(v) for v in g["size"]],
                  rgba=[float(v) for v in g["rgba"]],
                  contype=0, conaffinity=0)
    body.add_site(name=f"{name}_grasp", pos=[0.0, 0.0, -float(g["size"][2])],
                  size=[0.005, 0.005, 0.005])
    body.add_camera(name=f"{name}_{g['camera']['name']}",
                    fovy=float(g["camera"]["fovy"]))


def _fail_on_collision(mujoco, mjspec, names) -> None:
    """Spec-stage naming-collision gate (contract §6).

    The MJ model has ONE namespace across bodies/joints/sites/cameras, so
    the walk covers all four object kinds already present in the spec
    (camera rigs / props attach before the director).
    """
    taken: dict[str, int] = {}
    for b in _spec_bodies(mjspec.worldbody):
        taken.setdefault(b.name, mujoco.mjtObj.mjOBJ_BODY)
        for j in b.joints:
            taken.setdefault(j.name, mujoco.mjtObj.mjOBJ_JOINT)
        for s in getattr(b, "sites", []):
            taken.setdefault(s.name, mujoco.mjtObj.mjOBJ_SITE)
        for c in getattr(b, "cameras", []):
            taken.setdefault(c.name, mujoco.mjtObj.mjOBJ_CAMERA)
    for name, kind in names:
        if name in taken:
            raise ValueError(
                f"director gripper name {name!r} collides with an existing "
                f"model name (kind {int(kind)}); set director.gripper.name")


def _spec_bodies(body):
    yield body
    for child in body.bodies:
        yield from _spec_bodies(child)
