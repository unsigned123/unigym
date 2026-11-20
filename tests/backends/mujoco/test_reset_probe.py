"""ResetPrimitives.probe: scratch MuJoCo free-root collision/clearance queries."""

from __future__ import annotations

import numpy as np
import pytest

from unigym.backends.shared.mjfamily.props.reset_face import build_reset_primitives
from unigym.core.region import RegionDefinition
from unigym.initializers import (
    CollisionPairSpec,
    ProbeCapabilityError,
    ProbePose,
    ProbeRequest,
)
from unigym.initializers.placement import PlacementRule
from unigym.props import build_prop_manager


def _request(owner: str, pos, *, body: str | None = None, **kwargs) -> ProbeRequest:
    return ProbeRequest(
        overrides=(ProbePose(owner=owner, body=body, pos_w=pos, quat_wxyz=(1.0, 0.0, 0.0, 0.0)),),
        **kwargs,
    )


def _prop_manager(names: tuple[str, ...] = ("crate",), num_envs: int = 1):
    types = [
        {
            "name": name,
            "type": "box",
            "size": [0.1, 0.1, 0.1],
            "aabb": {
                "center": [0.0, 0.0, 0.05],
                "half_extents": [0.05, 0.05, 0.05],
            },
            "placement": {
                "mode": "fixed",
                "x_range": [0.5, 0.5],
                "y_range": [0.0, 0.0],
                "z_range": [0.3, 0.3],
            },
        }
        for name in names
    ]
    return build_prop_manager(
        {"types": types},
        num_envs=num_envs,
        env_spacing=2.0,
        device="cpu",
        backend="mujoco",
    )


def _articulated_prop_manager(asset, *, free: bool):
    return build_prop_manager(
        {
            "types": [
                {
                    "name": "laptop",
                    "type": "free_articulation" if free else "articulation",
                    "asset": str(asset),
                    "joints": {"lid_hinge": {"mode": "passive"}},
                    "placement": {
                        "mode": "fixed",
                        "x_range": [1.0, 1.0],
                        "y_range": [0.0, 0.0],
                        "z_range": [0.5, 0.5],
                    },
                }
            ]
        },
        num_envs=1,
        env_spacing=2.0,
        device="cpu",
        backend="mujoco",
    )


def _laptop_asset(tmp_path, *, free: bool):
    freejoint = '<freejoint name="laptop_free"/>' if free else ""
    asset = tmp_path / ("laptop_free.xml" if free else "laptop_anchored.xml")
    asset.write_text(
        f"""<mujoco model="laptop">
  <worldbody>
    <body name="laptop_root">
      {freejoint}
      <geom name="laptop_geom" type="box" size="0.1 0.1 0.02"/>
      <body name="lid" pos="0 0 0.02">
        <joint name="lid_hinge" type="hinge" axis="0 1 0"/>
        <geom name="lid_geom" type="box" size="0.1 0.1 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>""",
        encoding="utf-8",
    )
    return asset


def test_probe_uses_a_scratch_copy_and_leaves_live_state_untouched(tiny_env_factory):
    """A penetrating probe mutates neither live qpos/qvel nor derived xpos."""
    env = tiny_env_factory()
    primitives = build_reset_primitives(env)
    live = env.mj_datas[0]
    qpos = live.qpos.copy()
    qvel = live.qvel.copy()
    xpos = live.xpos.copy()

    result = primitives.probe(_request("tiny", (0.0, 0.0, -0.2)))

    assert result.accepted is False
    assert result.reason == "penetration"
    assert result.distances_m[0] < 0.0
    assert np.array_equal(live.qpos, qpos)
    assert np.array_equal(live.qvel, qvel)
    assert np.array_equal(live.xpos, xpos)
    assert env._active_data is live


@pytest.mark.parametrize(
    ("clearance", "accepted", "reason"),
    [
        (0.0, True, None),
        (0.9, False, "insufficient_clearance"),
    ],
)
def test_probe_checks_surface_clearance(
    tiny_env_factory, clearance: float, accepted: bool, reason: str | None
):
    """Positive clearance uses the signed geom distance, not an origin distance."""
    env = tiny_env_factory()
    primitives = build_reset_primitives(env)
    z = 0.9

    result = primitives.probe(_request("tiny", (0.0, 0.0, z), min_clearance_m=clearance))

    assert result.accepted is accepted
    assert result.reason == reason
    assert result.distances_m[0] == pytest.approx(z - 0.08, abs=1e-9)


@pytest.mark.parametrize(
    ("z", "allow_support", "accepted", "reason"),
    [
        (0.08, True, True, None),
        (0.08, False, False, "insufficient_clearance"),
        (-0.2, True, False, "penetration"),
    ],
)
def test_explicit_pairs_keep_the_support_near_zero_contact_exception(
    tiny_env_factory, z: float, allow_support: bool, accepted: bool, reason: str | None
):
    """Support contact is allowed only near zero; overlap is never excused."""
    env = tiny_env_factory()
    primitives = build_reset_primitives(env)
    pair = CollisionPairSpec(
        "base_geom",
        "floor",
        allow_support_contact=allow_support,
    )

    result = primitives.probe(_request("tiny", (0.0, 0.0, z), pairs=(pair,), min_clearance_m=0.05))

    assert result.accepted is accepted
    assert result.reason == reason


def test_probe_resolves_a_free_rigid_prop(tiny_env_factory):
    """A declared free prop is probed by its owner name like a robot root."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    primitives = build_reset_primitives(env)

    clear = primitives.probe(_request("crate", (0.5, 0.0, 0.3)))
    hit = primitives.probe(_request("crate", (0.5, 0.0, -0.1)))

    assert clear.accepted is True
    assert hit.accepted is False
    assert hit.reason == "penetration"


def test_probe_resolves_a_free_articulation_prop(tiny_env_factory, tmp_path):
    """A free articulation prop is probed by its prop name through its free root."""
    env = tiny_env_factory(
        prop_manager=_articulated_prop_manager(_laptop_asset(tmp_path, free=True), free=True)
    )
    primitives = build_reset_primitives(env)
    prop_art = env.prop_backend.articulations()["laptop"]
    live_qpos = env.mj_datas[0].qpos[prop_art.free_qadr : prop_art.free_qadr + 7].copy()

    clear = primitives.probe(_request("laptop", (1.0, 0.0, 0.5)))
    hit = primitives.probe(_request("laptop", (1.0, 0.0, -0.2)))

    assert clear.accepted is True
    assert hit.accepted is False
    assert hit.reason == "penetration"
    assert np.array_equal(
        env.mj_datas[0].qpos[prop_art.free_qadr : prop_art.free_qadr + 7], live_qpos
    )


def test_probe_rejects_an_anchored_articulation_prop(tiny_env_factory, tmp_path):
    """An anchored articulation prop has a fixed root, not an unknown owner."""
    env = tiny_env_factory(
        prop_manager=_articulated_prop_manager(_laptop_asset(tmp_path, free=False), free=False)
    )
    primitives = build_reset_primitives(env)

    with pytest.raises(
        ProbeCapabilityError,
        match=r"overrides\[0\]\.owner 'laptop'.*anchored articulation prop.*fixed root",
    ):
        primitives.probe(_request("laptop", (1.0, 0.0, 0.5)))


def test_automatic_pairs_exclude_the_owners_own_subtree(tiny_env_factory):
    """The robot's own links are candidates, not obstacles to its root."""
    env = tiny_env_factory()
    primitives = build_reset_primitives(env)

    result = primitives.probe(_request("tiny", (0.0, 0.0, 0.9), min_clearance_m=0.05))

    assert result.accepted is True
    assert len(result.distances_m) == 3
    assert all(distance > 0.05 for distance in result.distances_m)


def test_automatic_pairs_still_include_floor_collision(tiny_env_factory):
    """A subtree link that reaches into the floor rejects the candidate."""
    env = tiny_env_factory()
    primitives = build_reset_primitives(env)

    result = primitives.probe(_request("tiny", (0.0, 0.0, 0.6)))

    assert result.accepted is False
    assert result.reason == "penetration"


def test_automatic_pairs_reject_coincident_owner_overrides(tiny_env_factory):
    """Two coincident free props are cross-paired and rejected."""
    env = tiny_env_factory(prop_manager=_prop_manager(("crate", "block")))
    primitives = build_reset_primitives(env)
    pose = (0.5, 0.0, 0.5)
    request = ProbeRequest(
        overrides=(
            ProbePose("crate", None, pose, (1.0, 0.0, 0.0, 0.0)),
            ProbePose("block", None, pose, (1.0, 0.0, 0.0, 0.0)),
        )
    )

    result = primitives.probe(request)

    assert result.accepted is False
    assert result.reason == "penetration"


def test_automatic_pairs_keep_support_body_geoms_with_near_zero_contact(tiny_env_factory):
    """Support grants near-zero contact; it is never removed from the world."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    primitives = build_reset_primitives(env)
    pose = (0.5, 0.0, 0.0)

    without_support = primitives.probe(_request("crate", pose, min_clearance_m=0.05))
    with_support = primitives.probe(
        _request("crate", pose, min_clearance_m=0.05, support_body="world")
    )

    assert without_support.accepted is False
    assert without_support.reason == "insufficient_clearance"
    assert with_support.accepted is True
    assert with_support.distances_m[0] == pytest.approx(0.0, abs=1e-9)
    assert len(with_support.distances_m) > 1


def test_support_body_near_zero_contact_does_not_cover_its_subtree(
    tiny_env_factory, tiny_mjcf, tmp_path
):
    """Only the named support body gets the contact exception, not its children."""
    scene = tmp_path / "tiny_arm_table.xml"
    scene.write_text(
        tiny_mjcf.read_text(encoding="utf-8").replace(
            "</worldbody>",
            """
    <body name="table" pos="0.5 0 0.05">
      <geom name="table_top" type="box" size="0.3 0.3 0.05"/>
      <body name="table_trim" pos="0 0 0.16">
        <geom name="table_trim_geom" type="box" size="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>""",
        ),
        encoding="utf-8",
    )
    env = tiny_env_factory(mjcf=scene, prop_manager=_prop_manager())
    primitives = build_reset_primitives(env)

    result = primitives.probe(
        _request("crate", (0.5, 0.0, 0.1), min_clearance_m=0.05, support_body="table")
    )

    assert result.accepted is False
    assert result.reason == "insufficient_clearance"
    assert sum(1 for distance in result.distances_m if abs(distance) <= 1e-9) == 2


def test_probe_rejects_body_scoped_overrides(tiny_env_factory):
    """V1 has no arbitrary articulated internal-body pose capability."""
    env = tiny_env_factory()
    primitives = build_reset_primitives(env)

    with pytest.raises(
        ProbeCapabilityError,
        match=r"overrides\[0\]\.body 'base'.*free roots only",
    ):
        primitives.probe(_request("tiny", (0.0, 0.0, 0.5), body="base"))


def test_probe_rejects_an_unknown_owner_with_the_request_field(tiny_env_factory):
    """An unknown owner names the request path and the declared candidates."""
    env = tiny_env_factory()
    primitives = build_reset_primitives(env)

    with pytest.raises(
        ProbeCapabilityError,
        match=r"overrides\[0\]\.owner 'ghost'.*articulations: \['tiny'\]",
    ):
        primitives.probe(_request("ghost", (0.0, 0.0, 0.5)))


def test_probe_rejects_an_unknown_explicit_geom(tiny_env_factory):
    """An unknown explicit geom is a capability failure, not a value error."""
    env = tiny_env_factory()
    primitives = build_reset_primitives(env)
    pair = CollisionPairSpec("base_geom", "ghost_geom")

    with pytest.raises(
        ProbeCapabilityError,
        match=r"pairs\[0\]\.other_geom geom 'ghost_geom' was not found",
    ):
        primitives.probe(_request("tiny", (0.0, 0.0, 0.9), pairs=(pair,)))


def test_probe_rejects_an_unknown_support_body(tiny_env_factory):
    """An unknown support body is a capability failure with its request field."""
    env = tiny_env_factory()
    primitives = build_reset_primitives(env)

    with pytest.raises(
        ProbeCapabilityError,
        match=r"support_body body 'ghost' was not found",
    ):
        primitives.probe(_request("tiny", (0.0, 0.0, 0.9), support_body="ghost"))


def test_probe_rejects_a_fixed_root(tiny_env_factory, tiny_mjcf, tmp_path):
    """A fixed articulation root has no freejoint to override."""
    fixed = tmp_path / "tiny_arm_fixed.xml"
    fixed.write_text(
        tiny_mjcf.read_text(encoding="utf-8").replace('<freejoint name="root"/>', ""),
        encoding="utf-8",
    )
    env = tiny_env_factory(mjcf=fixed)
    primitives = build_reset_primitives(env)

    with pytest.raises(ProbeCapabilityError, match="fixed root"):
        primitives.probe(_request("tiny", (0.0, 0.0, 0.5)))


def test_probe_rejects_a_free_root_without_collision_geoms(tiny_env_factory, tiny_mjcf, tmp_path):
    """A candidate subtree with no collision geoms is a capability failure."""
    source = tiny_mjcf.read_text(encoding="utf-8")
    for name in ("base_geom", "link_geom", "tip_geom"):
        source = source.replace(f'name="{name}"', f'name="{name}" contype="0" conaffinity="0"')
    path = tmp_path / "tiny_no_collision.xml"
    path.write_text(source, encoding="utf-8")
    env = tiny_env_factory(mjcf=path)
    primitives = build_reset_primitives(env)

    with pytest.raises(ProbeCapabilityError, match="no collision geoms"):
        primitives.probe(_request("tiny", (0.0, 0.0, 0.9)))


def test_probe_applies_the_override_orientation(tiny_env_factory):
    """The scratch pose uses the normalized override quaternion."""
    env = tiny_env_factory()
    primitives = build_reset_primitives(env)
    pos = (0.0, 0.0, 0.9)

    identity = primitives.probe(
        ProbeRequest(overrides=(ProbePose("tiny", None, pos, (1.0, 0.0, 0.0, 0.0)),))
    )
    rotated = primitives.probe(
        ProbeRequest(overrides=(ProbePose("tiny", None, pos, (1.0, 0.0, 1.0, 0.0)),))
    )

    assert identity.accepted is True
    assert rotated.accepted is True
    assert rotated.distances_m != identity.distances_m


def test_probe_candidate_still_returns_region_probe_results(tiny_env_factory):
    """The legacy region entry delegates to ``probe`` with the proprietary pairs."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    env._placement_regions = {
        "shelf": RegionDefinition(
            name="shelf",
            owner="floor",
            kind="support",
            width_m=1.0,
            depth_m=1.0,
        )
    }
    primitives = build_reset_primitives(env)
    rule = PlacementRule(object_name="crate", relation="on", region_key="shelf")

    clear = primitives.probe_candidate(rule, (0.5, 0.0, 0.3), 0.0, 0.01)
    hit = primitives.probe_candidate(rule, (0.5, 0.0, -0.1), 0.0, 0.01)

    assert clear.accepted is True
    assert hit.accepted is False
    assert hit.reason == "penetration"
