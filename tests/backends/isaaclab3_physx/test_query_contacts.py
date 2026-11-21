"""Real PhysX filtered contact manifold (P1C-3).

Two overlapping dynamic crates are the smallest contact-producing filtered
pair on this lane: neither side is static terrain, so the PhysX filtered
contact view reports their crate-to-crate contacts.  The group opts in with
``manifold: true``; the query then returns body-named points with the engine's
position and separation plus a non-negative force magnitude.  The engine
contact-normal direction is not exposed because this filtered view does not
prove its orientation.
"""

from __future__ import annotations

import math

import pytest
import torch

from unigym.backends.shared.isaaclab3_common.contacts import (
    MANIFOLD_MAX_CONTACT_DATA_COUNT_PER_PRIM,
)
from unigym.query import QueryCapabilityError

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

ARTICULATION = "cartpole"
SENSOR_PROP = "crate_a"
FILTER_PROP = "crate_b"
# ContactPoint carries compiled asset body names (the MuJoCo P1A contract), not
# the neutral prop names used to declare the group.
SENSOR_BODY = "Prop_crate_a"
FILTER_BODY = "Prop_crate_b"


def _two_crate_props(num_envs: int = 1):
    """Two 0.1 m boxes with a 0.06 m x-overlap at z=0.3."""
    from unigym.props import build_prop_manager

    def crate(name: str, x: float):
        return {
            "name": name,
            "type": "box",
            "size": [0.1, 0.1, 0.1],
            "aabb": {
                "center": [0.0, 0.0, 0.05],
                "half_extents": [0.05, 0.05, 0.05],
            },
            "placement": {
                "mode": "fixed",
                "x_range": [x, x],
                "y_range": [0.0, 0.0],
                "z_range": [0.3, 0.3],
            },
        }

    return build_prop_manager(
        {"types": [crate(SENSOR_PROP, 0.5), crate(FILTER_PROP, 0.54)]},
        num_envs=num_envs,
        env_spacing=4.0,
        device="cuda:0",
        backend="isaaclab3_physx",
    )


def _crate_pair_manager():
    from unigym.contacts.manager import build_contact_manager

    return build_contact_manager(
        [
            {
                "name": "crate_pair",
                "sensor_refs": [{"owner": "object", "name": SENSOR_PROP}],
                "filter_refs": [{"owner": "object", "name": FILTER_PROP}],
                "frame": "world",
                "manifold": True,
            }
        ],
        backend="isaaclab3_physx",
        articulation_names=(ARTICULATION,),
        prop_names=[SENSOR_PROP, FILTER_PROP],
    )


def _step(env, steps: int = 3) -> None:
    from unigym.runtime.actions import NamedAction

    action = NamedAction.zeros(
        env.num_envs,
        ARTICULATION,
        ["slider_to_cart", "cart_to_pole"],
        "position",
        device=env.device,
    )
    for _ in range(steps):
        env.step((action,))


def test_physx_filtered_manifold_reports_crate_pair_contacts(env_factory) -> None:
    """An opted-in pairwise group returns body-named PhysX contact points."""
    manager = _crate_pair_manager()
    props = _two_crate_props()
    assert manager is not None
    assert props is not None
    props.require_contact_sensors(manager.referenced_prop_names())
    env = env_factory(1, contact_manager=manager, prop_manager=props)
    query = env.query

    assert query.capabilities.supports("contacts.manifold") == "partial"
    sensor = env.scene["contact_crate_pair_s0"]
    assert sensor.cfg.track_contact_points is True
    assert sensor.cfg.max_contact_data_count_per_prim == MANIFOLD_MAX_CONTACT_DATA_COUNT_PER_PRIM
    assert list(sensor.body_names) == [SENSOR_BODY]

    env.reset_all()
    _step(env)
    manifold = query.contacts.manifold("crate_pair")

    assert tuple(manifold.env_ids.tolist()) == (0,)
    assert manifold.group == "crate_pair"
    assert manifold.points, "expected at least one crate_a/crate_b PhysX contact"
    for point in manifold.points:
        assert point.env_id == 0
        assert (point.body1, point.body2) == (FILTER_BODY, SENSOR_BODY)
        assert point.geom1 is None and point.geom2 is None
        assert point.normal_w is None
        assert point.frame_w is None
        assert point.friction_force_w is None
        assert tuple(point.pos_w.shape) == (3,)
        assert bool(torch.isfinite(point.pos_w).all())
        # PhysX can report a near-contact point with a zero normal impulse; the
        # fixture is transient, so a finite non-negative force magnitude is the
        # honest bound (a > 0 assertion would be a flaky fixture property, not a
        # contract).
        assert math.isfinite(point.normal_force_n)
        assert point.normal_force_n >= 0.0
        assert math.isfinite(point.separation_m)


def test_physx_manifold_without_declared_groups_is_a_capability_error(cartpole_env) -> None:
    """A run without contacts: groups has no manifold source to report."""
    with pytest.raises(QueryCapabilityError, match="no contacts"):
        cartpole_env.query.contacts.manifold("crate_pair")
