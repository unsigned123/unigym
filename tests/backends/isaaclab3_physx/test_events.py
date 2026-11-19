"""A reset-mode CoM DR event applied at construction on the real PhysX lane."""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

POLE = "pole"
COM_OFFSET = 0.02


def _com_events(offset: float):
    """The yaml events list for an absolute pole-CoM x offset."""
    from unigym.backends.shared.isaaclab3_common.build import build_events

    return build_events(
        [
            {
                "func": "randomize_rigid_body_com_abs",
                "mode": "reset",
                "params": {
                    "asset_cfg": {"name": "robot", "body_names": [POLE]},
                    "com_range": {
                        "x": [offset, offset],
                        "y": [0.0, 0.0],
                        "z": [0.0, 0.0],
                    },
                },
            }
        ]
    )


def _pole_com(env) -> torch.Tensor:
    """Read the pole body CoM through the env's engine profile."""
    asset = env.scene["robot"]
    body_id = list(asset.body_names).index(POLE)
    profile = env.engine_profile
    assert profile is not None
    return torch.as_tensor(profile.get_body_com(asset)[0, body_id], dtype=torch.float32).clone()


def test_reset_com_event_applies_at_construction(env_factory):
    """Two reset-mode CoM ranges shift the pole CoM by their configured delta."""
    from unigym.backends.isaaclab3_physx.env import Isaaclab3PhysxEnv
    from unigym.backends.registry import ISAACLAB3_PHYSX_TRAITS, Trait

    assert Trait.EVENTS_DR in ISAACLAB3_PHYSX_TRAITS
    assert "EVENTS_DR" in {trait.name for trait in Isaaclab3PhysxEnv.BACKEND_CAPABILITIES.traits()}
    assert next(iter(_com_events(COM_OFFSET).values())).mode == "reset"

    plus_env = env_factory(1, events_cfg=_com_events(COM_OFFSET))
    plus_com = _pole_com(plus_env)
    # Close before the next construction: a build in the same process needs a
    # clean SimulationContext (see Isaaclab3EnvBase.close).
    plus_env.close()

    minus_env = env_factory(1, events_cfg=_com_events(-COM_OFFSET))
    minus_com = _pole_com(minus_env)

    expected = torch.zeros_like(plus_com)
    expected[0] = 2.0 * COM_OFFSET
    torch.testing.assert_close(
        plus_com - minus_com,
        expected,
        atol=1e-6,
        rtol=0,
    )
