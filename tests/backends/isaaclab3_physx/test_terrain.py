"""Real flat and mesh terrain presets on the PhysX lane."""

from __future__ import annotations

import pytest

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

ARTICULATION = "cartpole"
JOINT_ORDER = ("slider_to_cart", "cart_to_pole")
ENV_SPACING = 4.0


def test_flat_terrain_preset_builds_the_ground_entity(env_factory):
    """The flat preset's ``ground`` entity is part of the constructed scene."""
    from unigym.capabilities.traits import ISAACLAB3_PHYSX_TRAITS, Trait
    from unigym.backends.shared.isaaclab3_common.terrains import build_terrain

    assert Trait.TERRAIN in ISAACLAB3_PHYSX_TRAITS

    terrain_cfg = build_terrain(
        {"name": "flat", "params": {"friction": 0.7}},
        num_envs=2,
        env_spacing=ENV_SPACING,
    )
    assert set(terrain_cfg) == {"ground"}

    env = env_factory(2, terrain_cfg=terrain_cfg)
    assert env.num_envs == 2
    assert env.scene["ground"] is not None

    from unigym.runtime.actions import NamedAction

    step = env.step(
        (
            NamedAction.zeros(
                env.num_envs,
                ARTICULATION,
                list(JOINT_ORDER),
                "position",
                device=env.device,
            ),
        )
    )
    assert step.t == 0


def test_steps_terrain_preset_builds_the_terrain_entity_and_steps(env_factory):
    """The steps mesh preset's ``terrain`` entity builds and the env steps once."""
    from unigym.capabilities.traits import ISAACLAB3_PHYSX_TRAITS, Trait
    from unigym.backends.shared.isaaclab3_common.terrains import build_terrain
    from unigym.runtime.actions import NamedAction

    assert Trait.TERRAIN in ISAACLAB3_PHYSX_TRAITS

    terrain_cfg = build_terrain(
        {
            "name": "steps",
            "params": {
                "num_steps": 2,
                "step_width": 0.25,
                "step_height": 0.05,
                "platform_width": 1.0,
            },
        },
        num_envs=1,
        env_spacing=ENV_SPACING,
    )
    assert set(terrain_cfg) == {"terrain"}

    env = env_factory(1, terrain_cfg=terrain_cfg)
    assert env.scene["terrain"] is not None

    step = env.step(
        (
            NamedAction.zeros(
                env.num_envs,
                ARTICULATION,
                list(JOINT_ORDER),
                "position",
                device=env.device,
            ),
        )
    )
    assert step.t == 0
    assert env.num_envs == 1
