"""A real ``contacts: all`` world group contacts, reads and resets on PhysX."""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

ARTICULATION = "cartpole"
CONTACT_BODY = "pole"


def _raised_ground():
    """Ground whose top surface is inside the fixed Cartpole rail.

    The stock Cartpole rail is fixed at z=2 m and never contacts the default
    ground.  Raising the ground (box top at z=2.10) makes the pole's first
    physics step a real ground contact, so the test exercises the sensor's force
    pipeline rather than only its binding shape.  The reset teleport is clean:
    the sensor buffers are zero until the next physics step.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    return {
        "ground": AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.CuboidCfg(
                size=(10.0, 10.0, 0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 2.05)),
        )
    }


def test_contact_all_world_group_reads_world_forces(cartpole_usd, env_factory):
    """A robot ``body: "*"`` world group reports a world-frame normal force."""
    from unigym.backends.isaaclab3_physx.env import Isaaclab3PhysxEnv
    from unigym.backends.registry import ISAACLAB3_PHYSX_TRAITS, Trait
    from unigym.backends.shared.isaaclab3_common.usd import robot_rigid_bodies
    from unigym.contacts.manager import build_contact_manager
    from unigym.runtime.actions import NamedAction

    assert Trait.CONTACTS_GROUPS in ISAACLAB3_PHYSX_TRAITS
    assert "CONTACTS_GROUPS" in {
        trait.name for trait in Isaaclab3PhysxEnv.BACKEND_CAPABILITIES.traits()
    }

    manager = build_contact_manager(
        [
            {
                "name": "all",
                "sensor_refs": [{"owner": "robot", "articulation": ARTICULATION, "body": "*"}],
                "frame": "world",
            }
        ],
        backend="isaaclab3_physx",
        articulation_names=(ARTICULATION,),
    )
    assert manager.backend.backend_key == "isaaclab3_physx"

    env = env_factory(1, contact_manager=manager, terrain_cfg=_raised_ground())
    body_paths = [body.path for body in robot_rigid_bodies(str(cartpole_usd))]
    assert manager.backend.group_axes(manager.specs[0]) == (len(body_paths), 0)

    sensor = env.scene["contact_all_s0"]
    assert set(sensor.body_names) == {path.rsplit("/", 1)[-1] for path in body_paths}
    assert sensor.cfg.prim_path.endswith("/(.*)")
    contact_index = list(sensor.body_names).index(CONTACT_BODY)

    assert env.reset_all().tolist() == [0]
    after_reset = manager.read()["contact/all"]
    assert tuple(after_reset.shape) == (1, len(body_paths), 3)
    assert not bool(after_reset.abs().sum(dim=-1).any())

    env.step(
        (
            NamedAction.zeros(
                env.num_envs,
                ARTICULATION,
                ["slider_to_cart", "cart_to_pole"],
                "position",
                device=env.device,
            ),
        )
    )
    forces = manager.read()["contact/all"]
    assert tuple(forces.shape) == (env.num_envs, len(body_paths), 3)
    assert bool(torch.isfinite(forces).all())

    normal = forces[0, contact_index]
    assert normal[2] > 0.0
    assert normal[:2].abs().max() < 1e-3
    assert not bool(forces[0, :contact_index].abs().any())
    assert not bool(forces[0, contact_index + 1 :].abs().any())

    assert env.reset_all().tolist() == [0]
    cleared = manager.read()["contact/all"]
    assert tuple(cleared.shape) == (1, len(body_paths), 3)
    assert not bool(cleared.abs().sum(dim=-1).any())
