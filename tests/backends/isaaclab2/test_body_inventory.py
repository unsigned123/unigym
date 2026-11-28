"""USD-derived robot body inventory populations on the IsaacLab 2 dog lane."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("g1"),
]


def test_authored_population_keeps_fixed_merged_palm_links(g1_usd: Path) -> None:
    """AUTHORED is the raw USD rigid-body list, including merged palms."""
    from unigym.backends.isaaclab2.body_inventory import BodyPopulation, robot_body_inventory

    inventory = robot_body_inventory(g1_usd)
    authored = inventory.population(BodyPopulation.AUTHORED)

    left_palm = authored.resolve("left_hand_palm_link")
    right_palm = authored.resolve("right_hand_palm_link")
    assert left_palm.path == "left_hand/left_hand_palm_link"
    assert right_palm.path == "right_hand/right_hand_palm_link"
    assert authored.contains("left_hand_palm_link")
    assert authored.contains("right_hand_palm_link")


def test_simulated_population_excludes_fixed_merged_palm_links(g1_usd: Path) -> None:
    """SIMULATED is the PhysX body universe without the merged palms."""
    from unigym.backends.isaaclab2.body_inventory import BodyPopulation, robot_body_inventory

    inventory = robot_body_inventory(g1_usd)
    simulated = inventory.population(BodyPopulation.SIMULATED)
    simulated_paths = {body.path for body in simulated.bodies}

    assert not simulated.contains("left_hand_palm_link")
    assert not simulated.contains("right_hand_palm_link")
    assert "left_hand/left_hand_palm_link" not in simulated_paths
    assert "right_hand/right_hand_palm_link" not in simulated_paths
    assert "left_hand/left_hand_index_0_link" in simulated_paths
    assert simulated.contains("left_hand_index_0_link")


def test_palm_merged_into_names_the_wrist_parent(g1_usd: Path) -> None:
    """A merged palm still records the robot-relative parent it was merged into."""
    from unigym.backends.isaaclab2.body_inventory import BodyPopulation, robot_body_inventory

    inventory = robot_body_inventory(g1_usd)
    authored = inventory.population(BodyPopulation.AUTHORED)
    simulated_paths = {body.path for body in inventory.population(BodyPopulation.SIMULATED).bodies}

    left_palm = authored.resolve("left_hand_palm_link")
    right_palm = authored.resolve("right_hand_palm_link")
    assert left_palm.merged_into == "left_wrist_yaw_link"
    assert right_palm.merged_into == "right_wrist_yaw_link"
    assert left_palm.merged_into in simulated_paths
    assert right_palm.merged_into in simulated_paths


def test_robot_rigid_bodies_wrapper_matches_the_simulated_population(g1_usd: Path) -> None:
    """The compatibility wrapper preserves the simulated population slots."""
    from unigym.backends.isaaclab2.body_inventory import BodyPopulation, robot_body_inventory
    from unigym.backends.isaaclab2.usd import robot_rigid_bodies

    view = robot_body_inventory(g1_usd).population(BodyPopulation.SIMULATED)
    legacy = robot_rigid_bodies(str(g1_usd))

    assert [(body.path, body.collision) for body in legacy] == [
        (body.path, body.collision) for body in view.bodies
    ]


def test_duplicate_resolved_names_fail_with_both_paths() -> None:
    """A population rejects two bodies resolving to the same USD prim name."""
    from unigym.backends.isaaclab2.body_inventory import (
        BodyPopulation,
        RigidBody,
        RobotBodyInventory,
    )

    inventory = RobotBodyInventory(
        asset_path=Path("robot.usd"),
        bodies=(
            RigidBody(
                path="left_hand/palm_link", name="palm_link", collision=True, merged_into=None
            ),
            RigidBody(
                path="right_hand/palm_link", name="palm_link", collision=True, merged_into=None
            ),
        ),
    )

    with pytest.raises(ValueError, match=r"duplicate name 'palm_link'.*left_hand.*right_hand"):
        inventory.population(BodyPopulation.AUTHORED)


def test_duplicate_names_are_checked_per_population() -> None:
    """A merged duplicate stays visible in AUTHORED but not in SIMULATED."""
    from unigym.backends.isaaclab2.body_inventory import (
        BodyPopulation,
        RigidBody,
        RobotBodyInventory,
    )

    inventory = RobotBodyInventory(
        asset_path=Path("robot.usd"),
        bodies=(
            RigidBody(path="palm_link", name="palm_link", collision=True, merged_into=None),
            RigidBody(
                path="left_hand/palm_link",
                name="palm_link",
                collision=True,
                merged_into="palm_link",
            ),
        ),
    )

    with pytest.raises(ValueError, match=r"duplicate name 'palm_link'"):
        inventory.population(BodyPopulation.AUTHORED)

    simulated = inventory.population(BodyPopulation.SIMULATED)
    assert simulated.names() == ("palm_link",)
    assert simulated.resolve("palm_link").path == "palm_link"
