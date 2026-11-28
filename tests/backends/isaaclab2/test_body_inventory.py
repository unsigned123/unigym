"""USD-authored robot body inventory on the IsaacLab 2 G1 test pack."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.sim,
    pytest.mark.usefixtures("simulation_app"),
    pytest.mark.requires_test_packs("g1"),
]

G1_AUTHORED_BODIES = 46
G1_COLLISION_PRIMS = 31
G1_BODIES_WITH_COLLISION = 24


def test_authored_inventory_lists_all_46_bodies(g1_usd: Path) -> None:
    """AUTHORED is the raw USD rigid-body list, including both palms."""
    from unigym.backends.isaaclab2.body_inventory import robot_body_inventory

    inventory = robot_body_inventory(g1_usd)

    assert len(inventory.bodies) == G1_AUTHORED_BODIES
    assert len(set(inventory.paths())) == G1_AUTHORED_BODIES

    left_palm = inventory.resolve("left_hand/left_hand_palm_link")
    right_palm = inventory.resolve("right_hand/right_hand_palm_link")
    assert left_palm.path == "left_hand/left_hand_palm_link"
    assert right_palm.path == "right_hand/right_hand_palm_link"
    assert inventory.contains("left_hand_palm_link")
    assert inventory.contains("right_hand_palm_link")


def test_nested_finger_links_are_authored_bodies(g1_usd: Path) -> None:
    """Depth-2 hand links are authored bodies, not synthetic descendants."""
    from unigym.backends.isaaclab2.body_inventory import robot_body_inventory

    inventory = robot_body_inventory(g1_usd)
    paths = set(inventory.paths())

    assert "left_hand/left_hand_index_0_link" in paths
    assert "right_hand/right_hand_index_0_link" in paths
    assert "left_hand/left_hand_thumb_2_link" in paths
    assert "right_hand/right_hand_thumb_2_link" in paths
    assert inventory.resolve("left_hand_index_0_link").path == ("left_hand/left_hand_index_0_link")


def test_collision_scan_traverses_instance_proxies(g1_usd: Path) -> None:
    """The authored collision flags see prims inside native instance prototypes."""
    from pxr import Usd, UsdPhysics

    from unigym.backends.isaaclab2.body_inventory import robot_body_inventory

    inventory = robot_body_inventory(g1_usd)
    assert sum(1 for body in inventory.bodies if body.collision) == G1_BODIES_WITH_COLLISION

    stage = Usd.Stage.Open(str(g1_usd))
    assert stage is not None
    root = stage.GetDefaultPrim()
    assert root
    collision_prims = [
        prim
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        or any("Collision" in schema for schema in prim.GetAppliedSchemas())
    ]
    assert len(collision_prims) == G1_COLLISION_PRIMS


def test_merged_into_is_diagnostic_metadata_only(g1_usd: Path) -> None:
    """A merged palm records its parent but stays in the authored inventory."""
    from unigym.backends.isaaclab2.body_inventory import robot_body_inventory

    inventory = robot_body_inventory(g1_usd)
    left_palm = inventory.resolve("left_hand/left_hand_palm_link")
    right_palm = inventory.resolve("right_hand/right_hand_palm_link")

    assert left_palm.merged_into == "left_wrist_yaw_link"
    assert right_palm.merged_into == "right_wrist_yaw_link"
    assert inventory.contains(left_palm.merged_into)
    assert inventory.contains(right_palm.merged_into)
    assert inventory.contains(left_palm.path)
    assert inventory.contains(right_palm.path)
    assert len(inventory.bodies) == G1_AUTHORED_BODIES


def test_inventory_exposes_no_static_simulated_population(g1_usd: Path) -> None:
    """SIMULATED is a live PhysX measurement for B2, never a USD-derived set."""
    import unigym.backends.isaaclab2.body_inventory as body_inventory

    inventory = body_inventory.robot_body_inventory(g1_usd)

    assert not hasattr(body_inventory, "BodyPopulation")
    assert not hasattr(body_inventory, "BodyPopulationView")
    assert not hasattr(inventory, "population")


def test_importing_body_inventory_does_not_import_pxr() -> None:
    """The module stays importable in the base environment without OpenUSD."""
    code = (
        "import sys; "
        "import unigym.backends.isaaclab2.body_inventory; "
        "print('pxr' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


def test_authored_inventory_is_the_only_authored_surface(g1_usd: Path) -> None:
    """The old ``robot_rigid_bodies`` compatibility alias is retired."""
    import importlib.util

    from unigym.backends.isaaclab2.body_inventory import robot_body_inventory

    inventory = robot_body_inventory(g1_usd)
    assert len(inventory.bodies) == G1_AUTHORED_BODIES
    paths = inventory.paths()
    assert "left_hand/left_hand_palm_link" in paths
    assert "right_hand/right_hand_palm_link" in paths

    assert importlib.util.find_spec("unigym.backends.isaaclab2.usd") is None


def test_duplicate_durable_paths_fail_with_both_names() -> None:
    """Two rows with the same durable path are rejected with both names."""
    from unigym.backends.isaaclab2.body_inventory import (
        RigidBody,
        RobotBodyInventory,
    )

    with pytest.raises(ValueError, match=r"duplicate path 'pelvis'.*'pelvis'.*'pelvis_copy'"):
        RobotBodyInventory(
            asset_path=Path("robot.usd"),
            bodies=(
                RigidBody(path="pelvis", name="pelvis", collision=True, merged_into=None),
                RigidBody(path="pelvis", name="pelvis_copy", collision=True, merged_into=None),
            ),
        )


def test_duplicate_leaf_names_are_allowed_but_name_lookup_is_ambiguous() -> None:
    """Leaf names may collide; the full path is the durable key."""
    from unigym.backends.isaaclab2.body_inventory import (
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

    assert inventory.names() == ("palm_link", "palm_link")
    assert inventory.resolve("left_hand/palm_link").name == "palm_link"
    assert inventory.resolve("right_hand/palm_link").name == "palm_link"
    with pytest.raises(
        ValueError,
        match=r"name 'palm_link' is ambiguous.*'left_hand/palm_link'.*'right_hand/palm_link'",
    ):
        inventory.resolve("palm_link")
