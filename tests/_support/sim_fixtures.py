"""Temporary pack fixtures for the simulation-facade contract tests.

The helper writes one real ``unigym.pack/1`` manifest plus one
``unigym.robot/1`` document under a caller-owned ``tmp_path`` root and returns
the read-only :class:`~unigym.packs.packset.PackSet` with the minted
:class:`~unigym.packs.packset.RobotRef`.  Names are derived from the unique
temporary root (plus an optional tag) so activation cannot collide with another
test's fixture in the process-wide robot registry.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from unigym.packs.packset import PackSet, RobotRef, discover_set

__all__ = ["write_sim_pack"]


def _token(root: Path, tag: str) -> str:
    """A lowercase pack/robot token unique to the temporary root and tag."""
    raw = f"{root.name}_{tag}" if tag else root.name
    token = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")
    if not token or not token[0].isalpha():
        token = f"p_{token}"
    return token[:48]


def write_sim_pack(
    root: Path,
    *,
    tag: str = "",
    version: str = "1.0.0",
    joint_sets: Mapping[str, Sequence[str]] | None = None,
    default_joint_pos: Mapping[str, float] | None = None,
) -> tuple[PackSet, RobotRef]:
    """Write one manifest + robot document and discover the pack set over it."""
    token = _token(root, tag)
    pack_name = f"pack_{token}"
    robot_id = f"robot.{token}"

    sets = (
        {"body": ("shoulder", "elbow"), "hand": ("finger",)}
        if joint_sets is None
        else {name: tuple(joints) for name, joints in joint_sets.items()}
    )
    defaults = (
        {"shoulder": 0.0, "elbow": 0.0, "finger": 0.0}
        if default_joint_pos is None
        else dict(default_joint_pos)
    )
    full_order: list[str] = []
    for joints in sets.values():
        for joint in joints:
            if joint not in full_order:
                full_order.append(joint)
    robot = {
        "schema": "unigym.robot/1",
        "name": robot_id,
        "joint_sets": {name: list(joints) for name, joints in sets.items()},
        "default_joint_pos": defaults,
        "base_body": "base",
        "foot_bodies": ["left_foot"],
        "body_sets": {"grasp.left": ["left_hand"]},
        "body_bindings": {
            "base": {"mjcf": "base", "usd": "/World/base"},
            "left_foot": {"mjcf": "left_foot", "usd": "/World/left_foot"},
            "left_hand": {"mjcf": "left_hand", "usd": "/World/left_hand"},
        },
        "action_faces": {},
        "full_joint_order": full_order,
    }
    manifest = {
        "schema": "unigym.pack/1",
        "name": pack_name,
        "version": version,
        "api": 1,
        "trust": "content",
        "provides": {
            "providers": [
                {
                    "id": robot_id,
                    "kind": "robot",
                    "resource": f"robots/{token}.yaml",
                }
            ]
        },
    }

    robot_path = root / "robots" / f"{token}.yaml"
    robot_path.parent.mkdir(parents=True, exist_ok=True)
    robot_path.write_text(yaml.safe_dump(robot, sort_keys=False), encoding="utf-8")
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )

    pack_set = discover_set([root])
    return pack_set, pack_set.pack(pack_name).robot(robot_id)
