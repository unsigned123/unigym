"""Small real RobotProfile fixtures for tests that do not exercise robot packs."""

from __future__ import annotations

from unigym.assets.asset_ref import AssetRef
from unigym.robots.base import BodyBinding, RobotProfile


def fixture_profile(
    name: str = "fixture_arm",
    *,
    usd_path: str | None = None,
    base_height: float | None = None,
) -> RobotProfile:
    """Build a tiny typed profile with a body and hand face."""
    return RobotProfile(
        name=name,
        joint_sets={"body": ("shoulder", "elbow"), "hand": ("finger",)},
        default_joint_pos={"shoulder": 0.0, "elbow": 0.0, "finger": 0.0},
        assets=AssetRef(usd=usd_path) if usd_path is not None else None,
        action_faces={"hand": ("finger",)},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={
            "base": BodyBinding("base", {"mjcf": "base", "usd": "base"}),
        },
        base_height=base_height,
    )
