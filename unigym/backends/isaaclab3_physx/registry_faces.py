"""Engine-owned registry faces for the IsaacLab 3 PhysX identity.

The shared ``isaaclab3_common`` package stays backend-neutral, so the
concrete identity (name, profile) is injected here.  The stdlib-only
registry imports this module lazily through its face factories, which keeps
``import unigym.backends.registry`` free of any engine import.
"""

from __future__ import annotations

from unigym.backends.shared.isaaclab3_common import build as shared_build
from unigym.backends.shared.isaaclab3_common.asset_scene.engine import AssetSceneEngine
from unigym.backends.shared.isaaclab3_common.cameras.rigs import build_isaaclab_rig
from unigym.backends.shared.isaaclab3_common.contacts import IsaacLabContactBackend
from unigym.backends.shared.isaaclab3_common.director.gripper import IsaacLabGripper
from unigym.backends.shared.isaaclab3_common.director_ops import IsaacLabDirectorOps
from unigym.backends.shared.isaaclab3_common.launcher import kit_launcher
from unigym.backends.shared.isaaclab3_common.model import provider_for
from unigym.backends.shared.isaaclab3_common.props import build_isaaclab_prop_backend
from unigym.backends.shared.isaaclab3_common.robots import realizer_for
from unigym.backends.shared.isaaclab3_common.scene import IsaacLabSceneOps

from .profile import PHYSX_PROFILE

NAME = "isaaclab3_physx"
PROFILE = PHYSX_PROFILE

env_kwargs = shared_build.env_kwargs
build_terrain = shared_build.build_terrain
build_events = shared_build.build_events


class PhysXContactBackend(IsaacLabContactBackend):
    """The PhysX identity of the shared IsaacLab contact backend."""

    backend_key = NAME


class PhysXDirectorOps(IsaacLabDirectorOps):
    """The PhysX identity of the shared IsaacLab director ops."""

    backend = NAME


class PhysXSceneOps(IsaacLabSceneOps):
    """The PhysX identity of the shared IsaacLab scene ops."""

    backend = NAME


def model_provider():
    """Return the model-provider class labelled with this identity."""
    return provider_for(NAME)


def scene_engine():
    """Construct this identity's asset-scene engine."""
    return AssetSceneEngine(NAME, ("usd", "urdf"))


def prop_backend():
    """Build this identity's shared prop backend face."""
    return build_isaaclab_prop_backend(NAME)


def camera_backend():
    """Return a rig builder wired to this identity's renderer profile."""
    renderer_cfg = PROFILE.renderer_cfg
    if renderer_cfg is None:
        raise RuntimeError(f"{NAME}: engine profile declares no renderer cfg")

    def build(spec, render_groups=None):
        return build_isaaclab_rig(
            spec,
            render_groups=render_groups,
            renderer_cfg=renderer_cfg(),
            depth_background=PROFILE.depth_background,
        )

    return build


def contact_backend():
    """Return this identity's contact backend class."""
    return PhysXContactBackend


def director_ops():
    """Return this identity's director-ops class."""
    return PhysXDirectorOps


def scene_ops():
    """Return this identity's scene-ops class."""
    return PhysXSceneOps


def gripper_backend():
    """Return this identity's director gripper class."""
    return IsaacLabGripper


def robot_realizer():
    """Return this identity's robot realizer closure."""
    return realizer_for(NAME)


__all__ = [
    "NAME",
    "PROFILE",
    "build_events",
    "build_terrain",
    "camera_backend",
    "contact_backend",
    "director_ops",
    "env_kwargs",
    "gripper_backend",
    "kit_launcher",
    "model_provider",
    "prop_backend",
    "robot_realizer",
    "scene_engine",
    "scene_ops",
]
