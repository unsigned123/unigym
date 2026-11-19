"""IsaacLab 3 virtual gripper: scene cfg field + the kinematic runtime hand.

A fixed-base KINEMATIC rigid cuboid
(``RigidBodyPropertiesCfg(kinematic_enabled=True)``) under
``{ENV_REGEX_NS}/Director_<name>``; move =
``write_root_link_pose_to_sim_index``.

The scene Cfg classes are resolved through the neutral props ``_resolve``
table (G4); isaaclab/torch imports stay function-local so this module is
import-safe without the kit.  The gripper cfg ``quat`` is repo-internal wxyz
and is converted with :func:`unigym.core.quat.to_isaaclab_cfg` before it
enters the IsaacLab 3 ``InitialStateCfg`` (xyzw); runtime reads reorder
``xyzw -> wxyz``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

GRIPPER_SCENE_FIELD = "director_gripper"
GRIPPER_PRIM_PREFIX = "Director_"


def isaaclab_scene_fields(gripper_cfg: dict, classes: dict | None = None) -> dict:
    """``{scene_field: RigidObjectCfg}`` for the kinematic hand (pre-clone).

    Kinematic = infinite-mass pose-driven body: pushes dynamics, ignores
    gravity, movable by root-pose writes every control step.
    """
    g = gripper_cfg
    if classes is None:
        from unigym.props.base import _resolve as _resolve_classes

        table = dict(_resolve_classes(None))
    else:
        table = classes
    if "CuboidCfg" not in table:
        from isaaclab.sim.spawners import CuboidCfg  # lazy (isaaclab zone)

        table = {**table, "CuboidCfg": CuboidCfg}
    # gripper cfg quat is repo-internal wxyz; the IsaacLab 3 cfg rot= is xyzw.
    from unigym.core.quat import to_isaaclab_cfg

    cfg = table["RigidObjectCfg"](
        prim_path=(f"{{ENV_REGEX_NS}}/{GRIPPER_PRIM_PREFIX}{g['name']}"),
        spawn=table["CuboidCfg"](
            size=[abs(float(v)) * 2.0 for v in g["size"]],
            rigid_props=table["RigidBodyPropertiesCfg"](kinematic_enabled=True),
        ),
        init_state=table["RigidObjectCfg"].InitialStateCfg(
            pos=tuple(float(v) for v in g["pos"]),
            rot=to_isaaclab_cfg(tuple(float(v) for v in g["quat"])),
        ),
        collision_group=0,
        debug_vis=False,
    )
    return {GRIPPER_SCENE_FIELD: cfg}


class IsaacLabGripper:
    """Runtime face of the IsaacLab 3 kinematic hand (pose-write only)."""

    def __init__(self, env, name: str):
        key = GRIPPER_SCENE_FIELD
        try:
            self._asset = env.scene[key]
        except (KeyError, AttributeError) as e:
            raise ValueError(
                f"director gripper asset {key!r} not in the scene (was the "
                f"run started with --direct and a director.gripper "
                f"section?)"
            ) from e
        self.name = name
        self.bid = None

    def get_pose(self, env) -> tuple:
        """Current kinematic body world pose ``(pos, quat wxyz)``."""
        import torch

        from unigym.core.quat import to_wxyz

        pos = _torch_view(self._asset.data.root_pos_w)[0].detach().cpu().numpy()
        # il3 data quat is xyzw; the director surface is wxyz.
        quat = (
            torch.as_tensor(to_wxyz(_torch_view(self._asset.data.root_quat_w)[0]))
            .detach()
            .cpu()
            .numpy()
        )
        return pos, quat

    def set_pose(self, env, pos=None, quat=None) -> None:
        """Write the root pose (``None`` keeps that component)."""
        import torch

        from unigym.core.quat import to_isaaclab_cfg

        cur_pos, cur_quat = self.get_pose(env)
        p = cur_pos if pos is None else np.asarray(pos, dtype=np.float32)
        q = cur_quat if quat is None else np.asarray(quat, dtype=np.float32)
        pose = torch.zeros(1, 7, device=env.device)
        pose[0, 0:3] = torch.tensor(np.asarray(p, dtype=np.float32))
        # repo-internal wxyz -> IsaacLab 3 write payload (xyzw).
        pose[0, 3:7] = torch.as_tensor(
            to_isaaclab_cfg(torch.tensor(np.asarray(q, dtype=np.float32)))
        )
        self._asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=[0])


def _torch_view(value) -> "torch.Tensor":
    """Explicit ``.torch`` payload of an IsaacLab 3 ``ProxyArray`` (else *value*)."""
    import torch

    payload = getattr(value, "torch", None)
    if payload is None:
        payload = value
    assert isinstance(payload, torch.Tensor)
    return payload
