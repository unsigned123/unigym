"""A deterministic, pure-torch ``EnvBackend`` for conformance tests.

The fake is the external-simulator boundary double: it implements the same
name-addressed step/reset surface the real backends expose, with a one-joint
toy state, so contract and conformance tests can run without MuJoCo, IsaacLab
or any asset.  It is intentionally not a physics engine.
"""

from __future__ import annotations

import math

import torch

from unigym.env.backends.base import EnvBackend
from unigym.runtime.actions import NamedAction
from unigym.runtime.controllers import ArticulationRuntime, ControlRuntime

ARTICULATION_NAME = "arm"
CONTROL_NAME = "body"
JOINT_NAME = "shoulder"


class FakeBackend(EnvBackend):
    """One-articulation, one-joint fake backend (CPU, deterministic).

    The articulation ``arm`` owns the joint ``shoulder``; the single body
    control group ``body`` drives it.  The action face is the articulation
    name (``arm``), because the ``EnvBackend`` body-face rule requires
    ``action_face == articulation``.

    Every step replaces the scalar joint position with the named action's
    position target, publishes ``joint_pos``/``prev_action`` in the observation
    cache and the full extra-info record set, and advances the episode clock by
    one second.  There is no physics, contact, camera, prop or event model.
    """

    def __init__(self, num_envs: int = 1, device: str = "cpu") -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError(f"num_envs must be a positive int, got {num_envs!r}")
        requested = torch.device(device)
        if requested.type != "cpu":
            raise ValueError(f"FakeBackend is CPU-only, got device {device!r}")

        self.num_envs = num_envs
        self.device = "cpu"
        self._closed = False
        self._joint_pos = torch.zeros(num_envs, 1, dtype=torch.float32, device=self.device)
        self._prev_action = torch.zeros_like(self._joint_pos)
        self._time_in_episode = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._episode_id = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._command: torch.Tensor | None = None

        self.articulations: dict[str, ArticulationRuntime] = {
            ARTICULATION_NAME: ArticulationRuntime(
                name=ARTICULATION_NAME,
                root="fixed",
                joint_names=(JOINT_NAME,),
                q0=(0.0,),
                foot_bodies=(),
            )
        }
        self.controls: dict[str, ControlRuntime] = {
            CONTROL_NAME: ControlRuntime(
                name=CONTROL_NAME,
                action_face=ARTICULATION_NAME,
                adapter="manual",
                articulation=ARTICULATION_NAME,
                joint_names=(JOINT_NAME,),
                q0=(0.0,),
                drive="position_pd",
                target="position",
                required=True,
            )
        }
        self._body_control = self.controls[CONTROL_NAME]
        joint_count = len(self._body_control.joint_names)

        self.perm_il2pol = torch.arange(joint_count, dtype=torch.int64)
        self.perm_pol2il = torch.argsort(self.perm_il2pol)
        self.policy_joint_pos_limits = (
            torch.full((num_envs, joint_count), -math.pi, dtype=torch.float32),
            torch.full((num_envs, joint_count), math.pi, dtype=torch.float32),
        )

        self.obs_cache: dict[str, torch.Tensor] = {}
        self._refresh_obs_cache()

    def step(
        self,
        actions: dict[str, NamedAction],
        raw_action: torch.Tensor | None = None,
        cmd: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | bool]]:
        """Apply one named action, refresh the cache, return ``(obs, extra)``."""
        self._ensure_open()
        self._joint_pos = self._resolve_targets(actions)
        if raw_action is not None:
            if not isinstance(raw_action, torch.Tensor):
                raise TypeError(
                    f"raw_action must be a torch tensor, got {type(raw_action).__name__}"
                )
            if tuple(raw_action.shape) != tuple(self._joint_pos.shape):
                raise ValueError(
                    f"raw_action shape {tuple(raw_action.shape)} != "
                    f"{tuple(self._joint_pos.shape)}"
                )
            self._prev_action = raw_action.to(device=self.device, dtype=torch.float32).clone()
        if cmd is not None:
            if not isinstance(cmd, torch.Tensor) or cmd.ndim == 0 or cmd.shape[0] != self.num_envs:
                raise ValueError(f"cmd must be a tensor with leading dim {self.num_envs}")
            self._command = cmd.to(device=self.device).clone()
        self._time_in_episode += 1.0
        self._refresh_obs_cache()
        return dict(self.obs_cache), self._build_extra()

    def reset_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Reset only the selected rows to the spawn state; return the ids."""
        self._ensure_open()
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids)
        ids = ids.to(device=self.device, dtype=torch.int64).reshape(-1)
        if ids.numel() == 0:
            return ids
        if bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError(
                f"FakeBackend num_envs={self.num_envs}, cannot reset env ids {ids.tolist()}"
            )
        unique = torch.unique(ids)
        self._joint_pos[unique] = 0.0
        self._prev_action[unique] = 0.0
        self._time_in_episode[unique] = 0.0
        self._episode_id[unique] += 1
        self._refresh_obs_cache()
        return ids

    def reset_all(self) -> torch.Tensor:
        """Reset every env row."""
        return self.reset_envs(torch.arange(self.num_envs, dtype=torch.int64))

    def to_policy_order(self, x_il: torch.Tensor) -> torch.Tensor:
        """Gather interaction-order joints into policy order (identity here)."""
        return x_il[:, self.perm_il2pol]

    def to_il_order(self, x_pol: torch.Tensor) -> torch.Tensor:
        """Gather policy-order joints into interaction order (identity here)."""
        return x_pol[:, self.perm_pol2il]

    def close(self) -> None:
        """Release the fake resources (idempotent)."""
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("FakeBackend is closed")

    def _resolve_targets(self, actions: dict[str, NamedAction]) -> torch.Tensor:
        if not isinstance(actions, dict):
            raise TypeError(
                f"actions must be a dict[str, NamedAction], got {type(actions).__name__}"
            )
        control = self._body_control
        accepted = {control.action_face}
        unknown = sorted(set(actions) - accepted)
        if unknown:
            raise ValueError(f"unknown action face(s) {unknown}; accepted: {sorted(accepted)}")
        if control.action_face not in actions:
            raise ValueError(
                f"missing required action face {control.action_face!r}; "
                f"accepted: {sorted(accepted)}"
            )
        entry = actions[control.action_face]
        if not isinstance(entry, NamedAction):
            raise TypeError(f"action entry must be a NamedAction, got {type(entry).__name__}")
        if entry.kind != control.target:
            raise ValueError(
                f"action face {control.action_face!r} kind {entry.kind!r} != "
                f"control target {control.target!r}"
            )
        if entry.num_envs != self.num_envs:
            raise ValueError(f"action batch {entry.num_envs} != num_envs {self.num_envs}")

        provided = list(entry.joint_names)
        unknown_joints = sorted(set(provided) - set(control.joint_names))
        missing_joints = sorted(set(control.joint_names) - set(provided))
        if unknown_joints or missing_joints:
            raise ValueError(
                f"action face {control.action_face!r} joint coverage mismatch; "
                f"unknown={unknown_joints}, missing={missing_joints}"
            )
        column = {name: index for index, name in enumerate(provided)}
        columns = [column[name] for name in control.joint_names]
        return entry.values[:, columns].to(device=self.device, dtype=torch.float32).clone()

    def _refresh_obs_cache(self) -> None:
        self.obs_cache["joint_pos"] = self._joint_pos.clone()
        self.obs_cache["prev_action"] = self._prev_action.clone()
        if self._command is not None:
            self.obs_cache["command"] = self._command.clone()

    def _build_extra(self) -> dict[str, torch.Tensor | bool]:
        prefix = f"articulation/{ARTICULATION_NAME}"
        zeros3 = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        return {
            f"{prefix}/joint_pos": self._joint_pos.clone(),
            f"{prefix}/joint_vel": torch.zeros_like(self._joint_pos),
            f"{prefix}/applied_torque": torch.zeros_like(self._joint_pos),
            f"{prefix}/gravity_b": torch.tensor(
                [0.0, 0.0, -1.0], dtype=torch.float32, device=self.device
            )
            .expand(self.num_envs, 3)
            .clone(),
            f"{prefix}/base_pos_w": zeros3.clone(),
            f"{prefix}/base_quat_w": torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device
            )
            .expand(self.num_envs, 4)
            .clone(),
            f"{prefix}/base_lin_vel_w": zeros3.clone(),
            f"{prefix}/base_ang_vel_w": zeros3.clone(),
            f"{prefix}/base_lin_vel_b": zeros3.clone(),
            f"{prefix}/base_ang_vel_b": zeros3.clone(),
            "time_in_episode": self._time_in_episode.clone(),
            "episode_id": self._episode_id.clone(),
            "reset_env_ids": torch.empty(0, dtype=torch.int64, device=self.device),
            "should_reset": False,
        }


__all__ = [
    "ARTICULATION_NAME",
    "CONTROL_NAME",
    "FakeBackend",
    "JOINT_NAME",
]
