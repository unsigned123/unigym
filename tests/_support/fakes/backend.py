"""A deterministic, pure-torch ``EnvBackend`` for conformance tests.

The fake is the external-simulator boundary double: it implements the same
name-addressed step/reset surface the real backends expose, with a one-joint
toy state, so contract and conformance tests can run without MuJoCo, IsaacLab
or any asset.  It is intentionally not a physics engine.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from unigym.env.backends.base import EnvBackend, EnvStep
from unigym.env.obs import ObservationConfig, ObservationRegistry, build_observation_schema
from unigym.runtime.actions import NamedAction
from unigym.runtime.controllers import ArticulationRuntime, ControlRuntime

ARTICULATION_NAME = "arm"
CONTROL_NAME = ARTICULATION_NAME
JOINT_NAME = "shoulder"


class FakeBackend(EnvBackend):
    """One-articulation, one-joint fake backend (CPU, deterministic).

    The articulation ``arm`` owns the joint ``shoulder``; the single body
    control row is also named ``arm`` (physical row key ``(arm, arm)``) with an
    explicit ``role == "body"``.  Action columns are joint-addressed.

    Every step replaces the scalar joint position with the named action's
    position target, publishes ``joint_pos``/``prev_action`` in the observation
    registry and the full record set, and advances the per-env step counter by
    one second.  There is no physics, contact, camera, prop or event model.
    ``joint_names`` defaults to the one-joint contract fixture; a caller that
    needs a multi-joint action group may declare more names (identity order).
    ``due_rows`` reports a configured backend-owned due mask on every step; it
    never resets a row inside ``step``.
    """

    def __init__(
        self,
        num_envs: int = 1,
        device: str = "cpu",
        joint_names: Sequence[str] = (JOINT_NAME,),
        *,
        due_rows: Sequence[int] = (),
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError(f"num_envs must be a positive int, got {num_envs!r}")
        requested = torch.device(device)
        if requested.type != "cpu":
            raise ValueError(f"FakeBackend is CPU-only, got device {device!r}")
        names = tuple(joint_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError(f"joint_names must be non-empty strings, got {joint_names!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"joint_names must be unique, got {joint_names!r}")
        due = tuple(due_rows)
        if any(isinstance(row, bool) or not isinstance(row, int) for row in due):
            raise ValueError(f"due_rows must be ints, got {due!r}")
        if any(row < 0 or row >= num_envs for row in due):
            raise ValueError(f"due_rows must be in [0, {num_envs}), got {due!r}")
        if len(set(due)) != len(due):
            raise ValueError(f"due_rows must be unique, got {due!r}")

        self.num_envs = num_envs
        self.device = "cpu"
        self.joint_names = names
        self.due_rows = due
        self._closed = False
        self._joint_pos = torch.zeros(num_envs, len(names), dtype=torch.float32, device=self.device)
        self._prev_action = torch.zeros_like(self._joint_pos)
        self._time_in_episode = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._episode_id = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self._command: torch.Tensor | None = None

        self.articulations: dict[str, ArticulationRuntime] = {
            ARTICULATION_NAME: ArticulationRuntime(
                name=ARTICULATION_NAME,
                root="fixed",
                joint_names=self.joint_names,
                q0=tuple(0.0 for _ in self.joint_names),
                foot_bodies=(),
            )
        }
        self.controls: dict[tuple[str, str], ControlRuntime] = {
            (ARTICULATION_NAME, CONTROL_NAME): ControlRuntime(
                name=CONTROL_NAME,
                adapter="manual",
                articulation=ARTICULATION_NAME,
                joint_names=self.joint_names,
                q0=tuple(0.0 for _ in self.joint_names),
                drive="position_pd",
                target="position",
                required=True,
                role="body",
            )
        }
        self._body_control = self.controls[(ARTICULATION_NAME, CONTROL_NAME)]
        joint_count = len(self._body_control.joint_names)

        self.perm_il2pol = torch.arange(joint_count, dtype=torch.int64)
        self.perm_pol2il = torch.argsort(self.perm_il2pol)
        self.policy_joint_pos_limits = (
            torch.full((num_envs, joint_count), -math.pi, dtype=torch.float32),
            torch.full((num_envs, joint_count), math.pi, dtype=torch.float32),
        )

        self.observations = ObservationConfig(
            robot=(ARTICULATION_NAME,), command=True, prev_action=True
        )
        self.obs = ObservationRegistry(
            self.observations,
            build_observation_schema(
                self.observations,
                full_joint_names={ARTICULATION_NAME: self.joint_names},
                policy_joint_names={ARTICULATION_NAME: self.joint_names},
            ),
            num_envs=num_envs,
            device=self.device,
        )
        self._t = 0
        self._refresh_obs()

    def step(
        self,
        actions: Sequence[NamedAction],
        raw_action: torch.Tensor | None = None,
        cmd: torch.Tensor | None = None,
    ) -> EnvStep:
        """Apply one named action, refresh the registry, return the step."""
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
        self._refresh_obs(raw_action, cmd)
        due = torch.tensor(self.due_rows, dtype=torch.int64, device=self.device)
        step = EnvStep(
            t=int(self._t),
            obs=self.obs.snapshot(),
            record=self.record(),
            reset_env_ids=due,
            should_reset=bool(due.numel()),
        )
        self._t += 1
        return step

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
        self._refresh_obs()
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

    def _resolve_targets(self, actions: Sequence[NamedAction]) -> torch.Tensor:
        if isinstance(actions, (str, bytes)) or not isinstance(actions, Sequence):
            raise TypeError(
                f"actions must be a sequence of NamedAction, got {type(actions).__name__}"
            )
        control = self._body_control
        owned = {joint: index for index, joint in enumerate(control.joint_names)}
        columns: dict[str, tuple[torch.Tensor, int]] = {}
        for entry in actions:
            if not isinstance(entry, NamedAction):
                raise TypeError(f"action entry must be a NamedAction, got {type(entry).__name__}")
            if entry.articulation != control.articulation:
                raise ValueError(
                    f"unknown action articulation {entry.articulation!r}; "
                    f"accepted: {[control.articulation]}"
                )
            if entry.kind != control.target:
                raise ValueError(
                    f"action joint kind {entry.kind!r} != control target {control.target!r}"
                )
            if entry.num_envs != self.num_envs:
                raise ValueError(f"action batch {entry.num_envs} != num_envs {self.num_envs}")
            for column, joint in enumerate(entry.joint_names):
                if joint not in owned:
                    raise ValueError(
                        f"unknown action joint {joint!r}; accepted: {list(control.joint_names)}"
                    )
                if joint in columns:
                    raise ValueError(
                        f"action joint {joint!r} is addressed more than once in one tick"
                    )
                columns[joint] = (entry.values, column)
        missing = sorted(joint for joint in owned if joint not in columns)
        if missing:
            raise ValueError(f"missing action joints {missing}; accepted: {list(owned)}")
        ordered = [columns[joint][0][:, columns[joint][1]] for joint in control.joint_names]
        return torch.stack(ordered, dim=1).to(device=self.device, dtype=torch.float32).clone()

    def _refresh_obs(self, raw_action=None, cmd=None) -> None:
        self.obs.t = int(self._t)
        prefix_name = ARTICULATION_NAME
        zeros3 = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.obs.set_robot(prefix_name, "joint_pos", self._joint_pos.clone())
        self.obs.set_robot(prefix_name, "joint_pos_rel", self._joint_pos.clone())
        self.obs.set_robot(prefix_name, "joint_vel", torch.zeros_like(self._joint_pos))
        self.obs.set_robot(prefix_name, "applied_torque", torch.zeros_like(self._joint_pos))
        self.obs.set_robot(
            prefix_name,
            "gravity_b",
            torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)
            .expand(self.num_envs, 3)
            .clone(),
        )
        self.obs.set_robot(prefix_name, "base_pos_w", zeros3.clone())
        self.obs.set_robot(
            prefix_name,
            "base_quat_w",
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
            .expand(self.num_envs, 4)
            .clone(),
        )
        for leaf in (
            "base_lin_vel_w",
            "base_ang_vel_w",
            "base_lin_vel_b",
            "base_ang_vel_b",
        ):
            self.obs.set_robot(prefix_name, leaf, zeros3.clone())
        self.obs.set_robot(
            prefix_name,
            "base_yaw",
            torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device),
        )
        self.obs.set_command(cmd if cmd is not None else self._command)
        self.obs.set_prev_action(raw_action if raw_action is not None else self._prev_action)

    def record(self) -> dict[str, object]:
        record: dict[str, object] = dict(self.obs.record())
        record["time_in_episode"] = self._time_in_episode.clone()
        record["episode_id"] = self._episode_id.clone()
        return record


__all__ = [
    "ARTICULATION_NAME",
    "CONTROL_NAME",
    "FakeBackend",
    "JOINT_NAME",
]
