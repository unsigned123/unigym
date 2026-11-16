"""Gymnasium facades over one run.

Two distinct Gym semantics, never conflated:

* :class:`GymEnvRunner` -- the classic single-env ``gym.Env`` contract
  (``num_envs == 1``).  It runs with ``auto_reset=False``: a done env stays
  in its terminal state until the caller calls :meth:`reset`, and ``step``
  after a done raises (Gymnasium 1.0 semantics).  The returned ``obs`` is
  the PRE-reset observation; the post-reset observation is only returned by
  :meth:`reset`.
* :class:`GymVectorRunner` -- the batched ``gym.vector.VectorEnv`` contract
  (``num_envs >= 1``) with SameStep autoreset: the engine's reset chain runs
  inside the step that reports ``terminated/truncated``, ``obs`` is the
  POST-reset batch, and the pre-reset values of the done envs ride
  ``infos["final_observation"]`` / ``infos["final_info"]``.

Both borrow the :class:`~unigym.runner.sync.SyncRunner` tick verbatim:
termination, sink callbacks, reset chain and RNG order are unchanged; only
the action plumbing and the Gymnasium return shapes differ.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch

from unigym.runner.state import read_extra
from unigym.runner.sync import (
    SyncRunner,
    _adapter_joint_targets,
    _instance_action_groups,
    _instance_control_dt,
)
from unigym.runtime.actions import NamedAction


def _as_numpy(value):
    """Detach/convert one extra value to a numpy array (no copies lost)."""
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _batch_infos(extra: dict, num_envs: int) -> dict:
    """Per-env ``(N, ...)`` arrays from one batched extra dict.

    ``reset_env_ids`` is an id vector (not an env-indexed field) and is
    dropped; scalar fields (e.g. ``should_reset``) are broadcast to ``(N,)``.
    """
    infos: dict = {}
    for key, value in extra.items():
        if key == "reset_env_ids":
            continue
        arr = _as_numpy(value)
        if arr.ndim == 0:
            arr = np.full((num_envs,), arr.item())
        elif arr.shape[0] != num_envs:
            continue
        infos[key] = arr
    return infos


def _per_env_values(values: Mapping[str, object], env_id: int) -> dict:
    """One env's slice of a batched dict (pre-reset final_* values)."""
    out: dict = {}
    for key, value in values.items():
        if torch.is_tensor(value):
            out[key] = value[env_id].detach().cpu().numpy().copy()
        else:
            arr = _as_numpy(value)
            out[key] = arr[env_id].copy() if arr.ndim else arr.item()
    return out


def _reject_mapping_action(action) -> None:
    """Fail loud on the removed mapping/dict action form.

    Actions are a sequence of self-describing NamedAction entries; a mapping
    would silently lose the articulation identity and must never fall through
    to ``action_fn`` or be treated as a per-env value.
    """
    if isinstance(action, Mapping):
        raise TypeError(
            "received a mapping action, but the action contract is a sequence "
            "of joint-addressed NamedAction entries; build "
            "NamedAction(articulation=..., values=..., joint_names=..., "
            "kind=...) instead of a dict"
        )


class _GymBase(SyncRunner):
    """Shared best-effort post-reset 17-key extra reader."""

    def _extra_now(self) -> dict[str, object]:
        """Best-effort post-reset 17-key extra (the shared seam reader)."""
        return read_extra(self.env)


class GymEnvRunner(_GymBase):
    """Classic single-env ``gym.Env`` facade (``num_envs == 1``)."""

    @classmethod
    def from_instance(
        cls,
        instance,
        reward_fn: Callable | None = None,
        action_fn: Callable | None = None,
        max_steps: int | None = None,
    ) -> GymEnvRunner:
        """Build the single-env Gym facade from a live RunInstance."""
        return cls(instance, reward_fn=reward_fn, max_steps=max_steps, action_fn=action_fn)

    def __init__(
        self,
        instance,
        reward_fn: Callable | None = None,
        max_steps: int | None = None,
        action_fn: Callable | None = None,
    ):
        num_envs = int(getattr(instance.env, "num_envs", 0))
        if num_envs != 1:
            raise ValueError(
                f"GymEnvRunner requires exactly one env (num_envs=1), got "
                f"{num_envs}; use GymVectorRunner for batched runs"
            )
        spec = instance.spec
        runner_cfg = spec.runner or {}
        super().__init__(
            spec.cfg,
            instance.profile,
            instance.adapter,
            instance.sampler,
            instance.term_group,
            instance.env,
            instance.sink,
            reset_on_termination=instance.reset_on_termination,
            reset_mask_fn=runner_cfg.get("reset_mask_fn"),
            replay=instance.replay,
            auto_reset=False,
            adapter_joint_targets=_adapter_joint_targets(instance),
            action_groups=_instance_action_groups(instance),
            control_dt=_instance_control_dt(instance),
        )
        self.instance = instance
        self.reward_fn = reward_fn
        self.action_fn = action_fn
        self.max_steps = None if max_steps is None else int(max_steps)
        self._steps = 0
        self._finished = False

    def reset(self, seed=None, options=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Reset all envs; return ``(post_reset_obs, 17_key_info)``."""
        super().reset(None)
        self._steps = 0
        self._finished = False
        obs = dict(self.env.obs_cache)
        return obs, self._extra_now()

    def step(self, action):
        """Advance one tick; return the Gymnasium 5-tuple.

        ``obs`` is the PRE-reset observation and ``info`` the PRE-reset
        extra.  A done episode must be followed by :meth:`reset`.
        """
        if self._finished:
            raise RuntimeError("episode finished; call reset()")
        _reject_mapping_action(action)
        if isinstance(action, (list, tuple)) and (not action or isinstance(action[0], NamedAction)):
            actions = action
        elif self.action_fn is not None:
            actions = self.action_fn(action)
        else:
            raise TypeError(
                "GymEnvRunner.step expects a sequence of NamedAction (or pass "
                f"an action_fn); got {type(action).__name__}"
            )

        result = self.tick(actions_override=actions)
        self._steps += 1
        obs = result.obs
        extra = result.extra
        reward = float(self.reward_fn(obs, extra)) if self.reward_fn is not None else 0.0
        terminated = bool(result.terminated.any()) if result.terminated is not None else False
        due = bool(result.reset_env_ids.numel())
        step_truncated = self.max_steps is not None and self._steps >= self.max_steps
        truncated = (due or step_truncated) and not terminated
        self._finished = bool(terminated or truncated)
        return obs, reward, terminated, truncated, dict(extra)


class GymVectorRunner(_GymBase):
    """Batched ``gym.vector.VectorEnv`` facade (``num_envs >= 1``).

    SameStep autoreset is the declared semantics: the engine reset chain
    runs inside the step that reports the done envs, so ``obs`` is the
    POST-reset batch while the pre-reset values of the done envs are carried
    in ``infos["final_observation"]`` / ``infos["final_info"]``.
    """

    @classmethod
    def from_instance(
        cls, instance, reward_fn: Callable | None = None, action_fn: Callable | None = None
    ) -> GymVectorRunner:
        """Build the batched Gym facade from a live RunInstance."""
        return cls(instance, reward_fn=reward_fn, action_fn=action_fn)

    def __init__(
        self, instance, reward_fn: Callable | None = None, action_fn: Callable | None = None
    ):
        num_envs = int(getattr(instance.env, "num_envs", 0))
        if num_envs < 1:
            raise ValueError(f"GymVectorRunner requires num_envs >= 1, got {num_envs}")
        spec = instance.spec
        runner_cfg = spec.runner or {}
        super().__init__(
            spec.cfg,
            instance.profile,
            instance.adapter,
            instance.sampler,
            instance.term_group,
            instance.env,
            instance.sink,
            reset_on_termination=instance.reset_on_termination,
            reset_mask_fn=runner_cfg.get("reset_mask_fn"),
            replay=instance.replay,
            auto_reset=True,
            adapter_joint_targets=_adapter_joint_targets(instance),
            action_groups=_instance_action_groups(instance),
            control_dt=_instance_control_dt(instance),
        )
        self.instance = instance
        self.reward_fn = reward_fn
        self.action_fn = action_fn
        self.num_envs = num_envs

    def _vector_actions(self, actions):
        """Named-action sequence, or ``action_fn`` applied once per env."""
        _reject_mapping_action(actions)
        if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
            if not actions or isinstance(actions[0], NamedAction):
                return actions
        if self.action_fn is None:
            raise TypeError(
                "GymVectorRunner.step expects a sequence of NamedAction (or "
                f"pass an action_fn); got {type(actions).__name__}"
            )
        if isinstance(actions, (list, tuple)) and len(actions) == self.num_envs:
            per_env = [self.action_fn(a) for a in actions]
        else:
            per_env = [self.action_fn(actions) for _ in range(self.num_envs)]
        merged: dict[tuple[str, str, frozenset[str]], list[NamedAction]] = {}
        for entries in per_env:
            for entry in entries:
                merged.setdefault(
                    (entry.articulation, entry.kind, frozenset(entry.joint_names)), []
                ).append(entry)
        out: list[NamedAction] = []
        for (articulation, kind, joint_set), batch in merged.items():
            first = batch[0]
            for entry in batch[1:]:
                if entry.joint_names != first.joint_names:
                    raise ValueError(
                        "GymVectorRunner action "
                        f"(articulation={articulation!r}, kind={kind!r}, "
                        f"joints={sorted(joint_set)}) has inconsistent ordered "
                        "joint_names across envs"
                    )
            out.append(
                NamedAction(
                    articulation=articulation,
                    values=torch.cat([entry.values for entry in batch], dim=0),
                    joint_names=first.joint_names,
                    kind=first.kind,
                )
            )
        return tuple(out)

    def reset(self, seed=None, options=None):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Reset all envs; return ``(post_reset_obs, batched_infos)``."""
        super().reset(None)
        obs = dict(self.env.obs_cache)
        return obs, _batch_infos(self._extra_now(), self.num_envs)

    def step(self, actions):
        """Advance one tick with SameStep autoreset; return the 5-tuple."""
        result = self.tick(actions_override=self._vector_actions(actions))
        obs_pre = result.obs
        extra_pre = result.extra
        n = self.num_envs

        if self.reward_fn is None:
            rewards = np.zeros(n, dtype=np.float64)
        else:
            raw = self.reward_fn(obs_pre, extra_pre)
            rewards = _as_numpy(raw).astype(np.float64).reshape(-1)
            if rewards.size == 1:
                rewards = np.full(n, rewards.item())
            elif rewards.size != n:
                raise ValueError(f"reward_fn returned {rewards.size} rewards for {n} envs")

        terminateds = _as_numpy(result.terminated).astype(bool).reshape(-1)
        due = _as_numpy(result.reset_env_ids).astype(np.int64).reshape(-1)
        due_mask = np.zeros(n, dtype=bool)
        due_mask[due] = True
        truncateds = (due_mask & ~terminateds).astype(bool)

        obs = dict(self.env.obs_cache)  # post-reset (SameStep)
        infos = _batch_infos(self._extra_now(), n)
        final_observation = np.empty(n, dtype=object)
        final_info = np.empty(n, dtype=object)
        final_observation[:] = None
        final_info[:] = None
        for env_id in due:
            final_observation[env_id] = _per_env_values(obs_pre, int(env_id))
            final_info[env_id] = _per_env_values(extra_pre, int(env_id))
        infos["final_observation"] = final_observation
        infos["final_info"] = final_info
        return obs, rewards, terminateds, truncateds, infos


# Backwards-compatible name: the old single-env facade IS GymEnvRunner now.
GymRunner = GymEnvRunner


__all__ = ["GymEnvRunner", "GymRunner", "GymVectorRunner"]
