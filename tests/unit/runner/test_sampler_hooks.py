"""Sampler runner wiring: no runner-side binding, episode-boundary feedback."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch

from unigym.commands.base import CommandSampler
from unigym.core.episode_result import EpisodeResultView
from unigym.env.backends.base import EnvStep
from unigym.env.obs import ObsSnapshot, ObservationSchema
from unigym.runner.sync import SyncRunner
from unigym.terminations import TerminationGroup


class _EmptyObs:
    """Observation registry double whose snapshots carry no values."""

    def snapshot(self) -> ObsSnapshot:
        return ObsSnapshot({}, schema=ObservationSchema({}), num_envs=1, device="cpu")


class _ScriptedEnv:
    """One-env backend double with a fixed backend-owned due cadence."""

    def __init__(self, length: int = 2) -> None:
        self.num_envs = 1
        self.device = "cpu"
        self.controls = {}
        self.articulations = {}
        self.observations = SimpleNamespace(command=False, prev_action=False)
        self.obs = _EmptyObs()
        self._length = length
        self._step = 0
        self._t = 0
        self.reset_calls = 0

    def step(self, actions, raw_action=None, cmd=None) -> EnvStep:
        del actions, raw_action, cmd
        self._step += 1
        due = self._step >= self._length
        result = EnvStep(
            t=self._t,
            obs=self.obs.snapshot(),
            record={"time_in_episode": torch.zeros(1)},
            reset_env_ids=(
                torch.tensor([0], dtype=torch.int64) if due else torch.zeros(0, dtype=torch.int64)
            ),
            should_reset=due,
        )
        self._t += 1
        return result

    def reset_envs(self, env_ids: torch.Tensor) -> torch.Tensor:
        self.reset_calls += 1
        self._step = 0
        return env_ids

    def close(self) -> None:
        return None


class _NoopAdapter:
    """Minimal adapter: no actions, no history, no self-commands."""

    control_dt = 0.02
    generates_command = False

    def act(self, obs):
        return None

    def postprocess_action(self, raw):
        return []

    def compute_obs(self, snapshot, env):
        return torch.zeros(1, 1)

    def compute_obs_collect(self, frame, snapshot, env):
        return frame

    def on_reset(self, env_ids):
        return None


class _RecordingSink:
    """Sink double that records callback order and can publish result views."""

    def __init__(self, order: list, results=None) -> None:
        self.order = order
        self.results = results

    def on_step(self, t: int, bundle: dict) -> None:
        del t, bundle

    def on_episode_end(self, env_ids, bundle: dict) -> None:
        del env_ids
        self.order.append("sink")
        if self.results is not None:
            bundle["episode_results"] = self.results

    def on_step_end(self, t: int, bundle: dict) -> None:
        del t, bundle

    def on_run_end(self) -> dict:
        return {}


class _HookSampler(CommandSampler):
    """Sampler double recording every lifecycle hook call."""

    def __init__(self, order: list, *, fail: bool = False) -> None:
        super().__init__(dim=1, device="cpu", num_envs=1)
        self.order = order
        self.fail = fail
        self.bound: list = []
        self.episode_end_calls: list = []
        self.reset_calls: list = []

    def bind_env(self, env) -> None:
        self.bound.append(env)

    def on_episode_end(self, env_ids, results) -> None:
        self.order.append("sampler")
        self.episode_end_calls.append((env_ids.tolist(), results))
        if self.fail:
            raise RuntimeError("sampler on_episode_end failed")

    def reset(self, env_ids=None) -> torch.Tensor:
        self.order.append("reset")
        ids = self._resolve_ids(env_ids)
        self.reset_calls.append(ids.tolist())
        self.cmd[ids] = 0.0
        return self.cmd

    def update(self, step_dt: float, cmd: torch.Tensor) -> torch.Tensor:
        del step_dt
        self.cmd = cmd
        return self.cmd


def _runner(sink, sampler, env, *, auto_reset: bool = True) -> SyncRunner:
    return SyncRunner(
        cfg={"collect": {"steps": 4}},
        profile=None,
        adapter=_NoopAdapter(),
        sampler=sampler,
        term_group=TerminationGroup(),
        env=env,
        sink=sink,
        auto_reset=auto_reset,
        control_dt=0.02,
    )


def _view() -> EpisodeResultView:
    return EpisodeResultView(task="acme/reach", success=True, metrics={"score": 1.0})


def test_runner_does_not_bind_the_sampler():
    """Binding belongs to the instance build seam; runners never rebind."""
    env = _ScriptedEnv()
    sampler = _HookSampler([])
    _runner(_RecordingSink([]), sampler, env)
    _runner(_RecordingSink([]), sampler, env)

    assert sampler.bound == []


def test_runner_calls_sampler_on_episode_end_between_sink_and_reset():
    """Boundary order is sink.on_episode_end -> sampler.on_episode_end -> reset."""
    env = _ScriptedEnv(length=2)
    order: list = []
    sampler = _HookSampler(order)
    runner = _runner(_RecordingSink(order), sampler, env)

    runner.tick()
    order.clear()
    runner.tick()

    assert order == ["sink", "sampler", "reset"]
    assert sampler.reset_calls[-1] == [0]


def test_runner_forwards_per_env_episode_result_views():
    """The sampler receives the sink-published views keyed by env id."""
    env = _ScriptedEnv(length=1)
    view = _view()
    sink = _RecordingSink([], results={0: (view,)})
    sampler = _HookSampler([])
    runner = _runner(sink, sampler, env)

    runner.tick()

    env_ids, results = sampler.episode_end_calls[0]
    assert env_ids == [0]
    assert dict(results) == {0: (view,)}


def test_runner_results_mapping_is_read_only_to_the_sampler():
    """The sampler observes a wrapped mapping and cannot add or replace keys."""
    env = _ScriptedEnv(length=1)
    view = _view()
    sampler = _HookSampler([])
    runner = _runner(_RecordingSink([], results={0: (view,)}), sampler, env)

    runner.tick()

    _, results = sampler.episode_end_calls[0]
    assert isinstance(results, Mapping)
    with pytest.raises(TypeError):
        operator.setitem(results, 1, ())
    assert dict(results) == {0: (view,)}


def test_runner_passes_empty_results_when_the_sink_publishes_none():
    """A collect-only boundary (no episode_results key) yields an empty mapping."""
    env = _ScriptedEnv(length=1)
    sampler = _HookSampler([])
    runner = _runner(_RecordingSink([]), sampler, env)

    runner.tick()

    _, results = sampler.episode_end_calls[0]
    assert dict(results) == {}


def test_runner_calls_sampler_on_episode_end_without_auto_reset():
    """auto_reset=False skips the reset chain but keeps the sampler feedback."""
    env = _ScriptedEnv(length=2)
    order: list = []
    sampler = _HookSampler(order)
    runner = _runner(_RecordingSink(order), sampler, env, auto_reset=False)

    runner.tick()
    order.clear()
    runner.tick()

    assert order == ["sink", "sampler"]
    assert env.reset_calls == 0
    assert len(sampler.reset_calls) == 1  # the initial _start reset only


def test_runner_propagates_a_sampler_on_episode_end_failure():
    """A raising sampler hook is run-critical: the tick raises."""
    env = _ScriptedEnv(length=1)
    sampler = _HookSampler([], fail=True)
    runner = _runner(_RecordingSink([]), sampler, env)

    with pytest.raises(RuntimeError, match="on_episode_end failed"):
        runner.tick()
