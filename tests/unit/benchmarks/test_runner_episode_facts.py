"""Runner -> BenchmarkSink episode facts: local ticks + reset outcome path."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from unigym.benchmarks.base import EpisodeResult, Task
from unigym.benchmarks.episode import RunIdentity
from unigym.benchmarks.termination import EpisodeOutcome
from unigym.env.backends.base import EnvStep
from unigym.env.obs import ObsSnapshot, ObservationSchema
from unigym.initializers.base import ResetOutcome
from unigym.runner.sinks.benchmark import BenchmarkSink
from unigym.runner.sync import SyncRunner
from unigym.terminations import TerminationGroup


class _ScriptedObs:
    def __init__(self):
        self._snapshot = ObsSnapshot(
            {"articulation/tiny/joint_pos": torch.zeros(1, 2)},
            schema=ObservationSchema({}),
            num_envs=1,
            device="cpu",
        )

    def snapshot(self):
        return self._snapshot


class _ScriptedEnv:
    """Minimal backend double with an explicit per-episode reset chain."""

    def __init__(self, lengths=(2, 2), policy_dt=0.02):
        self.num_envs = 1
        self.device = "cpu"
        self.controls = {}
        self.articulations = {}
        self.observations = SimpleNamespace(command=False, prev_action=False)
        self.obs = _ScriptedObs()
        self._lengths = list(lengths)
        self._policy_dt = policy_dt
        self._step = 0
        self._episode_id = 0
        self._time = 0.0
        self._t = 0
        self._last: dict[int, object] = {}

    @property
    def last_reset_outcomes(self):
        return self._last

    def step(self, actions, raw_action=None, cmd=None):
        self._step += 1
        self._time += self._policy_dt
        length = self._lengths[self._episode_id]
        due = self._step >= length
        record = {
            "episode_id": torch.tensor([self._episode_id], dtype=torch.int64),
            "time_in_episode": torch.tensor([self._time]),
        }
        result = EnvStep(
            t=self._t,
            obs=self.obs.snapshot(),
            record=record,
            reset_env_ids=(
                torch.tensor([0], dtype=torch.int64) if due else torch.zeros(0, dtype=torch.int64)
            ),
            should_reset=due,
        )
        self._t += 1
        return result

    def reset_envs(self, env_ids):
        for _ in env_ids.tolist():
            self._episode_id += 1
            self._step = 0
            self._time = 0.0
            self._last = {
                0: ResetOutcome(
                    valid=self._episode_id != 1,
                    reason=("blocked" if self._episode_id == 1 else None),
                    metrics={"clearance": 0.0},
                )
            }
        return env_ids

    def close(self):
        return None


class _NoopAdapter:
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


class _ZeroTask(Task):
    name = "zero"

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={"m": 1.0}, success=None)


def _identity():
    return RunIdentity(
        backend="stub",
        device="cpu",
        num_envs=1,
        seed=0,
        policy_dt=0.02,
        sim_dt=0.005,
    )


def test_multi_episode_ticks_are_local_and_invalid_reset_reaches_episode(tmp_path):
    env = _ScriptedEnv(lengths=(2, 2))
    sink = BenchmarkSink(
        tasks=[_ZeroTask(params={}, meta={})],
        meta={"run_name": "r", "num_envs": 1, "policy_dt": 0.02, "seed": 0},
        min_episodes=2,
        max_steps=4,
        report_path=str(tmp_path / "benchmark_report.json"),
        run_identity=_identity(),
    )
    runner = SyncRunner(
        cfg={"collect": {"steps": 4}},
        profile=None,
        adapter=_NoopAdapter(),
        sampler=None,
        term_group=TerminationGroup(),
        env=env,
        sink=sink,
    )

    summary = runner.run(stop_when=sink.episodes_done)

    assert summary["episodes_total"] == 2
    first, second = sink.reporter.episodes
    # episode 2 starts at tick 1, not at the global run tick 3/4
    for ep in (first, second):
        assert ep.step_index.tolist() == [1, 2]
        assert ep.time_s.tolist() == [0.02, 0.04]
        assert ep.terminal_index is None or 0 <= ep.terminal_index <= ep.length
    assert first.reset_valid is None
    assert first.termination.primary.outcome is EpisodeOutcome.DEADLINE
    assert first.termination.primary.code == "runtime.episode_horizon"
    assert first.terminal_index == 2
    # the initializer's ResetOutcome is not silently dropped
    assert second.reset_valid is False
    assert second.reset_reason == "blocked"
    assert second.termination.primary.outcome is EpisodeOutcome.INVALID_RESET
    assert second.termination.primary.code == "initializer.invalid_reset"
    assert second.terminal_index == 0
