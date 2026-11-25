"""Runner-local lifecycle capture/restore in the rewind package."""

from __future__ import annotations

import pytest
import torch

from tests._support.fakes.backend import ARTICULATION_NAME, FakeBackend
from tests._support.fakes.external_sink import ResetAwareLifecycleSink
from unigym.adapters.manual import ManualAdapter
from unigym.core.episode_termination import EpisodeOutcome, Termination, TerminationSource
from unigym.runner.sync import SyncRunner
from unigym.runtime.actions import NamedAction
from unigym.runtime.rewind import RewindController, RewindGateError
from unigym.runtime.rewind_providers import RunnerLifecycleProvider
from unigym.terminations import TerminationGroup


def _actions() -> tuple[NamedAction, ...]:
    return (
        NamedAction(
            ARTICULATION_NAME,
            torch.zeros(1, 1),
            ("shoulder",),
            "position",
        ),
    )


def _runner(env: FakeBackend, sink) -> SyncRunner:
    return SyncRunner(
        cfg={"collect": {"steps": 4}},
        profile=None,
        adapter=ManualAdapter(),
        sampler=None,
        term_group=TerminationGroup(),
        env=env,
        sink=sink,
        control_dt=0.02,
    )


def _termination() -> Termination:
    return Termination(EpisodeOutcome.MANUAL, "custom.stop", TerminationSource.CUSTOM)


def test_runner_lifecycle_provider_restores_the_full_local_state():
    """A checkpoint reopens the closed row and restores episode-local facts."""
    env = FakeBackend()
    sink = ResetAwareLifecycleSink()
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions())
        provider = RunnerLifecycleProvider(runner)
        snapshot = provider.rewind_state()

        assert snapshot["closed"] == [False]
        assert snapshot["episode_step"] == [1]
        assert snapshot["started"] is True
        assert snapshot["reset_samples"][0] is not None

        runner.terminate([0], {0: _termination()}, reset=False)
        assert provider.rewind_state()["closed"] == [True]

        provider.apply_rewind_state(snapshot)

        restored = provider.rewind_state()
        assert restored["closed"] == [False]
        assert restored["episode_step"] == [1]
        assert restored["started"] is True
        runner.tick(actions_override=_actions())
        assert runner.ticks_done == 2
    finally:
        env.close()


def test_runner_lifecycle_provider_refusal_is_zero_change():
    """A malformed payload is refused by validate and apply without mutation."""
    env = FakeBackend()
    sink = ResetAwareLifecycleSink()
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions())
        provider = RunnerLifecycleProvider(runner)
        state = provider.rewind_state()
        bad = dict(state)
        bad["closed"] = [False, True]

        with pytest.raises(RewindGateError, match="num_envs"):
            provider.validate_rewind_state(bad)
        with pytest.raises(RewindGateError, match="num_envs"):
            provider.apply_rewind_state(bad)

        assert provider.rewind_state()["closed"] == [False]
        assert provider.rewind_state()["episode_step"] == [1]
        with pytest.raises(RewindGateError, match="missing lifecycle keys"):
            provider.validate_rewind_state({"closed": [False]})
    finally:
        env.close()


class _Obs:
    def __init__(self) -> None:
        self.state: dict = {}

    def rewind_state(self) -> dict:
        return dict(self.state)

    def restore_rewind_state(self, state: dict) -> None:
        self.state = dict(state)


class _SnapshotEnv:
    def __init__(self) -> None:
        self.obs = _Obs()
        self.value = 0

    def snapshot(self) -> dict:
        return {"value": self.value}

    def restore(self, state: dict) -> None:
        self.value = state["value"]


class _Adapter:
    rewindable = True

    def __init__(self) -> None:
        self.value = 0

    def on_rewind(self, restored_tick: int) -> None:
        del restored_tick

    def snapshot_internal(self) -> dict:
        return {"value": self.value}

    def restore_internal(self, state: dict) -> None:
        self.value = state["value"]


class _Bridge:
    def __init__(self) -> None:
        self.commands: dict = {}
        self._bundle: dict = {"obs_frame": None, "cmd": None, "record": {}}
        self._last_step = 2
        self.rewind_log: list = []
        self.out_dir = None
        self.paused = False

    def register_command(self, name: str, handler) -> None:
        self.commands[name] = handler

    def pause_for_rewind(self) -> None:
        self.paused = True


def _controller(runner: SyncRunner, provider: RunnerLifecycleProvider):
    bridge = _Bridge()
    controller = RewindController(
        bridge,
        env=_SnapshotEnv(),
        adapter=_Adapter(),
        engine=runner,
        backend_label="mujoco",
        providers=[provider],
    )
    return controller


def test_rewind_controller_round_trip_restores_lifecycle():
    """Snapshot -> external close -> rewind reopens the runner lifecycle."""
    env = FakeBackend()
    sink = ResetAwareLifecycleSink()
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions())
        provider = RunnerLifecycleProvider(runner)
        controller = _controller(runner, provider)
        snapshot = controller.cmd_snapshot(0, {})

        runner.terminate([0], {0: _termination()}, reset=False)
        assert provider.rewind_state()["closed"] == [True]

        controller.cmd_rewind(0, {"snap_id": snapshot["snap_id"]})

        assert provider.rewind_state()["closed"] == [False]
        assert provider.rewind_state()["started"] is True
    finally:
        env.close()


def test_rewind_controller_refuses_a_bad_lifecycle_payload_with_zero_change():
    """Pre-flight refusal leaves the closed runner and the engine untouched."""
    env = FakeBackend()
    sink = ResetAwareLifecycleSink()
    runner = _runner(env, sink)
    try:
        runner.tick(actions_override=_actions())
        provider = RunnerLifecycleProvider(runner)
        controller = _controller(runner, provider)
        snapshot = controller.cmd_snapshot(0, {})
        record = controller._snaps[snapshot["snap_id"]]
        record["payload"]["providers"][RunnerLifecycleProvider.name] = {
            "closed": [False],
            "episode_step": [1],
            "reset_samples": [None],
            "reset_valid": [None],
            "reset_reason": [None],
            "started": "yes",
        }

        runner.terminate([0], {0: _termination()}, reset=False)

        with pytest.raises(RewindGateError, match="started must be a bool"):
            controller.cmd_rewind(0, {"snap_id": snapshot["snap_id"]})

        assert provider.rewind_state()["closed"] == [True]
        assert controller.env.value == 0
    finally:
        env.close()
