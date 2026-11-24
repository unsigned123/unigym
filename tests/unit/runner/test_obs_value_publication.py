"""The runner publishes selected observations before the step value snapshot."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from tests._support.fakes.backend import ARTICULATION_NAME, JOINT_NAME, FakeBackend
from tests._support.fakes.external_sink import ValuePublishingSink, ViewProbeSink
from unigym.adapters.manual import ManualAdapter
from unigym.env.obs import (
    ObsConfigError,
    ObservationConfig,
    ObservationRegistry,
    build_observation_schema,
)
from unigym.runner.sinks import FanOutSink
from unigym.runner.sync import SyncRunner
from unigym.runtime.actions import NamedAction
from unigym.terminations import TerminationGroup
from unigym.values import ValueStore

JOINT_POS = f"articulation/{ARTICULATION_NAME}/joint_pos"
JOINT_VEL = f"articulation/{ARTICULATION_NAME}/joint_vel"
OBS_JOINT_POS = f"obs/{JOINT_POS}"
ACTIONS = (
    NamedAction(
        ARTICULATION_NAME,
        torch.full((1, 1), 0.25),
        (JOINT_NAME,),
        "position",
    ),
)


def _configured_env(
    values: Sequence[str],
    *,
    due_rows: Sequence[int] = (),
    cameras: dict | None = None,
) -> FakeBackend:
    """A fake backend whose observation registry selects *values*."""
    env = FakeBackend(due_rows=due_rows)
    raw: dict[str, object] = {
        "robot": [ARTICULATION_NAME],
        "command": True,
        "prev_action": True,
        "values": list(values),
    }
    if cameras is not None:
        raw["camera"] = list(cameras)
    config = ObservationConfig.from_mapping(raw)
    schema = build_observation_schema(
        config,
        full_joint_names={ARTICULATION_NAME: env.joint_names},
        policy_joint_names={ARTICULATION_NAME: env.joint_names},
        cameras=cameras,
    )
    env.observations = config
    env.obs = ObservationRegistry(config, schema, num_envs=env.num_envs, device=env.device)
    return env


def _runner(env: FakeBackend, sink, *, values: ValueStore | None) -> SyncRunner:
    return SyncRunner(
        cfg={"collect": {"steps": 4}},
        profile=None,
        adapter=ManualAdapter(num_envs=1),
        sampler=None,
        term_group=TerminationGroup(),
        env=env,
        sink=sink,
        control_dt=0.02,
        values=values,
    )


def test_step_and_final_contexts_see_the_current_tick_selected_observation() -> None:
    env = _configured_env([JOINT_POS, JOINT_VEL], due_rows=(0,))
    store = ValueStore(num_envs=1, device="cpu")
    probe = ViewProbeSink()
    runner = _runner(env, probe, values=store)
    try:
        runner.tick(actions_override=ACTIONS)
    finally:
        env.close()

    step_ctx = probe.step_contexts[0]
    assert OBS_JOINT_POS in step_ctx.values
    assert torch.equal(step_ctx.values[OBS_JOINT_POS], torch.full((1, 1), 0.25))
    assert f"obs/{JOINT_VEL}" in step_ctx.values
    assert JOINT_POS not in step_ctx.values
    assert "obs/articulation/arm/base_pos_w" not in step_ctx.values

    final_ctx = probe.final_step_contexts[0]
    assert OBS_JOINT_POS in final_ctx.values
    assert torch.equal(final_ctx.values[OBS_JOINT_POS], torch.full((1, 1), 0.25))


def test_unselected_observation_keys_are_never_published() -> None:
    env = _configured_env([JOINT_POS])
    store = ValueStore(num_envs=1, device="cpu")
    probe = ViewProbeSink()
    runner = _runner(env, probe, values=store)
    try:
        runner.tick(actions_override=ACTIONS)
    finally:
        env.close()

    names = set(store.snapshot())
    assert names == {OBS_JOINT_POS}
    assert f"obs/{JOINT_VEL}" not in names
    assert "obs/command" not in names
    assert "obs/prev_action" not in names


def test_no_store_is_a_clean_no_op() -> None:
    env = _configured_env([JOINT_POS])
    probe = ViewProbeSink()
    runner = _runner(env, probe, values=None)
    try:
        runner.tick(actions_override=ACTIONS)
    finally:
        env.close()

    assert runner.values is None
    assert len(probe.step_contexts[0].values) == 0


def test_empty_selection_with_a_store_publishes_nothing() -> None:
    env = _configured_env([])
    store = ValueStore(num_envs=1, device="cpu")
    probe = ViewProbeSink()
    runner = _runner(env, probe, values=store)
    try:
        runner.tick(actions_override=ACTIONS)
    finally:
        env.close()

    assert dict(store.snapshot()) == {}
    assert len(probe.step_contexts[0].values) == 0


def test_observation_owner_does_not_interfere_with_the_sink_owner() -> None:
    env = _configured_env([JOINT_POS], due_rows=(0,))
    store = ValueStore(num_envs=1, device="cpu")
    publishing = ValuePublishingSink(store.publisher("sink"), num_envs=1)
    probe = ViewProbeSink()
    runner = _runner(env, FanOutSink(publishing, probe), values=store)
    try:
        runner.tick(actions_override=ACTIONS)
    finally:
        env.close()

    snapshot = store.snapshot()
    assert set(snapshot) == {publishing.value, OBS_JOINT_POS}
    assert float(snapshot[publishing.value][0, 0]) == 1.0
    assert torch.equal(snapshot[OBS_JOINT_POS], torch.full((1, 1), 0.25))

    step_ctx = probe.step_contexts[0]
    assert OBS_JOINT_POS in step_ctx.values
    assert publishing.value not in step_ctx.values
    final_ctx = probe.final_step_contexts[0]
    assert OBS_JOINT_POS in final_ctx.values
    assert publishing.value in final_ctx.values


def test_camera_selection_fails_before_the_runner_starts() -> None:
    cameras = {"ego": {"cam/ego/rgb": ((3, 4, 4), "u8")}}
    config = ObservationConfig.from_mapping(
        {"robot": [ARTICULATION_NAME], "camera": ["ego"], "values": ["cam/ego/rgb"]}
    )
    schema = build_observation_schema(
        config,
        full_joint_names={ARTICULATION_NAME: (JOINT_NAME,)},
        cameras=cameras,
    )

    with pytest.raises(ObsConfigError, match="camera/depth"):
        ObservationRegistry(config, schema, num_envs=1, device="cpu")
