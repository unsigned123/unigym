"""Conformance: the state reader and the observation projection agree.

The fake backend is the shared conformance double for the ``EnvBackend`` step
boundary.  Its 17-key extra record and its observation cache are the two
sources :func:`unigym.runner.state.read_extra` and
:func:`unigym.sim.observation.build_observation` consume in production.  This
module proves they agree on every key, shape and joint column of one real
facade declaration; it makes no cross-backend physics claim.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from unigym.robots import restore_robot_registry, snapshot_robot_registry
from unigym.runner.state import read_extra
from unigym.runtime.actions import NamedAction
from unigym.sim import ArticulationHandle, Control, Observation, SimulationBuilder
from unigym.sim.observation import build_observation
from tests._support.fakes.backend import ARTICULATION_NAME, CONTROL_NAME, FakeBackend
from tests._support.sim_fixtures import write_sim_pack

JOINT_ORDER = ("shoulder", "elbow")
DEFAULT_JOINT_POS = {"shoulder": 0.1, "elbow": -0.2, "finger": 0.0}
PROJECTED_KEYS = (
    "joint_pos",
    "joint_vel",
    "applied_torque",
    "gravity_b",
    "base_pos_w",
    "base_quat_w",
    "base_lin_vel_b",
    "base_ang_vel_b",
)


@pytest.fixture(autouse=True)
def _isolate_robot_registry() -> Iterator[None]:
    """Restore the process-wide robot registry after the pack activation."""
    snapshot = snapshot_robot_registry()
    yield
    restore_robot_registry(snapshot)


def _handle(tmp_path: Path) -> ArticulationHandle:
    """A declared facade handle over the fake backend's two-joint arm."""
    pack_set, ref = write_sim_pack(
        tmp_path / "pack",
        joint_sets={"body": JOINT_ORDER},
        default_joint_pos=DEFAULT_JOINT_POS,
    )
    builder = SimulationBuilder(backend="mujoco", dt=0.02, physics_dt=0.005, pack_set=pack_set)
    return builder.add_articulation(
        ref,
        name=ARTICULATION_NAME,
        control=Control.position_pd(joint_set="body", kp=20.0, kd=1.0),
    )


def _observe(env: FakeBackend, handle: ArticulationHandle, *, t: int = 0) -> Observation:
    """Project the fake backend's live state through the shared reader."""
    return build_observation(
        t=t,
        num_envs=env.num_envs,
        state=read_extra(env),
        obs_cache=env.obs_cache,
        handles=(handle,),
        cameras=(),
        reset_env_ids=torch.zeros(0, dtype=torch.int64),
        reset_due_env_ids=torch.zeros(0, dtype=torch.int64),
        terminated=None,
    )


def test_reader_and_projection_agree_on_spawn_keys_shapes_and_columns(
    tmp_path: Path,
) -> None:
    """Every projected record key exists in the extra with the declared shape."""
    handle = _handle(tmp_path)
    env = FakeBackend(num_envs=1, joint_names=JOINT_ORDER)
    try:
        state = read_extra(env)
        obs = _observe(env, handle)

        assert handle.faces == {"body": JOINT_ORDER}
        assert tuple(env.controls[CONTROL_NAME].joint_names) == JOINT_ORDER
        prefix = f"articulation/{ARTICULATION_NAME}"
        for key in PROJECTED_KEYS:
            assert f"{prefix}/{key}" in state, f"backend extra lacks {prefix}/{key}"
        assert set(obs.raw) == set(state) - {"reset_env_ids", "should_reset"}

        record = obs.articulations[handle]
        assert record.joint_names == JOINT_ORDER
        assert tuple(handle.profile.joint_order) == JOINT_ORDER
        assert record.joint_pos.shape == (1, len(JOINT_ORDER))
        assert record.joint_vel.shape == record.joint_pos.shape
        assert record.applied_torque.shape == record.joint_pos.shape
        assert record.gravity_b.shape == (1, 3)
        assert record.base_pos_w.shape == (1, 3)
        assert record.base_quat_w.shape == (1, 4)
        assert record.base_lin_vel_b.shape == (1, 3)
        assert record.base_ang_vel_b.shape == (1, 3)
        assert torch.equal(record.joint_pos, torch.zeros(1, len(JOINT_ORDER)))
        q0 = torch.tensor(
            [DEFAULT_JOINT_POS[joint] for joint in JOINT_ORDER], dtype=record.joint_pos.dtype
        )
        assert torch.equal(record.joint_pos_rel, torch.zeros_like(record.joint_pos) - q0)
        for key in PROJECTED_KEYS:
            typed = getattr(record, key)
            assert torch.equal(typed, state[f"{prefix}/{key}"])
        assert obs.contacts == {}
        assert obs.reset_env_ids.numel() == 0
        assert obs.terminated is None
        assert record.prev_action is None
        assert record.command is None
    finally:
        env.close()


def test_two_env_rows_project_distinct_values(tmp_path: Path) -> None:
    """read_extra and build_observation agree row-for-row at num_envs=2."""
    handle = _handle(tmp_path)
    env = FakeBackend(num_envs=2, joint_names=JOINT_ORDER)
    target = torch.tensor([[0.25, -0.5], [-0.75, 0.4]], dtype=torch.float32)
    try:
        env.step({ARTICULATION_NAME: NamedAction(target, list(JOINT_ORDER), "position")})
        state = read_extra(env)
        obs = _observe(env, handle, t=1)

        prefix = f"articulation/{ARTICULATION_NAME}"
        record = obs.articulations[handle]
        assert obs.num_envs == 2
        assert record.joint_names == JOINT_ORDER
        assert record.joint_pos.shape == (2, len(JOINT_ORDER))
        assert record.gravity_b.shape == (2, 3)
        assert record.base_quat_w.shape == (2, 4)
        for key in PROJECTED_KEYS:
            typed = getattr(record, key)
            assert typed.shape[0] == 2, f"{key} lost the leading env axis: {typed.shape}"
            assert torch.equal(typed, state[f"{prefix}/{key}"])
        assert torch.equal(record.joint_pos, target)
        assert not torch.equal(record.joint_pos[0], record.joint_pos[1])
        time_in_episode = obs.raw["time_in_episode"]
        assert isinstance(time_in_episode, torch.Tensor)
        assert time_in_episode.tolist() == [1.0, 1.0]
        assert obs.reset_env_ids.tolist() == []
    finally:
        env.close()


def test_two_env_subset_reset_agrees_row_wise(tmp_path: Path) -> None:
    """A row-subset reset keeps the other row and the reader agrees per row."""
    handle = _handle(tmp_path)
    env = FakeBackend(num_envs=2, joint_names=JOINT_ORDER)
    target = torch.tensor([[0.25, -0.5], [-0.75, 0.4]], dtype=torch.float32)
    try:
        env.step({ARTICULATION_NAME: NamedAction(target, list(JOINT_ORDER), "position")})
        env.reset_envs(torch.tensor([1], dtype=torch.int64))
        state = read_extra(env)
        obs = _observe(env, handle, t=1)

        prefix = f"articulation/{ARTICULATION_NAME}"
        record = obs.articulations[handle]
        assert record.joint_pos.tolist() == [[0.25, -0.5], [0.0, 0.0]]
        for key in PROJECTED_KEYS:
            typed = getattr(record, key)
            assert torch.equal(typed, state[f"{prefix}/{key}"])
        time_in_episode = obs.raw["time_in_episode"]
        assert isinstance(time_in_episode, torch.Tensor)
        assert time_in_episode.tolist() == [1.0, 0.0]
        episode_id = obs.raw["episode_id"]
        assert isinstance(episode_id, torch.Tensor)
        assert episode_id.tolist() == [0, 1]
    finally:
        env.close()


def test_projection_columns_follow_the_acted_joint_values(tmp_path: Path) -> None:
    """A stepped action reaches the record column-for-column, never reordered."""
    handle = _handle(tmp_path)
    env = FakeBackend(num_envs=1, joint_names=JOINT_ORDER)
    target = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    try:
        env.step({ARTICULATION_NAME: NamedAction(target, list(JOINT_ORDER), "position")})
        obs = _observe(env, handle, t=1)
        record = obs.articulations[handle]

        prefix = f"articulation/{ARTICULATION_NAME}"
        raw_pos = obs.raw[f"{prefix}/joint_pos"]
        assert isinstance(raw_pos, torch.Tensor)
        assert torch.equal(record.joint_pos, raw_pos)
        assert torch.equal(record.joint_pos, target)
        q0 = torch.tensor(
            [DEFAULT_JOINT_POS[joint] for joint in JOINT_ORDER], dtype=record.joint_pos.dtype
        )
        assert torch.equal(record.joint_pos_rel, target - q0)
        assert record.joint_names == JOINT_ORDER
        for column, joint in enumerate(JOINT_ORDER):
            assert record.joint_pos[0, column] == target[0, column], joint
    finally:
        env.close()
