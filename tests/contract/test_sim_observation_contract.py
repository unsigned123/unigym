"""The public Observation shape, mapping freeze and handle-keyed contract."""

from __future__ import annotations

import operator
from dataclasses import FrozenInstanceError

import pytest
import torch

from unigym.robots.base import BodyBinding, RobotProfile
from unigym.sim import ArticulationHandle, CameraHandle, Observation, ObservationError
from unigym.sim.observation import build_observation


def _handle() -> ArticulationHandle:
    """One real handle with a two-joint body order."""
    profile = RobotProfile(
        name="arm",
        joint_sets={"body": ("shoulder", "elbow")},
        default_joint_pos={"shoulder": 0.0, "elbow": 0.0},
        assets=None,
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
    )
    return ArticulationHandle(
        name="arm",
        profile=profile,
        control=None,
        action_groups={},
        _owner=object(),
    )


def _state() -> dict[str, object]:
    """A complete namespaced state mapping for the fixture handle."""
    zeros3 = torch.zeros(1, 3, dtype=torch.float32)
    return {
        "articulation/arm/joint_pos": torch.tensor([[0.1, 0.2]]),
        "articulation/arm/joint_vel": torch.zeros(1, 2),
        "articulation/arm/applied_torque": torch.zeros(1, 2),
        "articulation/arm/gravity_b": torch.tensor([[0.0, 0.0, -1.0]]),
        "articulation/arm/base_pos_w": zeros3.clone(),
        "articulation/arm/base_quat_w": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "articulation/arm/base_lin_vel_b": zeros3.clone(),
        "articulation/arm/base_ang_vel_b": zeros3.clone(),
        "reset_env_ids": torch.zeros(0, dtype=torch.int64),
        "should_reset": False,
    }


def _observation() -> tuple[Observation, ArticulationHandle, CameraHandle]:
    """Build one observation with a published camera frame."""
    handle = _handle()
    camera = CameraHandle(name="cam", entry={"name": "cam"}, _owner=object())
    cache = {"cam/cam/rgb": torch.zeros(1, 4, 4, 3)}
    obs = build_observation(
        t=2,
        num_envs=1,
        state=_state(),
        obs_cache=cache,
        handles=(handle,),
        cameras=(camera,),
        reset_env_ids=torch.tensor([0], dtype=torch.int64),
        reset_due_env_ids=torch.zeros(0, dtype=torch.int64),
        terminated=torch.tensor([False]),
    )
    return obs, handle, camera


def test_observation_mappings_are_frozen_views() -> None:
    """Every mapping field is read-only after construction."""
    obs, handle, _ = _observation()

    with pytest.raises(TypeError):
        operator.setitem(obs.raw, "extra", torch.zeros(1))
    with pytest.raises(TypeError):
        operator.setitem(obs.obs_cache, "extra", torch.zeros(1))
    with pytest.raises(TypeError):
        operator.setitem(obs.articulations, handle, None)


def test_observation_records_are_frozen() -> None:
    """The Observation and its typed records are immutable value holders."""
    obs, handle, _ = _observation()
    record = obs.articulations[handle]

    with pytest.raises(FrozenInstanceError):
        setattr(obs, "t", 3)
    with pytest.raises(FrozenInstanceError):
        setattr(record, "joint_pos", torch.zeros(1, 2))


def test_articulations_are_keyed_by_handle_identity() -> None:
    """The typed record is reachable through the exact minted handle."""
    obs, handle, _ = _observation()

    assert list(obs.articulations) == [handle]
    assert obs.articulations[handle].handle is handle


def test_raw_and_obs_cache_stay_distinguishable() -> None:
    """raw is the live state minus bookkeeping; obs_cache is the step obs copy."""
    obs, handle, _ = _observation()

    assert obs.raw is not obs.obs_cache
    assert set(obs.raw) == set(_state()) - {"reset_env_ids", "should_reset"}
    assert set(obs.obs_cache) == {"cam/cam/rgb"}
    assert obs.articulations[handle].joint_pos is obs.raw["articulation/arm/joint_pos"]


def test_reset_masks_are_distinct_frozen_fields() -> None:
    """The performed and due masks are separate immutable Observation fields."""
    obs, _, _ = _observation()

    assert obs.num_envs == 1
    assert obs.reset_env_ids.tolist() == [0]
    assert obs.reset_due_env_ids.tolist() == []
    assert obs.reset_env_ids is not obs.reset_due_env_ids
    with pytest.raises(FrozenInstanceError):
        setattr(obs, "reset_due_env_ids", torch.zeros(0, dtype=torch.int64))


@pytest.mark.parametrize("num_envs", [0, -1, True, 1.5, "1"])
def test_num_envs_is_validated(num_envs: object) -> None:
    """num_envs is a positive int at the observation boundary."""
    with pytest.raises(ObservationError, match="num_envs"):
        build_observation(
            t=0,
            num_envs=num_envs,
            state=_state(),
            obs_cache={},
            handles=(_handle(),),
            cameras=(),
            reset_env_ids=torch.zeros(0, dtype=torch.int64),
            reset_due_env_ids=torch.zeros(0, dtype=torch.int64),
            terminated=None,
        )


def test_projection_does_not_build_a_second_observation_catalog() -> None:
    """The projection adds no keys to raw or obs_cache beyond the two sources."""
    state = _state()
    cache = {"cam/cam/rgb": torch.zeros(1, 4, 4, 3)}
    handle = _handle()
    camera = CameraHandle(name="cam", entry={"name": "cam"}, _owner=object())

    obs = build_observation(
        t=0,
        num_envs=1,
        state=state,
        obs_cache=cache,
        handles=(handle,),
        cameras=(camera,),
        reset_env_ids=torch.zeros(0, dtype=torch.int64),
        reset_due_env_ids=torch.zeros(0, dtype=torch.int64),
        terminated=None,
    )

    assert set(obs.raw) == set(state) - {"reset_env_ids", "should_reset"}
    assert set(obs.obs_cache) == set(cache)
    assert set(obs.contacts) == set()
    assert set(state) == {
        "articulation/arm/joint_pos",
        "articulation/arm/joint_vel",
        "articulation/arm/applied_torque",
        "articulation/arm/gravity_b",
        "articulation/arm/base_pos_w",
        "articulation/arm/base_quat_w",
        "articulation/arm/base_lin_vel_b",
        "articulation/arm/base_ang_vel_b",
        "reset_env_ids",
        "should_reset",
    }


def test_missing_state_key_raises_the_public_observation_error() -> None:
    """A missing required key crosses the facade as ObservationError."""
    handle = _handle()
    state = _state()
    del state["articulation/arm/joint_pos"]

    with pytest.raises(ObservationError, match="articulation/arm/joint_pos"):
        build_observation(
            t=0,
            num_envs=1,
            state=state,
            obs_cache={},
            handles=(handle,),
            cameras=(),
            reset_env_ids=torch.zeros(0, dtype=torch.int64),
            reset_due_env_ids=torch.zeros(0, dtype=torch.int64),
            terminated=None,
        )
