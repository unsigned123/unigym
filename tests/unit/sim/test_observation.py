"""The typed observation projection over synthetic step tensors."""

from __future__ import annotations

import pytest
import torch

from unigym.robots.base import BodyBinding, RobotProfile
from unigym.sim import ArticulationHandle, CameraHandle, ObservationError
from unigym.sim.observation import build_observation


def _handle(
    *,
    name: str = "arm",
    joints: tuple[str, ...] = ("shoulder", "elbow"),
    defaults: dict[str, float] | None = None,
) -> ArticulationHandle:
    """One real handle whose profile owns the body joint order."""
    q0 = {joint: 0.0 for joint in joints} if defaults is None else dict(defaults)
    profile = RobotProfile(
        name=name,
        joint_sets={"body": joints},
        default_joint_pos=q0,
        assets=None,
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
    )
    return ArticulationHandle(
        name=name,
        ref=None,
        profile=profile,
        control=None,
        action_groups={},
        _owner=object(),
    )


def _state(
    handle: ArticulationHandle,
    *,
    joint_pos: torch.Tensor | None = None,
    joints: int = 2,
    num_envs: int = 1,
) -> dict[str, object]:
    """A complete namespaced state mapping for one articulation."""
    prefix = f"articulation/{handle.name}"
    position = (
        torch.zeros(num_envs, joints, dtype=torch.float32) if joint_pos is None else joint_pos
    )
    zeros3 = torch.zeros(num_envs, 3, dtype=torch.float32)
    return {
        f"{prefix}/joint_pos": position,
        f"{prefix}/joint_vel": torch.zeros_like(position),
        f"{prefix}/applied_torque": torch.zeros_like(position),
        f"{prefix}/gravity_b": torch.tensor([[0.0, 0.0, -1.0]]).expand(num_envs, 3).clone(),
        f"{prefix}/base_pos_w": zeros3.clone(),
        f"{prefix}/base_quat_w": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(num_envs, 4).clone(),
        f"{prefix}/base_lin_vel_b": zeros3.clone(),
        f"{prefix}/base_ang_vel_b": zeros3.clone(),
        "time_in_episode": torch.zeros(num_envs, dtype=torch.float32),
        "reset_env_ids": torch.zeros(0, dtype=torch.int64),
        "should_reset": False,
    }


def _build(
    state: dict[str, object],
    handle: ArticulationHandle,
    *,
    obs_cache: dict[str, object] | None = None,
    cameras: tuple[CameraHandle, ...] = (),
    t: int = 0,
    num_envs: int = 1,
    reset_env_ids: torch.Tensor | None = None,
    reset_due_env_ids: torch.Tensor | None = None,
    terminated: torch.Tensor | None = None,
):
    """Project one synthetic state into an observation record."""
    return build_observation(
        t=t,
        num_envs=num_envs,
        state=state,
        obs_cache={} if obs_cache is None else obs_cache,
        handles=(handle,),
        cameras=cameras,
        reset_env_ids=(
            torch.zeros(0, dtype=torch.int64) if reset_env_ids is None else reset_env_ids
        ),
        reset_due_env_ids=(
            torch.zeros(0, dtype=torch.int64) if reset_due_env_ids is None else reset_due_env_ids
        ),
        terminated=terminated,
    )


def test_projection_names_columns_by_profile_order_and_derives_rel() -> None:
    """joint_names is profile.joint_order; joint_pos_rel is absolute minus q0."""
    handle = _handle(defaults={"shoulder": 0.5, "elbow": -0.25})
    state = _state(handle, joint_pos=torch.tensor([[1.0, 2.0]]))

    obs = _build(state, handle)
    record = obs.articulations[handle]

    assert record.joint_names == ("shoulder", "elbow")
    assert record.joint_pos.tolist() == [[1.0, 2.0]]
    assert record.joint_pos_rel.tolist() == [[0.5, 2.25]]


def test_missing_required_state_key_names_the_key() -> None:
    """A missing per-articulation key is an ObservationError, never a zero-fill."""
    handle = _handle()
    state = _state(handle)
    del state[f"articulation/{handle.name}/joint_vel"]

    with pytest.raises(ObservationError, match="articulation/arm/joint_vel"):
        _build(state, handle)


def test_joint_pos_column_count_must_match_the_profile_order() -> None:
    """A (N, J) disagreement with joint_order is refused."""
    handle = _handle()
    state = _state(handle, joint_pos=torch.zeros(1, 3), joints=3)

    with pytest.raises(ObservationError, match=r"shape \(N, 2\)"):
        _build(state, handle)


def test_missing_default_joint_pos_names_the_joint() -> None:
    """A joint without a valid profile default cannot derive joint_pos_rel."""
    handle = _handle(defaults={"shoulder": 0.0})
    state = _state(handle)

    with pytest.raises(ObservationError, match="'elbow'"):
        _build(state, handle)


def test_optional_prev_action_and_command_are_absent_then_copied() -> None:
    """The two optional cached tensors are None when absent and exact when present."""
    handle = _handle()
    state = _state(handle)
    prefix = f"articulation/{handle.name}"

    absent = _build(state, handle)
    assert absent.articulations[handle].prev_action is None
    assert absent.articulations[handle].command is None

    prev_action = torch.tensor([[0.1, 0.2]])
    command = torch.tensor([[1.0]])
    present = _build(
        state,
        handle,
        obs_cache={f"{prefix}/prev_action": prev_action, f"{prefix}/command": command},
    )
    record = present.articulations[handle]
    assert record.prev_action is prev_action
    assert record.command is command


def test_obs_cache_keeps_non_tensor_record_entries_exactly() -> None:
    """A record-like cache entry survives the exact copy; projection ignores it."""
    handle = _handle()
    state = _state(handle)
    prefix = f"articulation/{handle.name}"
    prev_action = torch.tensor([[0.1, 0.2]])
    probe = {"probe": [1, 2]}
    cache: dict[str, object] = {
        "contact_ops/foo": probe,
        "contact_ops/bar": ["raw", 3],
        f"{prefix}/prev_action": prev_action,
    }

    obs = _build(state, handle, obs_cache=cache)

    assert dict(obs.obs_cache) == cache
    assert obs.obs_cache["contact_ops/foo"] is probe
    assert obs.articulations[handle].prev_action is prev_action


def test_expected_cached_key_present_but_not_a_tensor_is_refused() -> None:
    """Projected cache keys keep the tensor contract despite the loose copy."""
    handle = _handle()
    state = _state(handle)
    prefix = f"articulation/{handle.name}"

    with pytest.raises(ObservationError, match=f"{prefix}/prev_action"):
        _build(state, handle, obs_cache={f"{prefix}/prev_action": {"oops": 1}})


def test_camera_frames_require_a_published_image_key() -> None:
    """Only cameras with a cached rgb/depth key get a frame; nothing is rendered."""
    handle = _handle()
    state = _state(handle)
    rgb_camera = CameraHandle(name="rgb_cam", entry={"name": "rgb_cam"}, _owner=object())
    depth_camera = CameraHandle(name="depth_cam", entry={"name": "depth_cam"}, _owner=object())
    silent_camera = CameraHandle(name="silent", entry={"name": "silent"}, _owner=object())
    rgb = torch.zeros(1, 4, 4, 3)
    depth = torch.zeros(1, 4, 4, 1)

    obs = _build(
        state,
        handle,
        cameras=(rgb_camera, depth_camera, silent_camera),
        obs_cache={
            "cam/rgb_cam/rgb": rgb,
            "cam/depth_cam/depth": depth,
        },
    )

    assert dict(obs.cameras) == {
        rgb_camera: obs.cameras[rgb_camera],
        depth_camera: obs.cameras[depth_camera],
    }
    assert obs.cameras[rgb_camera].rgb is rgb
    assert obs.cameras[rgb_camera].depth is None
    assert obs.cameras[depth_camera].rgb is None
    assert obs.cameras[depth_camera].depth is depth
    assert silent_camera not in obs.cameras


def test_contact_keys_are_collected_and_other_keys_stay_in_raw() -> None:
    """contact/ record tensors populate contacts; raw keeps every live key."""
    handle = _handle()
    state = _state(handle)
    contact = torch.tensor([[True]])
    state["contact/left_foot"] = contact
    state["object/cube/pos"] = torch.zeros(1, 3)

    obs = _build(state, handle)

    assert obs.contacts["contact/left_foot"] is contact
    assert "object/cube/pos" in obs.raw
    assert "object/cube/pos" not in obs.contacts


def test_raw_is_the_live_state_without_bookkeeping_keys() -> None:
    """reset_env_ids/should_reset never reach raw; the cache stays separate."""
    handle = _handle()
    state = _state(handle)
    cache = {"joint_pos": torch.tensor([[0.25, 0.5]])}

    obs = _build(state, handle, obs_cache=cache)

    assert "reset_env_ids" not in obs.raw
    assert "should_reset" not in obs.raw
    assert set(obs.raw) == set(state) - {"reset_env_ids", "should_reset"}
    assert dict(obs.obs_cache) == cache


def test_raw_rejects_a_value_that_is_neither_tensor_nor_bool() -> None:
    """The live state mapping is typed; an unexpected value is refused."""
    handle = _handle()
    state = _state(handle)
    state["time_in_episode"] = 1.5

    with pytest.raises(ObservationError, match="time_in_episode"):
        _build(state, handle)


def test_multienv_projection_has_num_envs_rows() -> None:
    """Every projected record carries the declared env count, not one row."""
    handle = _handle()
    state = _state(handle, num_envs=2)

    obs = _build(state, handle, num_envs=2)

    record = obs.articulations[handle]
    assert obs.num_envs == 2
    assert record.joint_pos.shape == (2, 2)
    assert record.joint_pos_rel.shape == (2, 2)
    assert record.joint_vel.shape == (2, 2)
    assert record.applied_torque.shape == (2, 2)
    assert record.gravity_b.shape == (2, 3)
    assert record.base_quat_w.shape == (2, 4)


def test_joint_pos_row_count_is_the_declared_num_envs() -> None:
    """joint_pos is validated against num_envs, never used to infer the rows."""
    handle = _handle()
    state = _state(handle, num_envs=1)

    with pytest.raises(ObservationError, match="expected 2"):
        _build(state, handle, num_envs=2)


def test_reset_due_env_ids_crosses_as_its_own_field() -> None:
    """The performed and due masks are distinct fields with their own ids."""
    handle = _handle()
    state = _state(handle, num_envs=2)

    obs = _build(
        state,
        handle,
        num_envs=2,
        reset_env_ids=torch.tensor([0], dtype=torch.int64),
        reset_due_env_ids=torch.tensor([1], dtype=torch.int64),
    )

    assert obs.reset_env_ids.tolist() == [0]
    assert obs.reset_due_env_ids.tolist() == [1]
    assert obs.reset_due_env_ids.dtype == torch.int64


def test_terminated_mask_is_copied_when_valid() -> None:
    """A (N,) bool termination mask is carried into the record."""
    handle = _handle()
    state = _state(handle, num_envs=2)

    obs = _build(state, handle, num_envs=2, terminated=torch.tensor([False, True]))

    assert obs.terminated is not None
    assert obs.terminated.tolist() == [False, True]


@pytest.mark.parametrize("num_envs", [0, -1, True, 1.5])
def test_num_envs_is_validated(num_envs: int) -> None:
    """The declared env count is a positive int, never a bool or a float."""
    handle = _handle()
    state = _state(handle)

    with pytest.raises(ObservationError, match="num_envs"):
        _build(state, handle, num_envs=num_envs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"reset_env_ids": [0]}, "must be a torch.Tensor"),
        ({"reset_env_ids": torch.tensor([0, 0], dtype=torch.int64)}, "duplicate"),
        ({"reset_env_ids": torch.tensor([2], dtype=torch.int64)}, "outside the legal range"),
        ({"reset_env_ids": torch.tensor([0], dtype=torch.int32)}, "1-D int64"),
        ({"reset_env_ids": torch.tensor([[0]], dtype=torch.int64)}, "1-D int64"),
        ({"reset_due_env_ids": torch.tensor([2], dtype=torch.int64)}, "reset_due_env_ids"),
    ],
)
def test_reset_masks_are_validated(kwargs: dict[str, object], match: str) -> None:
    """Both id masks are 1-D int64, unique and inside [0, num_envs)."""
    handle = _handle()
    state = _state(handle, num_envs=2)

    with pytest.raises(ObservationError, match=match):
        _build(state, handle, num_envs=2, **kwargs)


@pytest.mark.parametrize(
    ("terminated", "match"),
    [
        (torch.tensor([0.0, 0.0]), "must be a \\(2,\\) bool tensor"),
        (torch.zeros(1, dtype=torch.bool), "must be a \\(2,\\) bool tensor"),
        (torch.zeros(2, 2, dtype=torch.bool), "must be a \\(2,\\) bool tensor"),
        (torch.zeros(2, dtype=torch.int64), "must be a \\(2,\\) bool tensor"),
    ],
)
def test_terminated_mask_shape_and_dtype_are_validated(
    terminated: torch.Tensor, match: str
) -> None:
    """The termination mask is None or exactly (num_envs,) bool."""
    handle = _handle()
    state = _state(handle, num_envs=2)

    with pytest.raises(ObservationError, match=match):
        _build(state, handle, num_envs=2, terminated=terminated)


def test_optional_cached_rows_follow_num_envs() -> None:
    """A projected optional cache tensor also carries num_envs rows."""
    handle = _handle()
    state = _state(handle, num_envs=2)
    prefix = f"articulation/{handle.name}"

    with pytest.raises(ObservationError, match="env rows"):
        _build(
            state,
            handle,
            num_envs=2,
            obs_cache={f"{prefix}/prev_action": torch.zeros(1, 2)},
        )
