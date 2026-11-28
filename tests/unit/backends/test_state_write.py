"""Unit contract for the optional state-write helpers and the Run facade."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unigym._typing import JsonValue
from unigym.capabilities.traits import (
    ISAACLAB2_TRAITS,
    ISAACLAB3_NEWTON_TRAITS,
    ISAACLAB3_PHYSX_TRAITS,
    MJWARP_TRAITS,
    MUJOCO_TRAITS,
    Trait,
)
from unigym.env.backends.base import (
    BackendCapabilities,
    normalize_state_env_ids,
    normalize_state_quat,
    validate_state_tensor,
)
from unigym.run.api import Run


def test_normalize_env_ids_none_selects_every_env() -> None:
    """env_ids=None is the whole batch, returned as int64."""
    ids = normalize_state_env_ids(None, 3)

    assert ids.dtype == torch.int64
    assert ids.tolist() == [0, 1, 2]


def test_normalize_env_ids_preserves_input_order_and_rejects_out_of_range() -> None:
    """Subset order is preserved; ids outside the batch fail fast."""
    assert normalize_state_env_ids([2, 0], 3).tolist() == [2, 0]
    assert normalize_state_env_ids(torch.tensor([], dtype=torch.int64), 3).numel() == 0
    with pytest.raises(ValueError, match="outside"):
        normalize_state_env_ids([0, 3], 3)


def test_normalize_env_ids_rejects_floating_and_multi_dimensional_ids() -> None:
    """Float/bool ids and non-1-D selections are teaching errors."""
    with pytest.raises(TypeError, match="integer ids"):
        normalize_state_env_ids(torch.tensor([0.0]), 3)
    with pytest.raises(ValueError, match="must be 1-D"):
        normalize_state_env_ids(torch.zeros(1, 1, dtype=torch.int64), 3)


def test_validate_state_tensor_checks_shape_dtype_device_and_finiteness() -> None:
    """The boundary validates shape, dtype, device and finite values."""
    validate_state_tensor(torch.zeros(2, 3), field="pos_w", rows=2, cols=3, device="cpu")
    with pytest.raises(ValueError, match=r"pos_w shape"):
        validate_state_tensor(torch.zeros(2, 2), field="pos_w", rows=2, cols=3, device="cpu")
    with pytest.raises(TypeError, match="floating tensor"):
        validate_state_tensor(
            torch.zeros(2, 3, dtype=torch.int64),
            field="pos_w",
            rows=2,
            cols=3,
            device="cpu",
        )
    with pytest.raises(ValueError, match="contains non-finite"):
        validate_state_tensor(
            torch.tensor([[0.0, float("inf"), 0.0]]),
            field="pos_w",
            rows=1,
            cols=3,
            device="cpu",
        )


def test_normalize_state_quat_normalizes_and_rejects_zero_norm() -> None:
    """Quaternions are normalized by core; a zero norm is a teaching error."""
    unit = normalize_state_quat(torch.tensor([[2.0, 0.0, 0.0, 0.0]]), rows=1, device="cpu")

    assert unit[0].tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="zero-norm"):
        normalize_state_quat(torch.zeros(1, 4), rows=1, device="cpu")


def test_state_write_is_an_optional_trait() -> None:
    """Every registered backend declares STATE_WRITE after the C2 trait pass."""
    assert Trait.STATE_WRITE in MUJOCO_TRAITS
    assert Trait.STATE_WRITE in ISAACLAB2_TRAITS
    assert Trait.STATE_WRITE in MJWARP_TRAITS
    assert Trait.STATE_WRITE in ISAACLAB3_PHYSX_TRAITS
    assert Trait.STATE_WRITE in ISAACLAB3_NEWTON_TRAITS

    assert Trait.STATE_WRITE in BackendCapabilities(state_write=True).traits()
    assert Trait.STATE_WRITE not in BackendCapabilities().traits()


def _cfg() -> dict[str, JsonValue]:
    """A small valid config; ``build`` is stubbed in these unit tests."""
    return {
        "run_name": "r",
        "seed": 0,
        "env": {"backend": "mujoco", "num_envs": 1},
        "world_base": {"mjcf": "model.xml"},
        "collect": {"steps": 1},
        "articulations": [{"name": "arm", "joints": [{"name": "shoulder"}]}],
        "action_spec": [
            {
                "name": "arm",
                "articulation": "arm",
                "joints": ["shoulder"],
                "target": "position",
                "role": "body",
            }
        ],
        "policies": [{"adapter": "manual", "groups": [["arm", "arm"]]}],
        "observations": {"robot": ["arm"]},
    }


class _StateWriteRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, dict]] = []

    def set_root_state(self, env_ids, **kwargs):
        self.calls.append(("root", env_ids, kwargs))
        return torch.tensor([7], dtype=torch.int64)

    def set_joint_state(self, env_ids, **kwargs):
        self.calls.append(("joint", env_ids, kwargs))
        return torch.tensor([1, 0], dtype=torch.int64)

    def settle(self, env_ids):
        self.calls.append(("settle", env_ids, {}))
        return torch.tensor([0], dtype=torch.int64)


def test_run_delegates_state_writes_and_never_settles_implicitly(monkeypatch) -> None:
    """The facade forwards exactly one call per method; no hidden settle."""
    run = Run(_cfg())
    env = _StateWriteRecorder()
    monkeypatch.setattr(run, "build", lambda inspect=False: SimpleNamespace(env=env))
    zero = torch.zeros(1, 3)

    root_ids = run.set_root_state(
        None,
        name="arm",
        pos_w=zero,
        quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        lin_vel_w=zero,
        ang_vel_w=zero,
    )
    joint_ids = run.set_joint_state(
        [0, 1],
        articulation="arm",
        joint_names=("shoulder",),
        position=torch.zeros(2, 1),
    )
    settle_ids = run.settle(None)

    assert root_ids.tolist() == [7]
    assert joint_ids.tolist() == [1, 0]
    assert settle_ids.tolist() == [0]
    assert [call[0] for call in env.calls] == ["root", "joint", "settle"]
    assert env.calls[0][1] is None
    assert env.calls[1][1] == [0, 1]


def test_run_state_write_delegates_for_isaaclab3_physx(monkeypatch) -> None:
    """The C2 trait pass lights STATE_WRITE up for isaaclab3_physx."""
    cfg = _cfg()
    cfg["env"] = {"backend": "isaaclab3_physx", "num_envs": 1}
    cfg.pop("world_base")
    run = Run(cfg)
    env = _StateWriteRecorder()
    monkeypatch.setattr(run, "build", lambda inspect=False: SimpleNamespace(env=env))

    assert run.settle(None).tolist() == [0]
    assert env.calls[0][0] == "settle"


def test_run_state_write_delegates_for_mjwarp(monkeypatch) -> None:
    """mjwarp now declares STATE_WRITE, so the facade forwards to its env."""
    cfg = _cfg()
    cfg["env"] = {"backend": "mjwarp", "num_envs": 1}
    run = Run(cfg)
    env = _StateWriteRecorder()
    monkeypatch.setattr(run, "build", lambda inspect=False: SimpleNamespace(env=env))

    assert run.settle(None).tolist() == [0]
    assert env.calls[0][0] == "settle"
