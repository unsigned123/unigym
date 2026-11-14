"""``Control.position_pd`` validation and immutable control-spec contract."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from unigym.sim import Control, ControlSpec, SimulationConfigError


def test_position_pd_derives_the_face_name_from_the_joint_set() -> None:
    """A joint-set face uses the set name as its action face by default."""
    spec = Control.position_pd(joint_set="body", kp=40.0, kd=1.0)

    face = spec.faces["body"]
    assert face.name == "body"
    assert face.drive == "position_pd"
    assert face.target == "position"
    assert face.joint_set == "body"
    assert face.joints is None
    assert face.gains["kp"] == 40.0
    assert face.gains["kd"] == 1.0


def test_position_pd_keeps_explicit_joint_order_and_face() -> None:
    """An explicit-joint face keeps the caller's column order."""
    spec = Control.position_pd(joints=["elbow", "shoulder"], face="arm", kp=1.0, kd=0.1)

    face = spec.faces["arm"]
    assert face.joints == ("elbow", "shoulder")
    assert face.joint_set is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"joint_set": "body", "joints": ["shoulder"]},
    ],
)
def test_position_pd_requires_exactly_one_address_form(kwargs: dict[str, object]) -> None:
    """Neither or both address forms is a construction error."""
    with pytest.raises(SimulationConfigError, match="exactly one of joint_set or joints"):
        Control.position_pd(**kwargs, kp=1.0, kd=0.1)


def test_position_pd_explicit_joints_require_a_face() -> None:
    """A joints-only declaration cannot guess an action-face name."""
    with pytest.raises(SimulationConfigError, match="requires an explicit face"):
        Control.position_pd(joints=["shoulder"], kp=1.0, kd=0.1)


@pytest.mark.parametrize("joint_set", ["", "   "])
def test_position_pd_rejects_an_empty_joint_set(joint_set: str) -> None:
    """A joint-set name must be non-empty after trimming."""
    with pytest.raises(SimulationConfigError, match="non-empty"):
        Control.position_pd(joint_set=joint_set, kp=1.0, kd=0.1)


def test_position_pd_rejects_empty_and_duplicate_joint_lists() -> None:
    """Explicit joints must be non-empty and duplicate-free."""
    with pytest.raises(SimulationConfigError, match="must not be empty"):
        Control.position_pd(joints=[], face="arm", kp=1.0, kd=0.1)
    with pytest.raises(SimulationConfigError, match="duplicate joint names"):
        Control.position_pd(joints=["shoulder", "shoulder"], face="arm", kp=1.0, kd=0.1)


def test_position_pd_requires_kp_and_kd_keywords() -> None:
    """Gains are required keyword arguments, not hidden defaults."""
    with pytest.raises(TypeError):
        Control.position_pd(joint_set="body", kd=1.0)
    with pytest.raises(TypeError):
        Control.position_pd(joint_set="body", kp=1.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_position_pd_rejects_non_finite_scalar_gains(value: float) -> None:
    """A scalar gain must be finite."""
    with pytest.raises(SimulationConfigError, match="must be finite"):
        Control.position_pd(joint_set="body", kp=value, kd=1.0)


def test_position_pd_accepts_per_joint_gain_maps() -> None:
    """A non-empty per-joint gain map is kept per joint."""
    spec = Control.position_pd(
        joints=["shoulder", "elbow"],
        face="arm",
        kp={"shoulder": 40.0, "elbow": 20.0},
        kd={"shoulder": 1.0, "elbow": 0.5},
    )

    assert spec.faces["arm"].gains["kp"] == {"shoulder": 40.0, "elbow": 20.0}
    assert spec.faces["arm"].gains["kd"] == {"shoulder": 1.0, "elbow": 0.5}


def test_position_pd_rejects_empty_or_non_finite_gain_maps() -> None:
    """Gain maps must be non-empty with finite values."""
    with pytest.raises(SimulationConfigError, match="must not be an empty mapping"):
        Control.position_pd(joint_set="body", kp={}, kd=1.0)
    with pytest.raises(SimulationConfigError, match="must be finite"):
        Control.position_pd(joint_set="body", kp={"shoulder": math.nan}, kd=1.0)


def test_position_pd_rejects_non_finite_q0_overrides() -> None:
    """Standby overrides are finite numbers."""
    with pytest.raises(SimulationConfigError, match="q0"):
        Control.position_pd(joint_set="body", q0={"shoulder": math.inf}, kp=1.0, kd=0.1)


def test_control_specs_are_immutable_and_read_only() -> None:
    """Frozen faces and read-only mappings cannot be mutated after construction."""
    spec = Control.position_pd(joint_set="body", kp=1.0, kd=0.1)
    face = spec.faces["body"]

    with pytest.raises(FrozenInstanceError):
        face.name = "other"
    with pytest.raises(TypeError):
        spec.faces["other"] = face


def test_duplicate_face_names_are_rejected() -> None:
    """Two faces with the same name cannot share one ControlSpec."""
    face = Control.position_pd(joint_set="body", kp=1.0, kd=0.1).faces["body"]

    with pytest.raises(SimulationConfigError, match="duplicate face name"):
        ControlSpec.from_faces((face, face))


def test_control_spec_faces_must_match_their_keys() -> None:
    """A face mapping key must equal the face's own name."""
    face = Control.position_pd(joint_set="body", kp=1.0, kd=0.1).faces["body"]

    with pytest.raises(SimulationConfigError, match="does not match"):
        ControlSpec({"other": face})
