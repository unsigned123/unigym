"""ResetPrimitives.write_freejoint: yaw compatibility, quat and velocity math."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from unigym.backends.shared.mjfamily.props.reset_face import build_reset_primitives
from unigym.core import quat as quat_math

_FREE_BODY_XML = """
<mujoco model="free_quat_probe">
  <compiler angle="radian"/>
  <option timestep="0.0001" gravity="0 0 0"/>
  <worldbody>
    <body name="crate" pos="0 0 1">
      <freejoint name="crate_free"/>
      <geom name="crate_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


class _AddressTable:
    """The minimal ``prop_backend.free_addresses`` face the primitive uses."""

    def __init__(self, body_id: int, qadr: int, vadr: int) -> None:
        self._body_id = body_id
        self._qadr = qadr
        self._vadr = vadr

    def free_addresses(self, name: str):
        if name != "crate":
            raise KeyError(name)
        return self._body_id, self._qadr, self._vadr


def _primitives():
    model = mujoco.MjModel.from_xml_string(_FREE_BODY_XML)
    data = mujoco.MjData(model)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "crate_free")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "crate")
    qadr = int(model.jnt_qposadr[joint])
    vadr = int(model.jnt_dofadr[joint])
    env = SimpleNamespace(
        mj_model=model,
        _active_data=data,
        _settle_substeps=lambda substeps: None,
        _prop_extents=lambda name: (0.05, 0.1),
        prop_manager=None,
        prop_backend=_AddressTable(body, qadr, vadr),
    )
    return build_reset_primitives(env), model, data, body, qadr, vadr


def test_yaw_call_is_bit_exact_with_the_previous_pure_yaw_write() -> None:
    """The placement sampler's call shape keeps the original float bytes."""
    prims, _model, data, _body, qadr, vadr = _primitives()
    yaw = 0.37

    prims.write_freejoint("crate", [0.1, 0.2, 0.3], yaw)

    expected_quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    assert np.array_equal(data.qpos[qadr + 3 : qadr + 7], expected_quat)
    assert np.array_equal(data.qpos[qadr : qadr + 3], np.array([0.1, 0.2, 0.3]))
    assert np.array_equal(data.qvel[vadr : vadr + 6], np.zeros(6))


def test_quat_call_writes_the_exact_normalized_wxyz() -> None:
    """A non-unit wxyz quat is normalized; order never flips."""
    prims, _model, data, _body, qadr, vadr = _primitives()

    prims.write_freejoint("crate", [0.0, 0.0, 0.5], quat_wxyz=[2.0, 0.0, 0.0, 0.0])

    assert np.array_equal(data.qpos[qadr + 3 : qadr + 7], np.array([1.0, 0.0, 0.0, 0.0]))

    prims.write_freejoint("crate", [0.0, 0.0, 0.5], quat_wxyz=[1.0, 1.0, 0.0, 0.0])

    expected = np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0)
    assert np.allclose(data.qpos[qadr + 3 : qadr + 7], expected, atol=0.0, rtol=0.0)
    assert np.linalg.norm(data.qpos[qadr + 3 : qadr + 7]) == pytest.approx(1.0)
    assert np.array_equal(data.qvel[vadr : vadr + 6], np.zeros(6))


@pytest.mark.parametrize(
    ("quat", "message"),
    [
        ([0.0, 0.0, 0.0, 0.0], "zero norm"),
        ([1.0, np.inf, 0.0, 0.0], "must be finite"),
        ([1.0, 0.0, 0.0], r"shape \(4,\)"),
    ],
)
def test_bad_quat_fails_with_the_prop_name(quat, message: str) -> None:
    """Every quat validation error names the prop."""
    prims, *_ = _primitives()

    with pytest.raises(ValueError, match=message) as excinfo:
        prims.write_freejoint("crate", [0.0, 0.0, 0.5], quat_wxyz=quat)

    assert "'crate'" in str(excinfo.value)


def test_yaw_and_quat_together_fail() -> None:
    """Exactly one orientation source is accepted."""
    prims, *_ = _primitives()

    with pytest.raises(ValueError, match="exactly one") as excinfo:
        prims.write_freejoint("crate", [0.0, 0.0, 0.5], 0.1, quat_wxyz=[1.0, 0.0, 0.0, 0.0])

    assert "'crate'" in str(excinfo.value)


def test_neither_yaw_nor_quat_fails() -> None:
    """The orientation source is mandatory."""
    prims, *_ = _primitives()

    with pytest.raises(ValueError, match="exactly one"):
        prims.write_freejoint("crate", [0.0, 0.0, 0.5])


@pytest.mark.parametrize(
    ("pos", "message"),
    [
        ([0.0, 0.0], r"shape \(3,\)"),
        ([0.0, 0.0, np.nan], "must be finite"),
    ],
)
def test_bad_pos_fails_with_the_prop_name(pos, message: str) -> None:
    """Position shape/finite validation is explicit."""
    prims, *_ = _primitives()

    with pytest.raises(ValueError, match=message) as excinfo:
        prims.write_freejoint("crate", pos, yaw=0.0)

    assert "'crate'" in str(excinfo.value)


def test_absent_velocities_write_zero_qvel() -> None:
    """No velocity arguments -> the freejoint's whole 6-vector is zero."""
    prims, _model, data, _body, qadr, vadr = _primitives()
    data.qvel[vadr : vadr + 6] = [9.0] * 6

    prims.write_freejoint("crate", [0.0, 0.0, 0.5], quat_wxyz=[1.0, 0.0, 0.0, 0.0])

    assert np.array_equal(data.qvel[vadr : vadr + 6], np.zeros(6))
    assert np.array_equal(data.qpos[qadr + 3 : qadr + 7], np.array([1.0, 0.0, 0.0, 0.0]))


def test_supplied_velocities_write_world_linear_and_body_angular() -> None:
    """qvel linear is world; angular is R.T @ world (body frame)."""
    prims, _model, data, _body, _qadr, vadr = _primitives()
    quat = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]

    prims.write_freejoint(
        "crate",
        [0.0, 0.0, 0.5],
        quat_wxyz=quat,
        lin_vel_w=[0.1, -0.2, 0.3],
        ang_vel_w=[0.0, 0.0, 0.4],
    )

    assert np.allclose(data.qvel[vadr : vadr + 3], [0.1, -0.2, 0.3], atol=0.0, rtol=0.0)
    expected = quat_math.to_rotmat(np.asarray(quat)).T @ np.array([0.0, 0.0, 0.4])
    assert np.allclose(data.qvel[vadr + 3 : vadr + 6], expected, atol=0.0, rtol=0.0)
    assert data.qvel[vadr + 3 : vadr + 6].tolist() == pytest.approx([0.0, 0.4, 0.0])


@pytest.mark.parametrize("field", ["lin_vel_w", "ang_vel_w"])
def test_bad_velocity_fails(field: str) -> None:
    """Velocity shape/finite validation is explicit and prop-name bearing."""
    prims, *_ = _primitives()

    with pytest.raises(ValueError, match=field) as excinfo:
        prims.write_freejoint(
            "crate",
            [0.0, 0.0, 0.5],
            quat_wxyz=[1.0, 0.0, 0.0, 0.0],
            **{field: [0.0, 0.0]},
        )

    assert "'crate'" in str(excinfo.value)


def test_tensor_inputs_ride_the_numpy_path() -> None:
    """Torch tensors are accepted (detached CPU in the shared numpy path)."""
    torch = pytest.importorskip("torch")
    prims, _model, data, _body, qadr, vadr = _primitives()

    prims.write_freejoint(
        "crate",
        torch.tensor([0.0, 0.0, 0.5]),
        quat_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        lin_vel_w=torch.tensor([0.1, 0.0, 0.0]),
    )

    assert np.allclose(data.qpos[qadr + 3 : qadr + 7], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(data.qvel[vadr : vadr + 3], [0.1, 0.0, 0.0])


def test_full_quat_survives_forward_and_a_mujoco_step() -> None:
    """The written wxyz is what MuJoCo derives and integrates."""
    prims, model, data, body, qadr, _vadr = _primitives()
    quat = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
    pos = np.array([0.2, -0.1, 0.75])

    prims.write_freejoint("crate", pos, quat_wxyz=quat * 3.0)
    mujoco.mj_forward(model, data)

    assert np.allclose(data.xpos[body], pos, atol=1e-12)
    assert np.allclose(data.xquat[body], quat, atol=1e-12)

    mujoco.mj_step(model, data)

    assert np.allclose(data.qpos[qadr : qadr + 3], pos, atol=1e-12)
    assert np.allclose(data.qpos[qadr + 3 : qadr + 7], quat, atol=1e-12)
    assert abs(float(np.dot(data.xquat[body], quat))) == pytest.approx(1.0)
