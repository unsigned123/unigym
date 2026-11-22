"""MuJoCo runtime-query object/root/COM state behavior over real free props."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

from unigym.backends.mujoco.query import MujocoQuery
from unigym.core import quat as quat_math
from unigym.props import build_prop_manager
from unigym.query import (
    ComRef,
    ObjectRef,
    Query,
    QueryCapabilityError,
    QueryReferenceError,
)

_QUAT_90_Y = np.array([0.7071067811865476, 0.0, 0.7071067811865476, 0.0])


def _query(env) -> MujocoQuery:
    return MujocoQuery(env)


def _prop_manager(names: tuple[str, ...] = ("crate",), num_envs: int = 1):
    types = [
        {
            "name": name,
            "type": "box",
            "size": [0.1, 0.1, 0.1],
            "aabb": {
                "center": [0.0, 0.0, 0.05],
                "half_extents": [0.05, 0.05, 0.05],
            },
            "placement": {
                "mode": "fixed",
                "x_range": [0.5, 0.5],
                "y_range": [0.0, 0.0],
                "z_range": [0.3, 0.3],
            },
        }
        for name in names
    ]
    return build_prop_manager(
        {"types": types},
        num_envs=num_envs,
        env_spacing=2.0,
        device="cpu",
        backend="mujoco",
    )


def _free_articulation_manager(tmp_path):
    asset = tmp_path / "laptop_free.xml"
    asset.write_text(
        """<mujoco model="laptop">
  <worldbody>
    <body name="laptop_root">
      <freejoint name="laptop_free"/>
      <geom name="laptop_geom" type="box" size="0.1 0.1 0.02"/>
      <body name="lid" pos="0 0 0.02">
        <joint name="lid_hinge" type="hinge" axis="0 1 0"/>
        <geom name="lid_geom" type="box" size="0.1 0.1 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>""",
        encoding="utf-8",
    )
    return build_prop_manager(
        {
            "types": [
                {
                    "name": "laptop",
                    "type": "free_articulation",
                    "asset": str(asset),
                    "joints": {"lid_hinge": {"mode": "passive"}},
                    "placement": {
                        "mode": "fixed",
                        "x_range": [1.0, 1.0],
                        "y_range": [0.0, 0.0],
                        "z_range": [0.5, 0.5],
                    },
                }
            ]
        },
        num_envs=1,
        env_spacing=2.0,
        device="cpu",
        backend="mujoco",
    )


def _write_prop(
    env,
    name: str,
    *,
    env_id: int = 0,
    pos: np.ndarray,
    quat: np.ndarray,
    lin: np.ndarray,
    ang_local: np.ndarray,
) -> tuple[int, int, int]:
    body_id, qadr, vadr = env.prop_backend.free_addresses(name)
    data = env.mj_datas[env_id]
    data.qpos[qadr : qadr + 3] = pos
    data.qpos[qadr + 3 : qadr + 7] = quat
    data.qvel[vadr : vadr + 3] = lin
    data.qvel[vadr + 3 : vadr + 6] = ang_local
    mujoco.mj_forward(env.mj_model, data)
    return body_id, qadr, vadr


def test_object_state_reads_qpos_qvel_with_world_angular_conversion(tiny_env_factory) -> None:
    """Object pose comes from qpos and local angular qvel converts to world."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    query = _query(env)
    pos = np.array([0.4, 0.1, 0.5])
    lin = np.array([0.1, 0.2, 0.3])
    ang_local = np.array([0.3, -0.4, 0.5])
    _write_prop(
        env,
        "crate",
        pos=pos,
        quat=_QUAT_90_Y,
        lin=lin,
        ang_local=ang_local,
    )

    state = query.state.object_state(ObjectRef("crate"))
    body_id, qadr, vadr = env.prop_backend.free_addresses("crate")
    data = env.mj_datas[0]

    np.testing.assert_allclose(state.pos_w[0].numpy(), data.qpos[qadr : qadr + 3])
    np.testing.assert_allclose(state.quat_wxyz[0].numpy(), data.qpos[qadr + 3 : qadr + 7])
    np.testing.assert_allclose(state.lin_vel_w[0].numpy(), data.qvel[vadr : vadr + 3])
    expected = quat_math.to_rotmat(_QUAT_90_Y) @ data.qvel[vadr + 3 : vadr + 6]
    np.testing.assert_allclose(state.ang_vel_w[0].numpy(), expected)
    assert not np.allclose(state.ang_vel_w[0].numpy(), data.qvel[vadr + 3 : vadr + 6])


def test_object_state_matches_the_state_write_roundtrip(tiny_env_factory) -> None:
    """A prop written through set_root_state reads back the same world twist."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    pos = [0.4, 0.1, 0.5]
    quat = torch.from_numpy(_QUAT_90_Y).unsqueeze(0)
    lin = [0.1, 0.2, 0.3]
    ang = [0.0, 0.0, 0.4]
    env.set_root_state(
        None,
        name="crate",
        pos_w=torch.tensor([pos]),
        quat_wxyz=quat,
        lin_vel_w=torch.tensor([lin]),
        ang_vel_w=torch.tensor([ang]),
    )
    env.settle(None)

    state = _query(env).state.object_state(ObjectRef("crate"))
    data = env.mj_datas[0]
    _, _, vadr = env.prop_backend.free_addresses("crate")

    np.testing.assert_allclose(state.pos_w[0].numpy(), pos, atol=1e-12)
    np.testing.assert_allclose(state.quat_wxyz[0].numpy(), _QUAT_90_Y, atol=1e-12)
    np.testing.assert_allclose(state.lin_vel_w[0].numpy(), lin, atol=1e-12)
    np.testing.assert_allclose(state.ang_vel_w[0].numpy(), ang, atol=1e-12)
    expected_local = quat_math.to_rotmat(_QUAT_90_Y).T @ np.asarray(ang)
    np.testing.assert_allclose(data.qvel[vadr + 3 : vadr + 6], expected_local, atol=1e-12)


def test_object_com_state_matches_xipos_and_the_rigid_body_identity(tiny_env_factory) -> None:
    """COM velocity equals v_origin + omega x (xipos - xpos) and jacBodyCom @ qvel."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    query = _query(env)
    body_id, _, _ = _write_prop(
        env,
        "crate",
        pos=np.array([0.4, 0.1, 0.5]),
        quat=_QUAT_90_Y,
        lin=np.array([0.1, 0.2, 0.3]),
        ang_local=np.array([0.3, -0.4, 0.5]),
    )
    data = env.mj_datas[0]

    state = query.state.object_state(ObjectRef("crate"))
    com = query.state.object_com_state(ObjectRef("crate"))

    np.testing.assert_allclose(com.pos_w[0].numpy(), data.xipos[body_id])
    origin = state.pos_w[0].numpy()
    offset = np.asarray(data.xipos[body_id]) - np.asarray(data.xpos[body_id])
    expected = state.lin_vel_w[0].numpy() + np.cross(state.ang_vel_w[0].numpy(), offset)
    np.testing.assert_allclose(com.lin_vel_w[0].numpy(), expected, atol=1e-12)

    jacp = np.zeros((3, env.mj_model.nv), dtype=np.float64)
    jacr = np.zeros((3, env.mj_model.nv), dtype=np.float64)
    mujoco.mj_jacBodyCom(env.mj_model, data, jacp, jacr, body_id)
    np.testing.assert_allclose(com.lin_vel_w[0].numpy(), jacp @ data.qvel, atol=1e-12)
    assert np.linalg.norm(np.asarray(data.xipos[body_id]) - origin) > 0.0


def test_unknown_and_articulation_props_are_typed_errors(tiny_env_factory, tmp_path) -> None:
    """Unknown names are reference errors; non-rigid props are capability errors."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    query = _query(env)
    with pytest.raises(QueryReferenceError, match=r"ghost.*crate"):
        query.state.object_state(ObjectRef("ghost"))
    with pytest.raises(QueryReferenceError, match="use state.root_state/com_state"):
        query.state.object_state(ObjectRef("tiny"))

    articulated = tiny_env_factory(prop_manager=_free_articulation_manager(tmp_path))
    with pytest.raises(QueryCapabilityError, match="rigid free props only"):
        _query(articulated).state.object_com_state(ObjectRef("laptop"))


def test_object_reads_without_a_prop_backend_are_reference_errors(tiny_env) -> None:
    """No declared props cannot silently answer an object read."""
    query = _query(tiny_env)

    with pytest.raises(QueryReferenceError, match="no props are declared"):
        query.state.object_state(ObjectRef("crate"))
    with pytest.raises(QueryReferenceError, match="no props are declared"):
        query.state.object_com_state(ObjectRef("crate"))


def test_object_state_multi_env_selection_order_empty_and_detached(tiny_env_factory) -> None:
    """Multi-env reads preserve order, empty selects zero rows and results detach."""
    env = tiny_env_factory(num_envs=3, prop_manager=_prop_manager(num_envs=3))
    query = _query(env)
    env.set_root_state(
        [0],
        name="crate",
        pos_w=torch.tensor([[0.1, 0.0, 0.3]]),
        quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        lin_vel_w=torch.zeros(1, 3),
        ang_vel_w=torch.zeros(1, 3),
    )
    env.set_root_state(
        [2],
        name="crate",
        pos_w=torch.tensor([[0.9, 0.0, 0.3]]),
        quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        lin_vel_w=torch.zeros(1, 3),
        ang_vel_w=torch.zeros(1, 3),
    )
    env.settle(None)

    state = query.state.object_state(ObjectRef("crate"), env=[2, 0, 2])
    assert tuple(state.env_ids.tolist()) == (2, 0, 2)
    np.testing.assert_allclose(
        state.pos_w.numpy()[:, 0],
        [0.9, 0.1, 0.9],
        atol=1e-12,
    )

    empty = query.state.object_state(ObjectRef("crate"), env=[])
    assert tuple(empty.env_ids.shape) == (0,)
    assert tuple(empty.pos_w.shape) == (0, 3)
    assert tuple(query.state.object_com_state(ObjectRef("crate"), env=[]).lin_vel_w.shape) == (
        0,
        3,
    )

    live_qpos = env.mj_datas[0].qpos.copy()
    live_qvel = env.mj_datas[0].qvel.copy()
    live_xipos = env.mj_datas[0].xipos.copy()
    again = query.state.object_state(ObjectRef("crate"), env=0)
    again.pos_w[0, 0] = 123.0
    again.lin_vel_w[0, 0] = 456.0
    np.testing.assert_array_equal(env.mj_datas[0].qpos, live_qpos)
    np.testing.assert_array_equal(env.mj_datas[0].qvel, live_qvel)
    np.testing.assert_array_equal(env.mj_datas[0].xipos, live_xipos)


def _two_articulation_prop_env(tmp_path, tiny_mjcf, prop_manager):
    """Build a two-articulation run (``merged is None``) with one free prop."""
    from unigym.actuators import ActuationGroup
    from unigym.actuators.ir import resolve_actuation_groups
    from unigym.adapters.manual import ManualAdapter
    from unigym.backends.mujoco.env import MujocoEnv
    from unigym.env.obs import ObservationConfig
    from unigym.robots.base import BodyBinding, RobotProfile
    from unigym.runtime.actions import OrderedActionGroup, OrderedActionSpec, PolicyBinding
    from unigym.runtime.controllers import build_control_runtimes
    from unigym.scenes.articulation import ArticulationSpec

    base = tmp_path / "base_scene.xml"
    base.write_text(
        """<?xml version="1.0"?>
<mujoco model="base_scene">
  <compiler angle="radian"/>
  <worldbody>
    <geom name="ground" type="plane" size="4 4 0.1"/>
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    joints = ("elbow", "shoulder")
    source = "\n".join(
        line
        for line in tiny_mjcf.read_text(encoding="utf-8").splitlines()
        if 'geom name="floor"' not in line
    )
    specs = []
    profiles = {}
    groups = []
    for index, name in enumerate(("left", "right")):
        sign = -1.0 if index == 0 else 1.0
        robot = tmp_path / f"tiny_arm_{name}.xml"
        robot.write_text(
            source.replace('name="base_geom"', f'name="{name}_base_geom"')
            .replace('name="link_geom"', f'name="{name}_link_geom"')
            .replace('name="tip_geom"', f'name="{name}_tip_geom"'),
            encoding="utf-8",
        )
        profiles[name] = RobotProfile(
            name=name,
            joint_sets={"body": joints},
            default_joint_pos={joint: 0.0 for joint in joints},
            assets=None,
            extra_action_sets={},
            base_body="base",
            foot_bodies=(),
            body_sets={},
            body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
        )
        actuation = ActuationGroup(
            name="body",
            law="standard_pd",
            joints=joints,
            gains={"kp": 20.0, "kd": 1.0},
        )
        specs.append(
            ArticulationSpec(
                name=name,
                model={"mjcf": str(robot)},
                spawn={"pos": [sign, 0.0, 0.8]},
                actuation=(actuation,),
            )
        )
        groups.append(
            OrderedActionGroup(
                name=name,
                articulation=name,
                joints=joints,
                target="position",
                role="body",
            )
        )
    plans = {
        spec.name: resolve_actuation_groups(
            spec.actuation,
            articulation=spec.name,
            joints=joints,
        )
        for spec in specs
    }
    controls = build_control_runtimes(
        OrderedActionSpec(tuple(groups)),
        (PolicyBinding(adapter="manual", groups=tuple((g.articulation, g.name) for g in groups)),),
        tuple(specs),
        profiles,
        plans,
    )
    return MujocoEnv(
        robot_profile=profiles["left"],
        adapter=ManualAdapter(),
        num_envs=1,
        episode_length_s=100.0,
        decimation=4,
        sim_dt=0.005,
        world_base={"mjcf": str(base)},
        backend_params={},
        backend_options={"compose": {"auto_prefix": True}},
        device="cpu",
        articulations=tuple(specs),
        controls=controls,
        profiles=profiles,
        actuation_plans=plans,
        prop_manager=prop_manager,
        observations=ObservationConfig(robot=("left", "right")),
    )


def test_object_resolution_works_without_a_merged_ir(tmp_path, tiny_mjcf) -> None:
    """A multi-articulation run with merged=None still resolves a free prop."""
    env = _two_articulation_prop_env(tmp_path, tiny_mjcf, _prop_manager())
    try:
        assert env.model_set.merged is None
        query = _query(env)
        state = query.state.object_state(ObjectRef("crate"))
        com = query.state.object_com_state(ObjectRef("crate"))
        _, qadr, _ = env.prop_backend.free_addresses("crate")

        np.testing.assert_allclose(state.pos_w[0].numpy(), env.mj_datas[0].qpos[qadr : qadr + 3])
        assert tuple(com.pos_w.shape) == (1, 3)
    finally:
        env.close()


def test_object_capabilities_are_native_and_jacobian_object_is_native(
    tiny_env_factory,
) -> None:
    """P2-A/P2-B declare the object state reads and Jacobian native."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    query = _query(env)

    assert isinstance(query, Query)
    assert query.capabilities.supports("state.root_state") == "native"
    assert query.capabilities.supports("state.com_state") == "native"
    assert query.capabilities.supports("state.object_state") == "native"
    assert query.capabilities.supports("state.object_com_state") == "native"
    assert query.capabilities.supports("jacobian.object") == "native"


def test_object_com_state_rejects_a_non_world_frame(tiny_env_factory) -> None:
    """Only the world frame is representable; no silent frame substitution."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    query = _query(env)

    with pytest.raises(QueryCapabilityError, match="state.com_state.*local"):
        query.state.com_state(ComRef("tiny", body="base"), frame="local")


def _fixed_articulation_manager(tmp_path):
    asset = tmp_path / "laptop_anchored.xml"
    asset.write_text(
        """<mujoco model="laptop">
  <worldbody>
    <body name="laptop_root">
      <geom name="laptop_geom" type="box" size="0.1 0.1 0.02"/>
      <body name="lid" pos="0 0 0.02">
        <joint name="lid_hinge" type="hinge" axis="0 1 0"/>
        <geom name="lid_geom" type="box" size="0.1 0.1 0.01"/>
      </body>
    </body>
  </worldbody>
</mujoco>""",
        encoding="utf-8",
    )
    return build_prop_manager(
        {
            "types": [
                {
                    "name": "laptop",
                    "type": "articulation",
                    "asset": str(asset),
                    "joints": {"lid_hinge": {"mode": "passive"}},
                    "placement": {
                        "mode": "fixed",
                        "x_range": [1.0, 1.0],
                        "y_range": [0.0, 0.0],
                        "z_range": [0.5, 0.5],
                    },
                }
            ]
        },
        num_envs=1,
        env_spacing=2.0,
        device="cpu",
        backend="mujoco",
    )


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def test_object_jacobian_shape_layout_and_free_joint_columns(tiny_env_factory) -> None:
    """The object Jacobian is (E, 3, 6) with an object layout and free-joint columns."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    query = _query(env)
    body_id, _, vadr = _write_prop(
        env,
        "crate",
        pos=np.array([0.4, 0.1, 0.5]),
        quat=_QUAT_90_Y,
        lin=np.array([0.1, 0.2, 0.3]),
        ang_local=np.array([0.3, -0.4, 0.5]),
    )
    data = env.mj_datas[0]

    jac = query.kinematics.object_jacobian(ObjectRef("crate"))

    assert tuple(jac.transl.shape) == (1, 3, 6)
    assert tuple(jac.rot.shape) == (1, 3, 6)
    assert jac.frame == "world"
    assert jac.reference == "body_origin"
    layout = jac.layout
    assert layout.owner_kind == "object"
    assert layout.owner == "crate"
    assert layout.articulation is None
    assert layout.nv == 6
    assert layout.joint_names == ()
    assert layout.base_names == ("crate",)
    base_dofs = tuple(f"crate.{suffix}" for suffix in ("px", "py", "pz", "rx", "ry", "rz"))
    assert layout.base_dofs == base_dofs
    assert layout.order == base_dofs

    jacp = np.zeros((3, env.mj_model.nv), dtype=np.float64)
    jacr = np.zeros((3, env.mj_model.nv), dtype=np.float64)
    mujoco.mj_jacBody(env.mj_model, data, jacp, jacr, body_id)
    np.testing.assert_allclose(jac.transl[0].numpy(), jacp[:, vadr : vadr + 6])
    np.testing.assert_allclose(jac.rot[0].numpy(), jacr[:, vadr : vadr + 6])


def test_object_jacobian_com_reference_matches_velocity_and_origin_offset(
    tiny_env_factory,
) -> None:
    """COM/body-origin blocks relate by the offset term and reproduce the reads."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    query = _query(env)
    body_id, _, vadr = _write_prop(
        env,
        "crate",
        pos=np.array([0.4, 0.1, 0.5]),
        quat=_QUAT_90_Y,
        lin=np.array([0.1, 0.2, 0.3]),
        ang_local=np.array([0.3, -0.4, 0.5]),
    )
    data = env.mj_datas[0]
    qvel6 = np.asarray(data.qvel[vadr : vadr + 6], dtype=np.float64)

    state = query.state.object_state(ObjectRef("crate"))
    com = query.state.object_com_state(ObjectRef("crate"))
    origin_jac = query.kinematics.object_jacobian(ObjectRef("crate"), reference="body_origin")
    com_jac = query.kinematics.object_jacobian(ObjectRef("crate"), reference="com")

    np.testing.assert_allclose(
        com.lin_vel_w[0].numpy(), com_jac.transl[0].numpy() @ qvel6, atol=1e-12
    )
    np.testing.assert_allclose(
        state.ang_vel_w[0].numpy(), com_jac.rot[0].numpy() @ qvel6, atol=1e-12
    )

    offset = com.pos_w[0].numpy() - state.pos_w[0].numpy()
    expected = com_jac.transl[0].numpy() + _skew(offset) @ com_jac.rot[0].numpy()
    np.testing.assert_allclose(origin_jac.transl[0].numpy(), expected, atol=1e-12)
    np.testing.assert_allclose(origin_jac.rot[0].numpy(), com_jac.rot[0].numpy(), atol=1e-12)
    assert np.linalg.norm(offset) > 0.0
    assert np.linalg.norm(origin_jac.transl[0].numpy() - com_jac.transl[0].numpy()) > 0.0


def test_object_jacobian_translation_matches_finite_difference(tiny_env_factory) -> None:
    """Perturbing each free-joint qvel column reproduces the translational block."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    query = _query(env)
    _write_prop(
        env,
        "crate",
        pos=np.array([0.4, 0.1, 0.5]),
        quat=_QUAT_90_Y,
        lin=np.array([0.1, 0.2, 0.3]),
        ang_local=np.array([0.3, -0.4, 0.5]),
    )
    model = env.mj_model
    data = env.mj_datas[0]
    body_id, _, vadr = env.prop_backend.free_addresses("crate")
    qpos0 = data.qpos.copy()
    x0 = np.asarray(data.xpos[body_id], dtype=np.float64).copy()
    jac = query.kinematics.object_jacobian(ObjectRef("crate"))
    eps = 1e-6
    finite = np.zeros((3, 6), dtype=np.float64)
    try:
        for column in range(6):
            qvel = np.zeros(model.nv, dtype=np.float64)
            qvel[vadr + column] = 1.0
            qpos = qpos0.copy()
            mujoco.mj_integratePos(model, qpos, qvel, eps)
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            finite[:, column] = (np.asarray(data.xpos[body_id]) - x0) / eps
    finally:
        data.qpos[:] = qpos0
        mujoco.mj_forward(model, data)

    np.testing.assert_allclose(jac.transl[0].numpy(), finite, atol=1e-5)
    np.testing.assert_array_equal(data.qpos, qpos0)


def test_object_jacobian_rotational_false_zeros_only_the_rotational_block(
    tiny_env_factory,
) -> None:
    """The explicit rotational=False request zeroes only the angular block."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    query = _query(env)
    _write_prop(
        env,
        "crate",
        pos=np.array([0.4, 0.1, 0.5]),
        quat=_QUAT_90_Y,
        lin=np.array([0.1, 0.2, 0.3]),
        ang_local=np.array([0.3, -0.4, 0.5]),
    )

    full = query.kinematics.object_jacobian(ObjectRef("crate"), reference="com")
    transl_only = query.kinematics.object_jacobian(
        ObjectRef("crate"), reference="com", rotational=False
    )

    assert np.linalg.norm(full.rot[0].numpy()) > 0.0
    np.testing.assert_allclose(transl_only.transl[0].numpy(), full.transl[0].numpy())
    np.testing.assert_array_equal(transl_only.rot[0].numpy(), np.zeros((3, 6)))


def test_object_jacobian_rejects_unknown_and_non_rigid_props(tiny_env_factory, tmp_path) -> None:
    """Unknown names are reference errors; static/articulation props are capability errors."""
    env = tiny_env_factory(prop_manager=_prop_manager())
    query = _query(env)

    with pytest.raises(QueryReferenceError, match=r"ghost.*crate"):
        query.kinematics.object_jacobian(ObjectRef("ghost"))
    with pytest.raises(QueryReferenceError, match="use state.root_state/com_state"):
        query.kinematics.object_jacobian(ObjectRef("tiny"))
    with pytest.raises(QueryCapabilityError, match="jacobian.object.*local"):
        query.kinematics.object_jacobian(ObjectRef("crate"), frame="local")

    articulated = tiny_env_factory(prop_manager=_free_articulation_manager(tmp_path))
    with pytest.raises(QueryCapabilityError, match="rigid free props only"):
        _query(articulated).kinematics.object_jacobian(ObjectRef("laptop"))

    anchored = tiny_env_factory(prop_manager=_fixed_articulation_manager(tmp_path))
    with pytest.raises(QueryCapabilityError, match="free/anchored articulation prop"):
        _query(anchored).kinematics.object_jacobian(ObjectRef("laptop"))


def test_object_jacobian_multi_env_detached_and_live_state_unchanged(tiny_env_factory) -> None:
    """Multi-env selection preserves order and reads detach without mutating live state."""
    env = tiny_env_factory(num_envs=2, prop_manager=_prop_manager(num_envs=2))
    query = _query(env)
    live_qpos = env.mj_datas[0].qpos.copy()
    live_qvel = env.mj_datas[0].qvel.copy()
    live_xipos = env.mj_datas[0].xipos.copy()

    jac = query.kinematics.object_jacobian(ObjectRef("crate"), reference="com", env=[1, 0])

    assert tuple(jac.env_ids.tolist()) == (1, 0)
    assert tuple(jac.transl.shape) == (2, 3, 6)
    jac.transl[0, 0, 0] = 123.0
    jac.rot[0, 0, 0] = 456.0
    np.testing.assert_array_equal(env.mj_datas[0].qpos, live_qpos)
    np.testing.assert_array_equal(env.mj_datas[0].qvel, live_qvel)
    np.testing.assert_array_equal(env.mj_datas[0].xipos, live_xipos)


def test_object_jacobian_requires_a_prop_backend(tiny_env) -> None:
    """No declared props cannot silently answer an object Jacobian."""
    query = _query(tiny_env)

    with pytest.raises(QueryReferenceError, match="no props are declared"):
        query.kinematics.object_jacobian(ObjectRef("crate"))
