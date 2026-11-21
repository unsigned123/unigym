"""MuJoCo contact constraint-row Jacobian and wrench query behavior."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from unigym.backends.mujoco.query import MujocoQuery
from unigym.contacts.manager import build_contact_manager
from unigym.query import (
    EntityRef,
    QueryCapabilityError,
    QueryReferenceError,
    QueryStaleError,
)
from unigym.runtime.actions import NamedAction

PROP_CRATE = """    <body name="Prop_crate" pos="0.3 0 0.05">
      <freejoint/>
      <geom name="crate_geom" type="box" size="0.05 0.05 0.05" mass="0.5"/>
    </body>"""


def _source(
    *,
    condim: int | None = None,
    cone: str | None = None,
    sparse: bool = False,
    crate: bool = False,
) -> str:
    from tests.backends.mujoco.conftest import tiny_mjcf_source

    source = tiny_mjcf_source()
    if condim is not None:
        source = source.replace('<geom name="tip_geom"', f'<geom condim="{condim}" name="tip_geom"')
    if cone is not None:
        source = source.replace(
            '<option timestep="0.005" gravity="0 0 -9.81"/>',
            f'<option timestep="0.005" gravity="0 0 -9.81" cone="{cone}"/>',
        )
    if sparse:
        source = source.replace(
            '<option timestep="0.005" gravity="0 0 -9.81"/>',
            '<option timestep="0.005" gravity="0 0 -9.81" jacobian="sparse"/>',
        )
    if crate:
        source = source.replace("  </worldbody>", PROP_CRATE + "\n  </worldbody>", 1)
    return source


def _mjcf(tmp_path: Path, *, name: str = "query_contact_jacobian.xml", **options) -> Path:
    path = tmp_path / name
    path.write_text(_source(**options), encoding="utf-8")
    return path


def _manager():
    manager = build_contact_manager(
        [
            {
                "name": "tip_floor",
                "sensor_refs": [{"owner": "robot", "articulation": "tiny", "body": "tip"}],
                "filter_refs": [{"owner": "environment", "body": "world"}],
                "frame": "world",
            }
        ],
        backend="mujoco",
        articulation_names=("tiny",),
    )
    assert manager is not None
    return manager


def _step(env, steps: int = 30) -> None:
    action = NamedAction.zeros(env.num_envs, "tiny", ("elbow", "shoulder"), "position")
    for _ in range(steps):
        env.step((action,))


def _env(tiny_env_factory, tmp_path, **options):
    return tiny_env_factory(mjcf=_mjcf(tmp_path, **options), contact_manager=_manager())


def _point(env):
    query = MujocoQuery(env)
    return query, query.contacts.manifold("tip_floor").points[0]


def _layout_columns(env, layout):
    art = env.articulations["tiny"]
    columns: list[int] = []
    if layout.base_dofs:
        assert art.free_dof_adr is not None
        columns.extend(range(int(art.free_dof_adr), int(art.free_dof_adr) + 6))
    columns.extend(int(env.mj_model.joint(name).dofadr.item()) for name in layout.joint_names)
    assert len(columns) == layout.nv
    return columns


def _effective_rows(model, dim: int) -> int:
    cone = int(model.opt.cone)
    if cone == int(mujoco.mjtCone.mjCONE_ELLIPTIC):
        return dim
    if cone == int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        return max(1, 2 * (dim - 1))
    raise AssertionError(f"unexpected friction cone {cone}")


def _rows_from_addresses(data, contact_id: int) -> tuple[int, int]:
    """Return the contact's start row and stride from neighbouring contacts."""
    address = int(data.contact.efc_address[contact_id])
    if contact_id + 1 < int(data.ncon):
        stop = int(data.contact.efc_address[contact_id + 1])
    else:
        stop = int(data.nefc)
    return address, stop - address


def _rows(data, model, point, columns):
    address = int(data.contact.efc_address[point.contact_id])
    rows = _effective_rows(model, int(data.contact.dim[point.contact_id]))
    block = np.asarray(data.efc_J, dtype=np.float64).reshape(int(data.nefc), int(model.nv))
    return address, rows, block[address : address + rows, :][:, columns]


_PYRAMID_CONDIM3 = np.array(
    [
        [1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
    ]
)


def _row_frame_components(jacobian, frame_w):
    """Express each row's free-base linear part in the contact frame basis."""
    layout = jacobian.layout
    assert len(layout.base_dofs) == 6
    axes = [layout.order.index(name) for name in layout.base_dofs[:3]]
    return jacobian.rows.numpy()[:, axes] @ frame_w.T


@pytest.mark.parametrize("condim", [3, 4, 6])
def test_contact_jacobian_uses_the_effective_pyramidal_rows(
    tiny_env_factory, tmp_path, condim
) -> None:
    """Pyramidal contacts contribute paired friction rows, not contact.dim rows."""
    options = {} if condim == 3 else {"condim": condim}
    env = _env(tiny_env_factory, tmp_path, **options)
    _step(env)
    query, point = _point(env)
    jacobian = query.contacts.jacobian(point, "tiny")

    data = env.mj_data
    model = env.mj_model
    dim = int(data.contact.dim[point.contact_id])
    address, rows, expected = _rows(data, model, point, _layout_columns(env, jacobian.layout))
    next_address, next_rows = _rows_from_addresses(data, point.contact_id)
    assert int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
    assert dim == condim
    assert address == next_address
    assert rows == next_rows == _effective_rows(model, dim)
    assert jacobian.env_id == point.env_id
    assert jacobian.frame == "contact"
    assert jacobian.layout == query.kinematics.jacobian(EntityRef("body", "base")).layout
    assert jacobian.layout.nv == model.nv
    assert tuple(jacobian.rows.shape) == (rows, jacobian.layout.nv)
    assert point.frame_w is not None
    assert tuple(point.frame_w.shape) == (3, 3)
    row_frame = _row_frame_components(jacobian, point.frame_w.numpy())
    np.testing.assert_allclose(row_frame[:4], _PYRAMID_CONDIM3, atol=1e-12)
    if condim == 3:
        assert np.count_nonzero(np.abs(row_frame) > 1e-12, axis=1).tolist() == [2, 2, 2, 2]
    else:
        assert rows > 4
    np.testing.assert_allclose(jacobian.rows.numpy(), expected)


def test_contact_jacobian_uses_dim_rows_for_an_elliptic_cone(tiny_env_factory, tmp_path) -> None:
    """An elliptic cone contributes one efc row per contact dimension."""
    env = _env(tiny_env_factory, tmp_path, condim=4, cone="elliptic")
    _step(env)
    query, point = _point(env)
    jacobian = query.contacts.jacobian(point, "tiny")

    data = env.mj_data
    model = env.mj_model
    dim = int(data.contact.dim[point.contact_id])
    address, rows, expected = _rows(data, model, point, _layout_columns(env, jacobian.layout))
    next_address, next_rows = _rows_from_addresses(data, point.contact_id)
    assert int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)
    assert dim == 4
    assert address == next_address
    assert rows == next_rows == dim
    assert tuple(jacobian.rows.shape) == (dim, jacobian.layout.nv)
    assert point.frame_w is not None
    row_frame = _row_frame_components(jacobian, point.frame_w.numpy())
    np.testing.assert_allclose(row_frame[:3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(row_frame[3:], 0.0, atol=1e-12)
    np.testing.assert_allclose(jacobian.rows.numpy(), expected)


def test_contact_rows_are_column_scoped_to_the_articulation(tiny_env_factory, tmp_path) -> None:
    """A prop's DOFs never leak into the articulation's contact row block."""
    env = _env(tiny_env_factory, tmp_path, crate=True)
    _step(env)
    query, point = _point(env)
    jacobian = query.contacts.jacobian(point, "tiny")

    data = env.mj_data
    model = env.mj_model
    columns = _layout_columns(env, jacobian.layout)
    _, rows, expected = _rows(data, model, point, columns)
    assert jacobian.layout.nv < model.nv
    assert tuple(jacobian.rows.shape) == (rows, jacobian.layout.nv)
    np.testing.assert_allclose(jacobian.rows.numpy(), expected)


def test_contact_wrench_matches_the_constraint_generalized_force(
    tiny_env_factory, tmp_path
) -> None:
    """With only contact constraints, wrench_to_joint equals qfrc_constraint."""
    env = _env(tiny_env_factory, tmp_path)
    _step(env)
    query, point = _point(env)
    jacobian = query.contacts.jacobian(point, "tiny")
    wrench = query.contacts.wrench_to_joint(point, "tiny")

    data = env.mj_data
    model = env.mj_model
    dim = int(data.contact.dim[point.contact_id])
    assert int(data.nefc) == _effective_rows(model, dim)
    columns = _layout_columns(env, jacobian.layout)
    assert columns == list(range(int(model.nv)))
    assert tuple(wrench.shape) == (jacobian.layout.nv,)
    np.testing.assert_allclose(
        wrench.numpy(), np.asarray(data.qfrc_constraint, dtype=np.float64)[columns], atol=1e-9
    )


def test_contact_wrench_maps_the_contact_force_through_the_rows(tiny_env_factory, tmp_path) -> None:
    """wrench_to_joint is rows.T @ efc_force for the same contact rows."""
    env = _env(tiny_env_factory, tmp_path)
    _step(env)
    query, point = _point(env)
    jacobian = query.contacts.jacobian(point, "tiny")
    wrench = query.contacts.wrench_to_joint(point, "tiny")

    data = env.mj_data
    address, rows, _ = _rows(data, env.mj_model, point, _layout_columns(env, jacobian.layout))
    force = np.asarray(data.efc_force, dtype=np.float64).reshape(-1)[address : address + rows]
    expected = jacobian.rows.numpy().T @ force
    np.testing.assert_allclose(wrench.numpy(), expected)
    assert float(np.linalg.norm(wrench.numpy())) > 0.0


def test_contact_rows_and_wrench_are_detached_copies(tiny_env_factory, tmp_path) -> None:
    """Mutating a result never mutates the live contact solve."""
    env = _env(tiny_env_factory, tmp_path)
    _step(env)
    query, point = _point(env)
    live_j = np.array(env.mj_data.efc_J, copy=True)
    live_force = np.array(env.mj_data.efc_force, copy=True)

    jacobian = query.contacts.jacobian(point, "tiny")
    wrench = query.contacts.wrench_to_joint(point, "tiny")
    jacobian.rows[0, 0] = 123.0
    wrench[0] = 456.0

    np.testing.assert_array_equal(np.asarray(env.mj_data.efc_J), live_j)
    np.testing.assert_array_equal(np.asarray(env.mj_data.efc_force), live_force)


def test_sparse_efc_j_is_a_typed_capability_error(tiny_env_factory, tmp_path) -> None:
    """A sparse constraint Jacobian is refused instead of read as dense rows."""
    env = _env(tiny_env_factory, tmp_path, sparse=True)
    _step(env)
    query, point = _point(env)

    with pytest.raises(QueryCapabilityError, match="sparse"):
        query.contacts.jacobian(point, "tiny")
    with pytest.raises(QueryCapabilityError, match="sparse"):
        query.contacts.wrench_to_joint(point, "tiny")


def test_stale_contact_index_is_a_reference_error(tiny_env_factory, tmp_path) -> None:
    """A contact id outside the current contact list never returns rows."""
    env = _env(tiny_env_factory, tmp_path)
    _step(env)
    query, point = _point(env)
    stale = replace(point, contact_id=int(env.mj_data.ncon))

    with pytest.raises(QueryReferenceError, match="contact_id"):
        query.contacts.jacobian(stale, "tiny")


def test_reordered_contact_id_is_a_reference_error(tiny_env_factory, tmp_path) -> None:
    """An in-range id that names another contact pair is rejected as stale."""
    env = _env(tiny_env_factory, tmp_path, crate=True)
    _step(env)
    query, point = _point(env)
    data = env.mj_data
    current = tuple(int(value) for value in data.contact.geom[point.contact_id])
    other = next(
        index
        for index in range(int(data.ncon))
        if tuple(int(value) for value in data.contact.geom[index]) != current
    )
    stale = replace(point, contact_id=other)

    with pytest.raises(QueryReferenceError, match="contact_id"):
        query.contacts.jacobian(stale, "tiny")
    with pytest.raises(QueryReferenceError, match="contact_id"):
        query.contacts.wrench_to_joint(stale, "tiny")


def test_contact_point_env_id_outside_the_run_is_a_reference_error(
    tiny_env_factory, tmp_path
) -> None:
    """A point that names an env outside the run is a typed reference error."""
    env = _env(tiny_env_factory, tmp_path)
    _step(env)
    query, point = _point(env)
    stale = replace(point, env_id=env.num_envs)

    with pytest.raises(QueryReferenceError, match="env_id"):
        query.contacts.jacobian(stale, "tiny")


def test_unknown_contact_articulation_is_a_reference_error(tiny_env_factory, tmp_path) -> None:
    """The contact rows resolve the articulation before reading the solve."""
    env = _env(tiny_env_factory, tmp_path)
    _step(env)
    query, point = _point(env)

    with pytest.raises(QueryReferenceError, match="ghost"):
        query.contacts.wrench_to_joint(point, "ghost")


def test_contact_row_read_after_query_close_is_stale(tiny_env_factory, tmp_path) -> None:
    """A closed query handle rejects contact Jacobian reads."""
    env = _env(tiny_env_factory, tmp_path)
    _step(env)
    query, point = _point(env)
    query.close()

    with pytest.raises(QueryStaleError, match="stale"):
        query.contacts.jacobian(point, "tiny")
