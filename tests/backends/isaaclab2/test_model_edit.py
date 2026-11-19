"""IsaacLab 2 model edits: broadcast rows, frames, refusals, persistence."""

from __future__ import annotations

import numpy as np
import pytest

from unigym.backends.isaaclab2.model_edit import IsaacLab2ModelEditProvider
from unigym.model.edit import GeomEdit
from unigym.model.errors import ModelEditCapabilityError, ModelEditError
from unigym.runtime.actions import NamedAction

pytestmark = [
    pytest.mark.sim,
    pytest.mark.requires_test_packs("isaaclab_cartpole"),
]

ARTICULATION = "cartpole"
CART_JOINTS = ("slider_to_cart", "cart_to_pole")


def _body(env):
    ir = env.model_set.articulations[ARTICULATION]
    return ir.bodies[ir.bodies.names()[0]]


def _view(env):
    return env.robots[ARTICULATION].root_physx_view


def _body_index(env, name: str) -> int:
    return list(env.robots[ARTICULATION].body_names).index(name)


def _matrix_from_row(row) -> np.ndarray:
    values = np.asarray(row, dtype=float).reshape(3, 3)
    return values


def test_body_mass_edit_broadcasts_to_every_env_row(two_env) -> None:
    """v1 writes all env rows; the published model reflects the new mass."""
    body = _body(two_env)
    index = _body_index(two_env, body.name)
    with two_env.model_edit() as edit:
        edit.body(ARTICULATION, body.name).set(mass=2.0)

    masses = np.asarray(_view(two_env).get_masses().cpu())
    assert masses.shape[0] == 2
    assert np.allclose(masses[:, index], 2.0)
    assert two_env.model_set.articulations[ARTICULATION].bodies[body.name].mass == (
        pytest.approx(2.0)
    )
    assert two_env._model_revision == 1


def test_com_and_inertia_quat_publish_the_same_tensor(two_env) -> None:
    """A non-diagonal inertia round-trips through describe's ipos/iquat."""
    body = _body(two_env)
    index = _body_index(two_env, body.name)
    inertia = (0.2, 0.3, 0.4)
    quat = (0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
    com = (0.01, -0.02, 0.03)

    with two_env.model_edit() as edit:
        edit.body(ARTICULATION, body.name).set(
            com=com,
            inertia=inertia,
            inertia_quat=quat,
        )

    coms = np.asarray(_view(two_env).get_coms().cpu())
    assert np.allclose(coms[:, index, 0:3], com)
    published = two_env.model_set.articulations[ARTICULATION].bodies[body.name]
    assert published.ipos == pytest.approx(com)
    assert published.iquat is not None
    w, x, y, z = published.iquat
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    reconstructed = rotation @ np.diag(published.inertia) @ rotation.T
    expected = rotation @ np.diag(inertia) @ rotation.T
    assert np.allclose(reconstructed, expected, atol=1e-6)
    raw = _matrix_from_row(np.asarray(_view(two_env).get_inertias().cpu())[0, index])
    assert np.allclose(raw, expected, atol=1e-6)


def test_inertia_quat_alone_is_rejected(cartpole_env) -> None:
    """The PhysX frame rotation needs a diagonal inertia in the same call."""
    body = _body(cartpole_env)
    edit = cartpole_env.model_edit()
    edit.body(ARTICULATION, body.name).set(inertia_quat=(1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ModelEditError, match="requires body.inertia"):
        edit.commit()


def test_geom_fields_are_a_capability_error(cartpole_env) -> None:
    """No published geom IR means geom edits are declared absent."""
    provider = IsaacLab2ModelEditProvider()
    edit = GeomEdit(
        articulation=ARTICULATION,
        body="cartpole",
        geom="cartpole_geom",
        geom_id=None,
        values={},
    )
    with pytest.raises(ModelEditCapabilityError, match="geom fields are not editable"):
        provider.apply(cartpole_env, (), (edit,))


def test_body_edit_persists_across_reset(cartpole_env) -> None:
    """A reset does not rewrite the body mass."""
    body = _body(cartpole_env)
    index = _body_index(cartpole_env, body.name)
    with cartpole_env.model_edit() as edit:
        edit.body(ARTICULATION, body.name).set(mass=3.0)

    cartpole_env.reset_all()

    masses = np.asarray(_view(cartpole_env).get_masses().cpu())
    assert np.allclose(masses[:, index], 3.0)


def test_body_edit_does_not_change_state_or_observations(cartpole_env) -> None:
    """Commit writes the model only; the current state/obs are unchanged."""
    body = _body(cartpole_env)
    data = cartpole_env.robots[ARTICULATION].data
    before_qpos = data.joint_pos.clone()
    before_root = data.root_pos_w.clone()
    before_obs = {key: value.clone() for key, value in cartpole_env.obs.snapshot().values.items()}

    with cartpole_env.model_edit() as edit:
        edit.body(ARTICULATION, body.name).set(mass=2.0)

    assert bool((data.joint_pos == before_qpos).all())
    assert bool((data.root_pos_w == before_root).all())
    after_obs = cartpole_env.obs.snapshot().values
    for key, value in before_obs.items():
        assert bool((after_obs[key] == value).all()), key


def test_unknown_names_fail_before_any_physx_write(cartpole_env) -> None:
    """The IsaacLab IR has no world body; both names fail as unknown."""
    edit = cartpole_env.model_edit()
    with pytest.raises(ModelEditError, match="not in this model"):
        edit.body(ARTICULATION, "world")
    with pytest.raises(ModelEditError, match="not in this model"):
        edit.body(ARTICULATION, "bogus")


def test_step_guard_rejects_a_transaction_inside_step(cartpole_env, monkeypatch) -> None:
    """A transaction opened during step() is rejected."""
    original = cartpole_env._step_impl

    def probe(actions, raw_action=None, cmd=None):
        with pytest.raises(RuntimeError, match="between steps"):
            cartpole_env.model_edit()
        return original(actions, raw_action, cmd)

    monkeypatch.setattr(cartpole_env, "_step_impl", probe)
    action = (NamedAction.zeros(1, ARTICULATION, list(CART_JOINTS), "position"),)
    cartpole_env.step(action)
