"""Real MuJoCo model-edit transaction: arrays, derived fields, guards."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
import torch

from unigym.backends.registry import ModelEditField
from unigym.model.errors import ModelEditError
from unigym.runtime.actions import NamedAction


def _action(env):
    control = env.controls[("tiny", "tiny")]
    tensor = torch.zeros(env.num_envs, len(control.joint_names), dtype=torch.float32)
    return (NamedAction(control.articulation, tensor, control.joint_names, control.target),)


def _link_geom(env):
    bodies = env.model_set.merged.bodies
    geoms = env.model_set.merged.geoms
    return bodies["link"], geoms["link_geom"]


def _restitution_zeta(restitution: float) -> float:
    log_e = math.log(restitution)
    return -log_e / math.sqrt(math.pi * math.pi + log_e * log_e)


def test_body_and_geom_arrays_and_published_set_refresh(tiny_env_factory) -> None:
    """Every edited field lands in MjModel and the rebuilt ModelSet."""
    env = tiny_env_factory()
    link, geom = _link_geom(env)
    model = env.mj_model
    solref_before = model.geom_solref[geom.geom_id].copy()

    with env.model_edit() as edit:
        edit.body("tiny", "link").set(
            mass=2.0,
            com=(0.01, 0.0, -0.02),
            inertia=(0.02, 0.03, 0.04),
            inertia_quat=(1.0, 0.0, 0.0, 0.0),
        )
        edit.geom("tiny", body="link", geom="link_geom").set(
            friction=(0.9, 0.02, 0.001),
            restitution=0.2,
        )

    assert env._model_revision == 1
    assert model.body_mass[link.body_id] == pytest.approx(2.0)
    assert model.body_ipos[link.body_id] == pytest.approx([0.01, 0.0, -0.02])
    assert model.body_inertia[link.body_id] == pytest.approx([0.02, 0.03, 0.04])
    assert model.body_iquat[link.body_id] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert model.geom_friction[geom.geom_id] == pytest.approx([0.9, 0.02, 0.001])
    assert model.geom_solref[geom.geom_id][0] == pytest.approx(solref_before[0])
    assert model.geom_solref[geom.geom_id][1] == pytest.approx(_restitution_zeta(0.2))

    published_body = env.model_set.merged.bodies["link"]
    assert published_body.mass == pytest.approx(2.0)
    assert published_body.ipos == pytest.approx((0.01, 0.0, -0.02))
    assert published_body.inertia == pytest.approx((0.02, 0.03, 0.04))
    mj = env.model_set.merged.mj
    assert mj is not None
    published_geom = mj.geom_contact["link_geom"]
    assert published_geom.friction == pytest.approx((0.9, 0.02, 0.001))
    assert published_geom.solref is not None
    assert published_geom.solref[1] == pytest.approx(_restitution_zeta(0.2))
    assert env.model_edit  # the editor face exists


def test_body_simple_is_cleared_before_mj_set_const(tiny_env_factory) -> None:
    """A COM edit disables the simple-body shortcut before recomputation."""
    env = tiny_env_factory()
    link, _ = _link_geom(env)
    if not hasattr(env.mj_model, "body_simple"):
        pytest.skip("this mujoco build has no body_simple")
    env.mj_model.body_simple[link.body_id] = 1

    with env.model_edit() as edit:
        edit.body("tiny", "link").set(com=(0.02, 0.0, 0.0))

    assert env.mj_model.body_simple[link.body_id] == 0


def test_mj_set_const_refreshes_derived_fields(tiny_env_factory) -> None:
    """Mass/inertia edits recompute subtree mass and inverse weight."""
    env = tiny_env_factory()
    base = env.model_set.merged.bodies["base"]
    link, _ = _link_geom(env)
    subtreemass_before = float(env.mj_model.body_subtreemass[base.body_id])
    invweight_before = env.mj_model.body_invweight0[link.body_id].copy()

    with env.model_edit() as edit:
        edit.body("tiny", "link").set(
            mass=3.0,
            inertia=(0.5, 0.6, 0.7),
        )

    assert float(env.mj_model.body_subtreemass[base.body_id]) > subtreemass_before
    assert not np.allclose(env.mj_model.body_invweight0[link.body_id], invweight_before)


def test_edits_leave_state_and_warmstart_untouched(tiny_env_factory) -> None:
    """A model edit never perturbs qpos/qvel/ctrl or the solver warm-start."""
    env = tiny_env_factory()
    data = env.mj_datas[0]
    before = {
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "ctrl": data.ctrl.copy(),
        "qacc_warmstart": data.qacc_warmstart.copy(),
    }

    with env.model_edit() as edit:
        edit.body("tiny", "link").set(mass=2.0, com=(0.01, 0.0, 0.0))

    for key, value in before.items():
        assert np.array_equal(getattr(data, key), value), key


def test_mass_and_inertia_change_the_mass_matrix(tiny_env_factory) -> None:
    """The edited model is what the solver reads on the next forward pass."""
    env = tiny_env_factory()
    base = env.model_set.merged.bodies["base"]
    link, _ = _link_geom(env)
    data = env.mj_datas[0]
    mujoco.mj_forward(env.mj_model, data)
    baseline = data.M.copy()

    with env.model_edit() as edit:
        edit.body("tiny", "base").set(mass=4.0)
        edit.body("tiny", "link").set(inertia=(0.5, 0.6, 0.7))

    mujoco.mj_forward(env.mj_model, data)
    assert not np.allclose(data.M, baseline)
    assert env.mj_model.body_mass[base.body_id] == pytest.approx(4.0)


def test_direct_mode_solref_rejects_restitution_and_rolls_back(tiny_env_factory) -> None:
    """A failed restitution write restores the earlier friction write."""
    env = tiny_env_factory()
    _, geom = _link_geom(env)
    model = env.mj_model
    model.geom_solref[geom.geom_id][0] = -1000.0
    friction_before = model.geom_friction[geom.geom_id].copy()
    solref_before = model.geom_solref[geom.geom_id].copy()
    revision = env._model_revision

    edit = env.model_edit()
    edit.geom("tiny", body="link", geom="link_geom").set(
        friction=(0.1, 0.0, 0.0),
        restitution=0.2,
    )
    with pytest.raises(ModelEditError, match="direct-mode solref"):
        edit.commit()

    assert np.array_equal(model.geom_friction[geom.geom_id], friction_before)
    assert np.array_equal(model.geom_solref[geom.geom_id], solref_before)
    assert env._model_revision == revision


def test_edit_persists_across_reset(tiny_env_factory) -> None:
    """A reset re-writes state, never the model parameters."""
    env = tiny_env_factory()
    link, _ = _link_geom(env)
    with env.model_edit() as edit:
        edit.body("tiny", "link").set(mass=2.5)

    env.reset_all()

    assert env.mj_model.body_mass[link.body_id] == pytest.approx(2.5)
    assert env.model_set.merged.bodies["link"].mass == pytest.approx(2.5)


def test_snapshot_revision_guard_refuses_a_stale_restore(tiny_env_factory) -> None:
    """A checkpoint captured before an edit cannot be restored afterwards."""
    env = tiny_env_factory()
    state = env.snapshot()
    with env.model_edit() as edit:
        edit.body("tiny", "link").set(mass=2.0)

    with pytest.raises(ModelEditError, match="model revision"):
        env.restore(state)

    fresh = env.snapshot()
    env.restore(fresh)


def test_old_snapshot_without_revision_remains_compatible(tiny_env_factory) -> None:
    """A pre-model-edit payload (no meta.model_revision) still restores."""
    env = tiny_env_factory()
    state = env.snapshot()
    state["meta"].pop("model_revision", None)
    with env.model_edit() as edit:
        edit.body("tiny", "link").set(mass=2.0)

    env.restore(state)


def test_world_body_edits_are_rejected(tiny_env_factory) -> None:
    """The compiled world body (body_id=0) is not editable."""
    env = tiny_env_factory()
    with pytest.raises(ModelEditError, match="world body"):
        env.model_edit().body("tiny", "world")


def test_model_edit_is_rejected_during_step(tiny_env_factory, monkeypatch) -> None:
    """The step guard rejects a transaction opened inside step()."""
    env = tiny_env_factory()
    original = env._step_impl

    def probe(actions, raw_action=None, cmd=None):
        with pytest.raises(RuntimeError, match="between steps"):
            env.model_edit()
        return original(actions, raw_action, cmd)

    monkeypatch.setattr(env, "_step_impl", probe)
    env.step(_action(env))


def test_multi_articulation_named_edit_translates_to_the_compiled_body(tmp_path, tiny_mjcf) -> None:
    """A named edit on a prefixed multi-asset run writes the right body."""
    from tests.backends.mujoco.test_env import _two_articulation_env

    env = _two_articulation_env(tmp_path, tiny_mjcf)
    try:
        assert env.model_set.merged is None
        left = env.model_set.articulations["left"].bodies["link"]
        right = env.model_set.articulations["right"].bodies["link"]
        right_mass = float(env.mj_model.body_mass[right.body_id])

        with env.model_edit() as edit:
            edit.body("left", "link").set(mass=2.0)

        assert env.mj_model.body_mass[left.body_id] == pytest.approx(2.0)
        assert env.mj_model.body_mass[right.body_id] == pytest.approx(right_mass)
        assert env.model_set.articulations["left"].bodies["link"].mass == pytest.approx(2.0)
    finally:
        env.close()


def test_geom_field_matrix_is_declared(tiny_env_factory) -> None:
    """Both geom fields are writable on the mujoco backend."""
    env = tiny_env_factory()
    assert env.model_edit().fields >= {
        ModelEditField.GEOM_FRICTION,
        ModelEditField.GEOM_RESTITUTION,
    }
