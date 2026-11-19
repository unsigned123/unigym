"""Real mjwarp model-edit transaction: shared device push and next-step use."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unigym.model.errors import ModelEditError
from unigym.runtime.actions import NamedAction


def _action(env, value: float = 0.0):
    control = env.controls[("tiny", "tiny")]
    tensor = torch.full((env.num_envs, len(control.joint_names)), value, dtype=torch.float32)
    return (NamedAction(control.articulation, tensor, control.joint_names, control.target),)


def _ids(env):
    return (
        env.model_set.merged.bodies["link"],
        env.model_set.merged.geoms["link_geom"],
    )


def _assert_model_fields_equal(env) -> None:
    for name in env._MODEL_EDIT_FIELDS:
        device = getattr(env._wm, name)
        source = np.asarray(getattr(env.mj_model, name))
        actual = device.numpy() if device.ndim == 1 else device.numpy()[0]
        np.testing.assert_allclose(
            actual,
            source,
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"device field {name} differs from the CPU model",
        )


def _full_edit(edit):
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


def test_all_edited_and_derived_fields_reach_the_shared_device_model(
    tiny_env_factory,
) -> None:
    """The CPU model and the leading device slot agree after a full edit."""
    env = tiny_env_factory(num_envs=2)
    link, geom = _ids(env)
    assert env._wm.body_mass.shape[0] == 1

    with env.model_edit() as edit:
        _full_edit(edit)

    assert env._wm.body_mass.numpy()[0, link.body_id] == pytest.approx(2.0)
    assert env._wm.body_ipos.numpy()[0, link.body_id] == pytest.approx([0.01, 0.0, -0.02])
    assert env._wm.body_inertia.numpy()[0, link.body_id] == pytest.approx([0.02, 0.03, 0.04])
    assert env._wm.body_iquat.numpy()[0, link.body_id] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert env._wm.geom_friction.numpy()[0, geom.geom_id] == pytest.approx([0.9, 0.02, 0.001])
    assert env._wm.geom_solref.numpy()[0, geom.geom_id][0] == pytest.approx(
        env.mj_model.geom_solref[geom.geom_id][0]
    )
    _assert_model_fields_equal(env)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_cpu_and_device_edits_agree(tiny_env_factory, device) -> None:
    """The same transaction round-trips on CPU and a CUDA warp device."""
    env = tiny_env_factory(num_envs=2, device=device)
    link, _ = _ids(env)
    with env.model_edit() as edit:
        edit.body("tiny", "link").set(mass=2.5)
    assert env.mj_model.body_mass[link.body_id] == pytest.approx(2.5)
    assert env._wm.body_mass.numpy()[0, link.body_id] == pytest.approx(2.5)


def test_edit_does_not_touch_wdata_state_or_warmstart(tiny_env_factory) -> None:
    """Only the device model is pushed; batched Data state stays identical."""
    env = tiny_env_factory(num_envs=2)
    before = {
        "qpos": env._wdata.qpos.numpy().copy(),
        "qvel": env._wdata.qvel.numpy().copy(),
        "qacc_warmstart": env._wdata.qacc_warmstart.numpy().copy(),
    }
    model_before = {
        "body_pos": env._wm.body_pos.numpy().copy(),
        "eq_data": (
            None if getattr(env._wm, "eq_data", None) is None else env._wm.eq_data.numpy().copy()
        ),
    }

    with env.model_edit() as edit:
        _full_edit(edit)

    for name, value in before.items():
        np.testing.assert_array_equal(getattr(env._wdata, name).numpy(), value, err_msg=name)
    np.testing.assert_array_equal(env._wm.body_pos.numpy(), model_before["body_pos"])
    if model_before["eq_data"] is not None:
        np.testing.assert_array_equal(env._wm.eq_data.numpy(), model_before["eq_data"])
    assert env._wdata.qpos.shape[0] == 2


def test_next_warp_step_sees_the_edited_model(tiny_env_factory) -> None:
    """A changed inertia produces a different next-step trajectory."""
    baseline = tiny_env_factory(num_envs=1)
    edited = tiny_env_factory(num_envs=1)
    link, _ = _ids(edited)

    with edited.model_edit() as edit:
        edit.body("tiny", "link").set(inertia=(5.0, 5.0, 5.0))

    for _ in range(5):
        baseline.step(_action(baseline, 0.5))
        edited.step(_action(edited, 0.5))

    assert not np.allclose(baseline.mj_datas[0].qpos, edited.mj_datas[0].qpos, atol=1e-6)
    assert edited._wdata.qpos.shape[0] == 1
    assert edited.mj_model.body_inertia[link.body_id] == pytest.approx([5.0, 5.0, 5.0])


def test_edit_persists_across_reset(tiny_env_factory) -> None:
    """Reset re-seeds state; the edited CPU/device model stays in place."""
    env = tiny_env_factory(num_envs=2)
    link, _ = _ids(env)
    with env.model_edit() as edit:
        edit.body("tiny", "link").set(mass=2.5)

    env.reset_all()

    assert env.mj_model.body_mass[link.body_id] == pytest.approx(2.5)
    assert env._wm.body_mass.numpy()[0, link.body_id] == pytest.approx(2.5)
    _assert_model_fields_equal(env)


def test_snapshot_revision_guard_refuses_a_stale_restore(tiny_env_factory) -> None:
    """A payload captured before an edit cannot restore after it."""
    env = tiny_env_factory(num_envs=1)
    state = env.snapshot()
    with env.model_edit() as edit:
        edit.body("tiny", "link").set(mass=2.0)

    with pytest.raises(ModelEditError, match="model revision"):
        env.restore(state)

    env.restore(env.snapshot())


def test_refresh_failure_restores_cpu_and_device_model(tiny_env_factory, monkeypatch) -> None:
    """A refresh failure after the CPU write restores CPU and device rows."""
    env = tiny_env_factory(num_envs=2)
    link, _ = _ids(env)
    mass_before = float(env.mj_model.body_mass[link.body_id])
    device_before = env._wm.body_mass.numpy().copy()
    model_set_before = env.model_set
    revision_before = env._model_revision

    import unigym.backends.mjwarp.model_edit as module

    def fail_refresh(env: object, provider: object) -> None:
        raise RuntimeError("injected refresh failure")

    monkeypatch.setattr(module, "refresh_published_model", fail_refresh)
    edit = env.model_edit()
    edit.body("tiny", "link").set(mass=2.0)
    with pytest.raises(RuntimeError, match="injected refresh failure"):
        edit.commit()

    assert env.mj_model.body_mass[link.body_id] == pytest.approx(mass_before)
    np.testing.assert_allclose(env._wm.body_mass.numpy(), device_before)
    assert env.model_set is model_set_before
    assert env._model_revision == revision_before


def test_push_failure_restores_cpu_and_resyncs_device(tiny_env_factory, monkeypatch) -> None:
    """A failed device push rolls back the CPU edit and re-pushes the old model."""
    env = tiny_env_factory(num_envs=2)
    link, _ = _ids(env)
    mass_before = float(env.mj_model.body_mass[link.body_id])
    device_before = env._wm.body_mass.numpy().copy()
    revision_before = env._model_revision
    original_push = env._push_model_edit_fields
    calls: list[int] = []

    def flaky_push():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("injected push failure")
        return original_push()

    monkeypatch.setattr(env, "_push_model_edit_fields", flaky_push)
    edit = env.model_edit()
    edit.body("tiny", "link").set(mass=2.0)
    with pytest.raises(RuntimeError, match="injected push failure"):
        edit.commit()

    assert len(calls) >= 2
    assert env.mj_model.body_mass[link.body_id] == pytest.approx(mass_before)
    np.testing.assert_allclose(env._wm.body_mass.numpy(), device_before)
    assert env._model_revision == revision_before
