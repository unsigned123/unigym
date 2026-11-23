"""Watcher height/tilt/joint_limit delegate to canonical condition masks."""

from __future__ import annotations

import pytest
import torch

from unigym.evaluation import EnvView
from unigym.runtime.controllers import ArticulationRuntime
from unigym.watchers import WatchContext, WatchError
from unigym.watchers import templates as tpl


def _runtime(name: str, limits: tuple[torch.Tensor, torch.Tensor] | None) -> ArticulationRuntime:
    return ArticulationRuntime(
        name=name,
        joint_names=("joint",),
        q0=[0.0],
        joint_pos_limits=limits,
    )


def _ctx(record: dict, *, articulations: dict | None = None, num_envs: int = 2) -> WatchContext:
    view = EnvView(
        record=record,
        obs=None,
        env_id=None,
        num_envs=num_envs,
        device="cpu",
        articulations=articulations or {},
    )
    return WatchContext(t=0, bundle={"record": record}, view=view)


def _height_context() -> WatchContext:
    return _ctx({"base_pos_w": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.1]])})


def _tilt_context() -> WatchContext:
    quats = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.7071068, 0.0, 0.7071068, 0.0],
        ],
        dtype=torch.float64,
    )
    return _ctx({"base_quat_w": quats})


def _joint_context(q: torch.Tensor) -> WatchContext:
    limits = (torch.tensor([[-1.5]]), torch.tensor([[1.5]]))
    return _ctx(
        {"articulation/arm/joint_pos": q},
        articulations={"arm": _runtime("arm", limits)},
    )


def test_height_without_env_fires_on_any_env_but_keeps_env_zero_evidence():
    """No explicit env uses mask.any(); message/data keep the env-0 fields."""
    check = tpl.get_template("height")["check"]

    global_any = check(_height_context(), {"entity": "base", "z": 0.15})
    env_zero = check(_height_context(), {"entity": "base", "z": 0.15, "env": 0})
    env_one = check(_height_context(), {"entity": "base", "z": 0.15, "env": 1})

    assert global_any.fired is True
    assert global_any.level.value == "warn"
    assert global_any.data == {"entity": "base", "z": 1.0, "rule": "lt 0.15"}
    assert env_zero.fired is False
    assert env_one.fired is True
    assert env_one.data == {"entity": "base", "z": 0.1, "rule": "lt 0.15"}
    assert env_one.message == "base z=0.100 lt 0.15"


def test_height_band_and_op_modes_keep_their_existing_diagnostics():
    """Band and comparison modes still report the watcher's rule text."""
    check = tpl.get_template("height")["check"]

    band = check(_height_context(), {"entity": "base", "band": [0.0, 0.05]})
    comparison = check(_height_context(), {"entity": "base", "z": 0.5, "op": "gt", "env": 1})

    assert band.fired is True
    assert band.data == {"entity": "base", "z": 1.0, "rule": "outside band [0.0, 0.05]"}
    assert comparison.fired is False


def test_height_offline_context_still_evaluates_without_a_view():
    """The documented bundle-only fallback keeps working for height."""
    ctx = WatchContext(
        t=0,
        bundle={"record": {"base_pos_w": torch.tensor([[0.0, 0.0, 0.1]])}},
    )

    result = tpl.get_template("height")["check"](ctx, {"entity": "base", "z": 0.15})

    assert result.fired is True
    assert result.data["z"] == 0.1


def test_tilt_without_env_fires_on_any_env_and_honors_explicit_env():
    """Tilt uses global any by default and mask[env] when env is explicit."""
    check = tpl.get_template("tilt")["check"]

    global_any = check(_tilt_context(), {"entity": "base", "max_deg": 60.0})
    env_zero = check(_tilt_context(), {"entity": "base", "max_deg": 60.0, "env": 0})
    env_one = check(_tilt_context(), {"entity": "base", "max_deg": 60.0, "env": 1})

    assert global_any.fired is True
    assert global_any.data == {
        "entity": "base",
        "tilt_deg": 0.0,
        "max_deg": 60.0,
        "key": "record/base_quat_w",
        "env": 0,
    }
    assert env_zero.fired is False
    assert env_one.fired is True
    assert env_one.data["tilt_deg"] == 90.0
    assert env_one.data["env"] == 1
    assert env_one.message == "base tilt 90.0 deg > 60.0"


def test_tilt_offline_context_still_evaluates_without_a_view():
    """The bundle-only fallback also covers the quaternion tilt source."""
    ctx = WatchContext(
        t=0,
        bundle={
            "record": {
                "base_quat_w": torch.tensor([[0.7071068, 0.0, 0.7071068, 0.0]], dtype=torch.float64)
            }
        },
    )

    result = tpl.get_template("tilt")["check"](ctx, {"entity": "base", "max_deg": 60.0})

    assert result.fired is True
    assert result.data["tilt_deg"] == 90.0


def test_joint_limit_uses_watcher_magnitude_margin_and_reports_envs():
    """The magnitude mode keeps the watcher margin, env list and worst_over."""
    check = tpl.get_template("joint_limit")["check"]
    ctx = _joint_context(torch.tensor([[1.44], [1.39]]))

    global_any = check(ctx, {"articulation": "arm"})
    env_zero = check(ctx, {"articulation": "arm", "env": 0})
    env_one = check(ctx, {"articulation": "arm", "env": 1})

    assert global_any.fired is True
    assert global_any.data == {
        "articulation": "arm",
        "envs": [0],
        "worst_over": 0.04,
        "margin": 0.1,
    }
    assert global_any.message == "joint limit margin 0.1: envs [0]"
    assert env_zero.fired is True
    assert env_one.fired is False


def test_joint_limit_magnitude_boundary_at_the_watcher_margin():
    """A joint inside max(|limits|-margin) does not fire in magnitude mode."""
    check = tpl.get_template("joint_limit")["check"]
    ctx = _joint_context(torch.tensor([[1.39], [1.4]]))

    assert check(ctx, {"articulation": "arm"}).fired is True
    assert check(ctx, {"articulation": "arm", "env": 0}).fired is False
    assert check(ctx, {"articulation": "arm", "env": 1}).fired is True


def test_joint_limit_missing_joint_position_is_a_watch_error():
    """A missing joint-position leaf stays a data-shaped watcher failure."""
    limits = (torch.tensor([[-1.5]]), torch.tensor([[1.5]]))
    ctx = _ctx({}, articulations={"arm": _runtime("arm", limits)})

    with pytest.raises(WatchError, match="joint_pos"):
        tpl.get_template("joint_limit")["check"](ctx, {"articulation": "arm"})
