"""Run termination ownership: ``core`` (default) versus ``external`` mode."""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch

from tests._support.fakes.backend import FakeBackend
from unigym.adapters.manual import ManualAdapter
from unigym.errors import ConfigError
from unigym.run.build import _core_termination_group
from unigym.run.spec import build_spec
from unigym.runner.external import ExternalRunner
from unigym.runtime.actions import NamedAction


def _cfg(**overrides):
    cfg = {
        "run_name": "r",
        "seed": 7,
        "env": {"backend": "isaaclab2", "num_envs": 1},
        "collect": {"steps": 1},
        "articulations": [
            {"name": "arm", "joints": [{"name": "shoulder"}, {"name": "elbow"}]},
        ],
        "action_spec": [
            {
                "name": "arm",
                "articulation": "arm",
                "joints": ["shoulder", "elbow"],
                "target": "position",
                "role": "body",
            },
        ],
        "policies": [{"adapter": "manual", "groups": [["arm", "arm"]]}],
        "observations": {"robot": ["arm"]},
    }
    cfg.update(overrides)
    return cfg


def _external_cfg(**overrides):
    base = {
        "termination_mode": "external",
        "terminations": [],
        "runner": {"auto_reset": False, "reset_on_termination": False},
    }
    base.update(overrides)
    return _cfg(**base)


def _args():
    return Namespace(config="run.yaml")


class _NullSink:
    """The sink boundary the runner steps through; it records nothing."""

    def on_step(self, t, bundle) -> None:
        """Accept one pre-reset engine step."""

    def on_episode_end(self, env_ids, bundle) -> None:
        """Accept the episode-end data side."""

    def on_step_end(self, t, bundle) -> None:
        """Accept one post-reset engine step."""

    def on_run_end(self) -> dict:
        """Return the empty run summary."""
        return {}


def test_core_mode_is_the_default_and_keeps_the_config_terminations():
    """An absent mode is ``core`` and the ``terminations:`` section is unchanged."""
    terminations = [{"name": "fall", "articulation": "arm"}]

    spec = build_spec(_cfg(terminations=terminations), _args())

    assert spec.termination_mode == "core"
    assert spec.terminations == terminations
    assert _core_termination_group(spec, spec.cfg).names == ["fall"]


@pytest.mark.parametrize("terminations", [None, []], ids=["absent", "empty"])
def test_external_mode_builds_an_empty_termination_group(terminations):
    """External mode accepts no core termination source and builds no group."""
    cfg = _external_cfg()
    if terminations is None:
        cfg.pop("terminations")
    else:
        cfg["terminations"] = terminations

    spec = build_spec(cfg, _args())

    assert spec.termination_mode == "external"
    assert spec.runner == {"auto_reset": False, "reset_on_termination": False}
    assert len(_core_termination_group(spec, spec.cfg)) == 0


def test_external_mode_core_step_reports_no_terminations():
    """A driven external-mode lane returns an all-False core terminated mask."""
    spec = build_spec(_external_cfg(), _args())
    term_group = _core_termination_group(spec, spec.cfg)
    runner = ExternalRunner(
        {"collect": {"steps": 1}},
        None,
        ManualAdapter(),
        None,
        term_group,
        FakeBackend(),
        _NullSink(),
        reset_on_termination=False,
        auto_reset=False,
    )

    result = runner.step(
        (
            NamedAction(
                articulation="arm",
                values=torch.zeros(1, 1),
                joint_names=("shoulder",),
                kind="position",
            ),
        )
    )

    assert result.terminated is not None
    assert tuple(result.terminated.shape) == (1,)
    assert not bool(result.terminated.any())
    assert runner.auto_reset is False
    assert runner.reset_on_termination is False


@pytest.mark.parametrize(
    "terminations",
    [[{"name": "fall", "articulation": "arm"}], {"fall": True}],
    ids=["non-empty-list", "non-empty-mapping"],
)
def test_external_mode_rejects_a_core_termination_source(terminations):
    """External mode refuses any non-empty ``terminations:`` declaration."""
    with pytest.raises(ConfigError, match="terminations.*must be absent or empty"):
        build_spec(_external_cfg(terminations=terminations), _args())


@pytest.mark.parametrize(
    "runner",
    [{}, {"reset_on_termination": False}, {"auto_reset": True, "reset_on_termination": False}],
    ids=["missing", "missing-with-reset-false", "true"],
)
def test_external_mode_requires_explicit_auto_reset_false(runner):
    """An absent or true ``runner.auto_reset`` is a teaching config error."""
    with pytest.raises(ConfigError, match="requires an explicit runner.auto_reset: false"):
        build_spec(_external_cfg(runner=runner), _args())


def test_external_mode_rejects_reset_on_termination_true():
    """External mode never resets a terminated row inside the run."""
    runner = {"auto_reset": False, "reset_on_termination": True}

    with pytest.raises(ConfigError, match="requires runner.reset_on_termination: false"):
        build_spec(_external_cfg(runner=runner), _args())


def test_external_mode_accepts_an_explicit_null_episode_length():
    """``episode_length_s: null`` disables the core clock in external mode."""
    cfg = _external_cfg()
    cfg["env"]["episode_length_s"] = None

    spec = build_spec(cfg, _args())

    assert spec.termination_mode == "external"
    assert spec.cfg["env"]["episode_length_s"] is None


def test_external_mode_accepts_an_omitted_episode_length():
    """External mode owns truncation, so an absent clock is accepted as None."""
    cfg = _external_cfg()
    assert "episode_length_s" not in cfg["env"]

    spec = build_spec(cfg, _args())

    assert spec.termination_mode == "external"
    assert "episode_length_s" not in spec.cfg["env"]


@pytest.mark.parametrize(
    ("termination_mode", "env_section", "expected"),
    [
        ("external", {}, None),
        ("external", {"episode_length_s": None}, None),
        ("core", {}, 10.0),
        ("core", {"episode_length_s": 2.5}, 2.5),
    ],
    ids=["external-omitted", "external-null", "core-default", "core-explicit"],
)
def test_episode_clock_resolution_respects_the_termination_mode(
    termination_mode, env_section, expected
):
    """An omitted clock is disabled for external mode but keeps 10 s for core."""
    from types import SimpleNamespace

    from unigym.adapters.manual import ManualAdapter
    from unigym.backends.registry import backend_info
    from unigym.env.obs import ObservationConfig
    from unigym.run.build import _build_sensing

    spec = SimpleNamespace(
        enable_cameras=False,
        direct=False,
        articulations=(),
        termination_mode=termination_mode,
    )
    sensing = _build_sensing(
        spec,
        {"env": env_section},
        backend_info("mujoco"),
        "cpu",
        1,
        {},
        {},
        {},
        {},
        ManualAdapter(),
        0.02,
        None,
        ObservationConfig(robot=()),
    )

    assert sensing["episode_length_s"] == expected


def test_core_mode_rejects_a_null_episode_length():
    """Core mode always owns time, so a null clock is a teaching error."""
    cfg = _cfg()
    cfg["env"]["episode_length_s"] = None

    with pytest.raises(ConfigError, match="episode_length_s: null"):
        build_spec(cfg, _args())


def test_core_mode_keeps_the_numeric_duration_unchanged():
    """A numeric core duration still passes through untouched."""
    cfg = _cfg()
    cfg["env"]["episode_length_s"] = 2.5

    spec = build_spec(cfg, _args())

    assert spec.termination_mode == "core"
    assert spec.cfg["env"]["episode_length_s"] == 2.5


def test_external_mode_rejects_a_numeric_episode_length():
    """Two independent clocks (core duration and task budget) are refused."""
    cfg = _external_cfg()
    cfg["env"]["episode_length_s"] = 10.0

    with pytest.raises(ConfigError, match="two independent clocks"):
        build_spec(cfg, _args())


@pytest.mark.parametrize(
    "value", ["task", "", 1, None, True], ids=["unknown", "empty", "int", "none", "bool"]
)
def test_unknown_termination_mode_is_rejected(value):
    """Only ``core`` and ``external`` are supported termination modes."""
    with pytest.raises(ConfigError, match="termination_mode=.* is not supported"):
        build_spec(_cfg(termination_mode=value), _args())


def test_parse_episode_length_rejects_none():
    """The numeric-only parser leaves the null-clock branch to its callers."""
    from unigym.env.episode import parse_episode_length

    with pytest.raises(ValueError, match="disables the core episode clock"):
        parse_episode_length(None)
