"""Run termination ownership: ``core`` (default) versus ``external`` mode."""

from __future__ import annotations

from argparse import Namespace

import pytest
import torch
import yaml

from tests._support.fakes.backend import FakeBackend
from unigym.adapters.manual import ManualAdapter
from unigym.errors import ConfigError
from unigym.run.api import Run
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


@pytest.mark.parametrize("mode", ["core", "external"])
def test_config_round_trip_preserves_termination_mode(tmp_path, mode):
    """``to_config`` re-emits the top-level mode and it re-parses identically."""
    cfg = _external_cfg() if mode == "external" else _cfg()
    cfg["termination_mode"] = mode
    source = tmp_path / f"{mode}.yaml"
    source.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    out = tmp_path / f"{mode}.out.yaml"

    Run.from_config(str(source)).to_config(out)

    reloaded = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert reloaded["termination_mode"] == mode
    assert build_spec(reloaded, _args()).termination_mode == mode


@pytest.mark.parametrize(
    "value", ["task", "", 1, None, True], ids=["unknown", "empty", "int", "none", "bool"]
)
def test_unknown_termination_mode_is_rejected(value):
    """Only ``core`` and ``external`` are supported termination modes."""
    with pytest.raises(ConfigError, match="termination_mode=.* is not supported"):
        build_spec(_cfg(termination_mode=value), _args())
