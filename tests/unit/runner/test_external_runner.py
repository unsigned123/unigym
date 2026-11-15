"""``ExternalRunner.from_instance`` forwards the facade reset policy."""

from __future__ import annotations

from types import SimpleNamespace

from unigym.runner.external import ExternalRunner
from unigym.terminations import TerminationGroup


def _runner(runner_cfg):
    """Build an ExternalRunner from a minimal double carrying *runner_cfg*."""
    spec = SimpleNamespace(
        cfg={"collect": {"steps": 4}},
        runner=runner_cfg,
        adapter_name="manual",
    )
    instance = SimpleNamespace(
        spec=spec,
        profile=None,
        adapter=object(),
        sampler=None,
        term_group=TerminationGroup(),
        env=SimpleNamespace(controls=None),
        sink=None,
        replay=None,
        reset_on_termination=False,
    )
    return ExternalRunner.from_instance(instance)


def test_from_instance_forwards_auto_reset_false():
    """A facade spec's ``runner.auto_reset: false`` reaches the runner."""
    runner = _runner({"auto_reset": False})

    assert runner.auto_reset is False


def test_from_instance_defaults_auto_reset_to_true():
    """An absent ``runner.auto_reset`` key keeps the runner default."""
    runner = _runner({})

    assert runner.auto_reset is True
