"""``ExternalRunner.from_instance`` forwards the facade reset policy."""

from __future__ import annotations

from types import SimpleNamespace

from unigym.runner import make_runner
from unigym.runner.external import ExternalRunner
from unigym.runtime.rewind_providers import RunnerLifecycleProvider
from unigym.terminations import TerminationGroup


def _instance(runner_cfg, *, providers=None):
    """A minimal RunInstance double carrying *runner_cfg* and provider list."""
    spec = SimpleNamespace(
        cfg={"collect": {"steps": 4}},
        runner=runner_cfg,
        adapter_name="manual",
    )
    return SimpleNamespace(
        spec=spec,
        profile=None,
        adapter=object(),
        sampler=None,
        term_group=TerminationGroup(),
        env=SimpleNamespace(controls=None),
        sink=None,
        replay=None,
        reset_on_termination=False,
        rewind_providers=[] if providers is None else providers,
    )


def _runner(runner_cfg):
    """Build an ExternalRunner from a minimal double carrying *runner_cfg*."""
    return ExternalRunner.from_instance(_instance(runner_cfg))


def test_from_instance_forwards_auto_reset_false():
    """A facade spec's ``runner.auto_reset: false`` reaches the runner."""
    runner = _runner({"auto_reset": False})

    assert runner.auto_reset is False


def test_from_instance_defaults_auto_reset_to_true():
    """An absent ``runner.auto_reset`` key keeps the runner default."""
    runner = _runner({})

    assert runner.auto_reset is True


def test_make_runner_attaches_the_lifecycle_provider_once():
    """make_runner adds one runner_lifecycle provider and keeps other providers."""
    other = object()
    instance = _instance({}, providers=[other])

    runner = make_runner(instance, kind="external")
    again = make_runner(instance, kind="external")

    providers = instance.rewind_providers
    lifecycle = [p for p in providers if isinstance(p, RunnerLifecycleProvider)]
    assert len([p for p in providers if p is other]) == 1
    assert len(lifecycle) == 1
    assert lifecycle[0].runner is again
    assert again is not runner
