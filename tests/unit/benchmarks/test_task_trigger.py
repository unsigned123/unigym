"""Task trigger: opt-in cadence, construction and build-time validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from unigym.benchmarks.base import (
    TASK_REGISTRY,
    EpisodeResult,
    Task,
    TickResult,
    build_tasks,
)
from unigym.pack_api import discover

_ARTICULATIONS = {"articulations": {}}


class _EpisodeTask(Task):
    """Default-cadence task: ``score`` only."""

    name = "trigger_episode"

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={}, success=None)


class _TickableTask(Task):
    """Task that *can* run per tick but defaults to episode cadence."""

    name = "trigger_tickable"

    def on_tick(self, ctx):
        del ctx
        return TickResult()

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={}, success=None)


class _BothNoOverride(Task):
    """Declared ``both`` without overriding the default no-op."""

    name = "trigger_both_no_override"
    trigger = "both"

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={}, success=None)


class _BothWrongSignature(Task):
    """Declared ``both`` with an extra positional ``on_tick`` parameter."""

    name = "trigger_both_wrong_signature"
    trigger = "both"

    def on_tick(self, ctx, extra):
        del ctx, extra
        return TickResult()

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={}, success=None)


def _build_registered(cls: type[Task]) -> list[Task]:
    """Register one class-backed task and build it through ``build_tasks``."""
    TASK_REGISTRY[cls.name] = cls
    try:
        return build_tasks([{"name": cls.name}], dict(_ARTICULATIONS), discover([]))
    finally:
        TASK_REGISTRY.pop(cls.name, None)


def test_trigger_defaults_to_episode():
    """A task that declares no cadence is evaluated at episode end only."""
    task = _TickableTask()

    assert Task.trigger == "episode"
    assert task.trigger == "episode"


def test_trigger_param_overrides_the_class_attribute():
    """An authored ``trigger: both`` wins over an episode class default."""
    task = _TickableTask(params={"trigger": "both"})

    assert task.trigger == "both"
    assert task.params["trigger"] == "both"


def test_trigger_invalid_value_fails_at_construction():
    """A value outside ``episode|both`` is rejected at construction."""
    with pytest.raises(ValueError, match=r"trigger must be 'episode' or 'both'"):
        _TickableTask(params={"trigger": "tick"})


def test_build_accepts_an_overriding_task_with_a_both_param():
    """Build resolves the authored param and accepts a usable both hook."""
    TASK_REGISTRY[_TickableTask.name] = _TickableTask
    try:
        tasks = build_tasks(
            [{"name": _TickableTask.name, "trigger": "both"}],
            dict(_ARTICULATIONS),
            discover([]),
        )
    finally:
        TASK_REGISTRY.pop(_TickableTask.name, None)

    assert [task.trigger for task in tasks] == ["both"]


def test_trigger_both_without_an_on_tick_override_fails_at_build():
    """The inherited no-op is not a usable ``both`` hook."""
    with pytest.raises(ValueError, match=r"does not override the default no-op"):
        _build_registered(_BothNoOverride)


def test_trigger_both_with_a_wrong_on_tick_signature_fails_at_build():
    """``on_tick`` must be exactly ``on_tick(self, ctx)``."""
    with pytest.raises(ValueError, match=r"on_tick\(self, ctx\)"):
        _build_registered(_BothWrongSignature)


def test_formula_task_cannot_declare_trigger_both():
    """A formula section has no ``on_tick`` override and fails at build."""
    section = {
        "name": "formula_both",
        "class": "formula",
        "trigger": "both",
        "description": "trigger probe",
        "metrics": [
            {
                "id": "near_zero",
                "term": "frame_fraction",
                "key": "time_in_episode",
                "target": 0.0,
                "tol": 1.0,
            }
        ],
    }

    with pytest.raises(ValueError, match=r"does not override the default no-op"):
        build_tasks([section], dict(_ARTICULATIONS), discover([]))


def _write_plugin_pack(tmp_path: Path) -> Path:
    """Write a content pack whose Python plugin declares ``trigger: both``."""
    pack = tmp_path / "acme_bench"
    bench = pack / "benchmarks" / "hold"
    (bench / "tasks").mkdir(parents=True)
    (pack / "unigym.pack.yaml").write_text(
        """\
schema: unigym.pack/1
name: acme_bench
version: 1.0.0
api: 1
trust: content
provides:
  benchmarks:
    - id: acme_bench.hold
      resource: benchmarks/hold
""",
        encoding="utf-8",
    )
    (bench / "pack.yaml").write_text("name: acme_hold\nversion: 0.1.0\n", encoding="utf-8")
    (bench / "tasks" / "plugin.py").write_text(
        '''\
"""trusted-source trigger probe plugin."""

from unigym.benchmarks.base import EpisodeResult, Task


class PluginBoth(Task):
    """Declared both without overriding on_tick."""

    name = "plugin_both"
    trigger = "both"

    def score(self, ep, ctx):
        del ep, ctx
        return EpisodeResult(metrics={}, success=None)


TASK = PluginBoth
''',
        encoding="utf-8",
    )
    return pack


def test_pack_plugin_trigger_both_is_rejected_with_its_canonical_name(tmp_path: Path):
    """A pack plugin is validated like any task and named by canonical key."""
    runtime = discover([_write_plugin_pack(tmp_path)])

    with pytest.raises(ValueError, match=r"'acme_hold/plugin_both'.*does not override"):
        build_tasks([{"name": "acme_hold/plugin_both"}], dict(_ARTICULATIONS), runtime)


def test_wildcard_section_cannot_carry_a_trigger_param():
    """A wildcard section is exactly ``{name}`` -- per-task params are ambiguous."""
    with pytest.raises(ValueError, match=r"wildcard section .* extra keys \['trigger'\]"):
        build_tasks(
            [{"name": "builtin/*", "trigger": "both"}],
            dict(_ARTICULATIONS),
            discover([]),
        )
