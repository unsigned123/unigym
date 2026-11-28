"""Deprecated ``.live`` warnings and the ``live_for`` replacement."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import pytest

from unigym._typing import JsonValue
from unigym.model import ModelIR, ModelSet
from unigym.run.api import Run
from unigym.run.instance import RunInstance


def _cfg() -> dict[str, JsonValue]:
    """A small valid config; the tests fake the build and never start an env."""
    return {
        "run_name": "r",
        "seed": 0,
        "env": {"backend": "mujoco", "num_envs": 1},
        "world_base": {"mjcf": "model.xml"},
        "collect": {"steps": 1},
        "articulations": [{"name": "arm", "joints": [{"name": "shoulder"}]}],
        "action_spec": [
            {
                "name": "arm",
                "articulation": "arm",
                "joints": ["shoulder"],
                "target": "position",
                "role": "body",
            }
        ],
        "policies": [{"adapter": "manual", "groups": [["arm", "arm"]]}],
        "observations": {"robot": ["arm"]},
    }


class _LiveEnv:
    """The minimum env face the real ``live_for`` reads (no engine)."""

    def __init__(self) -> None:
        ir = ModelIR(name="arm", backend="mujoco", compiled=True)
        self.model_set = ModelSet({"arm": ir}, merged=ir)


def _instance() -> RunInstance:
    from unigym.backends.compose import backend_runtime

    return RunInstance(
        spec=SimpleNamespace(backend="mujoco"),
        profile=None,
        adapter=None,
        sampler=None,
        term_group=None,
        env=_LiveEnv(),
        bridge=None,
        live_provider=backend_runtime("mujoco").bindings.live_provider,
    )


def test_run_instance_live_warns_once_per_access_and_delegates(monkeypatch) -> None:
    """``RunInstance.live`` warns once per access and still calls ``live_for``."""
    instance = _instance()
    sentinel = object()
    calls: list[str | None] = []

    def live_for(name: str | None = None) -> object:
        calls.append(name)
        return sentinel

    monkeypatch.setattr(instance, "live_for", live_for)

    with pytest.warns(DeprecationWarning, match="live_for") as first:
        first_value = instance.live
    with pytest.warns(DeprecationWarning, match="live_for") as second:
        second_value = instance.live

    assert first_value is sentinel
    assert second_value is sentinel
    assert calls == [None, None]
    assert len(first) == 1
    assert len(second) == 1
    assert "query for typed reads" in str(first[0].message)


def test_run_live_warns_once_per_access_and_delegates(monkeypatch) -> None:
    """``Run.live`` warns once per access and still calls ``live_for``."""
    run = Run(_cfg())
    sentinel = object()
    calls: list[str | None] = []

    def live_for(name: str | None = None) -> object:
        calls.append(name)
        return sentinel

    monkeypatch.setattr(run, "live_for", live_for)

    with pytest.warns(DeprecationWarning, match=r"run\.build\(\)\.query") as first:
        first_value = run.live
    with pytest.warns(DeprecationWarning, match=r"run\.build\(\)\.query") as second:
        second_value = run.live

    assert first_value is sentinel
    assert second_value is sentinel
    assert calls == [None, None]
    assert len(first) == 1
    assert len(second) == 1
    assert "live_for(name)" in str(first[0].message)


def test_live_for_emits_no_deprecation_warning(monkeypatch) -> None:
    """The replacement ``live_for`` path warns on neither facade."""
    instance = _instance()
    run = Run(_cfg())
    monkeypatch.setattr(run, "build", lambda inspect=False: instance)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        instance_live = instance.live_for("arm")
        run_live = run.live_for("arm")

    deprecations = [item for item in caught if issubclass(item.category, DeprecationWarning)]
    assert deprecations == []
    assert type(instance_live) is type(run_live)
