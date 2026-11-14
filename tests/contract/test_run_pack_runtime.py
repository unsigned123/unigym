"""The additive ``Run(pack_runtime=...)`` seam (no ambient discovery)."""

from __future__ import annotations

from pathlib import Path
from typing import NoReturn

import pytest

from unigym._typing import JsonValue
from unigym.errors import ConfigError
from unigym.pack_api import PackRuntime, discover
from unigym.packs import PackError
from unigym.run.api import Run, load


def _runtime_lacking_the_preset() -> PackRuntime:
    """An explicit empty runtime: its catalog cannot resolve any preset."""
    return discover([])


def _runtime_cfg() -> dict[str, JsonValue]:
    """A config that needs a runtime to resolve its ``scene_preset``."""
    return {
        "run_name": "r",
        "seed": 0,
        "env": {
            "backend": "mujoco",
            "num_envs": 1,
            "episode_length_s": 10.0,
            "decimation": 4,
            "sim_dt": 0.005,
            "params": {},
        },
        "collect": {"steps": 1},
        "scene_preset": "missing.preset",
    }


def _plain_cfg() -> dict[str, JsonValue]:
    """A config that needs no runtime at all."""
    return {
        "run_name": "r",
        "seed": 0,
        "env": {
            "backend": "mujoco",
            "num_envs": 1,
            "episode_length_s": 10.0,
            "decimation": 4,
            "sim_dt": 0.005,
            "params": {},
        },
        "world_base": {"mjcf": "model.xml"},
        "collect": {"steps": 1},
        "articulations": [{"name": "arm", "joints": [{"name": "shoulder"}]}],
        "controls": [
            {
                "name": "arm",
                "adapter": "manual",
                "articulation": "arm",
                "joints": ["shoulder"],
                "drive": "position_pd",
                "target": "position",
            }
        ],
    }


def _forbidden_discovery(*args: object, **kwargs: object) -> NoReturn:
    """Fail the test if the ambient discovery path is touched."""
    raise AssertionError("Run must not discover when pack_runtime is supplied")


def test_injected_runtime_is_used_and_discovery_is_never_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supplied runtime reaches ``build_spec`` without ambient discovery."""
    runtime = _runtime_lacking_the_preset()
    monkeypatch.setattr("unigym.pack_api.discover", _forbidden_discovery)

    with pytest.raises(PackError, match="missing.preset"):
        _ = Run(dict(_runtime_cfg()), pack_runtime=runtime).spec


def test_injected_runtime_is_stored_on_the_spec_even_without_a_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An injected runtime is the spec's runtime, not a requirement-only branch."""
    runtime = _runtime_lacking_the_preset()
    monkeypatch.setattr("unigym.pack_api.discover", _forbidden_discovery)

    spec = Run(dict(_plain_cfg()), pack_runtime=runtime).spec

    assert spec.pack_runtime is runtime


def test_none_preserves_the_ambient_discovery_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``pack_runtime=None`` keeps the historical discovery path exactly."""
    runtime = _runtime_lacking_the_preset()
    calls: list[bool] = []

    def _spy_discovery(*, entry_points: bool = False) -> PackRuntime:
        calls.append(entry_points)
        return runtime

    monkeypatch.setattr("unigym.pack_api.discover", _spy_discovery)

    with pytest.raises(PackError, match="missing.preset"):
        _ = Run(dict(_runtime_cfg())).spec

    assert calls == [True]


def test_discover_runtime_false_raises_the_teaching_error_without_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The facade owner boundary refuses instead of discovering."""
    monkeypatch.setattr("unigym.pack_api.discover", _forbidden_discovery)

    with pytest.raises(ConfigError, match="pack_set=.*pack_runtime="):
        _ = Run(dict(_runtime_cfg()), discover_runtime=False).spec


def _write_runtime_config(tmp_path: Path) -> Path:
    """Write the pack-needing yaml the ``load`` seams read."""
    config = tmp_path / "run.yaml"
    config.write_text(
        "\n".join(
            [
                "run_name: r",
                "seed: 0",
                "env: {backend: mujoco, num_envs: 1, episode_length_s: 10.0,",
                "      decimation: 4, sim_dt: 0.005, params: {}}",
                "collect: {steps: 1}",
                "scene_preset: missing.preset",
            ]
        ),
        encoding="utf-8",
    )
    return config


def test_load_forwards_the_injected_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load(path, pack_runtime=...)`` reaches the same injected seam."""
    config = _write_runtime_config(tmp_path)
    runtime = _runtime_lacking_the_preset()
    monkeypatch.setattr("unigym.pack_api.discover", _forbidden_discovery)

    with pytest.raises(PackError, match="missing.preset"):
        _ = load(config, pack_runtime=runtime).spec


def test_load_forwards_discover_runtime_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load(path, discover_runtime=False)`` refuses without discovery."""
    config = _write_runtime_config(tmp_path)
    monkeypatch.setattr("unigym.pack_api.discover", _forbidden_discovery)

    with pytest.raises(ConfigError, match="pack_set=.*pack_runtime="):
        _ = load(config, discover_runtime=False).spec
