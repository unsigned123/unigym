"""Run composition activates pack adapters before building the instance."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from unigym.adapters import ADAPTER_REGISTRY
from unigym.run.api import Run

_ADAPTER_SOURCE = (
    "from __future__ import annotations\n\n"
    "from unigym.adapters import Adapter\n"
    "from unigym.pack_api import register_adapter\n\n\n"
    "class RunAdapter(Adapter):\n"
    "    def __init__(self, num_envs=1, device='cpu', **kwargs):\n"
    "        pass\n\n"
    "    def get_obs_requests(self):\n"
    "        return []\n\n"
    "    @property\n"
    "    def obs_dim(self):\n"
    "        return 0\n\n"
    "    @property\n"
    "    def control_dt(self):\n"
    "        return 0.02\n\n"
    "    def compute_obs(self, obs_cache, cmd, env):\n"
    "        return None\n\n"
    "    def act(self, obs):\n"
    "        return None\n\n"
    "    def postprocess_action(self, raw):\n"
    "        return {}\n\n\n"
    "def register(context):\n"
    "    register_adapter('acme.run', RunAdapter)\n"
)


@pytest.fixture(autouse=True)
def _isolate_run_activation():
    """Restore the adapter registry, ``sys.path`` and imported pack modules."""
    path_snapshot = list(sys.path)
    modules_snapshot = set(sys.modules)
    registry_snapshot = set(ADAPTER_REGISTRY)
    yield
    for name in set(ADAPTER_REGISTRY) - registry_snapshot:
        del ADAPTER_REGISTRY[name]
    sys.path[:] = path_snapshot
    for name in set(sys.modules) - modules_snapshot:
        del sys.modules[name]


def _write_pack(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": "unigym.pack/1",
        "name": "adapter_run_pack",
        "version": "1.0.0",
        "api": 1,
        "trust": "code",
        "entrypoint": "adapter_run_fixture:register",
        "provides": {},
    }
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    package = root / "python" / "adapter_run_fixture"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_ADAPTER_SOURCE, encoding="utf-8")


def _config() -> dict:
    return {
        "run_name": "adapter_run",
        "seed": 7,
        "env": {"backend": "isaaclab2", "num_envs": 1},
        "collect": {"steps": 1},
        "articulations": [
            {"name": "arm", "joints": [{"name": "shoulder"}, {"name": "elbow"}]},
        ],
        "controls": [
            {
                "name": "arm",
                "adapter": "acme.run",
                "articulation": "arm",
                "joints": ["shoulder", "elbow"],
                "drive": "position_pd",
                "target": "position",
            },
        ],
        "adapter": {"name": "acme.run"},
    }


def test_run_spec_activates_non_manual_pack_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_pack(tmp_path / "adapter_run_pack")
    monkeypatch.setenv("UNIGYM_PACKS", str(tmp_path / "adapter_run_pack"))
    assert "acme.run" not in ADAPTER_REGISTRY

    run = Run(_config())
    spec = run.spec

    assert spec.adapter_name == "acme.run"
    assert ADAPTER_REGISTRY["acme.run"].__name__ == "RunAdapter"


def test_run_spec_inspection_rebuild_reuses_pack_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spec inspection rebuild must not re-run pack activation side effects."""
    _write_pack(tmp_path / "adapter_run_pack")
    monkeypatch.setenv("UNIGYM_PACKS", str(tmp_path / "adapter_run_pack"))
    run = Run(_config())

    first = run.spec
    run._prepare_inspect()
    second = run.spec

    assert first.adapter_name == second.adapter_name == "acme.run"
    assert ADAPTER_REGISTRY["acme.run"].__name__ == "RunAdapter"
