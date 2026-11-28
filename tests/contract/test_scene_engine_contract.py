"""The scene-engine contract: the IsaacLab backend capability and its engine."""

from __future__ import annotations

import subprocess
import sys

import pytest

from unigym.backends.compose import backend_runtime
from unigym.capabilities.traits import Trait


def test_isaaclab2_declares_scene_compose_capability():
    """The backend declares scene composition without content/world-base faces."""
    runtime = backend_runtime("isaaclab2")
    spec = runtime.spec

    assert spec.has(Trait.SCENE_COMPOSE)
    assert spec.scene_provider_consumer is True
    assert runtime.bindings.scene_engine is not None
    assert spec.asset_formats == ("usd", "urdf")
    assert spec.content_fill is False
    assert spec.world_base_consumer is False


def test_isaaclab2_scene_engine_factory_returns_a_protocol_engine():
    """The composed factory builds an engine satisfying the neutral protocol."""
    from unigym.scenes.engine import SceneEngine

    factory = backend_runtime("isaaclab2").bindings.scene_engine
    assert factory is not None

    engine = factory()

    assert isinstance(engine, SceneEngine)
    assert engine.name == "isaaclab2"


def test_scene_engine_registry_agrees_with_the_backend_engine():
    """``scene_engine('isaaclab2')`` returns a fresh backend engine instance."""
    from unigym.backends.isaaclab2.asset_scene.engine import AssetSceneEngine
    from unigym.scenes.engine import SceneEngine, scene_engine

    engine = scene_engine("isaaclab2")

    assert isinstance(engine, SceneEngine)
    assert isinstance(engine, AssetSceneEngine)
    assert engine.name == "isaaclab2"


@pytest.mark.subprocess
def test_registry_import_is_lazy_and_imports_no_engine():
    """Importing the registry registers no engine and loads no heavy module."""
    code = (
        "import sys\n"
        "from unigym.backends.compose import backend_runtime\n"
        "from unigym.scenes.engine import registered_scene_engines\n"
        "registered = registered_scene_engines()\n"
        "assert registered == [], registered\n"
        "roots = ('isaaclab', 'pxr', 'omni')\n"
        "loaded = sorted(\n"
        "    name for name in sys.modules\n"
        "    if any(name == root or name.startswith(root + '.') for root in roots)\n"
        ")\n"
        "assert not loaded, loaded\n"
        "assert backend_runtime('isaaclab2').bindings.scene_engine is not None\n"
    )

    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
