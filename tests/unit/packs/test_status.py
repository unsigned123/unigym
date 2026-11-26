"""Activation status snapshots: ``PackItem.status`` carries the outcome."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from unigym.actuator_law import (
    ACTUATOR_LAW_REGISTRY,
    restore_actuator_laws,
    snapshot_actuator_laws,
)
from unigym.actuators.implementations import (
    ACTUATOR_IMPLEMENTATION_REGISTRY,
    restore_actuator_implementations,
    snapshot_actuator_implementations,
)
from unigym.pack_api import ItemStatus
from unigym.packs import PackError
from unigym.packs.loader import discover
from unigym.registration import reset_registration
from unigym.scenes.providers import (
    registered_scene_providers,
    restore_scene_providers,
    scene_provider_origin,
    snapshot_scene_providers,
)

_SUPPORT = Path(__file__).resolve().parents[2] / "_support" / "packs"
CONTENT_ROOT = _SUPPORT / "acme_content"
CODE_ROOT = _SUPPORT / "acme_code"
ACTUATORS_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "packs" / "acme_actuators"


@pytest.fixture(autouse=True)
def _isolate_registry_state():
    """Restore the registries and the pack modules activation can import.

    Third-party modules imported during a test stay loaded: some of them guard
    against re-initialization (torch refuses a second import in one process),
    and none of them carries pack or registry state.
    """
    path_snapshot = list(sys.path)
    scene_snapshot = snapshot_scene_providers()
    laws_snapshot = snapshot_actuator_laws()
    implementations_snapshot = snapshot_actuator_implementations()
    modules_snapshot = set(sys.modules)
    venv_prefix = Path(sys.prefix).resolve()
    reset_registration()
    yield
    sys.path[:] = path_snapshot
    restore_scene_providers(scene_snapshot)
    restore_actuator_laws(laws_snapshot)
    restore_actuator_implementations(implementations_snapshot)
    reset_registration()
    for name in set(sys.modules) - modules_snapshot:
        module = sys.modules.get(name)
        origin = getattr(module, "__file__", None)
        if origin is not None:
            try:
                Path(origin).resolve().relative_to(venv_prefix)
            except ValueError:
                pass
            else:
                continue
        del sys.modules[name]


def _write_pack(
    root: Path,
    document: dict[str, object],
    modules: dict[str, str] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    for module_name, source in (modules or {}).items():
        package = root / "python" / module_name
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(source, encoding="utf-8")
    return root


def _document(
    name: str,
    providers: list[dict[str, object]],
    *,
    entrypoint: str | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": "unigym.pack/1",
        "name": name,
        "version": "1.0.0",
        "api": 1,
        "trust": "code",
        "provides": {"providers": providers},
    }
    if entrypoint is not None:
        document["entrypoint"] = entrypoint
    return document


def _status_by_id(activated) -> dict[str, ItemStatus]:
    return {item.id: item.status for item in activated.items}


def test_activation_status_matrix() -> None:
    """Data stays declared, registered kinds register, inert kinds stay inert."""
    loader = discover([CONTENT_ROOT, CODE_ROOT, ACTUATORS_ROOT])

    content = loader.activate("acme_content")
    assert {item.status for item in content.items} == {ItemStatus.DECLARED}

    code = loader.activate("acme_code")
    assert _status_by_id(code) == {
        "acme_code.dynamic_assets": ItemStatus.INERT,
        "acme_code.table_top": ItemStatus.REGISTERED,
    }

    actuators = loader.activate("acme_actuators")
    assert _status_by_id(actuators) == {
        "acme.compliant_pd": ItemStatus.REGISTERED,
        "acme.compliant_pd.mujoco": ItemStatus.INERT,
    }
    assert "acme.compliant_pd" in ACTUATOR_LAW_REGISTRY
    assert ("acme.compliant_pd", "mujoco") not in ACTUATOR_IMPLEMENTATION_REGISTRY


def test_unknown_provider_kind_is_inert(tmp_path: Path) -> None:
    """An unrecognized provider kind is declared code that activation ignored."""
    root = tmp_path / "unknown_kind"
    root.mkdir()
    (root / "value.yaml").write_text("kind: unknown\n", encoding="utf-8")
    _write_pack(
        root,
        {
            "schema": "unigym.pack/1",
            "name": "unknown_kind",
            "version": "1.0.0",
            "api": 1,
            "trust": "content",
            "provides": {
                "providers": [{"id": "acme.mystery", "kind": "mystery", "resource": "value.yaml"}]
            },
        },
    )

    activated = discover([root]).activate("unknown_kind")

    assert _status_by_id(activated) == {"acme.mystery": ItemStatus.INERT}


def test_scene_mirror_is_adopted_and_pack_code_is_not_an_item(tmp_path: Path) -> None:
    """Adoption replaces the pack-code origin; pack-code names have no PackItem."""
    source = (
        '"""Adoption fixture."""\n\n'
        "from unigym.pack_api import register_scene_provider\n"
        "from unigym.scenes.scene_ir import Scene\n\n\n"
        "def table(params, context):\n"
        '    """Build the table."""\n'
        '    return Scene(name="table")\n\n\n'
        "def extra(params, context):\n"
        '    """Build the extra scene."""\n'
        '    return Scene(name="extra")\n\n\n'
        "def register(context):\n"
        "    register_scene_provider('acme.table', table)\n"
        "    register_scene_provider('acme.extra', extra)\n"
    )
    root = _write_pack(
        tmp_path / "adopted",
        _document(
            "adopted",
            [{"id": "acme.table", "kind": "scene", "entrypoint": "adopted_pack:table"}],
            entrypoint="adopted_pack:register",
        ),
        {"adopted_pack": source},
    )

    activated = discover([root]).activate("adopted")

    assert _status_by_id(activated) == {"acme.table": ItemStatus.ADOPTED}
    assert scene_provider_origin("acme.table").kind == "manifest"
    assert "acme.extra" in registered_scene_providers()
    assert "acme.extra" not in {item.id for item in activated.items}


def test_failed_activation_is_not_cached_and_retry_is_clean(tmp_path: Path) -> None:
    """A failure restores registries, caches nothing and permits a clean retry."""
    source = (
        '"""Fail-once fixture."""\n\n'
        "_first = True\n\n\n"
        "from unigym.pack_api import register_scene_provider\n"
        "from unigym.scenes.scene_ir import Scene\n\n\n"
        "def table(params, context):\n"
        '    """Build the table."""\n'
        '    return Scene(name="table")\n\n\n'
        "def register(context):\n"
        "    global _first\n"
        "    register_scene_provider('acme.retry', table)\n"
        "    if _first:\n"
        "        _first = False\n"
        '        raise RuntimeError("first attempt")\n'
    )
    root = _write_pack(
        tmp_path / "retry_pack",
        _document("retry_pack", [], entrypoint="retry_pack_module:register"),
        {"retry_pack_module": source},
    )
    loader = discover([root])

    with pytest.raises(PackError, match="first attempt"):
        loader.activate("retry_pack")

    assert loader.is_activated("retry_pack") is False
    assert registered_scene_providers() == []

    activated = loader.activate("retry_pack")

    assert loader.is_activated("retry_pack") is True
    assert activated is loader.activate("retry_pack")
    assert registered_scene_providers() == ["acme.retry"]
