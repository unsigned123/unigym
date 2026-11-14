"""Contract for ``kind: scene`` activation, provenance and provider params.

A pack activation imports and registers manifest ``kind: scene`` items, records
exact origins, applies the manifest parameter schema, adopts a matching pack-code
mirror without drift, and rolls the scene-provider, adapter and robot registries
back totally on any failure, including registrations a scene module performs at
import time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from unigym.adapters import (
    ADAPTER_REGISTRY,
    adapter_origin,
    restore_adapter_registry,
    snapshot_adapter_registry,
)
from unigym.pack_api import (
    ActivationHandle,
    PackCatalog,
    PackManifestV1,
    PackRuntime,
    ProvidedItem,
    discover,
)
from unigym.packs import PackError
from unigym.robots import (
    ROBOT_REGISTRY,
    restore_robot_registry,
    robot_origin,
    snapshot_robot_registry,
)
from unigym.scenes.provider_params import ProviderParamSpec, SceneProviderParamError
from unigym.scenes.providers import (
    ProviderOrigin,
    build_scene_from_provider,
    register_scene_provider,
    registered_scene_providers,
    restore_scene_providers,
    scene_provider_origin,
    scene_provider_providers,
    snapshot_scene_providers,
)
from unigym.scenes.scene_ir import Scene

PACKS_ROOT = Path(__file__).resolve().parents[1] / "_support" / "packs"
CODE_ROOT = PACKS_ROOT / "acme_code"
CODE_PYTHON_ROOT = (CODE_ROOT / "python").resolve()


@pytest.fixture(autouse=True)
def _isolate_scene_provider_state():
    """Keep fixture imports, ``sys.path`` and all provider registries isolated."""
    path_snapshot = list(sys.path)
    scene_snapshot = snapshot_scene_providers()
    adapter_snapshot = snapshot_adapter_registry()
    robots_snapshot = snapshot_robot_registry()
    _drop_modules()
    yield
    sys.path[:] = path_snapshot
    restore_scene_providers(scene_snapshot)
    restore_adapter_registry(adapter_snapshot)
    restore_robot_registry(robots_snapshot)
    _drop_modules()


def _drop_modules() -> None:
    for name in [
        name
        for name in sys.modules
        if name == "acme_pack" or name.startswith("acme_pack.") or name.startswith("scene_")
    ]:
        del sys.modules[name]


def _write_code_pack(
    root: Path,
    document: dict[str, object],
    modules: dict[str, str],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    for module_name, source in modules.items():
        package = root / "python" / module_name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(source, encoding="utf-8")


def _manifest(
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


_PROVIDER_SOURCE = (
    '"""Scene provider activation fixture."""\n\n'
    "from __future__ import annotations\n\n"
    "from unigym.scenes.scene_ir import Scene\n\n\n"
    "def table_top(params, context):\n"
    '    """Build the tabletop scene."""\n'
    '    return Scene(name="tabletop")\n'
)


def test_fixture_scene_provider_activation_records_exact_origin_and_schema() -> None:
    """The manifest item produces one exact origin and its normalized schema."""
    runtime = discover([CODE_ROOT])

    handle = runtime.activate("acme_code")

    assert handle == ActivationHandle(
        pack="acme_code",
        entrypoint="acme_pack:register",
        scene_providers=("acme_code.table_top",),
    )
    assert str(CODE_PYTHON_ROOT) in sys.path
    infos = scene_provider_providers(kind="scene")
    info = next(record for record in infos if record.name == "acme_code.table_top")
    expected = ProviderOrigin(
        kind="manifest",
        pack="acme_code",
        pack_version="1.0.0",
        root=str(CODE_ROOT.resolve()),
        discovered_by="explicit-root",
        face="providers",
        item_id="acme_code.table_top",
        entrypoint="acme_pack:table_top",
        activation_entrypoint="acme_pack:register",
    )
    assert info.origin == expected
    assert scene_provider_origin("acme_code.table_top") == expected
    assert info.params_schema == {
        "style": ProviderParamSpec(type="int", required=False, default=1),
        "seed": ProviderParamSpec(type="int", required=True),
    }

    scene = build_scene_from_provider("acme_code.table_top", {"seed": 2})
    assert scene.name == "tabletop"


def test_manifest_scene_provider_schema_is_enforced_before_dispatch() -> None:
    """The manifest schema rejects unknown, missing and mistyped values."""
    runtime = discover([CODE_ROOT])
    runtime.activate("acme_code")

    with pytest.raises(SceneProviderParamError, match="unknown parameter"):
        build_scene_from_provider("acme_code.table_top", {"seed": 1, "extra": 2})
    with pytest.raises(SceneProviderParamError, match="missing required parameter"):
        build_scene_from_provider("acme_code.table_top", {})
    with pytest.raises(SceneProviderParamError, match="must be int"):
        build_scene_from_provider("acme_code.table_top", {"seed": True})


def test_pack_code_only_scene_provider_reports_pack_code_origin(tmp_path: Path) -> None:
    """A provider registered by the entrypoint without an item is pack-code."""
    source = _PROVIDER_SOURCE + (
        "\n\nfrom unigym.pack_api import register_scene_provider\n\n\n"
        "def register(context):\n"
        "    register_scene_provider('acme.dynamic', table_top)\n"
    )
    root = tmp_path / "scene_pack_code"
    _write_code_pack(
        root,
        _manifest("scene_pack_code", [], entrypoint="scene_code_module:register"),
        {"scene_code_module": source},
    )
    runtime = discover([root])

    handle = runtime.activate("scene_pack_code")

    assert handle.scene_providers == ()
    assert "acme.dynamic" in registered_scene_providers()
    assert scene_provider_origin("acme.dynamic") == ProviderOrigin(
        kind="pack-code",
        pack="scene_pack_code",
        pack_version="1.0.0",
        root=str(root.resolve()),
        discovered_by="explicit-root",
        item_id=None,
        activation_entrypoint="scene_code_module:register",
    )


def test_manifest_scene_provider_name_must_be_free(tmp_path: Path) -> None:
    """A pre-existing registration is never silently replaced."""

    def existing(params, context):
        """Build the existing provider scene."""
        return Scene(name="existing")

    register_scene_provider("acme.table_top", existing)
    root = tmp_path / "scene_pack_duplicate"
    _write_code_pack(
        root,
        _manifest(
            "scene_pack_duplicate",
            [{"id": "acme.table_top", "kind": "scene", "entrypoint": "scene_dup_module:table_top"}],
        ),
        {"scene_dup_module": _PROVIDER_SOURCE},
    )
    runtime = discover([root])

    with pytest.raises(PackError, match="must be free"):
        runtime.activate("scene_pack_duplicate")

    assert registered_scene_providers() == ["acme.table_top"]
    assert scene_provider_origin("acme.table_top") == ProviderOrigin.unknown()


def test_scene_provider_items_are_validated_before_any_registration(tmp_path: Path) -> None:
    """A later bad item leaves no earlier scene provider registered."""
    root = tmp_path / "scene_pack_validate"
    _write_code_pack(
        root,
        _manifest(
            "scene_pack_validate",
            [
                {
                    "id": "acme.good",
                    "kind": "scene",
                    "entrypoint": "scene_validate_module:table_top",
                },
                {"id": "acme.bad", "kind": "scene", "entrypoint": "scene_validate_module:missing"},
            ],
        ),
        {"scene_validate_module": _PROVIDER_SOURCE},
    )
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate("scene_pack_validate")

    assert "acme.bad" in str(info.value)
    assert registered_scene_providers() == []


_SIDE_EFFECT_PROVIDER_SOURCE = (
    '"""Scene-only pack whose import registers an adapter and a robot."""\n\n'
    "from __future__ import annotations\n\n"
    "from unigym.adapters import Adapter\n"
    "from unigym.pack_api import register_adapter, register_robot\n"
    "from unigym.scenes.scene_ir import Scene\n\n\n"
    "class ImportedAdapter(Adapter):\n"
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
    "def make_imported_robot():\n"
    "    raise AssertionError('registration only; the factory is never called')\n\n\n"
    "register_adapter('acme.imported_adapter', ImportedAdapter)\n"
    "register_robot('acme.imported_robot', make_imported_robot)\n\n\n"
    "def table_top(params, context):\n"
    '    """Build the tabletop scene."""\n'
    '    return Scene(name="tabletop")\n'
)


def test_scene_only_pack_import_side_effects_are_rolled_back(tmp_path: Path) -> None:
    """A failed scene-only activation removes import-time adapter/robot names."""
    root = tmp_path / "scene_side_effect"
    _write_code_pack(
        root,
        _manifest(
            "scene_side_effect",
            [
                {
                    "id": "acme.table_top",
                    "kind": "scene",
                    "entrypoint": "scene_side_effect_module:table_top",
                },
                {
                    "id": "acme.bad",
                    "kind": "scene",
                    "entrypoint": "scene_side_effect_module:missing",
                },
            ],
        ),
        {"scene_side_effect_module": _SIDE_EFFECT_PROVIDER_SOURCE},
    )
    runtime = discover([root])
    adapters_before = set(ADAPTER_REGISTRY)
    robots_before = set(ROBOT_REGISTRY)

    with pytest.raises(PackError) as info:
        runtime.activate("scene_side_effect")

    assert "acme.bad" in str(info.value)
    assert set(ADAPTER_REGISTRY) == adapters_before
    assert "acme.imported_adapter" not in ADAPTER_REGISTRY
    assert set(ROBOT_REGISTRY) == robots_before
    assert "acme.imported_robot" not in ROBOT_REGISTRY
    assert registered_scene_providers() == []
    with pytest.raises(ValueError, match="acme.imported_adapter"):
        adapter_origin("acme.imported_adapter")
    with pytest.raises(AttributeError, match="unknown robot"):
        robot_origin("acme.imported_robot")


def test_scene_provider_target_requires_a_docstring(tmp_path: Path) -> None:
    """The provider factory contract is checked before registration."""
    source = (
        '"""Undocumented provider fixture."""\n\n'
        "from unigym.scenes.scene_ir import Scene\n\n\n"
        "def table_top(params, context):\n"
        '    return Scene(name="tabletop")\n'
    )
    root = tmp_path / "scene_pack_docstring"
    _write_code_pack(
        root,
        _manifest(
            "scene_pack_docstring",
            [{"id": "acme.table_top", "kind": "scene", "entrypoint": "scene_doc_module:table_top"}],
        ),
        {"scene_doc_module": source},
    )
    runtime = discover([root])

    with pytest.raises(PackError, match="docstring"):
        runtime.activate("scene_pack_docstring")

    assert registered_scene_providers() == []


def test_matching_pack_code_mirror_is_adopted_with_manifest_provenance(
    tmp_path: Path,
) -> None:
    """An entrypoint registration with the same schema is adopted, not duplicated."""
    source = _PROVIDER_SOURCE + (
        "\n\nfrom unigym.pack_api import register_scene_provider\n\n\n"
        "def register(context):\n"
        "    register_scene_provider(\n"
        "        'acme.table_top',\n"
        "        table_top,\n"
        "        params={'style': (int, 1)},\n"
        "    )\n"
    )
    root = tmp_path / "scene_pack_mirror"
    _write_code_pack(
        root,
        _manifest(
            "scene_pack_mirror",
            [
                {
                    "id": "acme.table_top",
                    "kind": "scene",
                    "entrypoint": "scene_mirror_module:table_top",
                    "params": {"style": {"type": "int", "default": 1}},
                }
            ],
            entrypoint="scene_mirror_module:register",
        ),
        {"scene_mirror_module": source},
    )
    runtime = discover([root])

    handle = runtime.activate("scene_pack_mirror")

    assert handle.scene_providers == ("acme.table_top",)
    origin = scene_provider_origin("acme.table_top")
    assert origin.kind == "manifest"
    assert origin.item_id == "acme.table_top"
    assert build_scene_from_provider("acme.table_top", {}).name == "tabletop"


def test_manifest_and_pack_code_schema_drift_fails_activation(tmp_path: Path) -> None:
    """A differing manifest/code mirror is an activation error and rolls back."""
    source = _PROVIDER_SOURCE + (
        "\n\nfrom unigym.pack_api import register_scene_provider\n\n\n"
        "def register(context):\n"
        "    register_scene_provider(\n"
        "        'acme.table_top',\n"
        "        table_top,\n"
        "        params={'style': (int, 2)},\n"
        "    )\n"
    )
    root = tmp_path / "scene_pack_drift"
    _write_code_pack(
        root,
        _manifest(
            "scene_pack_drift",
            [
                {
                    "id": "acme.table_top",
                    "kind": "scene",
                    "entrypoint": "scene_drift_module:table_top",
                    "params": {"style": {"type": "int", "default": 1}},
                }
            ],
            entrypoint="scene_drift_module:register",
        ),
        {"scene_drift_module": source},
    )
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate("scene_pack_drift")

    message = str(info.value)
    assert "acme.table_top" in message
    assert "differing keys" in message
    assert "style" in message
    assert registered_scene_providers() == []


def test_manifest_schema_without_a_pack_code_mirror_fails_activation(tmp_path: Path) -> None:
    """A manifest-only schema is drift when pack code registered the same name."""
    source = _PROVIDER_SOURCE + (
        "\n\nfrom unigym.pack_api import register_scene_provider\n\n\n"
        "def register(context):\n"
        "    register_scene_provider('acme.table_top', table_top)\n"
    )
    root = tmp_path / "scene_pack_manifest_only"
    _write_code_pack(
        root,
        _manifest(
            "scene_pack_manifest_only",
            [
                {
                    "id": "acme.table_top",
                    "kind": "scene",
                    "entrypoint": "scene_manifest_only_module:table_top",
                    "params": {"style": {"type": "int", "default": 1}},
                }
            ],
            entrypoint="scene_manifest_only_module:register",
        ),
        {"scene_manifest_only_module": source},
    )
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate("scene_pack_manifest_only")

    message = str(info.value)
    assert "acme.table_top" in message
    assert "differing keys" in message
    assert "declare the schema on both" in message
    assert registered_scene_providers() == []


def test_pack_code_schema_without_a_manifest_mirror_fails_activation(tmp_path: Path) -> None:
    """A pack-code-only schema is drift when the manifest item declares none."""
    source = _PROVIDER_SOURCE + (
        "\n\nfrom unigym.pack_api import register_scene_provider\n\n\n"
        "def register(context):\n"
        "    register_scene_provider(\n"
        "        'acme.table_top',\n"
        "        table_top,\n"
        "        params={'style': (int, 1)},\n"
        "    )\n"
    )
    root = tmp_path / "scene_pack_code_only"
    _write_code_pack(
        root,
        _manifest(
            "scene_pack_code_only",
            [
                {
                    "id": "acme.table_top",
                    "kind": "scene",
                    "entrypoint": "scene_code_only_module:table_top",
                }
            ],
            entrypoint="scene_code_only_module:register",
        ),
        {"scene_code_only_module": source},
    )
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate("scene_pack_code_only")

    message = str(info.value)
    assert "acme.table_top" in message
    assert "differing keys" in message
    assert "declare the schema on both" in message
    assert registered_scene_providers() == []


def test_hand_built_scene_item_with_a_resource_is_rejected(tmp_path: Path) -> None:
    """Activation re-checks that a kind: scene item is entrypoint-backed."""
    root = tmp_path / "scene_resource"
    (root / "python").mkdir(parents=True)
    (root / "value.yaml").write_text("kind: scene\n", encoding="utf-8")
    manifest = PackManifestV1(
        schema="unigym.pack/1",
        name="scene_resource",
        version="1.0.0",
        api=1,
        trust="code",
        entrypoint=None,
        requires={},
        provides={
            "providers": (
                ProvidedItem(
                    id="acme.table_top",
                    resource="value.yaml",
                    entrypoint="scene_resource_module:table_top",
                    kind="scene",
                ),
            )
        },
        root=root.resolve(),
    )
    runtime = PackRuntime(
        catalog=PackCatalog((manifest,)),
        root_by_pack={"scene_resource": root.resolve()},
    )

    with pytest.raises(PackError, match="must not declare"):
        runtime.activate("scene_resource")

    assert registered_scene_providers() == []


def test_entrypoint_registration_under_a_second_name_is_rejected(tmp_path: Path) -> None:
    """One factory owns exactly one scene-provider name."""
    source = _PROVIDER_SOURCE + (
        "\n\nfrom unigym.pack_api import register_scene_provider\n\n\n"
        "def register(context):\n"
        "    register_scene_provider('acme.dynamic', table_top)\n"
    )
    root = tmp_path / "scene_pack_alias"
    _write_code_pack(
        root,
        _manifest(
            "scene_pack_alias",
            [
                {
                    "id": "acme.table_top",
                    "kind": "scene",
                    "entrypoint": "scene_alias_module:table_top",
                }
            ],
            entrypoint="scene_alias_module:register",
        ),
        {"scene_alias_module": source},
    )
    runtime = discover([root])

    with pytest.raises(PackError, match="one factory owns exactly one name"):
        runtime.activate("scene_pack_alias")

    assert registered_scene_providers() == []
