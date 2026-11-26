"""Contract for the live per-pack registry projection.

``runtime_registrations`` is a read-only view over the existing registries, not
a second pack model: it keeps only records a given pack owns through a
``manifest`` or ``pack-code`` origin and drops core, inline, unknown and other
packs' entries.  The projection order is stable regardless of registration
order.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from types import MappingProxyType

import pytest

from unigym.actuator_law import (
    ActuatorLawDescriptor,
    register_actuator_law,
    restore_actuator_laws,
    snapshot_actuator_laws,
)
from unigym.adapters import (
    Adapter,
    register_adapter,
    restore_adapter_registry,
    snapshot_adapter_registry,
)
from unigym.initializers.base import (
    register as register_initializer,
    restore_initializers,
    snapshot_initializers,
)
from unigym.packs.inventory import runtime_registrations
from unigym.packs.schema import Pack, load_manifest_v1
from unigym.props.templates import (
    register_prop_template,
    restore_prop_templates,
    snapshot_prop_templates,
)
from unigym.provenance import ProviderOrigin
from unigym.robots import (
    register_robot,
    restore_robot_registry,
    snapshot_robot_registry,
)
from unigym.robots.base import RobotProfile
from unigym.scenes.pipeline import (
    SceneContentFiller,
    register_scene_content_filler,
    restore_scene_content_fillers,
    snapshot_scene_content_fillers,
)
from unigym.scenes.providers import (
    register_inline_scene_provider,
    register_scene_provider,
    restore_scene_providers,
    snapshot_scene_providers,
)
from unigym.scenes.scene_ir import Scene
from unigym.terminations import (
    register_termination,
    restore_termination_registry,
    snapshot_termination_registry,
)
from unigym.terminations.base import Termination

_REGISTRY_ORDER = (
    "adapters",
    "robots",
    "scene_providers",
    "scene_content_fillers",
    "prop_templates",
    "terminations",
    "initializers",
    "actuator_laws",
)


@pytest.fixture(autouse=True)
def _isolate_registries() -> Iterator[None]:
    """Restore every registry, ``sys.path`` and imported module around each test."""
    path_snapshot = list(sys.path)
    modules_snapshot = set(sys.modules)
    robots = snapshot_robot_registry()
    adapters = snapshot_adapter_registry()
    scene_providers = snapshot_scene_providers()
    content_fillers = snapshot_scene_content_fillers()
    templates = snapshot_prop_templates()
    terminations = snapshot_termination_registry()
    laws = snapshot_actuator_laws()
    initializers = snapshot_initializers()
    yield
    restore_robot_registry(robots)
    restore_adapter_registry(adapters)
    restore_scene_providers(scene_providers)
    restore_scene_content_fillers(content_fillers)
    restore_prop_templates(templates)
    restore_termination_registry(terminations)
    restore_actuator_laws(laws)
    restore_initializers(initializers)
    sys.path[:] = path_snapshot
    for name in set(sys.modules) - modules_snapshot:
        del sys.modules[name]


def _manifest_pack(tmp_path: Path, name: str) -> Pack:
    """Load a real empty-content pack document from ``tmp_path``."""
    root = tmp_path / name
    root.mkdir()
    (root / "unigym.pack.yaml").write_text(
        "schema: unigym.pack/1\n"
        f"name: {name}\n"
        "version: 1.0.0\n"
        "api: 1\n"
        "trust: content\n"
        "provides: {}\n",
        encoding="utf-8",
    )
    return load_manifest_v1(root)


def _origin(kind: str, pack: str, *, item_id: str | None = None) -> ProviderOrigin:
    """Build one attributable origin record."""
    return ProviderOrigin(kind=kind, pack=pack, item_id=item_id)


def _filler(**_kwargs) -> SceneContentFiller:
    """Return a fixture content filler (never called)."""
    raise AssertionError("fixture content filler is never called")


def _other_filler(**_kwargs) -> SceneContentFiller:
    """Return another fixture content filler (never called)."""
    raise AssertionError("fixture content filler is never called")


def _loose_filler(**_kwargs) -> SceneContentFiller:
    """Return an origin-less fixture content filler (never called)."""
    raise AssertionError("fixture content filler is never called")


def _robot_factory() -> RobotProfile:
    """Return a fixture robot profile (never called)."""
    raise AssertionError("fixture robot factory is never called")


def _scene_factory(params, context) -> Scene:
    """Return a fixture scene provider target."""
    raise AssertionError("fixture scene provider is never called")


def _inline_scene_factory(params, context) -> Scene:
    """Return a fixture inline scene provider target."""
    raise AssertionError("fixture inline scene provider is never called")


class _InventoryAdapter(Adapter):
    """Minimal adapter class; the registry only checks subclass identity."""


class _InventoryTermination(Termination):
    def check(self, ctx):
        return None


class _InventoryInitializer:
    name = "inventory_initializer"


def _prop_builder(prop_name, params):
    """Build one fixture prop."""
    return {"name": prop_name}


def test_runtime_registrations_filters_owner_and_kind(tmp_path: Path) -> None:
    """Manifest and pack-code records stay; core, inline, unknown and others drop."""
    pack = _manifest_pack(tmp_path, "inventory_pack")
    other = _manifest_pack(tmp_path, "other_pack")

    register_robot(
        "inventory_pack.robot",
        _robot_factory,
        origin=_origin("manifest", pack.name, item_id="inventory_pack.robot"),
    )
    register_scene_content_filler(
        "inventory_filler",
        _filler,
        origin=_origin("pack-code", pack.name),
    )
    register_scene_content_filler(
        "other_filler",
        _other_filler,
        origin=_origin("manifest", other.name, item_id="other.filler"),
    )
    register_scene_content_filler("loose_filler", _loose_filler)
    register_termination(
        "inventory_core_term",
        _InventoryTermination,
        origin=ProviderOrigin(kind="core", pack=pack.name),
    )
    register_scene_provider(
        "inventory_pack.inline_scene",
        _scene_factory,
        origin=_origin("inline", pack.name),
    )
    register_inline_scene_provider("inventory_inline", _inline_scene_factory)

    projected = runtime_registrations(pack)

    assert tuple(projected) == ("robots", "scene_content_fillers")
    assert [ref.name for ref in projected["robots"]] == ["inventory_pack.robot"]
    assert [ref.name for ref in projected["scene_content_fillers"]] == ["inventory_filler"]
    assert all(ref.origin.pack == pack.name for records in projected.values() for ref in records)
    assert "terminations" not in projected
    assert "scene_providers" not in projected
    assert all(
        ref.name != "inventory_core_term" for records in projected.values() for ref in records
    )
    assert isinstance(projected, MappingProxyType)


def test_runtime_registrations_follow_the_canonical_registry_order(tmp_path: Path) -> None:
    """Every registry is queried and rendered in the fixed canonical order."""
    pack = _manifest_pack(tmp_path, "inventory_order")
    origin = _origin("pack-code", pack.name)

    register_adapter("inventory.policy", _InventoryAdapter, origin=origin)
    register_robot("inventory.robot", _robot_factory, origin=origin)
    register_scene_provider("inventory.scene", _scene_factory, origin=origin)
    register_scene_content_filler("inventory_filler", _filler, origin=origin)
    register_prop_template("inventory_prop", params={"size": {"type": "string"}}, origin=origin)(
        _prop_builder
    )
    register_termination("inventory_term", _InventoryTermination, origin=origin)
    register_initializer(_InventoryInitializer, origin=origin)
    register_actuator_law(ActuatorLawDescriptor(name="inventory_law"), origin=origin)

    projected = runtime_registrations(pack)

    assert tuple(projected) == _REGISTRY_ORDER
    for records in projected.values():
        assert len(records) == 1
        assert records[0].origin == origin
