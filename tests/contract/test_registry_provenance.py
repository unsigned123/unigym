"""Contract for registry provenance: origins, window attribution and rollback.

Robot and adapter factories/classes keep their authoritative name -> value
registries; origins live in side tables.  Manifest providers write exact
``kind="manifest"`` origins, pack-code registrations during an activation
window get ``kind="pack-code"`` with ``item_id=None``, and a failed activation
restores values and origins exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from unigym.adapters import (
    ADAPTER_REGISTRY,
    Adapter,
    adapter_origin,
    adapter_providers,
    register_adapter,
    restore_adapter_registry,
    snapshot_adapter_registry,
)
from unigym.pack_api import PackRuntime, discover
from unigym.packs import PackError
from unigym.props.templates import (
    prop_template_origin,
    prop_template_providers,
    register_prop_template,
    restore_prop_templates,
    snapshot_prop_templates,
)
from unigym.provenance import ProviderInfo, ProviderOrigin
from unigym.robots import (
    ROBOT_REGISTRY,
    register_robot,
    restore_robot_registry,
    robot_origin,
    robot_providers,
    snapshot_robot_registry,
)
from unigym.scenes.pipeline import (
    register_scene_content_filler,
    restore_scene_content_fillers,
    scene_content_filler_origin,
    scene_content_filler_providers,
    snapshot_scene_content_fillers,
)
from unigym.terminations import (
    TERMINATION_REGISTRY,
    register_termination,
    restore_termination_registry,
    snapshot_termination_registry,
    termination_origin,
    termination_providers,
)


@pytest.fixture(autouse=True)
def _isolate_registries() -> None:
    """Restore every registry, ``sys.path`` and pack modules around each test."""
    path_snapshot = list(sys.path)
    modules_snapshot = set(sys.modules)
    robots_snapshot = snapshot_robot_registry()
    adapters_snapshot = snapshot_adapter_registry()
    content_snapshot = snapshot_scene_content_fillers()
    templates_snapshot = snapshot_prop_templates()
    terminations_snapshot = snapshot_termination_registry()
    yield
    restore_robot_registry(robots_snapshot)
    restore_adapter_registry(adapters_snapshot)
    restore_scene_content_fillers(content_snapshot)
    restore_prop_templates(templates_snapshot)
    restore_termination_registry(terminations_snapshot)
    sys.path[:] = path_snapshot
    for name in set(sys.modules) - modules_snapshot:
        del sys.modules[name]


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


def _write_pack(
    root: Path,
    document: dict[str, object],
    modules: dict[str, str] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    for module_name, source in (modules or {}).items():
        package = root / "python" / module_name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(source, encoding="utf-8")


_ADAPTER_SOURCE = (
    '"""Provenance adapter fixture."""\n\n'
    "from __future__ import annotations\n\n"
    "from unigym.adapters import Adapter\n\n\n"
    "class AcmeAdapter(Adapter):\n"
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
    "        return {}\n"
)

_ROBOT_SOURCE = (
    '"""Provenance robot fixture."""\n\n'
    "from unigym.robots.base import RobotProfile\n\n\n"
    "def make_arm():\n"
    "    return RobotProfile(\n"
    "        name='acme.robot',\n"
    "        joint_sets={'body': ('shoulder',)},\n"
    "        default_joint_pos={'shoulder': 0.0},\n"
    "        assets=None,\n"
    "        action_faces={},\n"
    "        base_body=None,\n"
    "        foot_bodies=(),\n"
    "        body_sets={},\n"
    "        body_bindings={},\n"
    "    )\n"
)

_ROBOT_DOCUMENT = (
    "schema: unigym.robot/1\n"
    "name: acme.arm\n"
    "joint_sets:\n"
    "  body: [shoulder]\n"
    "default_joint_pos:\n"
    "  shoulder: 0.0\n"
)


class _TestAdapter(Adapter):
    def __init__(self, num_envs=1, device="cpu", **kwargs):
        pass

    def get_obs_requests(self):
        return []

    @property
    def obs_dim(self):
        return 0

    @property
    def control_dt(self):
        return 0.02

    def compute_obs(self, obs_cache, cmd, env):
        return None

    def act(self, obs):
        return None

    def postprocess_action(self, raw):
        return {}


class _OtherTestAdapter(_TestAdapter):
    pass


def _factory() -> object:
    return object()


def _other_factory() -> object:
    return object()


def test_manual_adapter_is_a_core_origin() -> None:
    """The built-in ``manual`` entry reports ``kind="core"``."""
    assert adapter_origin("manual") == ProviderOrigin(kind="core")
    manual = next(info for info in adapter_providers() if info.name == "manual")
    assert manual.kind == "adapter"
    assert manual.origin == ProviderOrigin(kind="core")
    assert manual.active is True


def test_core_termination_seeds_are_core_origins() -> None:
    """The built-in terminations report ``kind="core"``."""
    assert set(TERMINATION_REGISTRY) >= {"fall", "base_contact", "joint_limits"}
    for name in ("fall", "base_contact", "joint_limits"):
        assert termination_origin(name) == ProviderOrigin(kind="core")
        info = next(info for info in termination_providers() if info.name == name)
        assert info.kind == "termination"
        assert info.origin == ProviderOrigin(kind="core")
    assert termination_providers(kind="robot") == ()


def test_live_registry_origins_round_trip_and_unknown_fallback() -> None:
    """Content fillers, prop templates and terminations follow the shared origin shape."""

    def filler(**_kwargs: object) -> object:
        return object()

    def builder(prop_name: str, params: object) -> dict[str, object]:
        """Build one fixture prop."""
        return {"name": prop_name}

    class AcmeTermination:
        def __init__(self, params: object, name: str | None = None) -> None:
            pass

    filler_origin = ProviderOrigin(kind="pack-code", pack="acme", item_id=None)
    template_origin = ProviderOrigin(kind="manifest", pack="acme", item_id="acme_prop")
    termination_origin_value = ProviderOrigin(kind="pack-code", pack="acme", item_id=None)

    register_scene_content_filler("acme_filler", filler, origin=filler_origin)
    register_scene_content_filler("acme_loose_filler", filler)
    register_prop_template(
        "acme_prop",
        params={"size": {"type": "string"}},
        origin=template_origin,
    )(builder)
    register_termination("acme_term", AcmeTermination, origin=termination_origin_value)
    register_termination("acme_loose_term", AcmeTermination)

    assert scene_content_filler_origin("acme_filler") == filler_origin
    assert scene_content_filler_origin("acme_loose_filler") == ProviderOrigin.unknown()
    assert prop_template_origin("acme_prop") == template_origin
    assert termination_origin("acme_term") == termination_origin_value
    assert termination_origin("acme_loose_term") == ProviderOrigin.unknown()

    assert [info.name for info in scene_content_filler_providers()] == [
        "acme_filler",
        "acme_loose_filler",
    ]
    assert [info.name for info in prop_template_providers()] == ["acme_prop"]
    assert [info.name for info in termination_providers("termination")] == [
        "acme_loose_term",
        "acme_term",
        "base_contact",
        "fall",
        "joint_limits",
    ]
    assert scene_content_filler_providers("robot") == ()
    assert prop_template_providers("robot") == ()
    assert all(isinstance(info, ProviderInfo) for info in termination_providers())


def test_robot_origin_round_trip_and_unknown_fallback() -> None:
    """A supplied origin round-trips; a loose registration reports unknown."""
    origin = ProviderOrigin(
        kind="manifest",
        pack="acme",
        pack_version="1.0.0",
        root="/packs/acme",
        discovered_by="explicit-root",
        face="providers",
        item_id="acme.robot",
        entrypoint="acme_robot:make_arm",
    )
    register_robot("acme.robot", _factory, origin=origin)
    register_robot("acme.loose", _other_factory)

    assert robot_origin("acme.robot") == origin
    assert robot_origin("acme.loose") == ProviderOrigin.unknown()
    assert [info.name for info in robot_providers(kind="robot")] == ["acme.loose", "acme.robot"]
    assert robot_providers(kind="adapter") == ()


def test_unknown_robot_origin_raises_the_registry_error() -> None:
    """An unknown name keeps the existing unknown-robot ``AttributeError``."""
    with pytest.raises(AttributeError, match="unknown robot 'acme.missing'"):
        robot_origin("acme.missing")


def test_adapter_origin_round_trip_and_unknown_fallback() -> None:
    """A supplied origin round-trips; a loose registration reports unknown."""
    origin = ProviderOrigin(kind="pack-code", pack="acme", item_id=None)
    register_adapter("acme.policy", _TestAdapter, origin=origin)
    register_adapter("acme.loose", _OtherTestAdapter)

    assert adapter_origin("acme.policy") == origin
    assert adapter_origin("acme.loose") == ProviderOrigin.unknown()
    assert [info.name for info in adapter_providers(kind="adapter")] == [
        "acme.loose",
        "acme.policy",
        "manual",
    ]
    assert adapter_providers(kind="robot") == ()


def test_unknown_adapter_origin_lists_the_registered_names() -> None:
    """An unknown adapter name is a teaching ``ValueError``."""
    with pytest.raises(ValueError, match="acme.missing"):
        adapter_origin("acme.missing")


def test_manifest_providers_write_exact_origins(tmp_path: Path) -> None:
    """A resource robot and an entrypoint adapter keep exact manifest origins."""
    root = tmp_path / "acme_manifest"
    robots_dir = root / "robots"
    robots_dir.mkdir(parents=True)
    (robots_dir / "acme_arm.yaml").write_text(_ROBOT_DOCUMENT, encoding="utf-8")
    _write_pack(
        root,
        _manifest(
            "acme_manifest",
            [
                {"id": "acme.arm", "kind": "robot", "resource": "robots/acme_arm.yaml"},
                {
                    "id": "acme.policy",
                    "kind": "adapter",
                    "entrypoint": "acme_manifest_pack:AcmeAdapter",
                },
            ],
        ),
        {"acme_manifest_pack": _ADAPTER_SOURCE},
    )
    runtime = discover([root])

    runtime.activate("acme_manifest")

    assert robot_origin("acme.arm") == ProviderOrigin(
        kind="manifest",
        pack="acme_manifest",
        pack_version="1.0.0",
        root=str(root.resolve()),
        discovered_by="explicit-root",
        face="providers",
        item_id="acme.arm",
        resource="robots/acme_arm.yaml",
        entrypoint=None,
        activation_entrypoint=None,
    )
    assert adapter_origin("acme.policy") == ProviderOrigin(
        kind="manifest",
        pack="acme_manifest",
        pack_version="1.0.0",
        root=str(root.resolve()),
        discovered_by="explicit-root",
        face="providers",
        item_id="acme.policy",
        resource=None,
        entrypoint="acme_manifest_pack:AcmeAdapter",
        activation_entrypoint=None,
    )


def test_pack_code_registration_gets_window_attribution(tmp_path: Path) -> None:
    """Names registered by the entrypoint are pack-code with ``item_id=None``."""
    source = (
        _ADAPTER_SOURCE
        + "\n\n"
        + _ROBOT_SOURCE
        + (
            "\n\nfrom unigym.pack_api import register_adapter, register_robot\n\n\n"
            "def register(context):\n"
            "    register_adapter('acme.policy', AcmeAdapter)\n"
            "    register_robot('acme.robot', make_arm)\n"
        )
    )
    root = tmp_path / "acme_window"
    _write_pack(
        root,
        _manifest("acme_window", [], entrypoint="acme_window_pack:register"),
        {"acme_window_pack": source},
    )
    runtime = discover([root])

    runtime.activate("acme_window")

    expected = ProviderOrigin(
        kind="pack-code",
        pack="acme_window",
        pack_version="1.0.0",
        root=str(root.resolve()),
        discovered_by="explicit-root",
        item_id=None,
        activation_entrypoint="acme_window_pack:register",
    )
    assert robot_origin("acme.robot") == expected
    assert adapter_origin("acme.policy") == expected


def test_registration_outside_an_activation_window_is_unknown() -> None:
    """A direct registration has no attributable pack and reports unknown."""
    register_robot("acme.robot", _factory)
    register_adapter("acme.policy", _TestAdapter)
    assert robot_origin("acme.robot").kind == "unknown"
    assert adapter_origin("acme.policy").kind == "unknown"


_WINDOW_REGISTRY_SOURCE = (
    '"""Window-attribution fixture for filler/template/termination registries."""\n\n'
    "from __future__ import annotations\n\n"
    "from unigym.pack_api import (\n"
    "    register_prop_template,\n"
    "    register_scene_content_filler,\n"
    "    register_termination,\n"
    ")\n\n\n"
    "def acme_filler(**_kwargs):\n"
    "    return object()\n\n\n"
    "def acme_prop(prop_name, params):\n"
    '    """Build a fixture prop."""\n'
    "    return {'name': prop_name}\n\n\n"
    "class AcmeTermination:\n"
    "    def __init__(self, params, name=None):\n"
    "        pass\n\n\n"
    "def register(context):\n"
    "    register_scene_content_filler('acme_filler', acme_filler)\n"
    "    register_prop_template('acme_prop', params={'size': {'type': 'string'}})(acme_prop)\n"
    "    register_termination('acme_term', AcmeTermination)\n"
)


def test_pack_code_registries_get_window_attribution(tmp_path: Path) -> None:
    """Filler, template and termination registrations get ``kind="pack-code"``."""
    root = tmp_path / "acme_window_registries"
    _write_pack(
        root,
        _manifest("acme_window_registries", [], entrypoint="acme_window_pack:register"),
        {"acme_window_pack": _WINDOW_REGISTRY_SOURCE},
    )
    runtime = discover([root])

    runtime.activate("acme_window_registries")

    expected = ProviderOrigin(
        kind="pack-code",
        pack="acme_window_registries",
        pack_version="1.0.0",
        root=str(root.resolve()),
        discovered_by="explicit-root",
        item_id=None,
        activation_entrypoint="acme_window_pack:register",
    )
    assert scene_content_filler_origin("acme_filler") == expected
    assert prop_template_origin("acme_prop") == expected
    assert termination_origin("acme_term") == expected


def test_failed_activation_restores_filler_template_and_termination_registries(
    tmp_path: Path,
) -> None:
    """A failed activation rolls the three pack-code registries back exactly."""
    content_before = snapshot_scene_content_fillers()
    templates_before = snapshot_prop_templates()
    terminations_before = snapshot_termination_registry()
    root = tmp_path / "acme_window_rollback"
    _write_pack(
        root,
        _manifest(
            "acme_window_rollback",
            [{"id": "acme.broken", "kind": "robot", "entrypoint": "acme_window_pack:missing"}],
            entrypoint="acme_window_pack:register",
        ),
        {"acme_window_pack": _WINDOW_REGISTRY_SOURCE},
    )
    runtime = discover([root])

    with pytest.raises(PackError, match="acme.broken"):
        runtime.activate("acme_window_rollback")

    assert snapshot_scene_content_fillers() == content_before
    assert snapshot_prop_templates() == templates_before
    assert snapshot_termination_registry() == terminations_before
    with pytest.raises(ValueError, match="acme_filler"):
        scene_content_filler_origin("acme_filler")
    with pytest.raises(ValueError, match="acme_prop"):
        prop_template_origin("acme_prop")
    with pytest.raises(ValueError, match="acme_term"):
        termination_origin("acme_term")


def test_failed_activation_restores_values_and_origins(tmp_path: Path) -> None:
    """A failed activation restores both registries and their origins exactly."""
    robots_before = snapshot_robot_registry()
    adapters_before = snapshot_adapter_registry()
    source = (
        _ADAPTER_SOURCE
        + "\n\n"
        + _ROBOT_SOURCE
        + (
            "\n\nfrom unigym.pack_api import register_adapter, register_robot\n\n\n"
            "def register(context):\n"
            "    register_adapter('acme.policy', AcmeAdapter)\n"
            "    register_robot('acme.robot', make_arm)\n"
        )
    )
    root = tmp_path / "acme_rollback"
    _write_pack(
        root,
        _manifest(
            "acme_rollback",
            [{"id": "acme.broken", "kind": "robot", "entrypoint": "acme_rollback_pack:missing"}],
            entrypoint="acme_rollback_pack:register",
        ),
        {"acme_rollback_pack": source},
    )
    runtime = discover([root])

    with pytest.raises(PackError, match="acme.broken"):
        runtime.activate("acme_rollback")

    assert snapshot_robot_registry() == robots_before
    assert snapshot_adapter_registry() == adapters_before
    assert "acme.robot" not in ROBOT_REGISTRY
    assert "acme.policy" not in ADAPTER_REGISTRY
    with pytest.raises(AttributeError, match="unknown robot"):
        robot_origin("acme.robot")
    with pytest.raises(ValueError, match="acme.policy"):
        adapter_origin("acme.policy")


def test_entry_point_location_seam_is_reported_on_live_origins(tmp_path: Path) -> None:
    """An entry-point root marks live origins with ``location_imported=True``."""
    root = tmp_path / "acme_seam"
    robots_dir = root / "robots"
    robots_dir.mkdir(parents=True)
    (robots_dir / "acme_arm.yaml").write_text(_ROBOT_DOCUMENT, encoding="utf-8")
    _write_pack(
        root,
        _manifest(
            "acme_seam",
            [{"id": "acme.arm", "kind": "robot", "resource": "robots/acme_arm.yaml"}],
        ),
    )
    base = discover([root])
    runtime = PackRuntime(
        catalog=base.catalog,
        root_by_pack=base.root_by_pack,
        origins={"acme_seam": "entry-point"},
    )

    runtime.activate("acme_seam")

    origin = robot_origin("acme.arm")
    assert origin.discovered_by == "entry-point"
    assert origin.location_imported is True
