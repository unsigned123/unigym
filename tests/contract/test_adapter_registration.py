"""Contract for dynamic and manifest-declared adapter registration.

Adapter classes live in packs.  A pack entrypoint may call
``register_adapter`` directly, or a manifest ``providers`` item may declare
``kind: adapter`` plus an ``entrypoint``; runtime activation imports that class
and registers it under the item id.  Both forms share one process-global
registry, so duplicate names fail loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from unigym.adapters import ADAPTER_REGISTRY, Adapter, build_adapter
from unigym.pack_api import discover
from unigym.packs import PackError
from unigym.robots import ROBOT_REGISTRY, restore_robots, snapshot_robots


@pytest.fixture(autouse=True)
def _isolate_adapter_registration():
    """Restore the adapter and robot registries, ``sys.path`` and pack modules."""
    path_snapshot = list(sys.path)
    modules_snapshot = set(sys.modules)
    registry_snapshot = set(ADAPTER_REGISTRY)
    robots_snapshot = snapshot_robots()
    yield
    for name in set(ADAPTER_REGISTRY) - registry_snapshot:
        del ADAPTER_REGISTRY[name]
    restore_robots(robots_snapshot)
    sys.path[:] = path_snapshot
    for name in set(sys.modules) - modules_snapshot:
        del sys.modules[name]


def _manifest(
    name: str,
    *,
    trust: str = "code",
    entrypoint: str | None = None,
    providers: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": "unigym.pack/1",
        "name": name,
        "version": "1.0.0",
        "api": 1,
        "trust": trust,
        "provides": {},
    }
    if entrypoint is not None:
        document["entrypoint"] = entrypoint
    if providers:
        document["provides"] = {"providers": providers}
    return document


def _adapter_source(class_name: str) -> str:
    """A minimal concrete Adapter subclass, optionally importing a sibling."""
    return (
        "from __future__ import annotations\n\n"
        "from unigym.adapters import Adapter\n\n\n"
        f"class {class_name}(Adapter):\n"
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


def _write_pack(
    root: Path,
    document: dict[str, object],
    modules: dict[str, str] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(document, sort_keys=False)
    (root / "unigym.pack.yaml").write_text(text, encoding="utf-8")
    for module_name, source in (modules or {}).items():
        package = root / "python" / module_name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(source, encoding="utf-8")


def test_dynamic_code_pack_registers_adapter(tmp_path: Path) -> None:
    source = _adapter_source("DynamicAdapter") + (
        "\n\nfrom unigym.pack_api import register_adapter\n\n\n"
        "def register(context):\n"
        "    register_adapter('acme.dynamic', DynamicAdapter)\n"
    )
    root = tmp_path / "adapter_dynamic"
    _write_pack(
        root,
        _manifest("adapter_dynamic", entrypoint="adapter_dynamic_pack:register"),
        {"adapter_dynamic_pack": source},
    )

    runtime = discover([root])
    runtime.activate_all()
    runtime.activate_all()

    registered = ADAPTER_REGISTRY["acme.dynamic"]
    assert issubclass(registered, Adapter)
    adapter = build_adapter({"name": "acme.dynamic"}, num_envs=1, device="cpu")
    assert isinstance(adapter, Adapter)


def test_manifest_adapter_provider_registers_under_item_id(tmp_path: Path) -> None:
    root = tmp_path / "adapter_static"
    _write_pack(
        root,
        _manifest(
            "adapter_static",
            providers=[
                {
                    "id": "acme.static",
                    "kind": "adapter",
                    "entrypoint": "adapter_static_pack:StaticAdapter",
                }
            ],
        ),
        {"adapter_static_pack": _adapter_source("StaticAdapter")},
    )

    runtime = discover([root])
    handles = runtime.activate_all()
    runtime.activate_all()

    assert handles[0].adapters == ("acme.static",)
    registered = ADAPTER_REGISTRY["acme.static"]
    assert issubclass(registered, Adapter)
    adapter = build_adapter({"name": "acme.static"}, num_envs=1, device="cpu")
    assert isinstance(adapter, Adapter)


def test_manifest_adapter_target_must_subclass_adapter(tmp_path: Path) -> None:
    root = tmp_path / "adapter_bad"
    _write_pack(
        root,
        _manifest(
            "adapter_bad",
            providers=[
                {
                    "id": "adapter_bad.not_adapter",
                    "kind": "adapter",
                    "entrypoint": "adapter_bad_pack:NotAdapter",
                }
            ],
        ),
        {"adapter_bad_pack": "class NotAdapter:\n    pass\n"},
    )

    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate_all()

    message = str(info.value)
    assert "adapter_bad" in message
    assert "adapter_bad.not_adapter" in message
    assert "adapter_bad_pack:NotAdapter" in message


def test_manifest_adapter_item_requires_an_entrypoint(tmp_path: Path) -> None:
    root = tmp_path / "adapter_resource"
    _write_pack(
        root,
        _manifest(
            "adapter_resource",
            trust="content",
            providers=[
                {
                    "id": "adapter_resource.item",
                    "kind": "adapter",
                    "resource": "adapter.yaml",
                }
            ],
        ),
    )
    (root / "adapter.yaml").write_text("kind: adapter\n", encoding="utf-8")

    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate_all()

    message = str(info.value)
    assert "adapter_resource" in message
    assert "adapter_resource.item" in message
    assert "no entrypoint" in message


def test_duplicate_dynamic_and_manifest_registration_fails_loudly(tmp_path: Path) -> None:
    source = _adapter_source("DuplicateAdapter") + (
        "\n\nfrom unigym.pack_api import register_adapter\n\n\n"
        "def register(context):\n"
        "    register_adapter('acme.duplicate', DuplicateAdapter)\n"
    )
    root = tmp_path / "adapter_duplicate"
    _write_pack(
        root,
        _manifest(
            "adapter_duplicate",
            entrypoint="adapter_duplicate_pack:register",
            providers=[
                {
                    "id": "acme.duplicate",
                    "kind": "adapter",
                    "entrypoint": "adapter_duplicate_pack:DuplicateAdapter",
                }
            ],
        ),
        {"adapter_duplicate_pack": source},
    )

    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate_all()

    message = str(info.value)
    assert "adapter_duplicate" in message
    assert "acme.duplicate" in message
    assert "already registered" in message


def test_manifest_adapter_partial_failure_leaves_no_registration(tmp_path: Path) -> None:
    source = _adapter_source("GoodAdapter") + "\n\ndef bad_target():\n    return None\n"
    root = tmp_path / "adapter_partial"
    _write_pack(
        root,
        _manifest(
            "adapter_partial",
            providers=[
                {
                    "id": "acme.good",
                    "kind": "adapter",
                    "entrypoint": "adapter_partial_pack:GoodAdapter",
                },
                {
                    "id": "acme.bad",
                    "kind": "adapter",
                    "entrypoint": "adapter_partial_pack:bad_target",
                },
            ],
        ),
        {"adapter_partial_pack": source},
    )

    runtime = discover([root])

    with pytest.raises(PackError) as first:
        runtime.activate_all()
    assert "acme.bad" in str(first.value)
    assert "acme.good" not in ADAPTER_REGISTRY

    with pytest.raises(PackError) as second:
        runtime.activate_all()
    assert "acme.bad" in str(second.value)
    assert "already registered" not in str(second.value)
    assert "acme.good" not in ADAPTER_REGISTRY


def test_manifest_provider_bad_name_fails_at_discovery(tmp_path: Path) -> None:
    """The adapter/provider grammar is enforced at manifest load, before registration."""
    root = tmp_path / "adapter_rollback"
    _write_pack(
        root,
        _manifest(
            "adapter_rollback",
            providers=[
                {
                    "id": "acme.good",
                    "kind": "adapter",
                    "entrypoint": "adapter_rollback_pack:GoodAdapter",
                },
                {
                    "id": "acme-bad",
                    "kind": "adapter",
                    "entrypoint": "adapter_rollback_pack:GoodAdapter",
                },
            ],
        ),
        {"adapter_rollback_pack": _adapter_source("GoodAdapter")},
    )

    with pytest.raises(PackError) as info:
        discover([root])

    assert "acme-bad" in str(info.value)
    assert "adapter/robot grammar" in str(info.value)
    assert "acme.good" not in ADAPTER_REGISTRY
    assert "acme-bad" not in ADAPTER_REGISTRY


def test_mixed_entrypoint_and_provider_failure_rolls_back_adapters(tmp_path: Path) -> None:
    """A later bad provider removes adapters registered by the pack entrypoint."""
    source = _adapter_source("EntrypointAdapter") + (
        "\n\nfrom unigym.pack_api import register_adapter\n\n\n"
        "def register(context):\n"
        "    register_adapter('acme.dynamic', EntrypointAdapter)\n"
    )
    root = tmp_path / "adapter_mixed"
    _write_pack(
        root,
        _manifest(
            "adapter_mixed",
            entrypoint="adapter_mixed_pack:register",
            providers=[
                {
                    "id": "acme.bad",
                    "kind": "adapter",
                    "entrypoint": "adapter_mixed_pack:not_an_adapter",
                }
            ],
        ),
        {"adapter_mixed_pack": source + "\n\ndef not_an_adapter():\n    return None\n"},
    )

    runtime = discover([root])

    with pytest.raises(PackError) as first:
        runtime.activate_all()
    assert "acme.bad" in str(first.value)
    assert "acme.dynamic" not in ADAPTER_REGISTRY

    with pytest.raises(PackError) as second:
        runtime.activate_all()
    assert "acme.bad" in str(second.value)
    assert "already registered" not in str(second.value)
    assert "acme.dynamic" not in ADAPTER_REGISTRY


_ROBOT_BODY = (
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

_ROBOT_SOURCE = (
    '"""Robot provider fixture."""\n\n' "from __future__ import annotations\n\n" + _ROBOT_BODY
)


def test_entrypoint_robot_rolls_back_when_adapter_provider_fails(tmp_path: Path) -> None:
    """A cross-kind failure removes robots registered by the pack entrypoint."""
    source = _ROBOT_SOURCE + (
        "\n\nfrom unigym.pack_api import register_robot\n\n\n"
        "def register(context):\n"
        "    register_robot('acme.robot', make_arm)\n\n\n"
        "def not_an_adapter():\n"
        "    return None\n"
    )
    root = tmp_path / "adapter_cross_robot"
    _write_pack(
        root,
        _manifest(
            "adapter_cross_robot",
            entrypoint="adapter_cross_robot_pack:register",
            providers=[
                {
                    "id": "acme.bad",
                    "kind": "adapter",
                    "entrypoint": "adapter_cross_robot_pack:not_an_adapter",
                }
            ],
        ),
        {"adapter_cross_robot_pack": source},
    )

    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate_all()

    assert "acme.bad" in str(info.value)
    assert "acme.robot" not in ROBOT_REGISTRY
    assert "acme.bad" not in ADAPTER_REGISTRY


def test_adapter_provider_rolls_back_when_robot_provider_fails(tmp_path: Path) -> None:
    """A later robot provider failure removes an earlier adapter registration."""
    source = (
        _adapter_source("GoodAdapter") + "\n\n" + _ROBOT_BODY + ("\nnot_a_factory = object()\n")
    )
    root = tmp_path / "adapter_cross_kind"
    _write_pack(
        root,
        _manifest(
            "adapter_cross_kind",
            providers=[
                {
                    "id": "acme.good",
                    "kind": "adapter",
                    "entrypoint": "adapter_cross_kind_pack:GoodAdapter",
                },
                {
                    "id": "acme.robot",
                    "kind": "robot",
                    "entrypoint": "adapter_cross_kind_pack:not_a_factory",
                },
            ],
        ),
        {"adapter_cross_kind_pack": source},
    )

    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate_all()

    assert "acme.robot" in str(info.value)
    assert "acme.good" not in ADAPTER_REGISTRY
    assert "acme.robot" not in ROBOT_REGISTRY
