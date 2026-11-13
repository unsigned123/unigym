"""Contract for the explicit static-root pack runtime.

Discovery reads fixture manifests only; resource lookup stays inside each
pack root; activation is explicit normal Python and cached per runtime.
Negative fixtures are copied into ``tmp_path`` so the shared support packs
stay untouched.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from unigym.pack_api import ActivationHandle, PackContext, PackRuntime, discover
from unigym.packs import PackError

PACKS_ROOT = Path(__file__).resolve().parents[1] / "_support" / "packs"
CONTENT_ROOT = PACKS_ROOT / "acme_content"
CODE_ROOT = PACKS_ROOT / "acme_code"
CODE_PYTHON_ROOT = (CODE_ROOT / "python").resolve()


def _drop_pack_modules() -> None:
    """Remove fixture pack modules so every test observes a fresh import."""
    for name in [
        name for name in sys.modules if name == "acme_pack" or name.startswith("acme_pack.")
    ]:
        del sys.modules[name]


@pytest.fixture(autouse=True)
def _isolate_pack_modules() -> None:
    """Keep fixture import state and sys.path from leaking between tests."""
    path_snapshot = list(sys.path)
    _drop_pack_modules()
    yield
    sys.path[:] = path_snapshot
    _drop_pack_modules()


@pytest.fixture(autouse=True)
def _isolate_robot_registry():
    """Restore the process-global external robot registry after every test."""
    from unigym.robots import restore_robots, snapshot_robots

    snapshot = snapshot_robots()
    yield
    restore_robots(snapshot)


def _spy_entrypoint(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Import the real fixture module and spy on its context-aware entrypoint.

    The fixture declares a one-argument ``register(context)`` entrypoint.  The
    spy records the activation call and the received context while proving the
    runtime imports the entrypoint from the fixture's ``python/`` boundary.
    """
    monkeypatch.syspath_prepend(str(CODE_PYTHON_ROOT))
    module = importlib.import_module("acme_pack")
    module_file = Path(module.__file__).resolve()
    assert module_file.is_relative_to(CODE_PYTHON_ROOT)
    calls: list[object] = []

    def record(context: object) -> None:
        calls.append(context)

    monkeypatch.setattr(module, "register", record)
    return calls


def test_discover_content_and_code_roots_in_order() -> None:
    runtime = discover([CONTENT_ROOT, CODE_ROOT])

    assert isinstance(runtime, PackRuntime)
    assert runtime.catalog.packs() == ("acme_content", "acme_code")
    assert [entry.id for entry in runtime.catalog.entries()] == [
        "acme_content.demo",
        "acme_content.tabletop",
        "acme_content.cup",
        "acme_code.dynamic_assets",
    ]
    assert runtime.root_by_pack["acme_content"] == CONTENT_ROOT.resolve()
    assert runtime.root_by_pack["acme_code"] == CODE_ROOT.resolve()


def test_resource_resolves_under_the_pack_root() -> None:
    runtime = discover([CONTENT_ROOT, CODE_ROOT])

    assert (
        runtime.resource("acme_content", "runs/demo.yaml")
        == (CONTENT_ROOT / "runs" / "demo.yaml").resolve()
    )
    assert runtime.resource("acme_content", "assets/cup/asset.yaml").is_file()


def test_discover_dedupes_canonical_duplicate_roots() -> None:
    runtime = discover([CONTENT_ROOT, CONTENT_ROOT, f"{CONTENT_ROOT}/."])

    assert runtime.catalog.packs() == ("acme_content",)
    assert len(runtime.catalog.manifests) == 1


def test_discover_rejects_duplicate_pack_names(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    shutil.copytree(CONTENT_ROOT, first)
    shutil.copytree(CONTENT_ROOT, second)

    with pytest.raises(PackError, match="duplicate pack name 'acme_content'"):
        discover([first, second])


def test_discover_rejects_missing_root_or_manifest(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(PackError, match="missing pack manifest"):
        discover([empty])
    with pytest.raises(PackError, match="missing pack manifest"):
        discover([tmp_path / "absent"])


@pytest.mark.parametrize(
    ("rel", "reason"),
    [
        ("", "non-empty relative path"),
        ("/etc/passwd", "must be relative to the pack root"),
        ("../outside.yaml", "must not contain '..'"),
        ("missing.yaml", "does not exist"),
    ],
)
def test_resource_rejects_illegal_or_missing_paths(tmp_path: Path, rel: str, reason: str) -> None:
    root = tmp_path / "acme_content"
    shutil.copytree(CONTENT_ROOT, root)
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.resource("acme_content", rel)

    message = str(info.value)
    assert "acme_content" in message
    assert reason in message


def test_resource_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "acme_content"
    shutil.copytree(CONTENT_ROOT, root)
    outside = tmp_path / "outside.yaml"
    outside.write_text("kind: outside\n", encoding="utf-8")
    (root / "escape.yaml").symlink_to(outside)
    runtime = discover([root])

    with pytest.raises(PackError, match="escapes the pack root"):
        runtime.resource("acme_content", "escape.yaml")


def test_discovery_does_not_import_pack_python() -> None:
    assert "acme_pack" not in sys.modules

    discover([CODE_ROOT, CONTENT_ROOT])

    assert "acme_pack" not in sys.modules


def test_discovered_roots_are_fixture_relative() -> None:
    runtime = discover([CONTENT_ROOT, CODE_ROOT])

    for root in runtime.root_by_pack.values():
        assert root.is_relative_to(PACKS_ROOT.resolve())


def test_code_activation_imports_and_calls_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = _spy_entrypoint(monkeypatch)
    runtime = discover([CODE_ROOT])

    handle = runtime.activate("acme_code")

    assert handle == ActivationHandle(pack="acme_code", entrypoint="acme_pack:register")
    assert len(contexts) == 1
    context = contexts[0]
    assert isinstance(context, PackContext)
    assert context.pack == "acme_code"
    assert context.root == CODE_ROOT.resolve()
    assert context.runtime is runtime
    assert str(CODE_PYTHON_ROOT) in sys.path


def test_activation_context_resolves_resources_inside_the_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts = _spy_entrypoint(monkeypatch)
    runtime = discover([CODE_ROOT])

    runtime.activate("acme_code")

    context = contexts[0]
    assert isinstance(context, PackContext)
    assert context.resource("unigym.pack.yaml") == (CODE_ROOT / "unigym.pack.yaml").resolve()
    with pytest.raises(PackError, match="acme_code"):
        context.resource("../outside.yaml")


def test_fixture_code_entrypoint_accepts_a_pack_context() -> None:
    runtime = discover([CODE_ROOT])

    handle = runtime.activate("acme_code")

    assert handle == ActivationHandle(pack="acme_code", entrypoint="acme_pack:register")


def test_content_activation_is_a_noop() -> None:
    runtime = discover([CONTENT_ROOT])

    handle = runtime.activate("acme_content")

    assert handle == ActivationHandle(pack="acme_content", entrypoint=None)
    assert "acme_pack" not in sys.modules


def test_activation_failure_wraps_pack_and_entrypoint(tmp_path: Path) -> None:
    root = tmp_path / "acme_code"
    shutil.copytree(CODE_ROOT, root)
    (root / "python" / "acme_pack" / "__init__.py").write_text(
        '"""Broken fixture entrypoint."""\n\n'
        "from __future__ import annotations\n\n\n"
        "def register(context: object) -> None:\n"
        '    raise RuntimeError("boom")\n',
        encoding="utf-8",
    )
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate("acme_code")

    message = str(info.value)
    assert "acme_code" in message
    assert "acme_pack:register" in message
    assert "boom" in message
    assert isinstance(info.value.__cause__, RuntimeError)


def test_repeated_activation_returns_the_cached_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_entrypoint(monkeypatch)
    runtime = discover([CODE_ROOT])

    first = runtime.activate("acme_code")
    second = runtime.activate("acme_code")

    assert first is second
    assert len(calls) == 1


def test_activate_all_activates_catalog_order_and_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _spy_entrypoint(monkeypatch)
    runtime = discover([CONTENT_ROOT, CODE_ROOT])

    first = runtime.activate_all()
    second = runtime.activate_all()

    assert [handle.pack for handle in first] == ["acme_content", "acme_code"]
    assert first == second
    assert all(earlier is later for earlier, later in zip(first, second, strict=True))
    assert len(calls) == 1


def test_activation_does_not_fall_back_to_a_zero_argument_entrypoint(tmp_path: Path) -> None:
    root = tmp_path / "acme_code"
    shutil.copytree(CODE_ROOT, root)
    (root / "python" / "acme_pack" / "__init__.py").write_text(
        '"""Zero-argument fixture entrypoint."""\n\n'
        "from __future__ import annotations\n\n\n"
        "def register() -> None:\n"
        "    return None\n",
        encoding="utf-8",
    )
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate("acme_code")

    message = str(info.value)
    assert "acme_code" in message
    assert "acme_pack:register" in message
    assert isinstance(info.value.__cause__, TypeError)


def test_runtime_module_has_no_legacy_loader_bindings() -> None:
    import unigym.packs.runtime as runtime_module

    for name in ("MountedPack", "load_pack_code", "mount", "pack_roots"):
        assert name not in runtime_module.__dict__


_ROBOT_SOURCE = (
    '"""Robot provider fixture."""\n\n'
    "from __future__ import annotations\n\n"
    "from unigym.robots.base import RobotProfile\n\n\n"
    "def make_arm():\n"
    "    return RobotProfile(\n"
    "        name='acme.arm',\n"
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

_NOT_A_FACTORY = "\nnot_a_factory = object()\n"


def _write_code_pack(
    root: Path,
    document: dict[str, object],
    modules: dict[str, str],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(document, sort_keys=False)
    (root / "unigym.pack.yaml").write_text(text, encoding="utf-8")
    for module_name, source in modules.items():
        package = root / "python" / module_name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(source, encoding="utf-8")


def _robot_manifest(
    name: str,
    providers: list[dict[str, str]],
    *,
    entrypoint: str | None = None,
    trust: str = "code",
) -> dict[str, object]:
    document: dict[str, object] = {
        "schema": "unigym.pack/1",
        "name": name,
        "version": "1.0.0",
        "api": 1,
        "trust": trust,
        "provides": {"providers": providers},
    }
    if entrypoint is not None:
        document["entrypoint"] = entrypoint
    return document


def test_manifest_robot_provider_registers_a_factory(tmp_path: Path) -> None:
    """A kind: robot item registers its entrypoint factory under the item id."""
    root = tmp_path / "robot_pack"
    _write_code_pack(
        root,
        _robot_manifest(
            "robot_pack",
            [{"id": "acme.arm", "kind": "robot", "entrypoint": "robot_success_module:make_arm"}],
        ),
        {"robot_success_module": _ROBOT_SOURCE},
    )

    runtime = discover([root])
    handle = runtime.activate("robot_pack")

    from unigym.robots import ROBOT_REGISTRY, build_robot_profile

    assert handle.robots == ("acme.arm",)
    assert "acme.arm" in ROBOT_REGISTRY
    assert build_robot_profile({"name": "acme.arm"}).name == "acme.arm"


def test_manifest_robot_item_requires_an_entrypoint(tmp_path: Path) -> None:
    """A resource-only kind: robot item is a teaching PackError."""
    root = tmp_path / "robot_resource"
    _write_code_pack(
        root,
        _robot_manifest(
            "robot_resource",
            [{"id": "acme.arm", "kind": "robot", "resource": "robot.yaml"}],
            trust="content",
        ),
        {},
    )
    (root / "robot.yaml").write_text("kind: robot\n", encoding="utf-8")
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate("robot_resource")

    message = str(info.value)
    assert "robot_resource" in message
    assert "acme.arm" in message
    assert "no entrypoint" in message


def test_manifest_robot_target_must_be_callable(tmp_path: Path) -> None:
    """A profile instance or other non-callable target fails before registration."""
    root = tmp_path / "robot_not_callable"
    _write_code_pack(
        root,
        _robot_manifest(
            "robot_not_callable",
            [
                {
                    "id": "acme.arm",
                    "kind": "robot",
                    "entrypoint": "robot_callable_module:not_a_factory",
                }
            ],
        ),
        {"robot_callable_module": _ROBOT_SOURCE + _NOT_A_FACTORY},
    )
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate("robot_not_callable")

    message = str(info.value)
    assert "acme.arm" in message
    assert "not callable" in message

    from unigym.robots import ROBOT_REGISTRY

    assert "acme.arm" not in ROBOT_REGISTRY


@pytest.mark.parametrize(
    ("item_id", "reason"),
    [
        ("Acme.Arm", "lowercase snake_case"),
        ("g1", "reserved"),
    ],
)
def test_manifest_robot_name_is_validated_before_registration(
    tmp_path: Path, item_id: str, reason: str
) -> None:
    """The item id uses the external grammar and cannot take a built-in name."""
    root = tmp_path / "robot_bad_name"
    _write_code_pack(
        root,
        _robot_manifest(
            "robot_bad_name",
            [{"id": item_id, "kind": "robot", "entrypoint": "robot_name_module:make_arm"}],
        ),
        {"robot_name_module": _ROBOT_SOURCE},
    )
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate("robot_bad_name")

    message = str(info.value)
    assert item_id in message
    assert reason in message

    from unigym.robots import ROBOT_REGISTRY

    assert item_id not in ROBOT_REGISTRY


def test_manifest_robot_provider_rejects_one_factory_under_two_names(tmp_path: Path) -> None:
    """Two provider ids for one factory fail before either name is registered."""
    root = tmp_path / "robot_alias"
    _write_code_pack(
        root,
        _robot_manifest(
            "robot_alias",
            [
                {"id": "acme.arm", "kind": "robot", "entrypoint": "robot_alias_module:make_arm"},
                {"id": "acme.other", "kind": "robot", "entrypoint": "robot_alias_module:make_arm"},
            ],
        ),
        {"robot_alias_module": _ROBOT_SOURCE},
    )
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate("robot_alias")

    message = str(info.value)
    assert "acme.arm" in message
    assert "acme.other" in message
    assert "exactly one name" in message

    from unigym.robots import ROBOT_REGISTRY

    assert "acme.arm" not in ROBOT_REGISTRY
    assert "acme.other" not in ROBOT_REGISTRY


def test_manifest_robot_provider_rejects_an_already_registered_name(tmp_path: Path) -> None:
    """A manifest item cannot replace a pre-existing external robot."""
    from unigym.robots import ROBOT_REGISTRY, register_robot
    from unigym.robots.base import RobotProfile

    def existing() -> RobotProfile:
        return RobotProfile(
            name="acme.arm",
            joint_sets={"body": ("shoulder",)},
            default_joint_pos={"shoulder": 0.0},
            assets=None,
            action_faces={},
            base_body=None,
            foot_bodies=(),
            body_sets={},
            body_bindings={},
        )

    register_robot("acme.arm", existing)
    root = tmp_path / "robot_duplicate"
    _write_code_pack(
        root,
        _robot_manifest(
            "robot_duplicate",
            [{"id": "acme.arm", "kind": "robot", "entrypoint": "robot_duplicate_module:make_arm"}],
        ),
        {"robot_duplicate_module": _ROBOT_SOURCE},
    )
    runtime = discover([root])

    with pytest.raises(PackError) as info:
        runtime.activate("robot_duplicate")

    assert "already registered" in str(info.value)
    assert ROBOT_REGISTRY["acme.arm"] is existing


def test_entrypoint_registered_robot_rolls_back_on_robot_provider_failure(
    tmp_path: Path,
) -> None:
    """A later bad robot provider removes robots the entrypoint registered."""
    source = (
        _ROBOT_SOURCE
        + _NOT_A_FACTORY
        + (
            "\n\nfrom unigym.pack_api import register_robot\n\n\n"
            "def register(context):\n"
            "    register_robot('acme.dynamic', make_arm)\n"
        )
    )
    root = tmp_path / "robot_mixed"
    _write_code_pack(
        root,
        _robot_manifest(
            "robot_mixed",
            [{"id": "acme.arm", "kind": "robot", "entrypoint": "robot_mixed_module:not_a_factory"}],
            entrypoint="robot_mixed_module:register",
        ),
        {"robot_mixed_module": source},
    )
    runtime = discover([root])

    with pytest.raises(PackError) as first:
        runtime.activate("robot_mixed")
    assert "acme.arm" in str(first.value)

    from unigym.robots import ROBOT_REGISTRY

    assert "acme.dynamic" not in ROBOT_REGISTRY
    assert "acme.arm" not in ROBOT_REGISTRY

    with pytest.raises(PackError) as second:
        runtime.activate("robot_mixed")
    assert "acme.arm" in str(second.value)
    assert "already registered" not in str(second.value)


def test_entrypoint_registered_robot_does_not_appear_in_the_handle(tmp_path: Path) -> None:
    """handle.robots records manifest kind: robot ids only, in declaration order."""
    source = _ROBOT_SOURCE + (
        "\n\nfrom unigym.pack_api import register_robot\n"
        "from unigym.robots.base import RobotProfile\n\n\n"
        "def make_dynamic():\n"
        "    return RobotProfile(\n"
        "        name='acme.dynamic',\n"
        "        joint_sets={'body': ('shoulder',)},\n"
        "        default_joint_pos={'shoulder': 0.0},\n"
        "        assets=None,\n"
        "        action_faces={},\n"
        "        base_body=None,\n"
        "        foot_bodies=(),\n"
        "        body_sets={},\n"
        "        body_bindings={},\n"
        "    )\n\n\n"
        "def register(context):\n"
        "    register_robot('acme.dynamic', make_dynamic)\n"
    )
    root = tmp_path / "robot_handle"
    _write_code_pack(
        root,
        _robot_manifest(
            "robot_handle",
            [{"id": "acme.arm", "kind": "robot", "entrypoint": "robot_handle_module:make_arm"}],
            entrypoint="robot_handle_module:register",
        ),
        {"robot_handle_module": source},
    )
    runtime = discover([root])

    handle = runtime.activate("robot_handle")

    assert handle.robots == ("acme.arm",)

    from unigym.robots import ROBOT_REGISTRY

    assert "acme.dynamic" in ROBOT_REGISTRY
    assert "acme.arm" in ROBOT_REGISTRY
