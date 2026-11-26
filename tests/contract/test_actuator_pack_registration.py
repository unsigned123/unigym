"""Pack-registered actuator laws and late-bound implementation factories."""

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
    ActuatorBuildContext,
    restore_actuator_implementations,
    snapshot_actuator_implementations,
)
from unigym.packs.actuators import materialize_actuator_implementations
from unigym.packs import PackError
from unigym.packs.loader import discover
from unigym.packs.schema import Face, ItemStatus, load_manifest_v1
from unigym.registration import (
    RegistrationPhase,
    enter_phase,
    reset_registration,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "packs" / "acme_actuators"


@pytest.fixture(autouse=True)
def _isolate_actuator_registries():
    laws = snapshot_actuator_laws()
    implementations = snapshot_actuator_implementations()
    modules = set(sys.modules)
    reset_registration()
    yield
    restore_actuator_laws(laws)
    restore_actuator_implementations(implementations)
    for name in set(sys.modules) - modules:
        del sys.modules[name]
    reset_registration()


def _write_pack(root: Path, *, law_id: str = "acme.compliant_pd") -> None:
    (root / "actuators").mkdir(parents=True)
    document = {
        "schema": "unigym.pack/1",
        "name": "acme_actuators_copy",
        "version": "1.0.0",
        "api": 1,
        "trust": "content",
        "provides": {
            "providers": [
                {
                    "id": law_id,
                    "kind": "actuator_law",
                    "resource": "actuators/law.yaml",
                }
            ]
        },
    }
    (root / "unigym.pack.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    (root / "actuators" / "law.yaml").write_text(
        "schema: unigym.actuator_law/1\n"
        f"name: {law_id}\n"
        "config_keys: [kp, kd]\n"
        "gains: {allowed: [kp, kd], required: [kp, kd]}\n"
        "drives: [position_pd]\n"
        "backends: [mujoco]\n",
        encoding="utf-8",
    )


def test_manifest_accepts_actuator_law_and_implementation_items() -> None:
    manifest = load_manifest_v1(_FIXTURE)
    law, implementation = manifest.provides["providers"]
    assert law.kind == "actuator_law"
    assert law.resource == "actuators/compliant_pd.law.yaml"
    assert law.entrypoint is None
    assert implementation.kind == "actuator_implementation"
    assert implementation.entrypoint == "acme_actuators:make_mujoco"
    assert dict(implementation.params) == {
        "law": "acme.compliant_pd",
        "backend": "mujoco",
        "kind": "mujoco_ctrl",
    }


def test_manifest_rejects_bad_actuator_item_shapes(tmp_path: Path) -> None:
    def write(items):
        root = tmp_path / f"pack{len(list(tmp_path.iterdir()))}"
        root.mkdir()
        (root / "unigym.pack.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema": "unigym.pack/1",
                    "name": "bad_pack",
                    "version": "1.0.0",
                    "api": 1,
                    "trust": "code",
                    "provides": {"providers": items},
                }
            ),
            encoding="utf-8",
        )
        return root

    with pytest.raises(PackError, match="cannot declare 'params:'"):
        load_manifest_v1(
            write(
                [
                    {
                        "id": "acme.law",
                        "kind": "actuator_law",
                        "entrypoint": "a:b",
                        "params": {},
                    }
                ]
            )
        )
    with pytest.raises(PackError, match="entrypoint-backed only"):
        root = write(
            [
                {
                    "id": "acme.impl",
                    "kind": "actuator_implementation",
                    "resource": "x.yaml",
                    "params": {"law": "a", "backend": "mujoco"},
                }
            ]
        )
        (root / "x.yaml").write_text("schema: x\n", encoding="utf-8")
        load_manifest_v1(root)
    with pytest.raises(PackError, match="unknown keys"):
        load_manifest_v1(
            write(
                [
                    {
                        "id": "acme.impl",
                        "kind": "actuator_implementation",
                        "entrypoint": "a:b",
                        "params": {"law": "a", "backend": "mujoco", "bogus": 1},
                    }
                ]
            )
        )


def test_activation_registers_law_and_leaves_implementation_inert() -> None:
    enter_phase(RegistrationPhase.ACTIVATE)
    runtime = discover([_FIXTURE])
    activated = runtime.activate("acme_actuators")

    laws = tuple(
        item.id
        for item in activated.items_for(
            face=Face.PROVIDERS, kind="actuator_law", status=ItemStatus.REGISTERED
        )
    )
    implementations = tuple(
        item.id
        for item in activated.items_for(
            face=Face.PROVIDERS, kind="actuator_implementation", status=ItemStatus.INERT
        )
    )
    assert laws == ("acme.compliant_pd",)
    assert implementations == ("acme.compliant_pd.mujoco",)
    descriptor = ACTUATOR_LAW_REGISTRY["acme.compliant_pd"]
    assert descriptor.requires_implementation
    assert descriptor.gain_keys == frozenset({"kp", "kd"})
    assert ("acme.compliant_pd", "mujoco") not in ACTUATOR_IMPLEMENTATION_REGISTRY
    assert "acme_actuators" not in sys.modules


def test_materialize_imports_and_registers_at_backend_build() -> None:
    enter_phase(RegistrationPhase.ACTIVATE)
    runtime = discover([_FIXTURE])
    runtime.activate("acme_actuators")
    enter_phase(RegistrationPhase.BACKEND_BUILD)

    materialized = materialize_actuator_implementations(runtime, "mujoco")

    assert materialized == ("acme.compliant_pd.mujoco",)
    implementation = ACTUATOR_IMPLEMENTATION_REGISTRY[("acme.compliant_pd", "mujoco")]
    assert implementation.kind == "mujoco_ctrl"
    assert "acme_actuators" in sys.modules
    context = ActuatorBuildContext(
        law=ACTUATOR_LAW_REGISTRY["acme.compliant_pd"],
        backend="mujoco",
        joint_names=("j1",),
        params={"kp": 1.0, "kd": 2.0},
    )
    result = implementation.factory(context)
    assert result == {
        "law": "acme.compliant_pd",
        "backend": "mujoco",
        "params": {"kp": 1.0, "kd": 2.0},
    }


def test_materialization_selects_only_the_requested_backend() -> None:
    enter_phase(RegistrationPhase.ACTIVATE)
    runtime = discover([_FIXTURE])
    runtime.activate("acme_actuators")
    enter_phase(RegistrationPhase.ASSEMBLE)

    assert materialize_actuator_implementations(runtime, "isaaclab2") == ()
    assert ACTUATOR_IMPLEMENTATION_REGISTRY.get(("acme.compliant_pd", "mujoco")) is None


def test_implementation_must_own_a_law_declared_by_the_same_pack(tmp_path: Path) -> None:
    root = tmp_path / "foreign"
    root.mkdir()
    (root / "actuators").mkdir()
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "unigym.pack/1",
                "name": "foreign_impl",
                "version": "1.0.0",
                "api": 1,
                "trust": "code",
                "provides": {
                    "providers": [
                        {
                            "id": "foreign.impl",
                            "kind": "actuator_implementation",
                            "entrypoint": "missing_module:make",
                            "params": {"law": "acme.compliant_pd", "backend": "mujoco"},
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    enter_phase(RegistrationPhase.BACKEND_BUILD)
    runtime = discover([root])
    with pytest.raises(PackError, match="does not own"):
        materialize_actuator_implementations(runtime, "mujoco")


def test_materialization_requires_the_backend_phase() -> None:
    enter_phase(RegistrationPhase.ACTIVATE)
    runtime = discover([_FIXTURE])
    runtime.activate("acme_actuators")
    enter_phase(RegistrationPhase.ASSEMBLE)

    with pytest.raises(Exception, match="BACKEND_BUILD"):
        materialize_actuator_implementations(runtime, "mujoco")


def test_materialization_requires_activation() -> None:
    enter_phase(RegistrationPhase.BACKEND_BUILD)
    runtime = discover([_FIXTURE])

    with pytest.raises(Exception, match="unknown actuator law"):
        materialize_actuator_implementations(runtime, "mujoco")


def test_activate_leaves_implementations_module_unimported_in_a_fresh_process(
    tmp_path: Path,
) -> None:
    import subprocess
    import sys

    root = tmp_path / "empty_pack"
    root.mkdir()
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "unigym.pack/1",
                "name": "empty_pack",
                "version": "1.0.0",
                "api": 1,
                "trust": "content",
                "provides": {},
            }
        ),
        encoding="utf-8",
    )
    code = (
        "import sys\n"
        "import unigym.packs.loader as runtime_mod\n"
        "from unigym.packs.actuators import materialize_actuator_implementations\n"
        "from unigym.registration import RegistrationPhase, enter_phase\n"
        f"runtime = runtime_mod.discover([{str(root)!r}])\n"
        "runtime.activate('empty_pack')\n"
        "assert 'unigym.actuators.implementations' not in sys.modules, "
        "'implementations imported during ACTIVATE'\n"
        "enter_phase(RegistrationPhase.BACKEND_BUILD)\n"
        "materialize_actuator_implementations(runtime, 'mujoco')\n"
        "assert 'unigym.actuators.implementations' in sys.modules, "
        "'implementations not imported at materialization'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_manifest_rejects_non_canonical_implementation_backend(tmp_path: Path) -> None:
    root = tmp_path / "bad_backend"
    root.mkdir()
    for backend in ("MuJoCo", "mujoco ", "isaaclab"):
        (root / "unigym.pack.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema": "unigym.pack/1",
                    "name": "bad_backend",
                    "version": "1.0.0",
                    "api": 1,
                    "trust": "code",
                    "provides": {
                        "providers": [
                            {
                                "id": "bad.impl",
                                "kind": "actuator_implementation",
                                "entrypoint": "a:b",
                                "params": {"law": "acme.law", "backend": backend},
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(PackError, match="not canonical"):
            load_manifest_v1(root)


def test_rollback_restores_law_and_initializer_origins(
    tmp_path: Path,
) -> None:
    from unigym.actuator_law import actuator_law_origin
    from unigym.initializers.base import INITIALIZER_REGISTRY, initializer_origin
    from unigym.packs import PackError

    root = tmp_path / "rollback_pack"
    package = root / "python" / "rollback_pack"
    package.mkdir(parents=True)
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "unigym.pack/1",
                "name": "rollback_pack",
                "version": "1.0.0",
                "api": 1,
                "trust": "code",
                "entrypoint": "rollback_pack:register",
                "provides": {},
            }
        ),
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        "from unigym.pack_api import ActuatorLawDescriptor, register_actuator_law\n"
        "from unigym.initializers.base import InitializerBase, register as register_init\n"
        "class RollbackInit(InitializerBase):\n"
        "    name = 'rollback.init'\n\n"
        "def register(context):\n"
        "    register_init(RollbackInit)\n\n"
        "    register_actuator_law(ActuatorLawDescriptor(name='rollback.law'))\n"
        "    raise RuntimeError('boom')\n",
        encoding="utf-8",
    )
    enter_phase(RegistrationPhase.ACTIVATE)
    runtime = discover([root])
    with pytest.raises(PackError, match="boom"):
        runtime.activate("rollback_pack")

    assert "rollback.law" not in ACTUATOR_LAW_REGISTRY
    assert "rollback.init" not in INITIALIZER_REGISTRY
    assert actuator_law_origin("rollback.law").kind == "unknown"
    assert initializer_origin("rollback.init").kind == "unknown"


def test_materialization_rejects_non_canonical_backend_without_schema() -> None:
    from types import SimpleNamespace

    from unigym.packs.actuators import materialize_actuator_implementations
    from unigym.packs.schema import EntrypointLocator, Face, PackItem
    from unigym.provenance import ProviderOrigin

    item = PackItem(
        pack="bad_pack",
        face=Face.PROVIDERS,
        kind="actuator_implementation",
        name="bad.impl",
        locator=EntrypointLocator(module="a", attribute="b", raw="a:b"),
        params={"law": "acme.law", "backend": "MuJoCo"},
        origin=ProviderOrigin(
            kind="manifest",
            pack="bad_pack",
            pack_version="1.0.0",
            face="providers",
            item_id="bad.impl",
            entrypoint="a:b",
        ),
    )

    class _Manifest(SimpleNamespace):
        def items_for(self, *, face=None, kind=None, status=None):
            if kind == "actuator_implementation":
                return (item,)
            return ()

    manifest = _Manifest(
        name="bad_pack",
        version="1.0.0",
        root=Path("."),
        entrypoint=None,
    )

    class _PackSet:
        def packs(self):
            return (manifest,)

    runtime = SimpleNamespace(pack_set=_PackSet(), origins={})
    enter_phase(RegistrationPhase.BACKEND_BUILD)
    with pytest.raises(PackError, match="not canonical"):
        materialize_actuator_implementations(runtime, "mujoco")
