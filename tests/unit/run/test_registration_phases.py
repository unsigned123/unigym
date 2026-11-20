"""Run construction enters the registration phases in the documented order."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from unigym.actuators.ir import ActuationGroup
from unigym.assets.asset_ref import AssetRef
from unigym.pack_api import discover
from unigym.registration import (
    RegistrationPhase,
    RegistrationSealedError,
    current_phase,
    is_sealed,
    reset_registration,
)
from unigym.robots.base import BodyBinding, RobotProfile
from unigym.run.api import Run
from unigym.runtime.actions import OrderedActionSpec, PolicyBinding
from unigym.world import World

TINY_ARM = Path(__file__).resolve().parents[2] / "_support" / "data" / "tiny_arm.xml"
JOINTS = ("shoulder", "elbow")
ACTION_SPEC = OrderedActionSpec.from_mapping(
    [
        {
            "name": "tiny",
            "articulation": "tiny",
            "joints": list(JOINTS),
            "target": "position",
            "role": "body",
        }
    ]
)
POLICIES = (PolicyBinding.from_mapping({"adapter": "manual", "groups": [["tiny", "tiny"]]}),)


@pytest.fixture(autouse=True)
def _isolate_registries():
    """Restore the law/implementation registries after each phase test."""
    from unigym.actuator_law import restore_actuator_laws, snapshot_actuator_laws
    from unigym.actuators.implementations import (
        restore_actuator_implementations,
        snapshot_actuator_implementations,
    )

    laws = snapshot_actuator_laws()
    implementations = snapshot_actuator_implementations()
    reset_registration()
    yield
    restore_actuator_laws(laws)
    restore_actuator_implementations(implementations)
    reset_registration()


def _profile() -> RobotProfile:
    return RobotProfile(
        name="tiny",
        joint_sets={"body": JOINTS},
        default_joint_pos={"shoulder": 0.1, "elbow": -0.2},
        assets=AssetRef(mjcf=str(TINY_ARM)),
        extra_action_sets={},
        base_body="base",
        foot_bodies=(),
        body_sets={},
        body_bindings={"base": BodyBinding("base", {"mjcf": "base"})},
        actuation=(
            ActuationGroup(
                name="body",
                law="standard_pd",
                joint_set="body",
                gains={"kp": 20.0, "kd": 1.0},
            ),
        ),
    )


def _floor_world(tmp_path: Path) -> Path:
    path = tmp_path / "world.xml"
    path.write_text(
        '<mujoco model="floor"><worldbody>'
        '<geom name="world_floor" type="plane" size="2 2 0.1"/>'
        "</worldbody></mujoco>",
        encoding="utf-8",
    )
    return path


def test_run_lifecycle_enters_spec_assemble_backend_build_then_seals(
    tmp_path: Path,
) -> None:
    reset_registration()
    run = Run(
        world=World.base(mjcf=_floor_world(tmp_path)),
        robot=_profile(),
        backend="mujoco",
        observations={"robot": ["tiny"]},
        action_spec=ACTION_SPEC,
        policies=POLICIES,
        steps=1,
        inspect=True,
    )
    try:
        assert current_phase() is None
        run.spec
        assert current_phase() is RegistrationPhase.SPEC
        assert not is_sealed()

        run.build(inspect=True)
        assert current_phase() is RegistrationPhase.RUNTIME
        assert is_sealed()
        from unigym.adapters import register_adapter

        with pytest.raises(RegistrationSealedError, match="RUNTIME"):
            register_adapter("late.policy", object)
    finally:
        run.close()


_ADAPTER_SOURCE = (
    "from unigym.adapters import Adapter\n"
    "from unigym.pack_api import register_adapter\n\n"
    "class RunAdapter(Adapter):\n"
    "    def __init__(self, num_envs=1, device='cpu', **kwargs):\n"
    "        pass\n"
    "    @property\n"
    "    def obs_dim(self):\n"
    "        return 0\n"
    "    @property\n"
    "    def control_dt(self):\n"
    "        return 0.02\n"
    "    def compute_obs(self, obs, env):\n"
    "        return None\n"
    "    def act(self, obs):\n"
    "        return None\n"
    "    def postprocess_action(self, raw):\n"
    "        return {}\n\n"
    "def register(context):\n"
    "    register_adapter('acme.run', RunAdapter)\n"
)


def _write_adapter_pack(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "unigym.pack/1",
                "name": "explicit_runtime_pack",
                "version": "1.0.0",
                "api": 1,
                "trust": "code",
                "entrypoint": "explicit_runtime_fixture:register",
                "provides": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    package = root / "python" / "explicit_runtime_fixture"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(_ADAPTER_SOURCE, encoding="utf-8")


def _adapter_config() -> dict:
    return {
        "run_name": "explicit_runtime",
        "seed": 7,
        "env": {"backend": "isaaclab2", "num_envs": 1},
        "collect": {"steps": 1},
        "articulations": [
            {"name": "arm", "joints": [{"name": "shoulder"}, {"name": "elbow"}]},
        ],
        "action_spec": [
            {
                "name": "arm",
                "articulation": "arm",
                "joints": ["shoulder", "elbow"],
                "target": "position",
                "role": "body",
            },
        ],
        "policies": [{"adapter": "acme.run", "groups": [["arm", "arm"]]}],
        "adapter": {"name": "acme.run"},
        "observations": {"robot": ["arm"]},
    }


def test_injected_pack_runtime_is_activated_when_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unigym.adapters import (
        ADAPTER_REGISTRY,
        restore_adapter_registry,
        snapshot_adapter_registry,
    )

    snapshot = snapshot_adapter_registry()
    monkeypatch.setenv("UNIGYM_PACKS", "")
    root = tmp_path / "explicit_runtime_pack"
    _write_adapter_pack(root)
    runtime = discover([root])
    assert "acme.run" not in ADAPTER_REGISTRY

    run = Run(_adapter_config(), pack_runtime=runtime, discover_runtime=False)
    try:
        spec = run.spec
        assert spec.adapter_name == "acme.run"
        assert "acme.run" in ADAPTER_REGISTRY
        assert current_phase() is RegistrationPhase.SPEC
    finally:
        restore_adapter_registry(snapshot)
        run.close()


def _write_law_pack(root: Path) -> None:
    (root / "actuators").mkdir(parents=True)
    (root / "unigym.pack.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "unigym.pack/1",
                "name": "law_only_pack",
                "version": "1.0.0",
                "api": 1,
                "trust": "content",
                "provides": {
                    "providers": [
                        {
                            "id": "acme.run_law",
                            "kind": "actuator_law",
                            "resource": "actuators/law.yaml",
                        }
                    ]
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (root / "actuators" / "law.yaml").write_text(
        "schema: unigym.actuator_law/1\n"
        "name: acme.run_law\n"
        "config_keys: [kp, kd]\n"
        "gains: {allowed: [kp, kd], required: [kp, kd]}\n"
        "drives: [position_pd]\n"
        "backends: [mujoco, mjwarp]\n",
        encoding="utf-8",
    )


def _law_only_config() -> dict:
    return {
        "run_name": "law_only",
        "seed": 7,
        "env": {"backend": "isaaclab2", "num_envs": 1},
        "collect": {"steps": 1},
        "articulations": [
            {
                "name": "arm",
                "joints": [{"name": "shoulder"}, {"name": "elbow"}],
                "actuation": [
                    {
                        "name": "body",
                        "law": "acme.run_law",
                        "joints": ["shoulder", "elbow"],
                        "gains": {"kp": 20.0, "kd": 1.0},
                    }
                ],
            }
        ],
        "action_spec": [
            {
                "name": "arm",
                "articulation": "arm",
                "joints": ["shoulder", "elbow"],
                "target": "position",
                "role": "body",
            },
        ],
        "policies": [{"adapter": "manual", "groups": [["arm", "arm"]]}],
        "observations": {"robot": ["arm"]},
    }


def test_law_only_config_requires_and_activates_an_injected_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unigym.actuator_law import ACTUATOR_LAW_REGISTRY
    from unigym.run.spec import requires_pack_runtime

    monkeypatch.setenv("UNIGYM_PACKS", "")
    root = tmp_path / "law_only_pack"
    _write_law_pack(root)
    runtime = discover([root])
    cfg = _law_only_config()
    assert "acme.run_law" not in ACTUATOR_LAW_REGISTRY
    assert requires_pack_runtime(cfg) is True

    run = Run(cfg, pack_runtime=runtime, discover_runtime=False)
    try:
        spec = run.spec
        assert spec.pack_runtime is runtime
        assert "acme.run_law" in ACTUATOR_LAW_REGISTRY
        assert spec.articulations[0].actuation[0].law == "acme.run_law"
    finally:
        run.close()
