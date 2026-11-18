"""Atomic observation config/schema/registry/snapshot/record contract."""

from __future__ import annotations

import pytest
import torch

from unigym.env.obs import (
    OBS_SCHEMA_VERSION,
    ObsConfigError,
    ObsContractError,
    ObservationConfig,
    ObservationRegistry,
    build_observation_schema,
)


def _config(**overrides) -> ObservationConfig:
    raw = {
        "robot": ["arm"],
        "contact": ["feet"],
        "object": ["cube"],
        "camera": ["ego"],
        "site": ["grip"],
        "command": True,
        "prev_action": True,
    }
    raw.update(overrides)
    return ObservationConfig.from_mapping(raw)


def _schema(config: ObservationConfig | None = None):
    return build_observation_schema(
        config or _config(),
        full_joint_names={"arm": ("shoulder", "elbow")},
        policy_joint_names={"arm": ("elbow",)},
        contacts={"feet": ((2,), "bool")},
        objects={"cube": ("lid",)},
        cameras={"ego": {"cam/ego/rgb": ((3, 4, 4), "u8")}},
        sites=("grip",),
    )


def _registry(num_envs: int = 1) -> ObservationRegistry:
    config = _config()
    return ObservationRegistry(config, _schema(config), num_envs=num_envs, device="cpu")


def _publish_all(registry: ObservationRegistry) -> None:
    n = registry.num_envs
    f32 = torch.float32
    registry.set_robot("arm", "joint_pos", torch.zeros(n, 1, dtype=f32))
    registry.set_robot("arm", "joint_pos_rel", torch.zeros(n, 2, dtype=f32))
    registry.set_robot("arm", "joint_vel", torch.zeros(n, 1, dtype=f32))
    registry.set_robot("arm", "joint_vel_full", torch.zeros(n, 2, dtype=f32))
    registry.set_robot("arm", "applied_torque", torch.zeros(n, 1, dtype=f32))
    registry.set_robot("arm", "gravity_b", torch.zeros(n, 3, dtype=f32))
    registry.set_robot("arm", "base_pos_w", torch.zeros(n, 3, dtype=f32))
    registry.set_robot(
        "arm", "base_quat_w", torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, 4).clone()
    )
    for leaf in (
        "base_lin_vel_w",
        "base_ang_vel_w",
        "base_lin_vel_b",
        "base_ang_vel_b",
    ):
        registry.set_robot("arm", leaf, torch.zeros(n, 3, dtype=f32))
    registry.set_robot("arm", "base_yaw", torch.zeros(n, 1, dtype=f32))
    registry.set_contact("feet", torch.zeros(n, 2, dtype=torch.bool))
    registry.set_object("cube", "pos", torch.zeros(n, 3, dtype=f32))
    registry.set_object("cube", "quat", torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(n, 4).clone())
    registry.set_object("cube", "joints/lid", torch.zeros(n, 1, dtype=f32))
    registry.set_site("grip", torch.zeros(n, 3, dtype=f32))
    registry.set_camera("cam/ego/rgb", torch.zeros(n, 3, 4, 4, dtype=torch.uint8))
    registry.set_command(torch.ones(n, 2, dtype=f32))
    registry.set_prev_action(torch.full((n, 1), 0.5, dtype=f32))


def test_config_parsing_rejects_missing_and_unknown_declarations() -> None:
    with pytest.raises(ObsConfigError, match="no `observations:`"):
        ObservationConfig.from_mapping(None)
    with pytest.raises(ObsConfigError, match="robot must name"):
        ObservationConfig.from_mapping({"robot": []})
    with pytest.raises(ObsConfigError, match="unknown key"):
        ObservationConfig.from_mapping({"robot": ["arm"], "junk": 1})
    with pytest.raises(ObsConfigError, match="duplicate names"):
        ObservationConfig.from_mapping({"robot": ["arm", "arm"]})


def test_config_name_validation_is_against_the_run_declarations() -> None:
    config = _config()

    config.validate(
        articulations=["arm"],
        contacts=["feet"],
        objects=["cube"],
        cameras=["ego"],
        sites=["grip"],
    )
    with pytest.raises(ObsConfigError, match="unknown cameras"):
        config.validate(
            articulations=["arm"],
            contacts=["feet"],
            objects=["cube"],
            cameras=["other"],
            sites=["grip"],
        )


def test_schema_meta_and_record_keys_are_canonical() -> None:
    schema = _schema()
    meta = schema.to_meta()

    assert meta["version"] == OBS_SCHEMA_VERSION
    assert "articulation/arm/joint_pos" in meta["keys"]
    keys = set(schema.record_keys())
    assert "articulation/arm/joint_pos" in keys
    assert "contact/feet" in keys
    assert "object/cube/pos" in keys
    assert "site/grip/pos" in keys
    assert "cam/ego/rgb" not in keys
    assert "command" not in keys
    assert "prev_action" not in keys
    assert "articulation/arm/joint_pos_rel" not in keys
    assert "articulation/arm/joint_vel_full" not in keys
    assert "articulation/arm/base_yaw" not in keys


def test_full_joint_velocity_leaf_has_full_order_and_is_not_an_artifact() -> None:
    """The full velocity leaf names every articulation joint and stays view-only."""
    schema = _schema()
    spec = schema["articulation/arm/joint_vel_full"]
    assert spec.shape == (2,)
    assert spec.joint_names == ("shoulder", "elbow")
    assert spec.source == "robot"
    assert spec.record is False
    assert "articulation/arm/joint_vel" in schema.record_keys()
    assert "articulation/arm/joint_vel_full" not in schema.record_keys()


def test_registry_publishes_typed_views_and_record() -> None:
    registry = _registry()
    _publish_all(registry)
    snapshot = registry.snapshot()

    assert torch.equal(snapshot.robot("arm").joint_pos, torch.zeros(1, 1))
    assert snapshot.robot("arm").joint_vel_full.shape == (1, 2)
    assert torch.equal(
        snapshot.robot("arm").joints(("shoulder",), leaf="joint_vel_full"),
        snapshot.robot("arm").joint_vel_full[:, :1],
    )
    assert torch.equal(
        snapshot.robot("arm").joints(("shoulder",), leaf="joint_pos_rel"),
        torch.zeros(1, 1),
    )
    assert not snapshot.contact("feet").any().any()
    assert torch.equal(snapshot.object("cube").pos_w, torch.zeros(1, 3))
    assert set(snapshot.object("cube").joints) == {"lid"}
    assert snapshot.camera("ego").rgb is not None
    assert snapshot.camera("ego").depth is None
    assert snapshot.camera("ego").t == 0
    assert torch.equal(snapshot.site("grip").pos_w, torch.zeros(1, 3))
    assert torch.equal(snapshot.command.value, torch.ones(1, 2))
    assert torch.equal(snapshot.prev_action.value, torch.full((1, 1), 0.5))

    record = snapshot.record()
    assert "cam/ego/rgb" not in record
    assert "command" not in record
    assert "prev_action" not in record
    assert "articulation/arm/joint_pos" in record
    assert "contact/feet" in record
    assert "object/cube/pos" in record
    assert "site/grip/pos" in record


def test_registry_validates_shape_and_dtype() -> None:
    registry = _registry()

    with pytest.raises(ObsContractError, match="shape"):
        registry.set_robot("arm", "gravity_b", torch.zeros(1, 2))
    with pytest.raises(ObsContractError, match="dtype"):
        registry.set_contact("feet", torch.zeros(1, 2))


def test_object_in_robot_frame_rotates_translation_and_orientation() -> None:
    registry = _registry()
    _publish_all(registry)
    registry.set_robot("arm", "base_pos_w", torch.tensor([[1.0, 0.0, 0.0]]))
    # 90 degrees about +z: world offset (1, 0, 0) is (0, 1, 0) in the base.
    half = 2**-0.5
    registry.set_robot("arm", "base_quat_w", torch.tensor([[half, 0.0, 0.0, half]]))
    registry.set_object("cube", "pos", torch.tensor([[2.0, 0.0, 0.0]]))

    pos_b, _ = registry.snapshot().object("cube").in_robot_frame("arm")
    assert pos_b[0].tolist() == pytest.approx([0.0, -1.0, 0.0], abs=1e-6)


def test_contact_detail_round_trips_in_the_snapshot() -> None:
    registry = _registry()
    _publish_all(registry)
    records = [{"normal": [0.0, 0.0, 1.0]}]
    registry.set_contact_detail("feet", records)

    snapshot = registry.snapshot()
    assert snapshot.contact("feet").detail() == records


def test_contact_detail_is_lazy_and_capability_gated() -> None:
    """Detail is never read during refresh; absent capability answers None."""
    registry = _registry()
    _publish_all(registry)
    calls: list[str] = []

    def provider() -> list[dict]:
        calls.append("read")
        return [{"v_t": 0.2}]

    registry.set_contact_detail_provider("feet", provider)
    snapshot = registry.snapshot()
    assert calls == []
    assert snapshot.contact("feet").detail() == [{"v_t": 0.2}]
    assert calls == ["read"]

    unbound = _registry()
    _publish_all(unbound)
    assert unbound.snapshot().contact("feet").detail() is None


def test_registry_rewind_state_round_trips_values_and_slots() -> None:
    registry = _registry()
    _publish_all(registry)
    registry.set_contact_detail("feet", [{"v_t": 0.1}])
    state = registry.rewind_state()

    registry.set_command(torch.zeros(1, 2))
    registry.set_contact_detail("feet", [])
    registry.restore_rewind_state(state)

    snapshot = registry.snapshot()
    assert torch.equal(snapshot.command.value, torch.ones(1, 2))
    assert snapshot.contact("feet").detail() == [{"v_t": 0.1}]


def test_multi_articulation_typed_access_is_independent() -> None:
    config = ObservationConfig.from_mapping({"robot": ["left", "right"]})
    schema = build_observation_schema(
        config,
        full_joint_names={"left": ("a",), "right": ("b",)},
    )
    registry = ObservationRegistry(config, schema, num_envs=2, device="cpu")
    registry.set_robot("left", "joint_pos", torch.full((2, 1), 1.0))
    registry.set_robot("right", "joint_pos", torch.full((2, 1), 2.0))

    snapshot = registry.snapshot()
    assert snapshot.robot("left").joint_pos.tolist() == [[1.0], [1.0]]
    assert snapshot.robot("right").joint_pos.tolist() == [[2.0], [2.0]]
