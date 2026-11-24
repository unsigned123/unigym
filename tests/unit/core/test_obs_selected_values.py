"""Explicit observation-value selection: config, validation and publication."""

from __future__ import annotations

import pytest
import torch

from unigym.env.obs import (
    ObsConfigError,
    ObservationConfig,
    ObservationRegistry,
    ObservationSchema,
    ObservationSpec,
    build_observation_schema,
)
from unigym.values import Value, ValueStore

ROBOT = "arm"
JOINT_POS = "articulation/arm/joint_pos"
JOINT_VEL = "articulation/arm/joint_vel"


class _Recorder:
    """Write-only publisher that records each call in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Value | None]] = []

    def publish(self, name: str, value: Value) -> None:
        self.calls.append(("publish", name, value))

    def clear(self, name: str) -> None:
        self.calls.append(("clear", name, None))


def _schema(config: ObservationConfig) -> ObservationSchema:
    return build_observation_schema(
        config,
        full_joint_names={ROBOT: ("shoulder",)},
        policy_joint_names={ROBOT: ("shoulder",)},
        contacts={"feet": ((2,), "bool")},
    )


def _registry(config: ObservationConfig, *, num_envs: int = 1) -> ObservationRegistry:
    return ObservationRegistry(config, _schema(config), num_envs=num_envs, device="cpu")


def test_config_values_default_parse_and_round_trip() -> None:
    assert ObservationConfig(robot=(ROBOT,)).values == ()
    config = ObservationConfig.from_mapping({"robot": [ROBOT], "values": [JOINT_VEL, JOINT_POS]})

    assert config.values == (JOINT_VEL, JOINT_POS)
    assert config.to_mapping()["values"] == [JOINT_VEL, JOINT_POS]
    assert ObservationConfig.from_mapping(config.to_mapping()) == config


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ([JOINT_POS, JOINT_POS], "duplicate names"),
        ([""], "non-empty strings"),
        ([1], "non-empty strings"),
    ],
)
def test_config_values_reject_duplicate_empty_and_non_string_entries(raw, match) -> None:
    with pytest.raises(ObsConfigError, match=match):
        ObservationConfig.from_mapping({"robot": [ROBOT], "values": raw})


def test_unknown_selected_key_is_rejected_at_registry_construction() -> None:
    config = ObservationConfig.from_mapping(
        {"robot": [ROBOT], "values": ["articulation/arm/not_published"]}
    )

    with pytest.raises(ObsConfigError, match="unknown observation key"):
        _registry(config)


@pytest.mark.parametrize("key", ["cam/ego/rgb", "sensor/frame/depth", "sensor/ir"])
def test_camera_and_depth_selected_keys_are_rejected_at_construction(key: str) -> None:
    schema = ObservationSchema(
        {
            JOINT_POS: ObservationSpec(JOINT_POS, (1,), "f32"),
            key: ObservationSpec(key, (1,), "f32", source="camera"),
        }
    )
    config = ObservationConfig(robot=(ROBOT,), values=(key,))

    with pytest.raises(ObsConfigError, match="camera/depth"):
        ObservationRegistry(config, schema, num_envs=1, device="cpu")


def test_selected_subset_publishes_only_obs_namespaced_keys_in_declared_order() -> None:
    config = ObservationConfig.from_mapping(
        {
            "robot": [ROBOT],
            "contact": ["feet"],
            "values": [JOINT_VEL, JOINT_POS],
        }
    )
    registry = _registry(config)
    registry.set_robot(ROBOT, "joint_pos", torch.full((1, 1), 1.0))
    registry.set_robot(ROBOT, "joint_vel", torch.full((1, 1), 2.0))
    registry.set_contact("feet", torch.zeros(1, 2, dtype=torch.bool))
    recorder = _Recorder()

    registry.publish_values(recorder)

    assert [call[:2] for call in recorder.calls] == [
        ("publish", f"obs/{JOINT_VEL}"),
        ("publish", f"obs/{JOINT_POS}"),
    ]
    assert torch.equal(recorder.calls[0][2], torch.full((1, 1), 2.0))
    assert torch.equal(recorder.calls[1][2], torch.full((1, 1), 1.0))
    names = [call[1] for call in recorder.calls]
    assert all(name.startswith("obs/") for name in names)
    assert not any("cam/" in name or name.endswith(("/rgb", "/depth")) for name in names)
    assert "obs/contact/feet" not in names


def test_missing_selected_key_clears_its_namespace_instead_of_publishing_stale() -> None:
    config = ObservationConfig.from_mapping({"robot": [ROBOT], "values": [JOINT_POS, JOINT_VEL]})
    registry = _registry(config)
    registry.set_robot(ROBOT, "joint_pos", torch.full((1, 1), 3.0))
    store = ValueStore(num_envs=1, device="cpu")
    publisher = store.publisher("observation")
    publisher.publish(f"obs/{JOINT_VEL}", torch.full((1, 1), -1.0))

    registry.publish_values(publisher)

    assert store.get(f"obs/{JOINT_VEL}") is None
    published = store.get(f"obs/{JOINT_POS}")
    assert published is not None
    assert torch.equal(published, torch.full((1, 1), 3.0))


def test_empty_selection_publishes_nothing() -> None:
    config = ObservationConfig.from_mapping({"robot": [ROBOT]})
    registry = _registry(config)
    registry.set_robot(ROBOT, "joint_pos", torch.full((1, 1), 1.0))
    recorder = _Recorder()
    store = ValueStore(num_envs=1, device="cpu")

    registry.publish_values(recorder)
    registry.publish_values(store.publisher("observation"))

    assert recorder.calls == []
    assert dict(store.snapshot()) == {}
