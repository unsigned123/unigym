"""Public contract for selected observation values and the publisher face."""

from __future__ import annotations

import inspect
from dataclasses import fields

import torch

from unigym.env.obs import (
    ObservationConfig,
    ObservationRegistry,
    ObservationSchema,
    build_observation_schema,
)
from unigym.values import Value, ValuePublisher, ValueStore

JOINT_POS = "articulation/arm/joint_pos"


class _RecordingPublisher:
    """Write-only face that records only ``publish``/``clear`` calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def publish(self, name: str, value: Value) -> None:
        del value
        self.calls.append(("publish", name))

    def clear(self, name: str) -> None:
        self.calls.append(("clear", name))


def _config(values: tuple[str, ...] = ()) -> ObservationConfig:
    return ObservationConfig.from_mapping({"robot": ["arm"], "values": list(values)})


def _registry(config: ObservationConfig) -> tuple[ObservationRegistry, ObservationSchema]:
    schema = build_observation_schema(
        config,
        full_joint_names={"arm": ("shoulder",)},
        policy_joint_names={"arm": ("shoulder",)},
    )
    return (
        ObservationRegistry(config, schema, num_envs=1, device="cpu"),
        schema,
    )


def test_values_is_a_public_last_field_with_an_empty_default() -> None:
    declared = fields(ObservationConfig)
    assert declared[-1].name == "values"
    assert declared[-1].default == ()
    assert ObservationConfig(robot=("arm",)).values == ()


def test_values_projects_through_to_mapping() -> None:
    config = _config((JOINT_POS,))

    assert config.to_mapping()["values"] == [JOINT_POS]
    assert ObservationConfig.from_mapping(config.to_mapping()) == config


def test_publish_values_signature_is_the_write_only_publisher_face() -> None:
    signature = inspect.signature(ObservationRegistry.publish_values)
    assert list(signature.parameters) == ["self", "publisher"]
    assert isinstance(ValueStore(1, "cpu").publisher("observation"), ValuePublisher)
    assert isinstance(_RecordingPublisher(), ValuePublisher)


def test_registry_does_not_auto_publish_or_mutate_the_schema() -> None:
    config = _config((JOINT_POS,))
    registry, schema = _registry(config)
    before = schema.to_meta()
    store = ValueStore(num_envs=1, device="cpu")
    registry.set_robot("arm", "joint_pos", torch.zeros(1, 1))

    assert dict(store.snapshot()) == {}
    recorder = _RecordingPublisher()
    registry.publish_values(recorder)

    assert schema.to_meta() == before
    assert recorder.calls == [("publish", f"obs/{JOINT_POS}")]


def test_publish_values_only_writes_the_obs_namespace() -> None:
    config = _config((JOINT_POS,))
    registry, _ = _registry(config)
    registry.set_robot("arm", "joint_pos", torch.full((1, 1), 2.0))
    store = ValueStore(num_envs=1, device="cpu")
    publisher = store.publisher("observation")

    registry.publish_values(publisher)

    assert list(store.snapshot()) == [f"obs/{JOINT_POS}"]
    value = store.get(f"obs/{JOINT_POS}")
    assert value is not None
    assert torch.equal(value, torch.full((1, 1), 2.0))
