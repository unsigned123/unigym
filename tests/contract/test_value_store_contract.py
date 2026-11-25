"""Public value-store contract: protocols, provider shape, no escape hatches."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

import unigym.values as values_module
from unigym.runtime.rewind import RewindStateProvider
from unigym.values import (
    VALUE_REWIND_SCHEMA,
    StepValues,
    ValueOwnershipError,
    ValuePublisher,
    ValueStore,
    ValueView,
)


def test_values_module_exports_the_required_public_api():
    """The documented names and the schema constant are module exports."""
    for name in (
        "Value",
        "ValueView",
        "ValuePublisher",
        "StepValues",
        "ValueStore",
        "ValueOwnershipError",
    ):
        assert hasattr(values_module, name)
    assert VALUE_REWIND_SCHEMA == 2


def test_value_ownership_error_is_a_value_error_with_a_teaching_message():
    """An ownership conflict is a ValueError naming the value and both owners."""
    store = ValueStore(num_envs=2, device="cpu")
    store.publish("alice", "x", torch.zeros(2))

    with pytest.raises(ValueOwnershipError) as excinfo:
        store.publish("bob", "x", torch.ones(2))

    assert isinstance(excinfo.value, ValueError)
    message = str(excinfo.value)
    assert "'x'" in message
    assert "alice" in message
    assert "bob" in message


def test_step_values_satisfies_the_value_view_protocol():
    """The boundary snapshot provides the read-only view face."""
    snapshot = ValueStore(num_envs=2, device="cpu").snapshot()

    assert isinstance(snapshot, ValueView)
    assert isinstance(snapshot, Mapping)


def test_bound_publisher_satisfies_the_value_publisher_protocol():
    """publisher(owner) returns the write-only publisher face."""
    publisher = ValueStore(num_envs=2, device="cpu").publisher("alice")

    assert isinstance(publisher, ValuePublisher)


def test_value_store_satisfies_the_read_and_rewind_faces():
    """The store is both a ValueView and a named RewindStateProvider."""
    store = ValueStore(num_envs=2, device="cpu")

    assert isinstance(store, ValueView)
    assert isinstance(store, RewindStateProvider)
    assert store.name == "values"


def test_snapshot_is_an_immutable_step_values_boundary():
    """snapshot() returns StepValues with no mutation API."""
    store = ValueStore(num_envs=2, device="cpu")
    store.publish("alice", "x", torch.zeros(2))

    snapshot = store.snapshot()

    assert isinstance(snapshot, StepValues)
    assert not hasattr(snapshot, "publish")
    assert not hasattr(snapshot, "clear")
    assert not hasattr(snapshot, "__setitem__")


def test_publisher_face_does_not_expose_read_or_rewind_operations():
    """A publisher cannot read values or drive the rewind face."""
    publisher = ValueStore(num_envs=2, device="cpu").publisher("alice")

    assert not hasattr(publisher, "get")
    assert not hasattr(publisher, "snapshot")
    assert not hasattr(publisher, "rewind_state")
    assert not hasattr(publisher, "apply_rewind_state")
    assert not hasattr(publisher, "__contains__")


def test_rewind_payload_has_the_documented_schema_shape():
    """rewind_state() emits exactly the schema-2 values/writers payload."""
    store = ValueStore(num_envs=2, device="cpu")
    store.publish("alice", "x", torch.zeros(2))

    state = store.rewind_state()

    assert set(state) == {"schema", "values", "writers"}
    assert state["schema"] == VALUE_REWIND_SCHEMA
    assert state["writers"] == {"x": "alice"}


def test_only_the_store_exposes_subscribe():
    """Subscriptions belong to the store; the view and publisher do not."""
    store = ValueStore(num_envs=2, device="cpu")
    snapshot = store.snapshot()
    publisher = store.publisher("alice")

    assert callable(store.subscribe)
    assert not hasattr(snapshot, "subscribe")
    assert not hasattr(publisher, "subscribe")
    assert not hasattr(ValueView, "subscribe")


def test_subscribe_returns_an_idempotent_unsubscribe_callable():
    """subscribe returns a callable that removes the registration repeatedly."""
    store = ValueStore(num_envs=2, device="cpu")

    unsubscribe = store.subscribe(lambda name, value: None)

    assert callable(unsubscribe)
    assert unsubscribe() is None
    assert unsubscribe() is None
