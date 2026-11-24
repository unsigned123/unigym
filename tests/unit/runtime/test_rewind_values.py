"""ValueStore rewind provider: schema-2 payloads and zero-change refusal."""

from __future__ import annotations

import pytest
import torch

from unigym.runtime.rewind import RewindGateError, RewindStateProvider
from unigym.values import VALUE_REWIND_SCHEMA, ValueOwnershipError, ValueStore


def _store() -> ValueStore:
    return ValueStore(num_envs=2, device="cpu")


def test_value_store_is_a_named_rewind_provider():
    """The store satisfies the provider protocol under the values key."""
    store = _store()

    assert store.name == "values"
    assert isinstance(store, RewindStateProvider)


def test_rewind_state_round_trips_values_and_writers():
    """Capture/restore replaces values and restores every ownership claim."""
    store = _store()
    store.publish("alice", "a", torch.tensor([1.0, 2.0]))
    store.publish("bob", "b", {"k": [1, 2]})
    state = store.rewind_state()

    store.publish("alice", "a", torch.tensor([9.0, 9.0]))
    store.clear("bob", "b")
    store.publish("alice", "c", torch.zeros(2))
    store.apply_rewind_state(state)

    assert torch.equal(store.get("a"), torch.tensor([1.0, 2.0]))
    assert store.get("b") == {"k": [1, 2]}
    assert store.get("c") is None
    with pytest.raises(ValueOwnershipError, match="alice"):
        store.publish("bob", "a", torch.zeros(2))
    with pytest.raises(ValueOwnershipError, match="bob"):
        store.publish("alice", "b", torch.zeros(2))
    store.publish("bob", "c", torch.zeros(2))


def test_rewind_state_returns_private_copies():
    """Mutating a captured payload cannot change the live store."""
    store = _store()
    store.publish("alice", "j", {"a": [1, {"b": 2}]})
    store.publish("alice", "x", torch.tensor([1.0, 2.0]))

    state = store.rewind_state()
    mutated_tree = state["values"]["j"]
    assert isinstance(mutated_tree, dict)
    mutated_tree["a"].append(99)
    mutated_tensor = state["values"]["x"]
    assert isinstance(mutated_tensor, torch.Tensor)
    mutated_tensor[0] = 42.0
    mutated_writers = state["writers"]
    assert isinstance(mutated_writers, dict)
    mutated_writers["j"] = "mallory"

    assert store.get("j") == {"a": [1, {"b": 2}]}
    assert torch.equal(store.get("x"), torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueOwnershipError, match="alice"):
        store.publish("mallory", "j", {"a": []})


def test_apply_restores_json_values_with_private_copies():
    """A restored JSON value is independent of both the store and the state."""
    store = _store()
    store.publish("alice", "j", {"a": [1]})
    state = store.rewind_state()
    store.publish("alice", "j", {"a": [2]})

    store.apply_rewind_state(state)

    assert store.get("j") == {"a": [1]}
    restored = store.get("j")
    assert isinstance(restored, dict)
    restored["a"].append(3)
    assert store.get("j") == {"a": [1]}


def test_old_flat_signal_style_payload_is_rejected():
    """The legacy flat {name: value} payload is not a values rewind state."""
    with pytest.raises(RewindGateError, match="flat|schema"):
        _store().validate_rewind_state({"a": torch.zeros(2)})


def test_wrong_schema_version_is_rejected():
    """Only the explicit schema-2 payload is accepted."""
    payload = {"schema": 1, "values": {}, "writers": {}}

    with pytest.raises(RewindGateError, match="schema must be 2"):
        _store().validate_rewind_state(payload)


def test_value_without_a_writer_is_rejected():
    """Every value name must have an ownership claim."""
    payload = {
        "schema": VALUE_REWIND_SCHEMA,
        "values": {"a": torch.zeros(2)},
        "writers": {},
    }

    with pytest.raises(RewindGateError, match="has no writer"):
        _store().validate_rewind_state(payload)


def test_unknown_payload_key_is_rejected():
    """The payload has exactly the required keys."""
    payload = {
        "schema": VALUE_REWIND_SCHEMA,
        "values": {},
        "writers": {},
        "extra": 1,
    }

    with pytest.raises(RewindGateError, match="keys must be exactly"):
        _store().validate_rewind_state(payload)


def test_apply_refuses_a_malformed_mixed_payload_without_change():
    """A bad value deep in the payload leaves every value and claim unchanged."""
    store = _store()
    store.publish("alice", "a", torch.tensor([1.0, 2.0]))
    store.publish("bob", "b", {"k": [1]})
    before = store.rewind_state()
    bad = {
        "schema": VALUE_REWIND_SCHEMA,
        "values": {
            "a": torch.tensor([5.0, 5.0]),
            "b": torch.zeros(3),
        },
        "writers": {"a": "alice", "b": "bob"},
    }

    with pytest.raises(RewindGateError, match="leading dimension"):
        store.apply_rewind_state(bad)

    assert torch.equal(store.get("a"), torch.tensor([1.0, 2.0]))
    assert store.get("b") == {"k": [1]}
    after = store.rewind_state()
    assert torch.equal(after["values"]["a"], before["values"]["a"])
    assert after["values"]["b"] == before["values"]["b"]
    assert after["writers"] == before["writers"]
    with pytest.raises(ValueOwnershipError, match="alice"):
        store.publish("mallory", "a", torch.zeros(2))


def test_apply_replaces_writers_instead_of_merging():
    """Restoring an empty state releases names claimed after the freeze."""
    store = _store()
    empty = store.rewind_state()
    store.publish("alice", "a", torch.zeros(2))

    store.apply_rewind_state(empty)

    assert "a" not in store
    store.publish("bob", "a", torch.ones(2))
    assert torch.equal(store.get("a"), torch.ones(2))


def test_cleared_claim_survives_a_rewind_round_trip():
    """A cleared name keeps its owner across capture and restore."""
    store = _store()
    store.publish("alice", "a", torch.zeros(2))
    store.clear("alice", "a")

    state = store.rewind_state()

    assert state["values"] == {}
    assert state["writers"] == {"a": "alice"}

    store.apply_rewind_state(state)

    assert "a" not in store
    with pytest.raises(ValueOwnershipError, match="alice"):
        store.publish("bob", "a", torch.ones(2))
    store.publish("alice", "a", torch.ones(2))
    assert torch.equal(store.get("a"), torch.ones(2))
