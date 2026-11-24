"""StepValues: immutable boundary snapshot with isolated reads."""

from __future__ import annotations

import torch

from unigym.values import StepValues, ValueStore


def test_step_values_is_a_mapping_view_of_the_snapshot():
    """StepValues behaves as a read-only mapping over the captured names."""
    store = ValueStore(num_envs=2, device="cpu")
    store.publish("alice", "x", torch.tensor([1.0, 2.0]))
    store.publish("alice", "j", {"a": [1]})

    snapshot = store.snapshot()

    assert isinstance(snapshot, StepValues)
    assert len(snapshot) == 2
    assert set(snapshot) == {"x", "j"}
    assert "x" in snapshot
    assert torch.equal(snapshot["x"], torch.tensor([1.0, 2.0]))
    assert snapshot["j"] == {"a": [1]}


def test_get_returns_default_for_a_missing_name():
    """A missing name answers None or the supplied default."""
    snapshot = StepValues({})

    assert snapshot.get("missing") is None
    assert snapshot.get("missing", "default") == "default"


def test_stored_json_none_is_present_not_missing():
    """A JSON None is a real value; presence is decided by containment."""
    store = ValueStore(num_envs=2, device="cpu")
    store.publish("alice", "n", None)

    snapshot = store.snapshot()

    assert "n" in snapshot
    assert snapshot.get("n") is None
    assert snapshot["n"] is None


def test_tensor_reads_are_isolated_copies():
    """Indexing, get and repeated reads never share a tensor instance."""
    store = ValueStore(num_envs=2, device="cpu")
    store.publish("alice", "x", torch.tensor([1.0, 2.0]))
    snapshot = store.snapshot()

    first = snapshot["x"]
    assert isinstance(first, torch.Tensor)
    first[0] = 9.0
    second = snapshot.get("x")
    assert isinstance(second, torch.Tensor)
    second[1] = 8.0

    assert torch.equal(snapshot["x"], torch.tensor([1.0, 2.0]))


def test_json_reads_are_deep_copies():
    """Nested mutation of a read result leaves the snapshot untouched."""
    store = ValueStore(num_envs=2, device="cpu")
    store.publish("alice", "j", {"a": [1, {"b": 2}]})
    snapshot = store.snapshot()

    got = snapshot["j"]
    assert isinstance(got, dict)
    got["a"][1]["b"] = 99
    got["a"].append(3)
    bundled = snapshot.get("j")
    assert isinstance(bundled, dict)
    bundled["a"].append(4)

    assert snapshot["j"] == {"a": [1, {"b": 2}]}


def test_snapshot_returns_an_independent_mapping():
    """The mapping handed out by snapshot() cannot change the snapshot."""
    store = ValueStore(num_envs=2, device="cpu")
    store.publish("alice", "x", torch.tensor([1.0, 2.0]))
    snapshot = store.snapshot()

    out = snapshot.snapshot()
    out["x"][0] = 7.0
    out["new"] = torch.zeros(2)

    assert torch.equal(snapshot["x"], torch.tensor([1.0, 2.0]))
    assert "new" not in snapshot


def test_constructor_copies_the_source_mapping_and_values():
    """External mutation of the source mapping cannot change the snapshot."""
    tensor = torch.tensor([1.0, 2.0])
    tree = {"a": [1]}
    source = {"x": tensor, "j": tree}

    snapshot = StepValues(source)
    tensor[0] = 9.0
    tree["a"].append(2)
    source["y"] = torch.zeros(2)

    assert torch.equal(snapshot["x"], torch.tensor([1.0, 2.0]))
    assert snapshot["j"] == {"a": [1]}
    assert "y" not in snapshot


def test_snapshot_does_not_see_later_store_publishes():
    """A retained snapshot is stable after the store moves on."""
    store = ValueStore(num_envs=2, device="cpu")
    store.publish("alice", "x", torch.tensor([1.0, 2.0]))

    snapshot = store.snapshot()
    store.publish("alice", "x", torch.tensor([5.0, 5.0]))
    store.publish("alice", "y", torch.zeros(2))

    assert torch.equal(snapshot["x"], torch.tensor([1.0, 2.0]))
    assert "y" not in snapshot


def test_step_values_exposes_no_mutation_api():
    """The boundary snapshot has no publish, clear or mapping mutation API."""
    snapshot = StepValues({})

    assert not hasattr(snapshot, "__setitem__")
    assert not hasattr(snapshot, "__delitem__")
    assert not hasattr(snapshot, "publish")
    assert not hasattr(snapshot, "clear")
    assert not hasattr(snapshot, "setdefault")
    assert not hasattr(snapshot, "pop")
    assert not hasattr(snapshot, "update")
