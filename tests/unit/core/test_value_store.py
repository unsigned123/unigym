"""ValueStore transport, ownership, copy isolation and subscriptions."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unigym.values import StepValues, ValueOwnershipError, ValueStore


def _store(num_envs: int = 2, device: str = "cpu") -> ValueStore:
    return ValueStore(num_envs=num_envs, device=device)


@pytest.mark.parametrize("num_envs", [0, -1, 1.5, True, "2", None])
def test_constructor_rejects_an_invalid_num_envs(num_envs):
    """The env count is a positive int, never a bool or non-int."""
    with pytest.raises(ValueError, match="num_envs"):
        ValueStore(num_envs=num_envs, device="cpu")


def test_constructor_normalizes_device_and_names_the_provider():
    """The device is a torch.device and the rewind name is the values key."""
    store = _store()

    assert store.device == torch.device("cpu")
    assert store.name == "values"


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        (torch.bool, (2,)),
        (torch.int64, (2,)),
        (torch.float32, (2,)),
        (torch.float32, (2, 3)),
    ],
)
def test_publish_round_trips_arbitrary_dtype_and_trailing_shape(dtype, shape):
    """Tensors keep their dtype and trailing shape with the per-env leading axis."""
    store = _store()
    value = torch.ones(shape, dtype=dtype)

    store.publish("alice", "x", value)

    stored = store.get("x")
    assert isinstance(stored, torch.Tensor)
    assert stored.dtype == dtype
    assert stored.shape == shape
    assert torch.equal(stored, value)


def test_publish_clones_the_callers_tensor():
    """Mutating the published tensor cannot change the stored value."""
    store = _store()
    value = torch.tensor([1.0, 2.0])

    store.publish("alice", "x", value)
    value[0] = 99.0

    assert torch.equal(store.get("x"), torch.tensor([1.0, 2.0]))


def test_get_and_snapshot_return_isolated_tensor_copies():
    """Read results cannot mutate the store, whether via get or snapshot."""
    store = _store()
    store.publish("alice", "x", torch.tensor([1.0, 2.0]))

    got = store.get("x")
    assert isinstance(got, torch.Tensor)
    got[0] = 7.0
    snapshot = store.snapshot()
    assert isinstance(snapshot, StepValues)
    bundled = snapshot["x"]
    assert isinstance(bundled, torch.Tensor)
    bundled[1] = 8.0

    assert torch.equal(store.get("x"), torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("shape", [(), (3,), (0,)])
def test_publish_rejects_a_wrong_leading_dimension(shape):
    """A tensor must be at least 1-D with leading dimension num_envs."""
    store = _store()

    with pytest.raises(ValueError, match="leading dimension"):
        store.publish("alice", "x", torch.zeros(shape))


def test_publish_rejects_a_wrong_device():
    """Tensor values must live on the store device."""
    store = _store()

    with pytest.raises(ValueError, match="device"):
        store.publish("alice", "x", torch.zeros(2, device="meta"))


def test_publish_rejects_a_sparse_layout():
    """Only dense (strided) tensors are transported."""
    store = _store()
    sparse = torch.zeros(2, 3).to_sparse()

    with pytest.raises(ValueError, match="dense"):
        store.publish("alice", "x", sparse)


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, 3, -1.5, "text", [1, "a", None], {"a": {"b": [1, 2]}}],
)
def test_publish_round_trips_json_scalars_and_trees(value):
    """JSON scalars, lists and string-keyed mappings round-trip unchanged."""
    store = _store()

    store.publish("alice", "v", value)

    assert store.get("v") == value
    assert "v" in store


def test_publish_normalizes_a_tuple_to_a_json_list():
    """Existing shared JSON semantics accept tuples as lists."""
    store = _store()

    store.publish("alice", "v", (1, 2))

    assert store.get("v") == [1, 2]


def test_publish_clones_a_nested_json_input():
    """Mutating the caller's JSON tree cannot change the stored value."""
    store = _store()
    value = {"a": [1, {"b": 2}]}

    store.publish("alice", "v", value)
    value["a"][1]["b"] = 99
    value["a"].append(3)

    assert store.get("v") == {"a": [1, {"b": 2}]}


def test_json_reads_are_independent_copies():
    """Mutating a returned JSON tree cannot change the stored value."""
    store = _store()
    store.publish("alice", "v", {"a": [1, 2]})

    got = store.get("v")
    assert isinstance(got, dict)
    got["a"].append(3)
    bundled = store.snapshot()["v"]
    assert isinstance(bundled, dict)
    bundled["a"].append(4)

    assert store.get("v") == {"a": [1, 2]}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_publish_rejects_non_finite_floats(bad):
    """NaN and infinities are not JSON-safe values."""
    with pytest.raises(ValueError, match="finite"):
        _store().publish("alice", "v", bad)


def test_publish_rejects_a_numpy_array():
    """Numpy arrays are not JSON-safe and are not tensors."""
    with pytest.raises(TypeError, match="JSON-safe"):
        _store().publish("alice", "v", np.zeros(2))


def test_publish_rejects_an_arbitrary_object():
    """Arbitrary objects are refused at the boundary."""
    with pytest.raises(TypeError, match="JSON-safe"):
        _store().publish("alice", "v", object())


def test_publish_rejects_a_non_string_mapping_key():
    """JSON objects must have string keys."""
    with pytest.raises(ValueError, match="keys must be strings"):
        _store().publish("alice", "v", {1: "x"})


def test_missing_name_is_absent_not_a_zero_value():
    """A never-published name answers None / absent."""
    store = _store()

    assert store.get("missing") is None
    assert "missing" not in store
    assert len(store.snapshot()) == 0


def test_same_owner_last_publish_wins():
    """A later publish by the same owner replaces the stored value."""
    store = _store()

    store.publish("alice", "x", torch.tensor([1.0, 1.0]))
    store.publish("alice", "x", torch.tensor([2.0, 2.0]))

    assert torch.equal(store.get("x"), torch.tensor([2.0, 2.0]))


def test_different_owner_cannot_republish_a_claimed_name():
    """An ownership conflict names the value and both owners."""
    store = _store()
    store.publish("alice", "x", torch.zeros(2))

    with pytest.raises(ValueOwnershipError) as excinfo:
        store.publish("bob", "x", torch.ones(2))

    assert isinstance(excinfo.value, ValueError)
    message = str(excinfo.value)
    assert "'x'" in message
    assert "alice" in message
    assert "bob" in message
    assert torch.equal(store.get("x"), torch.zeros(2))


def test_invalid_publish_does_not_claim_the_name():
    """A refused value never claims a name for its would-be owner."""
    store = _store()

    with pytest.raises(ValueError, match="leading dimension"):
        store.publish("alice", "x", torch.zeros(3))

    store.publish("bob", "x", torch.zeros(2))

    assert torch.equal(store.get("x"), torch.zeros(2))


def test_clear_removes_the_value_but_keeps_the_claim():
    """Clearing is not a release; only the claimer may publish again."""
    store = _store()
    store.publish("alice", "x", torch.zeros(2))

    store.clear("alice", "x")

    assert store.get("x") is None
    assert "x" not in store
    with pytest.raises(ValueOwnershipError, match="alice"):
        store.publish("bob", "x", torch.ones(2))

    store.publish("alice", "x", torch.ones(2))
    assert torch.equal(store.get("x"), torch.ones(2))


def test_clear_all_removes_only_that_owners_values():
    """clear(owner) drops that owner's values and leaves other owners alone."""
    store = _store()
    store.publish("alice", "a", torch.zeros(2))
    store.publish("alice", "b", torch.zeros(2))
    store.publish("bob", "c", torch.zeros(2))

    store.clear("alice")

    assert store.get("a") is None
    assert store.get("b") is None
    assert torch.equal(store.get("c"), torch.zeros(2))
    with pytest.raises(ValueOwnershipError, match="alice"):
        store.publish("bob", "a", torch.ones(2))


def test_clear_another_owners_name_is_a_conflict():
    """An owner cannot clear a name claimed by another owner."""
    store = _store()
    store.publish("alice", "x", torch.zeros(2))

    with pytest.raises(ValueOwnershipError, match="alice"):
        store.clear("bob", "x")

    assert torch.equal(store.get("x"), torch.zeros(2))


def test_clear_unclaimed_name_is_a_noop():
    """Clearing a never-claimed name does not create a claim."""
    store = _store()

    store.clear("alice", "never")
    store.publish("bob", "never", torch.zeros(2))

    assert torch.equal(store.get("never"), torch.zeros(2))


@pytest.mark.parametrize("owner", ["", 3, None])
def test_publish_rejects_an_invalid_owner(owner):
    """Owners are non-empty strings."""
    with pytest.raises(ValueError, match="owner"):
        _store().publish(owner, "x", torch.zeros(2))


@pytest.mark.parametrize("name", ["", 3, None])
def test_publish_rejects_an_invalid_name(name):
    """Names are non-empty strings."""
    with pytest.raises(ValueError, match="name"):
        _store().publish("alice", name, torch.zeros(2))


def test_publisher_is_bound_write_only_and_enforces_ownership():
    """A publisher publishes/clears as its owner and exposes no read face."""
    store = _store()
    publisher = store.publisher("alice")

    assert not hasattr(publisher, "get")
    assert not hasattr(publisher, "snapshot")
    assert not hasattr(publisher, "rewind_state")

    publisher.publish("x", torch.ones(2))
    assert torch.equal(store.get("x"), torch.ones(2))

    with pytest.raises(ValueOwnershipError, match="alice"):
        store.publish("bob", "x", torch.zeros(2))

    publisher.clear("x")
    assert store.get("x") is None
    with pytest.raises(ValueOwnershipError, match="alice"):
        store.publish("bob", "x", torch.zeros(2))


def test_publisher_rejects_an_invalid_owner():
    """The publisher factory validates its owner up front."""
    with pytest.raises(ValueError, match="owner"):
        _store().publisher("")


def test_subscribe_notifies_after_publish_with_an_isolated_copy():
    """A publish delivers the new name and a copy that cannot mutate the store."""
    store = _store()
    received = []
    store.subscribe(lambda name, value: received.append((name, value)))

    store.publish("alice", "x", torch.tensor([1.0, 2.0]))

    assert len(received) == 1
    name, value = received[0]
    assert name == "x"
    assert isinstance(value, torch.Tensor)
    assert torch.equal(value, torch.tensor([1.0, 2.0]))
    value[0] = 9.0
    assert torch.equal(store.get("x"), torch.tensor([1.0, 2.0]))


def test_each_subscriber_receives_its_own_copy():
    """One callback mutating its copy cannot change another callback's copy."""
    store = _store()
    seen = []

    def mutate(name, value):
        value[0] = 9.0

    store.subscribe(mutate)
    store.subscribe(lambda name, value: seen.append(value))

    store.publish("alice", "x", torch.tensor([1.0, 2.0]))

    assert torch.equal(seen[0], torch.tensor([1.0, 2.0]))
    assert torch.equal(store.get("x"), torch.tensor([1.0, 2.0]))


def test_subscribe_delivers_isolated_json_copies():
    """Mutating a delivered JSON tree cannot change the stored value."""
    store = _store()
    received = []
    store.subscribe(lambda name, value: received.append(value))

    store.publish("alice", "j", {"a": [1]})

    received[0]["a"].append(2)
    assert store.get("j") == {"a": [1]}


def test_subscribe_names_filter_matches_exact_names_only():
    """A names filter delivers only the exact listed names, never prefixes."""
    store = _store()
    seen = []
    store.subscribe(lambda name, value: seen.append(name), names=frozenset({"phase"}))

    store.publish("alice", "phase", torch.zeros(2))
    store.publish("alice", "phase_x", torch.zeros(2))
    store.publish("alice", "other", torch.zeros(2))

    assert seen == ["phase"]


def test_subscribe_without_names_receives_every_value():
    """names=None subscribes to every published name."""
    store = _store()
    seen = []
    store.subscribe(lambda name, value: seen.append(name))

    store.publish("alice", "a", torch.zeros(2))
    store.publish("alice", "b", {"k": 1})

    assert seen == ["a", "b"]


def test_subscriptions_notify_in_registration_order_without_duplicates():
    """Each publish calls every matching callback once, in registration order."""
    store = _store()
    order = []
    store.subscribe(lambda name, value: order.append("first"))
    store.subscribe(lambda name, value: order.append("second"))

    store.publish("alice", "x", torch.zeros(2))

    assert order == ["first", "second"]


def test_unsubscribe_stops_notifications_and_is_idempotent():
    """The returned callable removes one subscription and is safe to repeat."""
    store = _store()
    seen = []
    unsubscribe = store.subscribe(lambda name, value: seen.append(name))

    unsubscribe()
    unsubscribe()
    store.publish("alice", "x", torch.zeros(2))

    assert seen == []


def test_unsubscribe_only_removes_its_own_subscription():
    """Unsubscribing one callback leaves the others registered."""
    store = _store()
    seen = []
    first = store.subscribe(lambda name, value: seen.append("first"))
    store.subscribe(lambda name, value: seen.append("second"))

    first()
    store.publish("alice", "x", torch.zeros(2))

    assert seen == ["second"]


def test_clear_notifies_none_for_a_removed_value_only_once():
    """Clearing an existing value notifies once; a repeated clear is silent."""
    store = _store()
    seen = []
    store.subscribe(lambda name, value: seen.append((name, value)))
    store.publish("alice", "x", torch.zeros(2))
    seen.clear()

    store.clear("alice", "x")
    store.clear("alice", "x")

    assert seen == [("x", None)]


def test_names_filter_applies_to_clear_notifications():
    """A filtered subscription ignores clears of unlisted names."""
    store = _store()
    seen = []
    store.subscribe(lambda name, value: seen.append(name), names=frozenset({"phase"}))
    store.publish("alice", "phase", torch.zeros(2))
    store.publish("alice", "other", torch.zeros(2))
    seen.clear()

    store.clear("alice", "other")
    assert seen == []

    store.clear("alice", "phase")
    assert seen == ["phase"]


def test_clear_all_notifies_once_per_removed_value():
    """clear(owner) notifies each removed value and ignores other owners."""
    store = _store()
    seen = []
    store.subscribe(lambda name, value: seen.append((name, value)))
    store.publish("alice", "a", torch.zeros(2))
    store.publish("alice", "b", torch.zeros(2))
    store.publish("bob", "c", torch.zeros(2))
    seen.clear()

    store.clear("alice")
    store.clear("alice")

    assert seen == [("a", None), ("b", None)]


def test_clear_all_removes_every_value_before_a_raising_callback():
    """A raising callback stops later notifications but the removal is complete."""
    store = _store()
    store.publish("alice", "a", torch.zeros(2))
    store.publish("alice", "b", torch.zeros(2))
    seen = []

    class _Boom(Exception):
        pass

    def callback(name, value):
        seen.append(name)
        raise _Boom

    store.subscribe(callback)

    with pytest.raises(_Boom):
        store.clear("alice")

    assert seen == ["a"]
    assert "a" not in store
    assert "b" not in store


def test_failed_publish_does_not_notify():
    """A refused value never reaches subscribers."""
    store = _store()
    seen = []
    store.subscribe(lambda name, value: seen.append(name))

    with pytest.raises(ValueError, match="leading dimension"):
        store.publish("alice", "x", torch.zeros(3))

    assert seen == []


def test_refused_publish_does_not_notify():
    """An ownership conflict is refused before any notification."""
    store = _store()
    store.publish("alice", "x", torch.zeros(2))
    seen = []
    store.subscribe(lambda name, value: seen.append(name))

    with pytest.raises(ValueOwnershipError):
        store.publish("bob", "x", torch.ones(2))

    assert seen == []


def test_reads_and_snapshot_do_not_notify():
    """get, snapshot and containment are silent."""
    store = _store()
    seen = []
    store.subscribe(lambda name, value: seen.append(name))
    store.publish("alice", "x", torch.zeros(2))
    seen.clear()

    store.get("x")
    store.snapshot()
    "x" in store

    assert seen == []


@pytest.mark.parametrize("mutation", ["publish", "clear"])
def test_reentrant_store_mutation_from_a_callback_raises(mutation):
    """A callback that publishes or clears gets a teaching RuntimeError."""
    store = _store()
    failures = []

    def callback(name, value):
        try:
            if mutation == "publish":
                store.publish("alice", "y", torch.zeros(2))
            else:
                store.clear("alice", "x")
        except RuntimeError as err:
            failures.append(str(err))
            raise

    store.subscribe(callback)

    with pytest.raises(RuntimeError, match="during a subscription notification"):
        store.publish("alice", "x", torch.zeros(2))

    assert failures
    assert "y" not in store
    assert torch.equal(store.get("x"), torch.zeros(2))


def test_callback_exception_propagates_and_the_value_is_already_stored():
    """A raising callback propagates out of publish after the value is stored."""
    store = _store()

    class _Boom(Exception):
        pass

    def callback(name, value):
        raise _Boom

    store.subscribe(callback)

    with pytest.raises(_Boom):
        store.publish("alice", "x", torch.zeros(2))

    assert torch.equal(store.get("x"), torch.zeros(2))


def test_subscriptions_are_per_store_and_do_not_leak():
    """A subscription on one store never sees another store's publishes."""
    first = _store()
    second = _store()
    seen = []
    first.subscribe(lambda name, value: seen.append(name))

    second.publish("alice", "x", torch.zeros(2))
    assert seen == []

    first.publish("alice", "x", torch.zeros(2))
    assert seen == ["x"]


def test_subscribe_rejects_a_non_callable_callback():
    """A subscription callback must be callable."""
    with pytest.raises(TypeError, match="callable"):
        _store().subscribe(None)


@pytest.mark.parametrize("names", ["x", {"x"}, ["x"]])
def test_subscribe_rejects_names_that_are_not_frozensets(names):
    """The names filter must be a frozenset, never a string or bare set."""
    with pytest.raises(TypeError, match="frozenset"):
        _store().subscribe(lambda name, value: None, names=names)


@pytest.mark.parametrize("names", [frozenset({""}), frozenset({3})])
def test_subscribe_rejects_invalid_names(names):
    """Each filtered name must be a non-empty string."""
    with pytest.raises(ValueError, match="non-empty str"):
        _store().subscribe(lambda name, value: None, names=names)
