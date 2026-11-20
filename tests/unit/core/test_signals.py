"""SignalBus transport: per-env values, fail-fast validation, isolation."""

from __future__ import annotations

import pytest
import torch

from unigym.signals import SignalBus


@pytest.mark.parametrize(
    ("dtype", "shape"),
    [
        (torch.bool, (3,)),
        (torch.int64, (3,)),
        (torch.float32, (3,)),
        (torch.float32, (3, 4)),
    ],
)
def test_publish_round_trips_arbitrary_dtype_and_trailing_shape(dtype, shape):
    """Core stores any dtype/trailing shape with a per-env leading axis."""
    bus = SignalBus(num_envs=3, device="cpu")
    value = torch.ones(shape, dtype=dtype)

    bus.publish("user/name", value)

    stored = bus.get("user/name")
    assert stored is not None
    assert stored.dtype == dtype
    assert stored.shape == shape
    assert torch.equal(stored, value)


def test_publish_clones_the_callers_tensor():
    """Mutating the published tensor cannot change the stored signal."""
    bus = SignalBus(num_envs=2, device="cpu")
    value = torch.tensor([1.0, 2.0])

    bus.publish("x", value)
    value[0] = 99.0

    assert torch.equal(bus.get("x"), torch.tensor([1.0, 2.0]))


def test_get_and_snapshot_return_isolated_copies():
    """Read results cannot mutate the bus store."""
    bus = SignalBus(num_envs=2, device="cpu")
    bus.publish("a", torch.tensor([1.0, 2.0]))

    got = bus.get("a")
    assert got is not None
    got[0] = 7.0
    snap = dict(bus.snapshot())
    snap["a"][1] = 8.0
    snap["b"] = torch.zeros(2)

    assert torch.equal(bus.get("a"), torch.tensor([1.0, 2.0]))
    assert bus.get("b") is None


def test_missing_name_is_absent_not_a_zero_tensor():
    """A never-published signal answers None / absent."""
    bus = SignalBus(num_envs=2, device="cpu")

    assert bus.get("missing") is None
    assert dict(bus.snapshot()) == {}


def test_last_publish_wins():
    """A later publish replaces the stored value; nothing accumulates."""
    bus = SignalBus(num_envs=2, device="cpu")

    bus.publish("x", torch.tensor([1.0, 1.0]))
    bus.publish("x", torch.tensor([2.0, 2.0]))

    assert torch.equal(bus.get("x"), torch.tensor([2.0, 2.0]))


@pytest.mark.parametrize("name", ["", 3, None])
def test_publish_rejects_an_invalid_name(name):
    """Names must be non-empty strings."""
    bus = SignalBus(num_envs=2, device="cpu")

    with pytest.raises(ValueError, match="signal name"):
        bus.publish(name, torch.zeros(2))


def test_publish_rejects_a_non_tensor_value():
    """Only torch tensors are transported."""
    bus = SignalBus(num_envs=2, device="cpu")

    with pytest.raises(ValueError, match="torch.Tensor"):
        bus.publish("x", [0.0, 1.0])


@pytest.mark.parametrize("shape", [(), (3,), (0,)])
def test_publish_rejects_a_wrong_leading_dimension(shape):
    """The leading dimension must be num_envs."""
    bus = SignalBus(num_envs=2, device="cpu")

    with pytest.raises(ValueError, match="leading dimension"):
        bus.publish("x", torch.zeros(shape))


def test_publish_rejects_a_wrong_device():
    """Values must live on the bus device."""
    bus = SignalBus(num_envs=2, device="cpu")

    with pytest.raises(ValueError, match="device"):
        bus.publish("x", torch.zeros(2, device="meta"))


def test_constructor_rejects_an_invalid_num_envs():
    """The env count is validated at construction."""
    with pytest.raises(ValueError, match="num_envs"):
        SignalBus(num_envs=0, device="cpu")
