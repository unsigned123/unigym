"""Core environment step boundary: fields, validation and immutability."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields

import pytest
import torch

from unigym.evaluation import EnvStepContext, EnvView
from unigym.values import ValueStore


def _view(num_envs: int = 3, device: str = "cpu") -> EnvView:
    return EnvView(
        record={},
        obs=None,
        env_id=None,
        num_envs=num_envs,
        device=device,
        backend="fake",
    )


def _context(**overrides: object) -> EnvStepContext:
    num_envs = int(overrides.pop("num_envs", 3))
    device = str(overrides.pop("device", "cpu"))
    view = overrides.pop("view", None)
    values = overrides.pop("values", None)
    kwargs: dict[str, object] = {
        "view": _view(num_envs, device) if view is None else view,
        "values": (ValueStore(num_envs, device).snapshot() if values is None else values),
        "t": 0,
        "step_index": torch.zeros(num_envs, dtype=torch.int64),
        "time_s": None,
        "cmd": None,
        "reset_state": {},
    }
    kwargs.update(overrides)
    return EnvStepContext(**kwargs)  # type: ignore[arg-type]


def test_valid_vector_context_exposes_boundary_fields() -> None:
    """A vector view plus a value snapshot yields a complete context."""
    view = _view(num_envs=3)
    values = ValueStore(num_envs=3, device="cpu").snapshot()
    step_index = torch.arange(3, dtype=torch.int64)
    time_s = torch.arange(3, dtype=torch.float32)
    cmd = torch.zeros(3, 4)

    context = EnvStepContext(
        view=view,
        values=values,
        t=7,
        step_index=step_index,
        time_s=time_s,
        cmd=cmd,
        reset_state={0: {"done": True}, 2: [1, 2]},
    )

    assert context.view is view
    assert context.values is values
    assert context.t == 7
    assert context.step_index is step_index
    assert context.time_s is time_s
    assert context.cmd is cmd
    assert context.reset_state == {0: {"done": True}, 2: [1, 2]}


def test_num_envs_and_device_come_from_the_view() -> None:
    """``num_envs``/``device`` derive from the borrowed view."""
    context = _context(num_envs=4)

    assert context.num_envs == 4
    assert context.device == torch.device("cpu")
    assert isinstance(context.device, torch.device)


def test_non_tensor_view_is_rejected() -> None:
    """A view that is not an ``EnvView`` fails at construction."""
    with pytest.raises(TypeError, match="view"):
        _context(view="view")


def test_invalid_values_object_is_rejected() -> None:
    """A values object without the read-only value face is rejected."""
    with pytest.raises(TypeError, match="values"):
        _context(values=object())


@pytest.mark.parametrize("t", [-1, True, 1.5, "1", None])
def test_invalid_t_is_rejected(t: object) -> None:
    """``t`` must be a non-negative int, never bool/float/str/None."""
    with pytest.raises(TypeError, match="t"):
        _context(t=t)


@pytest.mark.parametrize(
    "step_index",
    [
        torch.zeros(4, dtype=torch.int64),
        torch.zeros(3, 1, dtype=torch.int64),
        torch.zeros(3, dtype=torch.float32),
        torch.zeros(3, dtype=torch.int64, device="meta"),
    ],
)
def test_invalid_step_index_is_rejected(step_index: torch.Tensor) -> None:
    """``step_index`` must be 1-D int32/int64 on the exact view device."""
    with pytest.raises((TypeError, ValueError), match="step_index"):
        _context(step_index=step_index)


def test_step_index_accepts_int32() -> None:
    """Both documented integer dtypes are accepted."""
    context = _context(step_index=torch.zeros(3, dtype=torch.int32))

    assert context.step_index.dtype == torch.int32


@pytest.mark.parametrize(
    "time_s",
    [
        torch.zeros(4, dtype=torch.float32),
        torch.zeros(3, 1, dtype=torch.float32),
        torch.zeros(3, dtype=torch.int64),
        torch.zeros(3, dtype=torch.float32, device="meta"),
    ],
)
def test_invalid_time_s_is_rejected(time_s: torch.Tensor) -> None:
    """``time_s`` must be a 1-D floating tensor on the view device."""
    with pytest.raises((TypeError, ValueError), match="time_s"):
        _context(time_s=time_s)


@pytest.mark.parametrize(
    "cmd",
    [
        torch.zeros(4, 2),
        torch.zeros(()),
        torch.zeros(3, 2, device="meta"),
    ],
)
def test_invalid_cmd_is_rejected(cmd: torch.Tensor) -> None:
    """``cmd`` keeps the env leading dimension and the view device."""
    with pytest.raises((TypeError, ValueError), match="cmd"):
        _context(cmd=cmd)


def test_cmd_allows_extra_trailing_dimensions() -> None:
    """``cmd`` may have any trailing shape as long as the leading axis is envs."""
    context = _context(cmd=torch.zeros(3, 2, 5))

    assert context.cmd is not None
    assert context.cmd.shape == (3, 2, 5)


@pytest.mark.parametrize("key", [True, "0", 1.0, None])
def test_reset_state_rejects_non_int_keys(key: object) -> None:
    """Reset keys must be real ints, not bools or other scalar types."""
    with pytest.raises(TypeError, match="reset_state"):
        _context(reset_state={key: {}})


@pytest.mark.parametrize("key", [-1, 3])
def test_reset_state_rejects_out_of_range_keys(key: int) -> None:
    """Reset keys must lie in ``[0, num_envs)``."""
    with pytest.raises(ValueError, match="reset_state"):
        _context(reset_state={key: {}})


@pytest.mark.parametrize("value", [torch.zeros(1), object(), {1, 2}, float("nan")])
def test_reset_state_rejects_non_json_values(value: object) -> None:
    """Reset values must be JSON-safe and finite."""
    with pytest.raises((TypeError, ValueError), match="reset_state"):
        _context(reset_state={0: value})


def test_reset_mapping_is_copied_and_read_only() -> None:
    """The context copies the caller's mapping and never aliases it."""
    source: dict[int, object] = {0: {"history": [1]}}
    context = _context(reset_state=source)

    source[0]["history"].append(2)
    source[1] = {"late": True}

    assert context.reset_state == {0: {"history": [1]}}
    assert context.reset_state[0] is not source[0]
    with pytest.raises(TypeError):
        context.reset_state[0] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        del context.reset_state[0]  # type: ignore[attr-defined]


def test_derived_context_is_frozen() -> None:
    """A constructed context rejects field mutation."""
    context = _context()

    with pytest.raises(FrozenInstanceError):
        context.t = 1  # type: ignore[misc]


def test_having_no_events_terminals_or_benchmark_vocabulary() -> None:
    """The boundary type owns no event/terminal/benchmark surface."""
    context = _context()

    for name in (
        "events",
        "terminals",
        "benchmark",
        "Termination",
        "BenchmarkStepContext",
        "TickFrame",
        "Probe",
    ):
        assert not hasattr(context, name)
        assert not hasattr(EnvStepContext, name)


def test_frozen_slots_subclass_can_add_a_defaulted_field() -> None:
    """Pack subclasses add defaulted fields without redefining the base."""

    @dataclass(frozen=True, slots=True)
    class PackStepContext(EnvStepContext):
        pack_score: float = 0.5

    base = _context()
    values = {field.name: getattr(base, field.name) for field in fields(EnvStepContext)}
    subclass = PackStepContext(**values)

    assert subclass.pack_score == 0.5
    assert subclass.num_envs == base.num_envs
    assert [field.name for field in fields(PackStepContext)][-1] == "pack_score"
