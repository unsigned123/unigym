"""Core-owned environment step boundary context.

An :class:`EnvStepContext` carries the read-only :class:`EnvView`, an immutable
:class:`~unigym.values.ValueView` snapshot of the run's managed values, tick and
episode position, and reset bookkeeping.  It is a data boundary only: it owns
no events, terminals, termination state, watcher state or pack vocabulary.

The view is borrowed: constructing a context never closes it, and the owner
keeps managing its lifetime.  The values are a boundary snapshot, not a live
store, so retaining the context never aliases mutable store state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch

from .._typing import JsonValue
from ..core.episode_termination import ensure_json_value
from ..values import ValueView
from .context import EnvView

__all__ = ["EnvStepContext"]


def _normalized_device(view: EnvView) -> torch.device:
    """Return the view's device normalized to ``torch.device``."""
    raw = view.device
    return raw if isinstance(raw, torch.device) else torch.device(raw)


def _require_tensor(
    value: object,
    *,
    field: str,
    num_envs: int,
    device: torch.device,
    floating: bool = False,
    exact_1d: bool = False,
) -> torch.Tensor:
    """Validate one per-env tensor field and return it unchanged."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"EnvStepContext.{field} must be a torch.Tensor, got {type(value).__name__}"
        )
    if exact_1d:
        if value.ndim != 1 or value.shape[0] != num_envs:
            raise ValueError(
                f"EnvStepContext.{field} must be 1-D with shape (num_envs={num_envs},), "
                f"got shape {tuple(value.shape)}"
            )
    elif value.ndim < 1 or value.shape[0] != num_envs:
        raise ValueError(
            f"EnvStepContext.{field} must have leading dimension num_envs={num_envs}, "
            f"got shape {tuple(value.shape)}"
        )
    if floating and not torch.is_floating_point(value):
        raise TypeError(
            f"EnvStepContext.{field} must be a floating tensor, got dtype {value.dtype}"
        )
    if value.device != device:
        raise ValueError(f"EnvStepContext.{field} must live on device {device}, got {value.device}")
    return value


@dataclass(frozen=True, slots=True)
class EnvStepContext:
    """Read-only boundary data for one environment step.

    Args:
        view: Borrowed read-only boundary view; the context never closes it.
        values: Immutable boundary snapshot of the run's managed values.
        t: Non-negative tick index.
        step_index: ``(num_envs,)`` int32/int64 tensor of per-env step counters.
        time_s: Optional ``(num_envs,)`` floating tensor of episode time.
        cmd: Optional per-env command tensor with leading dimension ``num_envs``.
        reset_state: Per-env JSON reset bookkeeping; stored read-only.

    The field set is intentionally minimal so packs can subclass with their
    own defaulted fields without redefining this boundary.
    """

    view: EnvView
    values: ValueView
    t: int
    step_index: torch.Tensor
    time_s: torch.Tensor | None
    cmd: torch.Tensor | None
    reset_state: Mapping[int, JsonValue]

    def __post_init__(self) -> None:
        view = self.view
        if not isinstance(view, EnvView):
            raise TypeError(f"EnvStepContext.view must be an EnvView, got {type(view).__name__}")
        num_envs = view.num_envs
        device = _normalized_device(view)

        if not isinstance(self.values, ValueView):
            raise TypeError(
                "EnvStepContext.values must satisfy the ValueView protocol, "
                f"got {type(self.values).__name__}"
            )

        if isinstance(self.t, bool) or not isinstance(self.t, int) or self.t < 0:
            raise TypeError(f"EnvStepContext.t must be a non-negative int, got {self.t!r}")

        if not isinstance(self.step_index, torch.Tensor):
            raise TypeError(
                "EnvStepContext.step_index must be a torch.Tensor, "
                f"got {type(self.step_index).__name__}"
            )
        if self.step_index.ndim != 1 or self.step_index.shape[0] != num_envs:
            raise ValueError(
                "EnvStepContext.step_index must be 1-D with shape "
                f"(num_envs={num_envs},), got shape {tuple(self.step_index.shape)}"
            )
        if self.step_index.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "EnvStepContext.step_index must have dtype torch.int32 or torch.int64, "
                f"got {self.step_index.dtype}"
            )
        if self.step_index.device != device:
            raise ValueError(
                f"EnvStepContext.step_index must live on device {device}, "
                f"got {self.step_index.device}"
            )

        if self.time_s is not None:
            _require_tensor(
                self.time_s,
                field="time_s",
                num_envs=num_envs,
                device=device,
                floating=True,
                exact_1d=True,
            )

        if self.cmd is not None:
            _require_tensor(
                self.cmd,
                field="cmd",
                num_envs=num_envs,
                device=device,
            )

        if not isinstance(self.reset_state, Mapping):
            raise TypeError(
                "EnvStepContext.reset_state must be a mapping, "
                f"got {type(self.reset_state).__name__}"
            )
        normalized: dict[int, JsonValue] = {}
        for key, value in self.reset_state.items():
            if isinstance(key, bool) or not isinstance(key, int):
                raise TypeError(
                    "EnvStepContext.reset_state keys must be int env ids, " f"got {key!r}"
                )
            if key < 0 or key >= num_envs:
                raise ValueError(
                    f"EnvStepContext.reset_state key {key} is outside " f"[0, num_envs={num_envs})"
                )
            normalized[key] = ensure_json_value(value, where=f"EnvStepContext.reset_state[{key}]")
        object.__setattr__(self, "reset_state", MappingProxyType(normalized))

    @property
    def num_envs(self) -> int:
        """The whole run's environment count, from the borrowed view."""
        return self.view.num_envs

    @property
    def device(self) -> torch.device:
        """The view's device, normalized to ``torch.device``."""
        return _normalized_device(self.view)
