"""Run-scoped generic signal channel (transport only).

A :class:`SignalBus` carries arbitrary tensors published by user code or
tests to a run's :class:`~unigym.commands.base.CommandSampler`.  Core owns
transport and boundary validation only: it never interprets a signal name,
dtype, trailing shape, phase or event.  Every value is per-env -- the leading
axis is ``num_envs``.  A missing name is absent, never a silent zero tensor.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import torch

__all__ = ["SignalBus", "SignalPublisher", "SignalView"]


class SignalPublisher(Protocol):
    """Write-only signal face handed to a custom sink at construction.

    A publisher is the closed-loop seam: a sink stores it and calls
    :meth:`publish` from its runner callbacks.  It deliberately exposes no read
    method; reading belongs to a sampler bound through
    :class:`~unigym.signals.SignalView`.  :class:`SignalBus` satisfies this
    protocol structurally, so the run's single bus is the one object behind the
    sink publisher, the sampler view and the rewind provider.
    """

    def publish(self, name: str, value: torch.Tensor) -> None:
        """Store *value* under *name* (transport rules are the bus's)."""
        ...


class SignalView(Protocol):
    """Read-only signal view bound to a command sampler at build."""

    def get(self, name: str) -> torch.Tensor | None:
        """Return the last published value for *name*, or ``None``."""
        ...

    def snapshot(self) -> Mapping[str, torch.Tensor]:
        """Return an isolated mapping of every currently published signal."""
        ...


class SignalBus:
    """One run's signal store, shared by producers and the sampler view.

    The bus is instance-owned and lives for the whole run: values persist
    until overwritten by a later :meth:`publish`.  There is no automatic
    clear or reset; the caller owns each signal's lifetime.  Values are
    detached and cloned on the way in and on every read, so neither the
    publisher's original tensor nor a read result can mutate the store.

    Args:
        num_envs: Number of envs; every signal's leading dimension must
            match it.
        device: Tensor device every published value must live on.
    """

    name = "signals"

    def __init__(self, num_envs: int, device: str | torch.device):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError(f"SignalBus num_envs must be a positive int, got {num_envs!r}")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._signals: dict[str, torch.Tensor] = {}

    def publish(self, name: str, value: torch.Tensor) -> None:
        """Store *value* under *name* (last publish wins).

        Raises:
            ValueError: *name* is not a non-empty string, *value* is not a
                tensor, its leading dimension is not ``num_envs``, or it
                lives on another device.
        """
        error = self._invalid_value(name, value)
        if error is not None:
            raise ValueError(error)
        self._signals[name] = value.detach().clone()

    def get(self, name: str) -> torch.Tensor | None:
        """Return an isolated copy of *name*, or ``None`` when absent."""
        value = self._signals.get(name)
        return None if value is None else value.detach().clone()

    def snapshot(self) -> Mapping[str, torch.Tensor]:
        """Return an isolated copy of every published signal."""
        return {name: value.detach().clone() for name, value in self._signals.items()}

    # ------------------------------------------------------------------ #
    # RewindStateProvider shape (name = "signals")
    # ------------------------------------------------------------------ #
    def rewind_state(self) -> dict[str, object]:
        """Return the bus state at a checkpoint (detached private copies)."""
        return {name: value.detach().clone() for name, value in self._signals.items()}

    def apply_rewind_state(self, state: dict[str, object]) -> None:
        """Replace the bus store with a state returned by :meth:`rewind_state`."""
        self._signals = {
            name: value.detach().clone() for name, value in self._checked_state(state).items()
        }

    def validate_rewind_state(self, state: dict[str, object]) -> None:
        """Pre-flight a signal payload read-only.

        The same judgment :meth:`apply_rewind_state` enforces, so a refused
        rewind is a zero-change rejection.

        Raises:
            RewindGateError: *state* is not a signal mapping, or a name/value
                violates the transport contract.
        """
        self._checked_state(state)

    def _checked_state(self, state: dict[str, object]) -> dict[str, torch.Tensor]:
        """Validate *state* and return the narrowed name -> tensor mapping."""
        from unigym.runtime.rewind import RewindGateError

        if not isinstance(state, Mapping):
            raise RewindGateError(
                f"signals rewind state must be a mapping, got {type(state).__name__}"
            )
        checked: dict[str, torch.Tensor] = {}
        for name, value in state.items():
            error = self._invalid_value(name, value)
            if error is not None:
                raise RewindGateError(f"signals rewind state: {error}")
            assert isinstance(name, str) and isinstance(value, torch.Tensor)
            checked[name] = value
        return checked

    def _invalid_value(self, name: object, value: object) -> str | None:
        """Return the transport violation for (*name*, *value*), or ``None``."""
        if not isinstance(name, str) or not name:
            return f"signal name must be a non-empty str, got {name!r}"
        if not isinstance(value, torch.Tensor):
            return f"signal {name!r} value must be a torch.Tensor, got {type(value).__name__}"
        if value.ndim < 1 or int(value.shape[0]) != self.num_envs:
            return (
                f"signal {name!r} value must have leading dimension "
                f"num_envs={self.num_envs}, got shape {tuple(value.shape)}"
            )
        if value.device != self.device:
            return (
                f"signal {name!r} value device {value.device} does not match "
                f"bus device {self.device}"
            )
        return None
