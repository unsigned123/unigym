"""Run-scoped managed value store and immutable boundary snapshot.

A :class:`ValueStore` carries arbitrary per-env tensors and JSON values
published by user code or sinks.  Core owns transport, ownership and boundary
validation only: it never interprets a value name, dtype, trailing shape or
JSON structure.

Ownership is first-writer-wins.  The first owner that publishes a name claims
it for the life of the store; another owner can never silently take the name
over, even after the value is cleared.  Clearing removes values, not claims.

Values are copied on the way in and on every read.  Tensor values are
detached clones; JSON values are deep copies.  A missing name is absent
(``None`` / ``False``), never a silent zero.

The store is instance-owned and single-threaded: publish, read, clear and
rewind all run on the owning thread (the engine thread in a live run).
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from typing import Protocol, TypeAlias, runtime_checkable

import torch

from unigym._typing import JsonValue
from unigym.core.episode_termination import ensure_json_value

__all__ = [
    "Value",
    "ValueView",
    "ValuePublisher",
    "StepValues",
    "ValueStore",
    "ValueOwnershipError",
    "VALUE_REWIND_SCHEMA",
]

#: Rewind payload schema owned by this module.  Schema 2 is the
#: ``{schema, values, writers}`` mapping; the old flat signal-style payload
#: (``{name: value}``) is deliberately rejected.
VALUE_REWIND_SCHEMA = 2

#: A managed value is either a per-env tensor or a JSON-safe tree.
Value: TypeAlias = torch.Tensor | JsonValue


class ValueOwnershipError(ValueError):
    """A value name is already claimed by a different owner.

    Claims are never reassigned.  The message names the value and both the
    requesting owner and the claiming owner so the conflict is actionable.
    """

    def __init__(self, name: str, owner: str, claimed_by: str):
        super().__init__(
            f"value {name!r} is already owned by {claimed_by!r}; "
            f"{owner!r} cannot publish or clear it (claims are not reassigned)"
        )
        self.name = name
        self.owner = owner
        self.claimed_by = claimed_by


@runtime_checkable
class ValueView(Protocol):
    """Read-only view over run-scoped managed values."""

    def get(self, name: str) -> Value | None:
        """Return an isolated copy of *name*, or ``None`` when absent."""
        ...

    def snapshot(self) -> Mapping[str, Value]:
        """Return an isolated mapping of every current value."""
        ...

    def __contains__(self, name: object) -> bool:
        """Return whether *name* is present."""
        ...


@runtime_checkable
class ValuePublisher(Protocol):
    """Write-only value face bound to one owner."""

    def publish(self, name: str, value: Value) -> None:
        """Store *value* under *name* for the bound owner."""
        ...

    def clear(self, name: str) -> None:
        """Remove the bound owner's value under *name* (the claim remains)."""
        ...


def _copy_value(value: Value) -> Value:
    """Return a detached/deep private copy of one managed value."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return copy.deepcopy(value)


class StepValues(Mapping[str, Value]):
    """Immutable snapshot of a :class:`ValueStore` at one step boundary.

    The mapping has no mutation API.  Constructing it copies the source
    values, and every read (``[]``, ``get``, iteration through the mapping
    mixins, ``snapshot``) hands out a fresh isolated copy, so a retained
    snapshot can be read long after the producing callback returned without
    aliasing live state.
    """

    def __init__(self, values: Mapping[str, Value]):
        self._values: dict[str, Value] = {
            name: _copy_value(value) for name, value in values.items()
        }

    def __getitem__(self, name: str) -> Value:
        return _copy_value(self._values[name])

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def snapshot(self) -> Mapping[str, Value]:
        """Return an isolated mapping of this snapshot's values."""
        return {name: _copy_value(value) for name, value in self._values.items()}


class _BoundValuePublisher:
    """Write-only publisher bound to one owner of a :class:`ValueStore`."""

    def __init__(self, store: ValueStore, owner: str):
        self._store = store
        self._owner = owner

    def publish(self, name: str, value: Value) -> None:
        self._store.publish(self._owner, name, value)

    def clear(self, name: str) -> None:
        self._store.clear(self._owner, name)


class ValueStore:
    """One run's managed values, shared by publishers and boundary readers.

    Args:
        num_envs: Number of envs; every tensor value's leading dimension must
            match it.
        device: Tensor device every tensor value must live on.

    The store is instance-owned and single-threaded.  Values persist until
    overwritten or cleared; ownership claims persist for the store's lifetime.
    """

    name = "values"

    def __init__(self, num_envs: int, device: str | torch.device):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError(f"ValueStore num_envs must be a positive int, got {num_envs!r}")
        self.num_envs = num_envs
        self.device = torch.device(device)
        self._values: dict[str, Value] = {}
        self._writers: dict[str, str] = {}

    def publisher(self, owner: str) -> ValuePublisher:
        """Return a write-only publisher bound to *owner*."""
        self._require_owner(owner)
        return _BoundValuePublisher(self, owner)

    def publish(self, owner: str, name: str, value: Value) -> None:
        """Store *value* under *name* on behalf of *owner* (last write wins).

        Raises:
            ValueOwnershipError: *name* is already claimed by another owner.
            ValueError: *owner*/*name* is not a non-empty string, or *value*
                violates the tensor/JSON transport contract.
            TypeError: *value* is not JSON-safe.
        """
        self._require_owner(owner)
        self._require_name(name)
        writer = self._writers.get(name)
        if writer is not None and writer != owner:
            raise ValueOwnershipError(name, owner, writer)
        stored = self._checked_value(name, value)
        self._writers[name] = owner
        self._values[name] = stored

    def get(self, name: str) -> Value | None:
        """Return an isolated copy of *name*, or ``None`` when absent."""
        value = self._values.get(name)
        return None if value is None else _copy_value(value)

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def snapshot(self) -> StepValues:
        """Return an immutable snapshot with isolated copies of all values."""
        return StepValues(self._values)

    def clear(self, owner: str, name: str | None = None) -> None:
        """Remove *owner*'s values while retaining ownership claims.

        With *name*, remove that one value; with ``None``, remove every value
        currently owned by *owner*.  Claims are never released.

        Raises:
            ValueOwnershipError: *name* is claimed by another owner.
            ValueError: *owner*/*name* is not a non-empty string.
        """
        self._require_owner(owner)
        if name is None:
            owned = [key for key in self._values if self._writers.get(key) == owner]
            for key in owned:
                del self._values[key]
            return
        self._require_name(name)
        writer = self._writers.get(name)
        if writer is not None and writer != owner:
            raise ValueOwnershipError(name, owner, writer)
        self._values.pop(name, None)

    # ------------------------------------------------------------------ #
    # RewindStateProvider shape (name = "values")
    # ------------------------------------------------------------------ #
    def rewind_state(self) -> Mapping[str, object]:
        """Return the schema-2 checkpoint payload with private copies."""
        return {
            "schema": VALUE_REWIND_SCHEMA,
            "values": {name: _copy_value(value) for name, value in self._values.items()},
            "writers": dict(self._writers),
        }

    def validate_rewind_state(self, state: Mapping[str, object]) -> None:
        """Validate a schema-2 payload read-only before any mutation.

        Raises:
            RewindGateError: *state* is not a schema-2 values payload or a
                name/value/writer pair violates the transport contract.
        """
        self._checked_payload(state)

    def apply_rewind_state(self, state: Mapping[str, object]) -> None:
        """Replace values and writers atomically from a checkpoint payload.

        The payload is validated fully first; a refused restore leaves every
        value and ownership claim unchanged.
        """
        values, writers = self._checked_payload(state)
        self._values = values
        self._writers = writers

    def _checked_payload(
        self, state: Mapping[str, object]
    ) -> tuple[dict[str, Value], dict[str, str]]:
        """Validate *state* and return copied values plus writers."""
        from unigym.runtime.rewind import RewindGateError

        if not isinstance(state, Mapping):
            raise RewindGateError(
                f"values rewind state must be a mapping, got {type(state).__name__}"
            )
        required = {"schema", "values", "writers"}
        keys = set(state.keys())
        if keys != required:
            if "schema" not in keys:
                raise RewindGateError(
                    "values rewind state must be the schema-2 payload with keys "
                    f"{sorted(required)}; got {sorted(keys, key=repr)} (old flat "
                    "signal-style payloads are not supported)"
                )
            raise RewindGateError(
                f"values rewind state keys must be exactly {sorted(required)}, "
                f"got {sorted(keys, key=repr)}"
            )
        schema = state["schema"]
        if isinstance(schema, bool) or not isinstance(schema, int) or schema != VALUE_REWIND_SCHEMA:
            raise RewindGateError(
                f"values rewind state schema must be {VALUE_REWIND_SCHEMA}, got {schema!r}"
            )
        raw_values = state["values"]
        raw_writers = state["writers"]
        if not isinstance(raw_values, Mapping):
            raise RewindGateError(
                f"values rewind state 'values' must be a mapping, got "
                f"{type(raw_values).__name__}"
            )
        if not isinstance(raw_writers, Mapping):
            raise RewindGateError(
                f"values rewind state 'writers' must be a mapping, got "
                f"{type(raw_writers).__name__}"
            )
        writers: dict[str, str] = {}
        for name, owner in raw_writers.items():
            if not isinstance(name, str) or not name:
                raise RewindGateError(
                    f"values rewind state writer name must be a non-empty str, got {name!r}"
                )
            if not isinstance(owner, str) or not owner:
                raise RewindGateError(
                    f"values rewind state writer for {name!r} must be a non-empty "
                    f"str, got {owner!r}"
                )
            writers[name] = owner
        values: dict[str, Value] = {}
        for name, value in raw_values.items():
            if not isinstance(name, str) or not name:
                raise RewindGateError(
                    f"values rewind state value name must be a non-empty str, got {name!r}"
                )
            if name not in writers:
                raise RewindGateError(f"values rewind state value {name!r} has no writer")
            try:
                values[name] = self._checked_value(name, value)
            except (TypeError, ValueError) as err:
                raise RewindGateError(f"values rewind state: {err}") from err
        return values, writers

    def _checked_value(self, name: str, value: object) -> Value:
        """Validate and return a private copy of one transported value."""
        if isinstance(value, torch.Tensor):
            if value.layout != torch.strided:
                raise ValueError(
                    f"value {name!r} must be a dense tensor, got layout {value.layout}"
                )
            if value.ndim < 1 or int(value.shape[0]) != self.num_envs:
                raise ValueError(
                    f"value {name!r} tensor must have leading dimension "
                    f"num_envs={self.num_envs}, got shape {tuple(value.shape)}"
                )
            if value.device != self.device:
                raise ValueError(
                    f"value {name!r} tensor device {value.device} does not match "
                    f"store device {self.device}"
                )
            return value.detach().clone()
        return ensure_json_value(value, where=f"value {name!r}")

    @staticmethod
    def _require_owner(owner: object) -> None:
        if not isinstance(owner, str) or not owner:
            raise ValueError(f"value owner must be a non-empty str, got {owner!r}")

    @staticmethod
    def _require_name(name: object) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"value name must be a non-empty str, got {name!r}")
