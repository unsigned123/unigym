"""Core-neutral episode termination facts.

``Termination`` is a CORE fact object.  It deliberately knows nothing about
``Task`` or Gymnasium's ``terminated`` / ``truncated`` vocabulary -- the RL
task layer maps outcomes onto that layer later.  Core owns:

* the outcome/source vocabulary (``EpisodeOutcome`` / ``TerminationSource``);
* namespaced, stable ``code`` strings (``env.force_limit``,
  ``initializer.invalid_reset``, ``runtime.episode_horizon`` ...);
* JSON-safe, namespaced, versioned extension payloads;
* the primary/all precedence (``INVALID_RESET > SUCCESS > FAILURE >
  DEADLINE > ENGINE > MANUAL``).

This module is numpy-only and importable without torch/isaaclab.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from unigym._typing import JsonValue

__all__ = [
    "EpisodeOutcome",
    "TerminationSource",
    "TerminationExtension",
    "Termination",
    "EpisodeTermination",
    "OUTCOME_PRIORITY",
    "validate_code",
    "ensure_json_value",
]


class EpisodeOutcome(StrEnum):
    """Core-neutral terminal outcome of one episode."""

    SUCCESS = "success"
    FAILURE = "failure"
    DEADLINE = "deadline"
    INVALID_RESET = "invalid_reset"
    ENGINE = "engine"
    MANUAL = "manual"


class TerminationSource(StrEnum):
    """Who observed/decided the termination (provenance, not severity)."""

    ENVIRONMENT = "environment"
    INITIALIZER = "initializer"
    RUNTIME = "runtime"
    CUSTOM = "custom"


#: Core precedence.  Lower rank wins; packs may provide a condition but may
#: never reorder this across outcomes.
OUTCOME_PRIORITY: dict[EpisodeOutcome, int] = {
    EpisodeOutcome.INVALID_RESET: 0,
    EpisodeOutcome.SUCCESS: 1,
    EpisodeOutcome.FAILURE: 2,
    EpisodeOutcome.DEADLINE: 3,
    EpisodeOutcome.ENGINE: 4,
    EpisodeOutcome.MANUAL: 5,
}

_CODE_NAMESPACES = frozenset(
    {"env", "initializer", "runtime", "engine", "manual", "task", "custom"}
)


def validate_code(code: str) -> str:
    """Validate a namespaced, stable termination code.

    The code is ``<namespace>.<stable_name>`` where each segment is
    lowercase snake_case (``env.force_limit``).  Unknown namespaces are
    rejected so report aggregation cannot silently invent schemas.
    """
    if not isinstance(code, str) or not code:
        raise ValueError(f"termination code must be a non-empty string, got {code!r}")
    namespace, sep, name = code.partition(".")
    if not sep or not name:
        raise ValueError(
            f"termination code {code!r} must be namespaced as "
            f"'<namespace>.<stable_name>' (e.g. 'env.force_limit')"
        )
    if namespace not in _CODE_NAMESPACES:
        raise ValueError(
            f"termination code {code!r}: unknown namespace {namespace!r} "
            f"(allowed: {sorted(_CODE_NAMESPACES)})"
        )
    for segment in code.split("."):
        if not segment or not all(c in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in segment):
            raise ValueError(
                f"termination code {code!r} must be lowercase snake_case "
                f"dot-separated (bad segment {segment!r})"
            )
    return code


def ensure_json_value(value: object, *, where: str = "extension data") -> JsonValue:
    """Validate/normalize a JSON-safe value (rejects tensors/live handles).

    Accepts ``None``/bool/int/str/finite float, lists/tuples and string-keyed
    mappings.  Arrays, tensors and arbitrary objects are refused at
    construction time -- reports are JSON artifacts, not object graphs.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where}: float {value!r} is not finite")
        return float(value)
    if isinstance(value, (list, tuple)):
        return [ensure_json_value(item, where=where) for item in value]
    if isinstance(value, Mapping):
        out: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where}: JSON object keys must be strings, got {key!r}")
            out[str(key)] = ensure_json_value(item, where=where)
        return out
    raise TypeError(
        f"{where}: {type(value).__name__} is not JSON-safe; extensions and "
        "report payloads may contain only JSON scalars, lists and mappings"
    )


def _validate_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(metrics, Mapping):
        raise TypeError(f"termination metrics must be a mapping, got {type(metrics).__name__}")
    for key, value in metrics.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"termination metric names must be non-empty strings, got {key!r}")
        try:
            number = float(value)
        except (TypeError, ValueError) as err:
            raise TypeError(
                f"termination metric {key!r} must be a scalar float, got {value!r}"
            ) from err
        if not math.isfinite(number):
            raise ValueError(f"termination metric {key!r} is not finite ({number!r})")
        out[key] = number
    return out


@dataclass(frozen=True)
class TerminationExtension:
    """One namespaced, versioned, JSON-safe extension payload.

    Core aggregation ignores extension contents (no cross-namespace merge);
    the per-episode document preserves them verbatim.
    """

    namespace: str
    schema_version: int
    data: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.namespace, str)
            or not self.namespace
            or not all(c in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in self.namespace)
        ):
            raise ValueError(
                f"termination extension namespace {self.namespace!r} must be a "
                "non-empty lowercase identifier"
            )
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise ValueError(
                f"termination extension {self.namespace!r}: schema_version must "
                f"be an int >= 1, got {self.schema_version!r}"
            )
        if not isinstance(self.data, Mapping):
            raise TypeError(f"termination extension {self.namespace!r}: data must be a mapping")
        normalized = ensure_json_value(dict(self.data), where=f"extension {self.namespace!r}")
        object.__setattr__(self, "data", normalized)

    def to_json(self) -> dict[str, JsonValue]:
        """JSON-safe extension document."""
        return {
            "namespace": self.namespace,
            "schema_version": int(self.schema_version),
            "data": dict(self.data),
        }


@dataclass(frozen=True)
class Termination:
    """One reason an episode ended (core fact, not a Task/Gym object)."""

    outcome: EpisodeOutcome
    code: str
    source: TerminationSource
    step_index: int | None = None
    time_s: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    extensions: Mapping[str, TerminationExtension] = field(default_factory=dict)
    priority: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.outcome, str) and not isinstance(self.outcome, EpisodeOutcome):
            try:
                object.__setattr__(self, "outcome", EpisodeOutcome(self.outcome))
            except ValueError as err:
                raise ValueError(
                    f"unknown episode outcome {self.outcome!r}; "
                    f"allowed: {[o.value for o in EpisodeOutcome]}"
                ) from err
        if not isinstance(self.outcome, EpisodeOutcome):
            raise TypeError(f"outcome must be an EpisodeOutcome, got {self.outcome!r}")
        validate_code(self.code)
        if isinstance(self.source, str) and not isinstance(self.source, TerminationSource):
            try:
                object.__setattr__(self, "source", TerminationSource(self.source))
            except ValueError as err:
                raise ValueError(
                    f"unknown termination source {self.source!r}; "
                    f"allowed: {[s.value for s in TerminationSource]}"
                ) from err
        if not isinstance(self.source, TerminationSource):
            raise TypeError(f"source must be a TerminationSource, got {self.source!r}")
        if self.step_index is not None:
            if not isinstance(self.step_index, int) or self.step_index < 0:
                raise ValueError(
                    f"termination step_index must be an int >= 0 or None "
                    f"(0 = reset tick), got {self.step_index!r}"
                )
        if self.time_s is not None:
            value = float(self.time_s)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"termination time_s must be finite and >= 0, got {value!r}")
            object.__setattr__(self, "time_s", value)
        object.__setattr__(self, "metrics", _validate_metrics(self.metrics))
        extensions: dict[str, TerminationExtension] = {}
        for key, extension in dict(self.extensions).items():
            if not isinstance(extension, TerminationExtension):
                raise TypeError(
                    f"termination extension {key!r} must be a TerminationExtension, "
                    f"got {type(extension).__name__}"
                )
            if key != extension.namespace:
                raise ValueError(
                    f"termination extension key {key!r} must equal its "
                    f"namespace {extension.namespace!r}"
                )
            extensions[key] = extension
        object.__setattr__(self, "extensions", extensions)
        if not isinstance(self.priority, int):
            raise TypeError(f"termination priority must be an int, got {self.priority!r}")

    # ---- derived / serialization ---------------------------------------- #
    def rank(self) -> int:
        """Core precedence rank (lower wins); ``priority`` cannot reorder it."""
        return OUTCOME_PRIORITY[self.outcome]

    def to_json(self) -> dict[str, JsonValue]:
        """JSON-safe termination document."""
        return {
            "outcome": self.outcome.value,
            "code": self.code,
            "source": self.source.value,
            "step_index": self.step_index,
            "time_s": self.time_s,
            "metrics": dict(self.metrics),
            "extensions": {key: ext.to_json() for key, ext in self.extensions.items()},
            "priority": int(self.priority),
        }


@dataclass(frozen=True)
class EpisodeTermination:
    """The episode's primary termination + every terminating reason."""

    primary: Termination
    all: tuple[Termination, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.primary, Termination):
            raise TypeError(f"primary must be a Termination, got {type(self.primary).__name__}")
        terms = tuple(self.all) if self.all else (self.primary,)
        if not all(isinstance(t, Termination) for t in terms):
            raise TypeError("all must contain only Termination objects")
        if self.primary not in terms:
            raise ValueError(
                "EpisodeTermination.primary must be present in all "
                f"(primary code {self.primary.code!r}, all codes "
                f"{[t.code for t in terms]})"
            )
        object.__setattr__(self, "all", terms)

    @classmethod
    def from_terminations(cls, terminations: Iterable[Termination]) -> EpisodeTermination:
        """Build an ``EpisodeTermination`` with core precedence applied.

        ``priority`` only breaks ties WITHIN one outcome; the core outcome
        rank always wins first.  The first declared termination wins a full
        tie.
        """
        terms = tuple(terminations)
        if not terms:
            raise ValueError("EpisodeTermination.from_terminations requires at least one term")
        indexed = list(enumerate(terms))
        _, primary = min(indexed, key=lambda pair: (pair[1].rank(), -pair[1].priority, pair[0]))
        return cls(primary=primary, all=terms)

    def to_json(self) -> dict[str, JsonValue]:
        """JSON-safe primary/all document."""
        return {
            "primary": self.primary.to_json(),
            "all": [term.to_json() for term in self.all],
        }


def outcome_counts(terminations: Iterable[Termination | None]) -> dict[str, int]:
    """Count terminal outcomes (``None`` entries are ignored)."""
    counts: dict[str, int] = {}
    for term in terminations:
        if term is None:
            continue
        key = term.outcome.value
        counts[key] = counts.get(key, 0) + 1
    return counts


def code_counts(terminations: Iterable[Termination | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for term in terminations:
        if term is None:
            continue
        counts[term.code] = counts.get(term.code, 0) + 1
    return counts


def source_counts(terminations: Iterable[Termination | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for term in terminations:
        if term is None:
            continue
        key = term.source.value
        counts[key] = counts.get(key, 0) + 1
    return counts
