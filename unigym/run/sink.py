"""Construction-side data contract for a custom sink.

A run may declare one custom :class:`unigym.sink.Sink` by config
(``sink: {factory: ..., options: ...}``) or by API
(``Run(..., sink=SinkSpec(...))``).  This module owns only the inert
declaration and its boundary validation: it never imports the factory target
eagerly and never invokes the factory.  Building the sink with a
:class:`SinkContext` belongs to the run builder.

The module stays import-light: it depends on the public sink protocol, the
shared JSON typing alias and the config error type, plus the standard library.
Construction-time types (``RunSpec``, ``RunIdentity``, ``OrderedActionGroup``)
are imported for typing only.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, cast

from unigym._typing import JsonValue
from unigym.errors import ConfigError
from unigym.sink import Sink

if TYPE_CHECKING:
    from unigym.benchmarks.episode import RunIdentity
    from unigym.run.spec import RunSpec
    from unigym.runtime.actions import OrderedActionGroup

__all__ = ["SinkContext", "SinkFactory", "SinkSpec", "resolve_sink_factory"]

#: ``module:attribute`` where both sides are dotted Python identifiers.
_FACTORY_PATH = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


def _empty_options() -> dict[str, JsonValue]:
    """Typed empty default for :attr:`SinkSpec.options`."""
    return {}


@dataclass(frozen=True)
class SinkSpec:
    """One custom sink declaration (config or API).

    ``factory`` is either an import path (``"pkg.mod:make_sink"``; config and
    API) or a callable (API only).  ``options`` travels unchanged to
    :attr:`SinkContext.options`; it is copied into a read-only mapping so a
    caller mutating its declaration after handing it over cannot change the
    run's intent.
    """

    factory: str | Callable[[SinkContext], Sink]
    options: Mapping[str, JsonValue] = field(default_factory=_empty_options)

    def __post_init__(self) -> None:
        factory = self.factory
        if isinstance(factory, str):
            if not factory:
                raise ConfigError(
                    "sink.factory must be a non-empty 'module:callable' string or a callable"
                )
            if _FACTORY_PATH.fullmatch(factory) is None:
                raise ConfigError(
                    f"sink.factory={factory!r} must be 'module:callable' "
                    "(dotted identifiers only)"
                )
        elif not callable(factory):
            raise ConfigError(
                f"sink.factory={factory!r} must be a non-empty 'module:callable' "
                "string or a callable"
            )
        options = self.options
        if not isinstance(options, Mapping):
            raise ConfigError(
                f"sink.options must be a string-keyed mapping, got {type(options).__name__}"
            )
        for key in options:
            if not isinstance(key, str):
                raise ConfigError(f"sink.options must be a string-keyed mapping, got key {key!r}")
        object.__setattr__(self, "options", MappingProxyType(dict(options)))


@dataclass(frozen=True)
class SinkContext:
    """Build-time context handed to a custom sink factory.

    It carries run metadata only.  There is deliberately no live env, writer,
    logger, bridge or signal bus here: those are runner-owned resources a sink
    receives through its callback bundle, never through construction.
    """

    spec: RunSpec
    mode: str
    out_dir: str
    num_envs: int
    policy_dt: float
    seed: int
    action_groups: tuple[OrderedActionGroup, ...]
    run_identity: RunIdentity | None
    options: Mapping[str, JsonValue]


#: The one factory shape: build a sink from its construction context.
SinkFactory: TypeAlias = Callable[[SinkContext], Sink]


def resolve_sink_factory(spec: SinkSpec, config_path: str) -> SinkFactory:
    """Resolve a validated :class:`SinkSpec` to its factory callable.

    A callable factory is returned unchanged.  A string factory is imported
    once (``module:attribute``, dotted attribute allowed) and must resolve to
    a callable.  Importing the target executes trusted config code; this is
    the same trust level as adapter and pack selection.  The factory itself is
    never called here.
    """
    factory = spec.factory
    if callable(factory):
        return factory
    module_name, _, attribute_path = factory.partition(":")
    try:
        target: object = importlib.import_module(module_name)
    except Exception as err:
        raise ConfigError(
            f"{config_path}: sink.factory={factory!r} could not be imported: {err} "
            "| hint: the factory path is 'module:callable' and the module must be "
            "importable from the run process"
        ) from err
    for attribute in attribute_path.split("."):
        try:
            target = getattr(target, attribute)
        except AttributeError as err:
            raise ConfigError(
                f"{config_path}: sink.factory={factory!r} could not be imported: "
                f"module {module_name!r} has no attribute {attribute!r}"
            ) from err
    if not callable(target):
        raise ConfigError(f"{config_path}: sink.factory={factory!r} is not callable")
    # The runtime callable check above is the only guarantee available here:
    # a factory's parameter/return contract is structural, not introspectable.
    return cast(SinkFactory, target)
