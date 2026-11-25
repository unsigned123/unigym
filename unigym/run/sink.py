"""Construction-side contract for run sinks.

A run selects sinks explicitly with ``sinks:`` (config) or ``Run(sinks=...)``
(API): an ordered list of :class:`CollectSinkSpec` (the core built-in
``collect`` kind) and/or :class:`SinkSpec` (a custom factory).  The singular
``sink:`` declaration and ``Run(sink=...)`` remain the one-custom-sink
shorthand.  This module owns the inert declarations and their boundary
validation (no eager factory import), plus the resolved-factory helpers:
:func:`resolve_sink_factory` imports a configured path once without calling it,
and :func:`build_custom_sink` invokes the resolved factory with a
:class:`SinkContext` and validates the returned sink's callback shape.

``CollectSinkSpec`` carries no construction resources: the run build owns the
built-in :class:`unigym.runner.sinks.CollectSink` and its writer/logger/
recorder/control facts.  Those never enter :class:`SinkContext`.

A custom sink stays a four-callback :class:`unigym.sink.Sink`.  Optional
capabilities (``on_rewind``, ``stop_when``, ``drain_terminations``) may be
declared but add no required callback; a declared capability that is not
callable fails at construction.  ``drain_terminations`` is the
:class:`unigym.sink.TerminalSink` capability, not a fifth callback.

The module stays import-light: it depends on the public sink protocol, the
shared JSON typing alias and the config error type, plus the standard library.
Construction-time types (``RunSpec``, ``RunIdentity``, ``OrderedActionGroup``)
are imported for typing only.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, cast

from unigym._typing import JsonValue
from unigym.errors import ConfigError
from unigym.sink import Sink

if TYPE_CHECKING:
    from unigym.core.identity import RunIdentity
    from unigym.run.spec import RunSpec
    from unigym.runtime.actions import OrderedActionGroup
    from unigym.values import ValuePublisher

__all__ = [
    "CollectSinkSpec",
    "SinkContext",
    "SinkDeclaration",
    "SinkFactory",
    "SinkSpec",
    "build_custom_sink",
    "effective_sink_name",
    "resolve_sink_factory",
    "validate_sink_declarations",
]

#: ``module:attribute`` where both sides are dotted Python identifiers.
_FACTORY_PATH = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")

#: The four non-optional runner callbacks every custom sink must implement.
_REQUIRED_CALLBACKS = ("on_step", "on_episode_end", "on_step_end", "on_run_end")
#: Optional capabilities: absent is fine, present must be callable.
_OPTIONAL_CALLBACKS = ("on_rewind", "stop_when", "drain_terminations")
#: A ``rewind_provider`` (or direct provider) payload-key + state callbacks.
_PROVIDER_CALLBACKS = ("rewind_state", "apply_rewind_state")

_MISSING: object = object()


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
    run's intent.  ``name`` is the optional stable identity used by a
    multi-sink fan-out; it is required when more than one sink is declared.
    """

    factory: str | Callable[[SinkContext], Sink]
    options: Mapping[str, JsonValue] = field(default_factory=_empty_options)
    name: str | None = None

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
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ConfigError(f"sink.name={self.name!r} must be a non-empty string when declared")
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
class CollectSinkSpec:
    """The core built-in ``collect`` sink selection.

    Inert declaration data: it names the build-owned :class:`CollectSink`
    kind, not a user factory, so it carries no options.  The build owns the
    actual construction with its private resources (writer/logger/recorder,
    language generator, control facts); none of those enter
    :class:`SinkContext`.
    """

    kind: str = "collect"
    name: str = "collect"

    def __post_init__(self) -> None:
        if self.kind != "collect":
            raise ConfigError(
                f"sink kind={self.kind!r} is not supported (the only core built-in "
                "kind is 'collect')"
            )
        if not isinstance(self.name, str) or not self.name:
            raise ConfigError("collect sink.name must be a non-empty string")


#: A parsed sink selection: a core built-in kind or a custom factory.
SinkDeclaration: TypeAlias = CollectSinkSpec | SinkSpec


def effective_sink_name(declaration: SinkDeclaration) -> str | None:
    """The declared name a fan-out uses for this entry (custom may be nameless)."""
    return declaration.name


def validate_sink_declarations(declarations: Sequence[SinkDeclaration], source: str) -> None:
    """Enforce the naming and built-in-cardinality rules on a parsed selection.

    A single entry may omit its name: the build returns that sink directly and
    its own summary shape is unchanged.  Two or more entries need a unique,
    explicit effective name so ``FanOutSink`` summaries are stable.  The core
    built-in ``collect`` kind may appear at most once: two collect sinks would
    share one writer/logger/recorder and corrupt the recorded output.
    """
    collect_indices = [
        index
        for index, declaration in enumerate(declarations)
        if isinstance(declaration, CollectSinkSpec)
    ]
    if len(collect_indices) > 1:
        first, second = collect_indices[0], collect_indices[1]
        raise ConfigError(
            f"{source}: sinks[{first}] and sinks[{second}] both select the core "
            "built-in 'collect' sink; only one collect entry is allowed"
        )
    if len(declarations) <= 1:
        return
    seen: dict[str, int] = {}
    for index, declaration in enumerate(declarations):
        name = effective_sink_name(declaration)
        if name is None:
            raise ConfigError(
                f"{source}: sinks[{index}] needs an explicit name when more than "
                "one sink is declared"
            )
        previous = seen.get(name)
        if previous is not None:
            raise ConfigError(
                f"{source}: sinks[{previous}] and sinks[{index}] share the name "
                f"{name!r}; sink names must be unique"
            )
        seen[name] = index


@dataclass(frozen=True)
class SinkContext:
    """Build-time context handed to a custom sink factory.

    It carries run metadata plus the run's write-only value publisher face.
    There is deliberately no live env, writer, logger or bridge here: those
    are runner-owned resources a sink receives through its callback bundle,
    never through construction.  ``values`` is the one closed-loop publisher
    face; it is not a read view, and ``None`` means the build provided no
    publisher (a sink that needs closed loop must fail fast rather than fall
    back).
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
    values: ValuePublisher | None = None


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


def build_custom_sink(factory: SinkFactory, ctx: SinkContext) -> Sink:
    """Invoke a resolved custom-sink factory exactly once and validate its shape.

    The factory receives the construction-time :class:`SinkContext` and owns
    building its sink.  The returned object is validated structurally: the four
    required callbacks must exist and be callable; ``on_rewind`` / ``stop_when``
    / ``drain_terminations`` are optional capabilities but must be callable when
    present; a declared ``rewind_provider`` must be a callable supplier or an
    already valid provider shape.  Validation never calls the callbacks.
    """
    sink = factory(ctx)
    missing = [name for name in _REQUIRED_CALLBACKS if not callable(getattr(sink, name, None))]
    if missing:
        raise TypeError(f"custom sink {type(sink).__name__} is missing callbacks: {missing}")
    for name in _OPTIONAL_CALLBACKS:
        callback = getattr(sink, name, _MISSING)
        if callback is not _MISSING and not callable(callback):
            raise TypeError(f"custom sink {type(sink).__name__}.{name} must be callable")
    _validate_rewind_provider(sink)
    return sink


def _validate_rewind_provider(sink: Sink) -> None:
    """Fail fast when a custom sink declares an invalid ``rewind_provider``.

    ``None`` means "no provider" (the same meaning the discovery walk gives
    it).  A callable is accepted as the supplier contract and is not invoked
    here; a non-callable value must already be a valid provider: a non-empty
    string ``name`` (the checkpoint payload key) plus callable
    ``rewind_state`` / ``apply_rewind_state``.
    """
    provider = getattr(sink, "rewind_provider", None)
    if provider is None or callable(provider):
        return
    missing: list[str] = []
    name = getattr(provider, "name", None)
    if not isinstance(name, str) or not name:
        missing.append("name")
    for callback in _PROVIDER_CALLBACKS:
        if not callable(getattr(provider, callback, None)):
            missing.append(callback)
    if missing:
        raise TypeError(f"custom sink {type(sink).__name__}.rewind_provider is missing: {missing}")
