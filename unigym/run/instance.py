"""RunInstance: the live resource bundle owned by one constructed run.

``build_instance`` returns this object; a Runner borrows it and never
closes it.  ``close()`` is idempotent and tears down the SHM bridge, any
owned recorder close hook and the env backend (the pre-split
``collect_cli`` finally path).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from unigym.backends.registry import CANONICAL_BACKENDS, Trait, backend_info
from unigym.query.errors import QueryStaleError
from unigym.sink import Sink

from .spec import RunSpec

if TYPE_CHECKING:
    from unigym.adapters.base import Adapter
    from unigym.cameras.manager import CameraManager
    from unigym.commands.base import CommandSampler
    from unigym.contacts.manager import ContactManager
    from unigym.director import Director
    from unigym.env.backends.base import EnvBackend
    from unigym.env.native import NativeHandle
    from unigym.io.writer import ShardWriter
    from unigym.logger import DataCollectLogger
    from unigym.model.live import LiveModel
    from unigym.props.base import PropManager
    from unigym.query.facade import Query
    from unigym.recorder import Recorder
    from unigym.robots.base import RobotProfile
    from unigym.runtime.actions import OrderedActionGroup
    from unigym.runtime.bridge import Bridge
    from unigym.runtime.controllers import (
        ResolvedControl,
        ResolvedControlPlan,
    )
    from unigym.runtime.rewind_providers import RewindStateProvider
    from unigym.terminations.composite import TerminationGroup
    from unigym.trace.replay import TraceReplay
    from unigym.values import ValueStore


if TYPE_CHECKING:

    class RunEnv(EnvBackend, Protocol):
        """Env backend plus the run-layer wiring and snapshot faces.

        ``adapter_groups`` (policy ownership) and ``adapter_joint_targets``
        (the joint -> kind validation partition) are stamped by
        :func:`build_instance`; the snapshot/restore pair is the rewind face
        (all registered backends implement it, and rewind gates on the
        SNAPSHOT trait before use).  ``effective_action_groups`` is the
        consumer-side action table (authored groups + backend-declared prop
        groups); it is metadata for sinks/runners, never backend routing input.
        """

        adapter_groups: dict[str, tuple[tuple[str, str], ...]]
        adapter_joint_targets: dict[str, dict[tuple[str, str], str]]
        effective_action_groups: tuple[OrderedActionGroup, ...]

        def snapshot(self) -> object:
            """Capture the engine state for a rewind checkpoint."""
            ...

        def restore(self, state: object) -> None:
            """Restore a state captured by :meth:`snapshot`."""
            ...

        def native(self, backend: str) -> NativeHandle:
            """Return the backend's unstable native escape hatch."""
            ...


def _empty_providers() -> list[RewindStateProvider]:
    """Typed empty default for :attr:`RunInstance.rewind_providers`."""
    return []


@dataclass
class RunInstance:
    """Live resources of one run (env / managers / sinks / bridge)."""

    spec: RunSpec
    profile: RobotProfile
    adapter: Adapter
    sampler: CommandSampler | None
    term_group: TerminationGroup
    env: RunEnv
    bridge: Bridge
    plan: ResolvedControlPlan | None = None
    values: ValueStore | None = None
    action_groups: tuple[OrderedActionGroup, ...] = ()
    camera_manager: CameraManager | None = None
    prop_manager: PropManager | None = None
    contact_manager: ContactManager | None = None
    recorder: Recorder | None = None
    director: Director | None = None
    sink: Sink | None = None
    writer: ShardWriter | None = None
    logger: DataCollectLogger | None = None
    replay: TraceReplay | None = None
    rewind_providers: list[RewindStateProvider] = field(default_factory=_empty_providers)
    stop_when: Callable[[], bool] | None = None
    artifacts_dir: Path | None = None
    out_dir: str | None = None
    reset_on_termination: bool = False
    _closed: bool = field(default=False, init=False, repr=False)
    _query: Query | None = field(default=None, init=False, repr=False)

    @property
    def query(self) -> Query:
        """The trait-gated on-demand runtime query facade.

        The query object is created lazily on first access and reused for the
        lifetime of this live env instance.  It is not an observation channel
        and never publishes or records.

        Raises:
            NotImplementedError: The live backend does not declare the
                ``RUNTIME_QUERY`` trait (or exposes no query factory).
            QueryStaleError: This run is already closed.
        """
        if self._closed:
            raise QueryStaleError(
                f"run.query for backend {self.spec.backend!r} is stale: the run is closed"
            )
        if self._query is not None:
            return self._query
        live = self.spec.backend
        info = backend_info(live)
        if Trait.RUNTIME_QUERY not in info.traits:
            raise NotImplementedError(
                f"backend {live!r} does not declare the RUNTIME_QUERY trait "
                f"(run.query is optional and this backend declares it absent)"
            )
        env_query = getattr(self.env, "query", None)
        if env_query is not None:
            self._query = env_query
            return env_query
        factory = info.query_backend
        if factory is None:
            raise NotImplementedError(
                f"backend {live!r} declares RUNTIME_QUERY but exposes no "
                f"env.query and the registry declares no query_backend factory"
            )
        self._query = factory()(self.env)
        return self._query

    @property
    def resolved_controls(self) -> tuple[ResolvedControl, ...]:
        """The frozen :class:`ResolvedControl` rows of :attr:`plan`."""
        return () if self.plan is None else tuple(self.plan.controls)

    @property
    def gated(self) -> bool:
        """True when the run carries the pause gate (``serve_pause``).

        "Paused/gated" is a run-level property, not a runner kind: the
        declaration is ``spec.bridges["serve_pause"]`` and the mechanism is
        the bridge's ``StepGate``.  This reads that declaration (the
        transient paused/stepping state is never duplicated here).
        """
        bridges = self.spec.bridges or {}
        return bool(bridges.get("serve_pause", False))

    @property
    def model(self):
        """The env's per-articulation :class:`~unigym.model.set.ModelSet`.

        The set holds one ``ModelIR`` per policy articulation plus the
        whole-scene ``merged`` IR (``None`` when the backend has no honest
        merge).  It is read from ``env.model_set`` (one source; no copy is
        stored here).

        Raises:
            RuntimeError: The wired env exposes no ``model_set`` (a backend
                without the assembly face).
        """
        model_set = getattr(self.env, "model_set", None)
        if model_set is None:
            name = type(self.env).__name__ if self.env is not None else "None"
            raise RuntimeError(
                f"RunInstance.model: env {name} has no model_set "
                f"(backend {self.spec.backend!r} does not build the v103.4 "
                f"ModelSet)"
            )
        return model_set

    def live_for(self, name: str | None = None) -> LiveModel:
        """Live proxies for one articulation's IR.

        Args:
            name: An articulation name from ``self.model.names()``.  ``None``
                binds the single articulation of a one-articulation run (or
                its merged IR); a multi-articulation run requires an explicit
                name (there is no first-binding fallback).

        Returns:
            A :class:`~unigym.model.live.LiveModel` bound to ``self.env``.

        Raises:
            RuntimeError: *name* was omitted for a multi-articulation run.
            KeyError: *name* is not a declared articulation.
        """
        from unigym.model.live import LiveModel

        model = self.model
        if name is None:
            if len(model) != 1:
                raise RuntimeError(
                    f"RunInstance.live_for: this run has "
                    f"{len(model)} articulations {model.names()}; pass an "
                    f"explicit name (no first-binding fallback)"
                )
            ir = next(iter(model.articulations.values()))
        else:
            ir = model[name]
        return LiveModel(ir, self.env)

    @property
    def live(self) -> LiveModel:
        """Deprecated: use :meth:`live_for` or :attr:`query`.

        Emits ``DeprecationWarning`` and preserves the historical binding:
        a fresh zero-copy proxy over the single articulation.  Migrate live
        proxy reads to ``live_for(name)`` and typed reads to ``query``.  The
        deprecation is warning-only: the delegation and returned value are
        unchanged.  Retained until zero-caller proof; removal requires a
        separate owner-approved change.

        A fresh binding is returned on every access; nothing mutable is
        cached here, and the proxies hold only a weak reference to the
        env.  Dynamic reads after :meth:`close` raise
        :class:`~unigym.model.StaleModelError`.

        Returns:
            A :class:`~unigym.model.live.LiveModel` bound to ``self.env``.

        Raises:
            RuntimeError: The run has several articulations (use
                :meth:`live_for` with a name), or the env's backend has no
                live binding yet (mujoco/mjwarp today).
        """
        warnings.warn(
            "RunInstance.live is deprecated; use live_for(name) for live "
            "proxies and query for typed reads",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.live_for()

    def native(self, backend: str) -> NativeHandle:
        """Return the live backend's unstable native escape hatch.

        Args:
            backend: A canonical backend name; it must match this run's live
                backend.  The removed ``isaaclab`` / ``isaaclab3`` aliases are
                rejected (no silent resolution).

        Returns:
            A :class:`~unigym.env.native.NativeHandle` of identity
            references (external packs only; unstable).

        Raises:
            ValueError: *backend* is not canonical, or does not match the live
                backend.
            NotImplementedError: The live backend does not declare the
                ``NATIVE_HANDLE`` trait (or exposes no native method).
        """
        live = self.spec.backend
        if not isinstance(backend, str) or backend not in CANONICAL_BACKENDS:
            raise ValueError(
                f"native(backend={backend!r}) is not a canonical backend | "
                f"hint: pass exactly one of: {', '.join(CANONICAL_BACKENDS)} "
                "(the 'isaaclab' / 'isaaclab3' aliases were removed)"
            )
        requested = backend
        if requested != live:
            raise ValueError(
                f"native(backend={backend!r}) does not match this run's live "
                f"backend {live!r} (canonical requested {requested!r})"
            )
        if Trait.NATIVE_HANDLE not in backend_info(live).traits:
            raise NotImplementedError(
                f"backend {live!r} does not declare the NATIVE_HANDLE trait "
                f"(env.native is an external-pack-only escape hatch)"
            )
        method = getattr(self.env, "native", None)
        if not callable(method):
            raise NotImplementedError(
                f"backend {live!r} declares NATIVE_HANDLE but its env exposes "
                f"no native() method"
            )
        return self.env.native(live)

    def model_edit(self):
        """Open one global model-edit transaction on the live backend.

        Returns:
            A :class:`~unigym.model.edit.ModelEditor` bound to this run.

        Raises:
            RuntimeError: The run is closed.
            NotImplementedError: The live backend does not declare
                ``MODEL_EDIT`` (or its env exposes no ``model_edit`` method).
        """
        if self._closed:
            raise RuntimeError("RunInstance is closed: model edits need a live backend")
        live = self.spec.backend
        if Trait.MODEL_EDIT not in backend_info(live).traits:
            raise NotImplementedError(
                f"backend {live!r} does not declare the MODEL_EDIT trait "
                f"(run.model_edit is optional and this backend declares it absent)"
            )
        method = getattr(self.env, "model_edit", None)
        if not callable(method):
            raise NotImplementedError(
                f"backend {live!r} declares MODEL_EDIT but its env exposes "
                f"no model_edit() method"
            )
        return method()

    def close(self) -> None:
        """Idempotent teardown: bridge -> recorder (if it owns one) -> env."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.bridge is not None:
                self.bridge.close()
        finally:
            try:
                close = getattr(self.recorder, "close", None)
                if callable(close):
                    close()
            finally:
                try:
                    if self._query is not None:
                        self._query.close()
                finally:
                    if self.env is not None:
                        self.env.close()
