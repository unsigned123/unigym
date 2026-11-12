"""RunInstance: the live resource bundle owned by one constructed run.

``build_instance`` returns this object; a Runner borrows it and never
closes it.  ``close()`` is idempotent and tears down the SHM bridge, any
owned recorder close hook and the env backend (the pre-split
``collect_cli`` finally path).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .spec import RunSpec

if TYPE_CHECKING:
    from unigym.adapters.base import Adapter
    from unigym.cameras.manager import CameraManager
    from unigym.commands.base import CommandSampler
    from unigym.contacts.manager import ContactManager
    from unigym.director import Director
    from unigym.env.backends.base import EnvBackend
    from unigym.io.writer import ShardWriter
    from unigym.logger import DataCollectLogger
    from unigym.model.live import LiveModel
    from unigym.props.base import PropManager
    from unigym.recorder import Recorder
    from unigym.robots.base import RobotProfile
    from unigym.runtime.bridge import Bridge
    from unigym.runtime.controllers import (
        ResolvedControl,
        ResolvedControlPlan,
    )
    from unigym.runtime.rewind_providers import RewindStateProvider
    from unigym.terminations.composite import TerminationGroup
    from unigym.trace.replay import TraceReplay


if TYPE_CHECKING:

    class RunEnv(EnvBackend, Protocol):
        """Env backend plus the run-layer wiring and snapshot faces.

        ``adapter_groups`` is stamped by :func:`build_instance`; the
        snapshot/restore pair is the rewind face (all registered backends
        implement it, and rewind gates on the SNAPSHOT trait before use).
        """

        adapter_groups: dict[str, tuple[str, ...]]

        def snapshot(self) -> object:
            """Capture the engine state for a rewind checkpoint."""
            ...

        def restore(self, state: object) -> None:
            """Restore a state captured by :meth:`snapshot`."""
            ...


class Sink(Protocol):
    """The engine sink contract (collect/benchmark/fan-out/bridge chain)."""

    def on_step(self, t: int, bundle: dict[str, object]) -> None:
        """Consume one pre-reset engine step."""

    def on_episode_end(self, env_ids: object, bundle: dict[str, object]) -> None:
        """Consume the episode-end data side before the reset chain."""

    def on_step_end(self, t: int, bundle: dict[str, object]) -> None:
        """Consume one post-reset engine step (metrics side)."""

    def on_run_end(self) -> dict[str, object]:
        """Flush the run and return the sink summary."""
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
        """Live, zero-copy proxies over :attr:`model`.

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
        return self.live_for()

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
                if self.env is not None:
                    self.env.close()
