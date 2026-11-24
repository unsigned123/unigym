"""The read-only, boundary-scoped evaluation context.

A live run owns one :class:`EnvView` per evaluation boundary; the lifetime is
owner-defined.  Termination and benchmark scoring create a view per callback
and close it after the callback.  The watcher engine retains the current
vector-scoped view until the next tick replaces it, so add/test/dump and
callable ``on_reset`` can read the live boundary between callbacks.  Closing a
view never closes the environment or the run-shared query; it only invalidates
the view's private references.  The view is a facade over the existing env
record/observation/query faces; it never exposes the backend, a mutable engine
handle or a write operation.

Values returned by :meth:`EnvView.record` and :meth:`EnvView.obs` are borrowed
from the boundary snapshot and must not be retained past the owning boundary
(the callback, or the watcher's current tick).  Query results are detached
copies produced by the query layer and are safe to keep, but the
:class:`~unigym.evaluation.query.EvalQuery` handle is bound to the view
lifetime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import torch

from .._typing import JsonValue
from ..env.obs import ObsSnapshot
from ..query.facade import Query
from ..runtime.controllers import (
    ArticulationRuntime,
    ControlKey,
    ControlRuntime,
    ResolvedControl,
    articulation_facts as _runtime_articulation_facts,
    build_control_plan,
)

if TYPE_CHECKING:
    from .query import EvalQuery

__all__ = [
    "EnvView",
    "EvalContext",
    "EvalQueryUnavailableError",
    "EvalViewError",
    "EvalViewStaleError",
]

Record = Mapping[str, object]


class EvalViewError(RuntimeError):
    """Base class for evaluation-view failures."""


class EvalViewStaleError(EvalViewError):
    """An evaluation view was used after its owner closed it."""


class EvalQueryUnavailableError(EvalViewError, NotImplementedError):
    """The live backend exposes no runtime query at this boundary."""


def _select_leading(value: object, rows: torch.Tensor, num_envs: int) -> object:
    """Return *value* restricted to *rows* when its leading axis is per-env."""
    if isinstance(value, torch.Tensor):
        if value.ndim >= 1 and value.shape[0] == num_envs:
            return value.index_select(0, rows.to(device=value.device))
        return value
    if isinstance(value, np.ndarray):
        if value.ndim >= 1 and value.shape[0] == num_envs:
            return value[rows.detach().cpu().numpy()]
        return value
    return value


def _select_tensor(value: torch.Tensor, rows: torch.Tensor, num_envs: int) -> torch.Tensor:
    """Return a per-env slice of a ``(num_envs, ...)`` tensor."""
    if value.ndim >= 1 and value.shape[0] == num_envs:
        return value.index_select(0, rows.to(device=value.device))
    return value


class EnvView:
    """Read-only, boundary-scoped view of one live environment.

    ``env_id is None`` selects the whole vector environment (termination and
    watcher scope).  ``env_id is int`` selects exactly one env (benchmark
    episode scope); scoped reads keep a leading axis of one so tensor code
    stays ``(N, ...)``.  A per-env view rejects every query selection outside
    that env.

    The constructor is internal wiring: callers pass the boundary record, the
    already-published observation snapshot, the read-only articulation runtime
    facts, the runtime control rows and an optional lazy query factory.  The
    control rows are projected to frozen
    :class:`~unigym.runtime.controllers.ResolvedControl` values; no mutable
    runtime control or backend object is retained on the public surface.
    """

    __slots__ = (
        "_art_names",
        "_articulations",
        "_backend",
        "_closed",
        "_controls",
        "_controls_runtime",
        "_device",
        "_env_id",
        "_num_envs",
        "_obs",
        "_parent",
        "_query",
        "_query_factory",
        "_record",
    )

    def __init__(
        self,
        *,
        record: Record,
        obs: ObsSnapshot | None,
        env_id: int | None,
        num_envs: int,
        device: str | torch.device = "cpu",
        backend: str = "unknown",
        query_factory: Callable[[], Query] | None = None,
        articulations: Mapping[str, ArticulationRuntime] | None = None,
        controls: Mapping[ControlKey, ControlRuntime] | None = None,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise EvalViewError(f"num_envs must be a positive int, got {num_envs!r}")
        if env_id is not None:
            if isinstance(env_id, bool) or not isinstance(env_id, int):
                raise EvalViewError(f"env_id must be None or an int, got {env_id!r}")
            if env_id < 0 or env_id >= num_envs:
                raise EvalViewError(f"env_id {env_id} is outside [0, num_envs={num_envs})")
        if not isinstance(backend, str) or not backend:
            raise EvalViewError(f"backend must be a non-empty str, got {backend!r}")
        if not isinstance(record, Mapping):
            raise EvalViewError(f"record must be a mapping, got {type(record).__name__}")
        if obs is not None and obs.num_envs != num_envs:
            raise EvalViewError(
                f"obs num_envs {obs.num_envs} does not match view num_envs {num_envs}"
            )
        if query_factory is not None and not callable(query_factory):
            raise EvalViewError("query_factory must be callable or None")
        if controls is not None and not isinstance(controls, Mapping):
            raise EvalViewError(
                f"controls must be a mapping or None, got {type(controls).__name__}"
            )

        runtime_controls: dict[ControlKey, ControlRuntime] = {}
        for key, control in (controls or {}).items():
            if not isinstance(control, ControlRuntime):
                raise EvalViewError(
                    f"controls[{key!r}] must be a ControlRuntime, got " f"{type(control).__name__}"
                )
            expected = (control.articulation, control.name)
            if key != expected:
                raise EvalViewError(
                    f"controls key {key!r} does not match "
                    f"(control.articulation, control.name)={expected!r}"
                )
            runtime_controls[key] = control

        self._record = MappingProxyType(dict(record))
        self._obs = obs
        self._env_id = env_id
        self._num_envs = num_envs
        self._device = device
        self._backend = backend
        self._query_factory = query_factory
        self._articulations = MappingProxyType(dict(articulations or {}))
        self._controls_runtime = MappingProxyType(runtime_controls)
        self._controls = MappingProxyType(
            {
                (row.articulation, row.name): row
                for row in build_control_plan(tuple(runtime_controls.values()))
            }
        )
        self._art_names = tuple(sorted(self._articulations))
        self._parent: EnvView | None = None
        self._query: EvalQuery | None = None
        self._closed = False

    # ---- lifetime ------------------------------------------------------- #
    def _ensure_open(self) -> None:
        if self._closed:
            scope = "vector" if self._env_id is None else f"env {self._env_id}"
            raise EvalViewStaleError(
                f"evaluation view ({scope}) is closed; values are only valid "
                f"inside the owning callback"
            )
        parent = self._parent
        if parent is not None:
            parent._ensure_open()

    def close(self) -> None:
        """Invalidate this view (idempotent); never closes the env or query.

        Closing a parent invalidates every derived child through the parent
        chain.  Closing a child never closes its parent.
        """
        self._closed = True
        self._query = None

    def _derive(self, record: Mapping[str, object], env_id: int | None) -> EnvView:
        """Build a child view sharing this view's data and facts.

        The child shares record references, the observation snapshot, the lazy
        query factory and the articulation/control facts; it copies no arrays
        and opens no data.  It is linked to this view as its parent so closing
        this view invalidates the child.
        """
        child = object.__new__(EnvView)
        child._record = record
        child._obs = self._obs
        child._env_id = env_id
        child._num_envs = self._num_envs
        child._device = self._device
        child._backend = self._backend
        child._query_factory = self._query_factory
        child._articulations = self._articulations
        child._controls_runtime = self._controls_runtime
        child._controls = self._controls
        child._art_names = self._art_names
        child._parent = self
        child._query = None
        child._closed = False
        return child

    # ---- identity ------------------------------------------------------- #
    @property
    def num_envs(self) -> int:
        """The whole run's environment count."""
        self._ensure_open()
        return self._num_envs

    @property
    def env_id(self) -> int | None:
        """``None`` for a vector view, otherwise the selected env."""
        self._ensure_open()
        return self._env_id

    @property
    def scope_size(self) -> int:
        """One for a per-env view, ``num_envs`` for a vector view."""
        self._ensure_open()
        return 1 if self._env_id is not None else self._num_envs

    @property
    def device(self) -> str | torch.device:
        """The boundary device reported by the live backend."""
        self._ensure_open()
        return self._device

    @property
    def controls(self) -> Mapping[ControlKey, ResolvedControl]:
        """The build-time control rows, keyed by ``(articulation, name)``.

        The mapping is read-only and every value is a frozen
        :class:`~unigym.runtime.controllers.ResolvedControl`; no mutable
        runtime control or backend address is exposed.
        """
        self._ensure_open()
        return self._controls

    @property
    def articulation_names(self) -> tuple[str, ...]:
        """The declared articulation names carried by this view, sorted."""
        self._ensure_open()
        return self._art_names

    # ---- query ---------------------------------------------------------- #
    @property
    def has_query(self) -> bool:
        """Whether a lazy runtime-query factory was supplied."""
        self._ensure_open()
        return self._query_factory is not None

    @property
    def query(self) -> EvalQuery:
        """The lazy, read-only query adapter.

        Raises:
            EvalQueryUnavailableError: The boundary has no runtime query; use
                :meth:`record`/:meth:`obs` instead.
            EvalViewStaleError: The view is closed.
        """
        self._ensure_open()
        if self._query_factory is None:
            raise EvalQueryUnavailableError(
                f"backend {self._backend!r} exposes no runtime query at this "
                f"evaluation boundary; use record()/obs() instead"
            )
        cached = self._query
        if cached is None:
            try:
                raw = self._query_factory()
            except NotImplementedError as exc:
                raise EvalQueryUnavailableError(
                    f"backend {self._backend!r} does not provide a runtime query "
                    f"at this evaluation boundary; use record()/obs() instead"
                ) from exc
            from .query import EvalQuery as _EvalQuery

            cached = _EvalQuery(self, raw)
            self._query = cached
        return cached

    # ---- scoped reads --------------------------------------------------- #
    def _selection(self) -> torch.Tensor | None:
        if self._env_id is None:
            return None
        return torch.tensor([self._env_id], dtype=torch.int64)

    def record(self) -> Record:
        """The scoped step record as a read-only mapping.

        Vector views return every env's rows.  Per-env views return only
        ``env_id``'s rows with a leading axis of one.  Values with a leading
        axis equal to ``num_envs`` are sliced; other values (body-name tuples,
        bookkeeping) are returned unchanged.  The values are borrowed from the
        boundary and must not be retained.
        """
        self._ensure_open()
        rows = self._selection()
        if rows is None:
            return MappingProxyType(dict(self._record))
        return MappingProxyType(
            {
                key: _select_leading(value, rows, self._num_envs)
                for key, value in self._record.items()
            }
        )

    def obs(self) -> ObsSnapshot | None:
        """The scoped observation snapshot, or ``None`` when unpublished.

        Per-env views return a same-schema snapshot carrying only that env's
        rows.  Per-contact detail is dropped by :meth:`ObsSnapshot.select`
        because detail records have no declared env axis.
        """
        self._ensure_open()
        if self._obs is None:
            return None
        if self._env_id is None:
            return self._obs
        return self._obs.select((self._env_id,))

    def articulation(self, name: str) -> Mapping[str, object]:
        """Return one articulation's ``articulation/<name>/<leaf>`` rows.

        The mapping is read-only and scoped like :meth:`record`.
        """
        self._ensure_open()
        if not isinstance(name, str) or not name:
            raise EvalViewError(f"articulation name must be a non-empty str, got {name!r}")
        prefix = f"articulation/{name}/"
        scoped = self.record()
        leaves = {
            key[len(prefix) :]: value for key, value in scoped.items() if key.startswith(prefix)
        }
        if not leaves:
            available = ", ".join(self._articulation_names()) or "<none>"
            raise EvalViewError(
                f"articulation {name!r} is not in the evaluation record; " f"available: {available}"
            )
        return MappingProxyType(leaves)

    def articulation_facts(self, name: str, control: str | None = None) -> Mapping[str, JsonValue]:
        """The neutral per-articulation facts for one ``(articulation, control)``.

        Reuses the existing ``runtime.controllers.articulation_facts`` shape
        (joint names, base body, foot bodies, body sets, contacts, action
        group).  ``control=None`` requires exactly one body-role control for
        *name*; zero or multiple fails with :class:`EvalViewError`.  An
        explicit *control* name selects that row.  The returned top-level
        mapping is read-only.
        """
        self._ensure_open()
        if not isinstance(name, str) or not name:
            raise EvalViewError(f"articulation name must be a non-empty str, got {name!r}")
        art = self._articulations.get(name)
        if art is None:
            available = ", ".join(self._art_names) or "<none>"
            raise EvalViewError(
                f"articulation {name!r} has no runtime facts in this view; "
                f"available: {available}"
            )
        runtime_control = self._fact_control(name, control)
        return MappingProxyType(dict(_runtime_articulation_facts(art, runtime_control)))

    def _fact_control(self, name: str, control: str | None) -> ControlRuntime:
        """Resolve the runtime control row for :meth:`articulation_facts`."""
        available = (
            ", ".join(sorted(key[1] for key in self._controls_runtime if key[0] == name))
            or "<none>"
        )
        if control is not None:
            if not isinstance(control, str) or not control:
                raise EvalViewError(
                    f"control name must be a non-empty str or None, got {control!r}"
                )
            runtime = self._controls_runtime.get((name, control))
            if runtime is None:
                raise EvalViewError(
                    f"articulation {name!r} has no control {control!r} in this view; "
                    f"available controls: {available}"
                )
            return runtime
        body = [
            runtime
            for runtime in self._controls_runtime.values()
            if runtime.articulation == name and runtime.role == "body"
        ]
        if len(body) == 1:
            return body[0]
        if not body:
            raise EvalViewError(
                f"articulation {name!r} has no body-role control in this view; "
                f"pass control= explicitly (available controls: {available})"
            )
        names = ", ".join(sorted(runtime.name for runtime in body))
        raise EvalViewError(
            f"articulation {name!r} has multiple body-role controls ({names}); "
            f"pass control= explicitly (available controls: {available})"
        )

    def joint_pos_limits(self, articulation: str) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-env lower/upper limits aligned with the articulation joint order.

        Reads the articulation runtime fact (owned by
        :class:`~unigym.runtime.controllers.ArticulationRuntime`), scoped like
        :meth:`record`.  Returns ``None`` when the asset declares no limits.
        """
        self._ensure_open()
        if not isinstance(articulation, str) or not articulation:
            raise EvalViewError(f"articulation name must be a non-empty str, got {articulation!r}")
        runtime = self._articulations.get(articulation)
        if runtime is None:
            available = ", ".join(sorted(self._articulations)) or "<none>"
            raise EvalViewError(
                f"articulation {articulation!r} has no runtime facts in this view; "
                f"available: {available}"
            )
        limits = runtime.joint_pos_limits
        if limits is None:
            return None
        rows = self._selection()
        if rows is None:
            return limits
        lower, upper = limits
        return (
            _select_tensor(lower, rows, self._num_envs),
            _select_tensor(upper, rows, self._num_envs),
        )

    # ---- derivation ----------------------------------------------------- #
    def with_record(self, record: Record) -> EnvView:
        """A child view over *record* with this view's scope and facts.

        Only the record is substituted; scope, observation snapshot, query
        factory and articulation/control facts are shared.  The child is
        independent to close, but closing this view invalidates it.
        """
        self._ensure_open()
        if not isinstance(record, Mapping):
            raise EvalViewError(f"record must be a mapping, got {type(record).__name__}")
        return self._derive(MappingProxyType(dict(record)), self._env_id)

    def for_env(self, env_id: int) -> EnvView:
        """A per-env child of this vector view scoped to *env_id*.

        Raises:
            EvalViewError: This view is already per-env, or *env_id* is not a
                valid env id.
        """
        self._ensure_open()
        if self._env_id is not None:
            raise EvalViewError(
                f"for_env is only valid on a vector view; this view is scoped to "
                f"env {self._env_id}"
            )
        if isinstance(env_id, bool) or not isinstance(env_id, int):
            raise EvalViewError(f"env_id must be an int, got {env_id!r}")
        if env_id < 0 or env_id >= self._num_envs:
            raise EvalViewError(f"env_id {env_id} is outside [0, num_envs={self._num_envs})")
        return self._derive(self._record, env_id)

    def _articulation_names(self) -> list[str]:
        names: set[str] = set()
        for key in self._record:
            parts = key.split("/")
            if len(parts) >= 3 and parts[0] == "articulation":
                names.add(parts[1])
        return sorted(names)


@dataclass(frozen=True, slots=True)
class EvalContext:
    """One boundary's live evaluation context.

    The owner creates exactly one context per callback and closes it after the
    callback.  The context is ephemeral: retaining it and reading later raises
    :class:`EvalViewStaleError`.
    """

    env: EnvView
    t: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.env, EnvView):
            raise EvalViewError(
                f"EvalContext.env must be an EnvView, got {type(self.env).__name__}"
            )
        if self.t is not None and (
            isinstance(self.t, bool) or not isinstance(self.t, int) or self.t < 0
        ):
            raise EvalViewError(f"EvalContext.t must be None or a non-negative int, got {self.t!r}")

    @property
    def env_id(self) -> int | None:
        """The selected env (``None`` for a vector context)."""
        return self.env.env_id

    @property
    def num_envs(self) -> int:
        """The whole run's environment count."""
        return self.env.num_envs

    def close(self) -> None:
        """Close the underlying view (idempotent)."""
        self.env.close()
