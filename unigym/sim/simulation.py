"""The live controller of the experimental simulation facade.

One :class:`Simulation` owns no engine object of its own: it borrows the
``Run`` / ``ExternalRunner`` pair built by :class:`SimulationBuilder`, keeps
one pending-action table, and caches the last produced ``StepResult`` state.

The pending-state rules are the facade contract:

* ``act`` validates and stages; it never advances time.
* ``step`` requires complete, fresh pending actions, lowers them to
  ``dict[str, NamedAction]``, ticks exactly once and clears the table before
  the new observation is built.  The ``step(actions=...)`` sugar stages its
  whole mapping atomically: any failed entry clears the table and leaves the
  tick/state untouched.
* ``observe`` is read-only and serves the cached state.
* ``reset`` clears pending and refreshes the state through the runner reset
  chain plus the shared request-free extra reader.
* an engine exception is fatal for the live simulation: the pending table is
  cleared, the simulation is marked faulted and the exception propagates.
* out-of-band runner ticks are detected through ``ticks_done`` drift and
  raise :class:`~unigym.sim.errors.StaleStateError` instead of applying a
  stale target.

``torch`` is imported inside the action-normalization and lowering paths
only, so importing ``unigym.sim`` stays torch-free.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .errors import (
    BindingError,
    ForeignHandleError,
    MissingActionError,
    PendingActionError,
    SimulationClosedError,
    SimulationConfigError,
    SimulationFaultedError,
    StaleStateError,
)
from .handles import ActionBinding, ArticulationHandle, CameraHandle
from .observation import Observation
from .pending import PendingActions, StagedAction

if TYPE_CHECKING:
    import torch

    from unigym.env.backends.base import EnvBackend
    from unigym.run.api import Run
    from unigym.runner.base import Runner, StepResult
    from unigym.runtime.actions import NamedAction
    from unigym.world import World

__all__ = ["Simulation"]

_STATE_BOOKKEEPING_KEYS = frozenset({"reset_env_ids", "should_reset"})


@runtime_checkable
class _StepRunner(Protocol):
    """The runner face the facade drives: an explicit ``step(actions)`` tick.

    ``ExternalRunner`` implements this on top of ``SyncRunner.tick``; the
    unit seam double implements the same method so the state machine can be
    proved without an engine.
    """

    def step(self, actions: dict[str, NamedAction]) -> StepResult: ...


class Simulation:
    """The live, single-clock controller returned by ``SimulationBuilder.build``.

    Args:
        backend: the resolved backend name.
        num_envs: the env count of the built run.
        world: the world declaration the builder handed to ``Run``.
        articulations: the declared articulation handles (builder-owned).
        cameras: the declared camera handles (builder-owned).
        runner: the live external runner of ``run`` (kind-pinned upstream).
        env: the live env backend the runner steps.
        run: the run that owns the instance and the runner.
        owner: the identity token the handles carry.
        state: the spawn state record from the request-free extra reader.
        obs_cache: the spawn observation cache.
        t: the policy tick that produced ``state`` (``0`` at spawn).
    """

    def __init__(
        self,
        *,
        backend: str,
        num_envs: int,
        world: World,
        articulations: Sequence[ArticulationHandle],
        cameras: Sequence[CameraHandle],
        runner: Runner,
        env: EnvBackend,
        run: Run,
        owner: object,
        state: Mapping[str, object],
        obs_cache: Mapping[str, object],
        t: int,
    ) -> None:
        if not isinstance(runner, _StepRunner):
            raise SimulationConfigError(
                "Simulation needs a runner whose step(actions) face is the explicit "
                f"tick entry point; got {type(runner).__name__}"
            )
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise SimulationConfigError(
                f"Simulation num_envs must be an int >= 1, got {num_envs!r}"
            )
        if isinstance(t, bool) or not isinstance(t, int) or t < 0:
            raise SimulationConfigError(f"Simulation t must be an int >= 0, got {t!r}")
        env_num_envs = getattr(env, "num_envs", num_envs)
        if env_num_envs != num_envs:
            raise SimulationConfigError(
                f"Simulation num_envs {num_envs} does not match env.num_envs {env_num_envs}"
            )

        handles = tuple(articulations)
        faces: dict[str, tuple[ArticulationHandle, tuple[str, ...], str]] = {}
        for handle in handles:
            if handle._owner is not owner:
                raise ForeignHandleError(
                    f"articulation handle {handle.name!r} belongs to another "
                    f"builder/simulation; handles are scoped to their owner"
                )
            control = handle.control
            if control is None:
                raise SimulationConfigError(
                    f"articulation {handle.name!r} has no control declaration; "
                    f"the facade drives declared control faces only"
                )
            for face_name, joints in handle.faces.items():
                if face_name in faces:
                    raise SimulationConfigError(
                        f"control face {face_name!r} is declared twice; face names are global"
                    )
                face = control.faces.get(face_name)
                if face is None:
                    raise SimulationConfigError(
                        f"articulation {handle.name!r} face {face_name!r} has no "
                        f"control-face declaration"
                    )
                faces[face_name] = (handle, tuple(joints), face.target)

        self.backend = backend
        self.num_envs = num_envs
        self.world = world
        self.articulations = handles
        self.cameras = tuple(cameras)
        self._runner = runner
        self._env = env
        self._run = run
        self._owner = owner
        self._faces = faces
        self._pending = PendingActions()
        self._state: dict[str, object] = dict(state)
        self._obs_cache: dict[str, object] = dict(obs_cache)
        self._state_t = t
        self._clock = int(runner.ticks_done)
        self._closed = False
        self._faulted = False

    # ------------------------------------------------------------------ #
    # public state
    # ------------------------------------------------------------------ #
    @property
    def t(self) -> int:
        """The policy tick that produced the cached state."""
        return self._state_t

    @property
    def closed(self) -> bool:
        """True after :meth:`close`; every later live call is refused."""
        return self._closed

    @property
    def faulted(self) -> bool:
        """True after an engine exception; only :meth:`close` is safe."""
        return self._faulted

    @property
    def run(self) -> Run:
        """The run that owns the built instance and the single clock."""
        return self._run

    @property
    def runner(self) -> Runner:
        """The live runner at the neutral seam (never a second clock).

        The builder path yields an ``ExternalRunner``, but the facade only
        requires the structural ``step(actions)`` seam, so the property is
        typed as the neutral :class:`~unigym.runner.base.Runner` protocol.
        """
        return self._runner

    # ------------------------------------------------------------------ #
    # observation
    # ------------------------------------------------------------------ #
    def observe(self) -> Observation:
        """Build the current observation from the cached state; never ticks."""
        self._ensure_operable()
        self._check_clock()
        return self._observation()

    # ------------------------------------------------------------------ #
    # action staging
    # ------------------------------------------------------------------ #
    def act(
        self,
        target: ActionBinding,
        values: torch.Tensor,
        *,
        face: str | None = None,
    ) -> None:
        """Validate and stage one action face; never advances time."""
        self._ensure_operable()
        self._check_clock()
        if not isinstance(target, ActionBinding):
            raise BindingError(
                "act target must be an ActionBinding bound from this simulation, "
                f"got {type(target).__name__}"
            )
        if target._owner is not self._owner or target.articulation._owner is not self._owner:
            raise ForeignHandleError(
                f"action binding for face {target.face!r} belongs to another "
                f"builder/simulation; handles are scoped to their owner"
            )
        if face is not None and face != target.face:
            raise BindingError(f"act face {face!r} does not match the binding face {target.face!r}")
        declared = self._faces.get(target.face)
        if declared is None:
            raise BindingError(
                f"action face {target.face!r} is not a declared control face; "
                f"declared faces: {sorted(self._faces)}"
            )
        handle, joints, kind = declared
        if target.articulation is not handle:
            raise ForeignHandleError(
                f"action binding face {target.face!r} names articulation "
                f"{target.articulation.name!r}, but the declared face belongs to "
                f"{handle.name!r}"
            )
        if target.kind != kind:
            raise BindingError(
                f"action face {target.face!r} binding kind {target.kind!r} does not "
                f"match the control target {kind!r}"
            )
        if len(target.joint_names) != len(joints) or set(target.joint_names) != set(joints):
            raise BindingError(
                f"action face {target.face!r} binding columns {list(target.joint_names)} "
                f"are not a permutation of the face joints {list(joints)}"
            )
        normalized = self._normalize_values(target, values, len(joints))
        self._pending.stage(target.face, StagedAction(binding=target, values=normalized))

    def step(
        self,
        actions: Mapping[ActionBinding, torch.Tensor] | None = None,
    ) -> Observation:
        """Advance exactly one tick with the staged actions and return the state.

        Args:
            actions: Optional sugar: one ``binding -> values`` entry per face.
                Every entry is staged with :meth:`act` and then the normal
                :meth:`step` runs; an already-staged simulation is refused.
                The canonical path is ``act`` followed by ``step()``.
        """
        self._ensure_operable()
        if actions is not None:
            if not isinstance(actions, Mapping):
                raise BindingError(
                    "step(actions=...) expects a mapping of ActionBinding -> tensor, "
                    f"got {type(actions).__name__}"
                )
            if self._pending.faces():
                raise PendingActionError(
                    f"step(actions=...) requires an empty pending table; already staged: "
                    f"{sorted(self._pending.faces())}"
                )
            self._check_clock()
            try:
                for binding, values in actions.items():
                    self.act(binding, values)
            except BaseException:
                self._pending.clear()
                raise
        return self._step()

    def _step(self) -> Observation:
        """Run the canonical one-tick path after all checks have passed."""
        self._ensure_operable()
        self._check_clock()
        staged = self._pending.faces()
        if not staged:
            raise PendingActionError(
                "step() needs a staged action; call act(...) first "
                "(the previous action is never reused)"
            )
        required = frozenset(self._faces)
        unknown = staged - required
        if unknown:
            raise PendingActionError(
                f"pending actions name undeclared face(s) {sorted(unknown)}; "
                f"declared faces: {sorted(required)}"
            )
        missing = required - staged
        if missing:
            raise MissingActionError(
                f"step() is missing staged action(s) for required face(s) "
                f"{sorted(missing)}; staged faces: {sorted(staged)}"
            )
        lowered = self._pending.lower()
        try:
            result = self._runner.step(lowered)
        except Exception:
            self._pending.clear()
            self._faulted = True
            raise
        self._pending.clear()
        self._cache_step(result)
        return self._observation()

    # ------------------------------------------------------------------ #
    # reset / close
    # ------------------------------------------------------------------ #
    def reset(self) -> Observation:
        """Reset every env, clear pending and refresh the cached spawn state."""
        self._ensure_operable()
        self._pending.clear()
        try:
            self._runner.reset(None)
            state = self._read_extra()
        except Exception:
            self._pending.clear()
            self._faulted = True
            raise
        self._state = dict(state)
        self._obs_cache = dict(self._env.obs_cache)
        self._state_t = int(self._runner.ticks_done)
        self._clock = int(self._runner.ticks_done)
        return self._observation()

    def close(self) -> None:
        """Close the owned run (and its runner/instance); idempotent."""
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        self._run.close()

    def __enter__(self) -> Simulation:
        """Enter the session; the simulation is already live."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Exit the session through the same idempotent close path."""
        self.close()

    # ------------------------------------------------------------------ #
    # internal
    # ------------------------------------------------------------------ #
    def _observation(self) -> Observation:
        """Build the A3 observation record from the cached state."""
        raw = {
            key: value for key, value in self._state.items() if key not in _STATE_BOOKKEEPING_KEYS
        }
        return Observation(
            t=self._state_t,
            raw=raw,
            obs_cache=self._obs_cache,
            articulations={},
            cameras={},
            contacts={},
        )

    def _cache_step(self, result: StepResult) -> None:
        """Adopt one successful step result as the cached state."""
        self._state = dict(result.extra)
        self._obs_cache = dict(result.obs)
        self._state_t = int(result.t)
        self._clock = int(self._runner.ticks_done)

    def _normalize_values(
        self,
        binding: ActionBinding,
        values: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        """Validate one value tensor and normalize it to ``(N, dim)``."""
        import torch

        if not isinstance(values, torch.Tensor):
            raise BindingError(
                f"action face {binding.face!r} values must be a torch.Tensor, "
                f"got {type(values).__name__}"
            )
        if not values.is_floating_point():
            raise BindingError(
                f"action face {binding.face!r} values must be floating, got dtype "
                f"{values.dtype}"
            )
        if values.ndim == 1:
            if self.num_envs != 1 or values.shape[0] != dim:
                raise BindingError(
                    f"action face {binding.face!r} values shape {tuple(values.shape)} "
                    f"is not a legal (dim,) vector for num_envs={self.num_envs}, "
                    f"dim={dim}"
                )
            normalized = values.reshape(1, dim)
        elif values.ndim == 2:
            expected = (self.num_envs, dim)
            if tuple(values.shape) != expected:
                raise BindingError(
                    f"action face {binding.face!r} values shape {tuple(values.shape)} "
                    f"does not match the expected (num_envs, dim)={expected}"
                )
            normalized = values
        else:
            raise BindingError(
                f"action face {binding.face!r} values must be (num_envs, dim) or "
                f"(dim,), got {values.ndim} dimensions"
            )
        return normalized.clone()

    def _read_extra(self) -> dict[str, object]:
        """Read the post-reset state through the shared request-free seam."""
        from unigym.runner.state import read_extra

        return read_extra(self._env)

    def _check_clock(self) -> None:
        """Refuse out-of-band ticking instead of applying a stale target."""
        ticks = int(self._runner.ticks_done)
        if ticks != self._clock:
            self._pending.clear()
            raise StaleStateError(
                f"the runner advanced out of band: facade clock {self._clock}, "
                f"runner ticks_done {ticks}; stage fresh actions after an "
                f"explicit reset or observe the new state"
            )

    def _ensure_operable(self) -> None:
        """Refuse every live call on a closed or faulted simulation."""
        if self._closed:
            raise SimulationClosedError("the simulation is closed; build a new SimulationBuilder")
        if self._faulted:
            raise SimulationFaultedError(
                "the simulation faulted during an engine tick; only close() is safe"
            )
