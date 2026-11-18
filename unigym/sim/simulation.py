"""The live controller of the experimental simulation facade.

One :class:`Simulation` owns no engine object of its own: it borrows the
``Run`` / ``ExternalRunner`` pair built by :class:`SimulationBuilder`, keeps
one pending-action table, and caches the last produced ``StepResult`` state.

The pending-state rules are the facade contract:

* ``act`` validates and stages; it never advances time.
* ``step`` requires complete, fresh pending actions, lowers them to
  ``tuple[NamedAction, ...]``, ticks exactly once and clears the table before
  the new observation is built.  The ``step(actions=...)`` sugar stages its
  whole mapping atomically: any failed entry clears the table and leaves the
  tick/state untouched.
* ``observe`` is read-only and serves the cached state.
* ``reset`` clears pending and refreshes the state through the runner reset
  chain plus the shared request-free extra reader; ``env_ids`` selects the
  rows the runner resets (``None`` resets every row).  A reset never advances
  the policy tick.
* ``step`` never resets a row: the producing ``StepResult`` reports the
  episode-clock due rows and the termination mask, and the caller decides
  which rows to reset next.
* an engine exception is fatal for the live simulation: the pending table is
  cleared, the simulation is marked faulted and the exception propagates.
* out-of-band runner ticks are detected through ``ticks_done`` drift and
  raise :class:`~unigym.sim.errors.StaleStateError` instead of applying a
  stale target.

``torch`` is imported inside the action-normalization and lowering paths
only, so importing ``unigym.sim`` stays torch-free.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

from .errors import (
    BindingError,
    ForeignHandleError,
    MissingActionError,
    ObservationError,
    PendingActionError,
    SimulationClosedError,
    SimulationConfigError,
    SimulationFaultedError,
    StaleStateError,
)
from .handles import ArticulationHandle, CameraHandle
from .observation import Observation, build_observation
from .pending import PendingActions, StagedAction

if TYPE_CHECKING:
    import torch

    from unigym.env.backends.base import EnvBackend
    from unigym.run.api import Run
    from unigym.runner.base import Runner, StepResult
    from unigym.runtime.actions import NamedAction, OrderedActionGroup, OrderedActionSpec
    from unigym.world import World

__all__ = ["Simulation"]


@runtime_checkable
class _StepRunner(Protocol):
    """The runner face the facade drives: an explicit ``step(actions)`` tick.

    ``ExternalRunner`` implements this on top of ``SyncRunner.tick``; the
    unit seam double implements the same method so the state machine can be
    proved without an engine.
    """

    def step(self, actions: Sequence[NamedAction]) -> StepResult: ...


def _reset_ids(
    value: object,
    what: str,
    *,
    num_envs: int,
    request: torch.Tensor | None,
) -> torch.Tensor:
    """Copy one runner reset-id vector, validating its contract.

    A runner result is a 1-D ``int64`` tensor of unique ids in ``[0, N)``.
    When ``request`` is given the result must also be a subset of the rows the
    caller asked for; a runner that quietly resets more or fewer rows is a
    contract breach, never a repaired value.
    """
    import torch

    if not isinstance(value, torch.Tensor):
        raise ObservationError(
            f"{what} reset_env_ids must be a torch.Tensor, got {type(value).__name__}"
        )
    if value.ndim != 1:
        raise ObservationError(f"{what} reset_env_ids must be 1-D, got shape {tuple(value.shape)}")
    if value.dtype != torch.int64:
        raise ObservationError(f"{what} reset_env_ids must be int64, got dtype {value.dtype}")
    ids = value.clone()
    if ids.numel() > 0 and bool(((ids < 0) | (ids >= num_envs)).any()):
        raise ObservationError(
            f"{what} reset_env_ids {ids} are outside the legal range [0, {num_envs})"
        )
    if ids.numel() > 1:
        ordered = ids.sort().values
        if bool((ordered[1:] == ordered[:-1]).any()):
            raise ObservationError(f"{what} reset_env_ids {ids} contain duplicate ids")
    if request is not None and not bool(torch.isin(ids, request.to(device=ids.device)).all()):
        raise ObservationError(
            f"{what} reset_env_ids {ids} are not a subset of the requested ids {request}"
        )
    return ids


def _terminated_mask(value: object, num_envs: int) -> torch.Tensor | None:
    """Copy one runner termination mask, validating its ``(N,)`` bool contract."""
    import torch

    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise ObservationError(
            f"StepResult terminated must be a torch.Tensor or None, " f"got {type(value).__name__}"
        )
    if value.dtype != torch.bool:
        raise ObservationError(f"StepResult terminated must be bool, got dtype {value.dtype}")
    if value.ndim != 1 or value.shape[0] != num_envs:
        raise ObservationError(
            f"StepResult terminated must have shape ({num_envs},), " f"got {tuple(value.shape)}"
        )
    return value.clone()


class Simulation:
    """The live, single-clock controller returned by ``SimulationBuilder.build``.

    Args:
        backend: the resolved backend name.
        num_envs: the env count of the built run.
        world: the world declaration the builder handed to ``Run``.
        articulations: the declared articulation handles (builder-owned).
        cameras: the declared camera handles (builder-owned).
        action_spec: the global ordered action template every staged group obeys.
        runner: the live external runner of ``run`` (kind-pinned upstream).
        env: the live env backend the runner steps.
        run: the run that owns the instance and the runner.
        owner: the identity token the handles carry.
        state: the spawn record from ``env.record()``.
        obs: the spawn typed :class:`~unigym.env.obs.ObsSnapshot`.
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
        action_spec: OrderedActionSpec,
        runner: Runner,
        env: EnvBackend,
        run: Run,
        owner: object,
        state: Mapping[str, object],
        obs: object,
        t: int,
    ) -> None:
        import torch

        from unigym.runtime.actions import OrderedActionSpec

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
        if not isinstance(action_spec, OrderedActionSpec):
            raise SimulationConfigError(
                "Simulation action_spec must be an OrderedActionSpec, "
                f"got {type(action_spec).__name__}"
            )
        env_num_envs = getattr(env, "num_envs", num_envs)
        if env_num_envs != num_envs:
            raise SimulationConfigError(
                f"Simulation num_envs {num_envs} does not match env.num_envs {env_num_envs}"
            )

        handles = tuple(articulations)
        handle_by_name: dict[str, ArticulationHandle] = {}
        for handle in handles:
            if handle._owner is not owner:
                raise ForeignHandleError(
                    f"articulation handle {handle.name!r} belongs to another "
                    f"builder/simulation; handles are scoped to their owner"
                )
            if handle.name in handle_by_name:
                raise SimulationConfigError(
                    f"articulation {handle.name!r} is declared twice; labels are unique"
                )
            handle_by_name[handle.name] = handle

        action_groups: dict[tuple[str, str], OrderedActionGroup] = {
            (group.articulation, group.name): group for group in action_spec.groups
        }
        spec_by_articulation: dict[str, dict[str, OrderedActionGroup]] = {}
        for group in action_spec.groups:
            spec_by_articulation.setdefault(group.articulation, {})[group.name] = group
        missing_handles = sorted(set(spec_by_articulation) - set(handle_by_name))
        if missing_handles:
            raise SimulationConfigError(
                f"action_spec names articulation(s) {missing_handles} with no declared "
                f"handle; declared handles: {sorted(handle_by_name)}"
            )
        for handle in handles:
            expected = spec_by_articulation.get(handle.name, {})
            actual = dict(handle.action_groups)
            if set(actual) != set(expected):
                raise SimulationConfigError(
                    f"articulation {handle.name!r} declares action groups "
                    f"{sorted(actual)} but the action_spec expects {sorted(expected)}"
                )
            for group_name, group in expected.items():
                if actual[group_name] != group:
                    raise SimulationConfigError(
                        f"articulation {handle.name!r} action group {group_name!r} does "
                        f"not match the action_spec template {group!r}"
                    )

        self.backend = backend
        self.num_envs = num_envs
        self.world = world
        self.articulations = handles
        self.cameras = tuple(cameras)
        self._runner = runner
        self._env = env
        self._run = run
        self._owner = owner
        self._action_spec = action_spec
        self._action_groups = action_groups
        self._pending = PendingActions()
        self._state: dict[str, object] = dict(state)
        self._obs = obs
        self._reset_env_ids = torch.empty(0, dtype=torch.int64, device=env.device)
        self._reset_due_env_ids = torch.empty(0, dtype=torch.int64, device=env.device)
        self._terminated: torch.Tensor | None = None
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
    @overload
    def act(
        self,
        target: OrderedActionGroup,
        values: torch.Tensor | Mapping[str, float | torch.Tensor],
    ) -> None: ...

    @overload
    def act(
        self,
        target: ArticulationHandle,
        values: torch.Tensor | Mapping[str, float | torch.Tensor],
    ) -> None: ...

    def act(
        self,
        target: OrderedActionGroup | ArticulationHandle,
        values: torch.Tensor | Mapping[str, float | torch.Tensor],
    ) -> None:
        """Validate and stage one ordered action group; never advances time.

        Two target forms are accepted:

        * ``act(group, tensor)``: the explicit :class:`OrderedActionGroup`
          path.  ``tensor`` is a floating ``(N, dim)`` tensor in the group's
          joint order, or ``(dim,)`` when ``N == 1``.
        * ``act(handle, values)``: a convenience for a handle that declares
          exactly one action group.  A mapping must cover the group joints
          exactly; its values are reordered into the group's joint order.  With
          ``N == 1`` a value may be a finite number or a floating ``(1,)``
          tensor, and the forms may mix.  With ``N > 1`` every value must be a
          floating ``(N,)`` tensor.  A tensor is accepted in group joint order.

        The target must name exactly one group of this simulation's
        :class:`OrderedActionSpec`; an unknown or mismatched group is refused.
        """
        self._ensure_operable()
        self._check_clock()
        import torch

        group = self._resolve_group(target)
        if isinstance(values, Mapping):
            tensor = self._named_values(group, values)
        elif isinstance(values, torch.Tensor):
            tensor = self._normalize_values(group, values)
        else:
            raise BindingError(
                f"act for action group {(group.articulation, group.name)!r} requires a "
                f"mapping of joint name -> value or a torch.Tensor, got "
                f"{type(values).__name__}"
            )
        self._pending.stage(StagedAction(group=group, values=tensor))

    def _resolve_group(
        self,
        target: OrderedActionGroup | ArticulationHandle,
    ) -> OrderedActionGroup:
        """Resolve one target to the authoritative declared action group."""
        from unigym.runtime.actions import OrderedActionGroup

        if isinstance(target, OrderedActionGroup):
            identity = (target.articulation, target.name)
            declared = self._action_groups.get(identity)
            if declared is None:
                raise BindingError(
                    f"action group {identity!r} is not declared by this simulation; "
                    f"declared groups: {sorted(self._action_groups)}"
                )
            if declared != target:
                raise BindingError(
                    f"action group {identity!r} does not match the declared template "
                    f"{declared!r}; stage the group object declared by the action_spec"
                )
            return declared
        if isinstance(target, ArticulationHandle):
            if target._owner is not self._owner or target not in self.articulations:
                raise ForeignHandleError(
                    f"articulation handle {target.name!r} belongs to another "
                    f"builder/simulation; handles are scoped to their owner"
                )
            groups = tuple(target.action_groups.values())
            if len(groups) != 1:
                declared = [(group.articulation, group.name) for group in groups]
                raise BindingError(
                    f"articulation {target.name!r} declares {len(groups)} action groups "
                    f"{declared}; act(handle, values) needs a single-group handle, so "
                    f"pass the OrderedActionGroup to act"
                )
            return groups[0]
        raise BindingError(
            "act target must be an OrderedActionGroup or an ArticulationHandle from "
            f"this simulation, got {type(target).__name__}"
        )

    def _named_values(
        self,
        group: OrderedActionGroup,
        values: Mapping[str, float | torch.Tensor],
    ) -> torch.Tensor:
        """Lower one named mapping to an ``(N, dim)`` tensor in group joint order.

        ``N == 1`` accepts finite numbers and floating ``(1,)`` tensors in any
        mix; numbers lower to float32 columns.  ``N > 1`` requires one floating
        ``(N,)`` tensor per joint with a uniform dtype and device.  Columns are
        stacked in the group's declared joint order, cloned, and then staged
        through the same explicit tensor path.
        """
        import torch

        identity = (group.articulation, group.name)
        provided = tuple(values)
        missing = [joint for joint in group.joints if joint not in values]
        extra = [joint for joint in provided if joint not in group.joints]
        if missing or extra:
            raise BindingError(
                f"act(handle, mapping) for action group {identity!r} must cover exactly "
                f"the group joints {list(group.joints)}; missing {missing}, "
                f"unexpected {extra}"
            )
        columns: list[float | torch.Tensor] = []
        tensor_device: torch.device | None = None
        reference: tuple[str, torch.dtype, torch.device] | None = None
        for joint in group.joints:
            value = values[joint]
            if isinstance(value, torch.Tensor):
                if not value.is_floating_point():
                    raise BindingError(
                        f"act(handle, mapping) action group {identity!r} joint "
                        f"{joint!r} value must be a floating tensor, got dtype "
                        f"{value.dtype}"
                    )
                if self.num_envs == 1:
                    if tuple(value.shape) != (1,):
                        raise BindingError(
                            f"act(handle, mapping) action group {identity!r} joint "
                            f"{joint!r} value shape {tuple(value.shape)} is not a legal "
                            f"(1,) vector for num_envs=1"
                        )
                else:
                    expected = (self.num_envs,)
                    if tuple(value.shape) != expected:
                        raise BindingError(
                            f"act(handle, mapping) action group {identity!r} joint "
                            f"{joint!r} value shape {tuple(value.shape)} does not match "
                            f"the expected (num_envs,)={expected}"
                        )
                    if reference is None:
                        reference = (joint, value.dtype, value.device)
                    else:
                        first_joint, first_dtype, first_device = reference
                        if value.dtype != first_dtype:
                            raise BindingError(
                                f"act(handle, mapping) action group {identity!r} joint "
                                f"{joint!r} value dtype {value.dtype} does not match "
                                f"joint {first_joint!r} dtype {first_dtype}; all per-env "
                                f"values must share one dtype"
                            )
                        if value.device != first_device:
                            raise BindingError(
                                f"act(handle, mapping) action group {identity!r} joint "
                                f"{joint!r} value device {value.device} does not match "
                                f"joint {first_joint!r} device {first_device}; all per-env "
                                f"values must share one device"
                            )
                if tensor_device is None:
                    tensor_device = value.device
                columns.append(value)
                continue
            if self.num_envs != 1:
                raise BindingError(
                    f"act(handle, mapping) action group {identity!r} joint {joint!r} "
                    f"value must be a floating torch.Tensor of shape "
                    f"({self.num_envs},) for num_envs={self.num_envs}; use "
                    f"torch.full(({self.num_envs},), {value!r}) for a constant"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BindingError(
                    f"act(handle, mapping) action group {identity!r} joint {joint!r} "
                    f"value must be a finite number, got {value!r}"
                )
            if not math.isfinite(float(value)):
                raise BindingError(
                    f"act(handle, mapping) action group {identity!r} joint {joint!r} "
                    f"value must be finite, got {value!r}"
                )
            columns.append(float(value))
        stacked: list[torch.Tensor] = []
        for column in columns:
            if isinstance(column, torch.Tensor):
                stacked.append(column)
            else:
                stacked.append(torch.tensor([column], dtype=torch.float32, device=tensor_device))
        return torch.stack(stacked, dim=1).clone()

    def step(
        self,
        actions: (
            Mapping[
                OrderedActionGroup | ArticulationHandle,
                torch.Tensor | Mapping[str, float | torch.Tensor],
            ]
            | None
        ) = None,
    ) -> Observation:
        """Advance exactly one tick with the staged actions and return the state.

        Args:
            actions: Optional sugar: one ``group -> values`` entry per declared
                action group.  Every entry is staged with :meth:`act` and then
                the normal :meth:`step` runs; an already-staged simulation is
                refused.  The canonical path is ``act`` followed by ``step()``.
        """
        self._ensure_operable()
        if actions is not None:
            if not isinstance(actions, Mapping):
                raise BindingError(
                    "step(actions=...) expects a mapping of action group -> values, "
                    f"got {type(actions).__name__}"
                )
            if self._pending.groups():
                raise PendingActionError(
                    f"step(actions=...) requires an empty pending table; already staged: "
                    f"{sorted(self._pending.groups())}"
                )
            self._check_clock()
            try:
                for target, values in actions.items():
                    self.act(target, values)
            except BaseException:
                self._pending.clear()
                raise
        return self._step()

    def _step(self) -> Observation:
        """Run the canonical one-tick path after all checks have passed."""
        self._ensure_operable()
        self._check_clock()
        staged = self._pending.groups()
        if not staged:
            raise PendingActionError(
                "step() needs a staged action; call act(...) first "
                "(the previous action is never reused)"
            )
        required = frozenset(self._action_groups)
        unknown = staged - required
        if unknown:
            raise PendingActionError(
                f"pending actions name undeclared action group(s) {sorted(unknown)}; "
                f"declared groups: {sorted(required)}"
            )
        missing = required - staged
        if missing:
            raise MissingActionError(
                f"step() is missing staged action(s) for required action group(s) "
                f"{sorted(missing)}; staged groups: {sorted(staged)}"
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
    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> Observation:
        """Reset every env (``None``) or the named rows, clearing pending.

        The reset ids are validated at the boundary and handed to
        ``runner.reset(ids)``; the runner owns the reset chain and the RNG
        order.  A reset never consumes an action and never advances the tick:
        ``t`` follows ``runner.ticks_done``.  The whole pending table is
        cleared because a staged action may belong to a row reset here.

        The reported ``reset_env_ids`` are the rows the runner actually reset
        (a unique ``int64`` subset of the request); ``reset_due_env_ids`` is
        empty and ``terminated`` is ``None`` because the previous tick's due
        and termination masks are no longer meaningful.
        """
        self._ensure_operable()
        ids = self._normalize_reset_ids(env_ids)
        self._pending.clear()
        import torch

        request = torch.arange(self.num_envs, dtype=torch.int64) if ids is None else ids
        try:
            result = self._runner.reset(None if ids is None else ids)
            state = self._env.record()
            reset_env_ids = _reset_ids(
                result.reset_env_ids,
                "runner reset",
                num_envs=self.num_envs,
                request=request,
            )
        except Exception:
            self._pending.clear()
            self._faulted = True
            raise
        self._state = dict(state)
        self._obs = self._env.obs.snapshot()
        self._reset_env_ids = reset_env_ids
        self._reset_due_env_ids = torch.empty(0, dtype=torch.int64, device=self._env.device)
        self._terminated = None
        self._state_t = int(self._runner.ticks_done)
        self._clock = int(self._runner.ticks_done)
        return self._observation()

    def _normalize_reset_ids(
        self, env_ids: Sequence[int] | torch.Tensor | None
    ) -> torch.Tensor | None:
        """Normalize one caller id argument to a 1-D ``int64`` CPU tensor.

        ``None`` means every row and stays ``None`` (the runner's v1 all-rows
        path).  An explicit set is validated as non-empty ints in ``[0, N)``
        with no duplicates; bools, floats and other non-int values are
        refused.  There is no silent dedupe, sort or truncation.
        """
        import torch

        if env_ids is None:
            return None
        if isinstance(env_ids, torch.Tensor):
            if env_ids.dtype == torch.bool or env_ids.is_floating_point() or env_ids.is_complex():
                raise SimulationConfigError(
                    f"reset(env_ids=...) must be an integer id tensor, got dtype "
                    f"{env_ids.dtype}; field 'env_ids', ids {env_ids}, "
                    f"legal ids [0, {self.num_envs})"
                )
            if env_ids.ndim != 1:
                raise SimulationConfigError(
                    f"reset(env_ids=...) must be a 1-D id tensor, got shape "
                    f"{tuple(env_ids.shape)}; field 'env_ids', legal ids "
                    f"[0, {self.num_envs})"
                )
            ids = env_ids.detach().to(device="cpu", dtype=torch.int64).clone()
            return self._validate_reset_ids(ids)
        if isinstance(env_ids, (str, bytes, bytearray)):
            raise SimulationConfigError(
                f"reset(env_ids=...) must be a sequence of ints, got "
                f"{type(env_ids).__name__}; field 'env_ids', legal ids "
                f"[0, {self.num_envs})"
            )
        try:
            values = list(env_ids)
        except TypeError as err:
            raise SimulationConfigError(
                f"reset(env_ids=...) must be a sequence of ints or a tensor, got "
                f"{type(env_ids).__name__}; field 'env_ids', legal ids "
                f"[0, {self.num_envs})"
            ) from err
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise SimulationConfigError(
                    f"reset(env_ids=...) ids must be ints, got {values!r}; field "
                    f"'env_ids', legal ids [0, {self.num_envs})"
                )
        return self._validate_reset_ids(torch.tensor(values, dtype=torch.int64))

    def _validate_reset_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Validate one normalized reset-id vector against the env count."""
        if ids.numel() == 0:
            raise SimulationConfigError(
                f"reset(env_ids=...) must name at least one row, got an empty id "
                f"set; field 'env_ids', legal ids [0, {self.num_envs})"
            )
        if bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise SimulationConfigError(
                f"reset(env_ids=...) ids {ids} are outside the legal range "
                f"[0, {self.num_envs}); field 'env_ids'"
            )
        if ids.numel() > 1:
            ordered = ids.sort().values
            if bool((ordered[1:] == ordered[:-1]).any()):
                raise SimulationConfigError(
                    f"reset(env_ids=...) ids {ids} contain duplicates; field "
                    f"'env_ids'; legal ids [0, {self.num_envs})"
                )
        return ids

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
        """Project the cached state through the typed observation builder."""
        return build_observation(
            t=self._state_t,
            num_envs=self.num_envs,
            state=self._state,
            snapshot=self._obs,
            handles=self.articulations,
            cameras=self.cameras,
            reset_env_ids=self._reset_env_ids,
            reset_due_env_ids=self._reset_due_env_ids,
            terminated=self._terminated,
        )

    def _cache_step(self, result: StepResult) -> None:
        """Adopt one successful step result as the cached state.

        The runner tick already happened, so the state, the exact
        ``dict(result.obs)`` copy, the due rows and the clock are committed
        together before the observation is projected.  A projection failure
        (for example an expected optional cache key present but not a tensor)
        then propagates as ``ObservationError`` while ``t`` and the facade
        clock still name the producing tick: the facade stays synchronised with
        the engine, is not faulted, and repeating ``observe`` reports the same
        honest error until a reset.  The due rows and termination mask are
        validated and copied before any field is assigned, so a rejected
        ``StepResult`` never leaves a partial adoption.

        A step performs no reset, so ``reset_env_ids`` is empty and
        ``reset_due_env_ids`` carries the producing tick's episode-clock due
        rows: they are what the caller should reset next, not what this tick
        reset.
        """
        import torch

        reset_due_env_ids = _reset_ids(
            result.reset_env_ids,
            "StepResult",
            num_envs=self.num_envs,
            request=None,
        )
        terminated = _terminated_mask(result.terminated, self.num_envs)
        reset_env_ids = torch.empty(0, dtype=torch.int64, device=self._env.device)
        self._state = dict(result.record)
        self._obs = result.obs
        self._reset_env_ids = reset_env_ids
        self._reset_due_env_ids = reset_due_env_ids
        self._terminated = terminated
        self._state_t = int(result.t)
        self._clock = int(self._runner.ticks_done)

    def _normalize_values(
        self,
        group: OrderedActionGroup,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """Validate one value tensor and normalize it to ``(N, dim)``."""
        import torch

        identity = (group.articulation, group.name)
        dim = len(group.joints)
        if not isinstance(values, torch.Tensor):
            raise BindingError(
                f"action group {identity!r} values must be a torch.Tensor, "
                f"got {type(values).__name__}"
            )
        if not values.is_floating_point():
            raise BindingError(
                f"action group {identity!r} values must be floating, got dtype " f"{values.dtype}"
            )
        if values.ndim == 1:
            if self.num_envs != 1 or values.shape[0] != dim:
                raise BindingError(
                    f"action group {identity!r} values shape {tuple(values.shape)} "
                    f"is not a legal (dim,) vector for num_envs={self.num_envs}, "
                    f"dim={dim}"
                )
            normalized = values.reshape(1, dim)
        elif values.ndim == 2:
            expected = (self.num_envs, dim)
            if tuple(values.shape) != expected:
                raise BindingError(
                    f"action group {identity!r} values shape {tuple(values.shape)} "
                    f"does not match the expected (num_envs, dim)={expected}"
                )
            normalized = values
        else:
            raise BindingError(
                f"action group {identity!r} values must be (num_envs, dim) or "
                f"(dim,), got {values.ndim} dimensions"
            )
        return normalized.clone()

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
