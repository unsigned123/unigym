"""Typed runner and reset-time state-reader doubles for facade unit tests.

``FakeRunner`` implements the runner boundary the simulation facade drives:
one explicit ``step(actions)`` tick over the shared ``FakeBackend``, an
explicit reset chain, a public tick clock, configurable due/terminated row
masks and deterministic fault injection.  It owns no loop and no thread;
``run()`` refuses instead of inventing a policy.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from unigym.runner.base import StepResult
from unigym.runtime.actions import NamedAction
from tests._support.fakes.backend import FakeBackend


def _row_ids(rows: Sequence[int], num_envs: int, what: str) -> tuple[int, ...]:
    """Validate one configured row set: unique ints in ``[0, num_envs)``."""
    ids = tuple(rows)
    if any(isinstance(row, bool) or not isinstance(row, int) for row in ids):
        raise ValueError(f"{what} must be ints, got {ids!r}")
    if any(row < 0 or row >= num_envs for row in ids):
        raise ValueError(f"{what} must be in [0, {num_envs}), got {ids!r}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{what} must be unique, got {ids!r}")
    return ids


class FakeRunner:
    """One explicit external tick over :class:`FakeBackend` (CPU, deterministic).

    ``fail_next_tick`` raises once from the next tick (after advancing the
    clock) to exercise the facade's fault path.  ``reset`` delegates to the
    backend's row reset and does not advance the clock, matching the
    synchronous runner contract.  ``reset_t`` overrides only the tick reported
    by the reset result, so callers can prove the facade clock follows
    ``ticks_done`` rather than a stale result tick.  ``last_actions`` is the
    exact mapping of the most recent ``step``/``tick`` call (the lowered

    ``terminated_rows`` and ``due_rows`` make the step result's termination
    mask and backend-owned due mask configurable.  They are reported only: the
    fake never resets a due row during ``step`` (the facade's
    ``runner.auto_reset: false`` policy), so the backend state still proves
    that a due tick does not clear the row.
    """

    def __init__(
        self,
        num_envs: int = 1,
        joint_names: Sequence[str] | None = None,
        *,
        terminated_rows: Sequence[int] = (),
        due_rows: Sequence[int] = (),
    ) -> None:
        self.env = (
            FakeBackend(num_envs=num_envs)
            if joint_names is None
            else FakeBackend(num_envs=num_envs, joint_names=joint_names)
        )
        self.fail_next_tick = False
        self.reset_t: int | None = None
        self.last_actions: tuple[NamedAction, ...] | None = None
        self.terminated_rows = _row_ids(terminated_rows, num_envs, "terminated_rows")
        self.due_rows = _row_ids(due_rows, num_envs, "due_rows")
        self._ticks = 0
        self._closed = False

    @property
    def ticks_done(self) -> int:
        """Policy ticks completed so far."""
        return self._ticks

    def step(self, actions: Sequence[NamedAction]) -> StepResult:
        """Advance one tick with caller-supplied named actions."""
        return self.tick(actions_override=actions)

    def tick(self, actions_override: Sequence[NamedAction] | None = None) -> StepResult:
        """Advance one tick; the external action sequence is required."""
        if self._closed:
            raise RuntimeError("FakeRunner is closed")
        if actions_override is None:
            raise RuntimeError("FakeRunner has no adapter path; pass named actions")
        self.last_actions = tuple(actions_override)
        if self.fail_next_tick:
            self.fail_next_tick = False
            self._ticks += 1
            raise RuntimeError("FakeRunner engine failure")
        step = self.env.step(actions_override)
        backend_ids = step.reset_env_ids
        reset_ids = (
            backend_ids if not self.due_rows else torch.tensor(self.due_rows, dtype=torch.int64)
        )
        terminated = torch.zeros(self.env.num_envs, dtype=torch.bool)
        if self.terminated_rows:
            terminated[list(self.terminated_rows)] = True
        result = StepResult(
            t=self._ticks,
            obs=step.obs,
            record=step.record,
            reset_env_ids=reset_ids,
            terminated=terminated,
        )
        self._ticks += 1
        return result

    def reset(self, env_ids: Sequence[int] | None = None) -> StepResult:
        """Reset the selected rows (all when ``None``); never advances ticks."""
        if self._closed:
            raise RuntimeError("FakeRunner is closed")
        if env_ids is None:
            due = self.env.reset_all()
        else:
            due = self.env.reset_envs(torch.as_tensor(list(env_ids), dtype=torch.int64))
        return StepResult(
            t=self._ticks if self.reset_t is None else self.reset_t,
            obs=self.env.obs.snapshot(),
            record=self.env.record(),
            reset_env_ids=due,
            terminated=None,
        )

    def run(self, stop_when: Callable[[], bool] | None = None) -> dict:
        """Refuse: this double is the single-step boundary, not a loop."""
        raise RuntimeError("FakeRunner is a single-step seam; drive step(actions) explicitly")

    def close(self) -> None:
        """Close the shared backend (idempotent)."""
        self._closed = True
        self.env.close()
