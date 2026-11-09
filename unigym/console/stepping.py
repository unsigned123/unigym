"""StepGate: the pause/step state machine behind the runtime debug console.

Replaces the bridge's inline freeze loop (``Bridge._freeze_loop``).
The bridge parks the engine inside ``sink.on_step`` (a step boundary -- the
step's physics, obs and recording are already complete) and consults the
gate:

    RUNNING ──pause──▶ PAUSED (engine parked inside wait())
    PAUSED(pending==0) ──request_step(n)──▶ pending=n, the park loop exits,
        the engine advances; at each later boundary ``arrive()`` consumes
        one pending step and re-parks once the budget is exhausted
    PAUSED ──resume──▶ RUNNING (pending cleared)
    PAUSED ──pause (repeat)──▶ idempotent; pending cleared (freeze NOW,
        pause always wins over an in-flight step budget)
    DONE run: request_step is refused by the console handler ("run finished")

``step`` semantics: one step = one full policy
iteration (act -> env.step -> obs -> record -> reset chain) driven by the
engine itself, so recording and episode clocks behave exactly like a
running rollout.  The physics-step concept is DELETED -- the wire contract
has no unit field (not NotImplemented: the field simply does not exist).

Pure logic + stdlib only (no torch/numpy): offline-unit-testable and safe
on the bridge's stdlib-only CLI import path.
"""

from __future__ import annotations

import time

__all__ = ["StepGate", "ConsoleError", "MAX_FAST_FORWARD"]

MAX_FAST_FORWARD = 100_000


class ConsoleError(ValueError):
    """Console command rejected (schema/state violation; lands in the ack,
    never escapes into the engine loop)."""


class StepGate:
    """Paused/pending_steps machine; all state is engine-thread-only.

    Every mutation happens on the engine's main thread (commands are
    dispatched from ``Bridge._serve_commands``, re-entrantly inside the
    park loop), so no lock is needed -- same single-thread discipline as
    the rest of the bridge.
    """

    def __init__(self, paused: bool = False):
        # paused=True (`unigym debug --pause`): born PAUSED --
        # the engine parks at the FIRST step boundary and waits for
        # resume/step; default keeps the historical start-RUNNING state.
        self.paused = bool(paused)
        self.pending_steps = 0

    # -- transitions ------------------------------------------------------- #
    def pause(self) -> None:
        """Enter paused (idempotent); an in-flight step budget is dropped."""
        self.paused = True
        self.pending_steps = 0

    def resume(self) -> None:
        """Back to RUNNING; any pending step budget is dropped."""
        self.paused = False
        self.pending_steps = 0

    def request_step(self, n: int = 1) -> int:
        """Queue *n* policy steps (paused only).  Returns *n*.

        Raises :class:`ConsoleError` on non-int/out-of-range/running state
        ("pause first").  The ack returns immediately; the engine then
        advances through the budget and re-parks (see ``arrive``).
        """
        if isinstance(n, bool) or not isinstance(n, int):
            raise ConsoleError(f"step: 'n' must be an integer, got {n!r}")
        if not 1 <= n <= MAX_FAST_FORWARD:
            raise ConsoleError(f"step: 'n' must be in [1, {MAX_FAST_FORWARD}], got {n}")
        if not self.paused:
            raise ConsoleError(
                "pause first: 'step' is only served while paused -- send "
                '{"type": "pause"} to POST /control (or run `unigym '
                "pause`) first, then re-queue this step"
            )
        self.pending_steps = int(n)
        return int(n)

    # -- engine-side hooks --------------------------------------------------- #
    def arrive(self) -> bool:
        """Step-boundary hook (``Bridge.tick``): True = park now.

        RUNNING -> never park.  Paused with a step budget: consume one
        step and keep the engine running until the budget is exhausted,
        then re-park (the boundary where the budget hit zero is parked).
        """
        if not self.paused:
            return False
        if self.pending_steps > 0:
            self.pending_steps -= 1
            if self.pending_steps > 0:
                return False
        return True

    def wait(self, env, service_fn, stop_fn=None, poll_interval: float = 0.02) -> None:
        """Park the engine: poll until resumed or a step budget is queued.

        ``service_fn`` (zero-arg) keeps the command block + broadcast
        frames alive while parked -- get_view/set_props/inspect/step/...
        all keep working on the frozen state.  ``stop_fn``: park must end
        (closed / shutdown_requested).

        ``env`` is the EnvBackend (may be None).  Freeze-period rendering
        fix (console contract §frozen-phase rendering): when the env exposes a kit
        simulator + scene (the isaaclab backend: ``env.sim`` /
        ``env.scene``), the park loop calls ``sim.render()`` AND
        ``env.scene.update(0)`` so camera pose changes made while paused
        become visible in the next get_view (the sensor annotators only
        refresh on the scene update -- sim.render() alone is not enough,
        2026-09-16 container-measured frame diff 0.00 -> 11.59 when the
        scene update was added).  mujoco has neither attribute: get_view
        renders on demand and needs no fix.  A raising hook degrades to
        no-op after one attempt (logged once) -- the park loop must never
        spin on a broken pipeline.
        """
        sim = getattr(env, "sim", None)
        render = getattr(sim, "render", None) if sim is not None else None
        scene_update = getattr(getattr(env, "scene", None), "update", None)
        hooks = []
        if render is not None:
            hooks.append((render, "sim.render", False))
        if scene_update is not None:
            hooks.append((scene_update, "scene.update", True))
        while self.paused and self.pending_steps == 0 and not (stop_fn is not None and stop_fn()):
            service_fn()
            for fn, label, needs_dt in hooks:
                try:
                    fn(0) if needs_dt else fn()
                except Exception as e:  # noqa: BLE001 -- degrade once
                    print(
                        f"[console] freeze-period {label} disabled after "
                        f"one failure: {type(e).__name__}: {e}",
                        flush=True,
                    )
                    hooks = [h for h in hooks if h[1] != label]
            time.sleep(poll_interval)

    # -- introspection ------------------------------------------------------- #
    @property
    def state(self) -> str:
        """RUNNING | PAUSED | STEPPING (paused with a live step budget)."""
        if not self.paused:
            return "RUNNING"
        return "STEPPING" if self.pending_steps > 0 else "PAUSED"
