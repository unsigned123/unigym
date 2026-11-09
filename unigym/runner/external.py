"""ExternalRunner: user-driven single-step loop.

The runner borrows the same :class:`~unigym.runner.sync.SyncRunner` engine
(reset chain, termination, sink callbacks, RNG order) and adds exactly one
face: :meth:`step`, which feeds a caller-supplied
``dict[str, NamedAction]`` through ``SyncRunner.tick(actions_override=...)``.
The adapter-driven path is untouched (``tick()`` with no override).

Use :class:`~unigym.adapters.manual.ManualAdapter` (or any adapter whose
``act`` is never needed) as the instance adapter.
"""

from __future__ import annotations

from collections.abc import Callable

from unigym.runner.base import StepResult
from unigym.runner.sync import SyncRunner, _adapter_groups


class ExternalRunner(SyncRunner):
    """A Runner whose clock is advanced explicitly via :meth:`step`.

    ``run()`` is intentionally not a rollout: without an ``action_fn`` it
    refuses and tells the caller to drive :meth:`step` themselves.  When an
    ``action_fn`` is supplied it is called as ``action_fn(t)`` for each tick
    and must return the named-action dict for that tick.
    """

    @classmethod
    def from_instance(cls, instance, action_fn: Callable | None = None) -> ExternalRunner:
        """Build from a live RunInstance (mirrors SyncRunner.from_instance)."""
        spec = instance.spec
        runner_cfg = spec.runner or {}
        return cls(
            spec.cfg,
            instance.profile,
            instance.adapter,
            instance.sampler,
            instance.term_group,
            instance.env,
            instance.sink,
            reset_on_termination=instance.reset_on_termination,
            reset_mask_fn=runner_cfg.get("reset_mask_fn"),
            replay=instance.replay,
            action_fn=action_fn,
            adapter_groups=_adapter_groups(instance),
        )

    def __init__(self, *args, action_fn: Callable | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_fn = action_fn

    def step(self, actions: dict) -> StepResult:
        """Advance one policy tick with caller-supplied named actions."""
        return self.tick(actions_override=actions)

    def run(self, stop_when: Callable[[], bool] | None = None) -> dict:
        """Drive ``action_fn(t)`` ticks, or refuse and point at :meth:`step`.

        A manual runner has no policy loop to run.  Supplying ``action_fn``
        keeps ``run()`` usable for scripts; without it the honest answer is
        an error telling the caller to step explicitly.
        """
        if self.action_fn is None:
            raise RuntimeError(
                "ExternalRunner.run() needs an action_fn (t -> "
                "dict[str, NamedAction]); drive ExternalRunner.step(actions) "
                "yourself for a manual loop"
            )
        self._start()
        while int(self._tick) < self.total_steps:
            if stop_when is not None and stop_when():
                break
            self.step(self.action_fn(int(self._tick)))
        return self.sink.on_run_end()
