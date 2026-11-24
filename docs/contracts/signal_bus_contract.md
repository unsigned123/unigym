# Signal bus contract

- Status: active
- Owner module: `unigym/signals.py`

A `SignalBus` is a run-scoped, semantics-free transport for per-env signal
tensors.  Producers publish values; a `CommandSampler` receives the bus through
the read-only `SignalView` protocol and interprets the names itself.  A custom
sink receives the write-only `SignalPublisher` face through
`SinkContext.signals`.  Core defines transport only: it never interprets a
name, dtype, trailing shape, phase or event, and it never maps a signal to a
command or an adapter observation.

Producer tiers:

- phase 2A: user code and tests call `SignalBus.publish` directly;
- phase 2C: a custom sink publishes from its runner callbacks through the
  publish-only `SinkContext.signals` face.

Watchers are debug-only and never receive or publish through this bus.  Both
producer tiers publish into the one run bus.  The sink's face is write-only, so
a sink cannot read back a value it or another producer published; reading is
reserved for the sampler's `SignalView`.

## Surface

```python
class SignalPublisher(Protocol):
    def publish(self, name: str, value: torch.Tensor) -> None: ...

class SignalView(Protocol):
    def get(self, name: str) -> torch.Tensor | None: ...
    def snapshot(self) -> Mapping[str, torch.Tensor]: ...

class SignalBus:
    name = "signals"

    def __init__(self, num_envs: int, device: str | torch.device): ...
    def publish(self, name: str, value: torch.Tensor) -> None: ...
    def get(self, name: str) -> torch.Tensor | None: ...
    def snapshot(self) -> Mapping[str, torch.Tensor]: ...
    def rewind_state(self) -> dict[str, object]: ...
    def apply_rewind_state(self, state: dict[str, object]) -> None: ...
    def validate_rewind_state(self, state: dict[str, object]) -> None: ...
```

`SignalPublisher` is write-only and exposes no read method; `SignalView` is its
read-only counterpart.  `SignalBus` structurally satisfies both, so a run has
one object behind the sink publisher, the sampler view and the rewind payload.

## Transport rules

- a signal name is any non-empty `str`; there is no registry and no reserved
  vocabulary;
- a value is a `torch.Tensor` whose leading dimension is `num_envs`; dtypes
  and trailing shapes are arbitrary (`bool`, integer, float, `(N,)`, `(N, k)`,
  ...);
- `publish` fails fast for an empty/non-string name, a non-tensor value, a
  wrong leading dimension, or a wrong device; the error names the offending
  field and the expected/actual value;
- values are detached and cloned into the bus, and `get` / `snapshot` return
  isolated copies, so no caller can mutate the store through a published or
  read tensor;
- last publish wins; a missing name returns `None` from `get` and is absent
  from `snapshot` (never a silent zero tensor);
- the bus does not clear or reset automatically, and it does not write
  commands.

## Lifecycle

`build_instance` creates exactly one bus per live `RunInstance` **before**
`_build_sinks` and before the bridge/wiring seam.  `_build_sinks` passes it to
a custom sink as `SinkContext.signals`, the bus is stored as
`RunInstance.signals`, and `sampler.bind_signals(signals)` is called exactly
once after `sampler.bind_env(env)` and before any runner exists.  Multiple
runners over one cached instance never rebind.  A sampler opts in by overriding
`bind_signals`; the base implementation is a no-op.  Watchers are not attached
to this bus; they read the step bundle and keep a debug snapshot only.

## Publication timing

The runner order is `env.step(t) -> sampler.update(t) -> sink.on_step(t)`.  A
custom sink therefore publishes a level value after tick `t`'s physics and
observation snapshot, and the next read is `sampler.update(t+1)`.  The command
returned by that update is written to `obs.command` at `t+1` and is first used
by `env.step(t+2)`: in-episode closed loop has two policy-tick latency from the
observed view to the acting command.  Publishing more than once per tick is
allowed but the transport does not deduplicate (last publish wins); producers
must namespace names (`<producer>/<signal>`).  Watchers are debug-only and
publish no signal values.

## Rewind

`SignalBus` implements the `RewindStateProvider` shape with
`name = "signals"`.  `rewind_state()` captures detached private copies of the
store, and `apply_rewind_state()` restores an isolated copy so a checkpoint
stays reusable.  `validate_rewind_state()` performs the same read-only
judgment before the controller touches the world; a malformed payload raises
`RewindGateError`, so a refused rewind is a zero-change rejection.  The bus is
appended to `RunInstance.rewind_providers` once at build.  Signals published by
a custom sink at tick `t` are part of the checkpoint payload at that boundary
and are restored with every other bus value; the sink must not expose the bus
as a second provider.  The watcher provider (`name = "watchers"`) restores only
specs, trigger state and template windows; it owns no signal transport.

## Out of scope

- `SIGNAL` watcher actions and any signal-to-command write;
- automatic signal-to-command or signal-to-observation mapping;
- phase/task/event vocabulary and template signal authoring;
- per-env vectorized watcher trigger state and online benchmark triggers.
