# Signal bus contract

- Status: active
- Owner module: `unigym/signals.py`

A `SignalBus` is a run-scoped, semantics-free transport for per-env signal
tensors.  Producers (phase 2A: user code and tests) publish values; a
`CommandSampler` receives the bus through the read-only `SignalView` protocol
and interprets the names itself.  Core defines transport only: it never
interprets a name, dtype, trailing shape, phase or event, and it never maps a
signal to a command or an adapter observation.

## Surface

```python
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

`build_instance` creates exactly one bus per live `RunInstance`, stores it as
`RunInstance.signals`, and calls `sampler.bind_signals(signals)` exactly once
after `sampler.bind_env(env)` and before any runner exists.  Multiple runners
over one cached instance never rebind.  A sampler opts in by overriding
`bind_signals`; the base implementation is a no-op.

## Rewind

`SignalBus` implements the `RewindStateProvider` shape with
`name = "signals"`.  `rewind_state()` captures detached private copies of the
store, and `apply_rewind_state()` restores an isolated copy so a checkpoint
stays reusable.  `validate_rewind_state()` performs the same read-only
judgment before the controller touches the world; a malformed payload raises
`RewindGateError`, so a refused rewind is a zero-change rejection.  The bus is
appended to `RunInstance.rewind_providers` once at build.

## Out of scope

- watcher-produced signals, `WatchResult.signals`, `SIGNAL` actions;
- per-env producer state and watcher rewind;
- phase/task/event vocabulary and template or expression authoring;
- hot-add behavior and online benchmark triggers;
- automatic signal-to-command or signal-to-observation mapping.
