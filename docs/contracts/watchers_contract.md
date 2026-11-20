# Watchers contract: signals

- Status: active
- Owner module: `unigym/watchers`

This record defines the watcher-produced signal face.  Transport rules (names,
dtypes, leading axis, isolation, validation) live in `signal_bus_contract.md`;
core owns no phase/event vocabulary and performs no signal-to-command mapping.

## `WatchResult.signals`

```python
@dataclass
class WatchResult:
    fired: bool
    level: Level = Level.INFO
    message: str = ""
    data: dict = field(default_factory=dict)
    actions: tuple = ()
    signals: Mapping[str, torch.Tensor] = field(default_factory=dict)
```

- names are user-defined non-empty strings;
- values are `torch.Tensor` objects whose leading dimension is `num_envs`;
  dtype and trailing shape are arbitrary;
- no name/dtype interpretation and no reserved names;
- `as_dict()` (event/dry-run logs) records signal **names and shapes** only,
  never raw tensors.

## Producers and publication

- callable and expression watchers may return a `WatchResult` carrying signals;
- templates cannot emit signals in this slice;
- every **evaluated** watcher tick publishes its `result.signals`, even when
  `fired` is false (level semantics).  The trigger/action layer stays
  independent: no signal writes a command and no `SIGNAL` action exists;
- a missing name is absent, never a silent zero tensor;
- values persist until overwritten; neither the engine nor the bus clears
  signals automatically (the user owns lifetime and may publish on reset);
- the whole mapping is validated before the first store, and a malformed
  mapping/value is a watcher failure: the existing fail-safe records an error
  event/strike and never crashes the engine loop.  No partial tick is stored;
- with no bus (offline engines) publication is a no-op.

## Rewind

`WatcherEngine` implements `RewindStateProvider` with `name = "watchers"`:

- the payload holds the serialized specs, each spec's trigger state
  (`prev` / `hold` / `cooldown_until` / `strikes` / `fired` / `disabled` / ...)
  and the framework-managed template window cache;
- `validate_rewind_state()` rejects malformed payloads read-only;
  `apply_rewind_state()` re-validates, re-compiles expr/guards, re-creates
  callable instances and restores the payload in place;
- specs ride the checkpoint, so a watcher hot-added or cleared after a snapshot
  is reverted by a rewind;
- the `SignalBus` provider (`name = "signals"`) restores the transported values;
  the two providers together restore producer and transport state.

## Stateless callable limitation

A callable watcher that publishes signals must be stateless in this slice.  The
rewind payload re-creates the instance from its spec; it cannot reproduce
internal instance state.  Stateful signal-producing callables are explicitly
unsupported (no silent restore promise).

`setup()` is different: it is the framework lifecycle hook, not opaque callable
state.  On restore the engine re-creates the callable and forces
`setup_done = False`, so the next evaluated tick runs `setup()` against the live
context before `check()`.  Restoring `setup_done = True` without re-running
setup is never valid.

## Out of scope

- template/expression signal authoring and a derived signal language;
- per-env vectorized edge/cooldown state;
- core phase/event/TaskRuntime types;
- benchmark online triggers;
- `SIGNAL` actions or any command write.
