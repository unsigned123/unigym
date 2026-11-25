# Value store contract

- Status: active
- Owner module: `unigym/values.py`

A `ValueStore` is the one run-scoped, semantics-free transport for managed
values.  Producers publish tensors or JSON-safe trees; a `CommandSampler` and
the step boundary read them through the read-only `ValueView` protocol.  A
custom sink receives the write-only `ValuePublisher` face through
`SinkContext.values`.  Core owns transport, ownership and boundary validation
only: it never interprets a value name, dtype, trailing shape or JSON structure,
and it never maps a value to a command or an adapter observation.

This contract replaces the retired flat signal API.  There is no `SignalBus`,
`SignalView`, `SignalPublisher`, `bind_signals` or `"signals"` rewind provider
in core; the old flat `{name: value}` checkpoint payload is rejected.

## Surface

```python
Value: TypeAlias = torch.Tensor | JsonValue

class ValueView(Protocol):
    def get(self, name: str) -> Value | None: ...
    def snapshot(self) -> Mapping[str, Value]: ...
    def __contains__(self, name: object) -> bool: ...

class ValuePublisher(Protocol):
    def publish(self, name: str, value: Value) -> None: ...
    def clear(self, name: str) -> None: ...

class StepValues(Mapping[str, Value]):
    # immutable snapshot; every read returns an isolated copy
    def snapshot(self) -> Mapping[str, Value]: ...

class ValueStore:
    name = "values"

    def __init__(self, num_envs: int, device: str | torch.device): ...
    def publisher(self, owner: str) -> ValuePublisher: ...
    def publish(self, owner: str, name: str, value: Value) -> None: ...
    def get(self, name: str) -> Value | None: ...
    def snapshot(self) -> StepValues: ...
    def subscribe(
        self,
        callback: Callable[[str, Value | None], None],
        *,
        names: frozenset[str] | None = None,
    ) -> Callable[[], None]: ...
    def clear(self, owner: str, name: str | None = None) -> None: ...
    def rewind_state(self) -> Mapping[str, object]: ...
    def validate_rewind_state(self, state: Mapping[str, object]) -> None: ...
    def apply_rewind_state(self, state: Mapping[str, object]) -> None: ...
```

`ValueView` is read-only.  `ValuePublisher` is write-only and bound to one
owner; it exposes no `get`, `snapshot` or rewind method.  The store satisfies
`ValueView`, and `publisher(owner)` returns the bound write-only face.  One
store is behind every publisher, the sampler view, the step snapshot and the
rewind payload.  `StepValues` is the immutable boundary snapshot: it supports
mapping reads only and never aliases live store state.

`ValueOwnershipError` (a `ValueError`) reports an ownership conflict and names
the value, the requesting owner and the claiming owner.

## Transport

A name is any non-empty `str`; there is no registry and no reserved vocabulary.
A value is exactly one of:

- a dense (strided) `torch.Tensor` whose leading dimension is `num_envs` and
  whose device is the store device.  Dtypes and trailing shapes are arbitrary
  (`bool`, integer, float, `(N,)`, `(N, k)`, ...);
- a JSON-safe tree (scalars, lists, string-keyed mappings) validated by the
  shared JSON boundary.  Floats must be finite and tuples normalize to lists.

Arbitrary Python objects, numpy arrays, non-string mapping keys, sparse tensors
and non-finite floats are refused at the boundary with an error naming the
field and the offending value.  Core transports values; it never performs
Python-object pickling or arbitrary object transport.

Values are copied on the way in and on every read.  Tensor values are detached
clones; JSON values are deep copies.  A caller cannot mutate the store through
a published or read value, and a retained `StepValues` cannot observe later
store mutations.

A missing name is absent: `get` returns `None`, `__contains__` is `False` and
`snapshot` omits it.  It is never a silent zero or default value.

The store is instance-owned and single-threaded.  Publish, read, clear,
snapshot and rewind all run on the owning thread (the engine thread in a live
run).

## Ownership

Writers are first-writer-wins.  The first owner to publish a name claims it for
the life of the store.  A different owner cannot publish or clear that name and
receives `ValueOwnershipError`; a refused publish does not claim the name.
`clear(owner, name)` removes the value but retains the claim, so only the
claiming owner can publish the name again.  `clear(owner)` removes every value
currently owned by that owner and leaves other owners' values and all claims
unchanged.  Clear never releases a claim.  A rewind restores the checkpoint's
writer map exactly (it does not merge), so names claimed after the checkpoint
are released and names present in the checkpoint keep their owner.  No
operation transfers a live claim to a different owner.

## Subscriptions

`subscribe(callback, *, names=None)` registers one synchronous value-change
callback and returns an idempotent unsubscribe callable.  Subscriptions are
run-scoped and instance-owned: they live on one store, are not module globals,
and are not part of the rewind payload.

- `names=None` delivers every name; otherwise only the exact listed names are
  delivered.  The filter is a `frozenset[str]` of non-empty names with no
  wildcard, pattern or DSL syntax.
- Callbacks run in registration order.  Unsubscribing removes that one
  subscription and is safe to call more than once.
- A successful `publish` notifies matching callbacks with `(name, value)`
  after the value is stored.  A `clear` that actually removes a value notifies
  once per removed value with `(name, None)`; clearing an absent name is
  silent.  `clear(owner)` notifies once for each value it removed.
- There is no notification for a refused publish, a read, `snapshot`, a
  `clear` that removed nothing, or a rewind restore.
- Callbacks receive isolated copies with the same semantics as `get` and
  `snapshot`; mutating a delivered value cannot change the store or another
  callback's copy.
- A callback exception propagates out of the `publish`/`clear` that fired it;
  the store mutation has already happened.  For `clear(owner)` every owned value
  is removed before notification starts, so a raising callback stops the
  remaining notifications but leaves the removal complete.  Callbacks must stay
  small and non-throwing.
- A callback must not mutate the store.  A re-entrant `publish`, `clear` or
  `apply_rewind_state` from a callback raises a teaching `RuntimeError` (store
  mutation during notification) instead of recursing or deadlocking.
- Subscribing and notifying are single-threaded on the owning thread, like
  every other store operation.

`StepValues` stays immutable and is not subscribable; `publisher(owner)` is
write-only and exposes no `subscribe`.  Only the store itself owns
subscriptions.

## Rewind

`ValueStore` implements the `RewindStateProvider` shape with `name = "values"`.
The payload is the explicit schema-2 mapping:

```text
{"schema": 2, "values": {name: value}, "writers": {name: owner}}
```

`rewind_state()` returns private copies.  `validate_rewind_state()` performs the
same read-only judgment before the controller touches the world, and
`apply_rewind_state()` validates the whole payload before replacing values and
writers atomically; a refused restore leaves every value and claim unchanged.
Restore replaces the writer map (it does not merge), so the ownership map
returns exactly to the captured state.  A cleared name keeps its owner across
capture and restore.

The old flat `{name: value}` signal-style payload, a wrong schema version, an
unknown key, a value without a writer or an invalid transported value raises
`RewindGateError`; it is not silently upgraded.

## One store per run

`build_instance` creates exactly one `ValueStore(num_envs, device)` per live
`RunInstance`, before `_build_sinks` and before `_build_bridge_and_wiring`:

- `_build_sinks` receives `values.publisher("sink")` as `SinkContext.values`;
- the store is stored as `RunInstance.values`;
- `sampler.bind_env(env)` and `sampler.bind_values(values)` run once at the
  live-instance construction seam, before any runner exists;
- the store is appended to `rewind_providers` exactly once as the `"values"`
  provider;
- the runner publishes the explicitly selected observations each tick through
  `values.publisher("observation")`.

A custom sink must not expose the store again as a second provider.  Watchers
are debug-only: they receive no publisher, publish no values, and join rewind as
`"watchers"` for specs, trigger state and template windows only.

Direct `SyncRunner` construction may pass `values=None`; it then creates no
store and every step boundary carries an empty `StepValues`.  Reusing one
cached `RunInstance` across runners never rebinds.

## Selected observation publication

`observations.values` selects canonical, small state observation keys; camera
and depth keys are rejected when the observation registry is built.  Each tick,
before the step boundary snapshot and `sink.on_step`, the runner publishes the
selected values under `obs/<key>` with owner `"observation"`; a selected key
that has no current value is cleared.  These names are ordinary managed values: they are visible to a
sampler's `ValueView`, to `bundle["step"].values`, and to the `"values"` rewind
payload.  Cameras and depth are never published.

## Publication timing

The runner order is `env.step(t) -> sampler.update(t) -> sink.on_step(t)`.  A
custom sink therefore publishes a value after tick `t`'s physics and observation
snapshot, and the sampler's next **polling** read is `update(t+1)`.  The command
returned by that update is written to `obs.command` at `t+1` and is first used
by `env.step(t+2)`: the polling in-episode closed loop has two policy-tick
latency from the observed view to the acting command.  Publishing more than once
per tick is allowed; last publish by the same owner wins.  Producers must
namespace names (`<producer>/<value>`); the transport has no registry.

A **subscription** callback runs synchronously inside that publish, still in
`on_step(t)`.  A control consumer can therefore mutate its live command buffer
in place and be acted on by `env.step(t+1)`: one policy-tick latency.  This
holds only when the callback updates the same buffer the runner holds for tick
`t` (the tensor returned by the current `update`, normally `self.cmd`) and that
`update` returns the owned buffer rather than a copy.  Reassigning a fresh
tensor in the callback does not propagate to the runner; the new value then
waits for the next `update` return and the polling path's two-tick latency.

## Out of scope

- automatic value-to-command or value-to-observation mapping;
- watcher value publication;
- phase/task/event vocabulary, typed value schemas or value registries;
- per-env vectorized watcher triggers;
- an event bus, asynchronous delivery, delivery history, persisted
  subscriptions or wildcard/pattern name matching.
