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

## Evaluation context

Watchers consume the read-only evaluation boundary from `unigym.evaluation`
(`EnvView`).  This is the C4 convergence recorded in the live-first plan; the
watcher trigger/action engine, key face, signal publication, rewind provider
and lifecycle are unchanged.

- `WatcherEngine.on_tick` builds exactly one vector-scoped `EnvView`
  (`env_id is None`) per tick from `bundle["env"]` or the engine's wired env.
  The view carries the step record and observation snapshot from the bundle,
  `num_envs`, `device`, the backend label, a lazy query factory when the env
  exposes `query`, and the articulation runtime facts.  The raw env is not
  retained by the context;
- `WatchContext.view: EnvView | None` is the read-only face.  `record`,
  `obs` and `num_envs` read through the view when it is present; the
  bundle-only fallback exists only for offline unit tests that construct a
  context without a live env;
- the view is retained across ticks so add/test/dump and callable `on_reset`
  read the current tick's boundary between callbacks.  Replacing `_last_ctx`
  on the next tick closes the previous view; it is not closed in the callback
  epilogue.  Closing a view never closes the env or the run-shared query;
- views are boundary handles, not checkpoint state.  The rewind payload is
  unchanged (specs, trigger state, template windows); a view is rebuilt from
  the live env after a rewind.

### Expression / callable namespace

The expr namespace and callable `check(ctx)` receive `view` as the read-only
handle.  `env` is **removed** from the expression namespace (owner-approved
watcher contract break): no raw env is reachable through `eval_namespace`.
`bundle` remains bound for compatibility, but `bundle["env"]` is not a watcher
API and must not be read by a predicate.  Watcher state reads go through
`view`, `record`, `obs` or the key face.

### `joint_limit` migration

`joint_limit` reads per-articulation limits through the view:

- `articulation` is the explicit selector; when omitted, the template infers
  the single articulation exposing `record/articulation/<name>/joint_pos`
  (multiple leaves require an explicit articulation);
- limits come from `ctx.view.joint_pos_limits(articulation)` in
  record/body-control policy order, the same order as the joint-position leaf;
- the template no longer reads `ctx.env`, `policy_joint_pos_limits` or
  `joint_pos_limits_by_articulation`;
- the retired fixed `key: record/joint_pos` is rejected with a migration hint;
  a namespaced `key: record/articulation/<name>/joint_pos` remains valid and
  selects that articulation.


## Condition-backed templates

`height`, `tilt` and `joint_limit` delegate their predicate to the canonical
conditions in `unigym.conditions` while the watcher template keeps ownership of
`level`, `message`, `data`, `actions` and signals.

- The condition returns a vector bool mask.  When the watcher declaration has no
  explicit `env` parameter, `WatchResult.fired = bool(mask.any())` (global any).
  When `env: <index>` is explicit, `fired = bool(mask[env])`, retaining the
  selected-env decision.
- This is a scalar trigger adapter, not per-env trigger state.  With no explicit
  `env`, a fire caused by one env still produces the template's existing
  evidence fields computed for the diagnostic env (env 0 for `height`/`tilt`);
  the message/data field shapes are unchanged.
- `joint_limit` keeps the watcher magnitude mode (default margin `0.1`) and
  still reports all violating env indexes in `data["envs"]` plus `worst_over`,
  even when an explicit `env` selects the fired decision.
- `height` keeps the threshold/op and inclusive-band forms; `tilt` keeps the
  quaternion-degree form.  Missing keys, missing views and missing articulation
  limits remain data-shaped `WatchError`s, so the engine's degrade/strike
  fail-safe is unchanged.
- Trigger edge/level/once/cooldown, action execution, signal publication and the
  rewind payload remain `WatcherEngine`-owned and unchanged.

## Out of scope

- template/expression signal authoring and a derived signal language;
- per-env vectorized edge/cooldown state;
- core phase/event/TaskRuntime types;
- benchmark online triggers;
- `SIGNAL` actions or any command write.
