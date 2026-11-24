# Watchers contract: lifecycle and values

- Status: active
- Owner module: `unigym/watchers`

This record defines the watcher evaluation lifecycle and the read-only values a
watcher can observe.  Watchers are debug/instrumentation only: they publish no
signals and Core transports no watcher-produced values.  Transport rules for
the run-scoped value store live in the values/runner contracts; this record only
states how watchers read them.

## `WatchResult`

```python
@dataclass
class WatchResult:
    fired: bool
    level: Level = Level.INFO
    message: str = ""
    data: dict = field(default_factory=dict)
    actions: tuple = ()
```

- there is no `signals` field (and no value payload);
- `as_dict()` (event/dry-run logs) emits exactly `fired`, `level`, `message`,
  `data` and `actions`; it never emits a `signals` key;
- a watcher result cannot publish, mutate or own a run value.

## Evaluation context

Watchers consume the C10 step boundary from `unigym.evaluation`
(`EnvStepContext`) and its borrowed read-only `EnvView`.

- `WatcherEngine.on_tick(t, bundle)` reads exactly `bundle["step"]` and requires
  an `EnvStepContext`.  It evaluates with one live
  `WatchContext(step=step, window_store=self.window_cache)`.  The engine builds
  no second `EnvView`; the step's view is the only evaluation view.
- The runner owns that boundary view.  The engine borrows it while the tick is
  live and never closes it.
- After evaluation (including fail-safe processing) the engine replaces its
  retained context with a `WatchDebugSnapshot` context: private copies of the
  step view's record plus non-camera observations and the step's immutable
  values snapshot, wrapped in a synthetic debug `EnvView`.  Camera/source-camera
  observation keys are excluded.  No live step or view is retained;
  `add`/`test`/`dump` between ticks read the debug snapshot.
- The debug snapshot intentionally carries record/obs/values but **no
  articulation runtime facts** (and no query).  Articulation/query-dependent
  templates (for example `joint_limit`) therefore require a live `step` context
  between ticks and otherwise fail with the existing teaching `WatchError`.
- `WatchContext.view` is a read-only property: `step.view` when `step` is
  present, otherwise `debug.view` (or `None`).  `record`, `obs` and `num_envs`
  read that view.  Only an explicitly constructed offline context with neither
  `step` nor `debug` falls back to the bundle snapshot.
- `WatchContext.values` returns `step.values` when the step is present,
  otherwise the debug snapshot's value mapping (empty when absent).
- `on_reset(env_ids, step=None, bundle=None)` receives the bridge's still-open
  final step and the final episode-end bundle when available.  It builds a
  temporary context over that step and bundle (so `ctx.bundle` carries
  `reset_facts`/`episode_results`/terminations) for the callable `on_reset`
  hooks and does not retain it; without a step it uses the debug context.  The
  engine never retains a live view for the reset hook.
- template window buffers are owned by the explicit context
  `window_store` (the engine passes its own `window_cache`), so two engines have
  isolated windows.

### Expression / callable namespace

The expr namespace and callable `check(ctx)` receive `view`, `values` and the
key face as read-only handles.  `env` is **absent** from the namespace and is
not a `WatchContext` field: no raw env is reachable through
`eval_namespace`.  `step` in the namespace remains the tick alias (`self.t`),
not the step context.  `bundle` remains bound for step-record and observation
facts, but it is not an env escape.

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
- views and debug snapshots are boundary handles, not checkpoint state; they
  are not part of the watcher payload;
- the watcher engine neither owns nor restores any signal transport.

## Stateless callable limitation

A callable watcher's rewind payload re-creates the instance from its spec; it
cannot reproduce internal instance state.  Stateful callable watchers are
explicitly unsupported (no silent restore promise).

`setup()` is different: it is the framework lifecycle hook, not opaque callable
state.  On restore the engine re-creates the callable and forces
`setup_done = False`, so the next evaluated tick runs `setup()` against the live
context before `check()`.  Restoring `setup_done = True` without re-running
setup is never valid.

## Condition-backed templates

`height`, `tilt` and `joint_limit` delegate their predicate to the canonical
conditions in `unigym.conditions` while the watcher template keeps ownership of
`level`, `message`, `data` and `actions`.

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
- Trigger edge/level/once/cooldown, action execution and the rewind payload
  remain `WatcherEngine`-owned and unchanged.

## Out of scope

- template/expression value or signal authoring;
- per-env vectorized edge/cooldown state;
- core phase/event/TaskRuntime types;
- benchmark online triggers;
- `SIGNAL` actions or any command write.
