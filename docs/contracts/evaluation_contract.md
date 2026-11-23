# Evaluation context contract

Status: `unigym.evaluation` defines the read-only, boundary-scoped context that
live termination consumes; benchmark scoring and watchers adopt the same
boundary.  Termination is live-only: there is no recorded/offline state path.

The context is a facade over the existing env `record` / observation / runtime
query faces at one engine-thread boundary.  It is not a new backend lifecycle,
not an observation channel and not a replay format.

## 1. Types

```text
EnvView                     read-only view of one live env boundary
EvalContext(env, t=None)    frozen, ephemeral context; no duplicated data fields
EvalQuery                   read-only, env-scoped query adapter
EvalViewError               base failure
EvalViewStaleError          access after close
EvalQueryUnavailableError   no runtime query at this boundary
```

`EnvView` is a concrete owner class with private slots, not a bare protocol, so
the wrong shape cannot be substituted.  It exposes `num_envs`, `env_id`,
`scope_size`, `device`, `has_query`, `query`, `record()`, `obs()`,
`articulation(name)`, `articulation_names`, `articulation_facts(name,
control=None)`, `joint_pos_limits(name)`, `controls`, `with_record(record)`,
`for_env(env_id)` and `close()`.  It exposes no `env`, `native`, `step`, `reset`,
`forward`, write, scratch, query `close`, mutable `ControlRuntime` /
`ArticulationRuntime` object or backend address/command/strategy field.

`EvalContext` carries only `env` and an optional `t`.  The record, observation
snapshot and episode payload are reached through `ctx.env` and the existing
scoring argument; they are not duplicated on the context.

## 2. Lifetime

- The owner creates exactly one view/context per callback boundary and closes
  it in `finally` after the callback.  The watcher engine is the exception:
  it retains one vector-scoped view per tick until the next tick replaces the
  context (so add/test/dump and `on_reset` can read it between callbacks) and
  closes the previous view at that replacement point.
- `close()` is idempotent.
- `close()` never closes the environment or the run-shared query handle.  The
  view only drops its private reference; the run/env still owns the query.
- Every accessor and every `EvalQuery` method after `close()` raises
  `EvalViewStaleError`.  A callback that retains the context and reads it on a
  later episode fails instead of silently scoring the next state.
- Derived views (`with_record`, `for_env`) are children of the view they came
  from.  A child's `close()` is independent and idempotent; closing the parent
  invalidates every descendant through the parent chain, and `_ensure_open`
  follows that chain.  Closing a child never closes its parent.
- If the run/env closes while a view is open, record/obs remain boundary data;
  query access raises the query layer's stale error.  The view owner still
  closes the view.

### 2.1 Per-tick benchmark contexts

A `trigger: both` benchmark task receives one VECTOR-scoped context per policy
tick (`env_id is None`, `t` = the policy tick), created and closed by
`EvaluationSession` inside its own callback.  It is separate from the runner's
boundary view (`bundle["view"]`, closed after `on_episode_end`) and from the
per-env episode-end context created by `EpisodeEvaluator`.  A task cannot reach
a raw env through it.

## 3. Scope

- `env_id is None`: vector scope.  `record()`/`obs()` return every env's rows;
  `query` accepts the caller's env selection unchanged.
- `env_id is int`: per-env scope.  `record()`/`obs()` return only that env's
  rows with a leading axis of one; `scope_size == 1`.
- A per-env `EvalQuery` fixes the env selection for every read and rejects a
  request or contact point outside `env_id` with `EvalViewError`.  It cannot
  read another env.
- `articulation(name)` parses `articulation/<name>/<leaf>` from the scoped
  record and returns a read-only mapping.  A missing articulation fails fast
  with the available names.
- `joint_pos_limits(name)` reads the articulation runtime fact
  (`ArticulationRuntime.joint_pos_limits`), scoped like `record()`.  A declared
  articulation with no limits returns `None`; an undeclared articulation fails
  fast.  Limits are `(lower, upper)`, each `(scope_size, len(joint_names))`, in
  the record/body-control policy joint order: the same order as the live
  `articulation/<name>/joint_pos` leaf.  This is not asset `joint_names` order
  when a body control reorders the joints.

### 3.1 Control and articulation facts

`EnvView` accepts runtime control rows at construction and projects them to
frozen `ResolvedControl` values; plan-row input is not part of this slice.

- `controls: Mapping[ControlKey, ResolvedControl]` is a read-only mapping keyed
  by `(articulation, control name)`.  Every constructor mapping key must equal
  `(control.articulation, control.name)`; a mismatch fails with
  `EvalViewError`.  Only neutral plan fields (`name`, `articulation`, `adapter`,
  `joint_set`, `source`, `joint_names`, `drive`, `target`, `q0`, `role`) are
  exposed; mutable runtime controls and their backend addresses, command and
  strategy are never exposed.
- `articulation_names` is the sorted tuple of declared articulation names.
- `articulation_facts(name, control=None)` returns the existing
  `runtime.controllers.articulation_facts` vocabulary (joint names, asset joint
  names, action group, base body, foot bodies, body sets, contacts) as a
  read-only top-level mapping.  `control=None` requires exactly one body-role
  control for that articulation; zero or multiple raises `EvalViewError`.  An
  explicit `control` name selects that `(articulation, control)` row and fails
  fast when it is undeclared.
- `joint_pos_limits(name)` remains the joint-limit authority and reads the
  private articulation runtime fact.

### 3.2 Derived views

- `with_record(record)` returns a child that preserves the parent's scope,
  observation snapshot, query factory and articulation/control facts and
  substitutes only the record.  It is scope-preserving: a per-env parent yields
  a per-env child.
- `for_env(env_id)` returns a per-env child of a vector view.  It raises
  `EvalViewError` on an already per-env view or an env id outside
  `[0, num_envs)`.
- Derived views share observation references and the underlying record values,
  the lazy query factory and the articulation/control facts; they copy no arrays
  and open no new data.  `with_record` substitutes the record mapping itself.
  Each child builds its own scoped `EvalQuery` lazily when read, so a parent and
  its children cannot leak one another's env selection.

## 4. No mutation

The user surface has no `step`, `reset`, state write, `forward`, scratch world,
native handle or raw backend object.  `EvalQuery` exposes only the read subset
(`backend`, `capabilities`, `model`, `state`, `kinematics`, `geometry`,
`contacts`, `actuators`, `require`) and has no `scratch()` and no `close()`.
It has no catch-all attribute forwarding that could leak either.

## 5. Query availability

Query is optional in the backend contract.  `EnvView.query` is lazy:

- constructing a view never calls `env.query`;
- no query factory means `has_query is False` and `query` raises
  `EvalQueryUnavailableError` naming the backend and telling the caller to use
  `record()`/`obs()`;
- there is no silent record-only fallback;
- when the factory exists, the query handle is created on first access and
  cached for the view lifetime.

## 6. Errors

- `EvalViewError`: invalid construction, out-of-scope request, unknown
  articulation.
- `EvalViewStaleError(EvalViewError)`: any access after close.
- `EvalQueryUnavailableError(EvalViewError, NotImplementedError)`: no runtime
  query at this boundary.

Errors identify the field/name and the expected/actual values.  No error is
converted into an empty result or a default.

## 7. Borrowed values and copies

- `record()` values are borrowed from the boundary snapshot.  They are valid
  only for the callback and must not be retained.  Values with a leading axis
  equal to `num_envs` are sliced for a per-env view; other values (body-name
  tuples, bookkeeping) are returned unchanged.  The mapping itself is
  read-only.
- `obs()` returns a same-schema `ObsSnapshot.select(env_ids)`.  Values keep
  their per-env leading axis; per-contact detail is dropped because detail
  records have no declared env axis.  The values are still borrowed.
- Query results are detached copies produced by the query layer and are safe
  to retain.  The `EvalQuery` handle itself is not safe after `close()`.

## 8. Boundaries (consumer plan)

The boundaries are:

- termination: vector view, after `env.step` and the observation snapshot,
  before reset; pre-reset state.  `Termination.check(ctx)` returns a bool
  `(num_envs,)` mask on the context device; `TerminationGroup` evaluates each
  term once and validates every mask.  The same view is exposed as
  `bundle["view"]` to the step sink and closed after `on_episode_end`.
  `build_terminations` inspects each instantiated term and rejects any `check`
  that is not exactly `check(self, ctx)`, so a legacy `check(self, state)` term
  fails at build;
- benchmark: per-env view inside `EvaluationSession.on_episode_end`, before reset,
  built by `EpisodeEvaluator` (`unigym.runner.evaluation.scoring`).
  `Task.score(ep, ctx)` and `Task.score_detail(ep, ctx, *, retain_frames=False)`
  receive one context scoped to `ep.env_id`; the evaluator closes it in
  `finally` after all tasks for that env.  A task exception is isolated and
  skipped; an old task signature is rejected by `build_tasks` at build time.  A
  sink wired without an env (direct unit use) builds an explicit record/obs-only
  context whose query is unavailable;
- watcher: one vector-scoped view (`env_id is None`) built by
  `WatcherEngine.on_tick` from the step bundle/env, retained until the next
  tick replaces it, then closed.  `WatchContext.record`/`obs`/`num_envs` read
  through it; the expression namespace exposes `view` but no raw `env`.
  `bundle["env"]` is not a watcher API.  Views are not checkpointed; the
  watcher rewind payload is unchanged.

Evaluation order is contract for the consumers that adopt it, not a convention
to reorder for convenience.

## 9. Shared condition predicates

`unigym.conditions` is the canonical predicate layer for genuinely duplicated
state conditions:

```text
Condition.evaluate(ctx: EvalContext) -> (ctx.env.scope_size,) bool
```

- `unigym.conditions` (the package import) is torch-free: it exports only the
  structural `Condition` protocol and `ConditionError` from
  `unigym.conditions.base`.  Concrete modules
  (`unigym.conditions.height`, `.tilt`, `.joint_limit`) may import torch.
- Domain packages (`unigym.terminations`, `unigym.benchmarks`) must import
  concrete conditions locally at use time, never at their package import time.
- A condition is live, read-only, effect-free and stateless.  It accepts only an
  `EvalContext`; it never receives a raw env, backend handle or mutable state,
  and it never produces a termination fact, `WatchResult`, `TickResult` or
  report field.
- `HeightCondition` reads an articulation `base_pos_w` leaf or an explicit
  record key and supports the termination threshold form and the watcher
  threshold/op or band forms.
- `TiltCondition` reads projected `gravity_b` or a quaternion (`base_quat_w`
  style) source; gravity thresholds are radians and quaternion thresholds are
  degrees.
- `JointLimitCondition` reads policy-order `joint_pos` plus the articulation's
  soft limits.  `band` mode matches `JointLimitTermination`; `magnitude` mode
  matches the watcher `joint_limit` template.  Missing limits produce an
  all-false mask; the watcher adapter still pre-validates limits to preserve its
  `WatchError`/fail-safe contract.
- Domain ownership is unchanged: terminations own fail-fast masks and episode
  facts; watchers own `WatchResult`, messages/data, signals, actions and the
  trigger layer; benchmark tasks own `TickResult`/`TickTrace`.  A task may
  compose a condition inside `on_tick(ctx)`; there is no core `Task.condition`
  API.

## 10. Non-goals

- No deterministic replay or offline re-scoring guarantee.
- No removal of `EpisodeData`, reports or history.
- No writable/scratch/native evaluation surface.
- No cross-env read from a per-env view.
- No recorded/offline termination state or reconstruction from stored episode
  data.

## 11. Removed episode filters

The episode filter subsystem is removed in this release.  There is no offline
filter chain, no generic recorded-termination filter and no compatibility shim;
a run config that declares an episode-filter section is rejected fail-fast at
spec construction with a migration hint.  Reintroducing filters later is a new,
data-only feature over `EpisodeData`, never a revival of the recorded
termination state path this contract deletes.
