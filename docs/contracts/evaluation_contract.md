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
`articulation(name)`, `joint_pos_limits(name)` and `close()`.  It exposes no
`env`, `native`, `step`, `reset`, `forward`, write, scratch or query `close`.

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
- If the run/env closes while a view is open, record/obs remain boundary data;
  query access raises the query layer's stale error.  The view owner still
  closes the view.

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
  before the step sink; pre-reset state.  `Termination.check(ctx)` returns a
  bool `(num_envs,)` mask on the context device; `TerminationGroup`
  evaluates each term once, validates every mask and closes the context
  before the sink observes the bundle.  `build_terminations` inspects each
  instantiated term and rejects any `check` that is not exactly
  `check(self, ctx)`, so a legacy `check(self, state)` term fails at build;
- benchmark: per-env view inside `BenchmarkSink.on_episode_end`, before reset.
  `Task.score(ep, ctx)` and `Task.score_detail(ep, ctx, *, retain_frames=False)`
  receive one context scoped to `ep.env_id`; the sink closes it in `finally`
  after all tasks for that env.  A task exception is isolated and skipped; an
  old task signature is rejected by `build_tasks` at build time.  A sink wired
  without an env (direct unit use) builds an explicit record/obs-only context
  whose query is unavailable;
- watcher: one vector-scoped view (`env_id is None`) built by
  `WatcherEngine.on_tick` from the step bundle/env, retained until the next
  tick replaces it, then closed.  `WatchContext.record`/`obs`/`num_envs` read
  through it; the expression namespace exposes `view` but no raw `env`.
  `bundle["env"]` is not a watcher API.  Views are not checkpointed; the
  watcher rewind payload is unchanged.

Evaluation order is contract for the consumers that adopt it, not a convention
to reorder for convenience.

## 9. Non-goals

- No deterministic replay or offline re-scoring guarantee.
- No removal of `EpisodeData`, reports or history.
- No writable/scratch/native evaluation surface.
- No cross-env read from a per-env view.
- No recorded/offline termination state or reconstruction from stored episode
  data.

## 10. Removed episode filters

The episode filter subsystem is removed in this release.  There is no offline
filter chain, no generic recorded-termination filter and no compatibility shim;
a run config that declares an episode-filter section is rejected fail-fast at
spec construction with a migration hint.  Reintroducing filters later is a new,
data-only feature over `EpisodeData`, never a revival of the recorded
termination state path this contract deletes.
