# Runner contract (benchmark facts)

The `SyncRunner` callback stream is the single source of episode facts.  The
`bundle` handed to sinks carries:

```text
t             policy tick counter
record        per-env backend record (canonical keys + contact facts)
terminated    (N,) bool, live term-group mask (never sink terminals)
alive         (N,) bool, not terminated and finite policy observation
term_metrics  per-group/total termination firing fractions
reset_state   {env_id: reset validity/reason state}
              (the due ids are StepResult.reset_env_ids, not a bundle key)
terminations  {env_id: EpisodeTermination}   # term group + core horizon
              plus the sink drain merged on this tick (see below)
step_index    (N,) int64 episode-local tick per env; each env advances once per
              policy tick and a reset env returns to 0 (frame 0 is tick 1)
time_s        (N,) float32 episode-local time (step_index * policy_dt), or None
              when no policy period is known
run_identity  RunIdentity (benchmark runs)
reset_facts   {env_id: {reset_index, reset_sample, reset_valid, reset_reason}}
              present on on_episode_end for the ending episode
episode_results {env_id: (EpisodeResultView, ...)}
              present on on_episode_end when an EvaluationSession scored the
              episode; per-env, task declaration order, read-only
view          the tick's one read-only boundary EnvView; valid at on_step /
              on_episode_end, None at on_step_end
step          EnvStepContext for this tick: the borrowed view plus the
              pre-on_step value snapshot; valid at on_step / on_episode_end,
              None at on_step_end
final_step    EnvStepContext for a due episode-end tick: same view plus the
              post-on_step value snapshot (sink publications included); built
              only on a due tick, never on rewind, None at on_step_end
```

Core composes no built-in `EvaluationSession`, so a run carries no
`episode_results` unless a custom sink writes them; a sink terminal request
ends envs but is not a benchmark `EpisodeResultView` (see
`command_sampler_contract.md`).

The bundle carries no live env.  Internal runtime hooks that need the engine
receive it at construction: `WatcherEngine.on_tick` reads exactly
`bundle["step"]` and retains only a debug snapshot between ticks, while
`Bridge`/`MetricsWriter` receive the step-clock capability at construction.
Termination and benchmark evaluation receive the read-only
`unigym.evaluation` context; a sink or task reads `bundle["view"]` rather than
walking a live backend.  The built-in `CollectSink` keeps no live env: it is
constructed with a read-only build-time facts snapshot.

## Boundary view lifetime

Each tick builds exactly one vector-scoped `unigym.evaluation.EnvView` from the
post-step record and observation snapshot, before the termination check.  The
runner owns that view and uses it for both termination and the sink callbacks:

- `bundle["view"]` is the same object at `on_step` and `on_episode_end`, and
  both callbacks may read it;
- `bundle["view"] is None` at `on_step_end`: the runner closes the view before
  the reset chain starts and before `on_step_end` runs;
- `on_step_end` still carries the current pre-reset `bundle["record"]` and
  `bundle["obs"]` facts; only the view is invalidated;
- a sink must not close the view, retain it past its callback, store it in
  rewind state or put it in a checkpoint provider payload.

The runner closes the view on every path: normal tick, termination-check
exception, sink exception, `on_episode_end` exception, reset-mask exception,
reset-chain failure and `RewindBoundary`.  A termination-check exception closes
the view and invokes no sink callback.  A rewind closes the view first and the
abandoned branch runs no `on_episode_end`, reset chain or `on_step_end`.

## Step value contexts

Each tick builds an `unigym.evaluation.EnvStepContext` (`bundle["step"]`) from
the runner-owned boundary view and an immutable `StepValues` snapshot of the
run's `ValueStore` taken **before** `sink.on_step`.  On a due episode-end tick,
after `on_step` returns (and after any sink publications) and before
`on_episode_end`, the runner builds a second context as `bundle["final_step"]`
with a fresh post-`on_step` snapshot.  The timing is fixed:

- `step.values` is the pre-`on_step` snapshot; sink publications during
  `on_step` are deliberately absent from it;
- `final_step.values` is the post-`on_step` snapshot and includes sink
  publications;
- `final_step` exists only on a due tick (`reset_env_ids` non-empty); a
  non-due tick has no `final_step` key;
- a rewind abandons the branch before the due block, so a rewind tick never
  builds `final_step`;
- the runner closes the boundary view, then sets `bundle["view"]`,
  `bundle["step"]` and (when present) `bundle["final_step"]` to `None` before
  the reset chain and `on_step_end`;
- the watcher engine reads exactly `bundle["step"]` during the tick and retains
  only its immutable debug snapshot between ticks.

## Sink protocol

The public contract is `unigym.sink.Sink`.  It is re-exported as `unigym.Sink`
and `unigym.runner.Sink`; `unigym.run.instance.Sink` resolves to the same
object.  It has exactly four required callbacks, driven only by `SyncRunner`:

- `on_step(t, bundle)` — pre-reset data side.
- `on_episode_end(env_ids, bundle)` — episode-end data side, before the reset
  chain.
- `on_step_end(t, bundle)` — post-reset metrics side.
- `on_run_end() -> dict` — flush the run and return the sink summary.

A sink may additionally implement an optional capability protocol:

- `unigym.sink.RewindAwareSink` — `on_rewind(info)` receives one rewind/fork
  mark.  It labels the abandoned branch; it is not state restoration.
- `unigym.sink.StoppingSink` — `stop_when() -> bool` declares a sink-owned
  early stop decision.  `SyncRunner.run` polls it at the top of every step;
  stopping the run remains the runner's job.
- `unigym.sink.TerminalSink` — `drain_terminations() -> Mapping[int,
  Termination]` declares one-shot per-env terminal facts.  A terminal sink is
  still a four-callback `Sink`; the drain is an optional capability, not a
  fifth required callback.  `Termination` is
  `unigym.core.episode_termination.Termination`.

These protocols describe optional capabilities only; publishing them adds no
new runner control flow.  `FanOutSink` composes sinks in construction order,
forwards every callback to each child and structurally satisfies `Sink`.  It
always exposes `drain_terminations` and merges the children that declare the
capability: a child that returns `None` or a non-mapping is a `TypeError`, and
the same env id returned by two children is a `ValueError` naming both
sources.  `on_run_end` keys summaries by sink class name; two children of the
same class collide and fail instead of silently overwriting.

After a successful `on_step`, the runner calls `drain_terminations()` exactly
once when the sink declares it.  The drain is a one-shot per-env episode-end
request; when `on_step` raises (including `RewindBoundary`) the runner never
drains, never calls `on_episode_end` and never resets.  The returned
`Mapping[int, Termination]` is validated before any runner state changes:

- keys are integer env ids in `[0, num_envs)`; `bool` keys are rejected;
- values are `unigym.core.episode_termination.Termination`, not an
  `EpisodeTermination` or a mapping;
- `source` is `TerminationSource.CUSTOM` and `code` starts with `custom.`;
- a missing `step_index` is stamped with the episode-local step (frame 0 is
  tick 1), never the global run tick; a provided value must equal it;
- a missing `time_s` is stamped with the tick's float32
  `episode_step * policy_dt` (the same value as `bundle["time_s"]`) when the
  period is known; a provided value must agree with that canonical value within
  float32 tolerance.

A malformed return (not a `Mapping`, a bad id, a non-`Termination`, a
non-custom source/code or a mismatched stamp) fails with a teaching error
before the bundle or due mask is mutated.

Lifecycle:

1. `on_step(t, bundle)` — pre-reset data side.  `bundle["step_index"]` and
   `bundle["time_s"]` are per-env EPISODE-LOCAL arrays: each env's own tick
   after this step (`1` on its first post-reset step), not the global run tick
   `t+1`.  They equal `t+1` only for an env that has never reset; with per-env
   resets, envs in one bundle can carry different episode-local ticks.
2. `drain_terminations()` — only after `on_step` returns.  The validated sink
   facts merge into `bundle["terminations"]` with
   `EpisodeTermination.from_terminations`, so core facts are preserved and the
   core outcome precedence still chooses the primary.  The sink env ids union
   into the due mask before reset-policy evaluation.  When `auto_reset` is
   false and the drain is non-empty, the tick raises a teaching `RuntimeError`
   before any due mutation: a sink terminal end requires the reset chain.
3. `on_episode_end(due, bundle)` — BEFORE `env.reset_envs`.  `due` is
   `unique(core episode clock ∪ sink terminal ids ∪ reset policy)`;
   `reset_facts` describe the episode's START (captured at its reset, tick 0);
   the termination facts are the final tick's.
4. `sampler.on_episode_end(due, bundle.get("episode_results") or {})` — AFTER
   the sinks (they publish the views) and BEFORE `sampler.reset`.  Runs even
   when `auto_reset` is false.
5. reset chain (`env.reset_envs` / `adapter.on_reset` / `sampler.reset`).
6. The runner captures the NEXT episode's pre-policy-step sample from
   `env.obs.snapshot()` and the backend's `ResetOutcome`.

`StepResult.reset_env_ids` is the final due set (core clock ∪ sink terminals ∪
reset policy).  `StepResult.terminated` remains the core term-group mask and
never includes sink terminal env ids.

## Custom sinks

A run may inject one custom sink by API
(`Run(..., sink=SinkSpec(factory=...))`) or by the top-level config section:

```yaml
sink:
  factory: acme_pack.sinks:make_benchmark_sink
  options:
    target_episodes: 10
```

A string factory is `module:callable`; a callable is API-only.  The factory is
resolved before env construction and called exactly once as
`factory(ctx: SinkContext) -> Sink`.  `SinkContext` carries construction
metadata: `spec`, `out_dir`, `num_envs`, `policy_dt`, `seed`, `action_groups`,
`run_identity`, `options` and the run's write-only `values` face.  It deliberately contains no live env, writer, logger or bridge; a sink
receives runner-owned resources through the callback bundle, never at
construction.  `values` is a `unigym.values.ValuePublisher` (never the read
view); `None` means the build provided no publisher, so a sink that needs closed
loop must fail fast rather than fall back.  A factory result missing a required
callback, declaring a non-callable optional callback (`on_rewind`, `stop_when`,
`drain_terminations`) or declaring an invalid rewind provider fails with a
teaching `TypeError` before the run starts.

### Sink value publication and closed-loop latency

A custom sink stores `ctx.values` and publishes level values from `on_step`:

```python
def on_step(self, t: int, bundle) -> None:
    self.values.publish("acme/probe", value)  # (num_envs, ...) tensor or JSON
```

Transport rules are exactly `value_store_contract.md` (non-empty name, dense
tensor with leading dimension `num_envs` or JSON-safe tree, store device,
detached/deep-copied, first-writer-wins ownership).  Names are namespaced by the
producer; there is no registry and the last publish by the same owner wins.
Watchers are debug-only, publish no values and never overwrite a sink value.

The runner order is fixed and unchanged:

```text
env.step(t) -> sampler.update(t) -> sink.on_step(t)
```

Therefore in-episode closed loop has **two policy-tick latency**:

1. a sink publishes in `on_step(t)`, after that tick's physics and observation
   snapshot;
2. the `CommandSampler.update` for tick `t+1` is the next read, because it runs
   after `env.step(t+1)`;
3. the command returned by `update(t+1)` is written to `obs.command` at `t+1`
   and is first used by `env.step(t+2)`.

This is the existing sampler placement, not a sink-seam bug.  Deterministic
direct-read tests must assert this two-tick latency, not an aspirational
one-tick loop.

Sink selection is explicit and mode-free:

- an explicit `sinks:` list (or `Run(sinks=...)`) is the ordered composition:
  each entry is the built-in `kind: collect` sink or a custom `factory:` sink;
- an absent selection keeps the historical default: one built-in
  `CollectSink`;
- the one-custom-sink `sink:` shorthand (or `Run(sink=...)`) selects that
  custom sink as the sole sink.

A custom sink therefore never silently gains or loses a `CollectSink`; declare
both entries in `sinks:` when both are wanted.  Adapter `extra_sinks` stay
prepended and the director/timeline/bridge tail remains appended.  `Director`,
`Bridge`, `MetricsWriter` and adapter `extra_sinks` are internal privileged
runtime hooks, not part of the public Sink surface or custom-Sink examples.

The removed `mode:` and top-level `benchmark:` config keys fail with teaching
`ConfigError`s: there is no compatibility alias and no built-in
benchmark/evaluation sink is selected by core.  Core has one physical deadline
authority, the core episode clock in `StepResult.reset_env_ids`; a custom sink
that needs a time or step budget ends episodes through its terminal requests.

`CollectSink` is built from a read-only facts snapshot (the resolved control
rows plus `{articulation: facts}`) and stores no live env.  During `on_step` it
derives policy controls from `bundle["view"]` and caches the resolved rows and
body-control facts from the first view that carries controls.  `on_step_end`
uses those cached/build rows (the view is already `None`), and `on_run_end`
writes the cached/build `{articulation: facts}` summary.  Multiple body
controls of one articulation keep the historical last-write-wins row.  A
zero-tick run has no view, so it still writes the build-time summary.

`stop_when` is optional and read-only.  `FanOutSink.stop_when` ORs its
children.  `build_instance` combines every selected sink's predicate (the
built-in `CollectSink` contributes none), so a selected custom sink owns the
early-stop decision.  The CLI passes `instance.stop_when` to `runner.run`; a
sink never stops a run itself.

### Stateful sinks and rewind

Rewind providers are discovered structurally, not by class name: a sink (or a
node reached through `rewind_provider`/`sinks`) is a provider when it has a
non-empty string `name` and callable `rewind_state` / `apply_rewind_state`;
`validate_rewind_state` is optional.  Provider names are checkpoint payload
keys, so duplicates fail at build time (and again in `RewindController`).
Custom providers should namespace their name (for example `acme.pick`).
`on_rewind(info)` only labels the abandoned branch; it does not restore state.

A stateful custom sink that must survive a checkpoint must expose that provider
shape or a `rewind_provider`.  Without one it is still valid, but its counters
and buffers are process truth and are not restored across a rewind; core does
not infer statefulness.

Termination evaluation is a live post-step/pre-reset boundary:

```text
env.step -> observation snapshot -> one vector EnvView (runner-owned)
         -> TerminationGroup.check_detail(ctx) once
         -> terminated / alive / term_metrics / terminations
         -> sink.on_step (same view in bundle["view"])
         -> due / reset mask -> sink.on_episode_end (same view)
         -> view.close(); bundle["view"] = None
         -> reset chain -> sink.on_step_end (view is None)
```

The bundle carries no `term_state` entry: `terminations` and `term_metrics`
are derived from the same one detail pass, and that same boundary view is the
one the sink observes during `on_step`/`on_episode_end`.  Termination is
live-only and never reads the env-level joint-limit compatibility views; joint
limits are owned by `ArticulationRuntime.joint_pos_limits` in record/policy
joint order.

The retained evaluation package is a second live boundary when an external sink/pack composes it:

```text
on_episode_end (before env.reset_envs)
         -> EpisodeBuffer.take(env_id)
         -> EpisodeEvaluator.evaluate(taken, bundle, env_id)
              -> assemble EpisodeData + core terminations
              -> one per-env EvalContext scoped to ep.env_id
              -> Task.score/episode incl. score_detail, declaration order
              -> ctx.close() in finally
              -> merge task terminations + validate the episode axis
         -> EvaluationSession forwards the EpisodeEvaluation
         -> publish episode_results for the sampler
```

`EpisodeEvaluator` (`unigym.runner.evaluation.scoring`) owns the episode facts,
the task invocation and the final termination merge.  `EvaluationSession`
(`unigym.runner.evaluation.session`) publishes `bundle["episode_results"]` and
delegates run-level accumulation and report writing to the injected
`EpisodeReporter` (for example `BenchmarkReporter` inside a composed evaluation session).

A benchmark task may read `ctx.env.record()`, `ctx.env.obs()` and the scoped
`ctx.env.query`; it cannot step, reset, write or read another env.  The bundle
has no live env field; `ctx.env` is the read-only task view.

Open-episode history is owned by `EpisodeBuffer`
(`unigym.runner.evaluation.buffer`): `EvaluationSession.on_step` appends the
normalized per-step record, `on_rewind` attaches a rewind mark to every open
buffer, and `on_episode_end` removes one env's steps and marks with `take`.
The rewind provider (`EvaluationSessionProvider`) snapshots/restores that buffer
(steps, marks, tick traces, episode stamp and the episode fuse) under the
`"benchmark"` provider name.  `FanOutSink.on_rewind` forwards to every child
recursively, so an evaluation session nested below director or adapter fan-outs
still receives the mark exactly once.  A tick-contract failure disables the task
for the run; that disabled set is process truth and a rewind never re-enables
it.

`build_instance` creates the run's one `ValueStore` **before** `_build_sinks`
and before `_build_bridge_and_wiring`.  `_build_sinks` hands
`values.publisher("sink")` to `SinkContext.values`, and the run build stores the
store as `RunInstance.values`; `sampler.bind_env(env)` and
`sampler.bind_values(values)` run at the live-instance construction seam,
before any runner exists.  `SyncRunner` never binds, so multiple runners over
one cached `RunInstance` cannot rebind.  The store is stored as
`RunInstance.values` and joins `rewind_providers` as the `"values"` provider
exactly once; a custom sink must not expose the store as a second provider.
Watchers are debug-only: they receive no value publisher and the watcher engine
joins as `"watchers"` for specs, trigger state and template windows only.  The
command-sampler lifecycle is documented in `command_sampler_contract.md`; the
transport is documented in `value_store_contract.md`; watcher lifecycle/values
are documented in `watchers_contract.md`.

Core-neutral rule: the runner and core sinks never use `Task` or Gymnasium
`terminated/truncated` vocabulary.  The task layer maps core outcomes later.
