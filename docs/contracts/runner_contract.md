# Runner contract (benchmark facts)

The `SyncRunner` callback stream is the single source of episode facts.  The
`bundle` handed to sinks carries:

```text
t             policy tick counter
record        per-env backend record (canonical keys + contact facts)
terminated    (N,) bool, live termination group
alive         (N,) bool, not terminated and finite policy observation
term_metrics  per-group/total termination firing fractions
reset_env_ids (N,) due ids, backend episode clock
terminations  {env_id: EpisodeTermination}   # term group + core horizon
step_index    canonical tick (t+1)
time_s        canonical time ((t+1) * policy_dt)
run_identity  RunIdentity (benchmark runs)
reset_facts   {env_id: {reset_index, reset_sample, reset_valid, reset_reason}}
              present on on_episode_end for the ending episode
episode_results {env_id: (EpisodeResultView, ...)}
              present on on_episode_end when an EvaluationSession scored the
              episode; per-env, task declaration order, read-only
env           live EnvBackend, sink wiring only (watchers/bridge)
```

`bundle["env"]` is sink wiring for the watcher/bridge transport, not a user
evaluation surface.  Termination and benchmark evaluation receive the read-only
`unigym.evaluation` context; a sink or task must not read `bundle["env"]` to
walk the backend.

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
  early stop decision (not yet polled by core); stopping the run remains the
  runner's job.

These protocols describe optional capabilities only; publishing them adds no
new runner control flow.  `FanOutSink` composes sinks in construction order,
forwards every callback to each child and structurally satisfies `Sink`.

Lifecycle:

1. `on_step(t, bundle)` — pre-reset data side.  Timeline columns are
   `step_index = t+1`, `time_s = (t+1) * policy_dt`.
2. `on_episode_end(due, bundle)` — BEFORE `env.reset_envs`.  `reset_facts`
   describe the episode's START (captured at its reset, tick 0); the
   termination facts are the final tick's.
3. `sampler.on_episode_end(due, bundle.get("episode_results") or {})` — AFTER
   the sinks (they publish the views) and BEFORE `sampler.reset`.  Runs even
   when `auto_reset` is false.
4. reset chain (`env.reset_envs` / `adapter.on_reset` / `sampler.reset`).
5. The runner captures the NEXT episode's pre-policy-step sample from
   `env.obs.snapshot()` and the backend's `ResetOutcome`.

Termination evaluation is a live post-step/pre-reset boundary:

```text
env.step -> observation snapshot -> one vector EvalContext
         -> TerminationGroup.check_detail(ctx) once
         -> terminated / alive / term_metrics / terminations
         -> ctx.close()
         -> sink.on_step
```

The bundle carries no `term_state` entry: `terminations` and `term_metrics`
are derived from the same one detail pass, and the context is closed before any
sink observes the bundle.  Termination is live-only and never reads the
env-level joint-limit compatibility views; joint limits are owned by
`ArticulationRuntime.joint_pos_limits` in record/policy joint order.

Benchmark evaluation is the second live boundary:

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
`EpisodeReporter` (`BenchmarkReporter` in benchmark runs).

A benchmark task may read `ctx.env.record()`, `ctx.env.obs()` and the scoped
`ctx.env.query`; it cannot step, reset, write or read another env.  The core
`env` bundle field is not the task context.

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

`build_instance` creates the run's one `SignalBus` **before**
`_build_bridge_and_wiring`, hands it to the watcher attach seam, calls
`sampler.bind_env(env)` and then `sampler.bind_signals(signals)` at the
live-instance construction seam, before any runner exists.  `SyncRunner` never
binds, so multiple runners over one cached `RunInstance` cannot rebind.  The bus
is stored as `RunInstance.signals` and joins `rewind_providers` as the
`"signals"` provider; the watcher engine joins as `"watchers"` (specs, trigger
state and template windows) so a checkpoint restores producer and transport
together.  The command-sampler lifecycle is documented in
`command_sampler_contract.md`; the transport is documented in
`signal_bus_contract.md`; watcher signal semantics are documented in
`watchers_contract.md`.

Core-neutral rule: the runner and core sinks never use `Task` or Gymnasium
`terminated/truncated` vocabulary.  The task layer maps core outcomes later.
