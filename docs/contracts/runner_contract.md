# Runner contract (benchmark facts)

The `SyncRunner` callback stream is the single source of episode facts.  The
`bundle` handed to sinks carries:

```text
t             policy tick counter
record        per-env backend record (canonical keys + contact facts)
terminated    (N,) bool, engine termination group
reset_env_ids (N,) due ids, backend episode clock
terminations  {env_id: EpisodeTermination}   # term group + core horizon
step_index    canonical tick (t+1)
time_s        canonical time ((t+1) * policy_dt)
run_identity  RunIdentity (benchmark runs)
reset_facts   {env_id: {reset_index, reset_sample, reset_valid, reset_reason}}
              present on on_episode_end for the ending episode
episode_results {env_id: (EpisodeResultView, ...)}
              present on on_episode_end when a BenchmarkSink scored the
              episode; per-env, task declaration order, read-only
```

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

`build_instance` calls `sampler.bind_env(env)` once at the live-instance
construction seam, before any runner exists.  `SyncRunner` never binds, so
multiple runners over one cached `RunInstance` cannot rebind.  The
command-sampler lifecycle is documented in `command_sampler_contract.md`.

Core-neutral rule: the runner and core sinks never use `Task` or Gymnasium
`terminated/truncated` vocabulary.  The task layer maps core outcomes later.
