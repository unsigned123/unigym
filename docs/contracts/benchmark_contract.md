# Benchmark contract

- Status: active
- Core base: `52b6bad`
- Schema: `benchmark_report` / `results`

Benchmark report is a deliberate break.  Core owns one episode contract;
pack/task code never re-derives timeline, deadlines or termination facts.

## Timeline

Canonical integer tick axis:

```text
reset sample:          tick=0, time=0
first post-step frame: tick=1, time=policy_dt
recorded frame i:      tick=i+1, time=(i+1)*policy_dt
```

`EpisodeData.step_index` is `int64 1..T`; `EpisodeData.time_s` is the
authoritative `step_index * policy_dt` mapping.  Recorders, benchmark sinks
and parity probes must use these arrays, never `arange * dt` starting at 0 or
`duration_s`.  `duration_s` is a compatibility summary only
(`time_s[-1]` when present).

`EpisodeData.reset_sample` is the pre-policy-step observation sample captured
by the reset chain (selected state keys, never camera frames).  Reset
legality is recorded in `reset_valid` / `reset_reason`; tasks must not infer
it from recorder frame 0.

## Deadline

`EpisodeLimits(timeout_s, timeout_steps)` has ONE integer authority.
`timeout_steps = floor(timeout_s / policy_dt + eps)`; declaring both requires
agreement, otherwise construction fails.  In `termination_mode: external`
the core `episode_length_s` is `None` and the task/benchmark
`timeout_steps` is the only deadline.  The core engine horizon (when
enabled) is recorded as `DEADLINE` / `runtime.episode_horizon`.

## Terminations

`unigym.benchmarks.termination` defines the core-neutral fact object:

- `EpisodeOutcome`: `success | failure | deadline | invalid_reset | engine | manual`
- `TerminationSource`: `environment | initializer | runtime | custom`
- codes are namespaced/stable (`env.force_limit`, `initializer.invalid_reset`,
  `runtime.episode_horizon`, `runtime.task_deadline`)
- `extensions` are JSON-safe, namespaced, versioned and ignored by core
  aggregation
- `metrics` are flat finite floats
- precedence (core-owned): `INVALID_RESET > SUCCESS > FAILURE > DEADLINE >
  ENGINE > MANUAL`; `priority` only breaks ties inside one outcome.

`EpisodeResult.success` stays task-judged and is never inferred from an
outcome.  Tasks may return `EpisodeResult.termination`; the sink validates
and includes it in the report.

## Reports

`benchmark_report.json`:

```json
{
  "schema": "benchmark_report",
  "meta": {"run": {"backend": "...", "device": "...", "num_envs": 8,
                   "seed": 0, "policy_dt": 0.02, "sim_dt": 0.005,
                   "reset_seed": 2, "world_id": "...",
                   "articulations": [{"name": "...", "profile": "..."}],
                   "observation_schema_version": 2, "model_revision": 0}},
  "terminal_outcome_counts": {},
  "terminal_code_counts": {},
  "terminal_source_counts": {},
  "episode_terminations": []
}
```

`results.json` (schema `results`) carries the same `run` identity and the
per-episode termination/extensions document.  Extension payloads are
preserved per episode; core never invents or merges schemas.
