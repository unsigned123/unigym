# Benchmark contract

- Status: active
- Core base: `52b6bad`
- Schema: `benchmark_report` / `results`

Benchmark report is a deliberate break.  Core owns one episode contract;
pack/task code never re-derives timeline, deadlines or termination facts.

## Runtime ownership

`EvaluationSession` (`unigym.runner.evaluation.session`) is the public runner
sink and episode-lifecycle owner.  It drives buffering, dispatches
`trigger: both` ticks, scores at episode end, publishes
`bundle["episode_results"]`, counts the early-stop fuse and forwards rewind
marks; it owns neither a scoring formula nor the report schema.

`EpisodeBuffer` (`unigym.runner.evaluation.buffer`) owns the open-episode step
slices, rewind marks and per-step tick samples and imports no benchmark type.
`EpisodeEvaluator` (`unigym.runner.evaluation.scoring`) assembles the
`EpisodeData`, creates the per-env context and invokes the tasks.
`EpisodeReporter` (protocol, `unigym.benchmarks.reporting`) owns run-level
observation; `BenchmarkReporter` is its benchmark implementation and owns
`benchmark_report.json` / `results.json`.  There is no `BenchmarkSink`.

## Timeline

Canonical integer tick axis:

```text
reset sample:          tick=0, time=0
first post-step frame: tick=1, time=policy_dt
recorded frame i:      tick=i+1, time=(i+1)*policy_dt
```

`EpisodeData.step_index` is `int64 1..T`; `EpisodeData.time_s` is the
authoritative `step_index * policy_dt` mapping.  Recorders, benchmark
evaluation and parity probes must use these arrays, never `arange * dt`
starting at 0 or `duration_s`.  `duration_s` is a compatibility summary only
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
outcome.  Tasks may return `EpisodeResult.termination`; the evaluator validates
it and the reporter includes it in the report.

## Task API

Benchmark tasks are live-first episode scorers:

```python
class Task:
    trigger: Literal["episode", "both"] = "episode"

    def on_tick(self, ctx: EvalContext) -> TickResult: ...
    def score(self, ep: EpisodeData, ctx: EvalContext) -> EpisodeResult: ...
    def score_detail(
        self, ep: EpisodeData, ctx: EvalContext, *, retain_frames: bool = False
    ) -> tuple[EpisodeResult, dict, dict]: ...
```

`trigger` is resolved once at construction from the authored `trigger:` param
(default: the class attribute) and validated at build time.  `episode` (the
default) calls only `score`.  `both` additionally calls `on_tick` once per
policy tick with one VECTOR-scoped context (`ctx.env.env_id is None`,
`ctx.t == tick`); it still calls `score` at episode end.  There is no
`tick`-only trigger.  A `both` task must override the default no-op `on_tick`
with exactly `on_tick(self, ctx)`; `FormulaTask` cannot satisfy `both` and is
rejected at build.

`on_tick` returns `TickResult(metrics: Mapping[str, torch.Tensor], fired:
torch.Tensor | None)`; every metric has a leading env axis `(num_envs, ...)`
and `fired` is bool `(num_envs,)`.  It is detection/annotation only: it cannot
step, reset, write or request termination.  The session converts each tick to
per-env CPU numpy `TickSample`s, so a per-tick contract failure (bad return
type, wrong shape/dtype) disables that task for the RUN (logged once; later
episodes score without it and the report shows `n_episodes: 0`).  The disable
set is process truth and is never reverted by a rewind.

A task may compose a canonical condition inside `on_tick`; there is no core
`Task.condition` field:

```python
from unigym.benchmarks.base import Task, TickResult
from unigym.conditions.height import HeightCondition


class LowBaseTask(Task):
    trigger = "both"

    def on_tick(self, ctx):
        mask = HeightCondition(articulation="g1", threshold=0.2).evaluate(ctx)
        return TickResult(fired=mask)
```

`TickResult` remains the benchmark-owned detection payload; the condition only
computes the mask and the task/session own trace and scoring effects.

Per-task histories are assembled into `EpisodeData.ticks: Mapping[str,
TickTrace] | None` (`None` when the run has no `both` task).  A trace's metric
arrays have a leading time axis `(T,)`/`(T, ...)` and `fired` is `(T,)` bool.
The key set must be stable across every step of one episode; a mismatch fails
that task for the episode (omitted from scores/views) without affecting other
tasks.  Ticks are inputs to `score`, never report fields; the report/results
schema is unchanged.

`EpisodeData` remains the episode history/timeline/metrics payload.  `ctx` is
the read-only, per-env `unigym.evaluation` context created by
`EpisodeEvaluator` (`unigym.runner.evaluation.scoring`) at episode end, before
reset, scoped to `ep.env_id`.  Tasks may read
`ctx.env.record()`, `ctx.env.obs()` and, when the backend exposes a runtime
query, `ctx.env.query`; they cannot mutate, step or reset the env or read
another env.  Values from `record()`/`obs()` are borrowed for the callback.

The evaluator builds one context per due env and closes it in `finally` after
every task for that env.  A task exception is isolated and skipped (the task is
omitted from the report and result views).  Signature errors are different: an
old `score(ep)` or a legacy `score_detail(..., retain_frames)` fails at build
time with a teaching error, so a broken task can never be silently omitted.

`unigym.benchmarks.reporting` defines the evaluator/reporter handoff types:
`TaskScore` (one successful task result plus declaration detail),
`EpisodeEvaluation` (the assembled `EpisodeData`, successful scores, views and
final merged termination) and the run-level `EpisodeReporter` protocol.  The
evaluator returns those values; report aggregation and file writing stay
outside it.

Built-in tasks (`stand`, `velocity_tracking`) accept `ctx` and ignore it; their
metrics are computed only from `EpisodeData`.  `FormulaTask` overrides both
faces and attaches predicate leaves/frames.  There is no offline re-scoring
guarantee: a report is a live run observation and cannot be reconstructed from
stored episode data alone.

## Result views

`unigym.core.episode_result.EpisodeResultView` is the read-only handoff used by
the command-sampler hook (command_sampler_contract.md):

```python
@dataclass(frozen=True)
class EpisodeResultView:
    task: str
    success: bool | None
    metrics: Mapping[str, float]
```

`EpisodeEvaluator` builds the views after each task returns successfully and
the `EvaluationSession` publishes them on the episode-end bundle as
`bundle["episode_results"] = {env_id: (view, ...)}`: one tuple per due env in
task declaration order.  A task that raises is omitted exactly as it is omitted
from the report.  The view copies its metrics into a read-only mapping and
carries no `EpisodeData`, numpy array, task object or `EvalContext`, so the
command layer never depends on the benchmark or evaluation layer.

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

## Migration

`BenchmarkSink` is removed in this release; there is no compatibility alias.
Use `EvaluationSession` (`unigym.runner.evaluation.session`) with the injected
`EpisodeBuffer`, `EpisodeEvaluator` and `BenchmarkReporter`.  The schema and
artifact constants (`REPORT_SCHEMA`, `RESULTS_SCHEMA`, `METRICS_NPZ`,
`RESULTS_JSON`) live in `unigym.benchmarks.reporting`.

The `Task` authoring API remains: `score(self, ep, ctx)`,
`score_detail(self, ep, ctx, *, retain_frames=False)` and `aggregate` are
unchanged.  `trigger="both"` and `on_tick(self, ctx)` are additive opt-in
surface; default `trigger="episode"` behavior is unchanged.  Tick results are
scoring inputs, not report fields, so the report/results schema does not change.
