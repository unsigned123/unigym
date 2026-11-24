# Documentation

The full documentation is being rewritten.
English and Chinese versions will live under `docs/en/` and `docs/zh/`.
See the root README for install and usage.

Process documents:

- [Engineering workflow](development.md) - workflow gates, roles, review and
  reporting.
- [Testing policy](testing.md) - test layers, markers, operations and red-test
  protocol.
- [Scene asset contract](contracts/scene_asset_contract.md) - file-backed
  `AssetInstance` fields, backend matrix, resolution, digest and refusals.
- [Model-edit contract](contracts/model_edit_contract.md) - global
  `run.model_edit()` body/geom transaction, field matrix and backend limits.
- [Evaluation context contract](contracts/evaluation_contract.md) - the
  read-only, boundary-scoped `EvalContext`/`EnvView`, scoped `EvalQuery`, and
  the live termination/benchmark boundaries.
- [Benchmark contract](contracts/benchmark_contract.md) - canonical
  episode timeline, deadline authority, the `EvaluationSession` /
  `EpisodeBuffer` / `EpisodeEvaluator` / `BenchmarkReporter` ownership split,
  live task signatures (including opt-in `trigger: both` ticks) and report
  schema.
- [Runner contract](contracts/runner_contract.md) - the sink callback bundle
  (timeline/reset/termination facts) and the live evaluation boundaries,
  including rewind fan-out and the `benchmark` rewind provider.
- [Command sampler contract](contracts/command_sampler_contract.md) - sampler
  lifecycle hooks, custom factory authoring and rewind participation.
- [Value store contract](contracts/value_store_contract.md) - the run-scoped
  `ValueStore`, `ValueView`/`ValuePublisher`/`StepValues` boundary, tensor and
  JSON transport, writer ownership, checkpoint payload and observation
  publication.
- [Report templates](templates/) - design brief, completion, review and blocked
  reports.

Usage documents:

- [Observations](observations.md) - config-declared typed observation registry,
  snapshots, records and artifact keys.
- [Actions](actions.md) - joint-addressed runtime values, authoring templates,
  per-joint routing/coverage and policy ownership rules.
- [World and `World.flat()`](world.md) - World/EnvSpec ownership, flat-ground
  resolution per backend, `<option>` inheritance and residual limits.
- [State-write contract](state_write_contract.md) - optional world-frame,
  wxyz, name-addressed root/joint writes and settle ordering.
- [NativeHandle escape hatch](native_handle.md) - external-pack-only,
  unstable `run.native`/`env.native` identity handles; core must not use it.
- [Simulation facade](sim_facade.md) - experimental `unigym.sim` usage, scope,
  error taxonomy and residual limits.
