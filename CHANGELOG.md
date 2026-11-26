# Changelog

## Unreleased

Breaking release: the pack model v2 cutover removes the transitional pack
modules and names. There is no compatibility shim.

- Removed the `unigym.packs.catalog` and `unigym.packs.runtime` modules and the
  `PackCatalog`, `CatalogEntry`, `PackInfo`, `ResourceInfo`, `ActivationHandle`,
  `ProviderInfo`, `AssetContext`, `PackManifestV1`, `ProvidedItem` and
  `PackRuntime` names. Packs are `Pack` values, stored items are `PackItem`
  values, live providers are `ProviderRef` projections, and the searchable,
  resource-resolving, activating owner is `PackLoader`.
- `PackSet` now owns the duplicate pack-name and duplicate `(face, name)` rule
  that lived in the deleted catalog module. `PackSet` stays a pure index; the
  loader owns `from_pack_set`, resources and activation.
- Removed `Pack.provides`; read `Pack.items` or call `Pack.items_for(...)`.
  `PackItem.resource`/`.entrypoint`/`.id` remain read-only projections over the
  stored locator and name.
- `unigym.pack_api` and `unigym.packs` export none of the removed names.

## 0.109.0

Additive release: the run-scoped `ValueStore` gains a synchronous
value-change subscription face.  There is no other semantic change and no
benchmark or evaluation behavior is restored.

- Added the additive run-scoped `ValueStore.subscribe(callback, *, names=None)`
  surface: synchronous value-change callbacks after a successful publish or a
  value-removing clear, registration order, isolated copies and an idempotent
  unsubscribe.  A `CommandSampler` bound to the concrete store can react at the
  publication boundary with one-tick latency; polling `ValueView` reads remain
  two-tick.  `StepValues` and `ValuePublisher` stay non-subscribable; there is
  no event bus, asynchronous delivery or history.
- Subscription filters are exact declared names (`names=None` means every
  name): there is no wildcard/pattern grammar, no event history and no rewind
  notification.  A callback must not mutate the store; a re-entrant publish,
  clear or rewind from a callback is a contract error.
- `ValueStore.subscribe` is the only new public surface.  Subsystem schemas and
  the 0.108 benchmark-removal contract are unchanged.

## 0.108.0

Breaking release: core no longer ships a built-in benchmark or evaluation loop.
The external `unigym-bench` package is the replacement.

- Removed the `unigym.benchmarks` package, the `unigym.runner.evaluation`
  package (`EvaluationSession`, `EpisodeBuffer`, `EpisodeEvaluator`), the
  benchmark task providers and their catalog, the built-in benchmark reporter,
  and the `benchmarks` CLI.  There is no compatibility shim.
- Removed the run `mode` (`collect` / `benchmark` / `both`).  A run now always
  uses the explicit sink selection (`sinks:`, the one-custom-sink `sink:`
  shorthand, or the built-in `CollectSink` default) and the run never selects a
  benchmark or evaluation sink.
- Removed `episode_length_s` and the episode clock/horizon they fed: the env
  key, the `Run(...)` argument, the `EnvSpec` field, the duration range and the
  benchmark `timeout_s` / `timeout_steps` deadline.  Core derives no due mask of
  its own; an episode ends through a backend-owned due mask, a sink terminal or
  an explicit `runner.terminate(...)` boundary.
- Removed the top-level `benchmark:` config section and the core
  `benchmark_report.json` / `results.json` artifacts.  A custom sink owns its
  scoring and report files.
- `meta.json` and `summary.json` no longer emit `episode_length_s` or
  `episode_length_range`.  The data schema version is unchanged because the
  remaining fields keep their meaning; consumers that read the removed keys
  must stop doing so.

## 0.107.0

Breaking release: the run-scoped channel is now the managed value store.

- Removed the `unigym.signals` module and its `SignalBus`, `SignalView` and
  `SignalPublisher` API.  There is no compatibility shim.
- A run owns one `unigym.values.ValueStore`; `SinkContext.values` is a
  write-only `ValuePublisher`, `RunInstance.values` exposes the store, and
  `CommandSampler.bind_values` receives the live `ValueView`.
- Managed values transport per-env dense tensors or JSON-safe trees, with
  first-writer-wins ownership and isolated copies on read.  The store is the
  `"values"` rewind provider with the schema-2 `{schema, values, writers}`
  payload; the old flat signal-style payload is rejected.
- `bundle["step"]` carries the pre-`on_step` `EnvStepContext` value snapshot and
  `bundle["final_step"]` the post-`on_step` snapshot on due ticks; both are
  `None` at `on_step_end`, and a rewind never builds a final context.
- Explicitly selected observations are published under `obs/<key>` through the
  store; cameras and depth are never published.
- Replaced `docs/contracts/signal_bus_contract.md` with
  `docs/contracts/value_store_contract.md` and updated the runner, command
  sampler, evaluation, watchers and pack API contracts.
