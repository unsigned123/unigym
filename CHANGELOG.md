# Changelog

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
