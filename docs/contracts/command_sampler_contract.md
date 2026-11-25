# Command sampler contract

- Status: active
- Owner module: `unigym/commands/`

A `CommandSampler` owns the per-env command vector written into
`obs.command`; it is the only control channel visible to the adapter.  The
sampler may also own higher-level lifecycle state (multi-stage commands,
episode-to-episode feedback) itself.  Core does not define a `phase`,
`TaskSignal` or task-event concept.

## Surface

```python
class CommandSampler(ABC):
    rewind_inert: bool = False
    rewind_capable: bool = False

    def reset(self, env_ids=None) -> torch.Tensor: ...
    def update(self, step_dt: float, cmd: torch.Tensor) -> torch.Tensor: ...
    def bind_env(self, env: EnvBackend) -> None: ...
    def bind_values(self, values: ValueView) -> None: ...
    def on_episode_end(self, env_ids, results) -> None: ...
    def rewind_state(self) -> dict[str, JsonValue]: ...
    def apply_rewind_state(self, state: dict[str, JsonValue]) -> None: ...
```

`self.cmd` is part of the sampler state.

## Lifecycle

- `bind_env(env)` is called exactly once by `build_instance` (the one
  live-instance construction seam) after the env and sampler exist and before
  any runner exists.  `SyncRunner` never binds, so `Run.runner("sync")` /
  `run.gym()` reusing the cached `RunInstance` cannot rebind.  It is never
  serialized.  The env is a read-only view (`env.obs`, `env.num_envs`,
  `env.device`).
- `bind_values(values)` is called exactly once immediately after `bind_env`, at
  the same build seam and before any runner exists.  The argument is the run's
  live read-only `ValueView` (`docs/contracts/value_store_contract.md`).  The
  default is a no-op; a sampler that consumes values stores the view and
  interprets names and dtypes itself.  Core never maps a value to a command.
  It is never serialized or replayed on restore.
- `on_episode_end(env_ids, results)` is called at every due episode boundary,
  after `sink.on_episode_end` and before `sampler.reset`, so prior-episode
  results can choose the next episode's command.  It also runs when
  `auto_reset` is false; only the reset chain is skipped.
- `results` is a read-only `Mapping[int, tuple[EpisodeResultView, ...]]`: one
  tuple per due env in declaration order, containing the completed-episode
  results a sink published.  A sink that publishes none passes `{}`.
- Both hooks default to no-ops; built-in samplers keep their behavior.
- A raising `on_episode_end` is run-critical and propagates.

## Episode results versus sink terminals

Core itself publishes no episode results: it selects no scoring sink, so the
runner-provided mapping is `{}` unless a custom sink writes
`bundle["episode_results"]`.  A sink terminal
from `TerminalSink.drain_terminations()` is an episode-end fact, not an
`EpisodeResultView`.  A pack that needs sink-driven feedback must carry it in
its own shared pack state or explicitly publish the documented result mapping.

## Authoring

```yaml
command:
  name: fixed          # built-in registry
  dim: 4
  values: [0, 0, 0, 0]
```

```yaml
command:
  module: "my_pkg.commands:make_sampler"   # external factory
  dim: 4
  params:
    seats: 6
```

- exactly one of `name` or `module` is required;
- a `module` section accepts only `module` / `dim` / optional `params`, with
  `dim` a positive int and `params` a mapping;
- the reference is `pkg.mod:factory` and resolves in the run process;
- the factory signature is
  `make_sampler(*, dim, device, num_envs, **params) -> CommandSampler`;
- a target that is not callable, or a return value that is not a
  `CommandSampler`, fails at build with a teaching error.

## Values

A sampler may read the run's managed values through the bound live `ValueView`
(`values.get(name)` / `values.snapshot()`).  Core owns transport only: names,
dtypes, shapes and meaning belong to the user.  There is no automatic
value-to-command mapping, and a sampler that keeps state derived from values
must include that state in its own `rewind_state()`.

Selected observation values are ordinary managed values published by the build
under `obs/<key>` (owner `"observation"`); a sampler reads them through the same
view.  The selected keys come from `observations.values`; cameras and depth are
never published.

## Value timing

The runner order is `env.step(t) -> sampler.update(t) -> sink.on_step(t)`.  A
custom sink publishes after tick `t`'s physics and observation snapshot, so the
sampler's first polling read of that value is `update(t+1)`.  Watchers are
debug-only and publish no values.  The command returned by `update(t+1)` is
written to `obs.command` at `t+1` and is first used by `env.step(t+2)`:
in-episode polling closed loop has two policy-tick latency from the observed
view to the acting command.  This is the fixed sampler placement; a sink
cannot shorten the polling path from its callback.

A sampler may instead subscribe through the concrete `ValueStore` at
`bind_values` (the `ValueView` parameter does not expose `subscribe`).  A
subscriber callback fired by a sink publish during `on_step(t)` can mutate the
sampler's live command buffer **in place**, and `env.step(t+1)` then uses the
new command: the subscription path has one policy-tick latency.  This works
only when the callback updates the same buffer the runner currently holds —
the tensor returned by the current tick's `update` (normally `self.cmd`).
Reassigning a fresh tensor in the callback does not propagate to the runner, and
neither does an `update` that returns a detached copy instead of the owned
buffer.  Pin that alias in tests.

## Rewind

The rewind gate accepts a sampler only when it declares `rewind_inert = True`
(no external state, e.g. `FixedSampler`) or `rewind_capable = True`
(implements both rewind methods).  A sampler with neither is refused.

- a capable sampler's `rewind_state()` joins every snapshot payload and the
  safety capture under `"sampler"`;
- restore applies the sampler state after env/observations/adapter and before
  the providers; rollback applies the safety state the same way;
- a capable sampler whose snapshot carries no sampler state is refused during
  preflight (no partial restore);
- `rewind_state` must be JSON/numpy-safe and must not contain the env, task
  objects or live tensors;
- `bind_env` is not replayed on restore: a restored sampler must not depend on
  having been rebound.
- `bind_values` is not replayed on restore either; the values themselves ride
  the `ValueStore` `"values"` rewind provider.

Episode feedback the sampler relies on must be folded into its own state in
`on_episode_end`; the runner does not snapshot `episode_results` separately.
