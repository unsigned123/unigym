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
- `on_episode_end(env_ids, results)` is called at every due episode boundary,
  after `sink.on_episode_end` and before `sampler.reset`, so prior-episode
  results can choose the next episode's command.  It also runs when
  `auto_reset` is false; only the reset chain is skipped.
- `results` is a read-only `Mapping[int, tuple[EpisodeResultView, ...]]`: one
  tuple per due env in task declaration order, containing only completed
  benchmark tasks.  A collect-only run passes `{}`.  There is no step-level
  benchmark trigger.
- Both hooks default to no-ops; built-in samplers keep their behavior.
- A raising `on_episode_end` is run-critical and propagates.

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

Benchmark feedback the sampler relies on must be folded into its own state in
`on_episode_end`; the runner does not snapshot `episode_results` separately.
