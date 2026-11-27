# tests/_support/fakes

Test doubles for external boundaries that cannot be exercised with real core
objects. This directory is not collected and contains no tests.

## Manifest

| Module | Protocol / boundary | Reset semantics |
| --- | --- | --- |
| `backend.py` | `unigym.env.backends.base.EnvBackend` — the simulator step/reset boundary | resets only the selected rows to spawn state (`q0`, zero `prev_action`, zero `time_in_episode`) and increments only their `episode_id`; `reset_all()` selects every row; `due_rows` reports a configured backend-owned due mask without resetting during `step` |
| `closing.py` | close-only manager/prop boundary for lifecycle fault injection | stateful counters; `RecordingClose` counts successful close calls, `FailingClose` raises on every close call |
| `command_sampler.py` | `unigym.commands.base.CommandSampler` — the custom sampler factory seam | `RecordingSampler` stores its factory parameters, records `bind_env` / `bind_values` / `on_episode_end` calls verbatim and declares no rewind capability; `ClosedLoopValueSampler` binds a `ValueView`, records the value observed by each `update` and the command that tick used, and turns the observed value into the next command (no rewind capability) |
| `runner.py` | the single-tick runner boundary (`step`/`tick`/`reset`/`close`/`ticks_done`) | `FakeRunner` owns one `FakeBackend`; `reset` selects the rows and never advances ticks, `reset_t` overrides only the result tick, `last_actions` records the lowered action mapping of the last tick, `terminated_rows` / `due_rows` set the step result's `(N,)` termination mask and backend-owned due ids (reported only, never reset during `step`), `close` is idempotent, `fail_next_tick` raises once after advancing the tick |
| `external_sink.py` | public `unigym.sink.Sink` plus the `SinkSpec` factory/context boundary (four callbacks, optional `stop_when` / `on_rewind` / terminal `drain_terminations`, rewind-provider shape, bridge metrics feed); `ExternalSink` / `make_external_sink` are the stateful real-run double and its importable factory; `ViewProbeSink` records the per-tick bundle and can inject a callback/`RewindBoundary` failure to prove boundary-view lifetime; `TerminalDrainSink` returns a fixed drain payload, `ScriptedTerminalSink` drains one queued payload per tick and records the episode-end stream, `TerminalRequestSink` / `make_terminal_request_sink` request one real-run terminal on a configured tick and record per-tick episode facts, `ValuePublishingSink` / `make_value_publishing_sink` publish one level value per tick through `SinkContext.values` (CPU-only) and record the publication log, and `NonCallableDrainSink` declares a non-callable terminal capability | construct one fresh instance per test; `RecordingSink.calls`, `StopPollingSink.polls`, `ProviderSink.supplier_calls`, `TerminalDrainSink.drain_calls`, `ScriptedTerminalSink` queues/counters/bundles, `TerminalRequestSink` counters/lists, `ValuePublishingSink` publication log, `ViewProbeSink` bundles and `ExternalSink` counters are the only state; no reset method and no shared state |
| `isaaclab.py` | `isaaclab` / `isaaclab.sensors` module boundary used by the IsaacLab contact-ref tests; the `fake_isaaclab` fixture installs a `ContactSensorCfg`-only module tree and returns the sensors module | `monkeypatch` restores `sys.modules` after the test; the module tree exposes only `sensors.ContactSensorCfg`; the `_Cfg` helper classes record constructor kwargs only; no sensors, articulations, assets, physics or backend behaviour is faked |
| `step_clock.py` | the bridge/metrics step-clock capability (`step_dt` plus optional `decimation`) | stateless; `StepClockEnv` carries only the configured period/decimation, never physics or observation state |

Every Python module in this directory (except the package marker) must appear
in the manifest table; `tests/meta/test_stub_policy.py` enforces this.

## `backend.py` — `FakeBackend`

Purpose: prove the backend-neutral `EnvBackend` contract and the conformance
layer without MuJoCo, IsaacLab, MJWarp, assets, network, subprocesses or a GPU.

Boundary: one articulation `arm` with a configurable identity joint list
(default one joint `shoulder`); one body control group `body` driving that
articulation (`position_pd` / `position`). The action-group name is the
articulation name `arm`, because the contract body-group rule is
`name == articulation`.

Behavior:

- `step(actions, raw_action=None, cmd=None)` consumes `dict[str, NamedAction]`,
  requires the named action group, validates kind/batch/joint coverage and
  replaces the scalar joint position with the target. `raw_action` is copied to
  the typed `prev_action` slot; `cmd`, when supplied, is copied to the
  `command` slot.
- `env.obs` is a config-declared `ObservationRegistry` and publishes the full
  robot state plus the two runtime slots.
- `record()` carries the canonical `articulation/arm/<key>` keys and the
  bookkeeping keys `time_in_episode` / `episode_id`; `step()` returns an
  `EnvStep` with the typed snapshot and reset masks.
- `perm_il2pol` / `perm_pol2il` are the identity permutation; policy joint
  limits are `(N, J)` float32 tensors.
- `close()` is idempotent.

Reset semantics:

- `reset_envs(ids)` preserves the given id order in its return value, resets
  only the selected rows, leaves every other row untouched, is a no-op for an
  empty id vector and fails fast for ids outside `[0, num_envs)`.
- `reset_all()` is `reset_envs(arange(num_envs))`.

Limitations (deliberate, no production seam is added for them):

- no physics, integration, contacts, feet or termination logic;
- no automatic reset (`reset_env_ids` is empty unless `due_rows` is configured;
  `step` never resets a due row itself);
- no cameras, depth, props, events, recorder, contacts, snapshots or actuator
  seam; no optional capability is advertised or silently faked;
- one articulation, CPU only, fixed identity joint order (the joint list is
  configurable, the order is declaration order);
- episode time advances by one second per step and carries no sim-time meaning.

One fake per external boundary: do not add per-test fakes, a fake hierarchy or
a configurable universal fake. `backend.py` is the conformance reference;
`closing.py` is the close-boundary fault injector; `runner.py` is the
single-tick runner seam. Add a new
manifest row only when a new external boundary appears.
