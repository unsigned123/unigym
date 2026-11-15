# Simulation facade (experimental)

The `unigym.sim` facade is a thin, object-first surface over the existing
`Run` / `ExternalRunner` / backend / IR stack. It is additive: the low-level
API is unchanged and remains the supported path for policy adapters, Gym-style
training and backend-specific work.

```text
SimulationBuilder -> Simulation -> Run / ExternalRunner / backend
```

## Experimental status

- `unigym.sim.SIM_API_VERSION` is `1`. There is no stable-API promise; an
  incompatible facade change bumps the version.
- The first spike is proven for **one env, one articulation** on `mujoco` and
  `isaaclab2`; its acceptance is `N == 1`. Multi-env (`N > 1`) is a backend
  capability — `mujoco` native per-env `MjData`, `mjwarp` one batched warp
  `Data` plus `N` CPU mirrors, `isaaclab2` native scene cloning — but the
  facade's `N > 1` contract is a separate, not-yet-landed plan (see
  [Scope and non-goals](#scope-and-non-goals)). Camera frames, contact parity
  and object/prop handles are deferred.
- `import unigym` stays torch/MuJoCo/IsaacLab-free. `unigym.sim` imports
  `torch` lazily at the action/observation boundary, so importing the facade
  is also torch-free.

## Usage: the four-line loop

```text
build once -> observe -> act (stage) -> step -> observe
```

`act` never advances time. `step` advances exactly one policy tick and returns
the observation produced by that tick. `observe` never ticks, but it is not
silently stale-tolerant: it raises `StaleStateError` if the runner advanced out
of band instead of returning the stale cache.

```python
import torch
import unigym
from unigym.packs import require
from unigym.world import World

packs = require("acme_robots")                 # one explicit PackSet
control = unigym.Control.position_pd(
    joint_set="body",
    kp=20.0,
    kd=1.0,
)

builder = unigym.SimulationBuilder(
    backend="mujoco",
    dt=0.02,               # policy/control period
    physics_dt=0.005,      # engine step; dt == physics_dt * substeps
    pack_set=packs,
)
builder.world = World.base(mjcf="robot_scene.xml")  # the world owns the asset
robot = builder.add_articulation(
    packs.pack("acme_robots").robot("acme.arm"),
    control=control,
)
binding = robot.bind_action("body", kind="position")

with builder.build() as sim:
    obs = sim.observe()                            # spawn state, t == 0
    print(obs.articulations[robot].joint_pos)

    for _ in range(3):
        sim.act(binding, torch.zeros((1, binding.dim), dtype=torch.float32))
        obs = sim.step()                           # one tick; first step keeps t == 0
```

`t` is 0-based and names the tick that produced the observation: spawn is
`t == 0`, the first `step()` also returns `t == 0`, and after `N` steps
`obs.t == N - 1`.

The named single-env form is equivalent when the handle declares exactly one
face:

```python
sim.act(robot, {"shoulder": 0.1, "elbow": -0.2})
obs = sim.step()
```

`step(actions={binding: values, ...})` is sugar: it stages every entry
atomically and then runs the normal `step()`. A failed entry clears the pending
table and leaves the tick/state untouched. The canonical path is `act` +
`step()`.

## Builder

```python
SimulationBuilder(
    *,
    backend: str,
    dt: float,
    physics_dt: float | None = None,
    substeps: int | None = None,
    seed: int = 0,
    num_envs: int = 1,
    pack_set: PackSet,
    name: str = "simulation",
    steps: int = 1000,
    episode_length_s: float = 20.0,
    artifacts: bool = False,
    device: str = "cpu",
)
```

- `backend` must be a registered backend name. The lookup never builds a
  backend.
- Exactly one of `physics_dt` / `substeps` is required and
  `dt == physics_dt * substeps` must hold with exact float arithmetic
  (`abs_tol=1e-12`). There is no silent re-quantization.
- Visible defaults are shown above; there are no hidden world, control or
  action defaults.
- `builder.world` is required and must declare a non-empty world asset. The
  world fragment may not declare `articulations` / `controls`; the builder is
  the single owner of those tables. `builder.terrain` is sugar for
  `builder.world.terrain`.
- `add_articulation(ref, *, name=None, control=...)` requires an explicit
  `Control` spec; there is no implicit control. Faces are resolved against the
  pack profile, face names are global, and faces of one articulation must be
  disjoint.
- `add_camera(entry, *, name=None)` accepts an existing `unigym.Camera.*`
  `cameras[]` entry mapping. `attach=BodyHandle` is deferred and raises
  `DeferredFeatureError`; use a native mount name through
  `Camera.attached(name, mount=...)`.
- After the first successful `build()`, declarations raise
  `BuilderSealedError`; a repeated `build()` returns the same `Simulation`.
- `builder.articulations` / `builder.cameras` are the declared handles. In
  config mode they are empty until `build()`.

## Pack acquisition

The facade never discovers packs. Every robot reference must belong to the
builder's explicit `PackSet`:

```python
import unigym.packs

packs = unigym.packs.installed()          # configured manifest-only scope
packs = unigym.packs.require("acme")      # filtered to exactly "acme"
packs = unigym.packs.discover_set(["/path/to/pack/root"])
```

`packs.pack(name).robot(id)` mints a `RobotRef` without importing pack Python.
`add_articulation` validates pack membership and the manifest version, then
loads the neutral robot profile through one explicit
`pack_set.to_runtime().activate(pack)` call; that activation registers the
robot process-wide (the existing pack boundary). Two packs that declare the
same robot id conflict at activation.

`builder.pack_runtime` is the activation-capable runtime over the same set. A
world preset is declared explicitly:

```python
builder.world = World.from_preset(preset, runtime=builder.pack_runtime)
```

## Actions

An `ActionBinding` is one face lowered to the existing `NamedAction` surface:
`binding.joint_names` is the column order and `binding.kind` is the action kind
(`"position"` for `Control.position_pd`).

- `binding = handle.bind_action(face, kind="position")`: columns default to the
  face's resolved joint order. An explicit `order=` must be a permutation of
  that face.
- `act(binding, values)` requires a floating `torch.Tensor` of shape
  `(num_envs, binding.dim)`, or `(binding.dim,)` when `num_envs == 1`. Values
  are normalized to `(N, dim)` and cloned.
- `act(handle, {joint: value})` is the single-env named form. Its keys must
  cover exactly the face joints; key order becomes the action column order.
  With more than one declared face, `face=` selects one.
- At most one pending entry per face. `step()` requires **every** declared
  control face to be staged; an unstaged face is `MissingActionError`. The
  previous action is never reused.
- All validation happens before the tick. An engine exception clears pending,
  marks the simulation `faulted`, and propagates; later live calls raise
  `SimulationFaultedError` until `close()`.

## Observations

One `Observation` belongs to the producing tick in `obs.t`: spawn is `t == 0`,
the first `step()` also returns `t == 0`, and after `N` steps `t == N - 1`.

| Field | Meaning |
| --- | --- |
| `t` | The 0-based policy tick that produced the cached state (`N - 1` after `N` steps). |
| `articulations` | `Mapping[ArticulationHandle, ArticulationObservation]`. |
| `cameras` | `Mapping[CameraHandle, CameraFrame]` for published frames only. |
| `contacts` | `contact/...` tensors present in the live state (no backend parity claim). |
| `reset_env_ids` | Env rows reset by the producing tick. |
| `terminated` | The producing tick's termination mask; `None` at spawn/reset. |
| `raw` | The live state mapping (`extra` without `reset_env_ids` / `should_reset`). |
| `obs_cache` | Exact copy of `StepResult.obs`; values are kept verbatim. |

`ArticulationObservation` exposes absolute `joint_pos`, derived
`joint_pos_rel = joint_pos - q0`, `joint_vel`, `applied_torque`, `gravity_b`,
`base_pos_w`, `base_quat_w` (`w-x-y-z`), `base_lin_vel_b`, `base_ang_vel_b`,
and the optional cached `prev_action` / `command`. Every tensor keeps the
leading env axis `(N, ...)`. `joint_names` is `profile.joint_order`, the order
of every `(N, J)` joint column.

Projection rules:

- A required `articulation/<name>/...` state key that is missing, has the wrong
  type or has the wrong shape is an `ObservationError`; there is no zero-fill.
- `q0` comes from `profile.default_joint_pos` in `joint_names` order. A joint
  without a default is an `ObservationError`.
- `prev_action` / `command` are read from `obs_cache`; absent means `None`.
  A present but non-tensor value is an `ObservationError`.
- `raw` drops the state bookkeeping keys (`reset_env_ids` / `should_reset`) and
  keeps every remaining live-state value verbatim; a value that is neither a
  `torch.Tensor` nor a `bool` is an `ObservationError`.
- Camera frames are read from `cam/<name>/rgb` / `cam/<name>/depth` in
  `obs_cache`; the facade never renders on demand. With the manual adapter the
  cache is normally empty, so no cameras appear.
- `obs_cache` keeps non-tensor record entries (for example `contact_ops/...`)
  verbatim; only the keys the facade projects must be tensors.

## One-clock rule

A `Simulation` owns no engine object of its own. It borrows the `Run` and its
`ExternalRunner`; `sim.runner` is the neutral `Runner` seam over that same
clock.

- `act` stages only; it never ticks.
- `observe` reads the cached state produced by the current tick; it never
  resynchronizes, and raises `StaleStateError` on clock drift.
- `step` ticks the one runner exactly once and caches the returned
  `StepResult`.
- `reset` resets every env through the same runner, clears pending and
  refreshes the cached state.
- Out-of-band ticks are detected via `runner.ticks_done` drift and raise
  `StaleStateError`; the facade never silently resyncs or applies a stale
  target.
- `sim.run.step(actions=...)` reaches the same clock and is supported but
  discouraged; `sim.run.step()` without actions fails in the manual adapter by
  design. The facade never starts a second runner.

## Error table

All facade errors derive from `unigym.errors.UniGymError`. `NoWorldError` and
`NoControlError` keep the historical `ConfigError` base; the rest share
`SimulationError`.

| Error | Raised when |
| --- | --- |
| `NoWorldError` | `build()` has no world, or the world declares no asset. |
| `NoControlError` | `add_articulation` is called without `control=`. |
| `SimulationConfigError` | Invalid declaration: dt arithmetic, labels, backend name, face/profile coverage, config-mode requirements. |
| `ForeignHandleError` | A handle, binding or pack ref belongs to another builder/simulation or pack set. |
| `BindingError` | Unknown face, illegal kind, bad column order, bad shape/dtype, ambiguous or partial named action. |
| `PendingActionError` | Duplicate face, empty pending on `step()`, or a staged face that is not declared. |
| `MissingActionError` | A required control face is not staged. |
| `StaleStateError` | The runner advanced out of band; the facade clock is behind. |
| `ObservationError` | A required state key is missing or invalid, or `q0` is unavailable. |
| `SimulationClosedError` | A live call after `close()`. |
| `SimulationFaultedError` | A live call after an engine exception; only `close()` is safe. |
| `BuilderSealedError` | A declaration after the first successful `build()`. |
| `ConfigModeError` | `add_articulation` / `add_camera` / `world=` on a `from_config` builder. |
| `DeferredFeatureError` | A declared out-of-scope feature, currently `attach=BodyHandle` cameras. |

## Config round-trip

`Simulation.to_config(path)` delegates to `Run.to_config(path)` and writes the
existing authoring surface (the world, articulation, control and camera
tables). The document also reloads through `Run.from_config`.

```python
sim.to_config("sim.yaml")

reloaded = SimulationBuilder.from_config("sim.yaml", pack_set=packs)
sim2 = reloaded.build()
```

The round-trip is **semantic**, not identity-preserving: same labels,
articulation/control routing, camera table, world asset, `dt` / `physics_dt` /
`substeps`, seed and backend, and the same reset observation on the same
backend and seed. Handles are not serialized and pack refs are not resurrected.

`from_config` uses the one `Run.from_config` parser and passes
`discover_runtime=False`: a config that references a pack needs `pack_set=` and
fails with a teaching `ConfigError` naming `pack_set=` / `pack_runtime=`
instead of falling back to ambient entry-point discovery. It refuses a config
with no manual control face, because the facade is not a policy-adapter driver.
A config-mode builder cannot declare new objects (`ConfigModeError`); its
handles are re-adopted from the live instance after `build()`.

## Acceptance evidence

The first-spike acceptance script
(`tests/_support/sim_facade_script.py`) has no backend branch. Lanes differ only
in the fixture they supply:

- `mujoco` (D0): the `facade_mjcf` fixture pack and the `tiny_arm` MJCF.
- `isaaclab2` (external lane): a lane-owned G1 fixture pack and an inline
  terrain world.

Claim: **same script, lane-owned fixture**. This is not a same-asset claim
(there is no dual-format test pack yet; see the residuals). The cross-backend
part is recorded A6 evidence; A7 did not rerun the `isaaclab2` lane.

## Scope and non-goals

In scope for `SIM_API_VERSION = 1`:

- One explicit `PackSet`, explicit world/control declarations, identity-scoped
  handles.
- `Control.position_pd` only.
- `N == 1` acceptance (the first-spike scope); the builder and handles keep
  the leading env axis, and `mujoco`/`mjwarp`/`isaaclab2` have `N > 1`
  backends (the `N > 1` facade plan is separate).
- `mujoco` and `isaaclab2` backend acceptance; `mjwarp` and `isaaclab3` are
  not part of the spike claim (`mjwarp` supports `N > 1`; it is out of this
  spike's lane/fixture scope, not out of capability).
- Explicit `to_config` / `from_config` semantic round-trip.

Non-goals:

- No Gym/VectorEnv integration; the `GymRunner` family stays the RL path.
- No change to `Run`, `make`, `NamedAction`, control-law, runner, contact or
  backend semantics.
- No removal or deprecation of the low-level API.
- No hidden defaults, pack discovery, or second parser.

Deferred (exact follow-up, not open questions):

| Deferred | Reason |
| --- | --- |
| Multi-env acceptance / `N > 1` promises | First-spike acceptance is `N == 1`; `mujoco` (native per-env `MjData`), `mjwarp` (batched warp + CPU mirrors) and `isaaclab2` (scene cloning) have `N > 1` backends, and the facade `N > 1` plan is separate. |
| Camera frames / rendering | `Run` hard-codes `enable_cameras=False`; the manual adapter requests no images. |
| `attach=BodyHandle` cameras | A canonical body id needs the build-selected native asset name. |
| Typed contacts / props / object handles | `contact/` keys are not backend-symmetric in `extra`. |
| Subset `joints=` action bindings | Full-face bindings only in this spike. |
| Same-asset cross-backend run | No dual-format (MJCF + USD/URDF) test pack in both lanes. |
| Per-env action masks | No consumer yet. |

## Honest residuals

- The facade configures the manual adapter; it is not a policy driver.
- Pack activation registers the robot profile process-wide. This is the
  existing pack boundary, not a facade promise.
- `read_extra` is called only at spawn and reset; a prop-enabled IsaacLab run
  has an `_build_extra` prop-manager side effect that is out of spike scope.
- Camera frames are opportunistic: they appear only if the backend publishes
  them in `obs_cache`.
- The full IsaacLab 2 dog lane is external and was not rerun for the
  documentation slice; the targeted facade test is recorded as passing.
- The facade is experimental: promotion requires independent review, a
  dual-format test pack, multi-env tests, camera/contact contracts and a
  revisited activation-scoped profile channel.
