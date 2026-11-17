# Simulation facade (experimental)

The `unigym.sim` facade is a thin, object-first surface over the existing
`Run` / `ExternalRunner` / backend / IR stack. It is additive: the low-level
API is unchanged and remains the supported path for policy adapters, Gym-style
training and backend-specific work.

```text
SimulationBuilder -> Simulation -> Run / ExternalRunner / backend
```

## Experimental status

- `unigym.sim.SIM_API_VERSION` is `4`. There is no stable-API promise; an
  incompatible facade change bumps the version.
- The facade is proven for **one articulation** on the tested coverage:
  `mujoco` at `N ∈ {1, 2, 3}` and `isaaclab2` at targeted `N ∈ {1, 2}` (full
  dog lane unverified); see [Multi-env](#multi-env). `mjwarp` has an `N > 1`
  backend, but its lane and fixtures are outside this spike; camera frames,
  contact parity and object/prop handles remain deferred.
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
group = robot.action_groups["body"]    # the ordered action template group

with builder.build() as sim:
    obs = sim.observe()                            # spawn state, t == 0
    print(obs.articulations[robot].joint_pos)

    for _ in range(3):
        sim.act(group, torch.zeros((1, len(group.joints)), dtype=torch.float32))
        obs = sim.step()                           # one tick; first step keeps t == 0
```

`t` is 0-based and names the tick that produced the observation: spawn is
`t == 0`, the first `step()` also returns `t == 0`, and after `N` steps
`obs.t == N - 1`.

The named single-env form is equivalent when the handle declares exactly one
action group:

```python
sim.act(robot, {"shoulder": 0.1, "elbow": -0.2})
obs = sim.step()
```

`step(actions={group: values, ...})` is sugar: it stages every entry
atomically and then runs the normal `step()`. A failed entry clears the pending
table and leaves the tick/state untouched. The canonical path is `act` +
`step()`. Finiteness is checked for Python numbers only; a tensor's
`NaN`/`inf` values are not inspected.

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
  world fragment may not declare `articulations` / `action_spec` / `policies`;
  the builder is the single owner of those tables. `builder.terrain` is sugar for
  `builder.world.terrain`.
- `add_articulation(profile, *, name=None, control=...)` requires an explicit
  `Control` spec; there is no implicit control. Authoring action groups are
  resolved against the `RobotProfile`, their authoring identity is
  `(articulation, name)`, and the groups of one articulation must be disjoint.
  This identity is a facade/authoring handle, not a runtime routing key.
- `add_camera(entry, *, name=None)` accepts an existing `unigym.Camera.*`
  `cameras[]` entry mapping. `attach=BodyHandle` is deferred and raises
  `DeferredFeatureError`; use a native mount name through
  `Camera.attached(name, mount=...)`.
- After the first successful `build()`, declarations raise
  `BuilderSealedError`; a repeated `build()` returns the same `Simulation`.
- `builder.articulations` / `builder.cameras` are the declared handles. In
  config mode they are empty until `build()`.

## Pack acquisition

The facade never discovers packs. Declaration-mode articulations are complete
`RobotProfile` objects; a config-mode run that references pack resources must
belong to the builder's explicit `PackSet`:

```python
import unigym.packs

packs = unigym.packs.installed()                  # configured manifest-only scope
packs = unigym.packs.require("acme")              # filtered to exactly "acme"
packs = unigym.packs.require("acme", entry_points=False)  # manifest-only
packs = unigym.packs.discover_set(["/path/to/pack/root"])
```

`packs.pack(name).robot(id)` returns the complete `RobotProfile` (a resource
document is read directly; an entrypoint factory must return a profile) without
activating the pack or mutating the robot registry.
`SimulationBuilder.add_articulation(profile, control=...)` consumes that
profile directly: no pack lookup, no activation and no implicit default
control. A renamed articulation (`name=...`) emits the existing
`params.profile` selector from `profile.name`.

`builder.pack_runtime` is the activation-capable runtime over the same set. A
world preset is declared explicitly:

```python
builder.world = World.from_preset(preset, runtime=builder.pack_runtime)
```

## Actions

Actions use the ordered action template as an **authoring handle** at the
facade. Each articulation handle carries a read-only `action_groups` mapping of
`OrderedActionGroup` objects, and `Simulation` holds the global
`OrderedActionSpec` built from those groups. A group fixes its authoring
`(articulation, name)` identity, joint order, target kind and body/aux role; the
same group name may appear on different articulations. The public facade
behavior is unchanged: `act(group, ...)` still stages exactly one declared
authoring group and `step()` still requires every declared group to be staged.

The boundary change is only in lowering: `PendingActions.lower()` produces
`tuple[NamedAction, ...]` with **no group field**. Each lowered entry carries
the staged group's articulation, its joint names in the group's declared order,
its values and its kind; backends then route every column by
`(articulation, joint_name)`. The group is an authoring handle, never part of
the runtime value.

- `group = handle.action_groups[name]` is the only action handle. There is no
  separate action-to-column binding layer and no column-order choice: the
  group's joint tuple is the tensor column order. A `PolicyBinding` is a
  separate declaration (adapter ownership and `control_dt`), not a column
  mapping.
- `act(group, values)` requires a floating `torch.Tensor` of shape
  `(num_envs, len(group.joints))`, or `(len(group.joints),)` when
  `num_envs == 1`. Values are normalized to `(N, dim)` and cloned.
- `act(handle, values)` is the single-group convenience. It is legal only when
  the handle declares exactly one action group; a multi-group handle must pass
  the `OrderedActionGroup` itself. A mapping must cover exactly the group
  joints; its values are reordered into the group's declared joint order, so
  mapping key order never changes the action layout. A tensor is accepted in
  the group's declared joint order, exactly like `act(group, tensor)`. At
  `N == 1` a value may be a finite number or a floating `(1,)` tensor; the
  per-env rules are in [Multi-env](#multi-env).
- An `OrderedActionGroup` is an ownerless template: staging one that does not
  exactly equal a declared group is a `BindingError` (there is no
  `ForeignHandleError` ownership claim for action groups).
- Every target is validated against the simulation's `OrderedActionSpec`: an
  unknown identity, a group whose joints/target differ from the template, a
  partial group, or a wrong value shape is a `BindingError` before staging.
- At most one pending entry per `(articulation, group)` identity. `step()`
  requires **every** declared group to be staged; an unstaged group is
  `MissingActionError`. The previous action is never reused.
- All validation happens before the tick. An engine exception clears pending,
  marks the simulation `faulted`, and propagates; later live calls raise
  `SimulationFaultedError` until `close()`.

## Multi-env

`num_envs=N` builds `N` rows on `mujoco` and `isaaclab2`. The leading env axis
is explicit: every action is `(N, dim)` and every projected tensor is
`(N, ...)`. The facade never broadcasts, fills or skips a row.

```python
sim.act(group, torch.zeros((N, len(group.joints)), dtype=torch.float32))  # exact (N, dim)
sim.act(robot, {"shoulder": torch.full((N,), 0.1), "elbow": torch.zeros(N)})
```

- `act(group, tensor)` requires an exact `(N, dim)` floating tensor; the
  `(dim,)` short form remains legal only at `N == 1`. A `(1, dim)` tensor,
  a `(dim,)` vector or a 3-D tensor at `N > 1` is a `BindingError` naming the
  actual and the expected `(num_envs, dim)`. `act(handle, tensor)` on a
  single-group handle has the same rule, with columns in the group's declared
  joint order.
- `act(handle, {joint: value})` is the per-env named form. At `N > 1` every
  value must be a floating `(N,)` tensor with one dtype and device; a Python
  number is refused with a `BindingError` naming the joint and the explicit
  broadcast, `torch.full((N,), value)`. The columns are stacked in the group's
  declared joint order into the same `(N, dim)` staging path. At `N == 1` finite
  numbers and `(1,)` tensors may mix.
- `reset(env_ids=...)` resets exactly the named rows; `reset()` (or
  `env_ids=None`) resets all rows. Ids are ints in `[0, N)`, non-empty and
  unique; a subset reset does not touch the other rows.
- `step()` never resets a row. It advances all rows one tick and reports:
  `obs.reset_env_ids` is empty, `obs.reset_due_env_ids` names the
  episode-clock due rows, and `obs.terminated` is the `(N,)` termination mask
  (`None` at spawn and after `reset()`). A due or terminated row keeps
  stepping until the caller resets it:

  ```python
  obs = sim.step()
  ids = torch.cat([obs.reset_due_env_ids,
                   torch.nonzero(obs.terminated).flatten()]).unique()
  if ids.numel():
      obs = sim.reset(env_ids=ids)
  ```

- The builder emits `runner.auto_reset: false`, which is how `step` keeps that
  promise. A config-mode build of a foreign config that leaves
  `runner.auto_reset` true is refused with a `SimulationConfigError` naming the
  key, rather than silently overriding the user's config.
- `num_envs > 1` together with a declared camera is a
  `DeferredFeatureError`; cameras at `N > 1` are not part of this contract.

## Observations

One `Observation` belongs to the producing tick in `obs.t`: spawn is `t == 0`,
the first `step()` also returns `t == 0`, and after `N` steps `t == N - 1`.

| Field | Meaning |
| --- | --- |
| `t` | The 0-based policy tick that produced the cached state (`N - 1` after `N` steps). |
| `num_envs` | The declared env count; every projected tensor has this many rows. |
| `articulations` | `Mapping[ArticulationHandle, ArticulationObservation]`. |
| `cameras` | `Mapping[CameraHandle, CameraFrame]` for published frames only. |
| `contacts` | `contact/...` tensors present in the live state (no backend parity claim). |
| `reset_env_ids` | The env rows **this operation performed** a reset on; empty after a step. |
| `reset_due_env_ids` | The episode-clock **due** rows at the producing tick; empty after a reset. |
| `terminated` | The producing tick's `(N,)` termination mask; `None` at spawn/reset. |
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
- `reset(env_ids=...)` resets the named rows (or every row when `None`) through
  the same runner, clears pending and refreshes the cached state. A subset
  reset does not move the clock and does not make the other rows stale.
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
| `SimulationConfigError` | Invalid declaration or caller argument: dt arithmetic, labels, backend name, group/profile coverage, config-mode requirements, `reset(env_ids=...)` ids (empty, non-int, out of range or duplicate), or a foreign config with `runner.auto_reset: true`. |
| `ForeignHandleError` | An articulation or camera-attach body handle, or a pack ref, belongs to another builder/simulation or pack set. |
| `BindingError` | Unknown action-group identity, a group that differs from the template, a bad shape/dtype, or an ambiguous/partial named action. |
| `PendingActionError` | Duplicate group identity, empty pending on `step()`, or a staged group that is not declared. |
| `MissingActionError` | A required action group is not staged. |
| `StaleStateError` | The runner advanced out of band; the facade clock is behind. |
| `ObservationError` | A required state key is missing or invalid, `q0` is unavailable, or a projected tensor/mask violates its `num_envs` contract (wrong row count, non-int64/out-of-range/duplicate reset ids, or a `terminated` mask that is not `(N,)` bool). |
| `SimulationClosedError` | A live call after `close()`. |
| `SimulationFaultedError` | A live call after an engine exception; only `close()` is safe. |
| `BuilderSealedError` | A declaration after the first successful `build()`. |
| `ConfigModeError` | `add_articulation` / `add_camera` / `world=` on a `from_config` builder. |
| `DeferredFeatureError` | A declared out-of-scope feature: `attach=BodyHandle` cameras, or camera frames at `N > 1`. |

## Config loading

`SimulationBuilder.from_config` builds a facade over an existing run config
(the world, articulation, control and camera tables), through the one
`Run.from_config` parser.

```python
reloaded = SimulationBuilder.from_config("sim.yaml", pack_set=packs)
sim = reloaded.build()
```

The document must load semantically: same labels, articulation/control
routing, camera table, world asset, `dt` / `physics_dt` / `substeps`, seed and
backend, and the same reset observation on the same backend and seed. Handles
are not serialized and pack refs are not resurrected.

`from_config` passes
`discover_runtime=False`: a config that references a pack needs `pack_set=` and
fails with a teaching `ConfigError` naming `pack_set=` / `pack_runtime=`
instead of falling back to ambient entry-point discovery. It refuses a config
with no manual policy binding, because the facade is not a policy-adapter
driver. A config-mode builder cannot declare new objects (`ConfigModeError`);
its handles and action groups are re-adopted from the live instance and the
`RunSpec.action_spec` after `build()`.

## Acceptance evidence

The shared acceptance script (`tests/_support/sim_facade_script.py`) has no
backend branch and takes `num_envs`. Lanes differ only in the fixture they
supply:

- `mujoco` (D0): the `facade_mjcf` fixture pack and the `tiny_arm` MJCF;
  `N ∈ {1, 2, 3}` run through the lane.
- `isaaclab2` (external lane): a lane-owned G1 fixture pack and an inline
  terrain world; targeted `N ∈ {1, 2}` on the recorded dog lane (full dog lane
  unverified).

Claim: **same script, lane-owned fixture**. This is not a same-asset claim
(there is no dual-format test pack yet; see the residuals). The targeted
`isaaclab2` part (`N ∈ {1, 2}`) is recorded dog-lane evidence, not a
cross-backend parity claim, and the full dog lane remains unverified.

## Scope and non-goals

In scope for `SIM_API_VERSION = 4`:

- One explicit `PackSet`, explicit world/control declarations, identity-scoped
  handles and ordered action groups from the run/action spec.
- `Control.position_pd` only.
- Tested multi-env coverage on `mujoco` at `N ∈ {1, 2, 3}` and targeted
  `isaaclab2` at `N ∈ {1, 2}` (full dog lane unverified): explicit `(N, dim)`
  actions, the per-env named form, `reset(env_ids=...)`, and due/terminated
  reporting with `step` never resetting a row.
- `mujoco` and `isaaclab2` backend acceptance; `mjwarp` and `isaaclab3` are
  not part of the spike claim (`mjwarp` supports `N > 1`; it is out of this
  spike's lane/fixture scope, not out of capability).
- Explicit `from_config` semantic loading.

Non-goals:

- No Gym/VectorEnv integration; the `GymRunner` family stays the RL path.
- No semantic change to `Run`, `make`, `NamedAction`, control-law, contact or
  backend behavior; the multi-env spike only forwards the existing reset
  policy and fixes the empty-group termination mask shape.
- No removal or deprecation of the low-level API.
- No hidden defaults, pack discovery, or second parser.

Deferred (exact follow-up, not open questions):

| Deferred | Reason |
| --- | --- |
| Camera frames / rendering | `Run` hard-codes `enable_cameras=False`; the manual adapter requests no images. |
| `attach=BodyHandle` cameras | A canonical body id needs the build-selected native asset name. |
| Typed contacts / props / object handles | `contact/` keys are not backend-symmetric in `extra`. |
| Subset `joints=` action groups | Full-group values only in this spike. |
| Same-asset cross-backend run | No dual-format (MJCF + USD/URDF) test pack in both lanes. |
| Per-env action masks | No consumer yet. |
| Per-env `episode_length_s` | The builder accepts one float for all rows; per-row episode clocks are read-only in `raw`. `None` disables the core clock and is only valid with `termination_mode: external` (an omitted value is `None` there, since the task owns truncation). |

## Honest residuals

- The facade configures the manual adapter; it is not a policy driver.
- Pack activation registers the robot profile process-wide. This is the
  existing pack boundary, not a facade promise.
- `read_extra` is called only at spawn and reset; a prop-enabled IsaacLab run
  has an `_build_extra` prop-manager side effect that is out of spike scope.
- Camera frames are opportunistic: they appear only if the backend publishes
  them in `obs_cache`.
- The full IsaacLab 2 dog lane is external and remains unverified; the targeted
  `N == 1` and `N == 2` facade tests are recorded as passing in isolation.
- The facade is experimental: this spike's independent review returned
  `APPROVE`, but promotion still requires a dual-format test pack,
  camera/contact contracts and a revisited activation-scoped profile channel.
