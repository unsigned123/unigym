# World, EnvSpec and `World.flat()`

## Ownership

A `World` owns the **environment only**:

- the scene asset (`scene_preset` / `world_base`) and
- static or dynamic `objects`.

A `World` never owns a robot, a spawn pose, an action layout or a policy
binding. `World.fragment()` emits `scene_preset` / `world_base` / `objects`
and never `articulations` / `action_spec` / `policies`. Those tables are the
owner of the `Run` / `EnvSpec` layer:

- robot identity and asset (`Run(robot=<RobotProfile>)`),
- root spawn (`Run(..., spawn={"pos": ...})`),
- the explicit action layout (`action_spec=`) and policy bindings
  (`policies=`).

`Run(world=...)` merges the world fragment into the run config exactly once
and marks the world built after the env compiles.

## `World.flat()`

`World.flat()` is the robot-free, backend-neutral flat ground used by RL
examples. Its declaration is deliberately empty (`fragment() == {}`): the
`Run` layer resolves it to the backend's own ground.

| Backend | Resolution |
| --- | --- |
| `mujoco`, `mjwarp` | `world_base: {mjcf: unigym/assets/worlds/flat_ground.xml}` |
| `isaaclab2` | `terrain: {name: flat, params: {}}` |
| any backend with neither a `world_base` consumer nor a native flat terrain (e.g. the `isaaclab3_*` stubs) | fails fast with `ConfigError`; the world would otherwise silently contribute no ground |

An explicit world declaration wins: when the caller also sets `world_base`
or `scene_preset` (or `terrain` for a terrain backend), the flat declaration
is not resolved on top of it.

## `<option>` inheritance during composition

`<option>` is a model-global MJCF block, and `MjSpec.attach` never transfers
child option values. The composition engine (`unigym/backends/shared/mjfamily/
scene/articulation_compose.py`) therefore applies one explicit rule on the
**single-articulation attach route** (a world base without the
`<!-- articulations -->` marker — the packaged `flat_ground.xml` takes this
route):

- Every `<option>` attribute the articulation asset explicitly declares is
  copied into the world base when the base does not declare that same
  attribute. The robot asset stays the source of its solver/contact settings
  (`cone`, `impratio`, ...); the world base stays the source of the attributes
  it declares.
- Same attribute declared by both sides: different values are a declaration
  conflict and fail fast with `ConfigError`; equal values pass.
- The marker / `<include>` route keeps MuJoCo's own include merge untouched
  (an included articulation's explicit option applies; this is the historical
  fused `scene.xml` behavior).
- Multiple articulation assets (or `auto_prefix`): options are not inherited;
  the world base is the single owner. There is no defined precedence between
  two robot assets, so a non-default physics setup must be declared on the
  world base.

The packaged `flat_ground.xml` intentionally declares no `<option>`, so the
Go2 (and every other single-robot flat-world profile) keeps the asset's
`cone` / `impratio` instead of silently falling back to MuJoCo defaults.

## Residual limits

- `unigym.sim.SimulationBuilder` requires `builder.world` to declare a
  non-empty world asset, so it still rejects `World.flat()` with
  `NoWorldError`. Use `Run(world=World.flat(), ...)` for flat RL tasks until
  the facade gains a world-less declaration form.
- The `isaaclab3_physx` / `isaaclab3_newton` backends are P1 stubs (canonical
  names with an empty registered trait set) and have no ground consumer;
  `World.flat()` on either is a loud `ConfigError` by design.
