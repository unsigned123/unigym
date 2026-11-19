# Design brief: IsaacLab 3 backends (PhysX + Newton)

Status: G2 approved (Q1-Q6 confirmed; Q1 revised 2026-11-19 to remove all
aliases). Target stack: IsaacLab 3.0.0rc1 / Isaac Sim 6.1.0.0 / Kit 110.3.0 /
Newton 1.6.1 / MuJoCo-Warp 3.12 / Warp 1.17 / Python 3.12.

## Goal

Add IsaacLab 3.x support to `unigym-core` as **two equal backends**:

- `isaaclab3_physx` — IsaacLab 3.x running `PhysxCfg`.
- `isaaclab3_newton` — IsaacLab 3.x running `NewtonCfg` (MJWarp solver first).

Both implement the existing capability protocols (`EnvBackend`, cameras,
contacts, props, director, terrain, events). Engine-neutral and IsaacLab-3.x
implementation lives **once** in
`unigym/backends/shared/isaaclab3_common/`; each backend package owns only its
`EngineProfile` values and genuinely engine-only adapters. The repo-internal
`wxyz` quaternion invariant is preserved at the data boundary.

The backend surface is **canonical-only**: `mujoco`, `mjwarp`, `isaaclab2`,
`isaaclab3_physx`, `isaaclab3_newton`.  There are no `isaaclab` or `isaaclab3`
aliases and no version-based family auto-selection; `backend` is required and
an omitted/ambiguous value is a teaching error.  Selection verifies the
explicitly chosen stack and never silently falls back.

The user-visible outcome: a config selects exactly one backend
(`backend: isaaclab3_physx` or `backend: isaaclab3_newton`) and gets honest
capability gating, no silent fallback, and one implementation to maintain.

## Non-goals

- No convergence with `isaaclab2`; this slice does not make `isaaclab2` import
  the shared package and does not touch its behaviour.
- No new `Trait` enum values. Capability differences are expressed by two
  backend identities and their trait tables.
- No new public `BackendInfo` fields unless the P0 probe proves a gap that
  existing fields cannot express.
- No Kamino / XPBD / VBD as separate backends; the solver is a Newton option.
- No kitless Newton in this slice. Newton first runs inside the same Kit launch
  path; kitless is a follow-up after the backend is green.
- No deformables, visuo-tactile sensors, `SNAPSHOT`/rewind, or motion-vector
  camera outputs.
- No golden/npz regression lane; `docs/testing.md` currently forbids it.
- No changes to the base `.venv` dependency set; IsaacLab 3 stays an external
  conda lane.
- No mujoco/mjwarp changes.

## P1b extraction plan

The il2 backend is a **reference, not a dependency**: the Non-goal above means
P1b does not modify `isaaclab2` and does not make it import this package.  The
A/B code is copied and adapted into `shared/isaaclab3_common`:

1. **A-class pure modules** first (no engine API): naming, cfg-class table, USD
   inventory, asset-scene names/inspect, terrain base + custom generators,
   neutral pose sampling.  `asset_scene/artifacts` is import-safe in this
   slice; its engine-backed call path is completed in B.
2. **B-class cfg builders** next, with the IsaacLab 3 cfg-class table and the
   `to_isaaclab_cfg` quaternion boundary: asset-scene engine/fields, static
   shapes, robot builder/realizer, prop spawn, actuator plan, terrain presets.
   `build.py` and `model.py` are P1b-B2; `build_events` stays a teaching stub
   until the C-class DR events land, and `model.describe` is PhysX-first (the
   Newton view gap is a P1b-C item).
3. **C/D env core** last: shared env base, articulation runtime, scene/model,
   cameras, contacts, director, DR events, prop handlers, build faces.  The
   shared `env.py` is the engine-coupled base (it inherits `ManagerBasedEnv`);
   backend `env.py` subclasses it and sets `backend_name` / `engine_profile` /
   `BACKEND_CAPABILITIES`.  Shared `env.py` is imported only by backend
   `env.py`, never by a package `__init__`.  These
   modules are engine-coupled by nature: they may use module-level
   torch/isaaclab imports, but they are never imported by the shared or
   backend package `__init__` — backend `env_class` factories import them
   lazily at construction.  The import-isolation meta test therefore covers
   the package `__init__` and the A/B leaf modules, not the C/D submodules.

Each slice keeps the il2 tree byte-identical and proves the shared package is
import-safe (no module-level engine import).  Duplication against il2 is
accepted for this release; a later convergence is a separate refactor.

## P2 slice plan

P2 turns the copied shared faces into **declared** capabilities, one backend
test at a time.  A trait is added only after the owning lane test passes on the
real stack; there is no batch declaration.  Every slice keeps `isaaclab2`
byte-identical, adds no new `Trait` value and no new public field.

- **P2-A lane harness + contacts**: real Cartpole fixtures on the PhysX lane
  (construct / reset / step / close), a `contacts: all` world-frame group, a
  real contact-producing step with a nonzero world-frame normal force, and
  post-reset zero buffers.  Declares `CONTACTS_GROUPS`; sets
  `BackendCapabilities.contacts=True`.  Newton follows in P3 only when the same
  test passes there.
- **P2-B props + free-root state**: rigid prop spawn/read, and a free-root
  asset for `set_root_state` / partial reset / `settle`.  PhysX status:
  `test_props.py` and the box `set_root_state` / partial `reset_envs` path in
  `test_state_write.py` verify the behavior, but **the props capability is
  deferred**.  `BackendCapabilities.props=True` projects
  `PROPS_RIGID + PROPS_ARTICULATION_URDF + PROPS_ARTICULATION_USD` together, so
  declaring only the rigid tier would fail the capability/trait subset
  invariant, while declaring the articulation tiers would claim them without a
  lane test.  The trait/bit land after the shared capability model can express
  partial prop support (or after all three tiers get lane tests).  Newton
  follows in P3.
- **P2-C director**: teleport / setvel / follow (kin attach) / gripper lane coverage.
  Declares `DIRECTOR_TELEPORT`, `DIRECTOR_SETVEL`, `DIRECTOR_ATTACH_SLAVE`,
  `DIRECTOR_GRIPPER` (no capability bit).  PhysX status: `test_director.py`
  drives the neutral `Director` through `handle()` and proves the rigid-prop
  teleport/setvel writes, the kinematic follow lock across `on_step` and
  release, the pin hold across `on_step` and release, and the gripper
  where/move/where round-trip.  Newton follows in P3.
- **P2-D cameras + recorder**: explicit profile `renderer_cfg`, RGB and depth
  rigs, recorder video. PhysX status: `test_cameras.py` proves a global world
  rig (RGB + `distance_to_camera` depth), a per-env body rig (RGB), and a
  per-instance world rig with a distinct env 1 state row, and writes a captured
  frame to a non-empty MP4, so `CAMERAS_RGB`, `CAMERAS_DEPTH_RIG`,
  `CAMERAS_GLOBAL`, `CAMERAS_PER_ENV`, `CAMERAS_PER_ENV_STATE` and
  `RECORDER_VIDEO` are declared with
  `BackendCapabilities.cameras=True` / `recorder=True`; the lane environment
  installs `imageio[ffmpeg]`. The depth RIG is proven while the `depth` alias
  bit is deferred (`depth_camera=False`, no `DEPTH_OBS_ALIASES` lane coverage);
  consequently an observation request selecting a depth stream is gate-rejected
  until that alias bit lands. No `CAMERAS_BATCH_RENDER`, per-env intrinsics or
  Newton camera trait is declared. Newton camera support stays undeclared
  unless the renderer path passes.
- **P2-E events + terrain closeout**: PhysX DR events lane coverage, the
  Newton `EVENTS_DR` decision, and terrain-preset lane tests.  PhysX status:
  `test_events.py` proves a reset-mode `randomize_rigid_body_com_abs` event
  applies at construction (two fixed-range envs differ by exactly twice the
  offset); `test_terrain.py` proves the flat `ground` and steps mesh `terrain`
  presets construct and step.  The steps/generator/native presets now pass only
  the engine-neutral material fields to IsaacLab 3's
  `RigidBodyMaterialBaseCfg`; the PhysX-only combine-mode kwargs were the
  shared-face bug found by this slice.  Consequence: IsaacLab 3 uses the
  solver's default combine mode (average) where isaaclab2 authored `multiply`;
  restoring multiply parity needs a PhysX-owned material factory and is a
  follow-up, not a neutral-preset responsibility.  Newton keeps `EVENTS_DR`
  undeclared until its P3 lane coverage.

P2-A owns only the lane harness, the contact test and the contact trait/cap
declarations.  A shared-face bug stops the slice as `BLOCKED`; it is fixed at
the shared owner boundary, not worked around in the test.

## P4 plan

P4 turns the lane-tested PhysX backend into a CI/nightly lane.  Newton stays
out of P4 until its P3 capability coverage exists; adding a stack-probe-only
Newton lane to nightly would advertise capability the lane does not test.

- **P4-A lane registry**: add `isaaclab3_physx` to `[nightly]`; leave
  `isaaclab3_newton` out and keep its `pytest_args` empty until P3.  Add a
  lane-registry meta test for the PhysX nightly entry.
- **P4-B workflows**: add default-skipped self-hosted PhysX jobs to
  `.github/workflows/ci.yml` and `nightly.yml`, gated by
  `vars.UNIGYM_RUN_ISAACLAB3_PHYSX == 'true'`, runner label
  `isaaclab3_physx`, and repository variables
  `UNIGYM_ISAACLAB3_PHYSX_ENV` / `UNIGYM_TEST_PACKS_WHEELHOUSE`.  The job
  installs the test-pack plugin + Cartpole wheel, checks that the lane
  environment provides `imageio_ffmpeg`, then runs
  `tools/ci/run_lane.py --lane isaaclab3_physx`.  The Isaac stack and the
  recorder encoder stay owned by the external conda environment; the job never
  installs Isaac or downloads PyPI packages.
- **P4-C docs/verification**: update the CI/nightly sections in
  `docs/testing.md`, the PhysX lane README and this brief; verify the lane
  registry, workflow YAML and `run_lane.py --dry-run`.

Non-goals: provisioning the external runner/environment from the repository;
changing Isaac dependencies; adding Newton nightly before P3; reworking the
existing d0 baseline failures.

Failure behaviour: a missing runner variable makes the job a visible skip (not
a fake pass); missing pack wheels or `imageio[ffmpeg]` fail the strict lane;
missing Isaac stays an external-environment failure.  Rollback removes the two
workflow jobs, the nightly entry, the meta assertion and the docs changes.

P4 status: PhysX registry/workflow/docs are in place and locally verified
(lane registry parse, workflow YAML, `run_lane.py --dry-run`).  The external
self-hosted runner and conda environment remain operational provisioning; the
nightly job is default-skipped until those variables exist.  Newton remains
P3-blocked and is deliberately absent from `[nightly]`.

## Files

Ownership: `shared/isaaclab3_common` owns the IsaacLab-3.x library; each
backend package owns its engine decisions and its env subclass.

```text
docs/designs/isaaclab3-backend.md                         - this brief
unigym/backends/shared/isaaclab3_common/                  - shared IsaacLab 3.x library
  __init__.py                                             - public re-exports (no engine import)
  README.md                                               - scope + dependency invariant
  engines.py                                              - EngineProfile dataclass (contract only)
  env.py                                                  - Isaaclab3EnvBase(ManagerBasedEnv)
  articulation_runtime.py                                 - joint/action/permutation runtime
  build.py                                                - env_kwargs / build_terrain / build_events
  scene.py                                                - SceneOps
  model.py                                                - USD/model introspection
  init_pose.py                                            - neutral wxyz pose sampling
  dr_events.py                                            - yaml events -> EventTermCfg
  contacts.py                                             - ContactBackend over il3 sensors
  cameras/{__init__,cfg,rigs}.py                          - CameraCfg/rigs, ProxyArray reads
  props/{__init__,spawn}.py                               - prop backend composition
  props/handlers/{ball,box,primitive,generic,articulation}.py
  director_ops.py                                         - director write paths
  director/gripper.py                                     - gripper runtime
  robots/{__init__,builder}.py                            - ArticulationCfg realizer
  actuators/plan.py                                       - ImplicitActuatorCfg plan
  asset_scene/{__init__,names,inspect,artifacts,engine,fields}.py
  static_shapes.py                                        - static collider payloads
  usd.py                                                  - USD inventory (pxr only)
  cfg_classes.py                                          - opaque external cfg-class table
  naming.py                                               - scene-field/prim naming
  terrains/{__init__,base,flat,steps,native,generator}.py
  terrains/custom/{__init__,_compat,gap,pit,slope_rough,unidirectional}.py

unigym/backends/isaaclab3_physx/                          - PhysX backend identity
  __init__.py                                             - BACKEND_INFO, lazy env factory
  env.py                                                  - Isaaclab3PhysxEnv + BACKEND_CAPABILITIES
  profile.py                                              - PHYSX EngineProfile values
  launcher.py                                             - kit launcher (AppLauncher)
  registry_faces.py                                       - engine-owned factories (camera/contact/director/...)

unigym/backends/isaaclab3_newton/                         - Newton backend identity
  __init__.py                                             - BACKEND_INFO, lazy env factory
  env.py                                                  - Isaaclab3NewtonEnv + BACKEND_CAPABILITIES
  profile.py                                              - NEWTON EngineProfile values
  launcher.py                                             - kit launcher (same launch path in this slice)
  registry_faces.py                                       - engine-owned factories + Newton-only adapters

unigym/backends/isaaclab3/                                - REMOVED in P1a; replaced by the two
                                                             canonical backend packages
unigym/backends/registry.py                               - trait tables, factories, canonical-only
                                                             names, StackProbe/selection_error
unigym/backends/__init__.py                               - dependency-direction docs
pyproject.toml                                            - import-linter contracts
ci/compat.toml                                            - environments + lanes (nightly in P4)
ci/envs/isaaclab3_physx.toml                              - pinned 3.x lane manifest
ci/envs/isaaclab3_newton.toml                             - pinned 3.x + newton lane manifest
tests/backends/isaaclab3_physx/**                         - PhysX backend lane
tests/backends/isaaclab3_newton/**                        - Newton backend lane
tests/unit/backends/test_isaaclab3_*.py                   - trait/registry/profile unit tests
tests/meta/test_isaaclab3_import_isolation.py             - import isolation
docs/testing.md                                           - lane/env/marker table
```

## Classes, protocols and registries

- **`EngineProfile`** (new, `shared/isaaclab3_common/engines.py`): frozen
  dataclass of the only engine-divergent decisions. Shared code calls it; it
  must never branch on `profile.name` in shared code.
  ```python
  @dataclass(frozen=True)
  class EngineProfile:
      name: str
      build_physics_cfg: Callable[[], object]          # PhysxCfg() | NewtonCfg(...)
      renderer_cfg: Callable[[], object] | None        # CameraCfg.renderer_cfg
      depth_background: float                           # inf (RTX) | 0.0 (NewtonWarp)
      default_joint_ordering: str | None               # "physx" | "mjwarp" | None
      get_body_com: Callable[[object], object]         # asset -> (N, B, 3) float32
      set_body_com: Callable[[object, object, object], None]  # (asset, coms, env_ids)
  ```
  Owner: `shared/isaaclab3_common` owns the contract; each backend's
  `profile.py` owns the concrete values.

  C3 note: CoM randomization (`dr_events.randomize_rigid_body_com_abs`) is
  engine-neutral and goes exclusively through `get_body_com`/`set_body_com`
  (both backed by `asset.data.body_com_pos_b` and `asset.set_coms_index`).
  PhysX writes a 7-dim `pos + identity xyzw quat` payload; Newton writes a
  3-dim position. There is no `root_physx_view` branch; shared code never
  inspects the backend identity.
- **`Isaaclab3EnvBase`** (new, `shared/.../env.py`): `ManagerBasedEnv` subclass
  carrying the existing `isaaclab2/env.py` lifecycle/step/reset logic with
  3.x data and write semantics. Takes an `EngineProfile` by construction.
- **`Isaaclab3PhysxEnv` / `Isaaclab3NewtonEnv`** (new, each backend `env.py`):
  thin subclasses. Each module defines its own `BACKEND_CAPABILITIES` because
  the capability gate reads `env_cls.__module__`
  (`unigym/env/backends/__init__.py:67`).
- **`ISAACLAB3_PHYSX_TRAITS` / `ISAACLAB3_NEWTON_TRAITS`** (changed,
  `registry.py`): per-backend trait tables. No new `Trait` values.
  - PhysX P1 start: `_CORE | {ENV_LAYOUT_TILED, MULTI_ENV, STATE_WRITE,
    MODEL_USD_PRIMS}`. P2 adds camera/contact/props/director/terrain/events
    traits only as each capability lands with a backend test.
  - Newton P3 start: the tested intersection (`_CORE`, `MULTI_ENV`,
    `ENV_LAYOUT_TILED`, `STATE_WRITE`, `PROPS_RIGID`,
    `PROPS_ARTICULATION_USD`, `PROPS_ARTICULATION_URDF`, `SCENE_COMPOSE`,
    `TERRAIN`, `INIT_POSE`, `SPAWN_XY_YAW`, `CONTACTS_GROUPS`,
    `DIRECTOR_TELEPORT/SETVEL/ATTACH_SLAVE`, `MODEL_USD_PRIMS`,
    `NATIVE_HANDLE`). Camera/`RECORDER_VIDEO`/`PROPS_LATCH`/`SNAPSHOT`/
    `CONTACTS_OVERRIDE` stay absent until proven.
- **`CANONICAL_BACKENDS`** (changed): `mujoco`, `mjwarp`, `isaaclab2`,
  `isaaclab3_physx`, `isaaclab3_newton`.
- **Aliases** (removed): no `isaaclab` / `isaaclab3` alias and no version-based
  family auto-selection.  `backend` is required; `None`, `""`, `isaaclab` and
  `isaaclab3` are rejected with a hint listing the five canonical names.
- **Selection decision** (new): a pure `StackProbe` value object plus
  `selection_error(canonical, probe)` decide stack presence/version errors.
  `select_backend(canonical)` is the only probe-backed entry; unit tests feed
  fake probes to the pure decision function (no monkeypatching).
- **Probes** (changed): `isaaclab3_newton` requires `isaaclab >= 3` **and**
  importable `newton` / `mujoco-warp` / `warp`. `isaaclab3_physx` requires
  only `isaaclab >= 3`. `isaaclab2` requires `isaaclab < 3`. Failures raise
  `BackendSelectionError{error, hint}`.
- **Build boundary**: `build_instance` verifies the explicitly chosen
  backend's stack before constructing any env; `env_kwargs` constructs the
  real env and reports teaching errors, never an internal `TypeError`.
- **`BackendInfo`** (unchanged fields in this slice): both entries fill
  `options_schema` (Newton: solver/substeps; PhysX: empty), `scopes` only where
  `max_envs` genuinely narrows, `camera_enable_required` (True for PhysX;
  Newton decided by P3 probe), `scene_provider_consumer=True`.
- **`BACKEND_CAPABILITIES`** (new per backend): a bit set matching that
  backend's declared traits. Do **not** use `BackendCapabilities.full()`.

## Interactions and data flow

```text
yaml/run spec
  -> registry.select_backend(canonical)       # explicit stack/newton probe
  -> registry.BACKEND_INFO[canonical].traits  # fail-fast capability gate
  -> backend package factories (env_kwargs, terrain, events, cameras, contacts,
     props, director, robot realizer, kit launcher)
  -> backend env class (thin) -> shared env base(profile)
  -> ManagerBasedEnv / InteractiveScene
```

Data boundary, one direction:

- **Read**: simulator data -> `ProxyArray.torch` -> (for quaternions)
  `unigym.core.quat.to_wxyz` -> repo-internal `wxyz`.
- **Write**: repo-internal `wxyz` -> `unigym.core.quat.to_xyzw` -> il3
  `write_*_index(..., env_ids=...)`.
- **Never** `torch.as_tensor(proxy_array)`; never implicit ProxyArray in
  `obs.set_robot` or TorchScript math.
- `profile.build_physics_cfg()` is applied once when the env cfg is built;
  `profile.renderer_cfg()` is applied to every declared camera that needs a
  sensor (Newton must not fall back to the default RTX renderer).

## Invariants and failure modes

Invariants:

1. Repo-internal quaternions are `wxyz`; only the il3 boundary converts.
2. `shared/isaaclab3_common` never imports `isaaclab2`, `isaaclab3_physx`,
   `isaaclab3_newton`, or mujoco-family packages.
3. `isaaclab3_physx` and `isaaclab3_newton` never import each other.
4. No module-level `isaaclab` / `pxr` / `omni` import outside the lazy
   boundaries; importing either backend package or the shared package loads no
   engine.
5. Declared trait == implemented protocol; a missing trait fails before
   construction with `{error, hint}`.
6. All simulator writes go through `*_index`; all data reads are explicit
   `.torch`; all cross-boundary quats are converted.
7. `default_root_state` / `root_state_w[...] = ...` are not used as a write
   path; writes go to the sim then `write_data_to_sim()` + `forward()`.
8. Cameras always get an explicit `renderer_cfg` from the profile (or a
   declared per-camera override); the default RTX fallback is not allowed on
   Newton.

Invalid input / failure behaviour:

- Unknown `env.backend_options.<name>` key -> `ValueError` naming the key and
  supported options (existing `validate_backend_options`).
- `isaaclab3_newton` requested without the Newton stack -> fail-fast with the
  missing distribution name and a hint to use `isaaclab3_physx`.
- A trait the backend does not declare -> gate error naming the trait and the
  backend's trait set.
- A camera spec without a renderer on Newton -> teaching error, never a silent
  RTX fallback.
- Construction before a `SimulationContext` must not let `FactoryBase` resolve
  an engine; P0 verifies the construction order and records the result.

## Compatibility and migration

- **Public API**: `known_backends()` / `CANONICAL_BACKENDS` are the five
  canonical names. `BACKEND_ALIASES` / `_BUILD_ALIASES` are removed;
  `build_name` only validates a canonical spelling and performs no probing.
  There is no implicit default: `build_name(None)`, `Run(backend=None)` and
  `build_env_backend(None)` raise a teaching error instead of falling back to
  `"isaaclab"`. Helper APIs that previously defaulted to `"isaaclab"`
  (`build_rig`, props, contacts) likewise require a canonical name.
- **Config/schema**: `backend` is required. `backend: isaaclab` and
  `backend: isaaclab3` are rejected with a hint listing the five canonical
  names; they are not aliases. `backend_options` namespaces are canonical-only
  (`isaaclab3_physx`, `isaaclab3_newton`); unknown cross-namespace keys still
  fail.
- **Executor bindings**: the executor `backends:` mapping is keyed by canonical
  env backend, not by family. A legacy `backends: {isaaclab: <executor>}` key
  is rejected with a teaching error and must be rewritten as one key per
  concrete backend (`isaaclab2`, `isaaclab3_physx`, `isaaclab3_newton`). The
  executor **name** `"isaaclab"` is unchanged: it is the docker image face, so
  `backends: {isaaclab2: isaaclab}` is still the correct binding. The
  `isaaclab` Python module root and the IsaacLab family docs keep their names;
  only backend selection and binding keys became canonical.
- **Migration**: every config that omitted `backend` or used
  `isaaclab`/`isaaclab3` must name `isaaclab2`, `isaaclab3_physx` or
  `isaaclab3_newton` explicitly, and every executor binding key must be
  renamed accordingly. This is intentional fail-fast, not a silent default.
- **Dependency/version lane**: new strict conda lanes pinned to the dog stack
  (isaaclab 3.0.0rc1, isaacsim 6.1.0.0, newton 1.6.1, mujoco-warp 3.12,
  warp-lang 1.17, python 3.12). `OMNI_KIT_ACCEPT_EULA=YES` is supplied only to
  these lanes.
- **Docs pass**: one docs pass (`docs/testing.md`, backend READMEs, contracts
  that name `isaaclab3` as a stub). Tests that assert the empty il3 trait set
  or the old canonical name are updated in the same change.
- **Rollback**: remove the two backend packages, their lanes and registry
  entries to restore the previous state. Configs must already name a canonical
  backend; there is no alias fallback to preserve.

## Tests

Lowest layers first:

- `tests/unit/backends/test_isaaclab3_registry.py` — both trait tables, trait
  rules, `TraitScope` validity, canonical names, alias rejection,
  `known_backends`, `build_name`, backend-option namespaces, and the **pure**
  `selection_error(canonical, StackProbe)` decision table.  No monkeypatching
  of version probes.
- `tests/unit/backends/test_isaaclab3_engine_profile.py` — the frozen
  EngineProfile contract (no engine branch).
- `tests/contract/test_registry_probe_isaaclab3.py` — `select_backend` error
  types/messages for a missing/wrong stack and for rejected aliases.
- `tests/meta/test_isaaclab3_import_isolation.py` — importing either backend or
  the shared package loads no `isaaclab`/`pxr`/`omni`/`newton`/`warp`.

Backend lanes (real sim):

- `tests/backends/isaaclab3_physx/` — mirror the `isaaclab2` lane coverage:
  construct / step / reset subset & all / close / action faces / state writes /
  data boundary quat / cameras / contacts / props / terrains / DR / director /
  robot realizer / multi-articulation.
- `tests/backends/isaaclab3_newton/` — smoke / reset / state writes / contact
  normal-force semantics / declared-absent traits fail-fast / ordering /
  determinism.

Lane registry:

- `ci/compat.toml`: `[environments.isaaclab3_physx]` and
  `[environments.isaaclab3_newton]` (both `strict = true` for their pinned
  manifests) plus `[lanes.isaaclab3_physx]` and `[lanes.isaaclab3_newton]`.
  In P1 the lanes are stack probes: they declare no `pytest_args` and are out
  of `[nightly]`. P4 adds `pytest_args = ["--strict-test-packs"]` once the
  backends consume test packs and joins `[nightly]`.
- `tests/meta/test_lane_registry.py` picks them up from `ci/compat.toml`; the
  two test directories are siblings, so no selector overlap.
- `docs/testing.md`: add the lanes, env vars and the EULA note.

Existing tests that must change in the same slice:

- `tests/unit/backends/test_state_write.py` (asserts il3 lacks `STATE_WRITE`).
- `tests/unit/backends/test_native_handle.py` (asserts il3 lacks
  `NATIVE_HANDLE`).
- `tests/unit/run/test_world_flat_resolution.py` (asserts il3 has no flat
  terrain).
- Any test asserting `select_backend("isaaclab3") == "isaaclab3"` or iterating
  the old canonical tuple.

Fast-suite impact: registry/marker/lane meta changes only. No heavy import in
`fast`; no sleeps, no machine paths, no network.

Boundary/failure cases: missing Newton stack; trait absent; unknown
backend-option; camera without renderer; `ProxyArray` leaking into obs; xyzw
double-conversion; `torch.as_tensor(ProxyArray)` regression.

## Risks, alternatives and resolved decisions

Risks:

- **ProxyArray / TorchScript**: `.torch` must be explicit. On the real rc1
  stack implicit torch operations (including `torch.as_tensor`) work but emit
  `DeprecationWarning`, and `ProxyArray.shape` is warp-shaped (for example
  `(N, B)` while `.torch` is `(N, B, 3)`). Highest-frequency mechanical risk;
  covered by unit + lane tests.
- **Camera/visualizer/recording**: rc1 moves the default renderer to RTX/Kit;
  `TiledCameraCfg` is a deprecated shim. Highest behavioral risk; keep camera
  traits last in PhysX and absent in Newton until proven.
- **Newton ordering/semantics**: default DFS vs PhysX BFS; friction is a force
  not a coefficient; CoM has no orientation; `root_physx_view` does not exist.
  Mitigation: `default_joint_ordering` in the profile; declare only tested
  traits; document restrictions in the Newton README.
- **`FactoryBase` early resolution**: without a `SimulationContext`,
  `_get_backend()` may resolve to Newton. P0 must verify our cfg-before-sim
  construction order.
- **Kit 110 pre-launch imports**: only `isaaclab.app` before launch; the
  backend packages stay import-safe.
- **Version churn**: rc1 has deprecation aliases that 3.1/3.2 remove. Pin the
  lane; do not rely on deprecated names where the canonical name is known.
- **Code drift**: two backends must not copy shared code. Enforced by
  import-linter and `shared/isaaclab3_common` ownership.
- **Scope creep**: kitless Newton, `PROPS_LATCH`, `SNAPSHOT` are follow-ups.

Alternatives considered:

- **One backend + `backend_options.physics`**: rejected because `TraitScope`
  can express only `max_envs`; capability truth would be lost or need a new
  mechanism.
- **Keep `isaaclab3` as the PhysX canonical name** (add only
  `isaaclab3_newton`): lower churn, but asymmetric. Q1 resolved in favour
  of the symmetric names.
- **Copy `isaaclab2` wholesale then diverge**: rejected; 45% of the code is
  version-neutral and the `mjfamily` precedent shows the shared-library shape.

Resolved decisions (Q1-Q6 confirmed):

- **Q1 confirmed (revised)**: canonical names only — `mujoco`, `mjwarp`,
  `isaaclab2`, `isaaclab3_physx`, `isaaclab3_newton`.  No `isaaclab` /
  `isaaclab3` aliases and no family auto-selection; `backend` is required.
  Selection uses the pure `StackProbe` / `selection_error` decision function;
  tests feed fake probes instead of monkeypatching version functions.
- **Q2 confirmed**: `shared/isaaclab3_common` is scoped to the IsaacLab 3.x
  family. A version-agnostic subset for `isaaclab2` is a separate refactor.
- **Q3 confirmed**: Newton runs inside the Kit lane in this slice; kitless is a
  follow-up.
- **Q4 confirmed**: `PROPS_LATCH` / `SNAPSHOT` stay out until the base backend
  is green.
- **Q5 confirmed**: the first Newton trait set is the tested intersection above
  (cameras absent until proven).
- **Q6 confirmed**: the import-linter additions are approved (shared-library
  privacy, il2/il3 independence, physx/newton independence, shared never
  imports a backend).

## P0 scope and deviations

P0 delivers the probes and the lane scaffolding only:

- two lane identities (`isaaclab3_physx`, `isaaclab3_newton`) with pinned
  manifests, markers, stack probes and docs;
- no `unigym/` product-code change in P0; P1a makes the canonical-only
  selection change and removes the old `isaaclab3` skeleton package;
- no `--strict-test-packs` (no test packs are consumed yet) and no `[nightly]`
  entry / GitHub job (no self-hosted runner and environment yet).

These are deliberate P0 boundaries, not silent fallbacks: the lane READMEs and
`docs/testing.md` state them, and P1 starts by closing the registry rename and
the lane provisioning (pytest in the lane environment) before any capability is
declared.

## P0 probe checklist (completed on the reference lane)

1. Registry probe on dog: `isaaclab_version() == (3, 0)` and
   `select_backend("isaaclab3_physx")` resolves explicitly (the old
   `select_backend("isaaclab3")` alias is intentionally removed in P1a).
2. Explicit import surfaces used by the backend all resolve without
   `fromlist=["*"]`.
3. `AppLauncher.add_app_launcher_args` adds no `--headless`/`--enable_cameras`;
   passing `--headless` is rejected; `AppLauncher(headless=True)` works.
4. `SimulationCfg(physics=PhysxCfg())` and
   `SimulationCfg(physics=NewtonCfg(MJWarpSolverCfg()))` construct; PhysX
   `SimulationContext.reset()` passes; Newton builds its solver but `reset()`
   on an empty stage fails for lack of a joint (needs a real scene).
5. `FactoryBase._get_backend()` returns `"newton"` when no
   `SimulationContext` exists; the construction-order guard is required in P1.
6. On the real GPU, `ProxyArray.torch`/`.clone`/`.detach` and even
   `torch.as_tensor(ProxyArray)` work (with deprecation warnings);
   `ProxyArray.shape` is warp-shaped, so explicit `.torch` remains the rule.
7. The default camera renderer resolves to `IsaacRtxRendererCfg`; an explicit
   `NewtonWarpRendererCfg` is accepted.
