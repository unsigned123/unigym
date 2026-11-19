# Design brief: IsaacLab 3 backends (PhysX + Newton)

Status: G2 approved (Q1-Q6 confirmed). Target stack: IsaacLab 3.0.0rc1 /
Isaac Sim 6.1.0.0 / Kit 110.3.0 / Newton 1.6.1 / MuJoCo-Warp 3.12 / Warp 1.17 /
Python 3.12.

## Goal

Add IsaacLab 3.x support to `unigym-core` as **two equal backends**:

- `isaaclab3_physx` — IsaacLab 3.x running `PhysxCfg`.
- `isaaclab3_newton` — IsaacLab 3.x running `NewtonCfg` (MJWarp solver first).

Both implement the existing capability protocols (`EnvBackend`, cameras,
contacts, props, director, terrain, events). Engine-neutral and IsaacLab-3.x
implementation lives **once** in
`unigym/backends/shared/isaaclab3_common/`; each backend package owns only its
`EngineProfile` values and genuinely engine-only adapters. The repo-internal
`wxyz` quaternion invariant is preserved at the data boundary. `isaaclab3`
remains a supported spelling (alias to `isaaclab3_physx`); `isaaclab`
auto-routing keeps PhysX as the default.

The user-visible outcome: a config can select `backend: isaaclab3_physx` or
`backend: isaaclab3_newton` and get honest capability gating, no silent
fallback, and one implementation to maintain.

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

unigym/backends/isaaclab3/                                - REMOVE in P1 (Q1 approved: symmetric names);
                                                             P0 keeps it as the probe target
unigym/backends/registry.py                               - trait tables, factories, aliases, probes
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
      set_body_com: Callable[[object, object, object], None]  # (asset, com, env_ids)
  ```
  Owner: `shared/isaaclab3_common` owns the contract; each backend's
  `profile.py` owns the concrete values.
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
- **Aliases** (changed): `isaaclab3` -> `isaaclab3_physx`; `isaaclab` auto
  -> PhysX for the 3.x major, unchanged for 2.x.
- **Probes** (changed): `isaaclab3_newton` requires `isaaclab >= 3` **and**
  importable `newton` / `mujoco-warp` / `warp`. `isaaclab3_physx` requires
  only `isaaclab >= 3`. Failures raise `BackendSelectionError{error, hint}`.
- **`BackendInfo`** (unchanged fields in this slice): both entries fill
  `options_schema` (Newton: solver/substeps; PhysX: empty), `scopes` only where
  `max_envs` genuinely narrows, `camera_enable_required` (True for PhysX;
  Newton decided by P3 probe), `scene_provider_consumer=True`.
- **`BACKEND_CAPABILITIES`** (new per backend): a bit set matching that
  backend's declared traits. Do **not** use `BackendCapabilities.full()`.

## Interactions and data flow

```text
yaml/run spec
  -> registry.select_backend(name)            # alias/version/newton probe
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
   boundaries; importing either backend package loads no engine.
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

- **Public API**: `known_backends()` / `CANONICAL_BACKENDS` gain a name and
  lose the `isaaclab3` canonical spelling; `isaaclab3` remains a config alias.
  `BackendInfo.name` values change accordingly. No other public type changes.
- **Config/schema**: `backend: isaaclab3` keeps working (alias). `isaaclab3`
  and `isaaclab3_physx` share the `backend_options` namespace; Newton uses
  `backend_options.isaaclab3_newton`. Unknown cross-namespace keys still fail.
- **Dependency/version lane**: new strict conda lanes pinned to the dog stack
  (isaaclab 3.0.0rc1, isaacsim 6.1.0.0, newton 1.6.1, mujoco-warp 3.12,
  warp-lang 1.17, python 3.12). `OMNI_KIT_ACCEPT_EULA=YES` is supplied only to
  these lanes.
- **Migration**: one docs pass (`docs/testing.md`, backend README, contracts
  that name `isaaclab3` as a stub). Tests that assert the empty il3 trait set
  or the old canonical name are updated in the same change.
- **Rollback**: purely additive. Removing the two backend packages, their
  lanes and registry entries restores the previous state; `isaaclab3` stays a
  valid alias in both directions, so user configs are never stranded.

## Tests

Lowest layers first:

- `tests/unit/backends/test_isaaclab3_traits.py` — both trait tables, trait
  rules, `TraitScope` validity.
- `tests/unit/backends/test_isaaclab3_registry.py` — aliases, probes,
  `known_backends`, `build_name`, backend-option namespaces.
- `tests/unit/backends/test_isaaclab3_profile.py` — profile values and the
  shared code's reliance on the contract (no engine branch).
- `tests/contract/test_registry_probe_isaaclab3.py` — selection/error messages.
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

- `ci/compat.toml`: `[environments.isaaclab3_physx]`,
  `[environments.isaaclab3_newton]`, `[lanes.isaaclab3_physx]`,
  `[lanes.isaaclab3_newton]`, both `strict = true`,
  `pytest_args = ["--strict-test-packs"]` and join `[nightly]` in P4; P0
  declares the lanes as stack probes without test packs and keeps them out of
  the nightly list.
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

- **Q1 confirmed**: symmetric canonical names `isaaclab3_physx` and
  `isaaclab3_newton`; `isaaclab3` becomes an alias to `isaaclab3_physx`.
  The P0 lanes use the final names; the registry rename lands in P1.
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
- no `unigym/` product-code change, so the registry keeps the current
  `isaaclab3` canonical name until P1;
- no `--strict-test-packs` (no test packs are consumed yet) and no `[nightly]`
  entry / GitHub job (no self-hosted runner and environment yet).

These are deliberate P0 boundaries, not silent fallbacks: the lane READMEs and
`docs/testing.md` state them, and P1 starts by closing the registry rename and
the lane provisioning (pytest in the lane environment) before any capability is
declared.

## P0 probe checklist (completed on the reference lane)

1. Registry probe on dog: `isaaclab_version() == (3, 0)`;
   `select_backend("isaaclab3")` passes.
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
