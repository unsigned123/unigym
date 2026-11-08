# `backends/shared/mjfamily` — MuJoCo family contract

Status: **G6** — the whole remaining MuJoCo family lives here:
`scene/`, `props/`, `director/`, `cameras/` (helpers) and `model/`.

## Members

- `unigym/backends/mujoco/` (CPU MuJoCo)
- `unigym/backends/mjwarp/` (MuJoCo Warp, CPU `MjData` mirror)

Both backends speak the same `mujoco.MjModel` / `mujoco.MjData` model
contract; code that is genuinely shared between them belongs here instead
of one backend importing the other.

## Rules

1. This package **may import `mujoco`** and `mujoco_warp` library modules.
2. It must **never import any `backends/<name>/env.py`**. Family code is a
   library, not an env orchestration; env state is passed in as an argument.
3. Only `unigym/backends/mujoco/**` and `unigym/backends/mjwarp/**` may
   import this package. Neutral layers and the `isaaclab2` / `isaaclab3`
   backends must not.
4. The two backend packages remain independent: `mujoco` must not import
   `mjwarp` (and vice versa). Cross-backend edges are ratcheted in
   `tests/meta/fixtures/backend_isolation_baseline.json` and can only
   shrink.

## `model/`

- `spec_walk.py` — pre-compile `MjSpec` walkers (`build_from_spec`).
- `model_walk.py` — post-compile `MjModel` readers (`describe_model`).
- `provider.py` — `MujocoModelProvider` (backend label overridable).
- `ir_access.py` — env `ModelIR` / name→id address helpers.

`mujoco/model/` re-exports the provider + address helpers; `mjwarp/model.py`
re-exports the provider as `MjwarpModelProvider` (`backend="mjwarp"`).

## `scene/`

`scene_render` / `scene_engine` / `scene_artifacts` / `scene_spawn` /
`mjcf_io` / `asset_render` / `placement_query` — the MjSpec composer,
engine registration, artifact serialization, spawn rewrites and asset
renderer.  `mujoco/<name>.py` keeps a thin compatibility re-export.

## `props/`

`props` / `props_articulation` / `props_usd` / `reset_face` / `multienv`
— the prop body builders, reset primitives and multi-env guards.

## `director/`

`ops.py` — `MujocoDirectorOps` (the mjwarp director subclasses it) and
`gripper.py` — the shared mocap/weld spec builders.

## `cameras/`

`helpers.py` — the pure camera helpers both backends need (`dc_` prefix,
Isaac->MuJoCo quaternion bridge, fov/mount resolution, ``<camera>`` MjSpec
injection).  The concrete `mujoco.Renderer` rig stays in
`backends/mujoco/cameras.py`; mjwarp resolves it through the neutral
registry face for its N=1 CPU-mirror path.

## Compatibility

Every moved `mujoco/<old>.py` remains as a thin re-export module
(`__getattr__` -> shared implementation), so historical imports in tests,
registry factories and docs keep working without duplicating code.
