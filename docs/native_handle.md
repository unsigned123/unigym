# NativeHandle escape hatch

Status: external-pack-only, unstable.

`NativeHandle` is a typed wrapper over backend-native objects.  It is an
explicit escape hatch, **not** a neutral API and **not** a compatibility
promise.  If an operation is not on the neutral `Run`/`EnvBackend` surface,
core does not support it merely because a handle can reach the native object.

Common model-parameter edits are **not** a reason to use this handle: they
live behind `Trait.MODEL_EDIT` as `run.model_edit()` (see
`contracts/model_edit_contract.md`).

## 0. Who may use it

Only external packs with a declared native dependency.  Core libraries, tasks,
examples, and in-repo packs must not call `env.native(...)`; backend internals
keep using their private native objects.  External packs own the causal
effects and cleanup of anything they mutate.

The existing public backend properties (`env.mj_model`, `env.mj_data`,
`env.mj_datas`, `env.sim`, `env.scene`, ...) are backend-internal aliases.
External code must use `env.native(backend)` (or `run.native(backend)`), never
those aliases.

## 1. Surface

```python
run.native("mujoco")    -> NativeHandle
run.native("mjwarp")    -> NativeHandle
run.native("isaaclab2") -> NativeHandle
```

The direct form is `env.native(backend)`.  The requested backend name must
match the live backend exactly; a mismatch is a `ValueError`.  The run facade
also gates on `Trait.NATIVE_HANDLE`, raising `NotImplementedError` with the
backend name and trait when the backend declares it absent (`isaaclab3`).

`NativeHandle` is a frozen dataclass with:

- `backend: str`
- `objects: Mapping[str, object]`
- `unstable: bool = True`
- `version: int = 1`

and `handle[name]`, `handle.name`, `handle.require(name)`, `handle.keys()`.
An unknown name raises an error listing the available names.  The handle is
not a `Mapping` implementation and has no JSON contract.

Handles return **identity references**: no copy, no settle, no state write, no
lazy engine import.  Names are stable only within `version = 1`; object types
are backend-version owned and may change without notice.

## 2. Version-1 contents

### mujoco

```python
{
    "model": env.mj_model,   # mujoco.MjModel
    "data": env.mj_data,     # mujoco.MjData (single-env view)
    "datas": env.mj_datas,   # per-env MjData handles
}
```

### mjwarp

```python
{
    "model": env.mj_model,
    "data": env.mj_data,
    "datas": env.mj_datas,
    "warp_model": env._wm,
    "warp_data": env._wdata,
}
```

### isaaclab2

```python
{
    "sim": env.sim,
    "scene": env.scene,
    "robot": env.robot,      # None for partitioned multi-articulation runs
}
```

## 3. External-pack pseudo-code

```python
# External pack only; never copy this pattern into core, tasks, examples,
# or in-repo packs.
handle = run.native("mujoco")
model = handle.require("model")
data = handle["data"]
```

The pack owns any native mutation, its ordering relative to steps, and the
resulting cleanup.  A native handle does not imply core support for the
operation.
