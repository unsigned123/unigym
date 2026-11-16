# State-write contract

Status: implementation contract for the optional batched state-write face.

The state-write face is **world-frame**, **wxyz**, **normalized by core**, and
**name-addressed**.  It writes simulator state directly; it is not an action
path and it does not advance policy time.

## 0. Surface

```python
run.set_root_state(
    env_ids=None, *, name, pos_w, quat_wxyz, lin_vel_w, ang_vel_w
) -> torch.Tensor
run.set_joint_state(
    env_ids=None, *, articulation, joint_names, position, velocity=None
) -> torch.Tensor
run.settle(env_ids=None) -> torch.Tensor
```

The state-write face is an **optional registry trait**:
`unigym.backends.registry.Trait.STATE_WRITE`, projected from
`BackendCapabilities.state_write`.  The required `EnvBackend` validator does not
include these methods, so a backend (or fake) that only collects remains valid.
The three methods are ordinary backend methods, not a second Protocol.
`Run` raises `NotImplementedError` naming the backend and the trait when the
live backend does not declare `STATE_WRITE`.

`env_ids` is a sequence or tensor of environment ids.  `None` means every env.
The input order is preserved (a repeated id is a repeated write).  Every
method returns the normalized `int64` id tensor.  Ids outside
`[0, num_envs)` fail fast.

## 1. Frame contract

- All input tensors are **world frame**.
- Quaternions are **`(w, x, y, z)`**; core normalizes them before dispatch.
  A zero-norm (`<= 1e-12`) quaternion is a teaching error.
- `lin_vel_w` is the world linear velocity **at the root link/body origin**,
  not at the center of mass.
- `ang_vel_w` is the world angular velocity.
- Inputs are `(K, 3)` / `(K, 4)` floating tensors on the env device, where
  `K = len(env_ids)`.  Non-finite values and wrong shapes/dtypes/devices fail
  fast.

## 2. `set_root_state`

- `name` addresses either a declared articulation with a **free root** or a
  **free rigid prop**.
- A fixed/welded articulation root is an honest error; use
  `set_joint_state` for its joints.
- Static or unbound props have no writable root and are an honest error.
- Free roots write `qpos[root:root+3] = pos_w`,
  `qpos[root+3:root+7] = quat_wxyz`, `qvel[dof:dof+3] = lin_vel_w`,
  `qvel[dof+3:dof+6] = R.T @ ang_vel_w` on MuJoCo.
- IsaacLab writes the world primitives directly through the asset's
  **link-origin** pose/velocity writers; the CoM-default velocity writer is
  never used.
- The call never settles implicitly.

## 3. `set_joint_state`

- `articulation` is a declared articulation name.
- `joint_names` is an explicit, duplicate-free sequence; it fixes the
  `position`/`velocity` columns.  Resolution is **by joint name**, never by
  simulator order.
- `position` is `(K, J)` absolute joint coordinate (rad; metres for slides).
- `velocity` is `(K, J)` or `None`; `None` leaves velocity unchanged.
- Unknown names are an honest error.  MuJoCo rejects free/ball joints
  explicitly; only 1-DOF joints are accepted in this version.
- The call never settles implicitly.

## 4. `settle`

`settle` recomputes derived state after writes and refreshes the observation
cache.  It must be called before reading observations, contacts, or camera
state that depends on the written pose.

Ordering (both backends):

1. MuJoCo: `mj_forward` for each touched `MjData` (per-env loop; no
   `mjBatch`).  IsaacLab: `scene.write_data_to_sim()` then `sim.forward()`.
2. Re-pose camera rigs through the existing reset hook (`on_reset(env_ids)`).
3. Refresh `obs_cache`.

`settle` does not change the episode clock, episode id, or action targets.
Without a settle, the raw engine state is written but derived quantities and
cached observations may be stale.

## 5. Lifecycle

A later `reset_envs` / `reset_all` overwrites written state with the declared
spawn pose, joint defaults and zero velocities.  State writes are therefore
episode state, not persistent configuration.

## 6. Backend mapping

| Concept | MuJoCo (`MujocoEnv`) | IsaacLab 2 (`IsaaclabEnv`) |
| --- | --- | --- |
| articulation root | articulation's freejoint `qpos`/`qvel` | `write_root_link_pose_to_sim` + `write_root_link_velocity_to_sim` |
| free rigid prop | prop backend `free_addresses(name)` | handler asset's `write_root_link_pose_to_sim` / `write_root_link_velocity_to_sim` |
| angular velocity | world -> body-local `R.T @ ang_vel_w` in qvel | world input, passed through the existing wxyz -> PhysX boundary |
| joint write | `jadr_art` / `jdr_art` qpos/qvel rows | `write_joint_state_to_sim(position, velocity, joint_ids, env_ids)` |
| settle | `mj_forward` + obs refresh | `write_data_to_sim` + `sim.forward` + obs refresh |

The MuJoCo freejoint convention is the one measured in the engine:
`qvel[0:3]` is world-frame linear velocity and `qvel[3:6]` is **body-frame**
angular velocity.  Observation and director helpers convert at the boundary;
world angular velocity is `R @ qvel[3:6]`.
