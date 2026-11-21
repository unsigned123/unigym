# Runtime query contract

Status: public, on-demand.  Requires `Trait.RUNTIME_QUERY`.

Runtime query is a stable, name-addressed read face for model facts,
live state, scratch worlds, geometry and Jacobians.  It is **not** an
observation channel:

- it never writes `env.obs`;
- it never publishes or records automatically;
- it never mutates live `MjData` (scratch uses a copy);
- it runs on the engine thread at a step boundary and is not thread-safe with
  `step`/`reset` (the backend adds no lock).

`obs` is unchanged.  Query is opt-in through the backend trait and the
`BackendInfo.query_backend` lazy factory.

## 0. Scope (P0A + P0B-1)

In scope:

- `run.query` (trait-gated) and the facade protocols;
- model facts from the published `ModelIR`;
- live body/site/COM/root state;
- per-env scratch copy with root/joint writes and `forward`;
- geometry distance (`mj_geomDistance`) and AABB;
- body/site/COM/world-point Jacobians.

Out of scope in P0A:

- contact manifolds, contact Jacobians, wrench-to-joint (P1);
- actuator runtime facts (P1);
- IsaacLab query implementations (P0B-2);
- any pack-specific consumer;
- deleting `NativeHandle`, `ModelIR`, `run.model`, `run.live`,
  `probe_candidate` or `ResetPrimitives.probe`.

`env.query` is an optional backend face; `EnvBackend` gains no required
member.  P0A ships the MuJoCo implementation and the neutral contract;
P0B-1 adds the mjWarp implementation over the same MuJoCo-family math and the
CPU `MjData` mirror.

## 1. Entry points

```python
run.query      # RunInstance.query -> Query (trait-gated, lazily constructed)
env.query      # optional backend face (MuJoCo implements it)
```

`RunInstance.query`:

- reads the registry truth (`backend_info(backend).traits`), never `hasattr`;
- reuses ``env.query`` as the single live instance when the backend exposes it,
  otherwise falls back to ``BackendInfo.query_backend``;
- caches the object for the live env instance;
- raises a teaching `NotImplementedError` when the backend does not declare
  `Trait.RUNTIME_QUERY`;
- raises `QueryStaleError` after the run is closed;
- is closed with the run.

MuJoCo's `env.query` caches one `MujocoQuery`; mjWarp's `env.query` caches one
`MjwarpQuery`.  In both backends `run.query` and `env.query` return the same
object.

## 2. Capabilities

```python
Support = Literal["native", "partial", "synthetic", "unsupported"]
```

`QueryCapabilities` groups per-operation `Feature(support, detail)` values by
subface (`model`, `state`, `scratch`, `jacobian`, `geometry`, `contacts`,
`actuators`).  `supports("<group>.<operation>")` returns the level;
`require(*ops)` raises `QueryCapabilityError` naming the backend, operation and
support for the first unsupported op.

Capabilities are part of the query contract, not the env capability bits.
There are no per-operation traits and no query bits in `BackendCapabilities`.

MuJoCo P0A declares:

| Group | Support |
|---|---|
| `model.*` | native |
| `state.*` | native |
| `scratch.*` | native (`copy`, `live_untouched`) |
| `jacobian.*` | native; `com` is partial (subtree COM has translation only) |
| `geometry.*` | native |
| `contacts.*` | unsupported (P1) |
| `actuators.*` | unsupported (P1) |

mjWarp P0B-1 reads the CPU `MjData` mirror, so only the model facts and live
state are native; scratch and the CPU math are labelled honestly:

| Group | Support |
|---|---|
| `model.*` | native (CPU mirror of the compiled model) |
| `state.*` | native (CPU mirror read) |
| `scratch.*` | synthetic (CPU mirror copy, not a device clone) |
| `jacobian.*` | synthetic (CPU MuJoCo math on the mirror); `com` is partial |
| `geometry.*` | synthetic (CPU MuJoCo math on the mirror) |
| `contacts.*` | unsupported (P1) |
| `actuators.*` | unsupported (P1) |

The contacts/actuators subfaces exist; every call raises a typed capability
error until P1.

## 3. Types

All names are backend-neutral; no raw engine id appears in a public signature.
Responses hold detached CPU torch copies.

- `EntityRef(kind, name, articulation=None)`: `kind` is
  `body | site | geom`.
- `ComRef(articulation, body=None, subtree=False)`: `body=None` selects the
  articulation base body; `subtree=True` reads subtree COM.
- `PointRef(entity, point_w)`: a world point anchored to a **body**.
- `Pose(env_ids, pos_w, quat_wxyz)`: world pose; `quat_wxyz` is `wxyz`.
- `DistanceResult(env_ids, signed_distance_m, closest_a_w, closest_b_w,
  normal_w, capped, mesh_exact)`: `capped` is true when the requested
  `max_distance_m` horizon was reached; `mesh_exact` is false in P0A.
- `AABB(env_ids, min_w, max_w)`.
- `DofLayout(articulation, base_dofs, base_names, joint_names, nv, order)`:
  `order` has one label per Jacobian column and equals
  `base_dofs + joint_names`.  A free base contributes six `base_dofs`
  labels (`<base>.px` ... `<base>.rz`) and one `base_names` entry; a fixed
  base contributes none.
- `Jacobian(env_ids, transl, rot, layout, frame, reference)`: `transl` and
  `rot` are `(E, 3, nv)`.

`env=None` selects every env for state/geometry reads.  `env` may be an int, a
sequence of ints or a tensor; output `env_ids` preserves the requested order.
An empty sequence or empty int64 tensor selects zero envs and returns
zero-row responses.  Scratch is one env: `scratch(None)` selects env 0.

## 4. Subfaces

```python
model.ir(articulation=None) -> ModelIR
model.body/geom/site/joint/actuator(name, *, articulation=None)
state.body_pose/site_pose(name, *, articulation=None, env=None) -> Pose
state.com_pose(ref, *, frame="world", env=None) -> Pose
state.root_pose(articulation, *, env=None) -> Pose
state.joint_position/joint_velocity(articulation, *, joint_names=None, env=None)
kinematics.jacobian(target, *, articulation=None, env=None,
                    frame="world", reference="body_origin", rotational=True)
geometry.distance(a, b, *, max_distance_m=1.0, env=None, scratch=None)
geometry.aabb(entity, *, env=None)
```

P0A supports `frame="world"` and `reference="body_origin"`.  `body_origin`
means the target's own origin: the body frame, the site frame, the subtree
COM, or the supplied world point.  Other values raise a capability error
instead of silently changing the frame.

A subtree COM has a defined translational Jacobian but no native rotational
Jacobian.  `ComRef(subtree=True)` with `rotational=True` raises
`QueryCapabilityError`; pass `rotational=False` to read the translational
block.  A body COM (`ComRef(subtree=False)`) keeps both blocks.

`EntityRef.articulation`, `PointRef.entity.articulation` and
`ComRef.articulation` select the articulation before the `articulation=`
keyword.  If both are present and disagree, the query fails with
`QueryReferenceError`; if a neutral name exists in several articulations and
neither selector is given, the query is ambiguous and also fails.  The same
resolved articulation drives both the Jacobian columns/layout and the target
lookup.

`state.com_pose` returns a world position; its `quat_wxyz` is the identity
quaternion because a center of mass is a point, not an oriented frame.

The Jacobian layout covers the articulation's compiled hinge/slide joints
plus its free root.  A compiled ball joint, a non-root free joint, or a
profile-filtered joint outside the neutral joint table is not silently
dropped or mis-sized: it raises `QueryCapabilityError` naming the
articulation, joint and compiled joint type.

`state.root_pose`, `scratch.set_root_state` and the Jacobian require a
resolved compiled free root.  If an articulation declares `root="free"` but
its base body / free joint cannot be mapped (for example
`profile.base_body=None`), the query raises `QueryCapabilityError` with a
"free root unresolved" message instead of reporting a fixed root or a
shortened layout.

Only 1-DOF hinge/slide joints are exposed by `joint_position`/
`joint_velocity`; another joint type raises a capability error naming the
joint.

## 5. Scratch

```python
with run.query.scratch(env_id) as scratch:
    scratch.set_root_state("robot", pos_w=..., quat_wxyz=...)
    scratch.set_joint_state("robot", ("joint",), position=...)
    scratch.forward()
    pose = scratch.body_pose("base")
    jac = scratch.jacobian(EntityRef("body", "base"))
    distance = scratch.geometry().distance(a, b)
```

MuJoCo semantics:

- `mj_copyData` from the live per-env `MjData`, then local writes;
- `forward()` runs `mj_forward` on the copy only;
- one active scratch per env and per query object; a second concurrent
  scratch for the same env fails loudly, while different envs may hold
  concurrent scratches;
- a subface captured from a scratch (`scratch.geometry()` / `scratch.jacobian()`)
  becomes stale when the scratch context closes and raises `QueryStaleError`;
- closing the context releases the copy; later reads raise
  `QueryStaleError`;
- the live `MjData` is never mutated by a scratch read or write.

### Geometry notes

- `distance` takes exactly one data source: live `env` (or `env=None` for all
  envs) or a `scratch` object, never both.
- `AABB` is conservative: primitive extents for box/sphere/capsule/cylinder/
  ellipsoid and the bounding radius for a mesh.  A plane or heightfield has no
  finite AABB and raises `QueryCapabilityError`.
- When the requested `max_distance_m` horizon is reached, `capped` is true and
  the returned closest points/normal are not meaningful for the true surface
  pair (MuJoCo returns the horizon with zero `fromto`).

## 6. mjWarp CPU mirror semantics

mjWarp steps the `mujoco_warp` device `Data` and refreshes one CPU `MjData`
mirror per world.  The shared MuJoCo-family query reads only that mirror:

- model facts come from the compiled `ModelIR` / `model_set`;
- live state is a read of the mirror (`env.mj_datas[e]`), never a device read;
- scratch is a private `mj_copyData` of the mirror plus `mj_forward`; it is a
  CPU copy, not a device clone;
- geometry (`mj_geomDistance`) and Jacobians run as CPU MuJoCo math on the
  mirror or scratch copy;
- the live mirror and the warp device `Data` are never mutated by a query;
- there is no implicit device-to-mirror sync and no lock.  The caller must
  query on the engine thread at a step boundary, after the backend has
  synchronized the mirrors.  A mid-step query may read stale mirror state; the
  contract does not repair it.

This is why the mjWarp capability object labels `scratch.*`, `jacobian.*` and
`geometry.*` as `synthetic` while `model.*` and `state.*` stay `native`.

## 7. Failures

| Case | Error |
|---|---|
| unsupported operation | `QueryCapabilityError` naming backend/op/support |
| unknown/ambiguous entity | `QueryReferenceError` with available names |
| bad shape/dtype/value | `QueryShapeError` (also a `ValueError`) |
| read after run/env/query close | `QueryStaleError` |
| concurrent step/reset | caller contract violation (engine-thread only) |
| mid-step / unsynchronized mjWarp mirror | caller contract violation; no silent device sync |

Failures are loud.  There is no hidden fallback, default substitute or silent
degradation, and query results are never published as observations.
