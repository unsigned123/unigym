# Runtime query contract

Status: public, on-demand.  Requires `Trait.RUNTIME_QUERY`.  Scope covers the
P0/P1 face plus P2-0 (object contract), P2-A (object/root/COM state), P2-B
(object Jacobian), P2-C (cleanup/deprecation only) and P2-D (gap
classification); P2-E adds the executable capability-matrix release gate.

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

## 0. Scope (P0A + P0B-1 + P1A-2 + P1B-1 + P1B-2 + P1C-1 + P1C-2 + P1C-3 + P2-0 + P2-A + P2-B + P2-C + P2-D + P2-E)

In scope:

- `run.query` (trait-gated) and the facade protocols;
- model facts from the published `ModelIR`;
- public MuJoCo-family geom contact parameters (`model.geom_contact`);
- live body/site/COM/root state;
- per-env scratch copy with root/joint writes and `forward` (MuJoCo family) or
  a per-env transaction scratch (IsaacLab 3);
- geometry distance (`mj_geomDistance`) and AABB;
- body/site/COM/world-point Jacobians;
- group-based contact manifolds for the MuJoCo family (P1A-1);
- contact constraint-row Jacobians and wrench-to-joint for the MuJoCo family
  (P1A-2);
- MuJoCo-family actuator facts and per-env runtime channels (P1B-1);
- IsaacLab static actuator facts (P1B-2);
- IsaacLab 3 body/body-COM Jacobians and composed site/world-point Jacobians
  (P1C-1);
- IsaacLab 3 transaction scratch with root/joint writes, `forward` and
  body/site/COM/Jacobian reads (P1C-2);
- IsaacLab 3 PhysX filtered pairwise contact manifolds for groups that declare
  `manifold: true` (P1C-3);
- the P2-0 object contract skeleton: `ObjectRef`, `RigidState`, `ComState`,
  `DofLayout` ownership, the new subface signatures, and the name-only
  free-prop resolver;
- P2-A MuJoCo-family object/root/COM state and world-frame velocity reads, with
  the four new state operations declared native on MuJoCo/mjWarp and typed
  unsupported on IsaacLab;
- P2-B the MuJoCo-family six-column object Jacobian: `jacobian.object` is
  native on MuJoCo and synthetic on mjWarp, with an object-owned `DofLayout`;
- P2-C cleanup and deprecation only: `run.live` and
  `ResetPrimitives.probe_candidate` warn and delegate to `run.live_for(name)` /
  `probe`, and `ContactManager.read_detail` stays until its callers migrate.
  P2 removes no public operation;
- P2-D the backend-gap classification in §"P2-D known gaps and classification":
  no gap is promoted to a support claim;
- P2-E the exact five-backend capability matrix gate
  (`tests/unit/query/test_capabilities_matrix.py`).

Out of scope:

- IsaacLab actuator runtime state;
- writing `ctrl`, gains or limits, and PD target conversion;
- actuator group semantics beyond the published row and its declared joint
  names;
- IsaacLab 2.x and IsaacLab 3 Newton contact manifolds, and every IsaacLab
  contact Jacobian/wrench-to-joint read;
- IsaacLab scratch geometry/contact reads;
- an IsaacLab 2.x scratch (IL2 has no transaction card);
- an IsaacLab subtree COM Jacobian (the engine exposes body COM only);
- all-contact world scans and an inactive/near-contact buffer;
- any pack-specific consumer;
- deleting `NativeHandle`, `ModelIR`, `run.model`, `run.live`,
  `probe_candidate`, `ResetPrimitives.probe` or `ContactManager.read_detail`
  (P2-C is warning-only deprecation, not removal).

`env.query` is an optional backend face; `EnvBackend` gains no required
member.  P0A ships the MuJoCo implementation and the neutral contract;
P0B-1 adds the mjWarp implementation over the same MuJoCo-family math and the
CPU `MjData` mirror.  P0B-2 adds the IsaacLab 2.x and IsaacLab 3.x
(`isaaclab3_physx` / `isaaclab3_newton`) model+pose subset documented in §8;
scratch geometry stays unsupported and the P0B-2 contact face stays
unsupported.  IsaacLab actuator runtime state stays unsupported.  P1B-2 adds
the IsaacLab static actuation facts subset; P1C-1 adds the IsaacLab 3 Jacobian
subset; P1C-2 adds the IsaacLab 3 transaction scratch; P1C-3 adds the PhysX
filtered pairwise contact manifold (the IsaacLab 2.x / Newton contact manifold
stays unsupported).  P2-0 adds the object contract skeleton; P2-A adds the
MuJoCo-family object/root/COM state; P2-B adds the six-column object Jacobian;
P2-C deprecates (but does not remove) the legacy `run.live`/`probe_candidate`
faces; P2-D classifies the remaining gaps; P2-E adds the executable matrix gate
and changes no capability value.

## 1. Entry points

```python
run.query      # RunInstance.query -> Query (trait-gated, lazily constructed)
env.query      # optional backend face (MuJoCo/mjWarp/IsaacLab implement it)
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
`MjwarpQuery`; IsaacLab 2.x caches one `Isaaclab2Query`; IsaacLab 3.x caches one
`Isaaclab3Query` that reports its concrete `isaaclab3_physx` or
`isaaclab3_newton` identity.  In every backend `run.query` and `env.query`
return the same object, and closing the env closes that handle
(`QueryStaleError` on later use).

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

MuJoCo declares (P0A + P1A-2):

| Group | Support |
|---|---|
| `model.*` | native |
| `model.geom_contact` | native (compiled `ModelIR` `mj.geom_contact` row) |
| `state.*` | native |
| `state.root_state`, `state.com_state`, `state.object_state`, `state.object_com_state` | native (free-joint `qpos`/`qvel`, `xipos`/`subtree_com`, `mj_jacBodyCom`/`mj_jacSubtreeCom`) |
| `scratch.*` | native (`copy`, `live_untouched`) |
| `jacobian.*` | native; `com` is partial (subtree COM has translation only); `object` is the six-column free-prop Jacobian |
| `geometry.*` | native |
| `contacts.manifold` | native (group-based active contacts) |
| `contacts.jacobian`, `contacts.wrench_to_joint` | native (dense contact constraint rows; effective rows follow `model.opt.cone`) |
| `actuators.facts` | native (compiled `ModelIR` actuator row) |
| `actuators.state` | native (live `data.ctrl`/`actuator_force`/`actuator_length`/`actuator_velocity`) |

mjWarp P0B-1/P1A-2 reads the CPU `MjData` mirror, so only the model facts and
live state are native; scratch, the CPU math and the contact mirror are
labelled honestly:

| Group | Support |
|---|---|
| `model.*` | native (CPU mirror of the compiled model) |
| `model.geom_contact` | native (CPU mirror of the compiled `mj.geom_contact` row; not a warp device read) |
| `state.*` | native (CPU mirror read) |
| `state.root_state`, `state.com_state`, `state.object_state`, `state.object_com_state` | native (same free-joint/COM math on the CPU mirror, not a warp device read) |
| `scratch.*` | synthetic (CPU mirror copy, not a device clone) |
| `jacobian.*` | synthetic (CPU MuJoCo math on the mirror); `com` is partial; `object` is the six-column free-prop Jacobian |
| `geometry.*` | synthetic (CPU MuJoCo math on the mirror) |
| `contacts.manifold` | synthetic (CPU contact mirror, not a warp device read) |
| `contacts.jacobian`, `contacts.wrench_to_joint` | synthetic (dense CPU mirror contact rows; effective rows follow `model.opt.cone`) |
| `actuators.facts` | synthetic (compiled CPU mirror model row) |
| `actuators.state` | synthetic (CPU mirror runtime channels; not a warp device read) |

IsaacLab 2.x / 3.x P0B-2 share the model+pose subset; P1C-1 adds the IsaacLab 3
Jacobian subset, P1C-2 the IsaacLab 3 transaction scratch and P1C-3 the PhysX
filtered pairwise contact manifold (the IsaacLab 2.x / Newton manifold stays
unsupported):

| Group | Support |
|---|---|
| `model.ir/body/joint/actuator` | native (published `ModelIR`) |
| `model.geom/site` | unsupported (the IsaacLab `ModelIR` carries no geom/site facts; none are synthesized) |
| `model.geom_contact` | unsupported (the IsaacLab `ModelIR` has no `mj` extension; MuJoCo-family contact parameters are not synthesized) |
| `state.body_pose/root_pose/joint_position/joint_velocity` | native (live articulation buffers) |
| `state.site_pose` | partial (robot-profile sites only: body + local offset; no USD prim/FrameTransformer) |
| `state.com_pose` | partial (body COM position; identity `quat_wxyz` by contract; subtree unsupported) |
| `state.root_state`, `state.com_state`, `state.object_state`, `state.object_com_state` | unsupported (IsaacLab exposes live articulation pose/joint buffers only; no typed rigid-object/root/COM velocity read face in P2) |
| `scratch.*` (IsaacLab 2.x) | unsupported (no transaction card) |
| `scratch.copy/set_root_state/set_joint_state/forward/body_pose/site_pose/com_pose/jacobian` (IsaacLab 3.x) | synthetic (exclusive per-env transaction over the live state; no snapshot copy) |
| `scratch.live_untouched` (IsaacLab 3.x) | partial (captured state restored on close; not bit-exact) |
| `scratch.geometry` (IsaacLab 3.x) | unsupported (no geometry/contact copy in a transaction) |
| `jacobian.*` (IsaacLab 2.x) | unsupported (no native Jacobian API) |
| `jacobian.body` (IsaacLab 3.x) | native (engine body link Jacobian at the body origin) |
| `jacobian.com` (IsaacLab 3.x) | partial (native body COM Jacobian; a subtree COM Jacobian is unsupported) |
| `jacobian.site` (IsaacLab 3.x) | synthetic (body link Jacobian composed with the robot-profile site local offset) |
| `jacobian.point` (IsaacLab 3.x) | synthetic (body link Jacobian composed with the world-point offset) |
| `jacobian.object` (IsaacLab 2.x / 3.x) | unsupported (no typed rigid-object read face; direct calls raise `QueryCapabilityError`) |
| `geometry.*` | unsupported (P1) |
| `contacts.manifold` (IsaacLab 3 PhysX) | partial (filtered pairwise manifolds for groups declared `manifold: true`; no contact-normal direction, geom ids, per-point tangent frame or friction) |
| `contacts.manifold` (IsaacLab 2.x / IsaacLab 3 Newton) | unsupported |
| `contacts.jacobian`, `contacts.wrench_to_joint` | unsupported (no IsaacLab contact constraint rows) |
| `actuators.facts` | partial (static `ModelIR` actuator-group facts plus published USD joint names; `slot=None`, no ctrlrange/gear/gains/limits) |
| `actuators.state` | unsupported (an IsaacLab group has no single scalar ctrl/force/length/velocity row) |

The contact/actuator subfaces exist on every backend; an operation the backend
does not implement raises a typed capability error instead of returning a
default.

### P2 object capability catalog

P2-0 added the names `state.root_state`, `state.com_state`,
`state.object_state`, `state.object_com_state` and `jacobian.object` to the
capability catalog.  `StateCaps` and `JacobianCaps` give them the safe default
`Feature("unsupported", "not declared")`.  P2-A declares the four state
operations explicitly: native on MuJoCo and mjWarp, and unsupported with a
teaching detail on IsaacLab 2 / IsaacLab 3.  P2-B declares `jacobian.object`
native on MuJoCo and synthetic on mjWarp.  IsaacLab 2 inherits the shared
unsupported teaching detail; IsaacLab 3 replaces the Jacobian group without
declaring `object`, so it keeps the safe `Feature("unsupported", "not declared")`
default.  On every IsaacLab identity the direct `object_state`,
`object_com_state` and `object_jacobian` methods raise `QueryCapabilityError`;
no IsaacLab object read or Jacobian is claimed.  Newton has no object lane and
never inherits PhysX evidence.

The five-backend matrix above is asserted operation-by-operation by
`tests/unit/query/test_capabilities_matrix.py`: support levels, scratch
mode/isolation, the exact `state`/`jacobian` member sets and the absence of
per-operation traits.  P2-E changes no capability value.

### P2-D known gaps and classification

This section classifies the remaining runtime-query backend gaps.  It records
current behavior only: P2-D adds no implementation, promotes no operation and
changes no capability value above.  A backend operation is a support claim only
when a validated backend lane backs it; items marked "follow-up" or "no lane"
stay at the declared `unsupported` or `partial` level.  The classification does
not turn any gap into a new P2 implementation.

| Gap | Classification | Rationale |
|---|---|---|
| Sparse `efc_J` | unsupported; dense-only | `contacts.jacobian`/`wrench_to_joint` map dense `data.efc_J` only.  A sparse layout (explicit, or `mjJAC_AUTO` at `nv >= 60`) raises `QueryCapabilityError`; compressed values are never reshaped as dense rows.  Dense constraint rows are already tested; sparse support is a follow-up. |
| Mesh-exact geometry distance | unsupported/follow-up; `mesh_exact` false | `geometry.distance` uses `mj_geomDistance`, which is not a mesh-mesh exact distance.  `DistanceResult.mesh_exact` stays `false`; there is no exact-mesh support claim. |
| IL3 contact normal/tangent/friction/geom ids | partial (existing P1C-3) | PhysX filtered manifolds expose position, signed separation and a non-negative normal-force magnitude.  `normal_w`/`frame_w`, per-point friction and geom ids are `None` because the engine filtered view does not prove them.  No change in P2-D. |
| Configurable IL contact point cap | unsupported/follow-up | The PhysX manifold reserves `MANIFOLD_MAX_CONTACT_DATA_COUNT_PER_PRIM = 4` contact slots per sensor body and env as a fixed backend declaration.  There is no per-group or per-run configuration. |
| IL contact multi-env/multi-filter validation | partial/follow-up | The PhysX manifold validates the resolved filter width against the engine filter axis, rejects env ids outside the sensor slots and preserves the requested env order.  Cross-group and more general multi-filter validation remains follow-up. |
| IL2 contact manifold | unsupported; no lane | IsaacLab 2 has no contact manifold implementation and no lane that would back a support claim; the direct read raises `QueryCapabilityError`. |
| Newton runtime query completeness | unsupported; no lane | Newton shares the IsaacLab query code but has no validated real-env lane in P2, so no runtime-query completeness claim is made.  PhysX evidence is never reused for Newton. |
| Newton contact normal/separation | unsupported; no lane | The Newton filtered contact manifold is not implemented; `contacts.manifold` is declared unsupported and raises `QueryCapabilityError`. |
| IL scratch free-root real restore | partial; transaction | An IsaacLab 3 scratch writes the live state-write face and, on close, restores the captured free-root pose/velocity plus captured 1-DOF joint position/velocity and forwards again.  `set_root_state` on a fixed root raises.  It is a transaction, not a snapshot copy. |
| IL scratch isolation/bit-exact | partial; `restored`, documented | `ScratchCaps.isolation == "restored"`, not `"bit_exact"`: live buffers are touched while the transaction is open.  The MuJoCo family keeps `mode="copy"` and `isolation="bit_exact"`. |
| IL actuator runtime | unsupported; no lane | IsaacLab actuator groups expose no single scalar `ctrl`/`force`/`length`/`velocity` row, so `actuators.state` is unsupported and raises `QueryCapabilityError`; static `actuators.facts` remains `partial`. |
| Query performance/caching/batch | documented per-env loop; no cache | State, geometry, Jacobian and contact reads iterate the selected per-env data (`mj_datas` or the articulation buffers) in requested order.  Query keeps no live-value cache and calls no implicit `forward`; "batch" is the env selection vector, not a vectorized engine call. |

## 3. Types

All names are backend-neutral; no raw engine id appears in a public signature.
Responses hold detached CPU torch copies.

- `EntityRef(kind, name, articulation=None)`: `kind` is
  `body | site | geom`.
- `ComRef(articulation, body=None, subtree=False)`: `body=None` selects the
  articulation base body; `subtree=True` reads subtree COM.
- `PointRef(entity, point_w)`: a world point anchored to a **body**.
- `Pose(env_ids, pos_w, quat_wxyz)`: world pose; `quat_wxyz` is `wxyz`.
- `ObjectRef(name)`: a declared rigid free prop addressed by its public prop
  name.  There is no raw body/joint id and no `articulation` field; P2 resolves
  only rigid free props.  `name` is non-empty.
- `RigidState(env_ids, pos_w, quat_wxyz, lin_vel_w, ang_vel_w,
  frame="world", reference="body_origin")`: one rigid body/root frame-origin
  pose and world-frame twist.  `lin_vel_w` is the world linear velocity at the
  frame origin; `ang_vel_w` is the world angular velocity.  Only
  `frame="world"` and `reference="body_origin"` are representable.
- `ComState(env_ids, pos_w, lin_vel_w, frame="world")`: a center-of-mass point
  position and world linear velocity.  A COM has no orientation and no angular
  velocity; a rigid body's angular velocity is available from `RigidState`.
- `DistanceResult(env_ids, signed_distance_m, closest_a_w, closest_b_w,
  normal_w, capped, mesh_exact)`: `capped` is true when the requested
  `max_distance_m` horizon was reached; `mesh_exact` is false in P0A.
- `AABB(env_ids, min_w, max_w)`.
- `DofLayout(articulation, base_dofs, base_names, joint_names, nv, order,
  owner_kind="articulation", owner=None)`: `order` has one label per Jacobian
  column and equals `base_dofs + joint_names`.  A free base contributes six
  `base_dofs` labels (`<base>.px` ... `<base>.rz`) and one `base_names` entry;
  a fixed base contributes none.  An articulation layout keeps
  `owner_kind="articulation"`, a non-empty `articulation` and `owner=None`.
  An object layout (rigid free prop) is explicitly not an articulation:
  `articulation=None`, `owner_kind="object"`, `owner` is the prop name,
  `joint_names=()`, six `base_dofs`, one `base_names` entry, `nv=6` and
  `order == base_dofs`.
- `Jacobian(env_ids, transl, rot, layout, frame, reference)`: `transl` and
  `rot` are `(E, 3, nv)`.
- `ContactPoint(env_id, group, body1, body2, geom1, geom2, pos_w, normal_w,
  frame_w, separation_m, normal_force_n, friction_force_w, contact_id)`:
  one active contact point.  `body*`/`geom*` keep the backend contact order;
  a geom with no compiled name is `None`.  `normal_w` points from geom1 to
  geom2, and `frame_w` rows are the world contact axes (normal, tangent1,
  tangent2).  The forces act on the **geom2 body** (`mj_contactForce`
  convention); `separation_m` is signed (positive = separated).
  `contact_id` indexes the env's current contact list and is not a stable
  cross-step identifier.
- `ContactManifold(env_ids, group, points)`: the selected env ids and the
  group's points in backend contact order, env-major.  No active contact is an
  empty tuple, not an error.
- `ContactJacobian(env_id, frame, rows, layout)`: one contact's constraint-row
  block.  `frame` is `"contact"`; `rows` is `(rows, layout.nv)` and holds the
  contact's effective constraint Jacobian in the articulation's DOF order.
  Rows are MuJoCo constraint rows, not a per-row axis list: elliptic contacts
  with `condim >= 3` have rows 0..2 equal to `frame_w`'s normal/tangent axes,
  pyramidal rows are friction-pyramid face normals (no pure normal row for
  `condim=3`), and `condim` 4/6 torsion/rolling rows are outside `frame_w`.
  The tensor is a detached CPU float64 copy.
- `ActuatorFacts(name, target, kind, slot, slot_kind, ctrlrange, gear,
  gainprm, biasprm, joint_names)`: one actuator row.  `slot` is the backend
  slot and `slot_kind` names its namespace; the MuJoCo family uses
  `slot_kind="mujoco_actuator"` and the compiled `MjModel` actuator id, while
  IsaacLab uses `slot_kind="isaaclab_group"` and `slot=None`.
  `target` is the neutral target name (`None` when unnamed);
  `ctrlrange`/`gear`/`gainprm`/`biasprm` are `None` when the compiled row
  does not declare them (and always `None` for an IsaacLab group).
  `joint_names` lists the neutral joints the actuator drives, in declared
  order: the target joint for a MuJoCo joint actuator, the published group
  names for an IsaacLab actuator group, and empty when the backend publishes
  no such list.
- `ActuatorRuntime(env_ids, ctrl, force, length, velocity)`: one actuator's
  `(E,)` runtime channels for the selected envs.  A channel is `None` when the
  backend cannot read it; every tensor is a detached CPU copy.
- `GeomContact(name, friction, solref, solimp, margin, gap, priority, condim)`:
  one geom's MuJoCo-family contact parameters, keyed by the geom's canonical IR
  name.  `friction`/`solref`/`solimp` are compiled parameter tuples and
  `margin`/`gap`/`priority`/`condim` are scalar compiled values; a `None`
  parameter was not published.  The row carries no raw `MjExt` or engine id.

`env=None` selects every env for state/geometry reads.  `env` may be an int, a
sequence of ints or a tensor; output `env_ids` preserves the requested order.
An empty sequence or empty int64 tensor selects zero envs and returns
zero-row responses.  Scratch is one env: `scratch(None)` selects env 0.

## 4. Subfaces

```python
model.ir(articulation=None) -> ModelIR
model.body/geom/site/joint/actuator(name, *, articulation=None)
model.geom_contact(name, *, articulation=None) -> GeomContact
state.body_pose/site_pose(name, *, articulation=None, env=None) -> Pose
state.com_pose(ref, *, frame="world", env=None) -> Pose
state.root_pose(articulation, *, env=None) -> Pose
state.root_state(articulation, *, env=None) -> RigidState
state.com_state(ref, *, frame="world", env=None) -> ComState
state.object_state(ref: ObjectRef, *, env=None) -> RigidState
state.object_com_state(ref: ObjectRef, *, env=None) -> ComState
state.joint_position/joint_velocity(articulation, *, joint_names=None, env=None)
kinematics.jacobian(target, *, articulation=None, env=None,
                    frame="world", reference="body_origin", rotational=True)
kinematics.object_jacobian(ref: ObjectRef, *, reference="body_origin"|"com",
                           env=None, frame="world", rotational=True) -> Jacobian
geometry.distance(a, b, *, max_distance_m=1.0, env=None, scratch=None)
geometry.aabb(entity, *, env=None)
contacts.manifold(group, *, env=None) -> ContactManifold
contacts.jacobian(point, articulation) -> ContactJacobian
contacts.wrench_to_joint(point, articulation) -> torch.Tensor  # (nv,)
actuators.rows(articulation=None) -> tuple[ActuatorFacts, ...]
actuators.facts(name, *, articulation=None) -> ActuatorFacts
actuators.state(name, *, articulation=None, env=None) -> ActuatorRuntime
```

P2-A implements the four state reads on the MuJoCo family; P2-B implements the
six-column object Jacobian.  A backend that has not declared
`state.root_state`, `state.com_state`, `state.object_state`,
`state.object_com_state` or `jacobian.object` raises `QueryCapabilityError`
through the safe default.  On IsaacLab 2 / IsaacLab 3 the four state reads and
`object_jacobian` are declared unsupported and their direct methods raise
`QueryCapabilityError` rather than `AttributeError`.  The declared levels are
pinned by `tests/unit/query/test_capabilities_matrix.py`.

P0A supports `frame="world"` and `reference="body_origin"`.  `body_origin`
means the target's own origin: the body frame, the site frame, the subtree
COM, or the supplied world point.  Other values raise a capability error
instead of silently changing the frame.

A subtree COM has a defined translational Jacobian but no native rotational
Jacobian.  `ComRef(subtree=True)` with `rotational=True` raises
`QueryCapabilityError`; pass `rotational=False` to read the translational
block.  A body COM (`ComRef(subtree=False)`) keeps both blocks.  IsaacLab 3
exposes only a body COM Jacobian, so there `ComRef(subtree=True)` raises for
both blocks.

IsaacLab 3 reads the native body link / body COM Jacobian rows.  The native
column axis is the six world-frame base DoFs (floating base only) followed by
the articulation's joint order, which is exactly the `DofLayout` order, so the
engine rows are used without reordering.  A fixed base omits the fixed root
from the public body order: the query resolves the root's public index from the
live backend root body (falling back to the declared base binding), leaves
indices before the root unchanged, shifts indices after it down by one, and
raises `QueryCapabilityError` only when the requested body is the root itself.
A site or `PointRef` attached to that fixed root body raises the same error,
because its composed row would start from the missing body link row.
Site and `PointRef` rows are composed from the body link row plus the offset
from the body origin (`J_lin + omega x r`); the angular block is
reference-point invariant.

The base label of a floating articulation is the declared canonical base body
(`ArticulationRuntime.base_body_id`, from `profile.base_body`), not the
backend-native USD binding.  An undeclared base body is a free-root mapping
failure: the query raises `QueryCapabilityError` instead of labelling the six
base columns with an inferred body.  A default `ComRef` (no `body`) resolves the
live body through the declared base's USD binding, matching `state.com_pose`,
and raises `QueryReferenceError` when no base body is declared.

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

### Geom contact parameters

`model.geom_contact(name, *, articulation=None)` returns the frozen
`GeomContact` row from the selected articulation's published `ModelIR` `mj`
extension.  It is a MuJoCo-family surface: MuJoCo and mjWarp declare it
`native`; IsaacLab 2.x, IsaacLab 3.x PhysX and IsaacLab 3.x Newton declare it
`unsupported` and the direct method raises `QueryCapabilityError`.

- `name` resolves through the existing geom table, so an unknown name raises
  `QueryReferenceError` naming the available geoms.
- `articulation=None` reads the merged whole-scene IR when it exists; a
  multi-articulation run without a merged IR raises `QueryReferenceError` and
  must pass `articulation=...`.
- A selected IR without the `mj` extension raises `QueryCapabilityError`
  (`model.geom_contact`, `unsupported`); a missing `mj.geom_contact` row raises
  `QueryReferenceError` naming that extension's rows.  Neither case returns a
  default row.
- The row carries the compiled contact parameters (`friction`, `solref`,
  `solimp`, `margin`, `gap`, `priority`, `condim`); no `MjExt` or raw engine
  handle crosses the public boundary.

### Object, root and COM state (P2-A)

`state.root_state`, `state.com_state`, `state.object_state` and
`state.object_com_state` are MuJoCo-family reads.  All public velocities are
world-frame and every result is a detached CPU copy; reads never call
`mj_forward`/`mj_subtreeVel`, never mutate live `MjData` and do not use
`data.cvel`.

- A free joint stores world linear velocity in `qvel[dof:dof+3]` and
  **body-local** angular velocity in `qvel[dof+3:dof+6]`.  The public angular
  velocity is `R(quat_wxyz) @ qvel_local`, the inverse of the
  `scratch.set_root_state` world-to-local convention.
- `state.root_state` is articulation-only and reads the compiled free-joint
  `qpos`/`qvel`; `lin_vel_w` is at the body-frame origin.  A fixed root raises
  `QueryReferenceError`; a declared-but-unresolved free root raises
  `QueryCapabilityError`, matching `state.root_pose`.  The free root's six DOFs
  are never conflated with `joint_velocity`.
- `state.com_state(ComRef(subtree=False))` reads body COM `xipos` and
  `mj_jacBodyCom @ qvel`; `subtree=True` reads `subtree_com` and
  `mj_jacSubtreeCom @ qvel`.  `ComState` carries linear velocity only: a
  subtree COM has no defined rigid angular velocity, and a body/object angular
  velocity is available from `RigidState`.  A fixed body COM velocity is the
  engine's exact zero, not a fabricated value.
- `state.object_state(ObjectRef)` resolves a rigid free prop through the
  name-only free-prop resolver.  Position and `wxyz` quaternion come from the
  free-joint `qpos`; linear velocity is world `qvel` at the body origin and
  angular velocity is converted from body-local as above.
- `state.object_com_state(ObjectRef)` reads the prop's single rigid body
  `xipos` and `mj_jacBodyCom @ qvel`.  It has no angular velocity.
- Prop resolution never calls `_ir()`/`model_set.merged`, so a
  multi-articulation run with no merged IR still resolves a rigid free prop.
  An unknown name raises `QueryReferenceError` listing the free props; an
  articulation name passed as an `ObjectRef` raises `QueryReferenceError` with
  a hint to use `state.root_state`/`state.com_state`; a static, anchored or
  free-articulation prop raises `QueryCapabilityError`.
- An empty env selection returns zero-row results; multi-env selection preserves
  the requested order.  Angle and position reads are engine values, never
  renormalized or repaired by the query.

### Object Jacobian (P2-B)

`kinematics.object_jacobian(ref: ObjectRef, *, reference="body_origin"|"com",
env=None, frame="world", rotational=True)` is the MuJoCo-family six-column
Jacobian of one rigid free prop.  It returns the shared `Jacobian` type with an
object-owned layout:

```python
base_dofs = (f"{prop}.px", f"{prop}.py", f"{prop}.pz",
             f"{prop}.rx", f"{prop}.ry", f"{prop}.rz")
DofLayout(
    articulation=None,
    base_dofs=base_dofs,
    base_names=(prop,),
    joint_names=(),
    nv=6,
    order=base_dofs,
    owner_kind="object",
    owner=prop,
)
```

The layout is explicitly **not** an articulation: `articulation is None`,
`owner_kind="object"` and `owner` is the prop name.  The six columns are the
compiled free-joint `qvel` order:

```text
columns 0..2 = free-joint world linear qvel (vx, vy, vz)
columns 3..5 = free-joint body-local angular qvel (wx, wy, wz)
```

The Jacobian output rows are world-frame.  `J @ qvel6` therefore yields the
world twist: the translational block is a world linear velocity and the
rotational block, combined with the body-local angular `qvel` basis, is the
world angular velocity.

- `reference="body_origin"` reads `mj_jacBody`; `reference="com"` reads
  `mj_jacBodyCom`.  Both return `transl` and `rot` as `(E, 3, 6)`; the
  rotational block is reference-point invariant.
- `rotational=False` returns a zero rotational block and the real
  translational block.  That explicit request is the only reason for a zero
  block; an unsupported channel raises `QueryCapabilityError`.
- The body-origin and body-COM translational blocks satisfy
  `J_origin.transl = J_com.transl + skew(com - origin) @ J_com.rot`.
- Resolution reuses the P2-A name-only `_object_addresses`: no `_ir()` and no
  `model_set.merged`, so a multi-articulation run without a merged IR still
  resolves the prop.
- `frame` other than `"world"` and `reference` other than `"body_origin"`/
  `"com"` raise `QueryCapabilityError`.  A static, anchored or
  free-articulation prop raises `QueryCapabilityError`; an unknown name raises
  `QueryReferenceError` listing the free props; an articulation name passed as
  an `ObjectRef` raises `QueryReferenceError` with the state-read hint.
- Reads never call `mj_forward`, never mutate live `MjData`, and every result
  is a detached CPU copy.  An empty env selection returns zero-row results;
  multi-env selection preserves the requested order.
- IsaacLab 2 / IsaacLab 3 declare `jacobian.object` unsupported, and the direct
  `object_jacobian` method raises `QueryCapabilityError`.

### Contacts (P1A-1 / P1A-2 / P1C-3)

`contacts.manifold(group, *, env=None)` is **group-based**.  `group` is
required and resolves against the run's declared `contacts:` groups; the
group owns the sensor/filter body ids, so there is no all-contact world scan
and no implicit default group.

- An unknown group raises `QueryReferenceError` naming the declared groups.
- A run without a contact manager, or a group with no resolved sensor bodies,
  raises `QueryCapabilityError`; it never returns an empty result as a
  substitute for "no group configured".
- A group with no active contact returns an empty `points` tuple; this is not
  an error.
- A `general` group matches a contact when one side is a sensor body and the
  other side is not; a `pairwise` group additionally requires the other side
  to be a declared filter body.
- Points are in backend contact order, env-major, and every result is a
  detached CPU copy.  Querying never mutates the contact list and never
  publishes an observation.

#### IsaacLab 3 PhysX filtered manifold (P1C-3)

A contact group opts in with `manifold: true` in its `contacts:` entry.  The
flag is additive and defaults to `false`; the existing force/observation path is
unchanged.  Only a pairwise group (non-empty `filter_refs`) may opt in; a
general group with `manifold: true` fails at spec construction.

- The PhysX contact sensor is declared with `track_contact_points=True` and an
  explicit `max_contact_data_count_per_prim`; the sensor still reports the
  force matrix used by the existing contact backend. The cap is a fixed
  backend declaration (`4` contact points per sensor body and env), not an
  implicit engine default.
- `contacts.manifold(group)` reads the sensor's filtered contact view per env.
  `body1` is the filtered partner and `body2` is the sensor body; this is the
  group's declaration order only, not a proven contact-normal direction.
- `pos_w` and `separation_m` come from the engine.  `separation_m` is the
  engine's signed contact separation; it is never fabricated when the stack
  does not expose it.  `normal_force_n` is the non-negative magnitude of the
  engine's normal force scalar.
- The engine contact normal direction is **not exposed**: this filtered view
  does not prove its one-sided/two-sided orientation, so `normal_w` and
  `frame_w` are `None`.  PhysX contact sensors also expose no geom names and no
  per-point friction force, so `geom1`/`geom2` and `friction_force_w` are
  `None`.  `body1`/`body2` are always present.
- `contact_id` is unique within one manifold read (env-local) and is not a
  stable cross-step identifier.  A group without the opt-in, an unbound group,
  and every IL2/Newton manifold read raise typed capability errors.
- No all-contact scan, step, observation or record write is performed.

### Contact constraint rows (P1A-2)

`contacts.jacobian(point, articulation)` and
`contacts.wrench_to_joint(point, articulation)` map one manifold point's
constraint rows into an articulation's DOF layout.  `point` must come from a
current manifold of the same env; `articulation` owns the output layout.

- The row block starts at `data.contact.efc_address[point.contact_id]`.  Its
  effective row count follows the friction cone, **not** `contact.dim`:
  elliptic uses `dim` rows; pyramidal uses `max(1, 2 * (dim - 1))` paired
  friction rows.  It is sliced from the dense `data.efc_J` and column-scoped to
  the articulation's `DofLayout`; the result is `(rows, nv)`, never the full
  model DOF width.
- `ContactPoint.frame_w` is a basis for the three contact axes (normal,
  tangent1, tangent2), **not** a per-constraint-row axis label:
  - elliptic contacts with `condim >= 3` have rows 0..2 equal to those three
    frame axes; rows 3+ are torsion/rolling constraints;
  - pyramidal rows are friction-pyramid face normals.  For `condim=3` they are
    `(n + t1), (n - t1), (n + t2), (n - t2)`; there is no pure normal row.
    Write a row as a linear combination of `frame_w` axes rather than treating
    each row as one axis;
  - `condim` 4/6 torsion and rolling rows are **not** described by `frame_w`.
- `wrench_to_joint` is `rows.T @ data.efc_force[rows]` in the same row block
  and DOF order.  The sign/frame follow the MuJoCo `efc_J`/`efc_force`
  convention; no zero wrench or zero Jacobian is substituted when the rows are
  unavailable.
- Only a dense `efc_J` is mapped.  A sparse constraint Jacobian, or an
  auto-selected sparse layout (`mjJAC_AUTO` with `nv >= 60`), raises
  `QueryCapabilityError`; compressed values are never reshaped as dense rows.
- A `contact_id` outside the current contact list raises
  `QueryReferenceError`.  A point whose `contact_id` still exists but now names
  a different geom/body pair is stale and also raises `QueryReferenceError`;
  refetch the manifold instead of reusing the old point.  A point whose
  `env_id` is outside the run raises `QueryReferenceError`.  A contact with
  `efc_address < 0` or a row range outside `nefc` raises `QueryCapabilityError`;
  the query never guesses a row block.
- An unknown or ambiguous articulation raises `QueryReferenceError`.  A
  compiled joint shape the layout cannot represent raises
  `QueryCapabilityError`, exactly as for `kinematics.jacobian`.
- Results are detached CPU float64 copies.  The contact list, the constraint
  solve and live `MjData` are not mutated and no observation is published.
- `contacts.jacobian` and `contacts.wrench_to_joint` are declared `unsupported`
  on IsaacLab.  They raise `QueryCapabilityError`.

### Actuators (P1B-1 + P1B-2)

`actuators.rows(articulation=None)` returns every compiled actuator's facts in
`ModelIR` order; `actuators.facts(name, ...)` resolves one neutral actuator
name, and `actuators.state(name, ..., env=None)` reads its live channels.

- Facts come from the compiled `ModelIR` actuator row: `slot` is the compiled
  `MjModel` actuator id, `slot_kind` is `"mujoco_actuator"`, and
  `ctrlrange`/`gear`/`gainprm`/`biasprm` are the compiled values.  An
  undeclared parameter tuple is `None`, never a zero-filled substitute.
  Facts are static: a row that is not compiled still reports its neutral
  identity with `slot=None`; only `state` requires a compiled slot.
- A MuJoCo joint actuator reports its target joint in `joint_names`; a
  non-joint target reports an empty tuple.
- IsaacLab actuator facts come from the published `ModelIR` actuator row and
  the `ext["usd"].attrs["actuator/<name>"]["joint_names"]` list when present.
  They are facts-only: `slot=None`, `slot_kind="isaaclab_group"`,
  `ctrlrange`/`gear`/`gainprm`/`biasprm` are `None`, and an absent joint list
  is an empty tuple.  No control range, gear or gain is fabricated from the
  group configuration.
- Runtime reads `data.ctrl`, `data.actuator_force`, `data.actuator_length` and
  `data.actuator_velocity` at the compiled slot, one value per selected env in
  the requested order.  `env=None` selects every env; an empty selection
  returns zero rows.
- An unknown actuator name, or an ambiguous `articulation=None` on a
  multi-articulation run without a merged IR, raises `QueryReferenceError`
  naming the available actuators.  `facts` returns `slot=None` for an
  uncompiled row; `state` on a row with no compiled slot raises
  `QueryReferenceError` because there is no live channel to read.
- IsaacLab declares `actuators.state` unsupported and raises
  `QueryCapabilityError`: an actuator group has no single scalar
  `ctrl`/`force`/`length`/`velocity` channel compatible with the MuJoCo-family
  row.  `actuators.facts` is `partial`.
- A backend with no actuator face declares `actuators.facts`/`state`
  unsupported and raises `QueryCapabilityError`; it never returns zero rows.
- Querying an actuator never writes `ctrl`, gains or limits, never converts a
  target and never runs `mj_forward`.  The mjWarp implementation reads the CPU
  mirror only: no device sync and no live mirror mutation.
- Results are detached CPU copies.  The actuator subface publishes no
  observation and records nothing.

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

IsaacLab 3 transaction semantics (P1C-2):

- the capability object declares `scratch.mode = "transaction"` and
  `scratch.isolation = "restored"`; the MuJoCo family keeps the default
  `mode = "copy"`, `isolation = "bit_exact"`;
- IsaacLab exposes no snapshot/restore, so `scratch` is an explicit per-env
  transaction, not a copy.  Enter captures every free-root articulation's root
  pose/velocity and every articulation's joint position/velocity for the
  selected env;
- `set_root_state` writes one free-root articulation through the env's live
  state-write face; a fixed-root articulation raises `QueryCapabilityError`.
  The live asset's `is_fixed_base` is the binding fact, so a runtime root
  declared `free` over a fixed USD base is rejected, and such an articulation
  never enters the restore-root set;
- `set_joint_state` writes explicit absolute 1-DOF hinge/slide joint positions
  and optional velocities through the live state-write face;
- `forward()` runs `scene.write_data_to_sim()` + `sim.forward()`; there is no
  implicit sync, and the caller must forward after writes before reading;
- `body_pose`, `site_pose`, `com_pose` and `jacobian` read the live articulation
  buffers inside the transaction; `geometry()` returns an unsupported typed
  stub whose `distance`/`aabb` calls raise `QueryCapabilityError`
  (`scratch.geometry`) because a transaction carries no geometry/contact copy;
- close restores the captured free-root pose/velocity and the captured 1-DOF
  hinge/slide joint position/velocity, runs the same write/forward flush, and
  preserves an in-flight body exception.  A non-1-DOF joint is not written back
  because the scratch write face cannot modify it.  A restore/flush failure
  while a body exception is active is attached to that exception as a note; it
  never replaces the body exception.  Isolation is `restored`, not bit-exact:
  the live buffers are touched while the transaction is open;
- one active scratch per env and per query object; a second concurrent scratch
  for the same env fails loudly, while different envs may hold concurrent
  scratches;
- the caller must not step, reset or open a second scratch during the
  transaction.  The query adds no lock; this is an engine-thread contract.

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
- geometry (`mj_geomDistance`), Jacobians and the contact manifold run as CPU
  MuJoCo math on the mirror or scratch copy;
- `contacts.manifold` reads the mirror contact list and force solve; it is
  `synthetic`, not a native warp device read;
- `contacts.jacobian` and `contacts.wrench_to_joint` read the mirror
  `efc_J`/`efc_force` contact rows with CPU MuJoCo math; they are `synthetic`;
  the mirror must carry a dense `efc_J` (sparse raises `QueryCapabilityError`);
- the live mirror and the warp device `Data` are never mutated by a query;
- there is no implicit device-to-mirror sync and no lock.  The caller must
  query on the engine thread at a step boundary, after the backend has
  synchronized the mirrors.  A mid-step query may read stale mirror state; the
  contract does not repair it.

This is why the mjWarp capability object labels `scratch.*`, `jacobian.*`,
`geometry.*`, `contacts.manifold`, `contacts.jacobian`,
`contacts.wrench_to_joint` and `actuators.facts`/`actuators.state` as
`synthetic` while `model.*` and `state.*` stay `native`.

## 7. Failures

| Case | Error |
|---|---|
| unsupported operation | `QueryCapabilityError` naming backend/op/support |
| unknown/ambiguous entity, actuator or contact group | `QueryReferenceError` with available names |
| `model.geom_contact` on an IR without an `mj` extension | `QueryCapabilityError` (`unsupported`; no default row) |
| `model.geom_contact` on a geom absent from `mj.geom_contact` | `QueryReferenceError` with the extension's rows |
| `actuators.state` on a row with no compiled slot | `QueryReferenceError` (no fabricated slot; `facts` may report `slot=None`) |
| contact query without a declared `contacts:` group | `QueryCapabilityError` (no all-contact fallback) |
| contact id outside the current contact list | `QueryReferenceError` with the available contact ids |
| in-range contact id now naming a different geom/body pair | `QueryReferenceError` (stale point; refetch the manifold) |
| point names an env outside the run | `QueryReferenceError` with the available env ids |
| contact constraint rows unavailable (`efc_address < 0`, bad row range/`nefc`) | `QueryCapabilityError` (no zero row block or wrench) |
| sparse `efc_J` (explicit or auto-selected) | `QueryCapabilityError` (dense constraint Jacobians only) |
| bad shape/dtype/value | `QueryShapeError` (also a `ValueError`) |
| read after run/env/query close | `QueryStaleError` |
| concurrent step/reset | caller contract violation (engine-thread only) |
| mid-step / unsynchronized mjWarp mirror | caller contract violation; no silent device sync |

Failures are loud.  There is no hidden fallback, default substitute or silent
degradation, and query results are never published as observations.

## 8. IsaacLab model + pose subset (P0B-2)

IsaacLab 2.x and the IsaacLab 3.x PhysX/Newton identities share one query
implementation (`unigym.backends.shared.isaaclab_query`); the backend modules
only bind the env type, capability identity and live quaternion order.

Semantics:

- model facts come from the published `ModelIR` / `model_set` exactly as
  `run.model` exposes them; `model.geom` and `model.site` raise
  `QueryCapabilityError` because the IsaacLab IR carries no geom/site facts and
  P0B-2 does not synthesize them;
- the live IsaacLab 3 IR is compiled from the env-0 articulation asset root,
  normalizing `{ENV_REGEX_NS}`, `/World/envs/env_.*`, `/World/envs/env_[^/]+`
  and concrete `/World/envs/env_<n>` spellings.  If the declared prim is absent
  from the live stage the provider falls back to
  `root_physx_view.prim_paths[0]` (the resolved asset root in this IL3 version,
  not the root link).  An unresolved root prim that yields no joint facts
  raises `ModelIRError` instead of publishing an `unknown` joint IR;
- live state reads the articulation data buffers
  (`body_pos_w`/`body_quat_w`, `root_pos_w`/`root_quat_w`, `joint_pos`/`joint_vel`);
  IsaacLab 3.x sim quaternions are `xyzw` and are converted to repo-internal
  `wxyz` at the read boundary;
- `state.root_pose` requires a `root="free"` articulation, matching the
  MuJoCo-family contract;
- `state.com_pose` reads the body pose plus the `ModelIR` body `ipos` offset.
  The `Pose` quaternion is identity because a center of mass is a point (the
  same documented contract as the MuJoCo-family query); subtree COM and
  non-world frames raise `QueryCapabilityError`;
- `state.site_pose` supports only robot-profile sites: the profile's site body
  binding and local `pos`/`quat_wxyz` are composed with the live body pose.
  There is no USD prim, `FrameTransformer` or generated prop-site fallback;
- `joint_position`/`joint_velocity` read the articulation's declared joint
  names; unknown names raise `QueryReferenceError` with the available list.  A
  requested joint whose `ModelIR` row is not a hinge/slide 1-DOF joint
  (ball/fixed/other) raises `QueryCapabilityError`; the live column is never
  returned;
- `actuators.facts` reports static actuator-group facts only (`slot=None`,
  `slot_kind="isaaclab_group"`, no ctrlrange/gear/gains/limits, joint names
  from the published USD attrs).  `actuators.state` and `geometry` are declared
  unsupported and raise `QueryCapabilityError`; IsaacLab 3 PhysX
  `contacts.manifold` is `partial` for opted-in pairwise groups (P1C-3), while
  the IsaacLab 2.x / Newton contact manifold and every contact
  Jacobian/wrench-to-joint read stay unsupported;
- IsaacLab 3 `scratch` is the transaction documented in §5: root/joint writes
  go through the live state-write face and close restores the captured state.
  IsaacLab 2.x keeps `scratch` unsupported; scratch `geometry()` returns an
  unsupported typed stub whose `distance`/`aabb` calls raise on both;
- IsaacLab 3 `kinematics.jacobian` reads the native body link / body COM rows
  and composes site/world-point rows from the body link row; IsaacLab 2.x has
  no native Jacobian API and keeps `jacobian.*` unsupported;
- results are detached CPU torch copies.  Query never writes `env.obs` and
  never records.  Outside an active transaction scratch it never mutates live
  simulator state; a transaction scratch restores the captured state on close.

The PhysX/Newton capability identity is unit-covered; the Newton real-env lane
remains unverified until that lane environment is available.
