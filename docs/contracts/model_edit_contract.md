# Model-edit contract

Status: public, global-only.  Requires `Trait.MODEL_EDIT`.

`run.model` is the read-only `ModelSet`.  Model parameters are changed through
one global transaction:

```python
with run.model_edit() as edit:
    edit.body("go2", "FL_thigh").set(
        mass=1.2,
        com=(0.01, 0.0, -0.02),
        inertia=(0.02, 0.03, 0.04),
        inertia_quat=(1.0, 0.0, 0.0, 0.0),
    )
    edit.geom("go2", body="FL_calf", geom="FL_calf_geom").set(
        friction=(0.9, 0.02, 0.001),
        restitution=0.2,
    )
```

`run.model_edit()` / `RunInstance.model_edit()` / `env.model_edit()` return a
`ModelEditor`.  `NativeHandle` remains the external-only long-tail escape
hatch; common model fields belong here.

## 0. Scope

- Global only: every edit applies to every env.  There is no `env_ids`.
- No raw solver parameters, geom density/mass, actuator gains, equality slots
  or site poses.
- No automatic mass -> inertia coupling.
- No undo/history: model edits are not rewound.
- IsaacLab2 has no published geom IR, so geom fields are a capability error.
- The IsaacLab 3 identities (`isaaclab3_physx` / `isaaclab3_newton`) declare
  no `MODEL_EDIT` trait, so every field below is a capability error.

## 1. Field matrix

The `isaaclab3_*` column covers both IsaacLab 3 identities
(`isaaclab3_physx` / `isaaclab3_newton`).

| Field | mujoco | mjwarp | isaaclab2 | isaaclab3_* |
|---|---|---|---|---|
| body.mass | yes | yes | yes | no |
| body.com | yes | yes | yes | no |
| body.inertia | yes | yes | yes | no |
| body.inertia_quat | yes | yes | yes, with inertia | no |
| geom.friction | yes | yes | no | no |
| geom.restitution | approx solref | approx solref | no | no |

Units and frames:

- `mass`: kg, strictly positive.
- `com`: metres, body-local.
- `inertia`: principal diagonal `kg m^2`, non-negative.
- `inertia_quat`: wxyz, normalized; on IsaacLab2 it requires `inertia` in the
  same `.set()` call.
- `friction`: `(slide, torsional, rolling)`, finite and non-negative.
- `restitution`: `[0, 0.999]`.

## 2. Transaction rules

- Names resolve against the published `ModelSet` immediately; unknown names
  fail with candidates.
- The world body (`body_id=0`) rejects body-parameter edits.
- A geom must belong to the named body.
- A field may appear at most once per transaction.
- `.set()` with no field is an error.
- Clean context exit commits; an exception rolls back.
- An editor is single-use: commit/rollback closes it.
- An empty transaction is a successful revision-preserving no-op.
- No partial write: every touched row is snapshot/restored on failure.
- A commit replaces `env.model_set`; the old `run.model` object remains a
  valid before snapshot.
- Edits are only legal between steps (`env._stepping` guard).
- The backend's `_model_revision` increments on every non-empty commit.

## 3. Backend semantics

### MuJoCo family

The shared writer edits `body_mass`, `body_ipos`, `body_inertia`,
`body_iquat`, `geom_friction` and `geom_solref`.  COM/inertia-quat edits clear
`body_simple` before `mj_setConst` (sameframe safety).  Restitution is an
approximation: `solref[0]` is retained and `solref[1]` receives the damping
ratio derived from the restitution.  Direct-mode `solref` (`solref[0] <= 0`)
is rejected.  The writer never calls `mj_forward` and never touches
qpos/qvel/warm-start.

mjwarp applies the same CPU edit + `mj_setConst`, then pushes the edited and
derived global model fields into the shared device model at leading index 0.
It never calls `put_model`/`put_data` and never changes `_wdata` state or
warm-start.

### IsaacLab2

Body mass/COM/inertia are written through
`root_physx_view.get_masses/set_masses`, `get_coms/set_coms` and
`get_inertias/set_inertias`; v1 broadcasts to every env row.  A non-identity
inertia frame is written as `R(iquat) @ diag(I) @ R(iquat)^T` (row-major).
`describe()` publishes the COM as `ipos` and the principal-axis quaternion as
`iquat`.  DR events may overwrite these values at reset; v1 does not reapply.

## 4. Snapshot revision

MuJoCo/mjwarp snapshot payloads carry `meta.model_revision`.  Restore refuses
a payload captured under a different revision.  Old payloads without the key
remain compatible.  Rewind never undoes a model edit.
