# Contacts contract (Scheme A: explicit typed body refs)

Status: breaking. The legacy string body lists, `auto` owner and the
`{contact_group: ...}` / `{prop: ...}` reference forms are REMOVED.

## 1. Declaration

A run-level `contacts:` entry is one group. The only body address is
`ContactBodyRef`:

```yaml
contacts:
  - name: crate_left_hand
    sensor_refs:
      - {owner: object, name: crate}
    filter_refs:
      - {owner: robot, articulation: left_hand, body: ".*"}
    frame: world
    format: contact
    threshold: 0.5
    aggregation: sum_magnitudes

  - name: crate_table
    sensor_refs: [{owner: object, name: crate}]
    filter_refs: [{owner: support, name: table, body: ".*"}]
    frame: world
    format: contact
    threshold: 0.001

  - name: wrist_hand_contact
    sensor_refs:
      - {owner: robot, articulation: left_arm, body: "left_wrist_.*"}
    filter_refs:
      - {owner: robot, articulation: left_hand, body: ".*"}
    frame: world
    format: contact
    threshold: 0.1
```

Group keys: `name`, `sensor_refs`, `frame` (required, no default),
`filter_refs` (optional), `format`, `component`, `threshold`, `aggregate`,
`history`, `aggregation`, `manifold`.

| key | rule |
|---|---|
| `manifold` | Optional bool, default `false`. Only a `pairwise` group (non-empty `filter_refs`) may set it `true`; `general` + `true` fails at construction. It opts the group into `q.contacts.manifold(name)`; currently only `isaaclab3_physx` implements the filtered read, while IsaacLab 2.x and Newton raise a typed capability error. The existing force/obs path is unchanged. |

Derived (never declared): `mode` (non-empty `filter_refs` -> `pairwise`,
otherwise `general`) and `owner` (`object` / `robot` / ... when all sensor
refs share one owner, otherwise `mixed`). Group-level `owner`, `mode`,
`articulation`, `sensor_bodies` and `filter_bodies` are removed keys and fail
with a teaching error.

## 2. ContactBodyRef

```text
owner: robot | object | support | fixture | environment
articulation: str | None
name: str | None
body: str | None
body_set: str | None
path: str | None
match: regex | exact      # default regex (fullmatch)
```

Per-owner rules:

| owner | required address | forbidden |
|---|---|---|
| `robot` | exactly one of `body` / `body_set` | `name`, `path` |
| `object` / `support` / `fixture` | `name`; `body` optional (scope root) | `articulation`, `body_set`, `path` |
| `environment` | exactly one of `body` / `path` | `articulation`, `name`, `body_set` |

- A robot ref must name `articulation` when the run declares more than one
  articulation. It is legal to omit it in a single-articulation run.
- `body: "*"` selects every body in the scope; `match: exact` makes `body`
  literal (including `"*"`).
- `path` is the backend-native escape hatch and cannot combine with other
  address fields.
- Unknown keys and invalid combinations fail immediately.

## 3. Inventory and resolution

Each backend builds a `ContactInventory` of `ContactBodyAddress` rows:

```text
owner, scope, body, native, prim_path, collision, contact_capable, key, root
```

Rows come from the declared robot articulations, declared object/fixture
props, scene support bodies and native environment bodies. Resolution is
explicit and per ref:

1. filter rows by owner (+ articulation for robot refs);
2. resolve `body_set` through the declared per-articulation body sets, or
   match `body` by regex/exact; an omitted scoped `body` takes the scope root;
3. keep backend-native row order inside each ref;
4. concatenate refs in declaration order.

There is no global regex union. Duplicate resolved addresses, unknown scopes
and ambiguous robot refs (no articulation in a multi-articulation inventory)
fail with the candidate canonical keys (for example
`robot/left_hand/left_hand_index_0_link`,
`object/crate/crate`, `environment/world/world`).

B/F order is therefore: declaration order of refs, then backend-native body
order within each ref. B is `sensor_refs`, F is `filter_refs`.

## 4. Output shapes and facts

```text
format     general        pairwise
raw        (N, B, 3)      (N, B, F, 3)
norm       (N, B)         (N, B, F)
contact    (N, B) bool    (N, B, F) bool
```

`contact/<name>` is the sensor/record key. `contact/<name>/force` and
`contact/<name>/active_bodies` are the benchmark force facts; `aggregation:
net | sum_magnitudes` keeps the existing math. Group metadata
(`meta["contacts"][<name>]`) carries the derived owner/owners, mode, frame,
unit and the explicit sensor/filter refs.

## 5. Backends

### MuJoCo / mjWarp

- Inventory is built from `MjModel`: articulation subtrees (fixed by the
  backend base-body bindings), `Prop_<name>*` scopes, free scene supports and
  the world body.
- `object` scopes are free/rigid props (`rigid_object`, `sphere`, `usd`,
  `box`, `free_articulation`); `fixture` scopes are anchored articulated props
  (`articulation`, `usd_articulation`).  The run layer supplies the fixture
  subset, so `free_articulation` is consistently an OBJECT and a fixture ref
  for it fails with candidates.
- Declared prop ownership is exclusive: a prop body nested under an
  articulation contributes only its object/fixture row, never a duplicate
  robot row.  The prop scope root is the top body of the prop's own subtree,
  not "parented to world".
- `support` scopes address free static scene bodies; `environment` addresses
  the world body by name or an explicit native path.  Static support and
  environment refs (including `environment.path`) are refused at build on the
  IsaacLab family, before any path can bypass the limitation.
- `read_detail` records carry `sensor_key`, `other_key`, `sensor_owner`,
  `other_owner`, `sensor_scope`, `other_scope`, `sensor_native`,
  `other_native` in addition to the force/velocity facts.
- `read_detail` is a deferred legacy face.  The query contact manifold does not
  yet expose equivalent fields (including the legacy `v_t` detail), so the
  internal observation providers keep using it and it must not be removed
  until that migration is lossless.  It emits no deprecation warning because
  it runs on the per-step observation path.
- mjWarp reuses the same backend over its CPU mirror.

### IsaacLab2 / IsaacLab3 (shared face)

- The IsaacLab2 and shared IsaacLab3 (`isaaclab3_physx` / `isaaclab3_newton`)
  faces use the same typed ref/inventory resolution.
- Object/fixture refs address `{ENV}/Prop_<name>` (root body only in this
  lane). Robot refs are anchored at the owning articulation prim leaf
  (`Robot`, then `Robot_<name>`).
- A referenced object prop must enable its physics contact reporter:
  `build_contact_manager` computes the referenced prop names and the run
  calls `PropManager.require_contact_sensors(...)`; the prop spawners pass
  `activate_contact_sensors=True`.
- General groups create one sensor cfg per sensor ref; pairwise groups create
  one sensor cfg per sensor ref with flattened filter prim paths in F order.
  Each pairwise sensor/filter ref must resolve exactly one body. `read_raw`
  asserts the sensor F axis equals the resolved filter count.
- Static ground/support filters are NOT representable (the rigid-prop contact
  sensor surface cannot report static pairs): they are refused at build with
  a clear limitation. Use the MuJoCo family for support/environment groups.
- `frame: body` remains unimplemented on the IsaacLab family.
- IsaacLab3 reads the normal-only ProxyArray views via their explicit
  `.torch` accessor; PhysX and Newton share this face.

## 6. Legacy rejection

`ContactGroupSpec.from_yaml` rejects `sensor_bodies`, `filter_bodies`,
`articulation`, `mode` and `owner` with a message naming the explicit ref
replacement. Mapping reference forms and raw string lists are rejected too.
There is no dual syntax and no compatibility warning path.
