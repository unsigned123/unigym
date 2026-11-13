# Pack API: manifest schema, catalog and runtime

Status: Phases 1 through 9 are implemented. `unigym.pack/1` plus
`PackRuntime` is the only pack loader:

```text
discover -> catalog -> resource/asset -> activate
```

The legacy `unigym.packs` mount, manifest, resolve and load modules are
deleted; there is no dual-read period, compatibility shim or implicit
repository mount. Scene references resolve only through the runtime `scenes`
face; run config references resolve only as absolute file paths or exact
runtime `runs` catalog ids; benchmark pack discovery reads only resource-backed
entries of the runtime `benchmarks` face. Asset references resolve once at the
composition/domain boundary into an `unigym.assets.plan.AssetPlan`: prop
templates, generated `AssetDefinition` files, scene assets and MJCF documents
reach backend code with absolute paths only. `UNIGYM_ASSET_ROOT` /
`UNIGYM_ASSET_PACK` were removed from the known placeholder vocabulary. Code
packs activate explicitly through `PackRuntime.activate_all()` at the daemon
and CLI composition boundaries. The asset CLI resolves pack roots and selection
faces through the explicit runtime; catalog build/search keep their external
collector contract because `tools/build_asset_catalog.py` is not part of the
public core.

Parent plan: the "unigym external pack API" draft and its phase briefs in the
planning repository (`plans/drafts/unigym-pack-api.md`,
`plans/drafts/pack-api-phase1-contract.md`,
`plans/drafts/pack-api-phase3-discovery.md`,
`plans/drafts/pack-api-phase3b-sources.md`,
`plans/drafts/pack-api-phase4-resource-lookup.md`,
`plans/drafts/pack-api-phase5-runtime-parity.md`,
`plans/drafts/pack-api-phase6a-scene-cutover.md`,
`plans/drafts/pack-api-phase6b-run-cutover.md`,
`plans/drafts/pack-api-phase7a-benchmark-cutover.md`,
`plans/drafts/pack-api-phase7b-asset-cutover.md`,
`plans/drafts/pack-api-phase9-retire-asset-root.md`,
`plans/drafts/legacy-loader-cutover.md`).

## Public imports

The pack contract is exported from `unigym.pack_api`:

```python
from unigym.pack_api import (
    ActivationHandle,
    AssetContext,
    CatalogEntry,
    PackCatalog,
    PackContext,
    PackManifestV1,
    PackRuntime,
    ProvidedItem,
    discover,
    load_asset_manifest,
    load_manifest_v1,
    register_adapter,
    register_content_provider,
    register_prop_template,
    register_robot,
    register_scene,
    register_termination,
)
```

The removed helper exports `find`, `resolve`, `code_pack_root`, `load_table`
and `asset_path` no longer exist. Pack code reaches its own resources only
through `PackContext`, and every resource resolves through `PackRuntime`.
`unigym.packs` exports only `PackError`; `MANIFEST_NAME`, `FACE_NAMES` and the
manifest types live in `unigym.packs.schema`. Importing `unigym.pack_api`
never imports pack Python.

## Manifest document

A pack root carries one `unigym.pack.yaml`:

```yaml
schema: unigym.pack/1
name: acme
version: 1.2.0
api: 1
trust: code
entrypoint: acme_unigym:register

requires:
  unigym: ">=0.107,<0.108"

provides:
  scenes:
    - id: acme.tabletop
      resource: scenes/tabletop.yaml
  runs:
    - id: acme.demo
      resource: runs/demo.yaml
  tasks:
    - id: acme.pick
      resource: tasks/pick.yaml
  assets:
    - id: acme.cup
      resource: assets/cup/asset.yaml
  providers:
    - id: acme.dynamic_assets
      kind: asset_generator
      entrypoint: acme_unigym.assets:register
    - id: acme.arm
      kind: robot
      resource: robots/acme_arm.yaml
```

### Required fields

| Field | Form |
| --- | --- |
| `schema` | exactly `unigym.pack/1` |
| `name` | `[a-z][a-z0-9_]*` |
| `version` | dotted numeric version, e.g. `1.2.0` |
| `api` | positive integer |
| `trust` | `content` or `code` |
| `provides` | mapping of declared face to a list of items |

The document is fail-closed. Unknown top-level keys and unknown faces are
rejected; missing or malformed required fields raise `unigym.packs.PackError`
with the offending field and the expected form.

### Optional fields

| Field | Form |
| --- | --- |
| `entrypoint` | `module:attribute`, allowed only with `trust: code` |
| `requires` | string to string mapping; stored, not evaluated |

There is no `shadow` key in `unigym.pack/1`. A manifest that declares it is
rejected as an unknown top-level key; duplicate pack names are always an error.

`requires` records the compatibility range a pack intends. Requirement
resolution belongs to a later loader phase and is not part of the runtime.

## Provided items

The face vocabulary is explicit:

```text
runs  scenes  tasks  assets  benchmarks  props  terminations  providers  selection
```

Each item is a mapping with:

| Key | Rule |
| --- | --- |
| `id` | required, non-empty string; namespaced by convention as `<pack>.<item>` |
| `resource` | exactly one of `resource` / `entrypoint` |
| `entrypoint` | exactly one of `resource` / `entrypoint`; `module:attribute` |
| `kind` | optional non-empty string naming a provider kind, e.g. `asset_generator`, `adapter` or `robot` |

An item with neither locator or with both locators is rejected. An
`entrypoint` on any item requires the manifest `trust: code`. Unknown item keys
are rejected. A `kind: robot` provider is either entrypoint-backed (a factory)
or resource-backed (one `unigym.robot/1` document); a `kind: adapter` provider
is entrypoint-only.

### Resource rules

`resource` is relative to the manifest root and must:

- be a non-empty relative path;
- not be absolute and not contain `..`;
- resolve inside the manifest root (symlink escapes are rejected);
- exist as a file or directory under the root.

The loader validates the path but does not parse the resource. Domain schemas
(scene, asset, task, run) stay owned by their existing modules.

### Trust model

- `trust: content` forbids executable declarations: a manifest-level
  `entrypoint` or any item `entrypoint` is rejected.
- `trust: code` allows one manifest-level `entrypoint` and item entrypoints.
  Trusted pack Python is normal Python; the schema only validates the string
  and never imports it.

`load_manifest_v1(root)` reads `unigym.pack.yaml` and validates the document.
It never imports pack Python, so enumerating manifests does not execute a pack.

## Catalog

`PackCatalog` is a pure, immutable view over already-loaded manifests:

```python
from unigym.pack_api import PackCatalog

catalog = PackCatalog((manifest_a, manifest_b))
catalog.packs()               # ("alpha", "beta"), declaration order
catalog.get("alpha")          # PackManifestV1
catalog.entries()             # every CatalogEntry, declaration order
catalog.entries(face="scenes")  # filtered by face
catalog.entry("scenes", "alpha.tabletop")  # one CatalogEntry by exact id
```

`CatalogEntry` carries `pack`, `face`, `id`, `resource`, `entrypoint` and
`kind`.

`catalog.entry(face, id)` is an exact lookup of one already-loaded item. It
validates `face` against the declared vocabulary, then matches `id` on that
face. An unknown face raises `PackError` with the known faces; an unknown id
raises `PackError` naming the face, the requested id and the known ids on that
face. The lookup is catalog-only: it never discovers, scans, parses or imports
a pack.

Catalog rules:

- pack order and per-pack item order are preserved;
- pack names must be unique;
- a `(face, id)` pair must be unique across the whole catalog;
- manifests and returned tuples are immutable from the caller's perspective;
- `get`, an unknown `entries(face=...)` and an unknown `entry(face, id)` raise
  `unigym.packs.PackError`;
- the catalog never discovers, scans or imports packs.

## Runtime

`discover` builds a runtime from explicit roots or from the configured pack
sources:

```text
discover(roots=None, entry_points=False) -> PackCatalog + root_by_pack
  -> runtime.catalog           enumerate packs and provided items
  -> runtime.resource(...)     resolve one pack-relative resource
  -> runtime.resource_for(...) resolve a catalog item's declared resource
  -> runtime.activate(...)     import a code pack's entrypoint once
  -> runtime.activate_all()    activate every catalog pack in catalog order
```

```python
from unigym.pack_api import discover

runtime = discover(["/packs/acme_code", "/packs/acme_content"])  # explicit
configured = discover()                                          # configured
installed = discover(entry_points=True)                          # opt-in entry points
```

### Explicit roots

Passing a non-`None` iterable reads only those roots, in order. Configured
sources are ignored, so `discover([])` is an explicit empty discovery. Every
root must contain `unigym.pack.yaml`, validated by `load_manifest_v1`; duplicate
pack names fail.

### Configured sources

`discover()` and `discover(None)` collect roots from these sources, in order:

1. `UNIGYM_PACKS`, an `os.pathsep`-separated list of pack roots;
2. `packs:` in the local overlay, `<repo>/unigym.local.yaml`;
3. installed `unigym.packs` entry points, name-sorted, only with
   `discover(entry_points=True)`;
4. every directory under `<UNIGYM_DEPS_ROOT>/unigym_packs`, sorted by name.

Rules:

- root order is preserved and canonical duplicate roots are visited once,
  including entry-point roots that repeat another source;
- every root must contain `unigym.pack.yaml`, validated by
  `load_manifest_v1`; a malformed or missing manifest in a configured root
  fails;
- duplicate pack names fail;
- an empty `UNIGYM_PACKS` entry, including a trailing or doubled separator, is
  a teaching error naming the variable and entry index;
- a missing local overlay is not an error; a malformed one fails instead of
  disabling discovery. `packs:` must be a list of non-empty strings, and
  relative entries resolve against the overlay's directory;
- an unset or empty `UNIGYM_DEPS_ROOT`, a missing deps directory, or a missing
  `unigym_packs` directory yields no roots and is not an error. A directory
  that exists under `unigym_packs` is a configured root and must carry a
  manifest;
- default discovery reads manifests and checks resource paths only. It never
  reads installed entry points and never imports pack Python.

#### Entry-point sources

`entry_points=True` is an explicit opt-in for configured discovery:

- an entry point in the `unigym.packs` group resolves to a path string
  (`str` or `os.PathLike`) or to a zero-argument callable returning one;
- resolving an entry point loads its target and may import it. This is why the
  opt-in exists: default manifest enumeration stays import-free;
- the name is the `EntryPoint` name; roots are ordered by that name;
- a broken entry point fails naming the entry point and the reason: a load
  error, a raising callable, a wrong target type or a target without a valid
  `unigym.pack.yaml`;
- `entry_points=True` with explicit roots is a teaching `PackError`, not a
  silently ignored keyword.

There is no implicit repository mount in this runtime. A resource is addressed
through an explicit pack root or an absolute path outside the catalog; the
checkout itself is not added as a hidden fallback pack.

`discover(..., config_path=...)` is an optional keyword-only override for the
local-overlay path, for embedding and tests. It only affects configured
discovery; explicit roots ignore it. `None` keeps `<repo>/unigym.local.yaml`.

`PackRuntime` is an immutable catalog plus a `pack -> root` mapping. The
constructor rejects a mapping that does not name exactly the catalog's packs.

### Resources

`runtime.resource(pack, rel)` returns an absolute `Path` inside that pack's
root. It applies the same rules as the schema: `rel` must be non-empty and
relative, must not contain `..`, must resolve inside the root and must exist.
Errors name both the pack and the path.

`runtime.resource_for(face, id)` looks up the catalog item through
`catalog.entry(face, id)` and resolves its declared `resource` through
`runtime.resource(entry.pack, entry.resource)`, preserving the same path and
existence checks. It returns the absolute resource `Path`.

Lookup is catalog-only: `resource_for` never activates or imports pack code. An
entrypoint-only item has no resource path, so it raises `PackError` naming the
pack, face, id and entrypoint and pointing the caller at
`runtime.activate(pack)`.

### Asset references (Phase 7b-1 foundation)

`runtime.asset_path(value)` resolves one declared asset reference to an absolute
`Path`:

```text
pack:<pack>/<rel>        -> <pack root>/<rel>, texture leaves canonicalized
/absolute/path           -> expanded absolute path, no existence check
~/path                   -> expanded absolute path, no existence check
bare relative reference  -> teaching PackError
```

Rules:

- only a non-empty string is accepted;
- an absolute or `~` path is a run config's own asset: it is expanded and
  returned as an absolute `Path` without a pack lookup or an existence check,
  matching the run config's own absolute path;
- a `pack:<pack>/<rel>` reference is canonicalized first with the existing pure
  `unigym.assets.refs.canonical_asset_path` rule, so a texture leaf such as
  `assets/generators/tex/a/wood.png` names
  `assets/generators/tex/a/wood/wood.png` before the lookup;
- the canonicalized reference then resolves through `runtime.resource(pack,
  rel)`, inheriting the non-empty, relative-only, `..`, inside-root, symlink
  escape and existence checks;
- a bare relative reference is rejected because it has no owning pack; the
  teaching error names the two accepted forms;
- an unknown pack, illegal path or absent resource is a `PackError` that names
  both the pack and the original reference.

`AssetContext` is the frozen low-level path resolver:

```python
@dataclass(frozen=True)
class AssetContext:
    runtime: PackRuntime

    def resolve(self, value: str) -> str: ...
```

`AssetContext.resolve` delegates to `runtime.asset_path` and returns the path as
a string. It owns no second resolution rule and carries no module-level or
ambient state.

### Boundary-resolved assets (Phase 7b-2)

`unigym.assets.plan.AssetPlan` is the one asset boundary:

```python
plan = AssetPlan(AssetContext(runtime))
plan.resolve("pack:acme/assets/meshes/cup.stl")
plan.resolve_definition(asset)   # all mesh/texture files absolute
plan.resolve_document(mjcf_text) # placeholders + pack: file leaves resolved
```

Rules:

- `RunSpec.pack_runtime` carries the runtime without serialization;
  `build_instance` builds the plan from it and hands it to `PropManager` and
  the env backend;
- `build_spec(..., runtime=...)` builds an `AssetContext` and passes it to
  `expand_placeholders`; `Run`, `collect_cli` and `describe_cli` build
  `discover(entry_points=True)` at their composition boundary when the config
  declares a scene, runs in `benchmark`/`both` mode, or carries any `pack:` /
  asset-reference string;
- `unigym.core.paths.expand_placeholders(obj, context=None)` keeps
  `${UNIGYM_REPO}` and `${UNIGYM_DEPS_ROOT}`; `${UNIGYM_ASSET_ROOT}` and
  `${UNIGYM_ASSET_PACK}` are teaching `PlaceholderError`s, and a `pack:`
  reference without a context is one too;
- `PropManager` resolves template entries and generated payload files at
  construction; backend `add_ir_body` / `_add_declarations` accept absolute
  `mesh.file` / `tex.file` only and perform no resolution;
- the scene engine's `compose(scene, asset_plan)` resolves scene material
  textures and instance payload files at the scene boundary;
- `load_spec` parses a resolved MJCF document only and rejects one that still
  carries `${...}` or a `pack:` reference (old content is not migrated);
- no backend module imports a path or asset-anchor helper; generic path tokens
  live in `unigym.core.paths`, asset path policy lives in `unigym.assets.refs`,
  and the old `unigym.packs.refs` resolver was deleted.

### Path tokens and asset path policy (Phase 9)

The generic path helpers have one owner, `unigym.core.paths`:

```python
from unigym.core.paths import (
    PlaceholderError,
    deps_root,
    expand_placeholders,
    known_tokens,
    repo_root,
)
```

The asset path mapping has one owner, `unigym.assets.refs`:

```python
from unigym.assets.refs import canonical_asset_path, texture_payload_rel
```

Rules:

- `unigym.core.asset_root` is deleted; there is no compatibility shim and no
  second owner for either helper family;
- only `${UNIGYM_REPO}` and `${UNIGYM_DEPS_ROOT}` remain known path
  placeholders, and `known_tokens()` names exactly those two;
- `${UNIGYM_ASSET_ROOT}` and `${UNIGYM_ASSET_PACK}` stay teaching
  `PlaceholderError`s rather than becoming unknown-token typos;
- a whole-string `pack:<pack>/<rel>` reference still requires an explicit
  `AssetContext`; expansion without one is a teaching `PlaceholderError`;
- placeholders expand once at the run/template/asset boundary, recursively over
  the parsed document; backend render code never expands them;
- `PackRuntime.asset_path` applies `assets.refs.canonical_asset_path` before its
  resource lookup, preserving the exact texture-leaf spellings;
- `unigym doctor` no longer reports `platform.asset_root` or
  `platform.asset_face`; it still sources the dependency root through
  `unigym.core.paths.deps_root` for executor discovery.

### Cameras and depth

Cameras are declared explicitly under the run's `cameras:` list. The manager
serves `cam/<name>/rgb` and `cam/<name>/depth`; a camera only renders the
streams named in its `types` (default `["rgb", "depth"]`).

Depth is served only by an explicitly declared camera. The adapter-facing
`depth_image` / `depth_image_raw` obs aliases map to a declared camera named
`depth`; if an adapter requests either alias and the run declares no `depth`
camera, the run fails with a teaching error. There is no implicit depth shim
camera.

Migration from the removed top-level `camera_depth:` key: declare the camera
under `cameras:` and reproduce the classic depth-encoder geometry explicitly:

```yaml
cameras:
  - name: depth
    rig:
      type: body
      attach: pelvis
      pos: [0.33, 0.0, 0.043]
      pitch_down_deg: 5.0
      prim_path: "{ENV_REGEX_NS}/depth_camera"
    types: [depth]
    width: 106
    height: 60
    hfov_deg: 87.0
    far_clip: 20.0
    depth_type: distance_to_camera
    interval: 1
```

`prim_path` and `attach` are optional: a body rig without `attach` uses the
effective profile's `base_body`. The removed `CameraSpec.depth_body_default`
factory has no replacement; code that constructed it must declare the same
`CameraSpec` explicitly under `cameras:`.

The `depth_image` alias applies
`preprocess_depth` and therefore requires the classic `106x60` render; use
`depth_image_raw` plus adapter-side preprocessing for any other geometry.

### Robot profiles and backend realization

Robot definitions are backend-neutral authoring data. A profile factory returns
`RobotProfile` with:

```text
joint_sets / default_joint_pos
assets (AssetRef: usd | mjcf | urdf refs, or None)
action_faces
base_body / foot_bodies / body_sets   (canonical body ids)
body_bindings                          (canonical id -> per-format native name)
base_height                            (free-base spawn height, or None)
```

`foot_bodies` is semantic metadata (it never implies a sensor). `body_bindings`
is robot metadata, not backend logic: the backend's selected asset format picks
the native name. Each backend declares an ordered, non-empty
`BackendInfo.asset_formats` preference tuple: `("mjcf",)` for
mujoco/mjwarp and `("usd", "urdf")` for isaaclab2/isaaclab3. For each
articulation the run layer selects the first preferred format present on the
effective `AssetRef` (`spec.model` when set, otherwise `profile.assets`). There
is no fallback or conversion: an effective asset with no compatible format
fails with a teaching error. The selected format is used for body bindings,
contact-group native names, the camera base mount and selected-format asset
path resolution; it is not one global backend format.

A profile with no effective `AssetRef` remains legal for backends whose
provider or scene owns the asset (for example the MuJoCo-family scene MJCF).
There is no asset format to select in that case, so the run layer uses the
backend's first declared format as the binding-key format; a backend robot
realizer that needs an asset (IsaacLab) still fails explicitly.
`base_height` is the neutral free-base spawn height: G1
declares `0.78`, GR1 and other robots declare `None`. `ArticulationSpec.spawn`
is the run-level spawn authority; when it is absent, an explicit
`env.params.spawn_height` wins over the profile `base_height`, whose final
fallback is `0.0`.

Contact inventories are declared per articulation:

```yaml
articulations:
  - name: g1
    contacts: all          # every rigid-body link under the robot asset
    # contacts: none       # explicit opt-out from the omitted default
    # contacts:
    #   - {name: feet, bodies: [left_ankle_roll, right_ankle_roll]}
```

- `all` materializes every rigid-body link under the robot asset subtree
  (world, props and other articulations excluded); collisionless bodies stay
  as zero contact channels. It contributes one synthetic inventory group
  named `all`.
- Omitting `contacts` while the run declares a run-level `contacts:` section
  defaults to `all` with a build warning that states the per-body sensor cost
  (on isaaclab one `ContactSensorCfg` per rigid body per env clone).
- `none` suppresses the implicit default; explicitly declared inventory groups
  and explicit run-level `sensor_bodies` lists still work.
- Adapter contact-observation requests never synthesize contact groups; they
  only pre-register requests on a manager built from the run config.
- MuJoCo/mjwarp expand `all` at bind over the referenced articulation's model
  bodies (articulation-binding scoped). IsaacLab expands it pre-clone through
  the inventory in `unigym/backends/isaaclab2/usd.py`, with fixed-joint merged
  links excluded and one sensor per body. That inventory walks a USD asset, so
  an articulation that selects its URDF `AssetRef` cannot use `contacts: all`;
  declare explicit contact groups (or provide the USD reference) instead.

A run-level `contacts:` group owns exactly one articulation:

```yaml
contacts:
  - name: cart_all
    sensor_bodies: {contact_group: {articulation: cartpole, group: all}}
    frame: world
  - name: humanoid_feet
    articulation: g1              # explicit owner for a native body list
    sensor_bodies: [left_ankle_roll_link, right_ankle_roll_link]
    frame: world
    filter_bodies: {prop: cup}    # backend-neutral: the declared prop
```

- The `{contact_group: {articulation, group}}` reference records the owner;
  a direct `articulation: <name>` key may declare the same owner for a native
  body list. Both must agree when both are present.
- A run with more than one articulation requires every contact group to name
  its owner; omission is a teaching error listing the declared articulations.
  A single-articulation run keeps the historical owner-less group.
- A bare filter body is a body NAME: the MuJoCo family matches it as a
  body-name regex (one expression may expand to several bodies, so F grows
  accordingly), while IsaacLab requires it to match exactly one body (F is
  one column per declared expression). For an owned group the MuJoCo family
  resolves a filter in the owner's subtree PLUS the free bodies outside
  every articulation -- the world body (`world`, whose static floor geoms
  make `filter_bodies: [world]` a ground-contact filter) and declared prop
  bodies -- and never another articulation's body. An explicit prim path
  (`/...` or `{ENV_REGEX_NS}/...`) is IsaacLab-only; it is not a body-name
  regex on the MuJoCo family (use the native body name, e.g. `world`, or a
  `{prop: <name>}` reference for a backend-neutral prop).
- Sensor prim paths are scoped to the owner's scene leaf: `Robot` for the
  first declared articulation and `Robot_<name>` for the others. The
  MuJoCo family resolves bare sensor bodies in the same articulation and
  never matches another articulation's body when the group names an owner.
- `contacts: all` expands only that articulation's pre-clone inventory, never
  a union across articulations. With several all-mode assets an omitted owner
  is ambiguous and fails; reference the articulation explicitly.
- Duplicate resolved sensor/filter bodies and colliding derived scene field
  names fail fast instead of silently overwriting or double-counting rows.
  This is a deliberate tightening shared by BOTH the IsaacLab and MuJoCo-family
  backends (not a backend quirk): overlapping filter expressions that resolve
  the same body (including the single-articulation MuJoCo case) used to
  produce duplicate F rows and are now rejected at bind time. IsaacLab also
  rejects a bare filter expression matching zero or several bodies, because
  its filter axis is one column per declared expression.
- The env binds a constructor-provided manager after compiling the
  articulation/base-body tables, so `articulation:` owners and
  `contacts: all` resolve on the normal run path as well as on direct
  manager construction.

`RobotProfile` carries no backend object and no backend
switch. There is no `with_isaaclab`, no `articulation_cfg`, no `body_parts`
and no `RobotAsset` field.

A backend that needs an engine object declares `BackendInfo.robot_realizer`:

```text
realizer(profile, *, spec, params, asset_plan) -> backend realization
```

`spec` is the run-level `ArticulationSpec`; its `spawn` mapping is the only
robot-root spawn authority.

- isaaclab2/isaaclab3 build the `ArticulationCfg` from the neutral profile in
  `unigym/backends/isaaclab2/robots/`;
- mujoco/mjwarp consume the neutral profile directly and build no engine robot
  object;
- the run layer validates the realizer callable and passes the same
  profile/params/asset plan; no engine import happens in `unigym.robots`.

G1's isaaclab realization is built through the generic builder: the neutral
profile supplies q0/body facts, the effective `AssetRef` (`spec.model` when
set, otherwise the profile's `assets`) is the spawn asset (the 43-DoF
with-hand `g1.usd` from an installed robot/asset pack), and the run-level
`ArticulationSpec.actuation` groups are the only actuator source. The realizer
selects the first backend-preferred format present on that effective asset:
IsaacLab prefers USD and also accepts URDF (`UrdfFileCfg`), with no MJCF
conversion or template. There is no `G1_29DOF_CFG` template and no hardcoded
G1 actuator table and no `profile.name` dispatch: any effective profile with a
USD or URDF asset and run-level actuation builds through the same generic path.
The realizer resolves each
group's `joint_set` through the profile vocabulary (`resolve_group_joints`)
without a coverage check; the run layer owns driven-joint coverage.
Realizer-only robot-section params (`prim_path`, `spawn_options`,
`collision_group`, `debug_vis`, `soft_joint_pos_limit_factor`) are split off
before the neutral profile factory.  `soft_joint_pos_limit_factor` defaults to
the generic `1.0`; a profile that wants the old G1 parity declares `0.9`
explicitly.  When `spec.spawn` is absent, `default_pos` is
`(0, 0, profile.base_height)` only when `base_height` is not `None`; otherwise
the builder fails because there is no spawn default.  Neutral actuation is
declared on the articulation (`ArticulationSpec.actuation`) and translated by
the backend; `env.params.actuators`, `controls[].gains` and the top-level
`actuator:` override are removed.  GR1 declares no neutral USD/URDF asset, so
the backend realizer rejects it because its effective asset has no compatible
format, not because of its name.  Its MJCF remains a scene resource, not a
neutral robot asset.

The drive/law matrix is explicit and closed:

```text
position_pd    -> standard_pd or computed_torque_pd
position_servo -> native_servo
torque         -> direct_torque
velocity       -> direct_velocity
effort/current -> unsupported (no native direct-drive law)
```

`direct_torque` writes the target torque to compiled `motor` actuators;
`direct_velocity` writes the target velocity to compiled velocity servos.
These are ctrl-space writes: MuJoCo's `ctrl` is the actuator-space value and
the compiled force is `gear * (gainprm[0] * ctrl + bias)`.  For
`direct_torque` (a fixed-gain, bias-free motor) the seam therefore requires
`gear * gainprm[0] == 1`, not just `gear == 1`: a general actuator with
`biastype="none"` and `gainprm="2 0 0"` still looks like a motor but would
double every commanded torque.  `direct_velocity` keeps the `gear == 1` rule
and additionally requires an affine `kv * (ctrl - qvel)` servo
(`gain == -bias[2]`, no constant bias).  A model that cannot honor the named
joint-space unit fails at build with a teaching error rather than silently
commanding a different plant.  Both laws require an empty `gains`/`parts`
payload (the compiled actuator owns the plant) and are supported by the
MuJoCo family only.  The IsaacLab backends have no native direct-drive
actuator mapping and reject these laws with a teaching error; direct drive is
never approximated with a position PD.  `BACKEND_LAW_CAPABILITIES` is the
machine-readable backend capability matrix.

Dynamic `register_robot` / manifest `kind: robot` registration is the explicit
external-robot path: a pack registers a backend-neutral factory or resource
and `build_robot_profile(name)` resolves it without a core factory table.
G1 and GR1 neutral profiles are pack data, not core built-ins.

Migration note: `unigym.robots.g1`, `unigym.robots.gr1`,
`unigym.robots.g1_constants`, `unigym.robots.gr1_constants` and the
`make_g1_profile` / `make_gr1_profile` factories are removed. Install and
activate a robot pack before calling `build_robot_profile` for those names.
`GR1_COTRAIN_POSE` remains in `unigym.robots.gr1_reset` because it is reset
semantics, not neutral profile data.

### Robot data documents

An external robot may be declared as data, with no pack Python:

```yaml
schema: unigym.robot/1
name: acme.arm
joint_sets:
  body: [shoulder, elbow]
default_joint_pos:
  shoulder: 0.0
  elbow: 0.0
base_body: base
body_bindings:
  base:
    mjcf: base
    usd: /World/base
assets:
  usd: pack:acme/robots/acme_arm.usd
```

`unigym.robots.load_robot_profile(path)` builds the neutral `RobotProfile`:

- required: `schema` (exactly `unigym.robot/1`), `name`, `joint_sets`,
  `default_joint_pos`;
- optional: `base_body`, `base_height`, `foot_bodies`, `body_sets`,
  `body_bindings`, `assets`, `action_faces`, `full_joint_order`;
- only top-level unknown keys fail; `body_bindings.<id>.<format>` is data, not
  a top-level field;
- `body_bindings` builds `BodyBinding(canonical=<key>, names=<value>)`;
- `assets` uses the same `{mjcf|urdf|usd: <ref>}` mapping as `AssetRef`; a
  `pack:` reference stays an unresolved string; `assets: null` and an omitted
  key both become `None`;
- every other invariant stays with `RobotProfile`; errors name the file, the
  field and the expected/actual form.

The loader imports `yaml` but never an engine, a backend or pack Python. It
does not know the provider item id; the pack resource provider enforces
`profile.name == item.id`.

### Robot registration

Dynamic registration and manifest providers share the external robot registry:

```python
from unigym.pack_api import register_robot

def register(context):
    register_robot("acme.arm", make_arm_profile)
```

```yaml
provides:
  providers:
    - id: acme.arm
      kind: robot
      resource: robots/acme_arm.yaml
```

```yaml
provides:
  providers:
    - id: acme.arm
      kind: robot
      entrypoint: acme_robot:make_profile
```

For a manifest `kind: robot` item:

- the item declares exactly one locator:
  - `resource: robots/<name>.yaml` loads one `unigym.robot/1` document at
    activation and registers a constant, parameterless factory under the item
    id; a run config that passes non-empty `params` to it is a teaching
    `PackError`;
  - `entrypoint: module:make_profile` imports and registers a callable factory
    (the R1 form);
- a resource profile's `name` must equal the item id; a mismatch is a teaching
  `PackError` naming the pack, the item id and both names;
- a resource-only robot provider list requires no `python/` directory and never
  mutates `sys.path`; when resource and entrypoint items are mixed, the import
  boundary comes from the first item that declares an entrypoint;
- all robot items are loaded and validated before any registration, so a later
  invalid item leaves no earlier registration behind;
- an entrypoint target must be a callable factory (a function or class); a
  profile instance or other non-callable is a teaching `PackError`;
- the item `id` is the robot name used verbatim by
  `unigym.robots.build_robot_profile`;
- factory callability and free-name validation happen before any registration;
  entrypoint return-type and profile-name legality are proved when the factory
  is called by `build_robot_profile`;
- activation failures are wrapped with the pack, item id and locator, and
  restore the prior robot registry exactly.

The external registry is authoritative: names use the adapter-style grammar
`[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*`, are never trimmed, and must be free of
duplicate names. One factory owns exactly one name. A factory must be callable.
`ROBOT_REGISTRY`, `register_robot`, `snapshot_robots` and `restore_robots` live
in `unigym.robots`; `unigym.pack_api.register_robot` is the pack-facing wrapper.
The registry lives for the process lifetime, like `ADAPTER_REGISTRY`.

Manifest robot providers are processed after the pack-level entrypoint and
after adapter providers (or without a pack-level entrypoint). The activation
handle's `robots` field records the manifest `kind: robot` item ids in
declaration order; robots registered by the manifest-level entrypoint itself
are not recorded. A failed activation restores the robot registry exactly on
both failure branches, including entrypoint-registered robots.

### Asset CLI and selection (Phase 7b-3)

`unigym asset` builds `discover(entry_points=True)` at the command boundary when
runtime discovery is needed. There is no module-level runtime or cache.

- `unigym asset file <id>` checks the id namespace against
  `runtime.root_by_pack` and resolves the asset directory through
  `runtime.resource`, so the inside-root, missing-resource and symlink checks
  stay with the runtime owner. It never calls the removed `pack_roots()` /
  `repo_pack()` helpers. An unknown namespace is a teaching error naming the
  discovered packs. A generator id activates every catalog pack through
  `runtime.activate_all()` before reading the template registry; the registry
  and its error behavior are unchanged, and an activation failure is a teaching
  CLI error rather than a traceback.
- `unigym asset selection check` without `--dir` enumerates
  `runtime.catalog.entries("selection")` and resolves each item through
  `runtime.resource_for("selection", id)`. A directory resource is the
  selection directory; a file resource uses its parent directory. Resolved
  directories are canonical-deduped and only directories are checked. There is
  no `<pack-root>/selection/` subdirectory scan.
- An explicit `--dir` keeps its existing behavior and needs no runtime.
- `unigym asset catalog build` and `catalog search` still run the external
  collector `tools/build_asset_catalog.py`. The public core does not
  reimplement the collector; an absent collector fails with the existing
  teaching error.

### Scene resources

A run config's `scene:` value is an exact `scenes` catalog id, not a directory
scan:

```yaml
scene: acme.tabletop
```

Resolution goes through `runtime.resource_for("scenes", <id>)`, parses the
declared YAML mapping and keeps the existing content-key normalization and
three-layer merge (`scene defaults -> task overrides -> run overrides`). The
consumed `scene:` ref is removed after the merge, so `RunSpec.scene` is not a
leftover catalog-id string. The legacy `<family>/<name>` form and a direct
`scene.yaml` file are not scene references; both fail instead of falling back.

The runtime is required at the boundary:

- `apply_scene_bundle(cfg, runtime=None)` returns a config without a `scene:`
  key (or with `scene: null`) unchanged and needs no runtime;
- a present `scene:` ref without a runtime raises a teaching `ValueError`
  naming the ref;
- an unknown id raises the catalog `PackError` with the face and known ids;
- a resource that is not valid YAML or not a mapping fails at resolution.

`build_spec(cfg, args, *, runtime=None)`, `World.from_pack(ref, *,
runtime=...)` and a scene-bearing `World.from_config(source, *,
runtime=...)` follow the same boundary. `Run`, `collect_cli` and `describe_cli`
build `discover(entry_points=True)` at their composition boundary when the
config declares `scene:`, its mode is `benchmark`/`both`, or it carries a
`pack:`/asset reference string; there is no module-level default runtime or
cache.

### Run resources

A run config reference has exactly two accepted forms:

```text
/absolute/path/to/run.yaml     -> used verbatim (must be a file)
acme.demo                      -> exact PackRuntime "runs" catalog id
```

`unigym.run.config_ref.resolve_run_config(spec, runtime=None)` is the one
shared resolver for the CLI and the daemon. Its rules:

- an absolute path or a `~` path is expanded, resolved and must be a file;
- an exact `runs` catalog id resolves through
  `runtime.resource_for("runs", <id>)` and must be a file;
- a catalog id requires a runtime; without one the error names the spec and
  the accepted forms;
- relative filesystem paths and globs are teaching errors naming only the two
  accepted forms;
- there is no implicit `config/` directory, no CWD-relative resolution, no glob
  expansion and no repository fallback;
- an unknown id raises the catalog `PackError` naming the id and the known ids.

All failures are `RunConfigRefError` (a `ValueError`) or `PackError`, surfaced
as CLI teaching errors rather than tracebacks.

`client.cmd_run` builds `discover(entry_points=True)` once, only when at least
one spec is non-absolute. `cmd_debug` and `probe_cli.cmd_probe` build it only
for a non-absolute spec. Absolute paths work without constructing a runtime.

The daemon owns one explicit `pack_runtime` built at `DaemonService`
construction. `GET /api/configs` enumerates `runtime.catalog.entries("runs")`:
resource-backed items carry `{id, path, relative: <id>, ...summary}`;
entrypoint-only run items are reported in `unsupported` instead of being
listed or silently dropped. `benchmark_packs` is listed with
`discover_packs(runtime)`, so the daemon reports the same runtime `benchmarks`
resources the run uses. `POST /api/start` calls the same
shared `resolve_run_config` with the daemon runtime, so it accepts an absolute
path or an exact `runs` catalog id and rejects relative paths and globs.

### Benchmark resources

A benchmark pack is the existing directory (`pack.yaml`, `tasks/`,
`references/`, optional `templates/`) declared as one runtime resource:

```yaml
provides:
  benchmarks:
    - id: acme.bench
      resource: benchmarks/acme
```

Discovery is `builtin` first, then every resource-backed `benchmarks` catalog
entry in catalog order:

```text
PackRuntime.catalog.entries("benchmarks")
  -> runtime.resource_for("benchmarks", <id>)
  -> load_pack(<resource directory>)      # unchanged domain parser
  -> PackIndex(builtin + catalog packs)
```

`unigym.benchmarks.packs.discover_packs(runtime)` requires the runtime and never
scans the repository, dependencies or `benchmark_packs/` directories. The
resource must be a benchmark-pack directory: an entrypoint-only item, a file
resource or a missing resource fails with a `PackError` naming the catalog id.
Duplicate pack names still fail in `PackIndex`.

`unigym.benchmarks.base.build_tasks(sections, meta, runtime)` requires the
runtime; `<pack>/<task>` resolution, unique-bare names and `<pack>/*` wildcard
expansion are unchanged. `RunSpec.pack_runtime` carries the runtime to
`build_instance` and is never serialized into the config or ledger. A
benchmark/both run without a runtime fails at build; `Run`, `collect_cli` and
`describe_cli` build `discover(entry_points=True)` when the config declares
`scene:` or its mode is `benchmark`/`both`.

`benchmark.packs` in a run config is rejected with a teaching `ConfigError`:
packs are declared on the runtime `benchmarks` face. `benchmark.tasks`
resolution and wildcards are unchanged.

Analysis follows the same runtime: `analysis.gate.resolve_task_specs(config,
runtime)` requires it, `gate_doc` builds the configured runtime at its analysis
boundary when none is supplied, and `analyze --metric` does the same.
`unigym benchmarks list` builds the runtime; `unigym benchmarks validate <dir>`
keeps loading one explicit directory through `load_pack`.

### Pack context

A pack entrypoint receives exactly one explicit `PackContext`:

```python
@dataclass(frozen=True)
class PackContext:
    pack: str
    root: Path
    runtime: PackRuntime

    def resource(self, rel: str) -> Path: ...
```

`resource(rel)` delegates to `runtime.resource(pack, rel)`, so it applies the
same pack-relative rules and names the owning pack in errors. The context is
the pack's explicit handle to its own resources; it is not a global registry
and it does not inspect the caller's frame.

### Activation

`runtime.activate(pack)` is explicit and cached per runtime:

- a content pack, or a code pack with no manifest entrypoint and no adapter or
  robot providers, returns a no-op `ActivationHandle(pack=..., entrypoint=None)`;
- a code pack's `entrypoint` is split into `module:attribute`, imported with
  normal `importlib`, and called with one `PackContext` argument;
- after the manifest entrypoint (or without it), manifest `kind: adapter` items
  are imported and registered, then `kind: robot` items are registered. A robot
  item is entrypoint-backed or resource-backed; a resource-only robot list
  needs no `python/` directory and never mutates `sys.path`, while a mixed list
  adds the first entrypoint item's `python/` boundary. All robot items are
  loaded and validated before registration;
- there is no zero-argument fallback: a legacy `register()` entrypoint fails
  to activate and is not supported;
- the pack root's `python/` directory is the explicit import boundary. It is
  added to `sys.path` once before the import and kept there because the pack
  may import its own submodules later;
- there is no AST restriction, custom module loader or sandbox. `trust: code`
  means trusted Python that may import `unigym.pack_api`;
- repeated activation returns the same cached handle and does not re-execute
  the entrypoint;
- an activation failure raises `PackError` naming the pack and entrypoint,
  with the original exception as its cause.

`runtime.activate_all()` activates every catalog pack in catalog order and
returns a `tuple[ActivationHandle, ...]`. It uses the same per-runtime cache,
so a repeated call returns the same handles without re-running an entrypoint.
Content packs remain no-ops.

### Adapter registration

Adapters are registered in the existing process-global adapter registry; there
is no separate adapter face. Two explicit forms produce the same entry:

```python
# dynamic: pack entrypoint code
from unigym.pack_api import register_adapter

def register(context):
    register_adapter("acme.policy", AcmeAdapter)
```

```yaml
# static: manifest-declared provider
provides:
  providers:
    - id: acme.policy
      kind: adapter
      entrypoint: acme_adapter:AcmeAdapter
```

For a manifest `kind: adapter` item:

- an `entrypoint` is required; a resource-only adapter item fails activation
  with a teaching `PackError` naming the pack and item id;
- activation imports the target through the same `module:attribute` resolver
  used by the manifest-level entrypoint;
- the target must be a class that subclasses
  `unigym.adapters.base.Adapter`; an instance, factory or unrelated class is a
  teaching `PackError`;
- the item `id` is the adapter name a run config selects in `adapter.name`;
- activation failures are wrapped with the pack, item id and entrypoint.

Manifest adapter providers are processed after the pack-level entrypoint (or
without one). Manifest robot providers are processed after that. Other provider
kinds stay inert: they are catalog metadata and are not imported or registered
by activation.

Registration is process-global because the adapter registry is global.
`register_adapter(name, cls)` rejects a malformed name, a non-class, a
non-`Adapter` class and a duplicate name; a duplicate error names the class
already registered under that name. One class owns exactly one adapter name:
registering the same class under a second name is rejected as an alias.
Activation is idempotent per runtime through `ActivationHandle`, whose
`adapters` field records the manifest adapter ids that activation registered.
A dynamic registration and a manifest registration for the same name still
fail loudly on the second registration.

Manifest adapter activation is transactional within one pack: every
`kind: adapter` item is validated (entrypoint present, target importable, a
class that subclasses `Adapter`, and a free registry name) before any of them is
registered. Adapter names registered by the pack-level entrypoint are rolled
back if a later provider item fails. A later invalid item therefore leaves no
earlier adapter from that manifest registered, and retrying the same failing
pack reports the original defect rather than a leaked duplicate.

`unigym.pack_api.register_adapter` imports the adapter package lazily, so
importing `unigym.pack_api` never imports torch or the adapter registry.
`adapter_params_registry`, `build_adapter` and the spawn-params validation see a
registered adapter immediately.

Activation does not scan for adapters or import adapter modules on its own; the
only automatic step is the manifest-declared `kind: adapter` provider above.
A dynamic-only adapter has no manifest item and is legal.

### Code activation at composition boundaries

Activation is explicit at each process boundary, never a module-level flag or
cache:

- `DaemonService.__init__` discovers the configured runtime and calls
  `activate_all()` before the control plane starts;
- `collect_cli` and `describe_cli` always build `discover(entry_points=True)`,
  call `activate_all()` and pass that runtime to `build_spec`;
- `Run` builds `discover(entry_points=True)` and calls `activate_all()` before
  `build_spec` whenever the config needs a pack runtime, including any adapter
  name other than the built-in `manual`;
- the launcher does not load code packs globally; command modules own their
  activation at their composition boundary.

## Pack CLI

`unigym pack` is backed by the runtime and `discover(entry_points=True)`:

- `pack list [--json]` prints each runtime manifest's name, version, api,
  trust, declared provided-item counts per face and root;
- `pack path <name> [face]` prints the pack root, or for one declared face
  prints each item's absolute resource path or its entrypoint;
- `pack doctor [--name NAME] [--json]` revalidates the manifest schema,
  checks `BUILD_MANIFEST.txt` against the live tree and activates the pack
  through the runtime. It performs no legacy mount, api/requires or mount-face
  checks.

There is no `repo` pack and no mount table. The CLI fails with the known packs
or known faces, never with an implicit checkout fallback.

## Breaking removals

- `unigym.packs.mount`, `unigym.packs.manifest`, `unigym.packs.resolve` and
  `unigym.packs.load` are deleted; there are no shims.
- `unigym.pack_api.find`, `resolve`, `code_pack_root`, `load_table` and
  `asset_path` are removed.
- `load_mounted_code`, `pack_roots`, `repo_pack`, the `unigym_packs_<name>`
  custom module loader and the `faces:` manifest schema are gone.
- Old `faces:` packs no longer load; a pack must declare the
  `unigym.pack/1` document, and code packs must use a manifest entrypoint that
  accepts one `PackContext`.
- Out-of-tree importers of the removed helpers must migrate to the runtime and
  `PackContext`.

## Non-goals

The current phases deliberately do not implement:

- migration of existing runs and packs: old run configs and old pack manifests
  are not migrated;
- implementing the asset catalog collector in the public core; the external
  `tools/build_asset_catalog.py` and the explicit collector contract of
  `catalog build` / `catalog search` remain unchanged;
- a compatibility shim between the runtime and the deleted legacy loader;
- requirement or `requires` resolution;
- provider registration orchestration beyond manifest-declared `kind: adapter`
  and `kind: robot` items; other provider kinds stay catalog metadata;
- changes to benchmark, scene, asset or task domain parsers;
- a sandbox for trusted code packs.
