# Scene asset contract (file-backed)

Status: stable for the file-backed asset MVP. This contract is neutral: it
sits above the MuJoCo family (`mujoco`, `mjwarp`) and IsaacLab 2
(`isaaclab2`). It does not cover generated MJCF/`AssetDefinition` payloads,
articulations, or production fixture parity.

## 1. `AssetInstance`

One neutral scene declares file-backed assets as `AssetInstance` entries on
`Scene.asset_instances` (`unigym/scenes/scene_ir.py`):

```python
ASSET_INSTANCE_KINDS = ("static_rigid", "dynamic_rigid")

@dataclass(frozen=True)
class AssetInstance:
    id: str
    asset: AssetRef                 # unigym.assets.asset_ref.AssetRef
    pose: Pose
    source: str                     # mandatory "file:line" provenance
    kind: str = "static_rigid"
    collision: bool = True
```

| Field | Type | Rule |
| --- | --- | --- |
| `id` | `str` | Neutral snake-case name; unique within the scene and disjoint from every `Instance.id`. |
| `asset` | `AssetRef` | The IR does not coerce mappings; authoring edges call `AssetRef.from_mapping`. At least one of `mjcf` / `urdf` / `usd` must be non-empty. |
| `pose` | `Pose` | Initial placement only. `pose.frame` must be `"world"`; slot/region frames are refused. |
| `source` | `str` | Mandatory `file:line` (or `scene:<provider>#<id>`) provenance. |
| `kind` | `str` | Closed role: `"static_rigid"` or `"dynamic_rigid"`. There is no second free-form role. |
| `collision` | `bool` | Exact `bool`; `0` / `1` / truthy objects are refused. The only per-instance geometry toggle. |

Validation happens at construction and again at the consumer boundary
(`validate_asset_scene`, called by `validate_binding`). A malformed field is a
construction `ValueError`; the consumer re-raises it as `SceneBindingError`.

`Scene.materials` stays legal and never binds asset visuals. MuJoCo keeps its
terrain composition; IsaacLab refuses `scene.terrain` on an asset scene. With
asset instances present, `Scene.regions`, `Scene.slots` and
`engine_hints["regions"]` are `SceneBindingError`s: they are never silently
ignored.

## 2. One rigid root body

Each asset instance is **one rigid root body**:

- static rigid: exactly one root body with collision/visual geometry and no
  joint;
- dynamic rigid: exactly one root body with no actuation and no internal
  joints. On MuJoCo its only joint is one root freejoint. On IsaacLab it is a
  passive rigid root; a jointless URDF is allowed, an articulation root is
  refused.

The `pose` is initial placement only. Static assets do not move; dynamic
assets move under physics but are not re-placed on reset. There is no
prop-manager lifecycle, reset state, action face, contact group, sensor target
or `object/<name>` record key for a scene asset.

### Pose placement

- **MuJoCo**: the authored root body keeps its local pose verbatim. The
  instance pose lands on a temporary attach frame
  (`Scene_<scene>_<id>_attach`); the frame dissolves at compile, so the body
  compiles at the declared world pose. A dynamic root freejoint follows that
  frame.
- **IsaacLab**: the declared pose becomes the field's `init_state.pos` /
  `init_state.rot`.

## 3. Per-environment clone semantics

The engines do not share clone semantics:

- **MuJoCo family**: one model/world. Each environment has its own `MjData`,
  and the declared pose is the world pose.
- **IsaacLab 2**: one scene field per asset instance,
  `asset_<scene>_<id>`, at prim path
  `{ENV_REGEX_NS}/Scene_<scene>_<id>`. `InteractiveScene` clones the field once
  per env at the env origin. The declared pose is therefore applied as the
  env-relative init state: env `i` resolves to
  `env_origin_i + pose.xyz`. With one env this equals the world pose. This is
  the same rule the static primitive passthrough uses.

There is no per-env asset resampling, placement initializer, or reset
re-placement.

## 4. Backend matrix

| Asset kind | MuJoCo / mjwarp `.mjcf` | IsaacLab 2 `.usd` | IsaacLab 2 `.urdf` |
| --- | --- | --- | --- |
| `static_rigid` | Supported: exactly one root body, no joints, no worldbody-level geometry, no nested bodies, no actuators/sensors/equalities/tendons. | Supported when no prim recursively authors `PhysicsRigidBodyAPI` or `PhysicsArticulationRootAPI`. | **Refused** with a teaching error. |
| `dynamic_rigid` | Supported: exactly one root body and exactly one joint, a root freejoint. | Supported: `RigidObjectCfg` + `UsdFileCfg`; any `ArticulationRootAPI` is refused. | Best effort: jointless URDF → `RigidObjectCfg` + `UrdfFileCfg`. A declared `<joint>` is refused. |

The backend format preference is declared by the registry:

```python
backend_info("mujoco").asset_formats   # ("mjcf",)
backend_info("mjwarp").asset_formats   # ("mjcf",)
backend_info("isaaclab2").asset_formats  # ("usd", "urdf")
```

`isaaclab3` declares the same asset formats but has no registered asset-scene
engine; file-backed scene composition is an `isaaclab2` capability.

Explicit refusals:

- static URDF on IsaacLab (URDF conversion yields an articulation/anchored
  root);
- MJCF on IsaacLab;
- USD on MuJoCo;
- dynamic URDF with any `<joint>` (articulation-in-scene is unsupported).

`collision: false` disables collision on the realized asset
(`contype=0` / `conaffinity=0` on every geom of the MuJoCo root body;
`CollisionPropertiesCfg(collision_enabled=False)` on IsaacLab). For a dynamic
asset this means the body may fall through the world; that is documented, not
compensated.

### Dynamic URDF is best effort

The engine's Kit-free guard parses the URDF with `xml.etree.ElementTree` and
refuses any declared `<joint>`. A jointless URDF is then built as a
`RigidObjectCfg` (`fix_base=False`, `merge_fixed_joints=False`,
`collision_from_visuals=True`, `make_instanceable=True`, no joint drive). If
IsaacLab's own URDF conversion fails or produces an articulation, that failure
surfaces; the engine does not substitute an articulation. The dog lane accepts
only a `RigidObject` construction or a conversion error whose message names the
URDF/rigid/articulation boundary.

## 5. Resolution

Asset paths resolve once at the composition boundary through
`unigym.assets.plan`:

```python
AssetPlan.resolve(value) -> str
select_asset_format(asset, backend_formats) -> str
resolve_asset_instance(asset, asset_plan, formats) -> tuple[str, str]
```

Rules:

1. `select_asset_format` walks the backend's declared preference order, not the
   `AssetRef` declaration order. There is no conversion and no fallback to a
   different declared format.
2. The selected reference must be a non-empty string and resolve to an
   absolute path.
3. The resolved path must be an existing file; `resolve_asset_instance` raises
   `FileNotFoundError` otherwise.
4. `pack:<pack>/<rel>` references resolve through the owning pack runtime;
   absolute and `~` paths are accepted; a bare relative path is a `PackError`;
   `${UNIGYM_ASSET_ROOT}` and `${UNIGYM_ASSET_PACK}` are teaching
   `PlaceholderError`s.
5. There is no CWD-relative fallback, no environment-variable asset root, and
   no module-level asset runtime. `AssetPlan` is passed explicitly.
6. `describe` never receives the resolved path; only the declared `AssetRef`
   mapping.

### MJCF closure policy

`load_asset_spec` (`unigym/backends/shared/mjfamily/scene/mjcf_io.py`):

- a document without `${` and without `pack:` loads in place through
  `MjSpec.from_file`, preserving ordinary document-relative includes and
  `file=` leaves;
- otherwise the document text resolves once through
  `AssetPlan.resolve_document`; if any `${` or `pack:` marker remains, it
  fails closed;
- a rewritten document loads through `MjSpec.from_string`, which has no
  document base, so every remaining `file=` leaf must be absolute. A relative
  leaf is refused. There is no copying, rewriting, or CWD-relative fallback.

### USD/URDF closure

The engine resolves only the top-level `.usd` / `.urdf` path. Relative USD
references and URDF mesh paths are asset-owned and resolved by the USD stage or
URDF converter at load time. The engine synthesizes no path and claims no
texture/material closure.

## 6. Artifacts and digest

`ASSET_SCENE_DIGEST_VERSION = "unigym.scene.asset/1"` is the neutral asset
slice tag (`unigym/scenes/artifacts.py`). Each engine emits its own top-level
tag; the tags are distinct and are **not** compared for cross-engine
equality:

| Engine | Top-level `digest_version` | Asset slice |
| --- | --- | --- |
| MuJoCo family | `unigym.modelir/2` (plus `base` and `ext.mj`) | `asset_scene: {digest_version: "unigym.scene.asset/1", instances: [...]}` |
| IsaacLab 2 | `unigym.scene.asset/1` (plus `ext.scene`) | same tag and row shape |

The neutral row shape, sorted by id, is:

```json
{
  "id": "cup",
  "kind": "static_rigid",
  "format": "mjcf",
  "asset": {"mjcf": "pack:acme/assets/cup.mjcf"},
  "pose": {
    "xyz": ["0x0.0p+0", "0x0.0p+0", "0x1.0p+0"],
    "quat_wxyz": ["0x1.0p+0", "0x0.0p+0", "0x0.0p+0", "0x0.0p+0"]
  },
  "collision": true,
  "source": "scene.yaml:12"
}
```

- Floats are `float.hex()` strings.
- `asset` carries the declared `AssetRef` fields verbatim. An authored
  absolute reference therefore stays in the digest; use `pack:` references for
  a portable digest.
- `describe` / `describe.json` carries **no resolved path**. `scene.json`
  carries the declared rows. Resolution and payload artifacts may carry the
  resolved absolute path:
  - MuJoCo writes `scene.json`, `resolution.json`, `payload_index.json`,
    `composed.xml`, `compile.json` (and `fill.json` only when content ran);
    `resolution.json` / `payload_index.json` are the rows allowed to carry
    `resolved`.
  - IsaacLab 2 writes `scene.json`, `resolution.json`, `payload_index.json`,
    `compose.txt`, `describe.json`; `resolution.json` / `payload_index.json` /
    `compose.txt` may carry the resolved path, `describe.json` never does.
- `to_xml` is artifact/A-B text only; the digest path never consults it.

## 7. Scene-provider routing

A run's `scene_preset` (catalog id or inline provider) is resolved into the
internal `env.params.scene_provider*` keys. The selected backend composes the
neutral scene:

- MuJoCo family: `compose_scene(scene, asset_plan)` attaches asset instances
  after the primitive/prop loop and records `{id: resolved path}` for the
  artifacts;
- IsaacLab 2: `AssetSceneEngine.compose(scene, asset_plan)` builds asset fields
  and rows; `attach(cfg.scene, composed, ...)` assigns the fields **before**
  `InteractiveScene` clones the scene.

An `AssetPlan` is required only when the scene declares asset instances.
Declaring both `static_shapes` and a scene provider is a teaching error
(`static_shapes` + composed scene cannot both be composed). IsaacLab 2 still
rejects `content` (no content router) and the run layer still rejects
`world_base` (`world_base_consumer=False`).

## 8. Unsupported list

The following remain unsupported and fail closed rather than being dropped:

- generated MJCF / `AssetDefinition` payloads;
- free or anchored articulations, articulation-in-scene, actuated joints;
- texture/material/MDL parity and per-asset material overrides;
- `Scene.regions` / `Scene.slots` / `engine_hints["regions"]` and placement
  initializers / content fill for asset scenes;
- contacts, filters or sensors targeting scene assets;
- `world_base` composition on IsaacLab;
- prop-manager lifecycle and scene-asset reset/state/action surfaces;
- static URDF on IsaacLab; MJCF on IsaacLab; USD on MuJoCo;
- the full robocasa kitchen;
- large-scene N>1 performance parity;
- cross-engine `describe` digest equality.

## 9. Teaching failure matrix

Construction / consumer:

| Condition | Failure |
| --- | --- |
| `asset` is not an `AssetRef` | `ValueError`: `asset instance '<id>': asset must be an AssetRef` |
| `AssetRef` declares no format | `ValueError`: names the allowed `mjcf`/`urdf`/`usd` set |
| `pose` is not a `Pose` | `ValueError`: `pose must be a Pose` |
| non-`world` pose frame | `ValueError`: `pose frame '<frame>' is not 'world'` |
| missing/malformed `source` | `ValueError`: mandatory `file:line` provenance |
| unknown `kind` | `ValueError`: names `("static_rigid", "dynamic_rigid")` |
| non-bool `collision` | `ValueError`: `collision must be a bool` |
| duplicate asset id / collision with `Instance.id` | `ValueError`: duplicate or one-namespace message |
| asset instance contract or regions/slots/`engine_hints["regions"]` at the consumer | `SceneBindingError` |

Resolution:

| Condition | Failure |
| --- | --- |
| no format declared or no compatible backend format | `AssetRefError` naming the declared and supported sets |
| empty/whitespace-only selected reference | `AssetRefError` |
| resolution yields a relative path | `AssetRefError`; no CWD fallback |
| resolved path is missing | `FileNotFoundError` naming the reference and path |
| bare relative path / illegal `pack:` reference | `PackError` from the pack runtime |
| unresolved `${`/`pack:` leaf after document resolution | `ValueError` from `load_asset_spec` |
| relative `file=` leaf after a `pack:` rewrite | `ValueError` from `load_asset_spec` |
| USD/URDF loader failure | surfaced backend load error |

MuJoCo:

| Condition | Failure |
| --- | --- |
| zero or more than one root body | `ValueError`: expected exactly one root body |
| worldbody-level geometry | `ValueError`: move geometry under the root body |
| nested bodies | `ValueError`: nested bodies are unsupported |
| actuators/sensors/equalities/tendons | `ValueError`: section unsupported |
| static asset declares any joint | `ValueError`: static rigid assets declare no joints |
| dynamic asset is not exactly one root freejoint | `ValueError`: names the joint count/type |
| a hinge/slide/ball joint | `ValueError`: articulation-in-scene unsupported |
| `AssetRef` only declares `usd`/`urdf` | `AssetRefError` from format selection |

IsaacLab 2:

| Condition | Failure |
| --- | --- |
| `AssetRef` only declares `mjcf` | `AssetRefError`; backend formats are `("usd", "urdf")` |
| static USD authors `RigidBodyAPI` or `ArticulationRootAPI` | teaching `ValueError` at `attach`; provide a truly static USD or declare `dynamic_rigid` |
| dynamic USD authors `ArticulationRootAPI` | teaching `ValueError`: articulation-in-scene unsupported |
| static URDF | teaching `ValueError`; provide a static USD |
| dynamic URDF declares `<joint>` | teaching `ValueError`; only a jointless URDF maps to `RigidObjectCfg` |
| dynamic URDF conversion fails or yields an articulation | surfaced IsaacLab failure; no silent substitution |
| `scene.terrain` on an asset scene | teaching `ValueError`: the run terrain owns the ground |
| non-empty `scene.engine_hints` on an asset scene | teaching `ValueError`: MuJoCo-only hint vocabulary |
| field/prim collision | teaching `ValueError` naming the field |
| non-primitive `Instance` | teaching `ValueError`: use `AssetInstance` or MuJoCo |
| `static_shapes` plus a scene provider | teaching `ValueError`; declare one |
| asset scene without an `AssetPlan` | teaching `ValueError`; pass an `AssetPlan` |
| `content` | existing `ConfigError` / `ValueError` (no content router) |
| `world_base` | existing `ConfigError` (`world_base_consumer=False`) |
