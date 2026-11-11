# Pack API: manifest schema, catalog and runtime

Status: Phase 1, Phase 3, Phase 3b, the additive Phase 4 `(face, id)` lookup
and Phase 5 runtime parity are implemented. The runtime is a parallel path: the
legacy `unigym.packs` discovery and loader remain the active runtime, and nothing
in this document switches the legacy loader, mounts domain providers or migrates
scene/run/asset loading.

Parent plan: the "unigym external pack API" draft and its phase briefs in the
planning repository (`plans/drafts/unigym-pack-api.md`,
`plans/drafts/pack-api-phase1-contract.md`,
`plans/drafts/pack-api-phase3-discovery.md`,
`plans/drafts/pack-api-phase3b-sources.md`,
`plans/drafts/pack-api-phase4-resource-lookup.md`,
`plans/drafts/pack-api-phase5-runtime-parity.md`).

## Public imports

The pack contract is exported from `unigym.pack_api`:

```python
from unigym.pack_api import (
    ActivationHandle,
    CatalogEntry,
    PackCatalog,
    PackContext,
    PackManifestV1,
    PackRuntime,
    ProvidedItem,
    discover,
    load_manifest_v1,
)
```

Existing legacy exports (`find`, `resolve`, `register_scene`,
`register_termination`, ...) remain unchanged. Importing `unigym.pack_api`
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
| `kind` | optional non-empty string naming a provider kind, e.g. `asset_generator` |

An item with neither locator or with both locators is rejected. An
`entrypoint` on any item requires the manifest `trust: code`. Unknown item keys
are rejected.

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

- a content pack, or a code pack with no manifest entrypoint, returns a
  no-op `ActivationHandle(pack=..., entrypoint=None)`;
- a code pack's `entrypoint` is split into `module:attribute`, imported with
  normal `importlib`, and called with one `PackContext` argument;
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

Activation does not register providers or mutate a registry in this phase; a
pack entrypoint may use the existing public registration helpers directly if
it needs to.

## Non-goals

The current phases deliberately do not implement:

- switching the legacy `unigym.packs` loader onto this runtime; the legacy
  loader remains the active discovery path;
- migration of existing runs and packs: old run configs and old pack manifests
  are not migrated, and no legacy loader is switched by this phase;
- a compatibility shim between the runtime and legacy mounts;
- requirement or `requires` resolution;
- provider registration orchestration; `activate_all` only calls entrypoints
  in catalog order and does not register providers;
- migration of scene, run, task, asset, benchmark or termination loaders onto
  pack resources;
- a sandbox for trusted code packs.

The legacy loader stays active until a later discovery/catalog migration.
