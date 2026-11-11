# Pack API: manifest schema, catalog and runtime

Status: Phase 1 and Phase 3 contracts are implemented. The runtime is a
parallel path: the legacy `unigym.packs` discovery and loader remain the active
runtime, and nothing in this document switches discovery, mounts domain
providers or migrates scene/run/asset loading.

Parent plan: the "unigym external pack API" draft and its phase briefs in the
planning repository (`plans/drafts/unigym-pack-api.md`,
`plans/drafts/pack-api-phase1-contract.md`,
`plans/drafts/pack-api-phase3-discovery.md`).

## Public imports

The pack contract is exported from `unigym.pack_api`:

```python
from unigym.pack_api import (
    ActivationHandle,
    CatalogEntry,
    PackCatalog,
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

shadow: false
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
| `shadow` | boolean, default `false` |

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
```

`CatalogEntry` carries `pack`, `face`, `id`, `resource`, `entrypoint` and
`kind`.

Catalog rules:

- pack order and per-pack item order are preserved;
- pack names must be unique;
- a `(face, id)` pair must be unique across the whole catalog;
- manifests and returned tuples are immutable from the caller's perspective;
- `get` and an unknown `entries(face=...)` raise `unigym.packs.PackError`;
- the catalog never discovers, scans or imports packs.

## Runtime

`discover` builds a runtime from explicit, ordered roots:

```text
discover(roots) -> PackCatalog + root_by_pack
  -> runtime.catalog          enumerate packs and provided items
  -> runtime.resource(...)    resolve one pack-relative resource
  -> runtime.activate(...)    import a code pack's entrypoint once
```

```python
from unigym.pack_api import discover

runtime = discover(["/packs/acme_code", "/packs/acme_content"])
runtime.catalog.packs()                 # ("acme_code", "acme_content")
runtime.resource("acme_content", "runs/demo.yaml")
runtime.activate("acme_code")           # ActivationHandle
```

Discovery rules:

- only the roots the caller passes are read; there is no implicit environment,
  repository, YAML or entry-point lookup in this phase;
- root order is preserved and canonical duplicate roots are visited once;
- every root must contain `unigym.pack.yaml`, validated by
  `load_manifest_v1`;
- duplicate pack names fail;
- discovery reads manifests and checks resource paths only. It never imports
  pack Python.

`PackRuntime` is an immutable catalog plus a `pack -> root` mapping. The
constructor rejects a mapping that does not name exactly the catalog's packs.

### Resources

`runtime.resource(pack, rel)` returns an absolute `Path` inside that pack's
root. It applies the same rules as the schema: `rel` must be non-empty and
relative, must not contain `..`, must resolve inside the root and must exist.
Errors name both the pack and the path.

### Activation

`runtime.activate(pack)` is explicit and cached per runtime:

- a content pack, or a code pack with no manifest entrypoint, returns a
  no-op `ActivationHandle(pack=..., entrypoint=None)`;
- a code pack's `entrypoint` is split into `module:attribute`, imported with
  normal `importlib`, and called with no arguments;
- the pack root's `python/` directory is the explicit import boundary. It is
  added to `sys.path` once before the import and kept there because the pack
  may import its own submodules later;
- there is no AST restriction, custom module loader or sandbox. `trust: code`
  means trusted Python that may import `unigym.pack_api`;
- repeated activation returns the same cached handle and does not re-execute
  the entrypoint;
- an activation failure raises `PackError` naming the pack and entrypoint,
  with the original exception as its cause.

Activation does not register providers or mutate a registry in this phase; a
pack entrypoint may use the existing public registration helpers directly if
it needs to.

## Non-goals

The current phases deliberately do not implement:

- legacy discovery migration, root scanning by the old loader, or an
  environment/YAML/entry-point discovery switch;
- installed-PyPI `unigym.packs` entry-point or wheel discovery;
- a compatibility shim between the runtime and legacy mounts;
- requirement or `requires` resolution;
- provider registration orchestration, shadow/conflict policy or activation
  ordering;
- migration of scene, run, task, asset, benchmark or termination loaders onto
  pack resources;
- a sandbox for trusted code packs.

The legacy loader stays active until a later discovery/catalog migration.
