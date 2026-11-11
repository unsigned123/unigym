# Pack API Phase 1: manifest schema and catalog

Status: Phase 1 contract only. The legacy `unigym.packs` discovery and loader
remain the active runtime path. Nothing in this document switches discovery,
imports pack code or changes how packs are mounted today.

Parent plan: the "unigym external pack API" draft and its Phase 1 brief in the
planning repository (`plans/drafts/unigym-pack-api.md`,
`plans/drafts/pack-api-phase1-contract.md`).

## Public imports

The Phase 1 contract is exported from `unigym.pack_api`:

```python
from unigym.pack_api import (
    CatalogEntry,
    PackCatalog,
    PackManifestV1,
    ProvidedItem,
    load_manifest_v1,
)
```

Existing legacy exports (`find`, `resolve`, `register_scene`,
`register_termination`, ...) remain unchanged.

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
| `requires` | string to string mapping; stored, not evaluated in Phase 1 |
| `shadow` | boolean, default `false` |

`requires` records the compatibility range a pack intends. Requirement
resolution belongs to a later loader phase; Phase 1 neither resolves nor
enforces it, including `requires.unigym`.

## Provided items

The Phase 1 face vocabulary is explicit:

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
are rejected in Phase 1.

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
  Trusted pack Python is normal Python; Phase 1 only validates the string and
  never imports it.

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
- the catalog never discovers, scans or imports packs. Construct it from
  manifests the caller already loaded.

## Non-goals (Phase 1)

Phase 1 deliberately does not implement:

- discovery, root scanning or entry-point enumeration;
- activation, provider execution or registry mutation;
- requirement or `requires` resolution;
- migration or removal of the legacy scene/benchmark/asset loaders and root
  mappings;
- replacement of the legacy `unigym.packs` manifest contract;
- the installed-PyPI `unigym.packs` entry-point package form;
- domain schemas for each provided face.

The legacy loader stays active until the Phase 2 discovery/catalog migration.
