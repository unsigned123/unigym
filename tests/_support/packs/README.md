# Pack fixture packs

Self-contained `unigym.pack/1` fixtures used by the pack API contract tests.
They are data, not tests: this directory is outside every collected test layer
and contains no `test_*.py` files.

## `acme_content`

A `trust: content` pack with one item on each of the `runs`, `scene_presets` and
`assets` faces. The run and asset resources are tiny placeholders; the
`scene_presets/tabletop.yaml` document is a real content-only
`unigym.scene_preset/1` preset (world asset, static `objects.types`, and
per-backend `backend_params`). It declares no provider and no content-fill keys.

`worlds/table.xml` is the minimal MJCF the preset world path names, so a
`pack:` world reference can be expanded by the run layer.

## `acme_code`

A `trust: code` pack with a manifest `entrypoint` (`acme_pack:register`) and two
`providers` items: `acme_code.dynamic_assets` (kind `asset_generator`, using the
same entrypoint) and `acme_code.table_top` (kind `scene`, entrypoint
`acme_pack:table_top`, with an inline `params` schema). The scene provider's
fixture factory returns a neutral `Scene`; activation registers it under the
item id. `python/acme_pack/__init__.py` also holds a harmless one-argument
`register(context)` callable.

Manifest and catalog loading must never import `acme_pack`: the module is
deliberately outside the import path, and `tests/contract/test_pack_fixtures.py`
asserts it stays absent from `sys.modules` after both fixtures load.

## Rules

- Fixtures are self-contained: no external assets, downloads or machine-local
  paths. Resource references stay pack-relative and must not use `..`.
- The code fixture is trust-`code` data. Phase 3 uses it for explicit static-root
  discovery and no-argument activation; wheel/entry-point tests arrive in a
  later phase and may reuse it.
- Keep the fixtures small. Do not copy real asset or benchmark packs here.
