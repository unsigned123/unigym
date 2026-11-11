# Pack fixture packs

Self-contained `unigym.pack/1` fixtures used by the pack API contract tests.
They are data, not tests: this directory is outside every collected test layer
and contains no `test_*.py` files.

## `acme_content`

A `trust: content` pack with one item on each of the `runs`, `scenes` and
`assets` faces. Each item points at a tiny placeholder YAML resource. The
resources are not full domain schemas; the pack schema only validates the
resource path and its existence, so the run/scene/asset parsers own their own
documents in later phases.

## `acme_code`

A `trust: code` pack with a manifest `entrypoint` (`acme_pack:register`) and a
`providers` item (`acme_code.dynamic_assets`, kind `asset_generator`) that uses
the same entrypoint. `python/acme_pack/__init__.py` holds a harmless
no-argument `register()` callable.

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
