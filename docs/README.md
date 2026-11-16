# Documentation

The full documentation is being rewritten.
English and Chinese versions will live under `docs/en/` and `docs/zh/`.
See the root README for install and usage.

Process documents:

- [Engineering workflow](development.md) - workflow gates, roles, review and
  reporting.
- [Testing policy](testing.md) - test layers, markers, operations and red-test
  protocol.
- [Scene asset contract](contracts/scene_asset_contract.md) - file-backed
  `AssetInstance` fields, backend matrix, resolution, digest and refusals.
- [Report templates](templates/) - design brief, completion, review and blocked
  reports.

Usage documents:

- [Actions](actions.md) - joint-addressed runtime values, authoring templates,
  per-joint routing/coverage and policy ownership rules.
- [State-write contract](state_write_contract.md) - optional world-frame,
  wxyz, name-addressed root/joint writes and settle ordering.
- [Simulation facade](sim_facade.md) - experimental `unigym.sim` usage, scope,
  error taxonomy and residual limits.
