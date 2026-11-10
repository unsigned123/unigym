# tests/backends/isaaclab2

Backend-local tests for the IsaacLab 2.3.2 backend.

These tests run only in the clean dog lane:

```text
UNIGYM_ISAACLAB2_ENV=unigym-isaaclab232
UNIGYM_TEST_ENV_STRICT=1
```

The directory is outside default `testpaths`; run it explicitly with the
IsaacLab 2 environment. The clean lane is recorded in
`plans/artifacts/isaaclab-2.3.2-env.md` in the planning repository.

Planned coverage:

- backend construction and `EnvBackend` contract;
- IsaacLab scene and articulation setup;
- named-action step / obs / extra;
- reset subset and reset-all;
- model IR topology;
- action-face and kind failure paths;
- close lifecycle.

No public-core asset pack is used. The test plan defines the allowed
IsaacLab-provided asset boundary and any `external_asset`/`sim` markers.
