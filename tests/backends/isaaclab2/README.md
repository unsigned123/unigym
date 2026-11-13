# tests/backends/isaaclab2

Backend-local tests for the IsaacLab 2.3.2 backend.

These tests run only in the clean dog lane:

```text
UNIGYM_ISAACLAB2_ENV=unigym-isaaclab232
UNIGYM_TEST_ENV_STRICT=1
UNIGYM_ISAACLAB2_ASSET_CACHE=<provisioned directory>
UNIGYM_ISAACLAB2_G1_ASSET=<provisioned 43-DoF g1.usd>
```

The directory is outside default `testpaths`; run it explicitly with the
IsaacLab 2 environment. The clean lane is recorded in
`plans/artifacts/isaaclab-2.3.2-env.md` in the planning repository.

## Asset cache

The Cartpole fixture reads its USD from `UNIGYM_ISAACLAB2_ASSET_CACHE`; the
directory must contain `cartpole.usd`. The G1 builder fixture reads
`UNIGYM_ISAACLAB2_G1_ASSET`; it must name the 43-DoF with-hand `g1.usd` root
whose `configuration/` folder sits beside it. Tests never download assets. A
missing cache or G1 asset fails the lane (strict); it is not skipped.

```bash
OMNI_KIT_ACCEPT_EULA=YES \
UNIGYM_ISAACLAB2_ASSET_CACHE=<provisioned directory> \
UNIGYM_ISAACLAB2_G1_ASSET=<provisioned 43-DoF g1.usd> \
UNIGYM_TEST_ENV_STRICT=1 \
conda run --no-capture-output -n unigym-isaaclab232 \
  python -m pytest tests/backends/isaaclab2 -q
```

## Contact inventory and ``contacts: all``

Contact sensors must exist before the scene clone. `unigym/backends/isaaclab2/usd.py`
is the lazy OpenUSD inventory owner: it walks the robot asset root, returns the
robot-relative rigid-body paths in traversal order, flags collision coverage,
and excludes links merged by a fixed joint (the G1 palm links). `contacts: all`
materializes one `ContactSensorCfg` per inventory body and binds one row per
body; collisionless bodies stay as zero rows. There is no wildcard all sensor.

Planned coverage:

- backend construction and `EnvBackend` contract;
- IsaacLab scene and articulation setup;
- named-action step / obs / extra;
- reset subset and reset-all;
- model IR topology;
- action-face and kind failure paths;
- close lifecycle;
- G1 contact-all per-body sensor rows (nested finger links, merged palms
  excluded).

No public-core asset pack is used. The test plan defines the allowed
IsaacLab-provided asset boundary and any `external_asset`/`sim` markers.
