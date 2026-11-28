# tests/backends/isaaclab2

Backend-local tests for the IsaacLab 2.3.2 backend.

These tests run only in the clean dog lane:

```text
UNIGYM_ISAACLAB2_ENV=unigym-isaaclab232
UNIGYM_TEST_ENV_STRICT=1
```

The environment also needs the `unigym-test-packs` plugin and the
`unigym-test-pack-isaaclab-cartpole` and `unigym-test-pack-g1` wheels installed.

The directory is outside default `testpaths`; run it explicitly with the
IsaacLab 2 environment. The clean lane is recorded in
`plans/artifacts/isaaclab-2.3.2-env.md` in the planning repository.

## Test packs

The Cartpole fixture resolves `isaaclab_cartpole.usd` through
`pack_runtime("isaaclab_cartpole")`; the pack ships the USD and its `Props/`
companion. The G1 fixture resolves `g1.usd` through `pack_runtime("g1")`; the
pack ships the complete `configuration/**` tree. Tests never read an asset
environment variable and never download assets. A missing Cartpole or G1 pack
fails the lane (strict); it is not skipped.

The same Cartpole pack also ships the tiny file-backed asset-scene fixtures:
`assets/static_cup.usda` (collision-only, no rigid-body/articulation schema),
`assets/dynamic_cube.usda` (`PhysicsRigidBodyAPI` plus collision and mass),
`assets/jointless_cube.urdf` (one link, no `<joint>`) and
`assets/jointed_cube.urdf` (the articulation-in-scene refusal). The asset-scene
tests resolve them through
`pack_runtime("isaaclab_cartpole").resolve("assets", <id>)` and pass
`AssetPlan(runtime)` to `IsaaclabEnv`.

```bash
conda run --no-capture-output -n unigym-isaaclab232 \
  python -m pip install --no-deps \
  <wheelhouse>/unigym_test_packs-*.whl \
  <wheelhouse>/unigym_test_pack_isaaclab_cartpole-*.whl \
  <wheelhouse>/unigym_test_pack_g1-*.whl

OMNI_KIT_ACCEPT_EULA=YES \
UNIGYM_TEST_ENV_STRICT=1 \
conda run --no-capture-output -n unigym-isaaclab232 \
  python -m pytest --strict-test-packs tests/backends/isaaclab2 -q
```

`test_asset_scene_kit.py` covers N=1/N=2 static and dynamic USD construction,
the cloned prim paths and poses, dynamic gravity motion through the
`RigidObject` data face, static/jointed URDF refusal, the jointless-URDF best
effort, the five build artifacts and the clean close lifecycle. It does not add
camera or contact managers; the existing manager tests remain that coverage. A
jointless URDF may either construct or fail during IsaacLab's URDF conversion;
the test accepts only an outcome whose error names the URDF/rigid/articulation
boundary. That conversion outcome is the dog-lane risk the fast lane cannot
prove.

## Contact inventory and ``contacts: all``

Contact sensors must exist before the scene clone. `unigym/backends/isaaclab2/body_inventory.py`
is the lazy OpenUSD authored-inventory owner: it walks the robot asset root,
returns the robot-relative rigid-body entries in traversal order and flags
collision coverage. It is asset inspection only: a fixed joint records
`merged_into` as diagnostic metadata but never removes the link (the G1 palms
stay authored bodies). At bind, `contacts: all` resolves against the measured
live PhysX bodies (`robot.data.body_names`, 46 including both palms) and uses
the authored inventory only to discover the per-parent sensor prim scopes
(`Robot/(.*)`, `Robot/left_hand/(.*)`, `Robot/right_hand/(.*)`). Each group
binds one row per live body; collisionless bodies stay as zero rows. The
sensor coverage is asserted set-equal to the live body set.

`robot.data.body_names` is the single live runtime body source for IsaacLab2
and IsaacLab3: `articulation.body_names` is only a convenience over the same
PhysX view and `asset.body_names` is never a live fallback. An absent or empty
live `data.body_names` fails fast in the runtime body table, the runtime query
and the env-site resolver (the kit-free `tests/unit/backends/test_isaaclab_bodies.py`
covers the env-site fail-fast). The authored USD inventory is used only for
asset inspection and the pre-clone sensor scopes above.

Planned coverage:

- backend construction and `EnvBackend` contract;
- IsaacLab scene and articulation setup;
- named-action step / obs / extra;
- reset subset and reset-all;
- model IR topology (the runtime body universe, palms included);
- action-group and kind failure paths;
- close lifecycle;
- G1 contact-all per-body sensor rows (nested finger links and both palms,
  unioning to the live 46-body set);
- two real articulations (Cartpole + G1) with one contact group per
  articulation: per-articulation `all` inventories and `Robot` /
  `Robot_<name>` prim scopes, never a global union.

Cartpole is provided by the installed `isaaclab_cartpole` test pack. The test
plan defines the allowed IsaacLab-provided asset boundary and any
`requires_test_packs`/`sim` markers.
