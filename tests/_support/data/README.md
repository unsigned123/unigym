# tests/_support/data

Shared, self-contained test assets. This directory is not collected.

## `tiny_arm.xml`

Purpose: one small, deterministic MuJoCo model for backend-local tests, so no
test depends on the production `assets/` tree, a benchmark pack, a wheel data
file, an external download, or a machine path.

Model: a free base body (`base`) with two hinge joints (`shoulder`, `elbow`)
and two torque motors (`shoulder_motor`, `elbow_motor`). Joint ranges are
declared in radians (`<compiler angle="radian"/>`) and the compiled order is
bodies `world, base, link, tip`, joints `root, shoulder, elbow`, actuators
`shoulder_motor, elbow_motor`. It exists purely to exercise the MuJoCo
backend's construction, action routing, limits/permutation and reset paths.

Provenance: original to this repository, authored for the pytest test suite.
It is not derived from any robot vendor file, benchmark asset or third-party
model.

Constraints (keep them when editing):

- MuJoCo primitives only (plane/box/capsule here);
- no `mesh`, `texture`, `hfield`, `include`, `file=` or network reference;
- no machine-specific paths;
- small enough for CPU-only, sub-second test construction.

License: MIT, matching the repository `LICENSE`.

Consumers:

- `tests/conftest.py` exposes it as the `tiny_mjcf` fixture, which copies the
  file into each test's `tmp_path` so tests can write artifacts beside it
  without touching the tracked asset;
- `tests/backends/mujoco/` builds a real `MujocoEnv` from the fixture.
