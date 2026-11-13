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

## `tiny_arm_velocity.xml`

Purpose: the same small MuJoCo model with compiled velocity servos, so the
`direct_velocity` law is proved against a real native velocity actuator rather
than a torque motor or a software PD approximation.

Model: a free base body (`base`) with two vertical-axis hinge joints
(`shoulder`, `elbow`) and two velocity servos (`shoulder_velocity`,
`elbow_velocity`). The horizontal links and wide joint ranges keep a short
velocity-tracking run free of gravity torque and joint-limit collisions. It is
primitive-only and self-authored.

Consumers:

- `tests/conftest.py` exposes it as the `tiny_velocity_mjcf` fixture;
- the MuJoCo and MuJoCo Warp `tiny_env_factory` fixtures accept an `mjcf`
  override and use it for the direct-velocity run and plant-mismatch tests.

License: MIT, matching the repository `LICENSE`.

## `tiny_arm.urdf`

Purpose: one small, deterministic URDF for the generic IsaacLab realizer, so
the URDF `AssetRef` path is exercised without a robot pack, benchmark asset or
machine path.

Model: a fixed-capable base (`base_link`) with two revolute joints (`shoulder`,
`elbow`) and primitive box/cylinder collision and visual geometry. It is
authored only to prove an `ArticulationCfg` builds with the real `UrdfFileCfg`.

Provenance: original to this repository, authored for the pytest test suite. It
is not derived from any robot vendor file, benchmark asset or third-party
model.

Constraints (keep them when editing):

- URDF primitives only (box/cylinder here);
- no `mesh`, `package://`, texture, include or network reference;
- no machine-specific paths;
- small enough for CPU-only, sub-second test construction.

License: MIT, matching the repository `LICENSE`.

Consumers:

- `tests/conftest.py` exposes it as the `tiny_urdf` fixture, which copies the
  file into each test's `tmp_path`;
- `tests/backends/isaaclab2/test_robot_realizer.py` builds the real URDF spawn.
