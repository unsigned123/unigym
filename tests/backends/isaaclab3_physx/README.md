# isaaclab3_physx backend lane

Lane-tested and nightly-enabled `isaaclab3_physx` backend identity. This is a
canonical backend name; `isaaclab3` and `isaaclab` are not aliases for it. The
lane owns real Cartpole fixtures plus environment/stack probes. P2 declares
a capability only after its lane test passes: `CONTACTS_GROUPS` is proven by
`test_contacts.py`, and the kinematic director traits
(`DIRECTOR_TELEPORT`, `DIRECTOR_SETVEL`, `DIRECTOR_ATTACH_SLAVE`,
`DIRECTOR_GRIPPER`) are proven by `test_director.py`, which drives the neutral
`Director` through its public `handle()` face. `test_cameras.py` proves a
global world rig (RGB + metric `distance_to_camera` depth), a per-env body rig
(RGB), and a per-instance world rig whose env 1 state row is read back
separately. The same test writes a captured frame to a non-empty MP4. The
camera/recorder traits (`CAMERAS_RGB`, `CAMERAS_DEPTH_RIG`, `CAMERAS_GLOBAL`,
`CAMERAS_PER_ENV`, `CAMERAS_PER_ENV_STATE`, `RECORDER_VIDEO`) are therefore
declared and `BackendCapabilities.cameras=True` / `recorder=True`. The lane
environment installs the `imageio[ffmpeg]` extra. The depth RIG is proven, but
the `DEPTH_OBS_ALIASES` bit stays deferred (`depth_camera=False`): no lane test
exercises the `depth` alias keys, and an observation request that selects a
depth stream is gate-rejected until that alias bit lands. `test_props.py` and
`test_state_write.py` verify a rigid box prop and the free-root `set_root_state`
/ partial-reset path, but the `props` capability is intentionally NOT declared
yet: the shared `props` bit projects all three `PROPS_*` traits at once, while
only the rigid tier is lane-proven. Articulated props remain undeclared until
their tests land; the earlier CoM DR write declares `EVENTS_DR`; its lane coverage is in
`test_events.py` (see `docs/designs/isaaclab3-backend.md`).

`test_events.py` proves a reset-mode absolute-CoM event applied at construction,
so the pre-existing `EVENTS_DR` declaration is lane-backed. `test_terrain.py`
proves both the flat `ground` preset and the steps mesh `terrain` preset
construct and step. Newton keeps `EVENTS_DR` undeclared until its P3 lane
coverage.

The lane declares `--strict-test-packs` because the contact, prop, director,
camera, event and terrain tests consume the `isaaclab_cartpole` test pack.

The lane launches Kit with `enable_cameras=True` in `conftest.py` because the
camera tests construct real RTX sensors. This is a session-wide offscreen
render path, so the other lane tests also run with camera extensions loaded.

The lane sets `OMNI_KIT_ACCEPT_EULA=YES` in `conftest.py` because the rc1
launcher requires non-interactive EULA acceptance. The lane runner does not
inject it.

Run it with a provisioned conda environment:

```bash
UNIGYM_ISAACLAB3_PHYSX_ENV=<env> python tools/ci/run_lane.py --lane isaaclab3_physx
```

CI/nightly: the self-hosted `isaaclab3_physx` job is default-skipped unless
`UNIGYM_RUN_ISAACLAB3_PHYSX == 'true'`. It requires the
`UNIGYM_ISAACLAB3_PHYSX_ENV` conda environment, the shared
`UNIGYM_TEST_PACKS_WHEELHOUSE`, and an environment that provides
`imageio_ffmpeg` for the recorder test.
