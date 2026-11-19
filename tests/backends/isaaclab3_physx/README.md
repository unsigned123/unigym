# isaaclab3_physx backend lane

P0 scaffolding for the IsaacLab 3 PhysX backend identity. The lane currently
owns environment and stack probes only; the backend implementation and the
capability tests land in P1/P2 (see `docs/designs/isaaclab3-backend.md`).

The lane sets `OMNI_KIT_ACCEPT_EULA=YES` in `conftest.py` because the rc1
launcher requires non-interactive EULA acceptance. The lane runner does not
inject it.

Run it with a provisioned conda environment:

```bash
UNIGYM_ISAACLAB3_PHYSX_ENV=<env> python tools/ci/run_lane.py --lane isaaclab3_physx
```
