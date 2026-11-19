# isaaclab3_newton backend lane

P0 scaffolding for the IsaacLab 3 Newton backend identity. The lane currently
owns environment and stack probes only; the backend implementation and the
capability/parity tests land in P3 (see `docs/designs/isaaclab3-backend.md`).

The lane sets `OMNI_KIT_ACCEPT_EULA=YES` in `conftest.py` because the rc1
launcher requires non-interactive EULA acceptance. Newton runs inside the Kit
lane in this slice; kitless is a follow-up.

Run it with a provisioned conda environment:

```bash
UNIGYM_ISAACLAB3_NEWTON_ENV=<env> python tools/ci/run_lane.py --lane isaaclab3_newton
```
