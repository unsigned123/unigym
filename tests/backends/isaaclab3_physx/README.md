# isaaclab3_physx backend lane

P2-in-progress lane for the `isaaclab3_physx` backend identity. This is a
canonical backend name; `isaaclab3` and `isaaclab` are not aliases for it. The
lane owns real Cartpole fixtures plus environment/stack probes. P2 declares a
capability only after its lane test passes: `CONTACTS_GROUPS` is proven by
`test_contacts.py`; props, director and cameras remain undeclared until their tests land;
the earlier CoM DR write declares `EVENTS_DR`, and its lane coverage lands in
P2-E (see `docs/designs/isaaclab3-backend.md`).

The lane declares `--strict-test-packs` because the contact test consumes the
`isaaclab_cartpole` test pack.

The lane sets `OMNI_KIT_ACCEPT_EULA=YES` in `conftest.py` because the rc1
launcher requires non-interactive EULA acceptance. The lane runner does not
inject it.

Run it with a provisioned conda environment:

```bash
UNIGYM_ISAACLAB3_PHYSX_ENV=<env> python tools/ci/run_lane.py --lane isaaclab3_physx
```
