# shared/isaaclab3_common

Engine-neutral library for the two IsaacLab 3 backend identities
(`isaaclab3_physx`, `isaaclab3_newton`).

Rules:

- This package is a library, not a backend: it never imports a concrete
  backend package or the MuJoCo family.
- It has no module-level engine import (`isaaclab*`, `pxr`, `omni`, `newton`,
  `warp`); function-local lazy imports are allowed behind the shared API.
- PhysX/Newton differences are injected through `EngineProfile`; shared code
  must not branch on `profile.name`.
- The public surface grows only when a second concrete engine needs it (see
  `docs/designs/isaaclab3-backend.md`).
