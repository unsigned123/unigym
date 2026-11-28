# Registration phase contract

Every pack-registerable registry is protected by one process-global,
monotonic registration phase.  The authoritative implementation is
`unigym.registration` (standard library only).

## Phases

```text
MANIFEST -> ACTIVATE -> SPEC -> ASSEMBLE -> BACKEND_BUILD -> RUNTIME
```

- Contexts may only enter forward; entering the current phase is a no-op.
- Bootstrap (`phase is None`) allows core built-in imports.  The first
  explicit phase entered by a run takes over.
- `RUNTIME` is terminal and sticky: after a live `RunInstance` is built no
  registry may mutate.  `RegistrationGate.seal_runtime()` records it.
- Tests reset the gate through the autouse fixture in `tests/conftest.py`.

## Face windows

| Face | Earliest | Latest |
|---|---|---|
| `RUN_CONFIG` | `MANIFEST` | `MANIFEST` |
| `SCENE_PRESET` | `MANIFEST` | `SPEC` |
| `ASSET` | `MANIFEST` | `SPEC` |
| `SELECTION` | `MANIFEST` | `MANIFEST` |
| `SCENE_PROVIDER` | `ACTIVATE` | `ACTIVATE` |
| `SCENE_PROVIDER_INLINE` | `SPEC` | `SPEC` |
| `SCENE_CONTENT_FILLER` | `ACTIVATE` | `ACTIVATE` |
| `PROP_TEMPLATE` | `ACTIVATE` | `ACTIVATE` |
| `ROBOT` | `ACTIVATE` | `ACTIVATE` |
| `ADAPTER` | `ACTIVATE` | `ACTIVATE` |
| `TERMINATION` | `ACTIVATE` | `ACTIVATE` |
| `INITIALIZER` | `ACTIVATE` | `BACKEND_BUILD` |
| `EXECUTOR` | `ACTIVATE` | `SPEC` |
| `ACTUATOR_LAW` | `MANIFEST` | `SPEC` |
| `ACTUATOR_IMPLEMENTATION` | `ASSEMBLE` | `BACKEND_BUILD` |

`ACTUATOR_IMPLEMENTATION` is additionally narrowed by the backend policy:

| Backend | Window |
|---|---|
| `mujoco` / `mjwarp` | `BACKEND_BUILD .. BACKEND_BUILD` |
| `isaaclab2` | `ASSEMBLE .. ASSEMBLE` |
| `isaaclab3_physx` / `isaaclab3_newton` | `ASSEMBLE .. ASSEMBLE` |

A backend policy may only narrow a global window, never widen it.  The
canonical mapping lives in `unigym.registration.BACKEND_REGISTRATION_POLICIES`
(registration-owned pure policy data; no backend graph).

## Key API

```python
from unigym.registration import (
    BackendRegistrationPolicy,
    PackFace,
    PhaseWindow,
    RegistrationGate,
    RegistrationPhase,
    require_registration,
)
```

`require_registration(face, name=..., backend=...)` is called at the entry of
every public `register_*` helper.  Its error names the face, the entry, the
current phase, the global window, any backend narrowing and a migration hint.

## Rules for registry owners

- Keep the gate call before any validation that mutates process state.
- Never catch `RegistrationError`/`RegistrationSealedError` and continue.
- A registry that is snapshotted during pack activation must restore values
  and origins together; `RegistrySnapshot` is the shared helper.
- Actuator law descriptors are pure data; implementation factories are
  imported only when materialized at the backend's legal phase.
