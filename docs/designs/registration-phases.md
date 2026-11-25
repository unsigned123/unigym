# Registration phases and the actuator law/implementation split

## Why phases

Pack registration used to be a set of independent process-global registries
with only local duplicate checks.  Nothing prevented a pack from mutating an
adapter, robot or actuator table after a run had already been built, which
made provenance, rollback and reproducibility dependent on call order.

The registration phase contract makes the lifecycle explicit:

```text
MANIFEST -> ACTIVATE -> SPEC -> ASSEMBLE -> BACKEND_BUILD -> RUNTIME
```

Each face declares the window in which registration is legal
(`docs/contracts/registration_phase_contract.md`).  The gate is monotonic and
`RUNTIME` is sticky: a live run seals the process against further registry
mutation.

## Where phases are entered

- `Run._build_spec` enters `MANIFEST`, then activates the pack runtime
  (manifest providers register under `ACTIVATE`).
- `build_spec` advances to at least `SPEC`.
- `build_instance` advances to `ASSEMBLE`.
- `_build_env_and_sampler` advances to `BACKEND_BUILD`.
- `build_instance` calls `seal_runtime()` immediately before returning the
  `RunInstance`.
- `PackRuntime.activate` advances to at least `ACTIVATE` around the
  entrypoint and manifest provider registration.  An explicitly injected
  runtime is activated when the config needs a pack.

## Actuator split

The actuator surface is split into two registries:

1. **Law descriptor** (`unigym.actuator_law`): pure stdlib data, available by
   `SPEC`.  It declares config keys, gain keys, required gains, parts support,
   the neutral drives it serves, backend capabilities, the compiled plant a
   backend must expose, backend-owned limits, command support and whether a
   backend implementation is required.  An `ActuationGroup` carries a frozen
   `params` mapping whose keys are validated against the descriptor's
   `config_keys`; `gains`/`parts` remain the shorthand authoring faces and the
   merged payload reaches every factory through
   `ActuatorBuildContext.params`.
2. **Implementation** (`unigym.actuators.implementations`): a late-bound
   factory keyed by `(law, backend)`.  MuJoCo/mjWarp factories return an
   `ActuatorBase` and are materialized at `BACKEND_BUILD`; IsaacLab2/3
   factories return a native cfg object or mapping and are materialized at
   `ASSEMBLE`.

The five built-in laws keep their concrete MuJoCo classes in
`ACTUATOR_REGISTRY` (the compatibility table) and register core descriptors in
`unigym.actuators.descriptors`.  The neutral IR derives its law/gain/parts
validation, drive matrix and `BACKEND_LAW_CAPABILITIES` compatibility view from
the descriptors; the hardcoded law/plant tables are retired.

## Pack manifest

The `unigym.pack/1` schema is unchanged.  Two additive provider kinds are
accepted on the `providers` face:

```yaml
providers:
  - id: acme.compliant_pd
    kind: actuator_law
    resource: actuators/compliant_pd.law.yaml

  - id: acme.compliant_pd.mujoco
    kind: actuator_implementation
    entrypoint: acme_actuators:make_mujoco
    params:
      law: acme.compliant_pd
      backend: mujoco
      kind: mujoco_ctrl
```

Rules:

- `actuator_law` is resource- or entrypoint-backed, never `params`; the
  descriptor name must equal the item id; entrypoints require `trust: code`.
- `actuator_implementation` is entrypoint-only, with flat `law`/`backend`
  (and optional `kind`) params.
- The implementation's law must be owned by the same pack.
- The backend must be canonical and the pair is unique at materialization.
- Implementation modules are not imported during `ACTIVATE` or `SPEC`.

## Provenance and rollback

Pack activation snapshots and restores every registry it can mutate: adapters,
robots, scene providers, content fillers, prop templates, terminations,
actuator laws and initializers.  Values and origins are restored together
through `RegistrySnapshot`.

## Non-goals

- no `unigym.pack/2` schema bump;
- no pack dependency graph or topological ordering;
- no runtime actuator hot-swap;
- no executor registry behaviour beyond its declared window;
- no changes to the existing actuator formulas.
