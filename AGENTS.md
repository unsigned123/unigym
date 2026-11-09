# Working on unigym-core

## Type annotation policy

This project targets Python 3.11. Type annotations are part of the public
contract, not decoration. New and modified code must follow these rules.

### 1. Public surfaces are fully typed

- Every public function, method, dataclass field, module-level constant with a
  non-obvious type, `Protocol` member, and `TypedDict` field must have explicit
  parameter and return annotations.
- Public API examples: `Run`, `RunSpec`, `ArticulationSpec`, `ControlGroup`,
  `ControlRuntime`, `NamedAction`, `Adapter`, `EnvBackend`, camera/contact
  protocols, and registry factories.
- Internal local variables are inferred; do not add annotation noise unless the
  type is genuinely non-obvious.

### 2. Use `from __future__ import annotations`

- Non-empty modules that contain annotations must start with
  `from __future__ import annotations` after the module docstring.
- Do not quote annotations for forward references when postponed evaluation is
  enabled. Remove redundant quotes from annotations only; never rewrite runtime
  strings, dict keys/values, `Literal[...]` values, or serialized data.
- Prefer `ruff --select UP037 --fix` for annotation quote removal.

### 3. Forbidden type escapes

Do not use these to make pyright quiet:

- `cast(...)` as a general-purpose escape;
- `object` as a substitute for an unknown type;
- `Any` as a substitute for understanding the data;
- blanket `# type: ignore` or file-level suppression.

`cast` is allowed only at a genuine runtime-validated boundary where the type
system cannot express the relationship. A permitted cast must have:
1. an adjacent runtime check (`isinstance`, validation, or assertion) that
   establishes the invariant, and
2. a short comment explaining why the cast is necessary.

`Any` is allowed only for genuinely arbitrary external payloads (for example an
opaque plugin extension payload) and must carry a comment. Prefer a recursive
`JsonValue` alias over `Any` for raw JSON/config data.

`object` is only acceptable when a value is intentionally opaque to the
consumer and no operation is performed on it. It is never a stand-in for
“unknown”.

### 4. Prefer real types over local narrowing hacks

Use, in order:

1. A concrete type or a common base class (`Mapping`, `Sequence`,
   `CameraRig`, `EnvBackend`, ...).
2. `Protocol` for duck-typed boundaries.
3. `TypedDict` for config/record mappings that are read by key.
4. `JsonValue` for raw JSON/config trees before validation.
5. `TypeGuard` / `isinstance` / `assert` for runtime narrowing.
6. A documented `cast` only when the above cannot express the invariant.

If two values share a base class, annotate the base class rather than a union
of concrete classes. If a value is genuinely dynamic, declare the field or
runtime attribute in its owner instead of casting at every consumer.

### 5. Unknown is not an error to hide

- Pyright `standard` does not report `Unknown` as an error. That is deliberate.
- Do not add `Any`, `object`, or `cast` just to remove an `Unknown`.
- If a type cannot be inferred without an architectural change, leave the
  `Unknown` and report it with its root cause.
- Public API `Unknown` reduction is a separate ratchet (`pyrightconfig.api.json`);
  zero is not required per change, but new public `Unknown` is discouraged.

### 6. Optional dependencies

- Optional imports (`PIL`, `imageio`, `av`, `aiortc`, Isaac Lab, etc.) stay
  lazy at the function boundary or under `TYPE_CHECKING`; they must not become
  base-module imports.
- Runtime failures should use `unigym._optional.require` or
  `MissingExtraError` with the correct extra name.

### 7. Checks

Run the relevant checks before committing:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m ruff check unigym examples tests
PYRIGHT_PYTHON_GLOBAL_NODE=false .venv/bin/pyright
PYRIGHT_PYTHON_GLOBAL_NODE=false .venv/bin/pyright -p pyrightconfig.api.json
```

`pyrightconfig.api.json` is the public-API ratchet: it enables Unknown
diagnostics and `reportUnnecessaryCast`. It is expected to be stricter than the
main `pyrightconfig.json`.

### 8. Review checklist

Before committing a typing change, verify:

- no unexplained `Any`, `object`, or `cast` was added;
- public signatures are concrete;
- runtime checks, not casts, establish narrowing;
- the main pyright baseline did not regress;
- tests and ruff pass;
- residual Unknowns are reported honestly rather than suppressed.
