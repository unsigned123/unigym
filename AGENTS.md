# Working on unigym-core

## 0. Prime directive

Your task is to deliver the requested change **without degrading the
architecture, contracts, code style, or long-term maintainability**.

Task completion does not justify a local hack, a broken abstraction, a weakened
test, or an unapproved contract change. A change that makes today's task pass
but makes the next change harder is a failure.

If the task requires violating a non-negotiable rule and no approved design
covers it, **stop and ask**. Do not silently choose an architectural
interpretation.

## 1. Read first and precedence

Read in this order:

1. this file - agent behavior and non-negotiable rules;
2. `docs/testing.md` - before touching any pytest test;
3. `ci/compat.toml` - supported environment lanes, once it exists;
4. the design/contract document for the area being changed;
5. the relevant plan record in the planning repository.

Authority:

| Question | Authority |
| --- | --- |
| Agent behavior, architecture boundaries, workflow | this file |
| Test layers, markers, commands, red-test handling | `docs/testing.md` |
| Supported Python/dependency/backend combinations | `ci/compat.toml` |
| Why the design exists | design docs and plan records |
| Current runtime behavior | the tests, unless a test is proven wrong |

Do not treat `AGENTS.local.md`, a scratch note, an old branch, or an agent's
memory as project policy. If documents conflict, stop and ask.

## 2. What this repository is

This is the public core repository for `unigym` (`unigym-core`).

- It contains the framework code, public API, and the `manual` adapter.
- It does not contain the private asset library, benchmark packs, scene packs,
  old benchmark data, or vendor-specific adapters.
- The base environment is self-contained and does not require Isaac Lab or
  MuJoCo Warp.
- Heavy backends run in separate environments and are selected by CI lanes.
- The public API is a long-lived contract, not a private implementation detail.

The test suite is being migrated to a layered architecture. Do not create a
parallel ad-hoc test layout while migration is in progress. Read
`docs/testing.md`.

## 3. Design philosophy

These principles are not optional.

1. **Explicit over implicit.**
   Do not add hidden defaults, magic inference, automatic injection, or silent
   fallback. Behavior that matters must be visible in configuration, the public
   API, or the design document.

2. **Simple over clever.**
   Prefer the smallest mechanism that states the invariant. Do not introduce a
   framework, registry, base class, or generic layer without a current need and
   a clear owner.

3. **Composition over inheritance.**
   Prefer small protocols and composition. Avoid deep hierarchies and
   inheritance used only for code reuse.

4. **One owner per concept.**
   Each behavior, invariant, and piece of policy has one module or layer that
   owns it. Do not duplicate a decision across layers.

5. **Boundaries over convenience.**
   Do not reach through an abstraction because it is faster to patch the
   symptom. If an abstraction is insufficient, propose a change at its owner
   boundary.

6. **Invariants at construction.**
   Validate configuration and state at the boundary where they enter the
   system. Do not repeatedly repair invalid state deep inside the call stack.

7. **Fail fast at the boundary.**
   Errors must identify the field and the expected/actual values. Do not
   swallow errors or replace them with a silent default.

8. **No hidden global state.**
   Avoid mutable module globals, ambient registries, import-time side effects,
   and test-only hooks in production code.

9. **Delete before add, simplify before generalize.**
   Remove dead code and obsolete concepts. Do not keep compatibility shims for
   unowned callers. Do not create speculative abstraction.

10. **Long-term maintainability wins.**
    If the requested change conflicts with these principles and the engineering
    plan has not explicitly authorized the trade-off, stop and ask.

Code style is part of maintainability: match surrounding style, keep the diff
focused, do not reformat unrelated code, and do not leave temporary workarounds
in the main path.

## 4. Architecture and ownership

Dependency direction and ownership are invariants:

- `unigym/core` is independent of concrete backends.
- Backend-specific behavior lives under `unigym/backends/<backend>`.
- Upper layers may depend on lower-level contracts; lower layers must not know
  about concrete upper-layer implementations.
- Optional dependencies are imported lazily, at the function boundary or under
  `TYPE_CHECKING`; they must not become base-module imports.
- Concrete backend classes, adapter classes, and runtime implementations are
  selected at composition boundaries, not hard-coded throughout the framework.
- Public API surfaces are small, typed, and stable.
- No layer may bypass encapsulation through private attributes, private
  imports, monkeypatching, or test-only hooks.

If a boundary appears insufficient:

1. describe the missing capability;
2. identify its owner;
3. propose a design change at that boundary;
4. ask for approval before implementing it.

Do not solve an architectural gap with a local special case.

Prefer structural enforcement over prose or source-text checks:

- `import-linter` for dependency direction;
- `pyright` for type contracts;
- `ruff` for code style;
- import smoke tests for optional-dependency isolation;
- behaviour tests for runtime contracts.

Do not add keyword-based AST scans to encode design principles. If a rule needs
enforcement, make the wrong shape structurally unavailable or use a real tool.

## 5. Contracts and compatibility

- The public API is a contract. Do not change its meaning silently.
- Configuration must be explicit. Do not add hidden defaults or automatic
  inference for behavior that users can observe.
- Invalid configuration must fail fast with a teaching error.
- Do not add silent fallback paths.
- Do not introduce backend-specific branching into shared code. Use protocols,
  capabilities, or backend-owned implementations.
- Do not add compatibility shims for internal callers without an approved
  migration plan.
- A breaking contract change requires a design decision, migration notes, and
  tests in the same change.
- If you are unsure whether a change is contract-breaking, stop and ask.

Specific runtime semantics belong in design documents and tests, not as a list
of special cases in this file.

### 5.1 Versioning

- Use the form `vMAJOR.MINOR.PATCH` for project and release versions, for
  example `v0.123.4`. Do not use internal plan IDs such as `v107` or `v106.1`
  in tracked documentation, plan records or code comments.
- Python package metadata must use the PEP 440 form without the leading `v`
  (`0.123.4`); release tags, documentation and plan records use `v0.123.4`.
  They refer to the same release.
- The package metadata is the single source of truth for the release version.
  Do not duplicate the literal in tests or comments.
- Contract and schema versions are explicit code constants, not comments.
- Do not put release history in code comments (`# v0.123.4: changed X`).
  History belongs in git, CHANGELOG and release notes.
- Comments may record a deprecation with a removal target, but it must match a
  real runtime deprecation and a tracked removal plan.
- Test version numbers must be checked through package metadata and format, not
  by asserting a hardcoded release literal.

### 5.2 Ownership and comments

- Human ownership belongs in `CODEOWNERS`, maintainer documentation or plan
  records, not in public code comments.
- Do not put personal names, handles, emails or "owner: <person>" markers in
  source comments.
- Code comments document responsibility, lifetime, mutation rights and
  invariants: "the caller owns the buffer", "the backend owns the name map",
  "the returned object is borrowed".
- A conceptual owner ("one owner per concept", "this module owns validation")
  is valid. Human ownership metadata in public code is not.
- Comments explain why an invariant exists, not who wrote it or which release
  introduced it.

## 6. Type annotation policy

This project targets Python 3.11. Type annotations are part of the public
contract, not decoration.

### 6.1 Public surfaces are fully typed

- Every public function, method, dataclass field, module-level constant with a
  non-obvious type, `Protocol` member, and `TypedDict` field must have explicit
  parameter and return annotations.
- Public API examples: `Run`, `RunSpec`, `ArticulationSpec`, `ControlGroup`,
  `ControlRuntime`, `NamedAction`, `Adapter`, `EnvBackend`, camera/contact
  protocols, and registry factories.
- Internal local variables are inferred; do not add annotation noise unless the
  type is genuinely non-obvious.

### 6.2 Use postponed annotations

- Non-empty modules that contain annotations must start with
  `from __future__ import annotations` after the module docstring.
- Do not quote annotations for forward references when postponed evaluation is
  enabled. Remove redundant quotes from annotations only; never rewrite runtime
  strings, dict keys/values, `Literal[...]` values, or serialized data.
- Prefer `ruff --select UP037 --fix` for annotation quote removal.

### 6.3 Forbidden type escapes

Do not use these to make pyright quiet:

- `cast(...)` as a general-purpose escape;
- `object` as a substitute for an unknown type;
- `Any` as a substitute for understanding the data;
- blanket `# type: ignore` or file-level suppression.

`cast` is allowed only at a genuine runtime-validated boundary where the type
system cannot express the relationship. A permitted cast must have:

1. an adjacent runtime check (`isinstance`, validation, or assertion) that
   establishes the invariant; and
2. a short comment explaining why the cast is necessary.

`Any` is allowed only for genuinely arbitrary external payloads (for example an
opaque plugin extension payload) and must carry a comment. Prefer a recursive
`JsonValue` alias over `Any` for raw JSON/config data.

`object` is only acceptable when a value is intentionally opaque to the
consumer and no operation is performed on it. It is never a stand-in for
"unknown".

### 6.4 Prefer real types over narrowing hacks

Use, in order:

1. a concrete type or a common base class (`Mapping`, `Sequence`, `CameraRig`,
   `EnvBackend`, ...);
2. `Protocol` for duck-typed boundaries;
3. `TypedDict` for config/record mappings that are read by key;
4. `JsonValue` for raw JSON/config trees before validation;
5. `TypeGuard` / `isinstance` / `assert` for runtime narrowing;
6. a documented `cast` only when the above cannot express the invariant.

If two values share a base class, annotate the base class rather than a union
of concrete classes. If a value is genuinely dynamic, declare the field or
runtime attribute in its owner instead of casting at every consumer.

### 6.5 Unknown is not an error to hide

- Pyright `standard` does not report `Unknown` as an error. That is deliberate.
- Do not add `Any`, `object`, or `cast` just to remove an `Unknown`.
- If a type cannot be inferred without an architectural change, leave the
  `Unknown` and report it with its root cause.
- Public API `Unknown` reduction is a separate ratchet
  (`pyrightconfig.api.json`); zero is not required per change, but new public
  `Unknown` is discouraged.

### 6.6 Optional dependencies

- Optional imports (`PIL`, `imageio`, `av`, `aiortc`, Isaac Lab, etc.) stay
  lazy at the function boundary or under `TYPE_CHECKING`; they must not become
  base-module imports.
- Runtime failures should use `unigym._optional.require` or
  `MissingExtraError` with the correct extra name.

## 7. Testing contract

Before touching pytest tests, read `docs/testing.md` and follow its red-test
protocol. That document is normative for test layers, markers, commands,
environment lanes and test conventions.

Non-negotiable testing rules:

- Existing tests are contracts. A red test is information, not permission to
  edit the test.
- Never delete, skip, xfail, weaken, broaden or hide a red test just to make CI
  green.
- Do not change a marker to remove a test from a required lane.
- Do not update golden or baseline data without explicit approval.
- Do not mock the internal function under test.
- Do not add a new pytest plugin, test dependency or test framework without
  design approval.
- New tests belong in the lowest layer that can prove the behavior.
- Default tests must not import heavy backends at module import time.
- Do not add machine-specific absolute paths, network calls or sleeps to fast
  tests.
- `fast` is derived, not hand-written. Version support is an environment lane,
  not a test marker.
- Full backend/version matrices run in CI, not in every local edit loop.

If a test is red and the root cause is not understood, stop and ask. Do not make
it green by guesswork.

## 8. Environments and dependencies

- The self-contained `.venv` is the primary base lane.
- Isaac Lab and MuJoCo Warp use external environments; do not install them into
  the base `.venv` without an approved design change.
- Environment resolution uses `UNIGYM_*_ENV` names. Do not hardcode conda
  prefixes, interpreter paths, `/home`, `/data`, or machine-specific defaults in
  tracked files.
- `tests/_support/envs.py` is tracked test infrastructure with no machine values.
- Supported combinations are the named coherent lanes in `ci/compat.toml`, not
  a full Cartesian product.
- Daily work uses the fixed D0/D2 lanes; the version matrix belongs to CI.
- Do not change dependency ranges, lock files, supported versions or lane
  definitions without an explicit design decision.
- Do not mutate shared conda environments as part of a test run.
- A missing required CI environment is a failure, not a silent skip.

## 9. Development workflow

The normative workflow is `docs/development.md`. This file keeps only the hard
constraints.

For any non-trivial change:

1. read `AGENTS.md`, `docs/testing.md` and the relevant design/contract;
2. resolve goal, non-goals, constraints and failure modes;
3. write the design brief (`docs/templates/design-brief.md`);
4. grill the brief before implementation or delegation;
5. implement the smallest coherent slice;
6. verify targeted tests, fast suite and relevant lanes;
7. obtain independent review (`docs/templates/review-report.md`);
8. fix blocking findings and re-review;
9. update docs/design/plan if the contract changed;
10. report results and unverified lanes.

If new information invalidates the design, stop and return to the brief. Do not
silently replan during implementation.

Do not perform unrelated refactors in the same change. Do not reformat files
you did not need to touch. Do not leave dead code or temporary debugging state.

## 10. Design gate and delegation

The main agent owns architecture and task decomposition. A subagent executes a
bounded brief; it does not make architectural decisions.

Use `docs/templates/design-brief.md` and the gates in `docs/development.md`.
Before writing code or dispatching a subagent, the brief must at least state:

- goal, non-goals and owned files;
- new/changed classes, protocols and registries;
- interactions, invariants and failure modes;
- compatibility, tests, lanes and rollback.

If any of these are materially unclear, **do not delegate**. Ask the user.
Unresolved structural questions block implementation; do not use "we will
figure it out while coding" for architecture.

For a trivial local fix, an abbreviated brief in the task prompt is acceptable,
but architectural decisions remain with the main agent.

## 11. Subagent protocol and stop conditions

All subagents are bound by this file and `docs/development.md`. A subagent brief
may add stricter rules, but may not relax these rules.

A subagent must **stop immediately and report `BLOCKED`** when:

- the design or requirement is ambiguous or conflicts with existing architecture;
- the task requires a new public API, dependency, version lane or compatibility
  shim;
- the task requires changing files outside the brief without approval;
- the implementation would need to bypass encapsulation, use private internals,
  monkeypatch production code, or add a special case;
- an existing test turns red and its root cause is not understood;
- the task requires deleting, skipping, weakening or broadening an existing test;
- a real bug outside the brief is discovered;
- the required environment or verification lane is unavailable;
- the change would degrade long-term maintainability;
- completion cannot be verified honestly.

A subagent must not:

- redesign the solution to make its task easier;
- expand scope or touch unowned paths;
- hide uncertainty behind a workaround;
- report success when a required lane was not run;
- commit or push unless explicitly authorized;
- treat "tests green" as proof of architectural correctness.

The main agent must review the actual diff and the test results, not only the
subagent's summary.

## 12. Prohibited shortcuts

Never solve a problem by:

- breaking encapsulation or reaching through private internals;
- monkeypatching production code;
- adding a special case for one backend or one caller in shared code;
- adding hidden fallback, automatic inference or silent default;
- copying implementation instead of using or extending the owner abstraction;
- weakening, deleting, skipping or hiding tests;
- weakening type annotations or suppressing diagnostics;
- adding a new dependency, test plugin or version lane without approval;
- performing unrelated refactors or reformatting;
- leaving temporary workarounds in the main path;
- committing secrets, credentials, machine paths or generated artifacts;
- committing or pushing without explicit instruction.

If the correct solution is larger than the requested task, stop and propose the
design change. Do not ship a shortcut and call the task complete.

## 13. Verification

Run the checks relevant to the change. For the current bootstrap state:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check unigym examples tests
PYRIGHT_PYTHON_GLOBAL_NODE=false .venv/bin/pyright
PYRIGHT_PYTHON_GLOBAL_NODE=false .venv/bin/pyright -p pyrightconfig.api.json
PYRIGHT_PYTHON_GLOBAL_NODE=false .venv/bin/pyright -p pyrightconfig.model.json
PYRIGHT_PYTHON_GLOBAL_NODE=false .venv/bin/pyright -p pyrightconfig.scene.json
PYRIGHT_PYTHON_GLOBAL_NODE=false .venv/bin/pyright -p pyrightconfig.runtime.json
PYRIGHT_PYTHON_GLOBAL_NODE=false .venv/bin/pyright -p pyrightconfig.run.json
```

After the test migration, use the commands in `docs/testing.md`.

Rules:

- Run targeted tests before broad tests.
- Local fast green is not CI full green.
- Do not claim a backend, GPU, Isaac or compatibility lane passed unless it was
  actually run.
- Report unverified lanes and remaining risks honestly.
- Do not run heavy GPU/Isaac suites unless the task requires it and access is
  assigned.

## 14. Reporting and completion

A change is complete only when:

- the requested behavior is implemented;
- architecture, boundaries and public contracts remain sound;
- relevant tests exist at the lowest useful layer;
- targeted tests and the fast suite pass;
- relevant lanes were run or explicitly reported as not run;
- independent review returned `APPROVE`;
- docs/design/plan are updated when the contract changed;
- no prohibited shortcut was introduced.

Use:

- `docs/templates/completion-report.md` for completed work;
- `docs/templates/review-report.md` for review results;
- `docs/templates/blocked-report.md` when blocked.

Do not report success for partial work. Do not omit unverified lanes. Do not
commit or push unless the user explicitly asks.
