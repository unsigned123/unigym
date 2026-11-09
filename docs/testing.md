# Testing in unigym-core

Status: normative test policy. The suite is being migrated to this structure;
some directories, markers and commands described here may not exist yet. Follow
this document for all new tests. Do not create a parallel ad-hoc test layout
while the migration is in progress.

Audience: contributors and coding agents.

Reading order:

1. `AGENTS.md` — type, optional-dependency and code-quality rules.
2. This document — test design, test operations and red-test policy.
3. `ci/compat.toml` — supported environment lanes, once it exists.
4. The design/plan record for the change you are making.

If this document and an existing test disagree, stop and investigate. Do not
silently rewrite the test to match current behavior. Existing tests are
contracts unless the contract is intentionally changed.

## 0. TL;DR

- Tests protect behavior, not implementation details.
- Fast feedback is local; the full environment matrix belongs to CI/CD.
- `tests/unit/**` mirrors the product domain tree for basic correctness.
- `tests/contract`, `tests/conformance`, `tests/integration` and `tests/meta`
  are flat, behavior-oriented suites.
- `tests/backends/<backend>/**` runs only in that backend's environment and is
  outside the default `testpaths`.
- Directory structure supplies most markers automatically. Hand-write only
  resource or capability markers.
- Do not hand-maintain a complete test table. Generate a catalog from pytest
  collection.
- A red test is information. Never delete it, skip it, weaken it or hide it just
  to make CI green.
- When in doubt, stop and ask. "Make it green" is not a diagnosis.

## 1. Precedence and sources of truth

| Question | Authority |
| --- | --- |
| Type annotations, optional imports, code quality | `AGENTS.md` |
| Test layers, markers, commands, red-test handling | this document |
| Supported Python/dependency/backend combinations | `ci/compat.toml` |
| Why the architecture exists | design docs and plan records |
| What the code currently does | the tests, unless a test is proven wrong |
| What tests exist and where | generated test catalog |

Rules:

- Tests are the source of truth for behavior, not a catalog file.
- A hand-written test table is not authoritative and must not be added.
- If a test is red, the default presumption is that the product code is wrong.
- If the contract itself changed intentionally, update the design/contract and
  the test together, with explicit review.

## 2. Testing principles

1. **Behavior over implementation.**
   Test the public contract: inputs, outputs, errors, state transitions and
   observable side effects. Do not assert private helper calls or internal data
   layout.

2. **Lowest layer that can prove the behavior.**
   Prefer unit, then contract, then conformance, then integration. Do not use a
   running simulator to prove pure data logic.

3. **One reason to fail.**
   A test should fail for one clear behavioral reason. Use parametrization for
   table-like variations instead of copying the same test body.

4. **Fast by default, complete in CI.**
   The edit loop must stay small. Heavy processes, real backends and external
   environments run in explicit lanes, not in every local run.

5. **Deterministic and isolated.**
   No sleeps, no network, no reliance on test order, no shared mutable global
   state, no ambient GPU selection, no machine-specific paths.

6. **Loud failure.**
   If a required dependency or capability is absent, fail with a useful message.
   Local optional-environment skips are allowed only when explicitly not
   required; strict CI must fail.

7. **No hidden debt.**
   A permanent skip, a broad `xfail`, a growing baseline or a weakened tolerance
   is architectural debt. It needs an owner and a removal plan.

8. **Tests are code.**
   Test changes receive the same review scrutiny as product changes. A test is
   not a scratch file.

## 3. Layers

| Layer | Directory | Purpose | Default local suite |
| --- | --- | --- | --- |
| L0 smoke | `tests/smoke` | install, import, public names, tiny end-to-end | yes |
| L1 unit | `tests/unit` | pure logic and basic correctness; mirrors product domains | yes |
| L2 contract | `tests/contract` | public API, protocols, config contracts, fail-fast behavior | yes |
| L3 conformance | `tests/conformance` | cross-backend behavior using the fake backend | yes |
| L4 backend | `tests/backends/<backend>` | one real backend's implementation behavior | no |
| L5 integration | `tests/integration` | CLI, daemon, real subprocesses, heavy paths | no |
| L6 meta | `tests/meta` | markers, imports, stubs, env provider, repository standards | yes |
| L7 regression | future `tests/regression` | golden/numerical regression | no, later |

Layer rules:

- `tests/unit` must not start a real backend or subprocess.
- `tests/contract` may use real pure objects but not a real simulator.
- `tests/conformance` uses the fake backend by default. Real backends are
  selected by CI lanes.
- `tests/backends` contains backend-local tests only. Cross-backend contract
  cases belong in `tests/conformance`, not duplicated here.
- `tests/integration` is always explicit/heavy and is not part of
  `make test-fast`.
- `tests/meta` tests the test system and repository rules, not product physics.
- `tests/regression` does not exist until the golden policy is activated.

## 4. Directory layout

Target layout:

```text
tests/
├── conftest.py                  # single auto-tagging entry point
├── _support/                    # not collected
│   ├── data/tiny_arm.xml
│   ├── envs.py                  # backend environment resolution
│   ├── factories.py
│   ├── assertions.py
│   └── fakes/
│       ├── backend.py
│       └── transport.py         # only when a real boundary needs it
├── smoke/
├── unit/
│   ├── core/
│   ├── model/
│   ├── scenes/
│   ├── runtime/
│   ├── run/
│   ├── terminations/
│   ├── filters/
│   ├── robots/
│   ├── adapters/
│   └── backends/                # framework-level registry tests only
├── backends/
│   ├── mujoco/
│   ├── mjwarp/
│   ├── isaaclab2/
│   └── isaaclab3/
├── contract/
├── conformance/
├── integration/
└── meta/
```

Mapping rule for unit tests:

```text
unigym/runtime/controls.py       -> tests/unit/runtime/test_controls.py
unigym/terminations/articulation.py
                                  -> tests/unit/terminations/test_articulation.py
unigym/backends/mujoco/env.py    -> tests/backends/mujoco/test_env.py
```

Mirror the product domain tree to a useful depth. Do not create a test file for
a module that has no independently testable behavior.

## 5. Markers

### 5.1 Automatic markers

Applied once by `tests/conftest.py` from a central directory-to-marker mapping:

| Marker | Source |
| --- | --- |
| `layer_smoke` | `tests/smoke/**` |
| `layer_unit` | `tests/unit/**` |
| `layer_contract` | `tests/contract/**` |
| `layer_conformance` | `tests/conformance/**` |
| `layer_backend` | `tests/backends/**` |
| `layer_integration` | `tests/integration/**` |
| `layer_meta` | `tests/meta/**` |
| `domain_core` | `tests/unit/core/**` |
| `domain_model` | `tests/unit/model/**` |
| `domain_scene` | `tests/unit/scenes/**` |
| `domain_runtime` | `tests/unit/runtime/**` |
| `domain_run` | `tests/unit/run/**` |
| `domain_terminations` | `tests/unit/terminations/**` |
| `domain_filters` | `tests/unit/filters/**` |
| `domain_robots` | `tests/unit/robots/**` |
| `domain_adapters` | `tests/unit/adapters/**` |
| `domain_backends` | `tests/unit/backends/**` |
| `backend_agnostic` | no specific backend |
| `backend_mujoco` | `tests/backends/mujoco/**` or conformance fixture |
| `backend_mjwarp` | `tests/backends/mjwarp/**` or conformance fixture |
| `backend_isaaclab2` | `tests/backends/isaaclab2/**` or conformance fixture |
| `backend_isaaclab3` | `tests/backends/isaaclab3/**` or conformance fixture |
| `fast` | smoke/unit/contract/meta without resource markers |
| `slow` | integration automatically; manual on heavy unit/backend tests |

Rules:

- `fast` is derived. Never hand-write it.
- A unit test that becomes heavy must add `slow`; the hook then removes `fast`.
- `slow` on a unit test does not make it an integration test.
- Do not create one marker per dependency or per version.

### 5.2 Manual markers

Only resource or capability requirements are hand-written:

| Marker | Meaning |
| --- | --- |
| `gpu` | requires explicit `CUDA_VISIBLE_DEVICES`; skip when absent |
| `sim` | requires a real heavy simulator/runtime |
| `subprocess` | starts real subprocesses |
| `network` | uses network |
| `serial` | cannot run concurrently |
| `requires_capabilities("...")` | conformance capability requirement |
| `known_deviation("...", ..., report="...")` | documented backend deviation |
| future `pack` / `external_asset` | provisioned assets |
| future `golden` | golden regression |

Rules:

- Unknown markers are collection errors (`--strict-markers`).
- `tests/meta/test_marker_registry.py` checks marker coverage and directory
  mapping.
- Do not use a marker to hide a red test or remove it from a required lane.

## 6. Environment lanes

Do not assume every backend exists in the self-contained `.venv`.

| Lane | Environment | Purpose |
| --- | --- | --- |
| D0 | primary `.venv`, Python 3.11 + MuJoCo 4.x | daily base iteration |
| D2 | external `UNIGYM_ISAACLAB2_ENV` | daily IsaacLab 2 iteration |
| C1 | Python 3.11 + oldest supported MuJoCo/NumPy pair | compatibility |
| C2 | Python 3.11 + latest MuJoCo/NumPy | primary CI |
| C3 | Python 3.12 + latest MuJoCo | nightly |
| W1 | MuJoCo Warp pinned pair | warp lane |
| I2 | external IsaacLab 2 environment | IsaacLab 2 CI |
| I3 | external IsaacLab 3 environment | IsaacLab 3 CI |

The supported set is the union of these validated lanes. It is not a promise
that every Python × MuJoCo × NumPy × backend combination works.

### 6.1 Environment resolution

`tests/_support/envs.py` is tracked test infrastructure. It contains no machine
paths, credentials or version defaults. It resolves:

```text
UNIGYM_TEST_PYTHON           explicit interpreter override
UNIGYM_MUJOCO_ENV            conda env for the MuJoCo lane
UNIGYM_MJWARP_ENV            conda env for the warp lane
UNIGYM_ISAACLAB2_ENV         conda env for IsaacLab 2
UNIGYM_ISAACLAB3_ENV         conda env for IsaacLab 3
UNIGYM_CONDA_EXE             optional conda executable override
UNIGYM_TEST_ENV_STRICT       "1" = missing environment fails
```

Resolution order:

1. explicit interpreter override;
2. named conda environment through `conda run`;
3. current interpreter, only when that lane is safe in it.

No tracked file may contain `/home/...`, `/data...`, a conda prefix, or a
machine-specific interpreter path.

### 6.2 Local and CI semantics

- Local missing optional environment: explicit skip with a teaching message.
- Required CI lane missing: fail, never skip.
- `CUDA_VISIBLE_DEVICES` is empty by default; GPU tests require explicit
  assignment.
- `OMNI_KIT_ACCEPT_EULA=YES` is supplied only to the IsaacLab 3 lane.

Example backend command:

```bash
conda run --no-capture-output -n "$UNIGYM_ISAACLAB2_ENV" \
  python -m pytest tests/backends/isaaclab2 -q
```

## 7. Running tests

### 7.1 Current bootstrap commands

Use the environment that owns the lane. For the primary `.venv`:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m pytest tests/test_control_contract.py -vv
.venv/bin/python -m pytest tests/test_control_contract.py::test_name -vv
```

Do not run the full backend matrix locally unless you are working on that
backend or preparing a release.

### 7.2 Target commands

Once the test Makefile exists:

```bash
make test-fast          # smoke + unit + contract + meta, target < 60 s
make test               # default testpaths, including fake conformance
make test-integration   # real CLI/daemon paths
make test-full          # all tests in the current environment
```

CI selects by markers, not by source path:

```bash
pytest -m fast
pytest -m "layer_conformance and not sim"
pytest -m "backend_mujoco and not slow"
```

### 7.3 Before commit

At minimum:

```bash
pytest -m fast -x -q
```

If the change touches a backend, public API, or dependency boundary, run the
relevant targeted lane/tests as well.

### 7.4 Before merge/release

The full required matrix is a CI responsibility. Do not claim a lane passed
unless you actually ran it. Report unrun lanes explicitly.

## 8. Test authoring conventions

### 8.1 Naming and documentation

- Test file: `test_<feature>.py`.
- Test name: `test_<subject>_<condition>_<expected>`.
- First docstring line: the protected behavior, not an implementation summary.
- Use comment only for non-obvious setup or a version/environment constraint.

Example:

```python
def test_termination_requires_explicit_articulation():
    """A per-articulation termination rejects a missing articulation."""
```

### 8.2 Structure

- Arrange, act, assert.
- One behavior per test; use `pytest.mark.parametrize` for variants.
- Assert observable results with precise, readable failure messages.
- When testing errors, match the field and the expected/actual value.
- Use `tmp_path` for filesystem work.
- Use `_support/factories.py` for repeated object construction; do not build
  production models with `SimpleNamespace`.

### 8.3 Dependencies and imports

- Do not import heavy backends at test-module import time.
- Import optional/heavy modules inside the test or fixture that needs them.
- Default tests must not import IsaacLab, Omni, MuJoCo Warp or a GL context.
- Do not add a new test dependency or pytest plugin without review.
- No network, sleep or wall-clock dependence in fast tests.
- Do not use absolute paths or machine-specific assets.

### 8.4 Assertions

- Test the contract, not private attributes.
- Do not assert log text unless log text is the contract.
- Do not use `assert result` when a specific value or error is expected.
- Do not weaken a tolerance to hide a regression. A tolerance change is a
  contract change and needs justification.
- Do not compare full large objects when a focused invariant is enough.

## 9. Where to add a test

| Change | Test location |
| --- | --- |
| Pure data/IR/quaternion/config logic | `tests/unit/<domain>/` |
| Public API shape, protocol or config contract | `tests/contract/` |
| Cross-backend behavior | `tests/conformance/` |
| One backend's implementation detail | `tests/backends/<backend>/` |
| CLI, daemon, subprocess, real run | `tests/integration/` |
| Marker/import/stub/env policy | `tests/meta/` |
| Bug fix | lowest layer that reproduces the bug |
| Install/import/tiny end-to-end | `tests/smoke/` |

Do not duplicate the same behavior across layers. Unit tests prove local
correctness; contract tests prove the boundary; conformance tests prove
agreement between backends.

## 10. Stubs and fakes

The goal is a small number of high-value test doubles, not a fake ecosystem.

Rules:

1. Prefer real core objects and a real tiny MJCF over faking them.
2. One fake per external protocol.
3. Fakes live in `tests/_support/fakes/`; no ad-hoc fakes in test files.
4. No `MagicMock` for core data. Use typed factories.
5. `monkeypatch` is for environment variables, missing optional dependencies or
   fault injection, not for replacing the function under test.
6. Fakes are deterministic, sleep-free, asset-free and network-free.
7. Unsupported capabilities raise explicitly; never return `None`.
8. Every fake is listed in `_support/fakes/README.md` with protocol, boundary,
   limitations and reset semantics.
9. `tests/meta/test_stub_policy.py` enforces the location and mock policy.
10. An editable install in an external environment is optional; use
    `pip install -e . --no-deps` only when entry points or packaging behavior
    are under test.

## 11. Red-test protocol

This is the most important section.

> Existing tests are contracts. Unless the contract is intentionally changed,
> the default presumption is that the product code is wrong, not the test.

### 11.1 Immediate rules

When an existing test turns red:

1. Stop.
2. Do not rerun until it passes.
3. Do not delete the test.
4. Do not add `skip` or `xfail`.
5. Do not weaken or broaden the assertion.
6. Do not change a marker to remove it from a required lane.
7. Do not hide it with `pytest -k`, a path change or a workflow selector.
8. Do not update a baseline or golden file without explicit approval.
9. Reproduce the exact test with verbose output.

```bash
pytest tests/path/test_file.py::test_name -vv
```

If the test is in the fast lane, it blocks new work until classified.

### 11.2 Classification before editing

| Classification | Required action |
| --- | --- |
| Product regression | Fix product code. Add a lower-layer regression test if coverage was missing. |
| Test bug | Fix the test while preserving its original intent. Never weaken it just to pass. |
| Intended contract change | Update the design/contract and the test in the same change. Get explicit review. |
| Environment/lane drift | Restore the declared environment or version. Do not add a skip. |
| Dependency/version incompatibility | Fix the code/test, or narrow the supported lane in `ci/compat.toml`. Do not hide it. |
| Flaky/non-deterministic | Prove it with repeated runs and logs. Fix determinism. Quarantine only with an issue, owner and expiry. |

### 11.3 Prohibited actions

Never make a red test green by:

- deleting it;
- adding a global or permanent skip;
- adding a broad `xfail`;
- changing the expected value to the current buggy output;
- widening a tolerance without a contract justification;
- mocking the internal function under test;
- hardcoding the current version;
- changing a required CI selector;
- updating golden/baseline data;
- making the test pass by test order or shared state.

### 11.4 Intentional behavior changes

If the behavior change is intentional:

- update the relevant design/contract first;
- update or replace the tests in the same change;
- explain why the old contract no longer holds;
- request explicit review;
- do not leave an obsolete test silently red.

Deleting a test is allowed only when the behavior or feature itself is removed
with an explicit design decision, not when the test is inconvenient.

### 11.5 Flaky tests

- Reproduce with repeated runs and record the seed/environment.
- Find the source: time, randomness, order, shared state, filesystem, GPU,
  subprocess, network or an actually unstable contract.
- Fix the source. Do not add global retries as the first response.
- If quarantine is unavoidable, require an issue, owner, reason, expiry and an
  allowlist entry. Quarantine is not a pass; CI must report it.
- A new quarantine requires approval. A permanent quarantine is forbidden.

### 11.6 Main is red

- If a recent commit caused the failure, revert or fix forward immediately.
- Do not stack unrelated changes on a red main.
- Do not merge a PR by excluding the failing test.
- Report the exact failing lane and command.

### 11.7 Report template

When you fix a red test, report:

```text
Command:
Expected:
Actual:
Classification:
Root cause:
Fix:
Tests rerun:
Lanes still unverified:
```

If the root cause is not understood, stop and ask. A green suite obtained by
weakening the test is a lie.

## 12. Generated test catalog

The catalog is generated, never hand-maintained.

```text
tests = source of truth
markers + docstrings = metadata
catalog = generated view
```

Target tool:

```text
tools/ci/test_catalog.py
```

Rules:

- Runs `pytest --collect-only`; does not execute tests.
- Uses a small collection hook/plugin to read collected items.
- Imports no torch, mujoco, isaaclab or omni directly.
- Records node id, file, line, layer, domain, backend, markers, fixtures and
  first docstring line.
- Does not record dependency versions; those belong to `ci/compat.toml`.
- Default scope follows default `testpaths`; backend catalogs are generated in
  their own lanes.
- CI publishes `test-catalog.json` as an artifact.
- If a catalog file is committed, CI regenerates and diffs it. Hand editing is
  forbidden.

## 13. Golden policy

Golden/regression tests are deliberately deferred.

- Do not create `tests/regression/` or `tests/golden/` yet.
- Do not add golden dependencies to the base dev extra.
- Do not add golden to `testpaths`.
- Future golden tests require a separate lane, small fixtures, explicit
  recording and human approval to update.
- Golden is not a replacement for unit, contract or conformance tests.

## 14. Agent quick checklist

### Before touching tests

- Read this document.
- Identify the layer, marker and lane.
- Run the relevant test in the current environment.
- Check whether the change is a bug fix, a contract change, or a new behavior.

### While writing tests

- Put the test in the lowest layer that can prove the behavior.
- Add the narrowest markers.
- Keep module import lightweight.
- Use factories, `tmp_path` and explicit assertions.
- Do not duplicate behavior across layers.
- Do not add a new fake unless a real external boundary requires it.

### Before reporting completion

- Run the targeted test.
- Run `pytest -m fast -x -q`, or the closest available fast suite.
- Run the relevant backend lane if backend code changed.
- Report commands, results and unverified lanes.
- Do not claim full CI passed unless it actually ran.

## 15. Review checklist

- [ ] Test is in the correct layer.
- [ ] Test name and first docstring line state the protected behavior.
- [ ] Markers are minimal and correct.
- [ ] No heavy import at module import time.
- [ ] No absolute path, machine path or ambient GPU selection.
- [ ] No sleep, network or order dependence.
- [ ] No mock of internal product logic.
- [ ] No new test dependency without review.
- [ ] Failure message identifies the field/value under test.
- [ ] No hidden skip/xfail and no weakened tolerance.
- [ ] Targeted test and fast suite pass.
- [ ] Relevant backend lane was run or explicitly reported unverified.

## 16. Common mistakes

- Importing IsaacLab or MuJoCo Warp in the primary `.venv`.
- Importing optional dependencies at test-module import time.
- Hardcoding `/home/...`, `/data1/...` or a conda prefix.
- Using `SimpleNamespace` as a production object.
- Mocking the function under test.
- Testing private attributes instead of the contract.
- Using a real simulator to test pure data logic.
- Duplicating one behavior in unit, contract and conformance.
- Adding a version tag instead of a lane.
- Making a red test green by changing its expected value.
- Adding a permanent skip because an external environment is inconvenient.
- Treating a generated catalog as the source of truth.

## 17. Definition of done for a test change

- The change has a test at the lowest layer that proves the behavior.
- The targeted test passes.
- The fast suite passes.
- The relevant backend/integration lane is run or explicitly reported as not
  run.
- Markers, docstring and layer are correct.
- No hidden skip, weakened assertion or baseline update was introduced.
- The final report includes commands, results and unverified lanes.
