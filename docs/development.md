# Engineering workflow

Status: normative workflow policy.

This document defines how changes are planned, delegated, implemented, reviewed,
committed and reported. It complements `AGENTS.md` (binding agent rules) and
`docs/testing.md` (test operations).

## 1. Authority and sources of truth

| Question | Authority |
| --- | --- |
| Agent behavior, architecture boundaries, prohibited shortcuts | `AGENTS.md` |
| Workflow gates, roles, review and reporting | this document |
| Test layers, markers, commands, red-test protocol | `docs/testing.md` |
| Current work, decisions and progress | planning repository |
| Supported dependency/environment lanes | `ci/compat.toml` once present |

Do not store volatile state here. Commit hashes, test counts, pyright baselines
and current phase status belong in the planning repository or CI artifacts.

## 2. Roles

### User / owner

- sets goals and priorities;
- approves design trade-offs and scope;
- explicitly authorizes commit and push;
- resolves questions that the agent must not decide alone.

### Main agent

- owns architecture, design quality and task decomposition;
- writes the design brief;
- decides what may be delegated;
- reviews subagent output and the actual diff;
- runs or verifies the final checks;
- reports results honestly.

### Subagent

- executes one bounded brief;
- does not make architectural decisions;
- does not expand scope;
- stops and reports `BLOCKED` when the brief is insufficient;
- does not commit or push unless explicitly authorized.

### Reviewer

- is independent from the implementer;
- is read-only and does not edit files during review;
- reviews the actual diff, not only the implementer's summary;
- runs the relevant commands independently;
- returns `APPROVE` or `BLOCK` with evidence.

## 3. Workflow gates

Every non-trivial change passes these gates:

```text
G0 Intake
  -> G1 Design brief
  -> G2 Design review / grill
  -> G3 Delegation
  -> G4 Implementation
  -> G5 Verification
  -> G6 Independent review
  -> G7 Remediation
  -> G8 Commit
  -> G9 Push
  -> G10 Plan/status update
```

| Gate | Required outcome |
| --- | --- |
| G0 | Goal, non-goals, scope and constraints are explicit |
| G1 | Design brief is written |
| G2 | Open design questions are resolved |
| G3 | Task is decomposed and delegated with a bounded brief |
| G4 | Smallest coherent implementation exists |
| G5 | Targeted tests, fast suite and relevant lanes pass |
| G6 | Independent reviewer returns `APPROVE` |
| G7 | Blocking findings are fixed and re-reviewed |
| G8 | Commit is created only when explicitly authorized |
| G9 | Push is performed only when explicitly authorized |
| G10 | Plan and status reflect the result and residual risks |

If new information invalidates the design, stop and return to G1. Do not
silently replan during implementation.

## 4. Design brief

Use `docs/templates/design-brief.md`.

A brief must state:

- goal and non-goals;
- files to add or modify, with ownership;
- new or changed classes, protocols, registries and responsibilities;
- module interactions and data flow;
- invariants, failure modes and error behavior;
- compatibility and migration impact;
- test layers, markers and lanes;
- risks, open questions and rollback path.

For a trivial local fix, an abbreviated brief in the task prompt is acceptable.
Architectural decisions remain with the main agent.

## 5. Design review / grill

Before implementation or delegation, the brief must survive an adversarial
review:

- Why this abstraction rather than a simpler one?
- What existing concept does it duplicate?
- Which module owns each invariant?
- What fails, where does the error surface, and what is the recovery path?
- Is the change reversible?
- What tests prove it at the lowest useful layer?
- Does it expand scope or public API unnecessarily?
- Is this a current need or speculative generality?
- What lanes remain unverified?

Unresolved structural questions block implementation. Do not use "we will
figure it out while coding" for architecture.

## 6. Delegation

The main agent writes the task brief. The subagent receives:

```text
Assigned files
Allowed changes
Forbidden changes
Invariants to preserve
Verification commands
Stop/BLOCKED conditions
Commit/push policy
```

Rules:

- one writer per path; parallel agents must use disjoint file sets;
- a subagent may not modify files outside its assignment;
- a subagent may not change the design to make its task easier;
- a subagent may not add private imports, monkeypatches, special cases or hidden
  fallbacks;
- a subagent reports changed files, commands, results, exceptions and risks;
- a subagent does not commit or push unless explicitly authorized.

The main agent must review the actual diff and rerun or verify the key checks.

## 7. Implementation

- implement the smallest coherent slice;
- follow the owner boundary;
- preserve public contracts unless the brief authorizes a change;
- remove obsolete code and dead paths;
- do not perform unrelated refactors;
- do not leave temporary debugging state;
- if the task requires a shortcut or a boundary violation, stop and report
  `BLOCKED`.

## 8. Verification

Required evidence depends on the change:

- targeted tests for the changed behavior;
- fast suite;
- relevant backend/integration lane;
- lint and type checks;
- no production changes outside the declared scope.

Rules:

- local fast green is not CI full green;
- an unpushed/unrun lane is not a passed lane;
- a missing required CI environment is a failure, not a silent skip;
- report exact commands and results.

## 9. Independent review

The reviewer must:

- read `AGENTS.md`, this document, the relevant design and the actual diff;
- verify behavior, architecture, boundaries and maintainability;
- look for private imports, hidden shortcuts, weakened tests and scope creep;
- run the relevant commands independently;
- return one verdict:
  - `APPROVE` — no blocking findings;
  - `BLOCK` — one or more required fixes.

The reviewer must not edit files, commit or push during review.

Review report format: `docs/templates/review-report.md`.

## 10. Remediation

After `BLOCK`:

1. fix each blocking finding;
2. add or update tests when the finding exposes a behavior gap;
3. re-run the relevant checks;
4. request a fresh review;
5. do not treat unreviewed remediation as complete.

Non-blocking findings may be deferred only when recorded in the plan with a
reason and a follow-up owner.

## 11. Commit and push

- commit only when the user explicitly asks;
- push only when the user explicitly asks;
- do not force-push unless explicitly authorized;
- never commit secrets, credentials, machine paths or generated artifacts;
- prefer atomic, logically grouped commits:
  - docs/policy;
  - production fix;
  - tests;
  - CI/tooling;
- do not mix unrelated changes in one commit;
- report commit hash and push state.

## 12. Reporting

Use the templates:

- `docs/templates/completion-report.md`
- `docs/templates/review-report.md`
- `docs/templates/blocked-report.md`
- `docs/templates/design-brief.md`

A report is evidence, not a transcript. Include exact commands and results.
Do not claim a lane passed unless it ran.

When blocked, use `docs/templates/blocked-report.md` and stop. Do not hide
uncertainty behind a workaround.

## 13. Persistent versus volatile information

Put in this document:

- stable workflow and report formats;
- role responsibilities;
- review and remediation rules;
- durable engineering principles.

Do not put here:

- current commit hashes or test counts;
- current phase status;
- per-task file lists;
- machine-specific paths or environment names;
- generated test catalogs;
- one-off review findings that belong in the plan or commit.

## 14. Anti-patterns

- starting implementation before the design gate;
- delegating an unresolved design;
- letting a subagent redesign the task;
- self-review;
- reviewing a summary instead of the diff;
- fixing a red test by weakening it;
- adding a special case instead of fixing the owner boundary;
- reporting partial work as complete;
- claiming a lane passed from an earlier run;
- committing or pushing without explicit approval;
- storing volatile state in `AGENTS.md` or this document.

## 15. Definition of done

A change is done only when:

- the requested behavior is implemented;
- architecture, boundaries and contracts remain sound;
- tests exist at the lowest useful layer;
- targeted tests and fast suite pass;
- relevant lanes were run or explicitly reported as not run;
- independent review returned `APPROVE`;
- docs/design/plan were updated when the contract changed;
- no prohibited shortcut was introduced;
- commit/push status is reported honestly.
