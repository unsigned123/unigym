# tests/_support/fakes

Test doubles for external boundaries that cannot be exercised with real core
objects. This directory is not collected and contains no tests.

## Manifest

| Module | Protocol / boundary | Reset semantics |
| --- | --- | --- |
| `backend.py` | `unigym.env.backends.base.EnvBackend` — the simulator step/reset boundary | resets only the selected rows to spawn state (`q0`, zero `prev_action`, zero episode clock) and increments only their `episode_id`; `reset_all()` selects every row |

Every Python module in this directory (except the package marker) must appear
in the manifest table; `tests/meta/test_stub_policy.py` enforces this.

## `backend.py` — `FakeBackend`

Purpose: prove the backend-neutral `EnvBackend` contract and the conformance
layer without MuJoCo, IsaacLab, MJWarp, assets, network, subprocesses or a GPU.

Boundary: one articulation `arm` with one joint `shoulder`; one body control
group `body` driving that articulation (`position_pd` / `position`). The action
face is the articulation name `arm`, because the contract body-face rule is
`action_face == articulation`.

Behavior:

- `step(actions, raw_action=None, cmd=None)` consumes `dict[str, NamedAction]`,
  requires the named action face, validates kind/batch/joint coverage and
  replaces the scalar joint position with the target. `raw_action` is copied to
  `obs_cache["prev_action"]`; `cmd`, when supplied, is copied to
  `obs_cache["command"]`.
- `obs_cache` always carries `joint_pos` and `prev_action` (plus `command` only
  when a command was supplied).
- `extra` carries the 13 namespaced per-articulation record keys under
  `articulation/arm/<key>`, the two bare bookkeeping keys `time_in_episode` and
  `episode_id`, and the control keys `reset_env_ids` / `should_reset`.
- `perm_il2pol` / `perm_pol2il` are the identity permutation; policy joint
  limits are `(N, J)` float32 tensors.
- `close()` is idempotent.

Reset semantics:

- `reset_envs(ids)` preserves the given id order in its return value, resets
  only the selected rows, leaves every other row untouched, is a no-op for an
  empty id vector and fails fast for ids outside `[0, num_envs)`.
- `reset_all()` is `reset_envs(arange(num_envs))`.

Limitations (deliberate, no production seam is added for them):

- no physics, integration, contacts, feet or termination logic;
- no automatic reset (`reset_env_ids` is always empty, `should_reset` is
  always `False`);
- no cameras, depth, props, events, recorder, contacts, snapshots or actuator
  seam; no optional capability is advertised or silently faked;
- one articulation, one joint, CPU only, fixed identity joint order;
- episode time advances by one second per step and carries no sim-time meaning.

One fake only: do not add per-test fakes, a fake hierarchy or a configurable
universal fake. Add a new manifest row only when a new external boundary
appears.
