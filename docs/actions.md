# Actions: ordered templates and named values

An action surface has two halves:

- an **authoring template**: `OrderedActionGroup` / `OrderedActionSpec`, which
  fixes group order, joint order and target kind;
- **self-describing values**: `NamedAction` / `tuple[NamedAction, ...]`, which
  carry the actual targets together with the identity and joint names they
  address.

Flat tensors are a serialization of the template, not the semantic contract.
Consumers route by `(articulation, group)` and resolve joint names; they never
guess a column from its position in a global simulator order.

All names in this document refer to `unigym.runtime.actions` unless stated
otherwise.

## Template/value duality

| Template | Value | Scope |
| --- | --- | --- |
| `OrderedActionGroup` | `NamedAction` | one action group |
| `OrderedActionSpec` | `tuple[NamedAction, ...]` | one complete action surface |

The field correspondence for one group is:

| `OrderedActionGroup` | `NamedAction` | Meaning |
| --- | --- | --- |
| `articulation` | `articulation` | owning articulation |
| `name` | `group` | group name within that articulation |
| `joints` | `joint_names` | ordered joint names |
| `target` | `kind` | target kind (`position`, `velocity`, ...) |
| `role` | — | backend body/aux routing hint, values have no role |

`target` is always one of `LEGAL_ACTION_KINDS`; `role` is `body` or `aux`
(default `aux`) and is never inferred from the group name.

`NamedAction.values` is the value tensor for that group: one column per entry
of `joint_names`, in the same order. A spec converts to values with
`unflatten`; values convert back to a spec-ordered flat tensor with `flatten`.

## Order semantics

`OrderedActionSpec` preserves declaration order. Flattening walks:

1. every group in `spec.groups` order, then
2. every joint in that group's `joints` order.

That traversal is the single source of truth for `dim_table`, `unflatten` and
`flatten`. A flat vector therefore has the layout

```text
[ group0.joint0, group0.joint1, ..., group1.joint0, ... ]
```

`dim_table()` returns one `ActionDim` row per flat column. `str(spec)` (and
therefore `print(spec)`) renders the same table:

```text
idx  articulation  group  joint        target
0    go1           legs   FL_hip_joint position
1    go1           legs   FL_thigh_joint position
...
12   go2           legs   FL_hip_joint position
...
```

The `index` column is exactly the column index in the flat tensor.

Three independent rules follow from that layout:

- The order of a `tuple[NamedAction, ...]` is **not** semantic. Entries are
  matched by `(articulation, group)` identity.
- The order of each `NamedAction.joint_names` **is** semantic: it is the column
  order of `values`, one-to-one. Reordering `joint_names` without reordering
  the matching value columns changes the commanded action.
- `OrderedActionSpec.flatten` accepts any tuple order and any per-entry
  permutation of that group's joint names, and restores the template order.

## Converting and inspecting

```python
import torch

from unigym.runtime.actions import NamedAction, OrderedActionGroup, OrderedActionSpec

JOINTS = ("hip_joint", "knee_joint", "ankle_joint")
spec = OrderedActionSpec(
    (
        OrderedActionGroup("legs", "go1", JOINTS, "position", role="body"),
        OrderedActionGroup("legs", "go2", JOINTS, "position", role="body"),
    )
)

flat = torch.arange(4 * spec.dim, dtype=torch.float32).reshape(4, spec.dim)
named = spec.unflatten(flat)          # one NamedAction per group, spec order
round_trip = spec.flatten(named)      # (4, spec.dim), spec order

assert len(named) == 2
assert [entry.articulation for entry in named] == ["go1", "go2"]
assert torch.equal(round_trip, flat)

# A value tuple may arrive in any order; flatten matches by identity.
assert torch.equal(spec.flatten((named[1], named[0])), flat)

# A per-entry joint permutation is legal and is undone in the flat result.
go1 = NamedAction(
    articulation="go1",
    group="legs",
    values=torch.flip(flat[:, : len(JOINTS)], dims=[1]),
    joint_names=tuple(reversed(JOINTS)),
    kind="position",
)
assert torch.equal(spec.flatten((named[1], go1)), flat)
```

`unflatten` slices views from the input tensor (`dtype` and device preserved,
no value copy). `flatten` checks that all entries agree on `num_envs`, dtype and
device, then concatenates in spec order. The two are bitwise inverses.

Other inspection helpers:

```python
rows = spec.dim_table()          # tuple[ActionDim, ...]
assert [row.index for row in rows] == list(range(spec.dim))
assert spec.dim == sum(len(group.joints) for group in spec.groups)
assert repr(spec).startswith("OrderedActionSpec(dim=")
print(spec)                      # the idx/articulation/group/joint/target table
```

Templates can also be parsed from explicit mappings for config-style code:

```python
group = OrderedActionGroup.from_mapping(
    {
        "name": "legs",
        "articulation": "go1",
        "joints": ["hip_joint", "knee_joint", "ankle_joint"],
        "target": "position",
        "role": "body",
    }
)
spec = OrderedActionSpec.from_mapping([group])
```

`joints` must be an explicit non-string sequence of joint names (a tuple, list
or any other `Sequence`). `joint_set` is an actuation selector and is rejected
by `OrderedActionGroup.from_mapping`.

## Identity and duplicate names

The identity of every group and value is `(articulation, group)`, not a bare
group name and not a flat column index.

- A group name must be unique **within one articulation**.
- The same group name on different articulations is legal.
- Joint names are scoped by articulation, so two articulations may declare the
  same joint names. This is why `go1` and `go2` can both use the identical 12
  joint names without suffixing.
- Within one articulation, action groups must own disjoint joints. The spec
  rejects overlap because routing a joint to two groups would be ambiguous.
- A `NamedAction` may not repeat a joint name inside its own `joint_names`.
- A `PolicyBinding` may not repeat an `(articulation, group)` identity.

These are contract-level rules. A backend still resolves each articulation's
names against its own asset and may have its own asset-name constraints.

## Two articulations with identical joint names: go1 and go2

Use one group name per articulation and the same 12 joint names on both:

```python
GO_JOINTS = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)

spec = OrderedActionSpec(
    (
        OrderedActionGroup("legs", "go1", GO_JOINTS, "position", role="body"),
        OrderedActionGroup("legs", "go2", GO_JOINTS, "position", role="body"),
    )
)
```

`spec.dim` is 24: the first 12 columns are `go1`, the next 12 are `go2`.
The identical names are unambiguous because `(articulation, group)` scopes
them.

### Option 1: `ExternalRunner` with two self-describing values

`ExternalRunner` is caller-driven. Its validation surface is the union of every
policy binding's groups; each declared identity must appear exactly once, but
the tuple order is irrelevant.

```python
from unigym.runner import ExternalRunner
from unigym.runtime.actions import NamedAction

runner = ExternalRunner.from_instance(instance)
runner.step(
    (
        NamedAction("go2", "legs", u_go2, GO_JOINTS, "position"),
        NamedAction("go1", "legs", u_go1, GO_JOINTS, "position"),
    )
)
```

Here `u_go1` and `u_go2` are `(num_envs, 12)` tensors in `GO_JOINTS` order.
Swapping the two entries does not change routing.

### Option 2: `SyncRunner` with one shared adapter owning both groups

Bind both identities to one adapter and have its `postprocess_action` return
both values every tick:

```python
from unigym.runtime.actions import PolicyBinding

policies = (
    PolicyBinding(
        adapter="go_legs_policy",
        groups=(("go1", "legs"), ("go2", "legs")),
        control_dt=0.02,
    ),
)
```

`SyncRunner.from_instance` accepts exactly one adapter key in
`env.adapter_groups`; several groups under that one adapter are supported. The
adapter must produce one `NamedAction` for every group it owns.

### Option 3: different policies for different articulations

The authoring layer can express a different owner per articulation:

```python
policies = (
    PolicyBinding("go1_policy", (("go1", "legs"),)),
    PolicyBinding("go2_policy", (("go2", "legs"),)),
)
```

This is a legal partition: each group has exactly one owner and every group has
an owner. However, the current `SyncRunner` is a single-adapter loop. A run
whose `env.adapter_groups` has more than one adapter is refused with a teaching
error. The currently supported multi-policy shape is therefore:

- one composite adapter that owns both groups and returns both `NamedAction`
  entries, or
- `ExternalRunner`, where the caller runs each policy and supplies the combined
  action tuple.

A true multi-policy `SyncRunner` with separate per-adapter action providers is
not implemented.

## Action groups are not actuation groups

`OrderedActionGroup` and `ActuationGroup` are orthogonal declarations:

- `OrderedActionGroup` is the policy-facing contract: articulation, group name,
  explicit joint order, target kind and body/aux role.
- `ActuationGroup` is the physics-facing contract: law, joints or `joint_set`,
  gains, limits, command parameters and parts.

They do not share identity, names, joint order or ownership. An action group
named `legs` does not require or create an actuation group named `legs`. The
action surface has no `joint_set`; actuation may use one.

The relationship is per joint and only about compatibility between the action
`target` and the resolved actuation law:

| Action target | Resolved actuation law(s) | Derived drive |
| --- | --- | --- |
| `position` | `standard_pd`, `computed_torque_pd` | `position_pd` |
| `position` | `native_servo` | `position_servo` |
| `velocity` | `direct_velocity` | `velocity` |
| `torque` | `direct_torque` | `torque` |

`effort` and `current` are legal `NamedAction.kind` values, but no direct-drive
actuation law is implemented for them today, so the build fails with a teaching
error instead of inventing a drive.

There is no 1:1 requirement in either direction. One action group may cover
joints served by several actuation groups, and an actuation group may cover
joints that belong to several action groups. The build resolves the laws per
joint and derives one `ControlRuntime` per action group:

```python
from unigym.runtime.controllers import build_control_runtimes

rows = build_control_runtimes(
    spec,
    policies,
    articulations,
    profiles,
    actuation_plans,
    default_control_dt=0.02,   # the adapter fallback period, when no binding sets one
)
# one row per OrderedActionGroup, in action-spec order
```

`ControlRuntime` is derived, not authored. Its `name` and `articulation` are
the action-group identity, its `adapter` comes from the owning
`PolicyBinding`, its `target` comes from the action group, its `drive` is
derived from target plus actuation laws, and its `control_dt` mirrors the
resolved policy clock. Backends consume these rows for addressing and drive
strategy; there is no separate action identity beside `(articulation, name)`.

### `q0` ownership

Action groups carry no `q0`: a standby position is not an authored
action-group field. It is resolved per joint into each derived
`ControlRuntime.q0` from articulation/joint and robot-profile facts, with this
precedence:

1. the articulation joint's declared `init` (`ArticulationSpec.joints[*].init`),
2. the robot profile's `default_joint_pos[joint]` (`RobotProfile`),
3. `0.0`, only for position drives (`position_pd` / `position_servo`).

A velocity or torque drive never falls back to zero: a joint with neither an
articulation init nor a profile default is a `ControlError`, not an invented
target. The resolution lives in `unigym/runtime/controllers.py::_control_q0`.

## `PolicyBinding`: ownership and `control_dt`

A `PolicyBinding` declares which adapter produces which groups:

```python
binding = PolicyBinding(
    adapter="go_legs_policy",
    groups=(("go1", "legs"), ("go2", "legs")),
    control_dt=0.02,
    agent="planner",
)
```

| Field | Rule |
| --- | --- |
| `adapter` | Non-empty adapter name that produces the values. |
| `groups` | Non-empty, unique `(articulation, group)` identities; order is preserved but is not an action layout. |
| `control_dt` | Optional positive finite period in seconds. |
| `agent` | Optional non-empty metadata; it does not route actions. |

Ownership rules:

- Every declared action group must be owned by exactly one binding.
- A group may not be claimed by two bindings.
- An adapter may own several groups; a group still has one owner.

`control_dt` is the timing authority. It overrides the adapter's own
`control_dt`; the adapter value is only the explicit fallback when no binding
declares a period:

```python
from unigym.runtime.actions import PolicyBinding, resolve_control_dt

binding = PolicyBinding("go_legs_policy", (("go1", "legs"),), control_dt=0.02)
assert resolve_control_dt((binding,), "go_legs_policy", fallback=0.01) == 0.02

plain = PolicyBinding("go_legs_policy", (("go1", "legs"),))
assert resolve_control_dt((plain,), "go_legs_policy", fallback=0.01) == 0.01
```

If bindings of one adapter declare conflicting periods, resolution fails; it
never picks one silently. If neither a binding nor the adapter supplies a
period, resolution also fails instead of inventing a default.

## Constructing `NamedAction` by hand

Manual construction is supported and is the normal path for external drivers.
The constructor validates the following:

- `articulation` and `group` are non-empty strings.
- `values` is a `torch.Tensor` of rank 2 with shape `(N, dim)`.
- `joint_names` is a non-empty sequence (not a bare string), with no duplicate
  names and exactly `dim` entries.
- `kind` is one of `LEGAL_ACTION_KINDS`: `position`, `velocity`, `torque`,
  `effort`, `current`.

```python
from unigym.runtime.actions import NamedAction

action = NamedAction(
    articulation="go1",
    group="legs",
    values=torch.zeros(2, 12, dtype=torch.float32),  # (num_envs, 12)
    joint_names=GO_JOINTS,                           # same 12, same order
    kind="position",
)

# Convenience constructor for an all-zero target.
zeros = NamedAction.zeros(2, "go1", "legs", GO_JOINTS, "position")
```

Values are targets in the units of `kind` (rad for `position`, rad/s for
`velocity`, N·m for `torque`/`effort`, A for `current`). Construction checks the
tensor shape and name agreement; it does not clip, reorder or validate values
against a spec. Coverage is checked by the consumer:

- `OrderedActionSpec.flatten` requires exactly one entry per spec identity, a
  joint-name permutation per group, and a matching `kind`.
- `SyncRunner` requires exactly the groups owned by its adapter, each once, with
  `kind` matching the control group's configured target.
- The simulation facade requires every declared group to be staged before
  `step()`.

The safe way to change column order is to reorder the value columns together
with `joint_names`, or to go through `OrderedActionSpec.flatten`, which
restores template order by name.

## Related modules

- `unigym/runtime/actions.py` — templates, values, bindings and converters.
- `unigym/runtime/controls.py` — closed drive/target table.
- `unigym/runtime/controllers.py` — `ControlRuntime` derivation.
- `unigym/actuators/ir.py` — `ActuationGroup` declarations.
- `unigym/runner/sync.py` / `unigym/runner/external.py` — action validation and
  tick loops.
- [`sim_facade.md`](sim_facade.md) — facade-level `act` / `step` rules.
- [`examples/ordered_actions.py`](../examples/ordered_actions.py) — executable
  conversion and ownership example (`PYTHONPATH=. .venv/bin/python
  examples/ordered_actions.py`, or the environment Python after activating the
  project environment; a bare `python` may lack the project dependencies).
