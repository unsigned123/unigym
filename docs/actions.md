# Actions: joint-addressed values and authoring templates

An action surface has two halves:

- an **authoring template**: `OrderedActionGroup` / `OrderedActionSpec`, which
  fixes declaration order, per-group joint order and target kind;
- **self-describing runtime values**: `NamedAction` /
  `tuple[NamedAction, ...]`, whose every column is addressed by its owning
  `(articulation, joint_name)` pair.

The runtime identity is **per column**, `(articulation, joint_name)`. Backends
route each column by that pair to the control/actuator that physically owns the
joint. They never use an action-group name, a group index or a global simulator
column order as an action identity.

Action groups survive as **authoring, policy-ownership, flat-layout and
diagnostic labels**. They are not backend routing identities and they are not
fields of the runtime value.

All names in this document refer to `unigym.runtime.actions` unless stated
otherwise.

## Runtime value: `NamedAction`

`NamedAction` is a frozen dataclass with exactly four fields:

| Field | Meaning |
| --- | --- |
| `articulation` | Owning articulation name (non-empty string). |
| `values` | `(N, D)` float tensor: one column per entry in `joint_names`. |
| `joint_names` | `D` joint names; `len(joint_names) == values.shape[1]`. |
| `kind` | One of `LEGAL_ACTION_KINDS`: `position`, `velocity`, `torque`, `effort`, `current`. |

There is no `group` field. There is no compatibility keyword and no alias.

One `NamedAction` covers one articulation and one homogeneous kind. It may
span several physical controls of that articulation when every addressed joint
expects that kind. Values are targets in the units of `kind` (rad for
`position`, rad/s for `velocity`, N·m for `torque`/`effort`, A for `current`).

## Authoring template: groups and spec

The template types are unchanged and are authoring-only:

| Type | Fields | Role |
| --- | --- | --- |
| `OrderedActionGroup` | `name`, `articulation`, `joints`, `target`, `role` | One named, declaration-ordered group. `role` is `body` or `aux` (default `aux`) and is never inferred from the name. |
| `OrderedActionSpec` | `groups` | A declaration-ordered sequence of groups. |
| `ActionDim` | `index`, `articulation`, `group`, `joint`, `target` | One flat-layout row. Its `group` is an authoring label. |

A template group is still an **authoring handle**: it names a joint batch, its
order and its kind. Its name is metadata. Two groups of one articulation must
own disjoint joints because two groups claiming one joint would be an
ambiguous template; that is a template rule, not a routing key.

`target` is always one of `LEGAL_ACTION_KINDS`. `joints` must be an explicit
non-string sequence of joint names. The action surface has no `joint_set`; that
is an *actuation* selector and `OrderedActionGroup.from_mapping` rejects it.

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
idx  articulation  group  joint           target
0    go1           legs   FL_hip_joint    position
1    go1           legs   FL_thigh_joint  position
...
12   go2           legs   FL_hip_joint    position
...
```

The `index` column is exactly the column index in the flat tensor. The `group`
column is the authoring label; it is not a runtime identity.

Three independent rules follow:

- The order of a `tuple[NamedAction, ...]` is **not** semantic. Entries are
  matched by `(articulation, joint_names, kind)`, not by sequence position.
- The order of each `NamedAction.joint_names` **is** semantic: it is the
  column order of `values`, one-to-one. Reordering `joint_names` without
  reordering the matching value columns changes the commanded action.
- `OrderedActionSpec.flatten` accepts any tuple order and any per-entry
  permutation of a template group's joint names, and restores template order.

## Converting: `unflatten` and `flatten`

`unflatten(flat)` emits one joint-addressed `NamedAction` per template group,
in declaration order:

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
named = spec.unflatten(flat)          # one joint batch per group, spec order
assert [entry.articulation for entry in named] == ["go1", "go2"]
assert [entry.joint_names for entry in named] == [JOINTS, JOINTS]
assert all(entry.kind == "position" for entry in named)
assert all(not hasattr(entry, "group") for entry in named)

round_trip = spec.flatten(named)      # (4, spec.dim), spec order
assert torch.equal(round_trip, flat)

# A value tuple may arrive in any order; flatten matches by joint identity.
assert torch.equal(spec.flatten((named[1], named[0])), flat)

# A per-entry joint permutation is legal and is undone in the flat result.
go1_permuted = NamedAction(
    articulation="go1",
    values=torch.flip(flat[:, : len(JOINTS)], dims=[1]),
    joint_names=tuple(reversed(JOINTS)),
    kind="position",
)
assert torch.equal(spec.flatten((named[1], go1_permuted)), flat)
```

`unflatten` slices views from the input tensor (`dtype` and device preserved,
no value copy). `flatten`:

1. requires a non-string sequence of `NamedAction`;
2. matches each entry to a template group by `(articulation, joint_names,
   kind)` using joint-set equality (so a per-entry permutation is legal);
3. fails loud when an entry matches no template, a template is matched twice,
   a template is left unaddressed, or `num_envs`/dtype/device disagree;
4. reorders each entry's columns by name into template joint order and
   concatenates in spec declaration order.

`flatten` is strict: it does not accept a subset, a superset or a split of one
template group. `project_groups` below is the tolerant projection used by
data/diagnostic consumers.

Other inspection helpers:

```python
rows = spec.dim_table()          # tuple[ActionDim, ...]
assert [row.index for row in rows] == list(range(spec.dim))
assert (rows[0].articulation, rows[0].group, rows[0].joint) == ("go1", "legs", JOINTS[0])
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

## Authoring projection: `project_groups`

`OrderedActionSpec.project_groups(actions)` returns one `ActionGroupValue` per
template group, in declaration order:

```python
projected = spec.project_groups(named)
assert [value.group.name for value in projected] == ["legs", "legs"]
assert torch.equal(projected[0].values, flat[:, : len(JOINTS)])
```

`ActionGroupValue.group` is the declared `OrderedActionGroup` and its `values`
are in that group's joint order. The projection matches **columns by joint
identity only**, so it tolerates an adapter that splits one group across
entries or merges several same-articulation, same-kind groups into one entry.
It fails loud on a missing, duplicated or wrong-kind joint column.

This is a data/diagnostic projection used by collect, watchers and trace
recording. Backends must not use it to route physics.

## Backend routing and coverage

Each backend builds a `JointActionIndex` from its physical control table once
per env. For every control row and every joint in `control.joint_names` it
records a `JointSlot` keyed by `(articulation, joint_name)`:

- `kind` comes from the control's `target`;
- `column` is the joint's index inside that control's `joint_names`;
- `required` and `prop` come from the control row.

Per tick the backend routes every `NamedAction` column:

1. a column whose `(articulation, joint_name)` is unknown fails naming the
   articulation and the joint, and lists the accepted joints for that
   articulation;
2. a column whose `kind` differs from the owning control's target fails naming
   the articulation, the joint, the entry kind and the expected target kind;
3. a joint addressed more than once in one tick fails naming the articulation
   and the joint;
4. every required control joint must be covered; a missing joint is named by
   articulation and joint names.

The authoritative error identity is always the articulation plus joint names,
never an action-group name. A control row's `name` appears only as optional
destination context.

One entry may span several physical controls of one articulation if all of the
addressed joints expect the same kind. Because a `NamedAction` has exactly one
`kind`, a merged entry cannot span different expected target kinds; that fails
per joint. This is stricter than a group-name-only world and is intentional.

Same joint names on different articulations are legal and independent: the
`(articulation, joint_name)` key disambiguates them. Prop joints are routable
the same way; a prop control's unique joint names resolve to that prop control,
and prop `required` semantics are preserved exactly.

## Identity and duplicate names

The identity of every routed column is `(articulation, joint_name)`, not a bare
joint name, not a group name and not a flat column index.

- A group name must be unique **within one articulation** in the authoring
  template.
- The same group name on different articulations is legal at the template and
  policy layers.
- Joint names are scoped by articulation, so two articulations may declare the
  same joint names. This is why `go1` and `go2` can both use the identical 12
  joint names without suffixing.
- Within one articulation, template groups must own disjoint joints.
- A `NamedAction` may not repeat a joint name inside its own `joint_names`.
- A `PolicyBinding` may not repeat an `(articulation, group)` identity.

These are contract-level rules. A backend still resolves each articulation's
names against its own asset and may have its own asset-name constraints.

## Two articulations with identical joint names: go1 and go2

Use one authoring group per articulation and the same 12 joint names on both:

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

`spec.dim` is 24: the first 12 flat columns are `go1`, the next 12 are `go2`.
The identical joint names are unambiguous because routing is keyed by
`(articulation, joint_name)`.

### Option 1: `ExternalRunner` with two self-describing values

`ExternalRunner` is caller-driven. Its validation surface is the union of every
adapter's `(articulation, joint) -> kind` partition; each addressed joint must
be owned exactly once per tick and the tuple order is irrelevant.

```python
from unigym.runner import ExternalRunner
from unigym.runtime.actions import NamedAction

runner = ExternalRunner.from_instance(instance)
runner.step(
    (
        NamedAction(
            articulation="go2",
            values=u_go2,                 # (num_envs, 12), GO_JOINTS order
            joint_names=GO_JOINTS,
            kind="position",
        ),
        NamedAction(
            articulation="go1",
            values=u_go1,                 # (num_envs, 12), GO_JOINTS order
            joint_names=GO_JOINTS,
            kind="position",
        ),
    )
)
```

Swapping the two entries does not change routing. `u_go1` and `u_go2` are
`(num_envs, 12)` tensors in `GO_JOINTS` order.

### Option 2: `SyncRunner` with one composite adapter owning both groups

Bind both `(articulation, group)` identities to one adapter. The adapter owns
the **joints** of those groups; its `postprocess_action` returns one
joint-addressed entry per articulation (or one merged entry per same-kind
articulation):

```python
from unigym.adapters.base import Adapter
from unigym.runtime.actions import PolicyBinding

policies = (
    PolicyBinding(
        adapter="go_legs_policy",
        groups=(("go1", "legs"), ("go2", "legs")),
        control_dt=0.02,
    ),
)


class GoLegsPolicy(Adapter):
    def postprocess_action(self, raw):
        return (
            self.named_action(raw[:, :12], articulation="go1",
                              joint_names=GO_JOINTS, kind="position"),
            self.named_action(raw[:, 12:], articulation="go2",
                              joint_names=GO_JOINTS, kind="position"),
        )
```

`SyncRunner.from_instance` accepts exactly one adapter key in
`env.adapter_groups`; several groups under that one adapter are supported.
Validation is against the adapter's joint partition, so one entry per group,
several groups merged into one same-articulation/kind entry, and a group split
across entries all remain legal as long as every owned joint is addressed
exactly once. `group=` is never passed to `NamedAction`.

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

- one composite adapter that owns both groups' joints and returns the
  corresponding `NamedAction` entries, or
- `ExternalRunner`, where the caller runs each policy and supplies the combined
  action tuple.

A true multi-policy `SyncRunner` with separate per-adapter action providers is
not implemented.

## Action groups are not actuation groups

`OrderedActionGroup` and `ActuationGroup` remain orthogonal declarations:

- `OrderedActionGroup` is the policy-facing **authoring** contract:
  articulation, group name, explicit joint order, target kind and body/aux
  role.
- `ActuationGroup` is the physics-facing contract: law, joints or `joint_set`,
  gains, limits, command parameters and parts.

They do not share identity, names, joint order or ownership. An action group
named `legs` does not require or create an actuation group named `legs`. The
action surface has no `joint_set`; actuation may use one.

Action/actuation orthogonality is unchanged by joint-addressed routing. The
relationship is per joint and only about compatibility between the action
`target` and the resolved actuation law:

| Action target | Resolved actuation law(s) | Derived drive |
| --- | --- | --- |
| `position` | `standard_pd`, `computed_torque_pd` | `position_pd` |
| `position` | `native_servo` | `position_servo` |
| `velocity` | `direct_velocity` | `velocity` |
| `torque` | `direct_torque` | `torque` |

The table is the built-in baseline. A pack may register an
`ActuatorLawDescriptor` whose `drives` names an existing drive; such a law is
derived exactly like a built-in one (the runtime drive is the drive common to
every resolved law that serves the target). The drive and target vocabularies
stay closed: a descriptor drive absent from `DRIVE_TARGET` cannot be selected.

`effort` and `current` are legal `NamedAction.kind` values, but no built-in
direct-drive actuation law serves them today, so the build fails with a teaching
error instead of inventing a drive. A pack may register a descriptor that serves
one of those drives; the closed `DRIVE_TARGET` vocabulary still owns the target
mapping.

There is no 1:1 requirement in either direction. One authored action group may
cover joints served by several actuation groups, and an actuation group may
cover joints that belong to several action groups.

`ControlRuntime` remains the derived **physical/build-time plan**: one row per
authored action group today, carrying articulation, joint names, target, drive,
q0, addresses and backend strategy. Its `(articulation, name)` key is a
physical control-row label for `env.controls` / actuator feeds. It is **not** an
action identity, and backend routing never uses it to match an action column;
routing always uses the addressed joint.

```python
from unigym.runtime.controllers import build_control_runtimes

rows = build_control_runtimes(
    spec,
    policies,
    articulations,
    profiles,
    actuation_plans,
    default_control_dt=0.02,   # the adapter fallback period when no binding sets one
)
# one physical row per OrderedActionGroup, in action-spec order
```

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

## `PolicyBinding`: authoring-only ownership and `control_dt`

A `PolicyBinding` declares which adapter produces which authored groups. It is
a policy/authoring declaration; it is not a runtime routing table:

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

- `articulation` is a non-empty string.
- `values` is a `torch.Tensor` of rank 2 with shape `(N, D)`.
- `joint_names` is a non-empty sequence (not a bare string), with no duplicate
  names and exactly `D` entries.
- `kind` is one of `LEGAL_ACTION_KINDS`: `position`, `velocity`, `torque`,
  `effort`, `current`.

```python
from unigym.runtime.actions import NamedAction

action = NamedAction(
    articulation="go1",
    values=torch.zeros(2, 12, dtype=torch.float32),  # (num_envs, 12)
    joint_names=GO_JOINTS,                          # same 12, same order
    kind="position",
)

# Convenience constructor for an all-zero target.
zeros = NamedAction.zeros(2, "go1", GO_JOINTS, "position")
assert zeros.values.shape == (2, len(GO_JOINTS))
```

Construction checks the tensor shape, kind and name/column agreement; it does
not clip, reorder or validate values against a spec. Coverage is checked by the
consumer:

- `OrderedActionSpec.flatten` requires every template group to be matched
  exactly once by `(articulation, joint-name set, kind)`, any tuple order and a
  per-entry permutation.
- `SyncRunner` / `ExternalRunner` require exactly the adapter-owned
  `(articulation, joint)` columns (or their union for the external caller),
  each addressed once, with `kind` matching the owning control's target.
- A backend requires every required control joint to be covered.
- The simulation facade requires every declared authoring group to be staged
  before `step()`.

The safe way to change column order is to reorder the value columns together
with `joint_names`, or to go through `OrderedActionSpec.flatten`, which restores
template order by name.

## Collect metadata and recorded columns

`meta["action_groups"]` keeps its playback-v3 key and value shape
(`{key, column, articulation, group, joint_names, kind}`), but its source is
**not** runtime action entries:

- it is the effective authoring table: `spec.action_spec.groups` plus
  backend-declared prop groups (`OrderedActionGroup(..., role="aux")`);
- it is written once at sink construction and does not depend on the first
  tick's runtime entries;
- `action_target/<group>` columns are projected from routed actions onto that
  table (in group-joint order), tolerating split/merged runtime entries via
  `project_groups`.

A sink built without an authoring table (test doubles / direct construction)
falls back to one deterministic joint-addressed column per runtime entry,
`action_target/<articulation>/<comma-joined joint_names>`. Normal `Run`-built
runs always supply the authoring projection and keep the historical
`action_target/<group>` segments.

## Related modules

- `unigym/runtime/actions.py` — templates, joint-addressed values, bindings and
  converters.
- `unigym/runtime/action_routing.py` — `JointActionIndex` per-joint routing.
- `unigym/runtime/controls.py` — closed drive/target table.
- `unigym/runtime/controllers.py` — `ControlRuntime` physical-plan derivation.
- `unigym/actuators/ir.py` — `ActuationGroup` declarations.
- `unigym/runner/sync.py` / `unigym/runner/external.py` — joint-partition
  validation and tick loops.
- [`sim_facade.md`](sim_facade.md) — facade-level `act` / `step` rules.
- [`examples/ordered_actions.py`](../examples/ordered_actions.py) — executable
  conversion, joint-addressed lowering and authoring-table example
  (`PYTHONPATH=. .venv/bin/python examples/ordered_actions.py`, or the
  environment Python after activating the project environment; a bare `python`
  may lack the project dependencies).
