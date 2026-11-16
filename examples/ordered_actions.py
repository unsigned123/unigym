"""Ordered action templates and named values, without an engine.

Run from the repository root::

    PYTHONPATH=. python examples/ordered_actions.py

Two articulations (``go1`` and ``go2``) declare the same 12 joint names.  The
example shows the template/value round-trip, identity-based ordering, a joint
permutation, and the policy binding that owns both groups.
"""

from __future__ import annotations

import torch

from unigym.runtime.actions import (
    NamedAction,
    OrderedActionGroup,
    OrderedActionSpec,
    PolicyBinding,
    resolve_control_dt,
)

GO_JOINTS = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)


def main() -> None:
    """Run the conversion and ownership examples."""
    spec = OrderedActionSpec(
        (
            OrderedActionGroup("legs", "go1", GO_JOINTS, "position", role="body"),
            OrderedActionGroup("legs", "go2", GO_JOINTS, "position", role="body"),
        )
    )
    assert spec.dim == 24

    flat = torch.arange(2 * spec.dim, dtype=torch.float32).reshape(2, spec.dim)
    named = spec.unflatten(flat)  # go1, go2 in declaration order
    assert [entry.articulation for entry in named] == ["go1", "go2"]

    # The value tuple order does not matter: flatten matches by identity.
    assert torch.equal(spec.flatten((named[1], named[0])), flat)

    # A per-entry joint permutation is legal and is undone in the flat result.
    go1_permuted = NamedAction(
        articulation="go1",
        group="legs",
        values=torch.flip(flat[:, : spec.dim // 2], dims=[1]),
        joint_names=tuple(reversed(GO_JOINTS)),
        kind="position",
    )
    assert torch.equal(spec.flatten((named[1], go1_permuted)), flat)

    # The dim table is the flat layout: index, identity, joint and target.
    table = spec.dim_table()
    assert [row.index for row in table] == list(range(spec.dim))
    assert (table[0].articulation, table[0].joint) == ("go1", GO_JOINTS[0])
    assert (table[12].articulation, table[12].joint) == ("go2", GO_JOINTS[0])

    # One adapter owns both groups; bindings are the timing authority.
    binding = PolicyBinding(
        adapter="go_legs_policy",
        groups=(("go1", "legs"), ("go2", "legs")),
        control_dt=0.02,
    )
    assert resolve_control_dt((binding,), "go_legs_policy", fallback=0.01) == 0.02

    zeros = NamedAction.zeros(2, "go1", "legs", GO_JOINTS, "position")
    assert zeros.values.shape == (2, len(GO_JOINTS))

    print(spec)
    print(f"\nspec.dim={spec.dim} groups={[(g.articulation, g.name) for g in spec.groups]}")


if __name__ == "__main__":
    main()
