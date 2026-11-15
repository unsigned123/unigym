"""The one shared simulation-facade acceptance script body.

Every backend lane runs :func:`run_facade_acceptance` unchanged: the function
builds a real ``Simulation`` from explicit declarations, reads the spawn
observation, stages one position action per policy tick, steps, and checks the
typed projection against the backend state.  The body has no backend branch:
``backend`` is only forwarded to ``SimulationBuilder``, and ``world`` /
``pack_set`` / ``robot_id`` / ``expected_joint_names`` are lane-provisioned
fixtures (the plan's B1: "same script, lane-owned fixture").

The acceptance evidence is the returned mapping: the resolved backend,
articulation label, joint order, producing tick index, env count and final joint
positions.  Lanes assert lane-specific facts on that evidence; they never fork
the body.

``num_envs`` defaults to one (the v1 single-env acceptance).  Above one it
stages one explicit ``(N, joint_count)`` target per tick whose rows differ by a
deterministic ``0.02 * row`` offset, so the same body proves per-row staging
and per-row observations without a backend branch.
"""

from __future__ import annotations

import torch

from unigym.packs.packset import PackSet, RobotRef
from unigym.sim import BindingOrder, Control, PendingActionError, SimulationBuilder
from unigym.world import World

__all__ = ["ARTICULATION_LABEL", "run_facade_acceptance"]

ARTICULATION_LABEL = "body"
"""The shared articulation label: it equals the ``body`` control face.

The backend contract requires one control whose ``action_face`` equals its
articulation (the body face), so the acceptance articulation is labelled
``body`` and the run config routes it to the pack profile through
``params.profile``.
"""


def _robot_ref(pack_set: PackSet, robot_id: str) -> RobotRef:
    """The one declared robot ref with this id, across the explicit pack set."""
    matches = [ref for pack in pack_set.packs() for ref in pack.robots() if ref.id == robot_id]
    if not matches:
        declared = sorted(ref.id for pack in pack_set.packs() for ref in pack.robots())
        raise AssertionError(
            f"facade acceptance robot {robot_id!r} is not declared by the pack set; "
            f"declared robots: {declared}"
        )
    if len(matches) > 1:
        raise AssertionError(
            f"facade acceptance robot {robot_id!r} is declared by several packs: "
            f"{sorted(ref.pack for ref in matches)}"
        )
    return matches[0]


def _position_target(joint_count: int, step: int, num_envs: int) -> torch.Tensor:
    """A deterministic ``(num_envs, joint_count)`` position target for one tick.

    Every row carries the same base pattern scaled by the tick index and is
    offset by ``0.02 * row``, so the rows of one tick are distinct by
    construction and the divergence is reproducible without randomness.
    """
    pattern = torch.linspace(-0.05, 0.05, joint_count, dtype=torch.float32)
    base = (pattern * float(step + 1)).reshape(1, joint_count)
    offsets = 0.02 * torch.arange(num_envs, dtype=torch.float32).reshape(num_envs, 1)
    return base + offsets


def run_facade_acceptance(
    *,
    pack_set: PackSet,
    robot_id: str,
    world: World,
    backend: str,
    expected_joint_names: tuple[str, ...],
    steps: int = 5,
    device: str = "cpu",
    num_envs: int = 1,
) -> dict[str, object]:
    """Run the shared build/observe/act/step acceptance and return its evidence.

    Args:
        pack_set: the explicit pack scope naming the robot.
        robot_id: the robot ref id declared by one pack of ``pack_set``.
        world: the lane-owned world declaration (the world owns the asset for
            the MJCF lane; an IsaacLab lane declares its own ground).
        backend: the registered backend name forwarded to the builder.
        expected_joint_names: the robot body joint order the observation must
            report, column for column.
        steps: the number of staged-action ticks to run (default 5).
        device: the backend device label (MuJoCo is CPU; IsaacLab needs cuda).
        num_envs: the environment count to build and drive (default 1).  Each
            tick stages one explicit ``(num_envs, joint_count)`` target, so a
            multi-env build exercises the same body as the single-env one.

    Returns:
        Evidence mapping: ``backend``, ``articulation``, ``joint_names``,
        ``steps``, ``t``, ``num_envs`` and ``final_joint_pos``.  With
        ``num_envs == 1`` ``final_joint_pos`` keeps the v1 flat row; above one
        env it is one flat vector per env row, in row order.

    Raises:
        AssertionError: any contract disagreement in the acceptance body; one
            case is a reused action, which the body reports by asserting that
            :meth:`Simulation.step` with an empty pending table raises
            :class:`~unigym.sim.errors.PendingActionError`.
    """
    joint_names = tuple(expected_joint_names)
    if not joint_names:
        raise AssertionError("facade acceptance needs a non-empty expected joint order")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise AssertionError(f"facade acceptance steps must be an int >= 1, got {steps!r}")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise AssertionError(f"facade acceptance num_envs must be an int >= 1, got {num_envs!r}")
    control = Control.position_pd(joint_set="body", kp=20.0, kd=1.0)
    builder = SimulationBuilder(
        backend=backend,
        dt=0.02,
        physics_dt=0.005,
        pack_set=pack_set,
        device=device,
        num_envs=num_envs,
    )
    builder.world = world
    robot = builder.add_articulation(
        _robot_ref(pack_set, robot_id), name=ARTICULATION_LABEL, control=control
    )

    sim = builder.build()
    try:
        expected_shape = (num_envs, len(joint_names))
        obs = sim.observe()
        record = obs.articulations[robot]
        prefix = f"articulation/{robot.name}"
        raw_pos = obs.raw[f"{prefix}/joint_pos"]
        assert isinstance(raw_pos, torch.Tensor)
        assert (
            obs.num_envs == num_envs
        ), f"spawn observation num_envs {obs.num_envs} != declared {num_envs}"
        assert record.joint_names == joint_names, (
            f"spawn observation joint order {record.joint_names} != declared "
            f"profile order {joint_names}"
        )
        assert (
            raw_pos.shape == expected_shape
        ), f"spawn joint_pos shape {tuple(raw_pos.shape)} != {expected_shape}"
        assert torch.equal(
            record.joint_pos, raw_pos
        ), "spawn typed joint_pos is not the backend state tensor"

        binding = robot.bind_action("body", kind="position", order=BindingOrder.DEFAULT)
        assert (
            binding.joint_names == joint_names
        ), f"action binding columns {binding.joint_names} != profile order {joint_names}"

        for index in range(steps):
            sim.act(binding, _position_target(len(joint_names), index, num_envs))
            obs = sim.step()
            record = obs.articulations[robot]
            raw_pos = obs.raw[f"{prefix}/joint_pos"]
            raw_vel = obs.raw[f"{prefix}/joint_vel"]
            assert isinstance(raw_pos, torch.Tensor)
            assert isinstance(raw_vel, torch.Tensor)
            assert obs.t == index, f"step {index}: producing tick {obs.t} != {index}"
            assert (
                obs.num_envs == num_envs
            ), f"step {index}: observation num_envs {obs.num_envs} != {num_envs}"
            assert record.joint_names == joint_names
            assert raw_pos.shape == record.joint_pos.shape == expected_shape, (
                f"step {index}: joint_pos shapes {tuple(raw_pos.shape)} / "
                f"{tuple(record.joint_pos.shape)} != {expected_shape}"
            )
            assert raw_vel.shape == record.joint_vel.shape == expected_shape, (
                f"step {index}: joint_vel shapes {tuple(raw_vel.shape)} / "
                f"{tuple(record.joint_vel.shape)} != {expected_shape}"
            )
            assert torch.equal(
                record.joint_pos, raw_pos
            ), f"step {index}: typed joint_pos is not the backend state tensor"
            assert torch.equal(
                record.joint_vel, raw_vel
            ), f"step {index}: typed joint_vel is not the backend state tensor"
            assert obs.reset_env_ids.tolist() == [], (
                f"step {index}: step reported reset_env_ids {obs.reset_env_ids.tolist()}; "
                f"a step never resets a row (due ids belong in reset_due_env_ids)"
            )
            try:
                sim.step()
            except PendingActionError:
                pass
            else:
                raise AssertionError(
                    f"step {index}: step() without a staged action must raise "
                    f"PendingActionError (the previous action is never reused)"
                )

        rows = [[float(value) for value in row] for row in record.joint_pos.tolist()]
        return {
            "backend": sim.backend,
            "articulation": robot.name,
            "joint_names": list(record.joint_names),
            "steps": steps,
            "t": sim.t,
            "num_envs": num_envs,
            "final_joint_pos": rows[0] if num_envs == 1 else rows,
        }
    finally:
        sim.close()
