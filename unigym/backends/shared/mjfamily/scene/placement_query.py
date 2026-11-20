"""Private MuJoCo free-root probes for initializer reset requests.

The backend half of :class:`unigym.initializers.ResetPrimitives.probe`: copy
the live reset ``MjData`` into a private scratch object, apply the requested
free-root overrides, run ``mj_forward`` and measure signed geom distances with
``mj_geomDistance``.  The live data, its qpos/qvel and the model are never
mutated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from unigym.initializers import (
    CollisionPairSpec,
    ProbeCapabilityError,
    ProbeRequest,
    ProbeResult,
)

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - capability is queried at runtime
    mujoco = None
    _MUJOCO_IMPORT_ERROR = exc

__all__ = [
    "ProbeError",
    "probe_free_poses",
]


class ProbeError(ValueError):
    """Malformed probe input (shape or type)."""


def _require_mujoco() -> None:
    if mujoco is None:  # pragma: no cover
        raise ProbeCapabilityError(
            f"MuJoCo placement probes are unavailable: {_MUJOCO_IMPORT_ERROR}"
        )


def _geom_id(model, name: str, *, field: str) -> int:
    assert mujoco is not None  # caller ran _require_mujoco
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if value < 0:
        raise ProbeCapabilityError(f"ProbeRequest.{field} geom {name!r} was not found")
    return int(value)


def _geom_name(model, geom: int, *, field: str) -> str:
    assert mujoco is not None  # caller ran _require_mujoco
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom))
    if not name:
        raise ProbeCapabilityError(
            f"ProbeRequest.{field}: collision geom id {int(geom)} has no name; "
            "automatic probe pairing needs named collision geoms"
        )
    return name


def _body_id(model, name: str, *, field: str) -> int:
    assert mujoco is not None  # caller ran _require_mujoco
    value = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if value < 0:
        raise ProbeCapabilityError(f"ProbeRequest.{field} body {name!r} was not found")
    return int(value)


def _is_collision_geom(model, geom: int) -> bool:
    """The model's collision truth: contype/conaffinity nonzero."""
    return bool(int(model.geom_contype[geom]) or int(model.geom_conaffinity[geom]))


def _subtree_bodies(model, root_body: int) -> set[int]:
    """The free root body plus every descendant body (parent ids precede)."""
    subtree = {int(root_body)}
    for body in range(int(root_body) + 1, int(model.nbody)):
        if int(model.body_parentid[body]) in subtree:
            subtree.add(body)
    return subtree


def _body_geoms(model, body_id: int, *, field: str) -> list[str]:
    """One body's named collision geoms (fail honestly on unnamed geoms)."""
    geoms: list[str] = []
    for geom in range(
        int(model.body_geomadr[body_id]),
        int(model.body_geomadr[body_id]) + int(model.body_geomnum[body_id]),
    ):
        if _is_collision_geom(model, geom):
            geoms.append(_geom_name(model, geom, field=field))
    return geoms


def _automatic_pairs(
    model,
    request: ProbeRequest,
    roots: Sequence[tuple[int, int]],
) -> tuple[CollisionPairSpec, ...]:
    """Pair every overridden owner's full subtree against the collision world.

    Each owner contributes its root body and all descendant bodies as
    candidates; same-owner internal pairs are excluded.  Different owners
    are cross-paired so overlapping overrides are detected.  The world set
    is every collision geom outside all overridden subtrees.  When
    ``support_body`` is declared, only pairs against that exact body allow
    a near-zero contact; it is never silently removed from the world set.
    """
    subtrees = [_subtree_bodies(model, body_id) for body_id, _qadr in roots]
    overridden_bodies: set[int] = set()
    for subtree in subtrees:
        overridden_bodies.update(subtree)

    owner_geoms: list[list[str]] = []
    for index, (pose, subtree) in enumerate(zip(request.overrides, subtrees)):
        field = f"overrides[{index}].owner {pose.owner!r}"
        geoms = [
            name for body_id in sorted(subtree) for name in _body_geoms(model, body_id, field=field)
        ]
        if not geoms:
            raise ProbeCapabilityError(f"ProbeRequest.{field} has no collision geoms to probe")
        owner_geoms.append(geoms)

    support_id: int | None = None
    if request.support_body is not None:
        support_id = _body_id(model, request.support_body, field="support_body")
    world_geoms = [
        geom
        for geom in range(int(model.ngeom))
        if _is_collision_geom(model, geom) and int(model.geom_bodyid[geom]) not in overridden_bodies
    ]

    pairs: list[CollisionPairSpec] = []
    for index, geoms in enumerate(owner_geoms):
        for candidate_geom in geoms:
            for other in world_geoms:
                pairs.append(
                    CollisionPairSpec(
                        candidate_geom,
                        _geom_name(model, other, field="automatic world pair"),
                        allow_support_contact=(
                            support_id is not None and int(model.geom_bodyid[other]) == support_id
                        ),
                    )
                )
        for other_index in range(index + 1, len(owner_geoms)):
            for candidate_geom in geoms:
                for other in owner_geoms[other_index]:
                    pairs.append(CollisionPairSpec(candidate_geom, other))
    return tuple(pairs)


def probe_free_poses(
    model,
    live_data,
    request: ProbeRequest,
    roots: Sequence[tuple[int, int]],
) -> ProbeResult:
    """Evaluate one :class:`ProbeRequest` on a private ``MjData`` copy.

    ``roots`` is ``(body_id, freejoint qpos address)`` for every override,
    in request order.  The live data is copied with ``mj_copyData``; the
    scratch copy receives the pose writes and a single ``mj_forward``.
    ``pairs=None`` pairs each overridden owner's full body subtree against
    the collision geoms outside every overridden subtree, skips same-owner
    internal pairs, and cross-pairs different overridden owners.  A
    declared ``support_body`` keeps its geoms in the world set but allows a
    near-zero contact for pairs against that exact body.  Explicit pairs
    are evaluated exactly as given.  A support pair's near-zero contact
    exception never excuses overlap.
    """
    _require_mujoco()
    assert mujoco is not None  # _require_mujoco established the capability
    roots = tuple(roots)
    if len(roots) != len(request.overrides):
        raise ProbeError(
            f"probe roots count {len(roots)} does not match request.overrides "
            f"count {len(request.overrides)}"
        )
    clearance = float(request.min_clearance_m)
    if not math.isfinite(clearance) or clearance < 0:
        raise ProbeError("ProbeRequest.min_clearance_m must be a finite non-negative number")

    probe_data = mujoco.MjData(model)
    mujoco.mj_copyData(probe_data, model, live_data)
    for index, (pose, (body_id, qadr)) in enumerate(zip(request.overrides, roots)):
        if not (0 <= int(body_id) < int(model.nbody)):
            raise ProbeError(
                f"probe overrides[{index}] body id {int(body_id)} is outside "
                f"[0, {int(model.nbody)})"
            )
        if not (0 <= int(qadr) and int(qadr) + 7 <= int(model.nq)):
            raise ProbeError(
                f"probe overrides[{index}] qpos address {int(qadr)} cannot hold "
                "a 7-wide freejoint pose"
            )
        quat = np.asarray(pose.quat_wxyz, dtype=np.float64)
        probe_data.qpos[qadr : qadr + 3] = np.asarray(pose.pos_w, dtype=np.float64)
        probe_data.qpos[qadr + 3 : qadr + 7] = quat / float(np.linalg.norm(quat))
    mujoco.mj_forward(model, probe_data)

    pair_list = (
        _automatic_pairs(model, request, roots) if request.pairs is None else tuple(request.pairs)
    )
    distances: list[float] = []
    for index, pair in enumerate(pair_list):
        geom1 = _geom_id(model, pair.candidate_geom, field=f"pairs[{index}].candidate_geom")
        geom2 = _geom_id(model, pair.other_geom, field=f"pairs[{index}].other_geom")
        tolerance = float(pair.penetration_tolerance_m)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ProbeError(
                f"ProbeRequest.pairs[{index}].penetration_tolerance_m must be "
                "finite and non-negative"
            )
        distance = float(
            mujoco.mj_geomDistance(
                model, probe_data, geom1, geom2, max(1.0, clearance + tolerance), None
            )
        )
        distances.append(distance)
        if distance < -tolerance:
            return ProbeResult(False, "penetration", tuple(distances))
        if clearance > 0 and distance < clearance:
            # A support exception is only for near-zero contact, never overlap.
            if not (
                pair.allow_support_contact and distance >= -tolerance and distance <= tolerance
            ):
                return ProbeResult(False, "insufficient_clearance", tuple(distances))
    return ProbeResult(True, None, tuple(distances))
