"""Obstacle placement strategy (isaaclab-free; torch-only).

Samples per-env obstacle placements ``(x, y, z, yaw_rad)``.

``mode="random"`` / ``mode="fixed"`` sample one obstacle per env within
configured ranges (classic behaviour: each env gets a single obstacle type
assigned by proportion).  ``mode="track"`` reproduces a fixed serpentine
track: every obstacle *type* owns one slot in ``track_pos`` (poses relative
to the env origin; the robot travels along +X), jittered per reset by
``random_x / random_y / random_yaw`` (jitter applied per reset).

The *strategy* owns the track-specific behaviour so the generic
:class:`~unigym.backends.isaaclab2.obstacles.ObstacleManager` never branches on
``mode == "track"``.  The manager asks for it through the opaque interface:

* :meth:`validate` — track mode checks one ``track_pos`` slot per type.
* :meth:`initial_pose_for` — track mode returns the type's baked spawn pose
  (``(x, y, yaw_rad)``); random/fixed return ``None``, meaning "place me on
  reset".
* :meth:`needs_reset_placement` — ``False`` for track (the course is baked
  into the spawn ``init_state`` and restored by ``scene.reset``), ``True``
  for random/fixed.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch

_SUPPORTED_MODES = ("random", "fixed", "track")


class PlacementStrategy:
    """Sample obstacle placements ``(x, y, z, yaw_rad)`` per environment."""

    def __init__(self, config: dict | None = None):
        config = dict(config or {})
        self.mode = str(config.get("mode", "random"))
        self.x_range = tuple(config.get("x_range", [1.0, 5.0]))
        self.y_range = tuple(config.get("y_range", [-1.0, 1.0]))
        # yaw ranges configured in degrees for human-friendly yaml; stored rad
        yaw_deg = tuple(config.get("yaw_range", [-10, 10]))
        self.yaw_range = (math.radians(yaw_deg[0]), math.radians(yaw_deg[1]))
        self.height_offset = float(config.get("height_offset", 0.0))
        self.x_range, self.y_range  # let bad shapes raise early

        # ---- track mode: one fixed slot per obstacle type ----
        # track_pos entry = [x, y, yaw_deg] relative to the env origin.
        self.track_pos: list[tuple[float, float, float]] = []
        for p in config.get("track_pos", []):
            x, y, ya = (float(v) for v in p)
            self.track_pos.append((x, y, math.radians(ya)))
        rx = tuple(config.get("random_x", [-0.25, 0.25]))
        ry = tuple(config.get("random_y", [-0.15, 0.15]))
        rya = tuple(config.get("random_yaw", [-5.0, 5.0]))
        self.random_x = (float(rx[0]), float(rx[1]))
        self.random_y = (float(ry[0]), float(ry[1]))
        self.random_yaw = (math.radians(rya[0]), math.radians(rya[1]))

        if self.mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"placement mode must be one of {_SUPPORTED_MODES}, "
                f"got '{self.mode}'")
        if self.mode == "track" and not self.track_pos:
            raise ValueError(
                "placement mode 'track' requires a non-empty 'track_pos' "
                "list of [x, y, yaw_deg] slots")

    @property
    def num_slots(self) -> int:
        """Number of track slots (one per obstacle type)."""
        return len(self.track_pos)

    def validate(self, specs: Sequence[Any]) -> None:
        """Raise if the placement config is inconsistent with ``specs``.

        ``track`` mode requires exactly one ``track_pos`` slot per obstacle
        type (the serpentine course is a fixed per-type assignment).  Random
        and fixed modes have no such constraint — one per-env placement is
        sampled for the env-assigned type.
        """
        if self.mode == "track" and self.num_slots != len(specs):
            raise ValueError(
                f"placement mode 'track' needs exactly one track_pos slot "
                f"per obstacle type ({len(specs)} types), got "
                f"{self.num_slots} slots")

    def needs_reset_placement(self) -> bool:
        """Whether ``on_reset`` must re-place obstacles each episode.

        ``track`` mode is a fixed course baked into the spawn ``init_state``
        (restored by ``scene.reset``), so no reset-time teleports — ``False``.
        Random/fixed sample a fresh placement on reset — ``True``.
        """
        return self.mode != "track"

    def initial_pose_for(self, spec_idx: int) -> tuple[float, float, float] | None:
        """Baked spawn pose ``(x, y, yaw_rad)`` for obstacle ``spec_idx``.

        ``track`` mode returns the type's slot from ``track_pos`` (env-local,
        no jitter) so the manager can anchor the clone at construction.
        Random/fixed return ``None``: the obstacle spawns inert at the env
        origin and is placed by :meth:`sample` on reset.
        """
        if self.mode != "track":
            return None
        if not 0 <= int(spec_idx) < self.num_slots:
            raise IndexError(f"track slot {spec_idx} out of range "
                             f"[0, {self.num_slots})")
        return self.track_pos[int(spec_idx)]

    def sample(self, num_positions: int, device: str = "cpu") -> torch.Tensor:
        """Return a ``(num_positions, 4)`` float tensor ``(x, y, z, yaw_rad)``.

        Defined for ``random``/``fixed`` modes; ``track`` mode raises (use
        :meth:`sample_slot`).
        """
        if self.mode == "track":
            raise NotImplementedError(
                "PlacementStrategy.sample() is not defined for mode='track'; "
                "use sample_slot(slot, n, device)")
        if num_positions <= 0:
            return torch.zeros((0, 4), device=device)
        if self.mode == "random":
            x = torch.empty(num_positions, device=device).uniform_(
                *self.x_range)
            y = torch.empty(num_positions, device=device).uniform_(
                *self.y_range)
            yaw = torch.empty(num_positions, device=device).uniform_(
                *self.yaw_range)
        else:  # fixed: midpoints
            x = torch.full((num_positions,),
                           0.5 * (self.x_range[0] + self.x_range[1]),
                           device=device)
            y = torch.full((num_positions,),
                           0.5 * (self.y_range[0] + self.y_range[1]),
                           device=device)
            yaw = torch.full((num_positions,),
                             0.5 * (self.yaw_range[0] + self.yaw_range[1]),
                             device=device)
        z = torch.full((num_positions,), self.height_offset, device=device)
        return torch.stack([x, y, z, yaw], dim=-1)

    def sample_slot(self, slot: int, num: int, device: str = "cpu") -> torch.Tensor:
        """Return ``(num, 4)`` env-local poses ``(x, y, z, yaw_rad)`` for a track slot.

        Base pose from ``track_pos[slot]`` plus per-reset iid uniform noise
        ``random_x / random_y / random_yaw``; ``z`` = ``height_offset``.
        """
        if self.mode != "track":
            raise NotImplementedError(
                "PlacementStrategy.sample_slot() requires mode='track'")
        if not 0 <= int(slot) < self.num_slots:
            raise IndexError(f"track slot {slot} out of range "
                             f"[0, {self.num_slots})")
        n = int(num)
        base_x, base_y, base_yaw = self.track_pos[slot]
        x = torch.full((n,), base_x, device=device)
        y = torch.full((n,), base_y, device=device)
        yaw = torch.full((n,), base_yaw, device=device)
        if self.random_x[0] != 0.0 or self.random_x[1] != 0.0:
            x += torch.empty(n, device=device).uniform_(*self.random_x)
        if self.random_y[0] != 0.0 or self.random_y[1] != 0.0:
            y += torch.empty(n, device=device).uniform_(*self.random_y)
        if self.random_yaw[0] != 0.0 or self.random_yaw[1] != 0.0:
            yaw += torch.empty(n, device=device).uniform_(*self.random_yaw)
        z = torch.full((n,), self.height_offset, device=device)
        return torch.stack([x, y, z, yaw], dim=-1)
