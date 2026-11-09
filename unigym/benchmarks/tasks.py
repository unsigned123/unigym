"""First benchmark tasks (B1 reference implementations).

Both tasks are pure episode judges: they read the recorded-key arrays of an
``EpisodeData`` and return an ``EpisodeResult``.  Units are documented per
parameter/metric in each docstring (metric keys themselves stay unit-less,
flat snake_case).

* ``stand``             -- upright/height quality of a standing policy.
* ``velocity_tracking`` -- per-frame body-frame velocity command tracking.
"""

from __future__ import annotations

import numpy as np

from .base import (
    EpisodeData,
    EpisodeResult,
    Task,
    register,
)


@register
class StandTask(Task):
    """Standing-quality judge.

    Metrics (per episode, frame = policy step):
        alive_fraction: fraction of frames where the projected gravity is
            within ``alive_tolerance`` of upright (dimensionless, 0..1).
        base_z_mean: mean base height over frames (m).
        base_z_var: population variance of base height (m^2).
        tilt_exceed_fraction: fraction of frames whose tilt angle (between
            body +z and world +z, degrees) exceeds ``tilt_max_deg``
            (dimensionless, 0..1).

    Success: ``alive_fraction >= alive_threshold`` AND ``height_range[0] <=
    base_z_mean <= height_range[1]`` (bounds inclusive).

    Params (yaml ``benchmark.tasks: [{name: stand, **params}]``):
        articulation: REQUIRED name of the articulation whose
            ``gravity_b`` / ``base_pos`` are judged (e.g. g1).  There is
            no primary binding and no unique-articulation inference.
        height_range: [lo, hi] acceptable mean base height, m
            (default [0.5, 1.0]).
        tilt_max_deg: tilt angle above which a frame counts as tilted,
            degrees (default 30.0).
        alive_tolerance: upright band around gravity_b z == -1
            (|g_z + 1| <= tol), dimensionless (default 0.2).
        alive_threshold: success threshold on alive_fraction, dimensionless
            (default 0.9).
    """

    name = "stand"
    requires_articulation = True

    def __init__(self, params: dict | None = None, meta: dict | None = None):
        super().__init__(params, meta)
        lo, hi = self.p.get("height_range", [0.5, 1.0])
        self._z_lo, self._z_hi = float(lo), float(hi)
        self._tilt_max_deg = float(self.p.get("tilt_max_deg", 30.0))
        self._alive_tol = float(self.p.get("alive_tolerance", 0.2))
        self._alive_thr = float(self.p.get("alive_threshold", 0.9))

    def score(self, ep: EpisodeData) -> EpisodeResult:
        g = np.asarray(ep.data[self.resolve_key(ep.data, "gravity_b")], dtype=float)  # (T, 3)
        z = np.asarray(ep.data[self.resolve_key(ep.data, "base_pos")], dtype=float)[:, 2]  # (T,)
        upright = np.abs(g[:, 2] + 1.0) <= self._alive_tol
        alive_fraction = float(upright.mean())
        # tilt: angle between body +z and world +z; cos(tilt) == -g_z
        tilt_deg = np.degrees(np.arccos(np.clip(-g[:, 2], -1.0, 1.0)))
        tilt_exceed_fraction = float((tilt_deg > self._tilt_max_deg).mean())
        base_z_mean = float(z.mean())
        base_z_var = float(z.var())
        success = alive_fraction >= self._alive_thr and self._z_lo <= base_z_mean <= self._z_hi
        return EpisodeResult(
            metrics={
                "alive_fraction": alive_fraction,
                "base_z_mean": base_z_mean,
                "base_z_var": base_z_var,
                "tilt_exceed_fraction": tilt_exceed_fraction,
            },
            success=bool(success),
        )


@register
class VelocityTrackingTask(Task):
    """Body-frame linear velocity command tracking judge.

    Per frame e_t = ||v_actual[:, :2] - cmd[:, :2]||_2 (m/s); a frame is a
    tracking success when ``e_t <= error_threshold``.

    Metrics (per episode):
        tracking_error_mean: mean per-frame tracking error (m/s).
        tracking_error_max: max per-frame tracking error (m/s).
        success_fraction: fraction of frames with error <=
            ``error_threshold`` (dimensionless, 0..1).

    Success: ``success_fraction >= success_fraction`` (param, dimensionless).

    Params:
        articulation: REQUIRED name of the articulation whose
            ``base_lin_vel`` / ``command`` are judged (e.g. g1).  There is
            no primary binding and no unique-articulation inference.
        error_threshold: per-frame error below which the frame tracks,
            m/s (default 0.5).
        success_fraction: episode success threshold on success_fraction,
            dimensionless (default 0.8).

    Requires a command channel: the run must configure a ``command:``
    sampler or a ``generates_command`` adapter (meta ``has_command``);
    construction raises ValueError otherwise.  Note the command's first two
    channels are interpreted as (vx, vy) in m/s.
    """

    name = "velocity_tracking"
    requires_articulation = True

    def __init__(self, params: dict | None = None, meta: dict | None = None):
        super().__init__(params, meta)
        if not self.meta.get("has_command", False):
            raise ValueError(
                "benchmark task 'velocity_tracking' requires a command "
                "channel: configure a yaml `command:` sampler or use an "
                "adapter with generates_command=True (none present)"
            )
        self._err_thr = float(self.p.get("error_threshold", 0.5))
        self._succ_frac = float(self.p.get("success_fraction", 0.8))

    def score(self, ep: EpisodeData) -> EpisodeResult:
        v = np.asarray(ep.data[self.resolve_key(ep.data, "base_lin_vel")], dtype=float)[
            :, :2
        ]  # (T, 2)
        cmd = np.asarray(ep.data[self.resolve_key(ep.data, "command")], dtype=float)[
            :, :2
        ]  # (T, 2)
        err = np.linalg.norm(v - cmd, axis=1)  # (T,)
        success_fraction = float((err <= self._err_thr).mean())
        return EpisodeResult(
            metrics={
                "tracking_error_mean": float(err.mean()),
                "tracking_error_max": float(err.max()),
                "success_fraction": success_fraction,
            },
            success=bool(success_fraction >= self._succ_frac),
        )
