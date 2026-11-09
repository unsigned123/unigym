"""Whitelist term templates for FormulaTask (B1.5).

Every template is a pure numpy function of an ``EpisodeData``:

    term_<name>(ep: EpisodeData, spec: dict) -> float

where ``spec`` is the yaml metric entry minus its ``id``/``term`` keys.
There is deliberately NO free-form expression language: all parameters are
structured data, ``post``/``reduction``/``op`` are whitelist enums, and a
``pred_spec`` free-expression field is *rejected* (v1 guardrail -- the DSL
is a fast path, not an eval hole; escape hatch is writing a Python Task,
docs/contracts/benchmark_contract.md §FormulaTask).

Frame-scalar derivation shared by ``frame_fraction``:

* ``dims`` selects columns (``axis`` is the single-column shorthand);
  omitted dims on a multi-column key selects all columns;
* ``key2`` turns the per-frame values into a per-frame *residual*
  ``x[t, cols] - y[t, cols]`` (frame-aligned; enables norm-of-difference
  predicates such as velocity tracking);
* ``post`` reduces the (possibly multi-column) selection to one scalar per
  frame: ``norm2`` = L2 norm; ``abs``/``x2`` require a single selected
  column.  Missing ``post`` requires exactly one selected column.

Numeric conventions: population statistics (``var`` ddof=0), inclusive
bounds everywhere, ``p90`` = ``np.percentile(.., 90)`` linear
interpolation.  An empty episode (T=0) raises ValueError in every template
(silent NaN would poison aggregate statistics).

Adding a template (checklist): implement ``term_<name>`` with a full
docstring (parameter table + units + return semantics), register it in
``TERM_REGISTRY``, add required params to ``TERM_REQUIRED_PARAMS`` and any
enum/param checks to ``validate_term_spec``, then unit-test against
hand-computed synthetic episodes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .base import EpisodeData, resolve_episode_key

# 14 templates -- the 11 pre-existing + contact_any (declarative
# contact criteria) + hold_streak (duration/streak gates, e.g. an
# external success-streak episode rule translated 1:1) +
# keypoint_in_region_fraction (tiered spawn-region criteria).

# default location of reference motion npz files (SONIC, reserved)
REFERENCES_DIR = Path(__file__).resolve().parents[2] / "references"

# one-sided comparison ops (whitelist; v1 constant-value operand only)
OPS = ("gt", "lt", "ge", "le")
_OP_FNS = {
    "gt": np.greater,
    "lt": np.less,
    "ge": np.greater_equal,
    "le": np.less_equal,
}

# per-frame scalarizing transforms (whitelist; no eval)
POSTS = ("norm2", "abs", "x2")

# reductions over per-frame scalars (whitelist)
REDUCTIONS = ("mean", "max", "p90")

# reductions over the (multi-column) last-minus-first vector (delta only)
DELTA_REDUCTIONS = ("mean", "max", "norm2")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _frames(ep: EpisodeData, key, articulation: str | None = None) -> np.ndarray:
    """Episode key -> float ndarray (T, .); missing key -> ValueError.

    Robot-state keys resolve under *articulation* (the task's explicit
    declaration); a missing declaration fails fast in the resolver.
    """
    if not isinstance(key, str):
        raise ValueError(f"key must be a recorded-key string, got {key!r}")
    key = resolve_episode_key(ep.data, key, articulation)
    if key not in ep.data:
        raise ValueError(f"episode data has no key '{key}' (available: {sorted(ep.data)})")
    return np.asarray(ep.data[key], dtype=float)


def _check_frames(x: np.ndarray, what: str) -> np.ndarray:
    """Empty episode (T=0) is a definition error, not a 0/NaN metric."""
    if x.ndim < 1 or x.shape[0] == 0:
        raise ValueError(f"{what}: empty episode (T=0) cannot be scored")
    return x


def _select(x: np.ndarray, spec: dict, what: str) -> np.ndarray:
    """Column selection: spec['dims'] (list) or spec['axis'] (int)."""
    dims, axis = spec.get("dims"), spec.get("axis")
    if dims is not None and axis is not None:
        raise ValueError(f"{what}: 'dims' and 'axis' are mutually exclusive")
    if x.ndim == 1:
        # a (T,) series is a single-column (T, 1) table (e.g. bool
        # scalars like time_in_episode); column 0.
        x = x[:, None]
        if axis is not None and int(axis) != 0:
            raise ValueError(f"{what}: single-column series has only axis 0")
    if dims is not None:
        return x[:, [int(d) for d in dims]]
    if axis is not None:
        return x[:, [int(axis)]]
    return x


def _frame_scalars(
    ep: EpisodeData, spec: dict, what: str, articulation: str | None = None
) -> np.ndarray:
    """Per-frame scalar series from spec: key (+key2 residual) -> cols ->
    post.  Used by frame_fraction; see module docstring for semantics."""
    x = _check_frames(_frames(ep, spec["key"], articulation), what)
    r = _select(x, spec, what)
    key2 = spec.get("key2")
    if key2 is not None:
        y = _check_frames(_frames(ep, key2, articulation), what)
        if y.shape[0] != x.shape[0]:
            raise ValueError(
                f"{what}: key2 '{key2}' has T={y.shape[0]} frames, "
                f"key '{spec['key']}' has T={x.shape[0]} (must align)"
            )
        r = r - _select(y, spec, what)
    post = spec.get("post")
    if post == "norm2":
        return np.linalg.norm(r, axis=1)
    if post is None or post in ("abs", "x2"):
        if r.shape[1] != 1:
            raise ValueError(
                f"{what}: post={post!r} needs exactly one selected column "
                f"(dims/axis of length 1); got {r.shape[1]} columns "
                f"(multi-column only supports post: norm2)"
            )
        v = r[:, 0]
        if post == "abs":
            return np.abs(v)
        if post == "x2":
            return np.square(v)
        return v
    raise ValueError(f"{what}: unknown post {post!r} (whitelist: {', '.join(POSTS)})")


def _scalar_reduction(v: np.ndarray, spec: dict, what: str) -> float:
    """spec['reduction'] over a per-frame series (default mean)."""
    red = spec.get("reduction", "mean")
    if red == "mean":
        return float(v.mean())
    if red == "max":
        return float(v.max())
    if red == "p90":
        return float(np.percentile(v, 90))
    raise ValueError(f"{what}: unknown reduction {red!r} (whitelist: {', '.join(REDUCTIONS)})")


# --------------------------------------------------------------------------- #
# templates
# --------------------------------------------------------------------------- #
def term_frame_fraction(ep: EpisodeData, spec: dict, *, articulation: str | None = None) -> float:
    """Fraction of frames satisfying a structured predicate.

    spec:
        key: recorded key of the per-frame array (T, .)
        axis: int column (shorthand for dims: [axis])          (optional)
        dims: list of int columns                              (optional)
        key2: residual reference key; per-frame values become
              x[t, cols] - y[t, cols] (frame-aligned)          (optional)
        post: norm2 | abs | x2  (scalarizing transform; norm2 is the only
              multi-column option)                             (optional)
        target: band center; predicate |v_t - target| <= tol   (form 1)
        tol: band half-width, default 0.0 (inclusive)          (optional)
        op: gt | lt | ge | le with value: constant threshold   (form 2)
        pred_spec: NOT SUPPORTED in v1 (rejected, see guardrail)

    Returns: float in [0, 1] -- fraction of frames where the predicate
    holds.  Forms 1/2 are mutually exclusive; exactly one is required.

    Units: input-dependent (e.g. dimensionless for gravity bands, m/s for
    norm-of-velocity residuals); the return is a dimensionless fraction.
    """
    v = _frame_scalars(ep, spec, "frame_fraction", articulation)
    if "pred_spec" in spec:
        raise NotImplementedError(
            "frame_fraction: pred_spec free-form expressions are rejected "
            "in v1 (whitelist-only DSL); use {target, tol} or {op, value}"
        )
    target = spec.get("target")
    if target is not None:
        if "op" in spec:
            raise ValueError(
                "frame_fraction: predicate forms are mutually exclusive "
                "({target, tol} vs {op, value})"
            )
        tol = float(spec.get("tol", 0.0))
        return float((np.abs(v - float(target)) <= tol).mean())
    if "op" in spec:
        op = spec["op"]
        if op not in _OP_FNS:
            raise ValueError(f"frame_fraction: unknown op {op!r} (whitelist: {', '.join(OPS)})")
        if "value" not in spec:
            raise ValueError("frame_fraction: op form requires a constant 'value'")
        return float(_OP_FNS[op](v, float(spec["value"])).mean())
    raise ValueError(
        "frame_fraction: missing predicate -- provide {target(, tol)} " "or {op, value}"
    )


def term_frame_mean(ep: EpisodeData, spec: dict, *, articulation: str | None = None) -> float:
    """Mean over frames of the (transformed) selected values.

    spec:
        key: recorded key (T, .)
        dims: list of int columns (default: all)               (optional)
        post: norm2 (per-frame L2 norm) | abs | x2             (optional)

    Returns: float -- mean over frames; with post=norm2 the mean of
    per-frame L2 norms; with abs/x2 the elementwise mean of |x| / x^2.
    Units: those of ``key`` (norm2: same, abs: same, x2: squared).
    """
    x = _check_frames(_frames(ep, spec["key"], articulation), "frame_mean")
    r = _select(x, spec, "frame_mean")
    post = spec.get("post")
    if post is None:
        return float(r.mean())
    if post == "norm2":
        return float(np.linalg.norm(r, axis=1).mean())
    if post == "abs":
        return float(np.abs(r).mean())
    if post == "x2":
        return float(np.square(r).mean())
    raise ValueError(f"frame_mean: unknown post {post!r} (whitelist: {', '.join(POSTS)})")


def term_frame_var(ep: EpisodeData, spec: dict, *, articulation: str | None = None) -> float:
    """Population variance (ddof=0) over frames of the transformed values.

    spec: identical to ``frame_mean`` (key, dims?, post?).

    Returns: float -- variance over frames (norm2: variance of per-frame
    L2 norms; abs/x2: variance of the elementwise transform).
    Units: squared units of the transformed quantity (m -> m^2).
    """
    x = _check_frames(_frames(ep, spec["key"], articulation), "frame_var")
    r = _select(x, spec, "frame_var")
    post = spec.get("post")
    if post is None:
        return float(r.var())
    if post == "norm2":
        return float(np.linalg.norm(r, axis=1).var())
    if post == "abs":
        return float(np.abs(r).var())
    if post == "x2":
        return float(np.square(r).var())
    raise ValueError(f"frame_var: unknown post {post!r} (whitelist: {', '.join(POSTS)})")


def _frame_series(
    ep: EpisodeData, spec: dict, what: str, articulation: str | None = None
) -> np.ndarray:
    """Per-frame series for final/min/max: selected columns, L2-reduced
    when more than one column is selected (raw scalar for one column)."""
    x = _check_frames(_frames(ep, spec["key"], articulation), what)
    r = _select(x, spec, what)
    if r.shape[1] == 1:
        return r[:, 0]
    return np.linalg.norm(r, axis=1)


def term_final(ep: EpisodeData, spec: dict, *, articulation: str | None = None) -> float:
    """Value of the last frame.

    spec:
        key: recorded key (T, .)
        dims: list of int columns; multi-column selections are L2-reduced
            per frame first (single column stays raw)         (optional)

    Returns: float -- the final-frame value (units of ``key``; L2 norm
    keeps those units).
    """
    return float(_frame_series(ep, spec, "final", articulation)[-1])


def term_min(ep: EpisodeData, spec: dict, *, articulation: str | None = None) -> float:
    """Minimum over frames (same spec/derivation as ``final``).

    Returns: float -- min over frames of the per-frame scalar series.
    """
    return float(_frame_series(ep, spec, "min", articulation).min())


def term_max(ep: EpisodeData, spec: dict, *, articulation: str | None = None) -> float:
    """Maximum over frames (same spec/derivation as ``final``).

    Returns: float -- max over frames of the per-frame scalar series.
    """
    return float(_frame_series(ep, spec, "max", articulation).max())


def term_tracking_error(ep: EpisodeData, spec: dict, *, articulation: str | None = None) -> float:
    """Per-frame ||state[cols] - cmd[cols]||_2 reduced over frames.

    spec:
        state: recorded key of the actual signal (T, .), e.g. base_lin_vel
        cmd: recorded key of the command signal (T, .), e.g. command
        dims: list of int columns (default [0, 1], i.e. (vx, vy) m/s)
        state_dims: per-signal column override for ``state``   (optional)
        cmd_dims: per-signal column override for ``cmd``       (optional)
            (lets e.g. base_pos z (col 2) track a height command
            living in a different column of the action_raw frame; when
            either override is absent ``dims`` applies to both signals)
        reduction: mean | max | p90 (default mean)

    Returns: float -- the reduced per-frame tracking error (units of the
    signal, m/s for velocities).  p90 = np.percentile(err, 90).
    """
    s = _check_frames(_frames(ep, spec["state"], articulation), "tracking_error")
    c = _check_frames(_frames(ep, spec["cmd"], articulation), "tracking_error")
    if s.shape[0] != c.shape[0]:
        raise ValueError(
            f"tracking_error: state T={s.shape[0]} != cmd T={c.shape[0]} "
            "(frame-aligned keys required)"
        )
    dims = spec.get("dims", (0, 1))
    cols_s = [int(d) for d in spec.get("state_dims", dims)]
    cols_c = [int(d) for d in spec.get("cmd_dims", dims)]
    err = np.linalg.norm(s[:, cols_s] - c[:, cols_c], axis=1)
    return _scalar_reduction(err, spec, "tracking_error")


def term_range_check_fraction(
    ep: EpisodeData, spec: dict, *, articulation: str | None = None
) -> float:
    """Fraction of frames whose selected values all lie in [lo, hi].

    spec:
        key: recorded key (T, .)
        dims: list of int columns (default: all)               (optional)
        lo: lower bound, inclusive (default -inf)              (optional)
        hi: upper bound, inclusive (default +inf)              (optional)

    Returns: float in [0, 1] -- a frame counts when EVERY selected value
    satisfies lo <= x <= hi.  Units: those of ``key`` for lo/hi.
    """
    x = _check_frames(_frames(ep, spec["key"], articulation), "range_check_fraction")
    r = _select(x, spec, "range_check_fraction")
    lo = float(spec.get("lo", -np.inf))
    hi = float(spec.get("hi", np.inf))
    if lo > hi:
        raise ValueError(f"range_check_fraction: lo ({lo}) > hi ({hi})")
    mask = np.all((r >= lo) & (r <= hi), axis=1)
    return float(mask.mean())


def term_delta(ep: EpisodeData, spec: dict, *, articulation: str | None = None) -> float:
    """Last-frame value minus first-frame value of the selected columns.

    spec:
        key: recorded key of the per-frame array (T, .)
        dims: list of int columns (default: all)               (optional)
        reduction: mean | max | norm2 -- reduces the (last-first) vector
            when MORE than one column is selected; required for
            multi-column selections, forbidden with a single one
                                                               (optional)

    Returns: float -- x[-1, cols] - x[0, cols] reduced to one scalar
    (single column: the raw difference; ``mean``/``max``: elementwise;
    ``norm2``: L2 norm of the difference vector).  Units: those of ``key``
    (norm2 keeps them).  A T=1 episode yields 0.0 (first and last frame
    coincide) -- no displacement is measurable, not an error.

    Typical uses: lift Δz, push/locomotion displacement along an axis,
    posture drift.
    """
    x = _check_frames(_frames(ep, spec["key"], articulation), "delta")
    r = _select(x, spec, "delta")
    d = r[-1] - r[0]
    red = spec.get("reduction")
    if red is None:
        if d.shape[0] != 1:
            raise ValueError(
                f"delta: multi-column selection ({d.shape[0]} columns) "
                f"requires a 'reduction' (whitelist: "
                f"{', '.join(DELTA_REDUCTIONS)})"
            )
        return float(d[0])
    if red == "mean":
        return float(d.mean())
    if red == "max":
        return float(d.max())
    if red == "norm2":
        return float(np.linalg.norm(d))
    raise ValueError(
        f"delta: unknown reduction {red!r} (whitelist: " f"{', '.join(DELTA_REDUCTIONS)})"
    )


def term_point_in_box_fraction(
    ep: EpisodeData, spec: dict, *, articulation: str | None = None
) -> float:
    """Fraction of frames whose 3D point lies inside an axis-aligned box.

    spec:
        key: recorded key of the point series (T, 3) -- xyz order
            (e.g. base_pos); a (T, .) array whose second axis is not 3 is
            a definition error
        center: [x, y, z] box center
        half_extents: [hx, hy, hz] half lengths (all >= 0)

    Returns: float in [0, 1] -- fraction of frames with ALL of
    center - half_extents <= point <= center + half_extents (inclusive
    bounds, frame-wise over each axis).  Units: those of ``key`` for
    center/half_extents (m for positions).
    """
    x = _check_frames(_frames(ep, spec["key"], articulation), "point_in_box_fraction")
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(
            f"point_in_box_fraction: key must be a (T, 3) xyz point " f"series, got shape {x.shape}"
        )
    center, half = spec.get("center"), spec.get("half_extents")
    try:
        c = np.asarray(center, dtype=float)
        h = np.asarray(half, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"point_in_box_fraction: center/half_extents must be [x, y, z] "
            f"number lists, got {center!r} / {half!r}"
        ) from exc
    if c.shape != (3,) or h.shape != (3,):
        raise ValueError(
            f"point_in_box_fraction: center/half_extents must have exactly "
            f"3 numbers each, got shapes {c.shape} / {h.shape}"
        )
    if (h < 0).any():
        raise ValueError(
            f"point_in_box_fraction: half_extents must be non-negative, " f"got {half!r}"
        )
    inside = np.all((x >= c - h) & (x <= c + h), axis=1)
    return float(inside.mean())


def term_reference_mpjpe(ep: EpisodeData, spec: dict, *, articulation: str | None = None) -> float:
    """Mean per-joint error of ``key`` against a reference motion npz.

    spec:
        key: recorded key of the per-frame signal (T, J) -- typically
            joint_pos (rad); (T, J, D) positional arrays also supported
        reference: npz path -- absolute, or relative to references/
            (e.g. g1_stand.npz; SONIC, reserved)

    Alignment (v1, frame-index based -- no time resampling): the episode
    must be AT LEAST as long as the reference; over-long episodes are
    clamped to the first R reference frames; shorter episodes raise
    ValueError.  The npz ``fps`` field is informational only in v1.

    Returns: float -- MPJPE = mean over frames and joints of the per-joint
    L2 error (rad for joint_pos; per-joint L2 over D for (T, J, D)).
    """
    key = spec["key"]
    x = _check_frames(_frames(ep, key, articulation), "reference_mpjpe")
    ref_path = Path(spec["reference"])
    if not ref_path.is_absolute():
        ref_path = REFERENCES_DIR / ref_path
    if not ref_path.is_file():
        raise ValueError(f"reference_mpjpe: reference file not found: {ref_path}")
    with np.load(ref_path) as npz:
        if key not in npz.files:
            raise ValueError(
                f"reference_mpjpe: reference npz {ref_path.name} has no "
                f"key '{key}' (available: {sorted(npz.files)})"
            )
        ref = np.asarray(npz[key], dtype=float)  # (R, J[, D])
    n_ref = ref.shape[0]
    if x.shape[0] < n_ref:
        raise ValueError(
            f"reference_mpjpe: episode shorter than reference "
            f"(T={x.shape[0]} < R={n_ref}); only over-long episodes are "
            "clamped, shorter ones are a definition error"
        )
    diff = x[:n_ref] - ref  # clamp over-long episodes to R frames
    per_joint = np.linalg.norm(diff, axis=-1) if diff.ndim == 3 else np.abs(diff)
    return float(per_joint.mean())


def term_contact_any(ep: EpisodeData, spec: dict, *, articulation: str | None = None) -> float:
    """Fraction of frames where a DECLARED contact group holds.

    The generic hand/contact criterion primitive : consumes a
    recorded ``contact/<name>`` key (the group must use ``format:
    contact``, i.e. boolean vs its threshold) and reduces the per-frame
    boolean table over all non-time axes (sensor bodies -- and filter
    bodies when the group keeps its F axis) with ``any`` or ``all``.

    spec:
        contact: recorded key of the contact group -- ``contact/<name>``
            (a bare group name without the ``contact/`` prefix is
            accepted and prefixed)
        reduce_bodies: any | all -- reduction over the body (B[, F])
            axes; ``any`` = at least one body in contact (held-style),
            ``all`` = every body in contact (full-closure style)

    Returns: float in [0, 1] -- fraction of frames where the reduced
    predicate holds.  Dimensionless.

    Typical uses: held / release / handover criteria over declared
    pairwise groups (e.g. hand_crate_left); the per-B any/all replaces
    the retired auto held-key family.
    """
    return float(_contact_series(ep, spec, "contact_any", articulation).mean())


# --------------------------------------------------------------------------- #
# keypoint-in-region (tiered spawn-region criteria) -- the geometry lives
# in benchmarks/regions.py (official semantics + citations in its module
# docstring); terms.py keeps the whitelist registration + delegation.
# --------------------------------------------------------------------------- #
from .regions import keypoint_region_series as _keypoint_region_series
from .regions import validate_spec as _validate_keypoint_region_spec
from .holding import contact_series as _contact_series
from .holding import hold_predicate_series as _hold_predicate_series
from .holding import validate_hold_specs as _validate_hold_specs


def term_keypoint_in_region_fraction(
    ep: EpisodeData, spec: dict, *, articulation: str | None = None
) -> float:
    """Fraction of frames where enough eroded bbox keypoints of a moving
    body lie inside a spawn region attached to another body.

    The tiered-container criteria primitive: the official generated
    success (tabletop_24dc.py:626-641) routes tiered_basket/tiered_shelf
    targets through ``OU.check_obj_in_receptacle(...,
    spawn_regions=[OU.get_highest_spawn_region(...)])`` --
    obj_in_region_with_keypoints (object_utils.py:229-283) instead of the
    contact + horizontal-distance band.  benchmarks/regions.py carries
    the full semantic transcription with citations (15 eroded keypoints,
    highest-z region selection, box/cylinder/sphere inclusive tests,
    min_num_points=2).

    spec (all structured; no eval):
        points: {pos, quat} -- recorded keys of the moving body's world
            pose (pos (T, 3); quat (T, 4) wxyz)
        bbox: {bottom [3], top [3], radius [2]} -- the body's LOCAL bbox
            model (scaled bottom_site/top_site pos and the
            horizontal_radius_site xy reading)
        region: {selector: highest_z, body: {pos, quat}, geoms: [...]}
            -- the spawn geoms LOCAL to the region body; each geom is
            {type: box|cylinder|sphere, pos [3], quat [4 wxyz]?
            (default identity), size [3]} (MuJoCo geom_size convention:
            box half extents; cylinder (radius, half_height, ignored);
            sphere radius only size[0] is used)
        min_points: int >= 1 (official: 2)
        erode: number in (0, 1] (official: 0.8)

    Returns: float in [0, 1] -- fraction of frames where the predicate
    holds.  Dimensionless.
    """
    return float(
        _keypoint_region_series(ep, spec, "keypoint_in_region_fraction", articulation).mean()
    )


def term_hold_streak(ep: EpisodeData, spec: dict, *, articulation: str | None = None) -> float:
    """Longest run of CONSECUTIVE frames where every declared per-frame
    predicate holds simultaneously (frame-level AND, then max run length).

    The duration/hold primitive: episodes whose gate is "the success
    condition must persist for N steps" (e.g. an external wrapper's
    success-streak episode rule) declare the per-frame predicates here and
    threshold the returned streak in the predicate tree.

    spec:
        specs: non-empty list of per-frame predicate declarations; each
            entry is EITHER a frame predicate (``key`` + optional
            axis/dims/key2/post + ``{target, tol}`` or ``{op, value}`` --
            the frame_fraction derivation and predicate forms verbatim)
            OR a contact predicate (``contact`` + ``reduce_bodies`` --
            the contact_any derivation) OR a region predicate (a nested
            ``keypoint_in_region:`` dict -- the
            keypoint_in_region_fraction derivation), optionally with
            ``negate: true`` (frame-level NOT).  All entries must be
            frame-aligned.

    Returns: float in [0, T] -- the maximum number of consecutive frames
    where the conjunction holds (0.0 when it never holds; an empty
    episode raises, as in every template).  Units: frames (dimensionless
    step count).
    """
    specs = spec.get("specs")
    if not isinstance(specs, (list, tuple)) or not specs:
        raise ValueError(
            "hold_streak: 'specs' must be a non-empty list of per-frame "
            f"predicate declarations, got {specs!r}"
        )
    series: np.ndarray | None = None
    for i, sub in enumerate(specs):
        s = _hold_predicate_series(ep, sub, f"hold_streak.specs[{i}]", articulation)
        series = s if series is None else (series & s)
    x = np.asarray(series, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], x, [0])))
    starts = np.flatnonzero(edges == 1)
    if starts.size == 0:
        return 0.0
    ends = np.flatnonzero(edges == -1)
    return float((ends - starts).max())


TERM_REGISTRY: dict[str, object] = {
    "frame_fraction": term_frame_fraction,
    "frame_mean": term_frame_mean,
    "frame_var": term_frame_var,
    "final": term_final,
    "min": term_min,
    "max": term_max,
    "tracking_error": term_tracking_error,
    "range_check_fraction": term_range_check_fraction,
    "delta": term_delta,
    "point_in_box_fraction": term_point_in_box_fraction,
    "reference_mpjpe": term_reference_mpjpe,
    "contact_any": term_contact_any,
    "hold_streak": term_hold_streak,
    "keypoint_in_region_fraction": term_keypoint_in_region_fraction,
}


# --------------------------------------------------------------------------- #
# per-frame retention (benchmark_contract §results -- metrics.npz)
# --------------------------------------------------------------------------- #
def _frames_fraction(ep, spec, articulation=None):
    """frame_fraction: the per-frame predicate series (reduction = mean)."""
    v = _frame_scalars(ep, spec, "frame_fraction", articulation)
    target = spec.get("target")
    if target is not None:
        tol = float(spec.get("tol", 0.0))
        return (np.abs(v - float(target)) <= tol).astype(np.float64)
    return _OP_FNS[spec["op"]](v, float(spec["value"])).astype(np.float64)


def _post_series(ep, spec, what, articulation=None):
    x = _check_frames(_frames(ep, spec["key"], articulation), what)
    r = _select(x, spec, what)
    post = spec.get("post")
    if post == "norm2":
        return np.linalg.norm(r, axis=1)
    if post in ("abs", "x2") and r.shape[1] == 1:
        return np.abs(r[:, 0]) if post == "abs" else np.square(r[:, 0])
    return r[:, 0] if r.shape[1] == 1 else r


def _frames_tracking_error(ep, spec, articulation=None):
    s = _check_frames(_frames(ep, spec["state"], articulation), "tracking_error")
    c = _check_frames(_frames(ep, spec["cmd"], articulation), "tracking_error")
    dims = spec.get("dims", (0, 1))
    return np.linalg.norm(
        s[:, [int(d) for d in spec.get("state_dims", dims)]]
        - c[:, [int(d) for d in spec.get("cmd_dims", dims)]],
        axis=1,
    )


def _frames_reference_mpjpe(ep, spec, articulation=None):
    x = _check_frames(_frames(ep, spec["key"], articulation), "reference_mpjpe")
    ref_path = Path(spec["reference"])
    if not ref_path.is_absolute():
        ref_path = REFERENCES_DIR / ref_path
    with np.load(ref_path) as npz:
        ref = np.asarray(npz[spec["key"]], dtype=float)
    diff = x[: ref.shape[0]] - ref
    per_joint = np.linalg.norm(diff, axis=-1) if diff.ndim == 3 else np.abs(diff)
    return per_joint.mean(axis=tuple(range(1, per_joint.ndim))) if per_joint.ndim > 1 else per_joint


def _frames_contact_any(ep, spec, articulation=None):
    return _contact_series(ep, spec, "contact_any", articulation).astype(np.float64)


def _frames_hold_streak(ep, spec, articulation=None):
    """hold_streak: the per-frame conjunction series (T,) -- exactly the
    per-step success signal an external success-streak wrapper would see
    (the sequence its streak counter consumed)."""
    series = None
    for i, sub in enumerate(spec.get("specs") or []):
        s = _hold_predicate_series(ep, sub, f"hold_streak.specs[{i}]", articulation)
        series = s if series is None else (series & s)
    if series is None:  # pragma: no cover -- schema
        raise ValueError("hold_streak: empty 'specs'")  # guards this
    return series.astype(np.float64)


def _frames_point_in_box(ep, spec, articulation=None):
    x = _check_frames(_frames(ep, spec["key"], articulation), "point_in_box_fraction")
    c = np.asarray(spec["center"], dtype=float)
    h = np.asarray(spec["half_extents"], dtype=float)
    return np.all((x >= c - h) & (x <= c + h), axis=1).astype(np.float64)


def _frames_keypoint_region(ep, spec, articulation=None):
    return _keypoint_region_series(ep, spec, "keypoint_in_region_fraction", articulation).astype(
        np.float64
    )


def _frames_range_check(ep, spec, articulation=None):
    x = _check_frames(_frames(ep, spec["key"], articulation), "range_check_fraction")
    sel = _select(x, spec, "r")
    return np.all(
        (sel >= float(spec.get("lo", -np.inf))) & (sel <= float(spec.get("hi", np.inf))), axis=1
    ).astype(np.float64)


# term -> per-frame series (the sequence its reduction consumed); None
# entries = no natural per-frame face (delta is an endpoint pair)
TERM_FRAMES: dict[str, Callable[..., object]] = {
    "frame_fraction": _frames_fraction,
    "frame_mean": lambda ep, spec, a=None: _post_series(ep, spec, "frame_mean", a),
    "frame_var": lambda ep, spec, a=None: _post_series(ep, spec, "frame_var", a),
    "final": lambda ep, spec, a=None: _frame_series(ep, spec, "final", a),
    "min": lambda ep, spec, a=None: _frame_series(ep, spec, "min", a),
    "max": lambda ep, spec, a=None: _frame_series(ep, spec, "max", a),
    "tracking_error": _frames_tracking_error,
    "range_check_fraction": _frames_range_check,
    "delta": lambda ep, spec, a=None: None,
    "point_in_box_fraction": _frames_point_in_box,
    "reference_mpjpe": _frames_reference_mpjpe,
    "contact_any": _frames_contact_any,
    "hold_streak": _frames_hold_streak,
    "keypoint_in_region_fraction": _frames_keypoint_region,
}


def metric_frames(term: str, ep: EpisodeData, spec: dict, articulation: str | None = None):
    """The per-frame series behind a term's scalar (float64 (T,)), or None
    when the template has no natural per-frame face (``delta``).  Purely
    the pre-reduction sequence -- metric_frames(...) reduced with the
    term's own reduction reproduces the scalar (verified in tests)."""
    return TERM_FRAMES[term](ep, spec, articulation)


# params each term requires in its spec (beyond id/term)
TERM_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "frame_fraction": ("key",),
    "frame_mean": ("key",),
    "frame_var": ("key",),
    "final": ("key",),
    "min": ("key",),
    "max": ("key",),
    "tracking_error": ("state", "cmd"),
    "range_check_fraction": ("key",),
    "delta": ("key",),
    "point_in_box_fraction": ("key", "center", "half_extents"),
    "reference_mpjpe": ("key", "reference"),
    "contact_any": ("contact", "reduce_bodies"),
    "hold_streak": ("specs",),
    "keypoint_in_region_fraction": ("points", "bbox", "region", "min_points", "erode"),
}


def get_term(name) -> object:
    """Whitelist lookup; unknown names raise ValueError listing options."""
    if name not in TERM_REGISTRY:
        raise ValueError(f"unknown term '{name}' (whitelist: {sorted(TERM_REGISTRY)})")
    return TERM_REGISTRY[name]


def _check_dims(spec: dict, errors: list[str], where: str, key: str = "dims") -> None:
    dims = spec.get(key)
    if dims is None:
        return
    if (
        not isinstance(dims, (list, tuple))
        or not dims
        or not all(isinstance(d, int) and not isinstance(d, bool) for d in dims)
    ):
        errors.append(f"{where}.{key}: must be a non-empty list of int columns, " f"got {dims!r}")


def validate_term_spec(term, spec: dict) -> list[str]:
    """Term-level schema checks (used by FormulaTask at construction).

    Returns a list of error strings (empty = valid); every message names
    the offending spec field.  Purely structural -- semantics (file
    contents, frame alignment) stay at score time.
    """
    errors: list[str] = []
    where = f"term {term!r}"
    for p in TERM_REQUIRED_PARAMS[term]:
        if p not in spec:
            errors.append(f"{where}: missing required param '{p}'")
    if term == "frame_fraction":
        if "pred_spec" in spec:
            errors.append(
                f"{where}.pred_spec: free-form expressions are not "
                "supported in v1 (whitelist-only DSL); use target/tol or "
                "op/value"
            )
        target, op, value = spec.get("target"), spec.get("op"), spec.get("value")
        if target is not None and op is not None:
            errors.append(
                f"{where}: predicate forms are mutually exclusive " "({target, tol} vs {op, value})"
            )
        if op is not None:
            if op not in OPS:
                errors.append(f"{where}.op: unknown op {op!r} (whitelist: " f"{', '.join(OPS)})")
            if value is None:
                errors.append(f"{where}: op form requires a constant 'value'")
        if target is None and op is None and "pred_spec" not in spec:
            errors.append(
                f"{where}: missing predicate -- provide " "{target(, tol)} or {op, value}"
            )
        if "tol" in spec and target is None:
            errors.append(f"{where}.tol: only valid together with 'target'")
        post = spec.get("post")
        if post is not None and post not in POSTS:
            errors.append(
                f"{where}.post: unknown post {post!r} (whitelist: " f"{', '.join(POSTS)})"
            )
        if post in ("abs", "x2") and spec.get("dims") is None and spec.get("axis") is None:
            errors.append(
                f"{where}.post: {post!r} needs pinned single column "
                "(dims/axis); multi-column only supports norm2"
            )
        if spec.get("dims") is not None and spec.get("axis") is not None:
            errors.append(f"{where}: 'dims' and 'axis' are mutually exclusive")
        _check_dims(spec, errors, where)
        if spec.get("axis") is not None and (
            not isinstance(spec["axis"], int) or isinstance(spec["axis"], bool)
        ):
            errors.append(f"{where}.axis: must be an int column, got " f"{spec['axis']!r}")
    elif term in ("frame_mean", "frame_var"):
        post = spec.get("post")
        if post is not None and post not in POSTS:
            errors.append(
                f"{where}.post: unknown post {post!r} (whitelist: " f"{', '.join(POSTS)})"
            )
        _check_dims(spec, errors, where)
    elif term in ("final", "min", "max"):
        _check_dims(spec, errors, where)
    elif term == "tracking_error":
        for p in ("state", "cmd"):
            if not isinstance(spec.get(p), str):
                errors.append(f"{where}.{p}: must be a recorded-key string")
        for p in ("state_dims", "cmd_dims"):
            _check_dims(spec, errors, where, key=p)
        red = spec.get("reduction", "mean")
        if red not in REDUCTIONS:
            errors.append(
                f"{where}.reduction: unknown reduction {red!r} (whitelist: "
                f"{', '.join(REDUCTIONS)})"
            )
        _check_dims(spec, errors, where)
    elif term == "range_check_fraction":
        _check_dims(spec, errors, where)
        lo, hi = spec.get("lo"), spec.get("hi")
        for name_, v in (("lo", lo), ("hi", hi)):
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                errors.append(f"{where}.{name_}: must be a number, got {v!r}")
        if (
            isinstance(lo, (int, float))
            and isinstance(hi, (int, float))
            and not isinstance(lo, bool)
            and not isinstance(hi, bool)
            and lo > hi
        ):
            errors.append(f"{where}: lo ({lo}) > hi ({hi})")
    elif term == "delta":
        _check_dims(spec, errors, where)
        red = spec.get("reduction")
        if red is not None and red not in DELTA_REDUCTIONS:
            errors.append(
                f"{where}.reduction: unknown reduction {red!r} (whitelist: "
                f"{', '.join(DELTA_REDUCTIONS)})"
            )
    elif term == "point_in_box_fraction":
        for p in ("center", "half_extents"):
            v = spec.get(p)
            ok = (
                isinstance(v, (list, tuple))
                and len(v) == 3
                and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
            )
            if not ok:
                errors.append(f"{where}.{p}: must be [x, y, z] numbers, got {v!r}")
            elif (
                p == "half_extents"
                and isinstance(v, (list, tuple))
                and any(isinstance(x, (int, float)) and x < 0 for x in v)
            ):
                errors.append(f"{where}.{p}: all half lengths must be >= 0, got {v!r}")
    elif term == "reference_mpjpe":
        ref = spec.get("reference")
        if isinstance(ref, str):
            p = Path(ref)
            if not p.is_absolute():
                p = REFERENCES_DIR / p
            if not p.is_file():
                errors.append(
                    f"{where}.reference: file not found: {p} (absolute "
                    f"path or relative to {REFERENCES_DIR})"
                )
        elif "reference" in spec:
            errors.append(f"{where}.reference: must be an npz path string, got {ref!r}")
    elif term == "contact_any":
        c = spec.get("contact")
        if "contact" in spec and (not isinstance(c, str) or not c):
            errors.append(
                f"{where}.contact: must be a recorded-key string "
                f"(contact/<name> or a bare group name), got {c!r}"
            )
        red = spec.get("reduce_bodies")
        if "reduce_bodies" in spec and red not in ("any", "all"):
            errors.append(f"{where}.reduce_bodies: must be 'any' or 'all', got {red!r}")
    elif term == "keypoint_in_region_fraction":
        errors.extend(_validate_keypoint_region_spec(spec, where))
    elif term == "hold_streak":
        # the per-entry form whitelist + the one-group-level rule live in
        # benchmarks/holding.py (validate_hold_specs)
        errors.extend(_validate_hold_specs(spec.get("specs"), where))
        return errors
    return errors
