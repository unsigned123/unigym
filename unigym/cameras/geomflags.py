"""Capture classification + seg helpers, and the LEGACY flags validator.

History: the flags triple ``{geom, contacts, segmentation}`` was the
v052/v053 capture payload; v054 replaces it with the RenderSpec
protocol (``cameras/spec.py``, camera_contract §14) and DELETES the
flags face everywhere but ONE place: :func:`validate_flags` survives as
the validator behind ``spec.expand_flags_alias`` (the CLI ``--flags``
deprecated alias expands its payload through it -- the alias machinery
itself).

What permanently lives here (engine-free, numpy in / numpy out):
the contype/conaffinity collision classification (:func:`collision_mask`),
the §8.1 load-time layout (:func:`reclassify_geom_groups` +
:data:`VISUAL_GROUP`), the seg palette/colorize/legend helpers, and the
per-config ``render_groups`` validator.  The mujoco rig
(``backends/mujoco/cameras.py``) feeds it model arrays and a name
resolver.

XML audit (the mapping's final form, humanoid_1 2026-10-14)
==========================================================
The design anticipated "geom group switching + name/contype fallback" -- the probe
found group conventions are NOT consistent across assets:

* ``scene_43dof.xml`` (G1): 50 visual-mesh geoms (group 1, contype 0,
  conaffinity 0) + 48 collision primitives (group 0, contype 1,
  conaffinity 1);
* ``assets/cracker_mjcf/object.xml`` (crate): visual/collision are BOTH
  group 0 -- the only discriminating signal is ``contype/conaffinity``;
* box props built by ``backends/shared/mjfamily/props/props.py`` are SINGLE geoms
  (default contype 1): they collide, so they are collision-class and
  render in both modes (honest: their collision geometry IS their
  visual geometry).

Final mapping (contract §14): **contype/conaffinity rule wins** -- a
geom is visual-only iff ``contype == 0 and conaffinity == 0``; a
per-config ``render_groups: {visual: [g...], collision: [g...]}``
override (env.params, camera-manager build arg) takes precedence when
declared, for assets that DO follow a group convention.

Render mechanics (flags v2, design §8.1): ``geom_group`` is a pure
visualization field (physics never reads it), so the rig applies the
classification ONCE at bind time -- visual-only geoms move PERMANENTLY
into the dedicated visual group (:data:`VISUAL_GROUP`) and the shared
MjModel stays read-only for the render face from then on.  Every later
capture is a pure per-render ``scene_option`` switch (collision mode
hides the visual group); the per-render mutate/restore of the first
flags round is gone -- zero model mutation on the render path, which is
what makes the shared model safe under a future multi-threaded
renderer.
"""
from __future__ import annotations

import numpy as np

__all__ = ["FLAG_KEYS", "GEOM_MODES", "VISUAL_GROUP",
           "validate_flags",
           "validate_render_groups", "reclassify_geom_groups",
           "collision_mask", "seg_palette",
           "colorize_seg", "build_legend", "legend_type_name"]

FLAG_KEYS = ("geom", "contacts", "segmentation")
GEOM_MODES = ("visual", "collision")
VISUAL_GROUP = 5
"""Reserved by the (legacy) flags face: the PERMANENT home of visual-only
geoms after the load-time reclassification.  The collision-truth
RenderSpec (``scene.geom_class: "collision"``) hides this group through
the per-render scene_option; nothing ever moves ``geom_group`` at render
time.  Per-config ``render_groups`` declaring collision geoms in this
group is a config error (fail-fast), and so is a model that natively
carries a collision-class geom here."""

SEG_BACKGROUND = (0, 0, 0)


def validate_flags(flags) -> dict:
    """Normalize one COMPLETE legacy flags payload; raise ValueError with
    the legal keys otherwise.

    SOLE surviving consumer: ``spec.expand_flags_alias`` (the CLI
    ``--flags`` deprecated alias -- the payload is expanded to a
    RenderSpec before anything else sees it).  Complete-or-error: every
    payload must carry ALL of :data:`FLAG_KEYS` (missing keys -> error
    listing them plus the legal values; unknown keys, a bad ``geom``
    mode or non-bool toggles are value errors).  Fail-fast, no silent
    flag dropping (a silently dropped ``geom: collision`` would lie to
    the caller about what the pixels show).
    """
    if not isinstance(flags, dict):
        raise ValueError(
            f"capture flags must be a complete dict {list(FLAG_KEYS)}, "
            f"got {flags!r} -- the alias expander has no defaults "
            f"(CLI --flags is deprecated: use --view/--spec)")
    unknown = sorted(set(flags) - set(FLAG_KEYS))
    if unknown:
        raise ValueError(
            f"unknown capture flag(s) {unknown}; legal flags: "
            f"{list(FLAG_KEYS)} (camera_contract §14)")
    missing = [k for k in FLAG_KEYS if k not in flags]
    if missing:
        raise ValueError(
            f"capture flags: missing required key(s) {missing}; the "
            f"payload must carry ALL of {list(FLAG_KEYS)} -- legal "
            f"values: geom {list(GEOM_MODES)}, contacts/segmentation "
            f"bool (CLI --flags is deprecated: use --view/--spec)")
    mode = flags["geom"]
    if mode not in GEOM_MODES:
        raise ValueError(
            f"capture flag 'geom' must be one of {list(GEOM_MODES)}, "
            f"got {mode!r}")
    out = {"geom": mode}
    for k in ("contacts", "segmentation"):
        v = flags[k]
        if not isinstance(v, bool):
            raise ValueError(
                f"capture flag {k!r} must be a bool, got {v!r}")
        out[k] = v
    return out


def validate_render_groups(render_groups) -> None:
    """Fail-fast the per-config ``render_groups`` declaration (build time).

    Legal shape: ``{"visual": [groups...], "collision": [groups...]}``
    (both optional, lists of ints, disjoint, no collision group in the
    reserved visual-only group).  Raises ``ValueError`` on anything else.
    """
    if not isinstance(render_groups, dict):
        raise ValueError(
            f"render_groups must be a dict with optional keys "
            f"['visual', 'collision'], got {render_groups!r}")
    unknown = set(render_groups) - {"visual", "collision"}
    if unknown:
        raise ValueError(
            f"render_groups: unknown key(s) {sorted(unknown)} "
            f"(legal: ['collision', 'visual'])")
    visual = set(render_groups.get("visual") or [])
    coll = set(render_groups.get("collision") or [])
    for side, groups in (("visual", visual), ("collision", coll)):
        for g in groups:
            if isinstance(g, bool) or not isinstance(g, int):
                raise ValueError(
                    f"render_groups.{side}: groups must be integers, "
                    f"got {g!r}")
    if visual & coll:
        raise ValueError(
            f"render_groups: group(s) {sorted(visual & coll)} are declared "
            f"both visual and collision (ambiguous)")
    if VISUAL_GROUP in coll:
        raise ValueError(
            f"render_groups: collision group {VISUAL_GROUP} is reserved by "
            f"the flags face (the visual-only home group)")


def collision_mask(contype, conaffinity, geom_group,
                   render_groups=None) -> np.ndarray:
    """Per-geom collision-class bool mask (the audit's final mapping).

    Default rule: collision iff ``contype != 0 or conaffinity != 0``
    (visual-only = both zero).  ``render_groups`` override:
    ``{"visual": [groups...], "collision": [groups...]}`` -- group lists
    must be disjoint; groups listed in neither stay on the default rule.
    """
    ct = np.asarray(contype) != 0
    ca = np.asarray(conaffinity) != 0
    mask = ct | ca
    if render_groups:
        validate_render_groups(render_groups)
        visual = set(render_groups.get("visual") or [])
        coll = set(render_groups.get("collision") or [])
        gg = np.asarray(geom_group).astype(int)
        for g in visual:
            mask[gg == g] = False
        for g in coll:
            mask[gg == g] = True
    return mask


def reclassify_geom_groups(contype, conaffinity, geom_group,
                           render_groups=None) -> np.ndarray:
    """The §8.1 load-time layout: geom_group with visual-only geoms moved
    PERMANENTLY into :data:`VISUAL_GROUP`.

    One-time, at rig bind (never per render): the returned array is what
    the shared MjModel keeps for its whole life -- every later collision
    capture is a pure ``scene_option`` switch.  Idempotent by
    construction (the mask reads contype/conaffinity, which the
    reclassification never touches).  Fail-fast when a collision-class
    geom sits in the reserved visual group natively -- hiding the visual
    group would silently drop it from ``geom: collision`` renders.
    """
    gg = np.asarray(geom_group).astype(int)
    natively_reserved = gg[(gg == VISUAL_GROUP)
                           & collision_mask(contype, conaffinity, gg,
                                            render_groups)]
    if natively_reserved.size:
        raise ValueError(
            f"reclassify: {natively_reserved.size} collision-class geom(s) "
            f"natively live in the reserved visual group {VISUAL_GROUP}; "
            f"remap them in the asset XML or via render_groups -- the "
            f"collision render hides that group whole")
    new = gg.copy()
    new[~collision_mask(contype, conaffinity, gg, render_groups)] = \
        VISUAL_GROUP
    return new


# 24-entry stable palette (no black; deterministic per id -- the legend
# carries the explicit color so consumers never recompute the hash).
_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 216, 177), (0, 0, 128), (128, 128, 128),
    (100, 60, 120), (40, 160, 160), (200, 90, 40), (90, 200, 90),
]


def seg_palette(n: int) -> np.ndarray:
    """First ``n`` palette colors, cycled; (n, 3) uint8."""
    idx = np.arange(max(int(n), 1)) % len(_PALETTE)
    return np.asarray(_PALETTE, dtype=np.uint8)[idx]


def colorize_seg(seg_ids: np.ndarray) -> np.ndarray:
    """(H, W) int id buffer -> (H, W, 3) uint8 colored PNG face.

    Negative ids (background / no hit) render black.  Valid ids map
    through the stable palette (id 0 = background color slot is
    impossible: MuJoCo ids are >= 0, < 0 = miss).
    """
    seg = np.asarray(seg_ids)
    out = np.zeros(seg.shape + (3,), dtype=np.uint8)
    valid = seg >= 0
    if valid.any():
        colors = seg_palette(int(seg[valid].max()) + 1)
        out[valid] = colors[seg[valid]]
    return out


def legend_type_name(obj_type: int) -> str:
    """MuJoCo mjtObj int -> stable label (engine-free table)."""
    return {1: "body", 5: "geom", 6: "site"}.get(
        int(obj_type), f"obj{int(obj_type)}")


def build_legend(entries: list[dict], mode: dict) -> dict:
    """Legend document from the rig's per-object rows (schema 1).

    ``entries``: ``{"id", "type", "name", "body", "color", "pixels"}``
    rows (named objects must carry a real name; unnamed geoms are
    labeled ``<type>_<id>`` -- honest, addressable, never silent).
    """
    return {"schema": 1, "mode": dict(mode), "objects": sorted(
        entries, key=lambda e: (-int(e.get("pixels", 0)), int(e["id"])))}
