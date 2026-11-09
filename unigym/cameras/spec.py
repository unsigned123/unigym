"""RenderSpec declarative compiler (render-layers design §3-§5): schema,
merge, registry.

A RenderSpec is PURE JSON describing one capture's rendering::

    {"camera": {"mode": "fixed"|"orbit"|"closeup", "target": ..., "d": ...,
                "el": ..., "az": ..., "orbit": {...}, "views": [...],
                "contact_select": {...}},
     "appearance": [{"select": {...}, "rgba": [...], "matid": -1}, ...],
     "decor": [{"gen": "cage_grid", "target": "crate", ...}, ...],
     "buffers": ["rgb"] | ["rgb", "seg"],
     "scene": {"geom_class": "all"|"collision", "shadow": bool|None, ...},
     "output": {"gif": bool, "gif_ms": int, "caption": bool, "legend": bool}}

Four layers (design §2): primitives (engine, L1.5) -> generators
(``decor`` registry) -> presets (named fragments) -> views (named
compositions).  This module owns the middle two registries and the
aspect-slot merge:

===================  ==========================================  =========
aspect               merge rule                                  conflict
===================  ==========================================  =========
camera/buffers/      exclusive slot: at most one declarer        error
output/scene
appearance           keyed by canonical selector; different      error on
                     selectors coexist (hand rule + floor rule)  same key
decor                additive list keyed by (gen, target)        error on
                                                                 same key
===================  ==========================================  =========

``combine`` is a pure, ORDER-INDEPENDENT function (A+B == B+A): every
conflict is a :class:`RenderConflictError` carrying the conflicting
aspect and BOTH source names -- build-time fail-fast, never a silent
override.  Named frames beyond the primary: ``camera.views`` selects
from the canonical set (top/side/front/iso, all relative to the camera
azimuth; probe5 viewpoint table).

Schema deltas vs the design sketch (documented in camera_contract §14.10):
``scene`` aspect added (collision-truth views + bitwise-neutral
scn-flag control); ``output.views`` dropped (duplicate of
``camera.views``); ``camera.contact_select`` added (the design's
``target: "contact_centroid"`` needs a contact filter -- the probe5 GT
centroid is hand-contact filtered); ``depth_tint`` carries explicit
shallow/deep tint colors (the probe gradient).

``camera.pos`` + ``camera.lookat`` (world metres) join the
schema -- the absolute-pose free-camera form (the camera aim
semantics), mutually exclusive with the spherical ``target``/``d``/
``el``/``az`` form in the SAME dict.  The backend converts it to the
spherical keys before rendering (:func:`spherical_camera_keys` -- the
deleted view_presets math, promoted to a first-class capability).
"""

from __future__ import annotations

import math

from .appearance import selector_key, validate_appearance, validate_selector

__all__ = [
    "RenderConflictError",
    "VIEW_MODES",
    "NAMED_VIEWS",
    "CAMERA_KEYS",
    "SCENE_KEYS",
    "OUTPUT_KEYS",
    "spherical_camera_keys",
    "register_preset",
    "register_view",
    "preset_names",
    "view_names",
    "get_preset",
    "get_view",
    "validate_fragment",
    "combine",
    "complete_spec",
    "resolve_render",
    "spec_declaration",
    "expand_flags_alias",
]

VIEW_MODES = ("fixed", "orbit", "closeup")
NAMED_VIEWS = ("top", "side", "front", "iso")
CAMERA_KEYS = (
    "mode",
    "target",
    "d",
    "el",
    "az",
    "orbit",
    "views",
    "contact_select",
    "pos",
    "lookat",
)
#: the spherical form the absolute-pose form (pos/lookat) excludes --
#: one pose language per camera dict, co-occurrence = teaching error
SPHERICAL_CAMERA_KEYS = ("target", "d", "el", "az")
ORBIT_KEYS = ("n", "step_deg", "ms")
SCENE_KEYS = ("geom_class", "shadow", "reflection", "skybox")
OUTPUT_KEYS = ("gif", "gif_ms", "caption", "legend")
EXCLUSIVE_ASPECTS = ("camera", "buffers", "output", "scene")

DEFAULT_ORBIT = {"n": 24, "step_deg": 15.0, "ms": 125}

# probe5 viewpoint table (findings.txt §D): named frames relative to the
# camera azimuth, lookat = target.
NAMED_VIEW_PARAMS = {
    "top": {"el": -89.9, "d": 0.24, "az_off": 0.0},
    "side": {"el": -8.0, "d": 0.20, "az_off": 90.0},
    "front": {"el": -8.0, "d": 0.20, "az_off": 180.0},
    "iso": {"el": -45.0, "d": 0.20, "az_off": 135.0},
}


class RenderConflictError(ValueError):
    """Two declarers claim one aspect slot / merge key (design §4).

    Attributes: ``aspect`` (the conflicting aspect) and ``sources``
    (both declarer names, order-normalized).  Build-time fail-fast: the
    merge never silently overrides."""

    def __init__(self, aspect: str, sources: list):
        self.aspect = aspect
        self.sources = sorted(str(s) for s in sources)
        super().__init__(
            f"render spec conflict on aspect {aspect!r}: declared by "
            f"{self.sources} -- combine() never silently overrides "
            f"(camera_contract §14)"
        )


# --------------------------------------------------------------------- #
# fragment validation (partial specs are legal: presets are fragments)
# --------------------------------------------------------------------- #
def spherical_camera_keys(pos, lookat) -> dict:
    """``(pos, lookat)`` world metres -> the EXACT MjvCamera spherical
    keys ``{"target", "d", "el", "az"}`` (the v062 deleted view_presets
    math, promoted to a first-class spec capability).

    The free-camera math is exact (empirically pinned, probe delta
    < 1e-7 m) -- with az/el in radians::

        forward = [cos(el)cos(az), cos(el)sin(az), sin(el)]
        cam_pos = lookat - d * forward            (el < 0 = camera above)

    so a desired ``(cam_pos, lookat)`` inverts to ``el = asin(-vz/d)``,
    ``az = atan2(-vy, -vx)``, ``d = |v|`` with ``v = cam_pos - lookat``
    (round-trip exact, unit-locked)."""
    px, py, pz = (float(x) for x in pos)
    lx, ly, lz = (float(x) for x in lookat)
    vx, vy, vz = px - lx, py - ly, pz - lz
    d = math.sqrt(vx * vx + vy * vy + vz * vz)
    if d < 1e-6:
        raise ValueError("camera pos == lookat (degenerate view)")
    el = math.degrees(math.asin(max(-1.0, min(1.0, -vz / d))))
    az = math.degrees(math.atan2(-vy, -vx))
    return {"target": [lx, ly, lz], "d": d, "el": el, "az": az}


def _check_camera(cam, label: str) -> dict:
    if not isinstance(cam, dict):
        raise ValueError(f"{label}: camera must be a dict, got {cam!r}")
    unknown = sorted(set(cam) - set(CAMERA_KEYS))
    if unknown:
        raise ValueError(
            f"{label}: unknown camera key(s) {unknown}; " f"legal: {list(CAMERA_KEYS)}"
        )
    out: dict = {}
    mode = cam.get("mode", "fixed")
    if mode not in VIEW_MODES:
        raise ValueError(
            f"{label}: camera.mode must be one of " f"{list(VIEW_MODES)}, got {mode!r}"
        )
    out["mode"] = mode
    # the absolute-pose form (pos/lookat, world metres -- the
    # camera(aim) semantics) is mutually exclusive with the spherical
    # form (target/d/el/az) in the SAME dict -- co-occurrence is a
    # teaching error, never a silent priority.  The backend converts
    # pos/lookat to the spherical keys before rendering.
    if ("pos" in cam) or ("lookat" in cam):
        clash = sorted(set(SPHERICAL_CAMERA_KEYS) & set(cam))
        if clash:
            raise RenderConflictError(
                "camera.pos/lookat",
                ["pos+lookat (absolute pose)", f"{'+'.join(clash)} (spherical)"],
            )
        if not ("pos" in cam and "lookat" in cam):
            raise ValueError(
                f"{label}: camera.pos and camera.lookat go TOGETHER "
                f"(world metres -- the absolute free-camera pose), got "
                f"pos={cam.get('pos')!r} lookat={cam.get('lookat')!r}"
            )
        for k in ("pos", "lookat"):
            v = cam[k]
            if (
                not isinstance(v, (list, tuple))
                or len(v) != 3
                or not all(
                    isinstance(x, (int, float))
                    and not isinstance(x, bool)
                    and math.isfinite(float(x))
                    for x in v
                )
            ):
                raise ValueError(
                    f"{label}: camera.{k} must be [x, y, z] finite "
                    f"numbers (world metres), got {v!r}"
                )
            out[k] = [float(x) for x in v]
    if "target" in cam:
        t = cam["target"]
        if t != "contact_centroid":
            if (
                not isinstance(t, (list, tuple))
                or len(t) != 3
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in t)
            ):
                raise ValueError(
                    f"{label}: camera.target must be 'contact_centroid' " f"or [x, y, z], got {t!r}"
                )
            t = [float(x) for x in t]
        out["target"] = t
    for k in ("d", "el", "az"):
        if k in cam:
            v = cam[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"{label}: camera.{k} must be a number, " f"got {v!r}")
            out[k] = float(v)
    if "orbit" in cam:
        orb = cam["orbit"]
        if not isinstance(orb, dict):
            raise ValueError(
                f"{label}: camera.orbit must be a dict " f"{list(ORBIT_KEYS)}, got {orb!r}"
            )
        unknown = sorted(set(orb) - set(ORBIT_KEYS))
        if unknown:
            raise ValueError(
                f"{label}: unknown camera.orbit key(s) " f"{unknown}; legal: {list(ORBIT_KEYS)}"
            )
        norm = dict(DEFAULT_ORBIT)
        for k in ORBIT_KEYS:
            if k in orb:
                v = orb[k]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                    raise ValueError(
                        f"{label}: camera.orbit.{k} must be " f"a positive number, got {v!r}"
                    )
                norm[k] = v
        norm["n"] = int(norm["n"])
        norm["ms"] = int(norm["ms"])
        out["orbit"] = norm
    if "views" in cam:
        views = cam["views"]
        if not isinstance(views, (list, tuple)) or not all(v in NAMED_VIEWS for v in views):
            raise ValueError(
                f"{label}: camera.views must be a list drawn from "
                f"{list(NAMED_VIEWS)}, got {views!r}"
            )
        out["views"] = list(dict.fromkeys(views))
    if "contact_select" in cam:
        sel = cam["contact_select"]
        out["contact_select"] = None if sel is None else validate_selector(sel)
    return out


def validate_fragment(spec, label: str = "spec") -> dict:
    """Structural validation of a (possibly partial) spec fragment.

    Unknown top-level keys fail-fast with the legal list; per-aspect
    entries are normalized (appearance selectors, decor params through
    the generator validators).  The result is a fragment dict with
    ONLY the declared aspects present."""
    if not isinstance(spec, dict):
        raise ValueError(f"{label}: render spec must be a dict, got " f"{spec!r}")
    legal = set(EXCLUSIVE_ASPECTS) | {"appearance", "decor"}
    unknown = sorted(set(spec) - legal)
    if unknown:
        raise ValueError(
            f"{label}: unknown render spec key(s) {unknown}; legal keys: "
            f"{sorted(legal)} (camera_contract §14)"
        )
    out: dict = {}
    if "camera" in spec:
        out["camera"] = _check_camera(spec["camera"], label)
    if "appearance" in spec:
        out["appearance"] = validate_appearance(spec["appearance"], label)
    if "decor" in spec:
        if not isinstance(spec["decor"], (list, tuple)):
            raise ValueError(f"{label}: decor must be a list of entries, " f"got {spec['decor']!r}")
        # lazy import: the decor package (generators) is the registry
        # this validation consults; keeping it function-local lets the
        # schema layer be imported (and unit-tested) standalone
        from .decor import validate_decor_entry

        out["decor"] = [validate_decor_entry(e, label) for e in spec["decor"]]
    if "buffers" in spec:
        bufs = spec["buffers"]
        if (
            not isinstance(bufs, (list, tuple))
            or not bufs
            or not all(b in ("rgb", "seg") for b in bufs)
        ):
            raise ValueError(
                f"{label}: buffers must be a non-empty list drawn from "
                f"['rgb', 'seg'], got {bufs!r}"
            )
        out["buffers"] = ["rgb"] + (["seg"] if "seg" in bufs else [])
    if "scene" in spec:
        sc = spec["scene"]
        if not isinstance(sc, dict):
            raise ValueError(f"{label}: scene must be a dict, got {sc!r}")
        unknown = sorted(set(sc) - set(SCENE_KEYS))
        if unknown:
            raise ValueError(
                f"{label}: unknown scene key(s) {unknown}; " f"legal: {list(SCENE_KEYS)}"
            )
        norm = {}
        if "geom_class" in sc:
            if sc["geom_class"] not in ("all", "collision"):
                raise ValueError(
                    f"{label}: scene.geom_class must be 'all'|'collision',"
                    f" got {sc['geom_class']!r}"
                )
            norm["geom_class"] = sc["geom_class"]
        for k in ("shadow", "reflection", "skybox"):
            if k in sc:
                v = sc[k]
                if v is not None and not isinstance(v, bool):
                    raise ValueError(
                        f"{label}: scene.{k} must be bool or "
                        f"null (null = engine default), got "
                        f"{v!r}"
                    )
                norm[k] = v
        out["scene"] = norm
    if "output" in spec:
        op = spec["output"]
        if not isinstance(op, dict):
            raise ValueError(f"{label}: output must be a dict, got {op!r}")
        unknown = sorted(set(op) - set(OUTPUT_KEYS))
        if unknown:
            raise ValueError(
                f"{label}: unknown output key(s) {unknown}; " f"legal: {list(OUTPUT_KEYS)}"
            )
        norm = {}
        for k in ("gif", "caption", "legend"):
            if k in op:
                if not isinstance(op[k], bool):
                    raise ValueError(f"{label}: output.{k} must be bool, " f"got {op[k]!r}")
                norm[k] = op[k]
        if "gif_ms" in op:
            v = op["gif_ms"]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                raise ValueError(f"{label}: output.gif_ms must be a " f"positive number, got {v!r}")
            norm["gif_ms"] = int(v)
        out["output"] = norm
    return out


# --------------------------------------------------------------------- #
# aspect merge (pure, order-independent)
# --------------------------------------------------------------------- #
def combine(entries: list) -> dict:
    """Merge validated fragments into one; ORDER-INDEPENDENT.

    ``entries``: list of ``(name, fragment)`` with fragments from
    :func:`validate_fragment`.  Exclusive aspects (camera/buffers/output/
    scene) accept ONE declarer; appearance merges by canonical selector
    key; decor merges by ``(gen, target)``.  Any double-claim raises
    :class:`RenderConflictError` with both source names."""
    if not entries:
        return {}
    merged: dict = {}
    owner: dict = {}

    def claim(aspect: str, name) -> None:
        if aspect in owner:
            raise RenderConflictError(aspect, [owner[aspect], name])
        owner[aspect] = name

    # canonical processing order = sorted by (name, aspect) so the ERROR
    # and the RESULT never depend on the caller's entry order
    for name, frag in sorted(entries, key=lambda e: str(e[0])):
        for aspect in EXCLUSIVE_ASPECTS:
            if aspect in frag:
                claim(aspect, name)
                val = frag[aspect]
                merged[aspect] = dict(val) if isinstance(val, dict) else list(val)
        if "appearance" in frag:
            slot = merged.setdefault("appearance", [])
            seen = {selector_key(e["select"]) for e in slot}
            for e in frag["appearance"]:
                key = selector_key(e["select"])
                if key in seen:
                    raise RenderConflictError(
                        f"appearance[{key}]", [_appearance_owner(merged, key), name]
                    )
                seen.add(key)
                slot.append(e)
        if "decor" in frag:
            slot = merged.setdefault("decor", [])
            seen = {(e["gen"], e["target"]) for e in slot}
            for e in frag["decor"]:
                key = (e["gen"], e["target"])
                if key in seen:
                    raise RenderConflictError(
                        f"decor[{key[0]}@{key[1]}]", [_decor_owner(merged, key), name]
                    )
                seen.add(key)
                slot.append(e)
    return merged


def _appearance_owner(merged: dict, key: str) -> str:
    return f"appearance[{key}] (first declarer)"


def _decor_owner(merged: dict, key: tuple) -> str:
    return f"decor[{key[0]}@{key[1]}] (first declarer)"


def complete_spec(fragment: dict) -> dict:
    """Fill the defaults -> the COMPLETE normalized spec (the
    ``effective_render`` echo).  Defaults = historical plain render:
    fixed rig camera, no appearance/decor, rgb buffer, engine scene
    flags, no caption -- view ``visual`` renders bitwise-identically to
    the flagless path by construction."""
    cam = {"mode": "fixed", "target": "contact_centroid", "contact_select": None}
    cam.update(fragment.get("camera", {}))
    if "pos" in cam or "lookat" in cam:
        # pos/lookat IS the pose -- the contact_centroid default
        # target must not linger beside it (the spherical form is
        # excluded from the same dict)
        cam.pop("target", None)
    if cam["mode"] == "orbit" and "orbit" not in cam:
        cam["orbit"] = dict(DEFAULT_ORBIT)
    spec = {
        "camera": cam,
        "appearance": list(fragment.get("appearance", [])),
        "decor": list(fragment.get("decor", [])),
        "buffers": list(fragment.get("buffers", ["rgb"])),
        "scene": {"geom_class": "all", "shadow": None, "reflection": None, "skybox": None},
        "output": {"gif": False, "gif_ms": 125, "caption": False, "legend": False},
    }
    spec["scene"].update(fragment.get("scene", {}))
    spec["output"].update(fragment.get("output", {}))
    return spec


# --------------------------------------------------------------------- #
# registries (presets + views)
# --------------------------------------------------------------------- #
PRESETS: dict[str, dict] = {}
VIEWS: dict[str, dict] = {}


def spec_declaration(fragment: dict) -> dict:
    """What a fragment actually writes (registration-time truth)."""
    decl: dict = {
        "aspects": sorted(a for a in EXCLUSIVE_ASPECTS if a in fragment),
        "appearance": sorted(selector_key(e["select"]) for e in fragment.get("appearance", [])),
        "decor": sorted(f"{e['gen']}@{e['target']}" for e in fragment.get("decor", [])),
    }
    return decl


def _check_claims(name: str, claims, fragment: dict) -> None:
    decl = spec_declaration(fragment)
    if claims is None:
        return
    if not isinstance(claims, dict):
        raise ValueError(
            f"preset {name!r}: claims must be a dict "
            f"{{aspects?, appearance?, decor?}}, got {claims!r}"
        )
    unknown = sorted(set(claims) - {"aspects", "appearance", "decor"})
    if unknown:
        raise ValueError(
            f"preset {name!r}: unknown claims key(s) "
            f"{unknown}; legal: ['appearance', 'aspects', "
            f"'decor']"
        )
    if sorted(claims.get("aspects", [])) != decl["aspects"]:
        raise ValueError(
            f"preset {name!r}: claims.aspects "
            f"{sorted(claims.get('aspects', []))} != the spec's actual "
            f"aspects {decl['aspects']} -- claims must match the spec "
            f"(fail-fast at registration, camera_contract §14)"
        )
    if sorted(claims.get("appearance", [])) != decl["appearance"]:
        raise ValueError(
            f"preset {name!r}: claims.appearance "
            f"{sorted(claims.get('appearance', []))} != the spec's "
            f"actual selector keys {decl['appearance']}"
        )
    if sorted(claims.get("decor", [])) != decl["decor"]:
        raise ValueError(
            f"preset {name!r}: claims.decor {sorted(claims.get('decor', []))}"
            f" != the spec's actual decor keys {decl['decor']}"
        )


def register_preset(name: str, claims=None, spec: dict | None = None):
    """Register a named fragment (design §5).  ``claims`` must equal the
    fragment's actual declaration (fail-fast at import -- a lying claim
    would mislead every future composition)."""
    frag = validate_fragment(spec or {}, f"preset {name}")
    _check_claims(name, claims, frag)
    if name in PRESETS:
        raise ValueError(f"preset {name!r} registered twice")
    PRESETS[name] = frag
    return frag


def register_view(name: str, *entries):
    """Register a named composition: ``combine`` runs NOW, so conflicts
    between a view's own presets fail at import, never at capture."""
    if name in VIEWS:
        raise ValueError(f"view {name!r} registered twice")
    VIEWS[name] = combine(list(entries))
    return VIEWS[name]


def get_preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(
            f"unknown preset {name!r}; available presets: "
            f"{sorted(PRESETS)} (camera_contract §14)"
        )
    return PRESETS[name]


def get_view(name: str) -> dict:
    if name not in VIEWS:
        raise ValueError(
            f"unknown view {name!r}; available views: " f"{sorted(VIEWS)} (camera_contract §14)"
        )
    return VIEWS[name]


def preset_names() -> list:
    return sorted(PRESETS)


def view_names() -> list:
    return sorted(VIEWS)


# --------------------------------------------------------------------- #
# consumer entry point
# --------------------------------------------------------------------- #
def resolve_render(render) -> tuple[dict, str]:
    """The compiler front-end: view name, view + overrides, or a full
    inline spec -> ``(complete spec, label)``.

    * ``{"view": "palm_pose_occl"}`` -- the registered view verbatim;
    * ``{"view": ..., <fragment keys>}`` -- the view COMBINED with the
      overrides (complementary keys merge; a key the view already owns
      raises RenderConflictError -- fail-fast, never silent);
    * any other dict -- a full inline spec (complete-or-error at
      :func:`complete_spec`)."""
    if not isinstance(render, dict):
        raise ValueError(
            f"render must be a dict: {{'view': name}}, {{'view': name, "
            f"...overrides}} or a full inline spec ({{}} = all defaults);"
            f" available views: {view_names()} (camera_contract §14)"
        )
    if "view" in render:
        name = render["view"]
        if not isinstance(name, str):
            raise ValueError(f"render.view must be a string, got {name!r}")
        label = f"view:{name}"
        entries = [(label, get_view(name))]
        rest = {k: v for k, v in render.items() if k != "view"}
        if rest:
            entries.append(
                (f"override[{sorted(rest)}]", validate_fragment(rest, "render override"))
            )
        return complete_spec(combine(entries)), label
    return complete_spec(validate_fragment(render, "render")), "spec"


def expand_flags_alias(flags: dict) -> dict:
    """Legacy flags triple -> COMPLETE RenderSpec (design §5 migration
    map; the ONLY surviving flags path -- the CLI
    ``--flags`` deprecated alias.  The CLI expands its payload through
    this function BEFORE anything hits the wire; the webui/daemon/
    bridge layers have no flags form at all).

    =======================  ==================================
    ``geom: "collision"``    scene.geom_class = "collision"
    ``geom: "visual"``       (no appearance; historical render)
    ``contacts: true``       preset ``contact_mjvis_simple``
    ``segmentation: true``   buffers = ["rgb", "seg"]
    =======================  =================================="""
    from .geomflags import validate_flags

    norm = validate_flags(flags)
    frag: dict = {}
    if norm["geom"] == "collision":
        frag["scene"] = {"geom_class": "collision"}
    decor = []
    if norm["contacts"]:
        decor.append({"gen": "contact_mjvis", "target": "*"})
    if decor:
        frag["decor"] = decor
    if norm["segmentation"]:
        frag["buffers"] = ["rgb", "seg"]
    return complete_spec(validate_fragment(frag, "flags alias"))


# --------------------------------------------------------------------- #
# builtin presets + views (probe5-faithful; design §5)
# --------------------------------------------------------------------- #
_WHITE = [1.0, 1.0, 1.0]
_CYAN = [0.05, 0.75, 0.95]
_ORANGE = [1.0, 0.55, 0.15]
_SLATE = [0.42, 0.47, 0.55]
_HAND_SEL = {"body_prefix": "right_hand"}
_HAND_CONTACTS = {"body_prefix": "right_hand"}

register_preset(
    "hand_solid",
    claims={"appearance": ["body_prefix:right_hand", "body_prefix:right_wrist"]},
    spec={
        "appearance": [
            {"select": _HAND_SEL, "rgba": _WHITE + [1.0]},
            {"select": {"body_prefix": "right_wrist"}, "rgba": _WHITE + [1.0]},
        ]
    },
)
register_preset(
    "hand_xray",
    claims={"appearance": ["body_prefix:right_hand", "body_prefix:right_wrist"]},
    spec={
        "appearance": [
            {"select": _HAND_SEL, "rgba": _WHITE + [0.35]},
            {"select": {"body_prefix": "right_wrist"}, "rgba": _WHITE + [0.35]},
        ]
    },
)
register_preset(
    "crate_shell_occl",
    claims={"appearance": ["body_prefix:crate"]},
    spec={"appearance": [{"select": {"body_prefix": "crate"}, "rgba": _CYAN + [0.0]}]},
)
register_preset(
    "crate_shell_xray",
    claims={"appearance": ["body_prefix:crate"]},
    spec={"appearance": [{"select": {"body_prefix": "crate"}, "rgba": _CYAN + [0.45]}]},
)
register_preset(
    "ghost_rest",
    claims={"appearance": ["body_prefix:"]},
    spec={"appearance": [{"select": {"body_prefix": ""}, "rgba": _ORANGE + [0.05]}]},
)
register_preset(
    "floor_opaque",
    claims={"appearance": ["body:world"]},
    spec={"appearance": [{"select": {"body": "world"}, "rgba": _SLATE + [1.0]}]},
)
register_preset(
    "crate_cage",
    claims={"decor": ["cage_grid@crate"]},
    spec={
        "decor": [
            {
                "gen": "cage_grid",
                "target": "crate",
                "r_dense": 0.03,
                "seg": 0.004,
                "radius": 0.0005,
                "depth_tint": {"mesh_body": "right_hand", "band": 0.006},
                "contact_select": _HAND_CONTACTS,
            }
        ]
    },
)
register_preset(
    "contact_marks",
    claims={"decor": ["contact_dots@*", "contact_normals@*"]},
    spec={
        "decor": [
            {
                "gen": "contact_dots",
                "target": "*",
                "r": 0.0035,
                "force_scale": 1.2,
                "contact_select": _HAND_CONTACTS,
            },
            {
                "gen": "contact_normals",
                "target": "*",
                "len": 0.02,
                "contact_select": _HAND_CONTACTS,
            },
        ]
    },
)
register_preset(
    "contact_mjvis_simple",
    claims={"decor": ["contact_mjvis@*"]},
    spec={"decor": [{"gen": "contact_mjvis", "target": "*"}]},
)
register_preset(
    "clean_collision", claims={"aspects": ["scene"]}, spec={"scene": {"geom_class": "collision"}}
)

_CAM_ORBIT = {
    "camera": {
        "mode": "orbit",
        "target": "contact_centroid",
        "d": 0.20,
        "el": -32.0,
        "az": 0.0,
        "orbit": {"n": 24, "step_deg": 15.0, "ms": 125},
        "views": ["top", "side"],
        "contact_select": _HAND_CONTACTS,
    }
}
_CAM_CLOSEUP = {
    "camera": {
        "mode": "closeup",
        "target": "contact_centroid",
        "d": 0.16,
        "el": -30.0,
        "az": 0.0,
        "orbit": {"n": 2, "step_deg": 180.0},
        "contact_select": _HAND_CONTACTS,
    }
}
_SCENE_CLEAN = {
    "scene": {"geom_class": "collision", "shadow": False, "reflection": False, "skybox": False}
}
_OUTPUT_LIVE = {"output": {"gif": True, "gif_ms": 125, "caption": True}}

register_view(
    "palm_pose_occl",
    ("hand_solid", get_preset("hand_solid")),
    ("crate_shell_occl", get_preset("crate_shell_occl")),
    ("crate_cage", get_preset("crate_cage")),
    ("contact_marks", get_preset("contact_marks")),
    ("ghost_rest", get_preset("ghost_rest")),
    ("floor_opaque", get_preset("floor_opaque")),
    ("camera", _CAM_ORBIT),
    ("scene", _SCENE_CLEAN),
    ("output", _OUTPUT_LIVE),
)
register_view(
    "palm_pose_xray",
    ("hand_xray", get_preset("hand_xray")),
    ("crate_shell_xray", get_preset("crate_shell_xray")),
    ("crate_cage", get_preset("crate_cage")),
    ("contact_marks", get_preset("contact_marks")),
    ("ghost_rest", get_preset("ghost_rest")),
    ("floor_opaque", get_preset("floor_opaque")),
    ("camera", _CAM_CLOSEUP),
    ("scene", _SCENE_CLEAN),
    ("output", _OUTPUT_LIVE),
)
register_view("clean_collision", ("clean_collision", get_preset("clean_collision")))
register_view("visual", ("empty", {}))
# per-camera style axis (camera unification): the WebUI style selector
# and camera(style) write these as a camera's live render state
register_view("contacts", ("contact_mjvis_simple", get_preset("contact_mjvis_simple")))
register_view(
    "cage_contacts",
    ("clean_collision", get_preset("clean_collision")),
    ("crate_cage", get_preset("crate_cage")),
    ("contact_marks", get_preset("contact_marks")),
)
