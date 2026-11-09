"""The closed IR vocabularies and the mechanical element-name guard.

The constant tables below are the payload's engine-portable vocabulary;
the name guard is the anti-leak face: an IR element may never be
NAMED after an engine element, so a renderer can never re-interpret a
payload string as engine structure.
"""
from __future__ import annotations

import re

# Closed IR vocabularies.  The shape set excludes the file-referencing
# engine types (sdf/hfield): those are not payload vocabulary.  A mesh
# geom is the ONE file-referencing form: ``type="mesh"`` + a logical
# :class:`MeshDef` table reference (S2.2) -- the declaration carries the
# file, the geom carries only the name.
GEOM_TYPES = ("box", "sphere", "capsule", "cylinder", "ellipsoid")
#: the mesh-reference geom type (legal only with a declared MeshDef).
MESH_GEOM_TYPE = "mesh"
#: the composite pilot: ``ball`` joins the closed set
#: -- the robocasa composite bodies carry real ball chains (the stove
#: plug's 20-joint wire).  A ball joint has no axis (the engine
#: quaternion-dofs face); ``range`` carries ``[0, max_angle_rad]`` and
#: ``springdamper`` the mass-spring parametrization.
JOINT_TYPES = ("hinge", "slide", "ball")
#: size-vector length per geom type (the shared geometry convention).
GEOM_SIZE_LEN = {"box": 3, "sphere": 1, "capsule": 2, "cylinder": 2, "ellipsoid": 3}
#: legal contact condition dimensions (engine-independent physics facts).
CONDIMS = (1, 3, 4, 6)
#: texture projection kinds (a closed, engine-portable pair).
TEXTURE_KINDS = ("2d", "cube")
#: site marker types (the composite families carry
#: non-sphere markers -- the stove's cylinder ``burner_on``).  ``None``
#: (the field default) keeps the engine default sphere.
SITE_TYPES = ("sphere", "cylinder", "box")

# The mechanical vocabulary guard (§12.4-1): an IR element may never be
# NAMED after an engine element.  Covers the MJCF top-level sections, the
# body-tree element names and the actuator ShortNames.
_ENGINE_ELEMENT_NAMES = frozenset(
    {
        "body",
        "geom",
        "joint",
        "site",
        "freejoint",
        "frame",
        "include",
        "asset",
        "default",
        "worldbody",
        "world",
        "actuator",
        "motor",
        "position",
        "velocity",
        "intvelocity",
        "general",
        "damper",
        "muscle",
        "adhesion",
        "contact",
        "pair",
        "equality",
        "tendon",
        "sensor",
        "keyframe",
        "numeric",
        "text",
        "tuple",
        "flex",
        "skin",
        "hfield",
        "texture",
        "material",
        "mesh",
        "skybox",
    }
)

_NAME_RE = re.compile(r"[a-z][a-z0-9_]*")

#: the composite-constructor geom-name tail (P3-b): the official
#: pot handles spell their side bars ``handle0_-`` / ``handle0_+`` --
#: a snake stem plus ONE ``+``/``-`` side tail (the robosuite twin pass
#: may append ``_vis`` after the tail).  Geom names only; every other
#: element kind stays strict snake (the P2 constructor namespace
#: deliberately did not extend to element names -- this tail face is
#: the declared exception the compile-level golden pins).
_GEOM_CONSTRUCTOR_TAIL_RE = re.compile(r"[a-z][a-z0-9_]*_[+\-](_vis)?")

#: the robocasa CONSTRUCTOR namespace: the asset-declaration
#: names of the extracted dressing payloads survive the source
#: constructor's spellings verbatim -- hyphens (``..._tex-model``) and
#: Maya-style CamelCase suffixes (``..._blinn1SG``) -- because the
#: compile-level golden pins them by name.  Asset-declaration names only
#: (element names stay strict snake); engine element names stay rejected.
_CONSTRUCTOR_SEGMENT_RE = re.compile(r"[A-Za-z0-9_\-]+")


def _require_name(kind: str, name: str) -> str:
    """Validate one IR element name (the mechanical vocabulary guard)."""
    tail_ok = kind == "geom" and _GEOM_CONSTRUCTOR_TAIL_RE.fullmatch(name)
    if not isinstance(name, str) or not (tail_ok or _NAME_RE.fullmatch(name)):
        raise ValueError(f"{kind} name {name!r} must be a lowercase snake_case identifier")
    if name in _ENGINE_ELEMENT_NAMES:
        raise ValueError(
            f"{kind} name {name!r} is an engine element name -- providers "
            f"speak the neutral IR vocabulary only (design v072 §12.4-1)"
        )
    return name


def require_neutral_name(kind: str, name: str) -> str:
    """Public vocabulary-guard face.

    The scenes IR shares the SAME mechanical engine-element-name
    ban -- one guard, two vocabularies.
    """
    return _require_name(kind, name)
