"""The declaration tables of a generated payload (textures, meshes, materials).

Declaration rows ride beside the body tree: the renderer adds them to
the child spec BEFORE the bodies, so the attach pass namespaces them
(and every logical reference to them) together with the tree.  A
``file`` field is the ONE path-like string in the IR -- opaque data
here (no IO, no resolution); the renderer expands ``${TOKEN}``
placeholders at child-spec build time.
"""
from __future__ import annotations

from dataclasses import dataclass

from ._elements import MatRef
from ._guards import (
    _require_bool,
    _require_finite,
    _require_nonneg,
    _require_quat,
    _require_rgba,
    _require_vec,
)
from ._vocab import (
    TEXTURE_KINDS,
    _CONSTRUCTOR_SEGMENT_RE,
    _ENGINE_ELEMENT_NAMES,
    _require_name,
)


@dataclass(frozen=True)
class TextureDef:
    """One texture declaration of a generated payload.

    ``file`` is an opaque path string -- the ONE path-like field in the
    IR.  It is never touched here (no IO, no resolution): the renderer
    expands ``${TOKEN}`` placeholders at child-spec build time and hands
    it to the engine's loader.  ``kind`` is the closed projection
    vocabulary (:data:`TEXTURE_KINDS`).

    The name gate is the CONSTRUCTOR NAMESPACE face, not the element
    guard: a texture's compiled identity is the source constructor's
    texture name (robocasa spells them segment-shaped -- ``flat/gray``,
    ``marble/marble_2_2d``; the v075 P1 verbatim bake must reproduce
    them), so each ``/`` segment follows the constructor segment rule
    instead of the whole string.  Payload-local texture declarations
    (the F1-F3 appliance generators) use plain lowercase snake_case
    names, which the same rule accepts.
    """

    name: str
    kind: str
    file: str

    def __post_init__(self):
        """Validate the name, the closed kind set and the file string."""
        segments = str(self.name).split("/")
        if not self.name or any(
            not _CONSTRUCTOR_SEGMENT_RE.fullmatch(segment) for segment in segments
        ):
            raise ValueError(
                f"texture name {self.name!r} must be constructor-namespace "
                f"segments joined by '/' (the robocasa texture namespace)"
            )
        object.__setattr__(self, "name", self.name)
        if self.kind not in TEXTURE_KINDS:
            raise ValueError(
                f"texture {self.name!r}: kind {self.kind!r} must be one of " f"{TEXTURE_KINDS}"
            )
        if not isinstance(self.file, str) or not self.file:
            raise ValueError(f"texture {self.name!r}: file must be a non-empty string")


@dataclass(frozen=True)
class MeshDef:
    """One mesh declaration of a generated payload (S2.2; TextureDef form).

    ``file`` is an opaque path string like :attr:`TextureDef.file` (no
    IO, no resolution here; the renderer expands ``${TOKEN}``
    placeholders at child-spec build time and hands it to the engine's
    OBJ loader).  ``scale`` is the optional per-axis mesh scale (the
    engine bakes it into the compiled vertices); a silent scale keeps
    the engine default.  ``refquat`` is the optional mesh orientation
    fact (§13 step 3, the robocasa data-object meshes): a non-zero
    (w, x, y, z) quaternion the engine applies to the loaded vertices
    BEFORE any geom placement -- per-mesh data, hence on the
    declaration, never on the referencing geom.
    """

    name: str
    file: str
    scale: tuple | None = None
    refquat: tuple | None = None

    def __post_init__(self):
        """Validate the name, the file string and the scale vector."""
        object.__setattr__(self, "name", _require_name("mesh", self.name))
        if not isinstance(self.file, str) or not self.file:
            raise ValueError(f"mesh {self.name!r}: file must be a non-empty string")
        if self.scale is not None:
            object.__setattr__(
                self, "scale", _require_vec(f"mesh {self.name!r}", "scale", self.scale, 3)
            )
        if self.refquat is not None:
            object.__setattr__(self, "refquat", _require_quat(f"mesh {self.name!r}", self.refquat))


@dataclass(frozen=True)
class MaterialDef:
    """One material declaration of a generated payload.

    ``texture`` is an OPTIONAL logical reference to a
    :class:`TextureDef` name of the SAME payload (the renderer resolves
    it inside the child spec, where the attach pass namespaces it like
    every element name); ``None`` is the texture-less color-only
    material (§13 step 3: 378 of the robocasa data-object materials
    carry rgba only).  Appearance attributes are optional; a silent one
    keeps the engine default.  ``specular`` (v072 §13 step 5) rides the
    same optional-appearance face -- the robosuite CustomMaterial metal
    family authors it beside shininess.
    """

    name: str
    texture: MatRef | None = None
    rgba: tuple | None = None
    shininess: float | None = None
    reflectance: float | None = None
    texuniform: bool | None = None
    texrepeat: tuple | None = None
    specular: float | None = None

    def __post_init__(self):
        """Validate the name/reference and every present attribute."""
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name in _ENGINE_ELEMENT_NAMES
            or not _CONSTRUCTOR_SEGMENT_RE.fullmatch(self.name)
        ):
            raise ValueError(
                f"material name {self.name!r} must be a constructor-"
                f"namespace identifier and never an engine element name"
            )
        if self.texture is not None and not isinstance(self.texture, MatRef):
            raise ValueError(f"material {self.name!r}: texture must be a MatRef")
        owner = f"material {self.name!r}"
        if self.rgba is not None:
            object.__setattr__(self, "rgba", _require_rgba(owner, self.rgba))
        for f in ("reflectance", "specular"):
            if getattr(self, f) is not None:
                object.__setattr__(self, f, _require_nonneg(owner, f, getattr(self, f)))
        if self.shininess is not None:
            # the MJCF shininess face: any finite float -- the upstream
            # lightwheel corpus authors negative specular exponents
            # (jar003's -1023 is the attested extreme) and MuJoCo
            # accepts them
            object.__setattr__(
                self, "shininess", _require_finite(owner, "shininess", self.shininess)
            )
        if self.texuniform is not None:
            object.__setattr__(
                self, "texuniform", _require_bool(owner, "texuniform", self.texuniform)
            )
        if self.texrepeat is not None:
            object.__setattr__(
                self, "texrepeat", _require_vec(owner, "texrepeat", self.texrepeat, 2)
            )
