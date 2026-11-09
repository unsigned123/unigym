"""Neutral asset-definition IR for provider-generated prop payloads.

Pure data + validation: no IO, no engine import, no path resolution.  A
provider speaks ONLY this vocabulary -- the frozen
dataclasses plus the closed type/name guards make it mechanically
impossible to spell an engine ELEMENT name (``body`` / ``geom`` /
``freejoint`` / actuator ShortNames / ...): there is no free-form field a
payload could carry that a renderer would re-interpret as engine
structure (ruling §12.4-1).  The renderer
(:mod:`unigym.backends.shared.mjfamily.scene.asset_render`) is the SOLE interpreter of
the neutrally-named ``solver_hints`` block; other backends ignore + warn
on a payload.

Neutrality contract (§12.6): IR neutrality is NOT physics equality -- the
same IR may legitimately compile to different inertia/contact behaviour
in different engines.  The golden pins ``this backend's product == this
backend's reference``; it does not promise cross-backend value equality.

S2.1 additions (cabinet pilot): joint anchor ``pos``; per-geom
``density`` / ``rgba`` / ``group`` / ``collidable``; per-site ``size`` /
``rgba``; and the texture/material declaration tables
(:class:`TextureDef` / :class:`MaterialDef`).  A texture ``file`` is the
ONE path-like string in the IR: it is opaque data (no IO, no resolution
here) that may carry ``${UNIGYM_REPO}``-style placeholders, expanded by
the renderer at child-spec build time; material/texture references stay
logical names (MatRef) that the attach pass namespaces like every other
element name.

S2.2 additions (drawer/microwave rollout): the mesh declaration table
(:class:`MeshDef`, the TextureDef pattern for engine meshes) and the
mesh-reference geom form.  A ``type="mesh"`` geom carries a logical
mesh NAME (:attr:`Geom.mesh`) resolved inside the payload's declared
:class:`MeshDef` table -- the file/scale data lives in the declaration
(never on the geom), the reference stays a name the attach pass
namespaces.  Geoms additionally carry per-geom
:class:`ContactHints` (:attr:`Geom.hints`) beside the payload-wide
``solver_hints`` block (the microwave visual-mesh geoms' authored
friction/solver parameters); both are NEUTRAL names interpreted only by
the renderer (ruling §12.4-1).

The package is a stable facade: the vocabulary guard lives in
``_vocab``, the scalar/vector guards in ``_guards``, the frozen
tree-element dataclasses in ``_elements``, the declaration tables in
``_declarations``, and the aggregate + its tree-level helpers in
``definition``.  Every historical ``unigym.props.asset_ir`` name is
re-exported here unchanged.

Naming note: the unprefixed names are the historical spelling;
new code should use the ``Asset*`` aliases to avoid collision with
``unigym.model`` entities.
"""
from __future__ import annotations

from ._declarations import MaterialDef as MaterialDef
from ._declarations import MeshDef as MeshDef
from ._declarations import TextureDef as TextureDef
from ._elements import Body as Body
from ._elements import ContactHints as ContactHints
from ._elements import Geom as Geom
from ._elements import Inertial as Inertial
from ._elements import Joint as Joint
from ._elements import JointHints as JointHints
from ._elements import MatRef as MatRef
from ._elements import Site as Site
from ._elements import SolverHints as SolverHints
from ._elements import iter_bodies as iter_bodies
from ._guards import _require_bool as _require_bool
from ._guards import _require_nonneg as _require_nonneg
from ._guards import _require_quat as _require_quat
from ._guards import _require_range as _require_range
from ._guards import _require_rgba as _require_rgba
from ._guards import _require_unique as _require_unique
from ._guards import _require_vec as _require_vec
from ._vocab import CONDIMS as CONDIMS
from ._vocab import GEOM_SIZE_LEN as GEOM_SIZE_LEN
from ._vocab import GEOM_TYPES as GEOM_TYPES
from ._vocab import JOINT_TYPES as JOINT_TYPES
from ._vocab import MESH_GEOM_TYPE as MESH_GEOM_TYPE
from ._vocab import SITE_TYPES as SITE_TYPES
from ._vocab import TEXTURE_KINDS as TEXTURE_KINDS
from ._vocab import _CONSTRUCTOR_SEGMENT_RE as _CONSTRUCTOR_SEGMENT_RE
from ._vocab import _ENGINE_ELEMENT_NAMES as _ENGINE_ELEMENT_NAMES
from ._vocab import _GEOM_CONSTRUCTOR_TAIL_RE as _GEOM_CONSTRUCTOR_TAIL_RE
from ._vocab import _NAME_RE as _NAME_RE
from ._vocab import _require_name as _require_name
from ._vocab import require_neutral_name as require_neutral_name
from .definition import AssetDefinition as AssetDefinition
from .definition import asset_roots as asset_roots
from .definition import iter_asset_bodies as iter_asset_bodies
from .definition import joint_names as joint_names
from .definition import validate_asset_definition as validate_asset_definition

# Unambiguous aliases: the unprefixed names above are the
# historical spelling; these never shadow the same-named unigym.model
# entities.
from ._declarations import MaterialDef as AssetMaterialDef  # noqa: F401
from ._declarations import MeshDef as AssetMeshDef  # noqa: F401
from ._declarations import TextureDef as AssetTextureDef  # noqa: F401
from ._elements import Body as AssetBody  # noqa: F401
from ._elements import ContactHints as AssetContactHints  # noqa: F401
from ._elements import Geom as AssetGeom  # noqa: F401
from ._elements import Inertial as AssetInertial  # noqa: F401
from ._elements import Joint as AssetJoint  # noqa: F401
from ._elements import JointHints as AssetJointHints  # noqa: F401
from ._elements import MatRef as AssetMatRef  # noqa: F401
from ._elements import Site as AssetSite  # noqa: F401
from ._elements import SolverHints as AssetSolverHints  # noqa: F401
