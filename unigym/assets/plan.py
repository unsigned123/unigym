"""Boundary-resolved asset plan for run, prop and scene construction.

Asset references are resolved exactly once, at the composition/domain
boundary, into an :class:`AssetPlan`.  Backend render code receives only
absolute paths and resolved MJCF text -- it never resolves a placeholder,
a ``pack:`` reference or a legacy asset token.

The plan is a thin adapter over :class:`unigym.packs.runtime.AssetContext`:
:meth:`AssetPlan.resolve` expands the remaining known path placeholders and
then delegates the final path decision to the context.  The context (and the
:class:`~unigym.packs.runtime.PackRuntime` it owns) stays the sole resolver.
There is no module-level or ambient runtime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from unigym.core.paths import expand_placeholders

if TYPE_CHECKING:
    from unigym.packs.runtime import AssetContext
    from unigym.props.asset_ir import AssetDefinition

_FILE_ATTR_RE = re.compile(r'(file=")([^"]*)(")')
"""MJCF ``file=`` attributes carry the asset leaves of a parsed document."""


@dataclass(frozen=True)
class AssetPlan:
    """One asset resolution boundary for a run or a composed scene.

    ``context`` owns path resolution; the plan exposes the narrow operations
    backend-adjacent code needs, so no backend imports the legacy asset
    anchor layer.
    """

    context: AssetContext

    def resolve(self, value: str) -> str:
        """Resolve one asset reference to an absolute path string.

        Known path placeholders are expanded first; the resulting reference
        then delegates to :meth:`AssetContext.resolve`.  A bare relative
        reference or a ``pack:`` reference without an owning runtime fails
        there with the teaching error.
        """
        return self.context.resolve(expand_placeholders(value, self.context))

    def resolve_definition(self, asset: AssetDefinition) -> AssetDefinition:
        """Return a copy of *asset* with every mesh/texture file absolute.

        Only the ``file`` fields change; every other declaration field is
        preserved verbatim.  The copy is frozen, matching the input IR.
        """
        meshes = tuple(replace(mesh, file=self.resolve(mesh.file)) for mesh in asset.meshes)
        textures = tuple(
            replace(texture, file=self.resolve(texture.file)) for texture in asset.textures
        )
        return replace(asset, meshes=meshes, textures=textures)

    def resolve_document(self, text: str) -> str:
        """Resolve path placeholders and ``pack:`` leaves of one MJCF document.

        Placeholders expand document-wide.  A ``file=`` attribute carrying a
        ``pack:`` reference resolves through the plan; every other attribute
        value is left verbatim (including a file-relative path, whose
        resolution belongs to the engine's document loader).
        """
        expanded = expand_placeholders(text, self.context)
        return _FILE_ATTR_RE.sub(lambda match: self._resolve_file_attr(match), expanded)

    def _resolve_file_attr(self, match: re.Match[str]) -> str:
        """Resolve one ``file="..."`` attribute of a document."""
        value = match.group(2)
        if value.startswith("pack:"):
            value = self.resolve(value)
        return match.group(1) + value + match.group(3)


__all__ = ["AssetPlan"]
