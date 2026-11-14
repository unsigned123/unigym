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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from unigym.assets.asset_ref import AssetRef, AssetRefError, select_asset_format
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


def resolve_asset_instance(
    asset: AssetRef,
    asset_plan: AssetPlan,
    formats: Sequence[str],
) -> tuple[str, str]:
    """Resolve one file-backed asset instance to ``(format, absolute path)``.

    The format follows the backend's declared preference order, never the
    ``AssetRef`` declaration order.  The selected reference resolves through
    *asset_plan* and must name an existing file: there is no conversion and
    no fallback to another declared format.

    Args:
        asset: The effective neutral asset reference.
        asset_plan: The boundary that resolves ``pack:``/absolute references.
        formats: The backend's declared format preference order.

    Returns:
        ``(format, absolute path)``, with *format* the selected member of
        :data:`~unigym.assets.asset_ref.ASSET_FORMATS`.

    Raises:
        AssetRefError: The preference list is invalid, no format is declared,
            the selected format's reference is empty or whitespace-only, or the
            resolved path is relative; there is no CWD-relative fallback.
        FileNotFoundError: The resolved path is not an existing file; the
            message names the declared reference, the resolved path and the
            selected format.
    """
    fmt = select_asset_format(asset, formats)
    raw = getattr(asset, fmt)
    if not isinstance(raw, str) or not raw.strip():
        raise AssetRefError(
            f"asset reference declares selected format {fmt!r} with an empty or "
            f"whitespace-only path; declare a real {fmt} path"
        )
    path = asset_plan.resolve(raw)
    if not Path(path).is_absolute():
        raise AssetRefError(
            f"asset reference {raw!r} ({fmt}) resolved to the relative path "
            f"{path!r}; the boundary must resolve to an absolute path -- there "
            f"is no CWD-relative fallback"
        )
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"asset reference {raw!r} ({fmt}) resolved to {path!r}, which is not an "
            f"existing file; file-backed assets must name an existing file "
            f"(no conversion, no fallback to another declared format)"
        )
    return fmt, path


__all__ = ["AssetPlan", "resolve_asset_instance"]
