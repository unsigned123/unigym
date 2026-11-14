"""USD authored-static inspection for the asset-scene attach step.

A file-backed asset declared ``static_rigid`` must actually be static: a USD
that authors ``UsdPhysics.RigidBodyAPI`` or ``UsdPhysics.ArticulationRootAPI``
declares simulation/articulation semantics, and silently freezing it would
change the declaration.  This module owns the single recursive check.

OpenUSD (``pxr``) is a Kit dependency and is imported lazily inside
:func:`inspect_usd_authoring`; importing this module never imports ``pxr``.
Needing USD inspection without OpenUSD, or failing to open the stage, is a
teaching error -- never an assumption that the asset is static.
"""

from __future__ import annotations

from dataclasses import dataclass

from unigym._optional import MissingExtraError

__all__ = ["UsdAuthoring", "inspect_usd_authoring"]


@dataclass(frozen=True)
class UsdAuthoring:
    """The simulation/articulation schemas authored in one USD stage.

    Attributes:
        rigid_body_paths: absolute prim paths carrying ``RigidBodyAPI``.
        articulation_root_paths: absolute prim paths carrying
            ``ArticulationRootAPI``.
    """

    rigid_body_paths: tuple[str, ...] = ()
    articulation_root_paths: tuple[str, ...] = ()


def _usd_modules():
    """Import the OpenUSD modules at the function boundary."""
    try:
        from pxr import Usd, UsdPhysics
    except ImportError as exc:
        raise MissingExtraError(
            "pxr", "usd", "USD asset inspection for static file-backed assets"
        ) from exc
    return Usd, UsdPhysics


def inspect_usd_authoring(path: str) -> UsdAuthoring:
    """Walk one USD stage and report authored rigid-body/articulation roots.

    Args:
        path: Absolute path to the USD file to inspect.

    Returns:
        The authored schema paths, sorted for deterministic reporting.

    Raises:
        MissingExtraError: OpenUSD is not available on this host.
        ValueError: The stage cannot be opened or traversed; the message
            names the asset and the failure, and never assumes static.
    """
    Usd, UsdPhysics = _usd_modules()
    try:
        stage = Usd.Stage.Open(path)
    except Exception as exc:  # noqa: BLE001 — OpenUSD raises its own hierarchy
        raise ValueError(
            f"USD asset {path!r}: cannot open the stage ({exc}); a static "
            f"declaration cannot be verified -- provide a readable USD"
        ) from exc
    if stage is None:
        raise ValueError(
            f"USD asset {path!r}: cannot open the stage; a static declaration "
            f"cannot be verified -- provide a readable USD"
        )
    rigid: list[str] = []
    articulation: list[str] = []
    try:
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid.append(prim_path)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation.append(prim_path)
    except Exception as exc:  # noqa: BLE001 — OpenUSD traversal errors are stage errors
        raise ValueError(
            f"USD asset {path!r}: cannot traverse the stage ({exc}); a static "
            f"declaration cannot be verified -- provide a readable USD"
        ) from exc
    return UsdAuthoring(tuple(sorted(rigid)), tuple(sorted(articulation)))
