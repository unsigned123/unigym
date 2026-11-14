"""USD and URDF inspection for the asset-scene compose/attach steps.

A file-backed asset declared ``static_rigid`` must actually be static: a USD
that authors ``UsdPhysics.RigidBodyAPI`` or ``UsdPhysics.ArticulationRootAPI``
declares simulation/articulation semantics, and silently freezing it would
change the declaration.  This module owns the single recursive USD check.

:func:`urdf_joint_names` is the Kit-free URDF guard: a URDF that declares any
``<joint>`` converts to an articulation, which has no passive
``RigidObjectCfg`` form.  It uses only :mod:`xml.etree.ElementTree`, so it runs
in ``compose`` without OpenUSD or Kit.

OpenUSD (``pxr``) is a Kit dependency and is imported lazily inside
:func:`inspect_usd_authoring`; importing this module never imports ``pxr``.
Needing USD inspection without OpenUSD, or failing to open the stage, is a
teaching error -- never an assumption that the asset is static.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass

from unigym._optional import MissingExtraError

__all__ = ["UsdAuthoring", "inspect_usd_authoring", "urdf_joint_names"]


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


def _local_tag(tag: object) -> str:
    """Return an XML element tag without its namespace prefix."""
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1]


def urdf_joint_names(path: str) -> tuple[str, ...]:
    """Return the ``<joint>`` names declared in one URDF document.

    This is the Kit-free guard for the dynamic-URDF route.  Only the standard
    library is used, so it runs in ``compose`` without OpenUSD or Kit; names
    are returned in document order and namespace prefixes are ignored.

    Args:
        path: Absolute path to the URDF file.

    Returns:
        The declared joint names in document order; empty for a jointless
        URDF.  A joint element without a name is reported as ``"<unnamed>"``.

    Raises:
        ValueError: The file cannot be read or is not well-formed XML; the
            message names the asset and never assumes the URDF is jointless.
    """
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        raise ValueError(
            f"URDF asset {path!r}: cannot read the file ({exc}); a jointless "
            f"URDF cannot be verified -- provide a readable URDF"
        ) from exc
    try:
        root = ElementTree.fromstring(data)
    except ElementTree.ParseError as exc:
        raise ValueError(
            f"URDF asset {path!r}: malformed URDF XML ({exc}); a jointless "
            f"URDF cannot be verified -- provide a well-formed URDF"
        ) from exc
    names: list[str] = []
    for element in root.iter():
        if _local_tag(element.tag) == "joint":
            name = element.get("name")
            names.append(name if name else "<unnamed>")
    return tuple(names)


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
