"""Pack-reference -> neutral robot profile loading.

A facade robot is always acquired from an explicit :class:`~unigym.packs.packset.PackSet`.
The reference must name a pack of that set with the same manifest version; a
foreign reference is refused before any activation or file read.

Data-backed references load the declared ``unigym.robot/1`` document through
``pack_set.resource_path(ref)``.  Entrypoint-backed references go through the
existing robot factory registry after one ``runtime.activate(ref.pack)`` call.
That activation is the explicit, process-wide pack-registry boundary; the
facade does not perform ambient discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from unigym._typing import JsonValue

from .errors import ForeignHandleError, SimulationConfigError

if TYPE_CHECKING:
    from unigym.packs.packset import PackSet, RobotRef
    from unigym.robots.base import RobotProfile

__all__ = ["load_profile_for_ref"]


def _require_membership(pack_set: PackSet, ref: RobotRef) -> None:
    """Reject a ref whose pack is absent or carries a different manifest version."""
    packs = {pack.name: pack for pack in pack_set.packs()}
    pack = packs.get(ref.pack)
    if pack is None:
        raise ForeignHandleError(
            f"robot ref {ref.id!r} names pack {ref.pack!r}, which is not in this "
            f"pack set; available packs: {sorted(packs)}"
        )
    if pack.version != ref.pack_version:
        raise ForeignHandleError(
            f"robot ref {ref.id!r} names pack {ref.pack!r} version "
            f"{ref.pack_version!r}, but this pack set declares {pack.version!r}"
        )


def load_profile_for_ref(pack_set: PackSet, ref: RobotRef) -> RobotProfile:
    """Load the neutral profile for *ref* through the explicit *pack_set*.

    Raises:
        ForeignHandleError: The ref's pack is absent from the set or its
            version does not match the set's manifest.
        SimulationConfigError: The loaded profile name does not equal the ref id.
        PackError: The declared resource is missing or malformed.
        RobotDataError: The robot document is malformed.
    """
    _require_membership(pack_set, ref)
    runtime = pack_set.to_runtime()
    runtime.activate(ref.pack)

    if ref.is_data:
        from unigym.robots import load_robot_profile

        profile = load_robot_profile(pack_set.resource_path(ref))
    else:
        from unigym.robots import build_robot_profile

        section: dict[str, JsonValue] = {"name": ref.id}
        try:
            profile = build_robot_profile(section)
        except ValueError as err:
            raise SimulationConfigError(
                f"robot ref {ref.id!r} in pack {ref.pack!r} could not build a " f"profile: {err}"
            ) from err

    if profile.name != ref.id:
        raise SimulationConfigError(
            f"robot ref {ref.id!r} in pack {ref.pack!r} loaded profile "
            f"{profile.name!r}; the profile name must equal the ref id"
        )
    return profile
