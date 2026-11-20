"""Shared observation-schema assembly for the simulator backends.

The run's :class:`~unigym.env.obs.ObservationConfig` names the sources; this
module turns the live articulation/contact/prop/camera/site declarations into
the concrete :class:`~unigym.env.obs.ObservationSchema` the registry publishes.
Torch-only (no engine import); every backend calls it after its managers are
bound.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from unigym.env.obs import ObservationConfig, ObservationSchema, build_observation_schema
from unigym.sites import SiteEntry, SiteSpec

if TYPE_CHECKING:
    from unigym.props.asset_ir import AssetDefinition
    from unigym.robots.base import RobotProfile

__all__ = [
    "build_schema",
    "build_site_inventory",
    "policy_joint_names",
    "selected_camera_frames",
]


def policy_joint_names(env) -> dict[str, tuple[str, ...]]:
    """``{articulation: body-control joint order}`` (full order when absent)."""
    out: dict[str, tuple[str, ...]] = {}
    controls = getattr(env, "controls", None) or {}
    for name, art in (getattr(env, "articulations", None) or {}).items():
        body = next(
            (
                control
                for control in controls.values()
                if getattr(control, "articulation", None) == name
                and getattr(control, "role", None) == "body"
                and not getattr(control, "prop", False)
            ),
            None,
        )
        out[name] = tuple(body.joint_names) if body is not None else tuple(art.joint_names)
    return out


def _camera_frames(camera_manager, names: Sequence[str]) -> dict[str, dict]:
    frames: dict[str, dict] = {}
    if camera_manager is None:
        return frames
    for name in names:
        prefix = f"cam/{name}/"
        table: dict[str, tuple[tuple[int, ...], str]] = {}
        for key in sorted(camera_manager.obs_keys()):
            if not key.startswith(prefix):
                continue
            frame = camera_manager.frame_request(key)
            if frame is None:
                continue
            dtype = "u8" if str(frame.dtype).endswith("uint8") else "f32"
            table[key] = (tuple(int(d) for d in frame.shape), dtype)
        if table:
            frames[name] = table
    return frames


def selected_camera_frames(config: ObservationConfig, camera_manager) -> dict[str, dict]:
    """Frame table for the cameras selected by ``observations.camera``."""
    return _camera_frames(camera_manager, config.camera)


def _contact_table(contact_manager) -> dict[str, tuple[tuple[int, ...] | None, str]]:
    if contact_manager is None:
        return {}
    table: dict[str, tuple[tuple[int, ...] | None, str]] = {}
    for spec in contact_manager.specs:
        shape = contact_manager.group_shape(spec.name)
        dtype = "bool" if spec.format == "contact" else "f32"
        table[spec.name] = (shape, dtype)
    return table


def _object_table(prop_manager) -> dict[str, tuple[str, ...]]:
    if prop_manager is None:
        return {}
    table: dict[str, tuple[str, ...]] = {}
    for handler in getattr(prop_manager, "_handlers", ()):
        joints = getattr(handler, "joints", None)
        table[handler.spec.name] = tuple(joints or ())
    return table


def build_site_inventory(
    profiles: Mapping[str, RobotProfile],
    asset_definitions: Mapping[str, AssetDefinition],
) -> Mapping[str, SiteEntry]:
    """Build the asset-level site inventory ``{name: SiteEntry}``.

    Profile sites are articulation-owned; generated prop payload sites are
    prop-owned.  Site names are run-wide, so a name declared by two owners is
    a build error rather than an arbitrary winner.  The returned mapping is
    read-only.
    """
    if not isinstance(profiles, Mapping):
        raise ValueError(f"site inventory profiles must be a mapping, got {profiles!r}")
    if not isinstance(asset_definitions, Mapping):
        raise ValueError(
            f"site inventory asset_definitions must be a mapping, got {asset_definitions!r}"
        )
    inventory: dict[str, SiteEntry] = {}

    def add(owner: str, kind: Literal["articulation", "prop"], spec: SiteSpec) -> None:
        entry = SiteEntry(owner=owner, kind=kind, spec=spec)
        previous = inventory.get(entry.spec.name)
        if previous is not None:
            raise ValueError(
                f"duplicate site name {entry.spec.name!r} declared by "
                f"{previous.kind} {previous.owner!r} and {entry.kind} {entry.owner!r}"
            )
        inventory[entry.spec.name] = entry

    for owner, profile in profiles.items():
        if not isinstance(owner, str) or not owner:
            raise ValueError(
                f"site inventory profile owner must be a non-empty string, got {owner!r}"
            )
        for spec in (getattr(profile, "sites", None) or {}).values():
            add(owner, "articulation", spec)

    for owner, asset in asset_definitions.items():
        if not isinstance(owner, str) or not owner:
            raise ValueError(f"site inventory prop owner must be a non-empty string, got {owner!r}")
        if asset.name != owner:
            raise ValueError(
                f"site inventory prop {owner!r} carries asset definition "
                f"{asset.name!r}; the owner and payload identity must match"
            )
        for spec in asset.site_specs().values():
            add(owner, "prop", spec)

    return MappingProxyType(inventory)


def build_schema(
    config: ObservationConfig,
    *,
    env,
    camera_manager,
    contact_manager,
    prop_manager,
    sites: Sequence[str],
) -> ObservationSchema:
    """Build the backend's observation schema from the live declarations."""
    articulations = getattr(env, "articulations", None) or {}
    full = {name: tuple(art.joint_names) for name, art in articulations.items()}
    return build_observation_schema(
        config,
        full_joint_names=full,
        policy_joint_names=policy_joint_names(env),
        contacts=_contact_table(contact_manager),
        objects=_object_table(prop_manager),
        cameras=_camera_frames(camera_manager, config.camera),
        sites=sites,
    )
