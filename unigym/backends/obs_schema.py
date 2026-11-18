"""Shared observation-schema assembly for the simulator backends.

The run's :class:`~unigym.env.obs.ObservationConfig` names the sources; this
module turns the live articulation/contact/prop/camera/site declarations into
the concrete :class:`~unigym.env.obs.ObservationSchema` the registry publishes.
Torch-only (no engine import); every backend calls it after its managers are
bound.
"""

from __future__ import annotations

from collections.abc import Sequence

from unigym.env.obs import ObservationConfig, ObservationSchema, build_observation_schema

__all__ = [
    "build_schema",
    "policy_joint_names",
    "require_sites",
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


def require_sites(backend: str, declared, inventory=None) -> tuple[str, ...]:
    """Validate ``observations.site`` against a backend site inventory.

    ``inventory=None`` means the backend has no site readback face: a
    non-empty selection fails at build; an empty selection is legal.
    """
    names = tuple(str(name) for name in (declared or ()))
    if inventory is None:
        if names:
            raise NotImplementedError(
                f"{backend} backend: observations.site is not implemented "
                f"(no site readback face); selected sites: {list(names)}"
            )
        return ()
    available = set(inventory)
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            f"{backend} backend: observations.site names unknown sites {unknown}; "
            f"declared sites: {sorted(available)}"
        )
    return names


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
