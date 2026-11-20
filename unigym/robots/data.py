"""Load external ``unigym.robot/1`` robot profile documents.

The document is authoring data only: joint sets, standby pose, canonical base
and foot facts, per-format body bindings, declared assets (with optional
per-format backend restrictions and opaque options), extra action sets and an
optional full joint order.  Loading builds a backend-neutral
:class:`~unigym.robots.base.RobotProfile` and never imports a backend, an
engine or pack Python.

This module owns the schema gate.  Only top-level unknown keys are rejected;
``body_bindings`` values are per-format native names, not top-level fields.
``assets`` accepts the same ``{mjcf|urdf|usd: <ref>}`` mapping as
:class:`~unigym.assets.asset_ref.AssetRef`; a pack reference stays an
unresolved string until the composition boundary resolves it.  The optional
``sites`` table declares asset-level site poses (``{name: {body, pos,
quat_wxyz}}``); the mapping key is the run-wide site observation name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, cast

import yaml

from unigym._typing import JsonValue
from unigym.assets.asset_ref import AssetRef, AssetRefError
from unigym.sites import SiteSpec

from .base import (
    BodyBinding,
    BodyBindingError,
    CanonicalBodyId,
    RobotProfile,
)

if TYPE_CHECKING:
    from unigym.actuators.ir import ActuationGroup

SCHEMA_ID = "unigym.robot/1"
"""The one robot document schema this loader accepts."""

_REQUIRED_KEYS = ("schema", "name", "joint_sets", "default_joint_pos")
_OPTIONAL_KEYS = (
    "base_body",
    "base_height",
    "foot_bodies",
    "body_sets",
    "body_bindings",
    "assets",
    "asset_backends",
    "asset_options",
    "extra_action_sets",
    "full_joint_order",
    "actuation",
    "sites",
)
_TOP_LEVEL_KEYS = frozenset((*_REQUIRED_KEYS, *_OPTIONAL_KEYS))


class RobotDataError(ValueError):
    """A robot profile document is malformed (file, field and fix are named)."""


def _error(path: Path, field: str, detail: str) -> RobotDataError:
    """Build a fail-closed teaching error naming file, field and problem."""
    return RobotDataError(f"robot profile {path}: field {field!r}: {detail}")


def _body_bindings(path: Path, value: object) -> dict[CanonicalBodyId, BodyBinding]:
    """Build the canonical-id -> :class:`BodyBinding` map from raw data."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _error(
            path,
            "body_bindings",
            f"expected a mapping of canonical id -> format map, got {type(value).__name__}",
        )
    bindings: dict[CanonicalBodyId, BodyBinding] = {}
    for canonical, names in value.items():
        if not isinstance(canonical, str):
            raise _error(
                path,
                "body_bindings",
                f"canonical id must be a string, got {canonical!r}",
            )
        if not isinstance(names, Mapping):
            raise _error(
                path,
                f"body_bindings.{canonical}",
                f"expected a mapping of format -> native name, got {type(names).__name__}",
            )
        try:
            # BodyBinding validates the format/native-name pairs at runtime.
            bindings[CanonicalBodyId(canonical)] = BodyBinding(
                CanonicalBodyId(canonical), cast(Mapping[str, str], names)
            )
        except BodyBindingError as err:
            raise _error(path, f"body_bindings.{canonical}", str(err)) from err
    return bindings


def _assets(path: Path, value: object) -> AssetRef | None:
    """Build the optional :class:`AssetRef`; ``None`` stays ``None``."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise _error(
            path,
            "assets",
            f"expected a mapping {{mjcf|urdf|usd: <ref>}}, got {type(value).__name__}",
        )
    try:
        # AssetRef.from_mapping validates the keys and values at runtime.
        return AssetRef.from_mapping(
            cast(Mapping[str, JsonValue], value), f"{path}: field 'assets'"
        )
    except AssetRefError as err:
        raise _error(path, "assets", str(err)) from err


def _asset_backends(path: Path, value: object) -> Mapping[str, tuple[str, ...]]:
    """Validate the optional ``asset_backends`` top-level shape."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _error(
            path,
            "asset_backends",
            f"expected a mapping of format -> backend names, got {type(value).__name__}",
        )
    return cast(Mapping[str, tuple[str, ...]], value)


def _asset_options(path: Path, value: object) -> Mapping[str, Mapping[str, JsonValue]]:
    """Validate the optional ``asset_options`` top-level shape."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _error(
            path,
            "asset_options",
            f"expected a mapping of format -> option mapping, got {type(value).__name__}",
        )
    return cast(Mapping[str, Mapping[str, JsonValue]], value)


def _actuation(path: Path, value: object) -> tuple[ActuationGroup, ...]:
    """Normalize the optional ``actuation`` group list."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise _error(
            path,
            "actuation",
            f"expected a list of ActuationGroup mappings, got {type(value).__name__}",
        )
    from unigym.actuators.ir import ActuationError, normalize_actuation_groups

    entries = list(value)
    if not all(isinstance(entry, Mapping) for entry in entries):
        raise _error(path, "actuation", "every entry must be a mapping")
    try:
        return normalize_actuation_groups(entries)
    except ActuationError as err:
        raise _error(path, "actuation", str(err)) from err


def _site_numbers(
    path: Path, name: str, field: str, value: object, length: int
) -> tuple[float, ...]:
    """Validate one site pose vector at the loader boundary."""
    if not isinstance(value, (list, tuple)):
        raise _error(
            path,
            f"sites.{name}",
            f"field {field!r} must be {length} finite numbers, got {value!r}",
        )
    # The runtime check above establishes a sequence; the cast only replaces
    # the builtin container's unknown item type at this YAML boundary.
    entries = cast(Sequence[object], value)
    if len(entries) != length:
        raise _error(
            path,
            f"sites.{name}",
            f"field {field!r} must be {length} finite numbers, got {value!r}",
        )
    numbers: list[float] = []
    for entry in entries:
        if (
            isinstance(entry, bool)
            or not isinstance(entry, (int, float))
            or not math.isfinite(entry)
        ):
            raise _error(
                path,
                f"sites.{name}",
                f"field {field!r} must be {length} finite numbers, got {value!r}",
            )
        numbers.append(float(entry))
    return tuple(numbers)


def _sites(path: Path, value: object) -> dict[str, SiteSpec]:
    """Build the optional ``sites`` table (key = SiteSpec name)."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _error(
            path,
            "sites",
            f"expected a mapping of site name -> pose mapping, got {type(value).__name__}",
        )
    sites: dict[str, SiteSpec] = {}
    # The isinstance check above establishes a mapping at the YAML boundary;
    # the cast only replaces the raw container's unknown key/value types.
    raw_sites = cast(Mapping[object, object], value)
    for name, entry in raw_sites.items():
        if not isinstance(name, str) or not name:
            raise _error(path, "sites", f"site names must be non-empty strings, got {name!r}")
        if not isinstance(entry, Mapping):
            raise _error(
                path,
                f"sites.{name}",
                f"expected a mapping {{body, pos, quat_wxyz}}, got {type(entry).__name__}",
            )
        # Same runtime-validated mapping boundary as above.
        raw_entry = cast(Mapping[str, object], entry)
        unknown = sorted(set(raw_entry) - {"body", "pos", "quat_wxyz"})
        if unknown:
            raise _error(
                path,
                f"sites.{name}",
                f"unknown key(s) {unknown}; allowed keys are ['body', 'pos', 'quat_wxyz']",
            )
        body = raw_entry.get("body")
        if not isinstance(body, str) or not body:
            raise _error(
                path,
                f"sites.{name}",
                f"field 'body' must be a non-empty string, got {body!r}",
            )
        pos = _site_numbers(path, name, "pos", raw_entry.get("pos", (0.0, 0.0, 0.0)), 3)
        quat = _site_numbers(
            path, name, "quat_wxyz", raw_entry.get("quat_wxyz", (1.0, 0.0, 0.0, 0.0)), 4
        )
        try:
            sites[name] = SiteSpec(
                name=name,
                body=body,
                pos=(pos[0], pos[1], pos[2]),
                quat_wxyz=(quat[0], quat[1], quat[2], quat[3]),
            )
        except (TypeError, ValueError) as err:
            raise _error(path, f"sites.{name}", str(err)) from err
    return sites


def load_robot_profile(path: str | Path) -> RobotProfile:
    """Load one ``unigym.robot/1`` document into a :class:`RobotProfile`.

    Args:
        path: The robot YAML document path.

    Returns:
        The validated neutral profile.  ``assets`` is an :class:`AssetRef` or
        ``None``; pack references inside it remain unresolved strings.

    Raises:
        RobotDataError: The file is unreadable or malformed, a required field
            is missing, an unknown top-level key is present, or the document
            fails :class:`RobotProfile` validation.  The message names the
            file, the field and the expected/actual form.
    """
    document_path = Path(path)
    try:
        text = document_path.read_text(encoding="utf-8")
    except OSError as err:
        raise RobotDataError(f"unreadable robot profile {document_path}: {err}") from err
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise RobotDataError(f"unreadable robot profile {document_path}: {err}") from err
    if not isinstance(document, Mapping):
        raise RobotDataError(
            f"robot profile {document_path} must be a mapping, " f"got {type(document).__name__}"
        )

    if "action_faces" in document:
        raise _error(
            document_path,
            "action_faces",
            "was renamed to 'extra_action_sets'; rename the key " "(no alias is accepted)",
        )

    unknown = sorted(str(key) for key in document if key not in _TOP_LEVEL_KEYS)
    if unknown:
        raise _error(
            document_path,
            "top level",
            f"unknown key(s) {unknown}; known keys are {sorted(_TOP_LEVEL_KEYS)}",
        )

    missing = [key for key in _REQUIRED_KEYS if key not in document]
    if missing:
        raise _error(
            document_path,
            ", ".join(missing),
            f"required field(s) missing; required fields are {list(_REQUIRED_KEYS)}",
        )

    schema = document.get("schema")
    if schema != SCHEMA_ID:
        raise _error(
            document_path,
            "schema",
            f"expected {SCHEMA_ID!r}, got {schema!r}",
        )

    raw_extra_action_sets = document.get("extra_action_sets")
    raw_foot_bodies = document.get("foot_bodies")
    raw_body_sets = document.get("body_sets")
    try:
        return RobotProfile(
            name=document["name"],
            joint_sets=document["joint_sets"],
            default_joint_pos=document["default_joint_pos"],
            assets=_assets(document_path, document.get("assets")),
            extra_action_sets={} if raw_extra_action_sets is None else raw_extra_action_sets,
            base_body=document.get("base_body"),
            foot_bodies=() if raw_foot_bodies is None else raw_foot_bodies,
            body_sets={} if raw_body_sets is None else raw_body_sets,
            body_bindings=_body_bindings(document_path, document.get("body_bindings")),
            asset_backends=_asset_backends(document_path, document.get("asset_backends")),
            asset_options=_asset_options(document_path, document.get("asset_options")),
            base_height=document.get("base_height"),
            full_joint_order=document.get("full_joint_order"),
            actuation=_actuation(document_path, document.get("actuation")),
            sites=_sites(document_path, document.get("sites")),
        )
    except (TypeError, ValueError) as err:
        if isinstance(err, RobotDataError):
            raise
        raise _error(document_path, "profile", str(err)) from err


__all__ = ["SCHEMA_ID", "RobotDataError", "load_robot_profile"]
