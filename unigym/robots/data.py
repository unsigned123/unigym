"""Load external ``unigym.robot/1`` robot profile documents.

The document is authoring data only: joint sets, standby pose, canonical base
and foot facts, per-format body bindings, declared assets, action faces and an
optional full joint order.  Loading builds a backend-neutral
:class:`~unigym.robots.base.RobotProfile` and never imports a backend, an
engine or pack Python.

This module owns the schema gate.  Only top-level unknown keys are rejected;
``body_bindings`` values are per-format native names, not top-level fields.
``assets`` accepts the same ``{mjcf|urdf|usd: <ref>}`` mapping as
:class:`~unigym.assets.asset_ref.AssetRef`; a pack reference stays an
unresolved string until the composition boundary resolves it.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import yaml

from unigym._typing import JsonValue
from unigym.assets.asset_ref import AssetRef, AssetRefError

from .base import (
    BodyBinding,
    BodyBindingError,
    CanonicalBodyId,
    RobotProfile,
)

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
    "action_faces",
    "full_joint_order",
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

    raw_action_faces = document.get("action_faces")
    raw_foot_bodies = document.get("foot_bodies")
    raw_body_sets = document.get("body_sets")
    try:
        return RobotProfile(
            name=document["name"],
            joint_sets=document["joint_sets"],
            default_joint_pos=document["default_joint_pos"],
            assets=_assets(document_path, document.get("assets")),
            action_faces={} if raw_action_faces is None else raw_action_faces,
            base_body=document.get("base_body"),
            foot_bodies=() if raw_foot_bodies is None else raw_foot_bodies,
            body_sets={} if raw_body_sets is None else raw_body_sets,
            body_bindings=_body_bindings(document_path, document.get("body_bindings")),
            base_height=document.get("base_height"),
            full_joint_order=document.get("full_joint_order"),
        )
    except (TypeError, ValueError) as err:
        if isinstance(err, RobotDataError):
            raise
        raise _error(document_path, "profile", str(err)) from err


__all__ = ["SCHEMA_ID", "RobotDataError", "load_robot_profile"]
