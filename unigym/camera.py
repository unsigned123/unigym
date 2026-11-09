"""Programmatic camera builder: ``unigym.Camera`` / ``CameraSet`` (S1b-ii).

``Camera.world`` / ``Camera.attached`` / ``Camera.chase`` produce the
**existing** ``cameras:`` list-entry schema (the same dict
:meth:`unigym.cameras.base.CameraSpec.from_yaml` parses).  Nothing here
introduces a second camera config: a :class:`CameraSet` is a
``MutableMapping`` of declared entries that the run facade copies into the
config before :func:`unigym.run.spec.build_spec`.

Validation stays lazy (the camera package imports torch): entries are
structurally checked on ``add`` and schema-validated when the run locks
the set at build time.  ``run.model.cameras`` remains the compiled IR
table; ``run.cameras`` lists the declared specs.
"""

from __future__ import annotations

import copy
import math
from collections.abc import MutableMapping

from unigym._typing import JsonValue

__all__ = ["Camera", "CameraSet"]


def _vec(value, length: int, what: str) -> list:
    """Validate a finite numeric vector and return a float list."""
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in value
        )
    ):
        raise ValueError(f"{what} must be a {length}-vector of finite " f"numbers, got {value!r}")
    return [float(v) for v in value]


def _positive(value, what: str) -> float:
    """Validate one finite positive scalar."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{what} must be a finite number > 0, got {value!r}")
    return float(value)


def _resolution(width, height) -> tuple:
    """Validate a positive integer resolution."""
    for label, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"camera {label} must be a positive int, " f"got {value!r}")
    return int(width), int(height)


def _entry(name, rig, types, width, height, hfov_deg, extra) -> dict:
    """Assemble one cameras-list entry (the existing schema)."""
    if not isinstance(name, str) or not name or "/" in name or name.strip() != name:
        raise ValueError(
            f"camera name {name!r} must be non-empty, trimmed and free of "
            f"'/' (it is part of the cam/<name>/<type> key)"
        )
    width, height = _resolution(width, height)
    if isinstance(types, str):
        types = (types,)
    entry = {
        "name": name,
        "rig": rig,
        "types": list(types or ("rgb",)),
        "width": width,
        "height": height,
        "hfov_deg": _positive(hfov_deg, "hfov_deg"),
    }
    entry.update(extra)
    return entry


class Camera:
    """Factories for the existing camera-spec dict schema."""

    @staticmethod
    def world(
        name,
        pos,
        lookat=None,
        quat=None,
        width=64,
        height=48,
        hfov_deg=87.0,
        *,
        types=("rgb",),
        **extra,
    ) -> dict:
        """A world-fixed camera (``pos`` + ``lookat`` or ``quat`` wxyz)."""
        rig = {"type": "world", "pos": _vec(pos, 3, "world camera pos")}
        if lookat is not None and quat is not None:
            raise ValueError("Camera.world takes lookat OR quat, not both")
        if lookat is not None:
            rig["lookat"] = _vec(lookat, 3, "world camera lookat")
        elif quat is not None:
            rig["rot"] = _vec(quat, 4, "world camera quat")
        else:
            raise ValueError("Camera.world needs lookat=[x,y,z] or " "quat=[w,x,y,z]")
        return _entry(name, rig, types, width, height, hfov_deg, extra)

    @staticmethod
    def attached(
        name,
        mount,
        pos=(0.0, 0.0, 0.0),
        quat=None,
        width=64,
        height=48,
        hfov_deg=87.0,
        *,
        types=("rgb",),
        **extra,
    ) -> dict:
        """A body-mounted camera (``mount`` = robot link name)."""
        if not isinstance(mount, str) or not mount:
            raise ValueError(
                f"Camera.attached mount must be a non-empty " f"link name, got {mount!r}"
            )
        rig = {
            "type": "body",
            "attach": mount,
            "pos": _vec(pos, 3, "attached camera pos"),
        }
        if quat is not None:
            rig["rot"] = _vec(quat, 4, "attached camera quat")
        return _entry(name, rig, types, width, height, hfov_deg, extra)

    @staticmethod
    def chase(
        name,
        target_env=0,
        pos=(0.0, 0.0, 0.0),
        lookat=(0.0, 0.0, 0.0),
        width=64,
        height=48,
        hfov_deg=87.0,
        *,
        types=("rgb",),
        **extra,
    ) -> dict:
        """A robot-yaw-relative chase camera for *target_env*."""
        if isinstance(target_env, bool) or not isinstance(target_env, int) or target_env < 0:
            raise ValueError(f"Camera.chase target_env must be an int >= 0, " f"got {target_env!r}")
        rig = {
            "type": "chase",
            "pos": _vec(pos, 3, "chase camera pos"),
            "lookat": _vec(lookat, 3, "chase camera lookat"),
            "target_env": int(target_env),
        }
        return _entry(name, rig, types, width, height, hfov_deg, extra)


class CameraSet(MutableMapping):
    """Ordered declared-camera set with a pre-build ``add``.

    Behaves like a mapping ``name -> cameras-list entry``.  ``add`` is
    pre-build only; after :meth:`lock` (run build) the declared set stays
    readable but immutable.  ``to_list`` yields config-ready copies.
    """

    def __init__(self, entries=(), *, locked: bool = False):
        self._entries: dict[str, dict[str, JsonValue]] = {}
        self._locked = bool(locked)
        for entry in entries or ():
            self.add(entry)

    # -- mutation ------------------------------------------------------ #
    def add(self, entry, *, name: str | None = None) -> dict:
        """Add one declared camera entry (dict or another CameraSet)."""
        if self._locked:
            raise RuntimeError(
                "CameraSet.add: the run is already built; cameras are "
                "declared pre-build only (create a new Run)"
            )
        if isinstance(entry, CameraSet):
            last: dict = {}
            for value in entry.values():
                last = self.add(value)
            return last
        if not isinstance(entry, dict):
            raise TypeError(f"CameraSet.add expects a dict entry, got " f"{type(entry).__name__}")
        value = copy.deepcopy(entry)
        if name is not None:
            value["name"] = name
        cam_name = value.get("name")
        if not isinstance(cam_name, str) or not cam_name:
            raise ValueError("camera entry needs a non-empty 'name'")
        self._entries[cam_name] = value
        return copy.deepcopy(value)

    def lock(self) -> CameraSet:
        """Validate every entry against CameraSpec, then freeze the set."""
        if not self._locked:
            from unigym.cameras.base import CameraSpec

            for value in self._entries.values():
                CameraSpec.from_yaml(value)
            self._locked = True
        return self

    def to_list(self) -> list[dict[str, JsonValue]]:
        """Config-ready copies of the declared entries (declaration order)."""
        return [copy.deepcopy(value) for value in self._entries.values()]

    def names(self) -> list:
        """Declared camera names in declaration order."""
        return list(self._entries)

    # -- mapping protocol --------------------------------------------- #
    def __getitem__(self, key: str) -> dict:
        return copy.deepcopy(self._entries[key])

    def __setitem__(self, key: str, value: dict) -> None:
        if self._locked:
            raise RuntimeError(
                "CameraSet is locked after build; cameras are declared " "pre-build only"
            )
        if not isinstance(value, dict):
            raise TypeError("camera entry must be a dict")
        entry = copy.deepcopy(value)
        entry.setdefault("name", key)
        if entry["name"] != key:
            raise ValueError(f"camera entry name {entry['name']!r} does not match key " f"{key!r}")
        self._entries[key] = entry

    def __delitem__(self, key: str) -> None:
        if self._locked:
            raise RuntimeError("CameraSet is locked after build")
        del self._entries[key]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        state = "locked" if self._locked else "open"
        return f"CameraSet({self.names()!r}, {state})"

    @property
    def locked(self) -> bool:
        """True once the run built (the declared set is read-only)."""
        return self._locked

    def unlock(self) -> None:
        """Re-open the set (tests/tools only; never called by ``Run``)."""
        self._locked = False
