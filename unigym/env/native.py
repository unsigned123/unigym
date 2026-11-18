"""The explicit, unstable native escape hatch (external packs only).

``NativeHandle`` is a typed wrapper over backend-native objects.  It is not a
neutral API and carries no compatibility promise: names are stable only within
``NativeHandle.version``, while the object types are backend-version owned.
Core libraries, tasks, examples, and in-repo packs must not call
``env.native(...)``; see ``docs/native_handle.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class NativeHandle:
    """Backend-native objects behind an explicitly unstable escape hatch.

    The handle returns **identity** references; it never copies, settles,
    writes, or imports an engine lazily.  External packs own every causal
    effect and the cleanup of objects they mutate.
    """

    backend: str
    objects: Mapping[str, object]
    unstable: bool = True
    version: int = 1

    def __post_init__(self) -> None:
        """Freeze the name table while keeping the referenced objects identical."""
        object.__setattr__(self, "objects", MappingProxyType(dict(self.objects)))

    def __getitem__(self, name: str) -> object:
        """Return the native object named *name* (identity, no copy)."""
        try:
            return self.objects[name]
        except KeyError:
            raise KeyError(self._unknown(name)) from None

    def __getattr__(self, name: str) -> object:
        """Attribute access for handle names (``handle.model``)."""
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.objects[name]
        except KeyError:
            raise AttributeError(self._unknown(name)) from None

    def require(self, name: str) -> object:
        """Return the named native object or fail with the available names."""
        try:
            return self.objects[name]
        except KeyError:
            raise KeyError(self._unknown(name)) from None

    def keys(self) -> tuple[str, ...]:
        """The native object names carried by this handle."""
        return tuple(self.objects)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self.objects))

    def _unknown(self, name: str) -> str:
        return (
            f"NativeHandle(backend={self.backend!r}, version={self.version}) has "
            f"no native object {name!r}; available: {list(self.keys())}"
        )


__all__ = ["NativeHandle"]
