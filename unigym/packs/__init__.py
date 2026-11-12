"""Pluggable content packs: the shared error type and public package boundary.

The pack contract is :mod:`unigym.pack_api`; discovery, catalog lookup,
resource resolution and activation are owned by :mod:`unigym.packs.runtime`.
A pack root carries a ``unigym.pack.yaml`` document validated by
:mod:`unigym.packs.schema`.  There is no implicit repository mount and no
mount order: every resource belongs to an explicitly discovered pack.
"""

from __future__ import annotations


class PackError(ValueError):
    """A pack manifest, resource or activation is invalid."""


__all__ = ["PackError"]
