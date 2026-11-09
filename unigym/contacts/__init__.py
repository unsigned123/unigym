"""contacts: declarative contact-force groups (feat/contact-manager).

Public surface:

* :func:`build_contact_manager` — collect_cli entry point; parses the yaml
  ``contacts:`` section into specs and dispatches the backend object
  (``isaaclab`` | ``mujoco``).  A missing/empty section returns ``None`` —
  no manager, zero code paths (baseline protocol).
* :class:`ContactManager` — holds the specs, serves ``contact/<name>``
  record keys through the SHARED format/component/aggregate unified layer
  (``contacts.base.apply_format``); backends only produce raw arrays.
* :class:`contacts.base.ContactGroupSpec` — yaml-declarable group spec
  (frame REQUIRED, no default).
* Declarative references : ``sensor_bodies: {body_parts:
  grasp.left}`` and ``filter_bodies: {prop: cracker}`` resolve against the
  robot profile body-parts groups and the run's declared props
  (``contacts.base.resolve_contact_entries``; name-addressing only, zero
  hardcoded bodies).  The retired auto held groups are now
  expressed as two declared pairwise groups.

Record channel: the env backends merge ``manager.read()`` into ``extra`` so
the keys flow through CollectSink/BenchmarkSink into npz / episode buffers
(L2 loop).  Obs path: adapters may request ``contact/<name>`` keys
(decoupled from recording).  Contract docs: ``docs/contracts/contacts_contract.md``.

The package is numpy/torch-only at import time; backend imports are lazy
(inside ``build_contact_manager``), mirroring the ``cameras/`` convention.
"""
from __future__ import annotations

from .base import (
    AGGREGATE_CONTACT,
    AGGREGATE_VALUE,
    BACKENDS,
    COMPONENTS,
    FORMATS,
    FRAMES,
    MODES,
    ContactGroupSpec,
    apply_format,
    contact_key,
    resolve_body_part_group,
    resolve_contact_entries,
    resolve_prop_filter,
)
from .manager import ContactManager, build_contact_manager

__all__ = [
    "AGGREGATE_CONTACT", "AGGREGATE_VALUE", "BACKENDS", "COMPONENTS",
    "FORMATS", "FRAMES", "MODES", "ContactGroupSpec", "ContactManager",
    "apply_format", "build_contact_manager", "contact_key",
    "resolve_body_part_group", "resolve_contact_entries",
    "resolve_prop_filter",
]
