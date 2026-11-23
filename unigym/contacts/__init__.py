"""contacts: declarative contact-force groups (feat/contact-manager).

Public surface:

* :func:`build_contact_manager` — collect_cli entry point; parses the yaml
  ``contacts:`` section into specs and dispatches the backend object
  (``mujoco`` | ``mjwarp`` | ``isaaclab2``).  A missing/empty section returns
  ``None`` — no manager, zero code paths (baseline protocol).
* :class:`ContactManager` — holds the specs, serves ``contact/<name>``
  record keys through the SHARED format/component/aggregate unified layer
  (``contacts.base.apply_format``); backends only produce raw arrays.
* :class:`contacts.base.ContactGroupSpec` — yaml-declarable group spec
  (frame REQUIRED, no default; ``sensor_refs`` / ``filter_refs`` explicit).
* :class:`contacts.refs.ContactBodyRef` — the only body reference: typed
  owner + address (no legacy string lists, no auto owner, no reference
  forms, no global regex union).
* :class:`contacts.inventory.ContactInventory` — backend-native address rows
  and per-ref resolution with duplicate/ambiguous/unknown failures.

Record channel: the env backends merge ``manager.read()`` into ``extra`` so
the keys flow through CollectSink/EvaluationSession into npz / episode buffers
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
)
from .inventory import ContactBodyAddress, ContactInventory
from .manager import ContactManager, build_contact_manager
from .refs import MATCHES, OWNERS, ContactBodyRef

__all__ = [
    "AGGREGATE_CONTACT",
    "AGGREGATE_VALUE",
    "BACKENDS",
    "COMPONENTS",
    "FORMATS",
    "FRAMES",
    "MATCHES",
    "MODES",
    "OWNERS",
    "ContactBodyAddress",
    "ContactBodyRef",
    "ContactGroupSpec",
    "ContactInventory",
    "ContactManager",
    "apply_format",
    "build_contact_manager",
    "contact_key",
]
