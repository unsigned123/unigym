"""ContactManager: one manager per run, shared unified layer, two backends.

Mirrors ``cameras/manager.py``: the yaml ``contacts:`` section is parsed into
:class:`contacts.base.ContactGroupSpec` entries; a backend object (isaaclab
or mujoco) owns ALL simulator interaction and only produces RAW force arrays
(``spec.raw_shape``); the format/component/aggregate semantics live ONCE in
``contacts.base.apply_format`` (two backends can never drift).

Lifecycle (driven by the env backends, mirroring props/cameras):

* ``build_contact_manager(cfg, backend, device)`` — collect_cli entry point;
  a missing/empty ``contacts:`` section returns ``None`` (zero-change
  guarantee for every existing config).
* ``setup_scene() -> {field: ContactSensorCfg}`` — isaaclab only; the env
  backend merges the returned dict into the scene cfg BEFORE cloning
  (same ``setattr`` pattern as props).
* ``bind_env(env)`` — after construction; resolves body names, grabs sensor
  handles (isaaclab) or indexes the MJ model (mujoco), and fails fast on
  resolution errors (no silent empty groups).
* ``read() -> {contact/<name>: (N, ...) tensor on env.device}`` — every
  policy step and published once per step by the env backend;
  values are freshly allocated tensors (no aliasing of backend buffers).

Observation path: the run's ``observations.contact`` list selects groups;
the backend reads every configured group once per step and publishes the
selected values into the typed observation registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Callable, Optional, Protocol, cast, runtime_checkable

import numpy as np
import torch

from .base import ContactGroupSpec, apply_format, contact_force_facts, contact_key


@runtime_checkable
class ContactBackend(Protocol):
    """Structural protocol both backend objects satisfy.

    Capability trait: :attr:`unigym.backends.registry.Trait.CONTACTS_GROUPS`
    (a backend without it never receives a ContactManager).  ``setup_scene``
    exists on both (mujoco: returns {} — contacts need no scene elements);
    ``read_raw`` returns a FRESHLY allocated array (numpy for mujoco, torch
    for isaaclab) with ``spec.raw_shape`` layout, forces already rotated
    into ``spec.frame``.
    """

    def setup_scene(self) -> dict: ...

    def bind_env(self, env) -> None: ...

    def read_raw(self, spec: ContactGroupSpec): ...

    def group_axes(self, spec: ContactGroupSpec) -> tuple[int, int]:
        """Resolved (B, F) counts post bind_env (obs catalog shapes)."""
        ...

    def close(self) -> None: ...


class ContactManager:
    """Holds the group specs + the backend; serves ``contact/<name>`` keys."""

    def __init__(self, specs: list[ContactGroupSpec], backend, device="cpu"):
        names = [s.name for s in specs]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"duplicate contact group names: {dupes}")
        self.specs: list[ContactGroupSpec] = list(specs)
        self.backend = backend
        self.device = device
        # director override table (sensor/record face ONLY --
        # physical contact computation is untouched; contract
        # director_contract.md §3.3).  name -> "on" | "off".
        self._overrides: dict[str, str] = {}

    # ---- keys ------------------------------------------------------------ #
    def keys(self) -> list[str]:
        """All sensor-record keys served (``contact/<name>`` per group)."""
        return [contact_key(s.name) for s in self.specs]

    def fact_keys(self) -> list[str]:
        """Force/active-body fact keys served (benchmark contract)."""
        return [key for s in self.specs for key in (s.force_key(), s.active_bodies_key())]

    def metadata(self) -> dict[str, dict]:
        """Static per-group provenance metadata for ``meta["contacts"]``.

        Owner/frame/unit/bodies are declarations, not per-step records.
        """
        return {s.name: s.metadata() for s in self.specs}

    def group_shape(self, name: str) -> tuple | None:
        """Per-env record shape of one group (batch axis stripped).

        Uses the backend's resolved sensor/filter counts (``group_axes``,
        available after ``bind_env``).  Returns ``None`` when the backend
        cannot resolve axes (pre-bind, or a minimal test fake).
        """
        axes = getattr(self.backend, "group_axes", None)
        if axes is None:
            return None
        spec = next(s for s in self.specs if s.name == name)
        b, f = axes(spec)
        return tuple(int(d) for d in spec.output_shape(1, b, f)[1:])

    # ---- director override face  --------------------------------- #
    def set_override(self, name: str, state: str) -> None:
        """Force one group's sensor/record value (director contact op).

        ``state`` is ``on`` (all-True) or ``off`` (all-False); only
        ``format: contact`` groups are overridable (force magnitudes have
        no honest forced value -- fail-fast otherwise).  Unknown groups
        raise KeyError.  Overrides SURVIVE resets (explicit director
        intent, contract §3.3).
        """
        if state not in ("on", "off"):
            raise ValueError(f"override state must be on|off, got {state!r}")
        self._override_spec(name)  # validates existence + format
        self._overrides[name] = state

    def clear_override(self, name: str) -> None:
        """Return one group to the physical reading (director contact op)."""
        self._override_spec(name)
        self._overrides.pop(name, None)

    def overrides(self) -> dict:
        """Live override table (name -> on|off); a copy."""
        return dict(self._overrides)

    def _override_spec(self, name: str):
        for s in self.specs:
            if s.name == name:
                if s.format != "contact":
                    raise ValueError(
                        f"group {name!r} has format {s.format!r} -- only "
                        f"format: contact groups are overridable (sensor "
                        f"face; forces have no honest forced value)"
                    )
                return s
        raise KeyError(name)

    # ---- per-step read ---------------------------------------------------- #
    def read(self) -> dict:
        """``{contact/<name>: (N, ...) torch tensor on self.device}``.

        dtype: float32 for raw/norm, bool for contact.  Values are freshly
        allocated (backend ``read_raw`` clones; ``apply_format`` never
        mutates and raw passthrough is cast into a new tensor when needed).
        Director overrides  apply AFTER the physical read: the
        sensor/record face sees the forced value, physics does not.
        """
        out = {}
        for spec in self.specs:
            # ONE raw read yields the sensor value AND both force facts.
            raw = self.backend.read_raw(spec)
            value = apply_format(raw, spec)
            force, active = contact_force_facts(raw, spec)
            state = self._overrides.get(spec.name)
            if state is not None:
                # forced value in the SAME dtype/shape as the physical
                # reading (torch or numpy -- apply_format is dual-stack)
                if torch.is_tensor(value):
                    value = torch.ones_like(value) if state == "on" else torch.zeros_like(value)
                else:
                    arr = np.asarray(value)
                    value = np.ones_like(arr) if state == "on" else np.zeros_like(arr)
            out[contact_key(spec.name)] = self._to_record_tensor(value)
            # Physical fact keys are never director-overridden (only the
            # sensor/record face is); they always describe the physics read.
            out[spec.force_key()] = self._to_record_tensor(force)
            out[spec.active_bodies_key()] = self._to_record_tensor(active)
        return out

    # ---- v057 detail face (decision package / slip criterion) ------------- #
    def has_detail(self, name: str) -> bool:
        """True when the bound backend implements the per-contact detail face."""
        self._spec(name)
        return callable(getattr(self.backend, "read_detail", None))

    def detail_keys(self) -> list[str]:
        """The ops detail keys served (``contact_ops/<name>`` per group)."""
        return [f"contact_ops/{s.name}" for s in self.specs]

    def read_detail(self, name: str) -> list[dict]:
        """Per-contact records for one group (JSON-ready list of dicts;
        contacts/backends/mujoco.py ``read_detail`` carries the record
        schema + the v_t computation).  A backend without the detail
        face fails fast -- a silent empty list would read as 'no
        contact' and poison the slip criterion."""
        fn = getattr(self.backend, "read_detail", None)
        if not callable(fn):
            raise NotImplementedError(
                f"contact backend {type(self.backend).__name__} has no "
                f"per-contact detail face (contact_ops/{name})"
            )
        detail = cast(Callable[[ContactGroupSpec], list[dict]], fn)
        return detail(self._spec(name))

    def _spec(self, name: str) -> ContactGroupSpec:
        for s in self.specs:
            if s.name == name:
                return s
        raise KeyError(name)

    def _to_record_tensor(self, value) -> torch.Tensor:
        if torch.is_tensor(value):
            t = value.detach()
            return t if str(t.device) == str(torch.device(self.device)) else t.to(self.device)
        return torch.as_tensor(np.asarray(value)).to(self.device)

    # ---- env lifecycle passthrough ---------------------------------------- #
    def setup_scene(self) -> dict:
        return self.backend.setup_scene()

    def bind_env(self, env) -> None:
        self.backend.bind_env(env)

    def close(self) -> None:
        self.backend.close()


def _validate_articulation_scope(
    specs: list[ContactGroupSpec],
    articulation_names: Optional[Sequence[str]],
) -> Optional[tuple[str, ...]]:
    """Validate every group's owner against the run articulations.

    Returns the articulation order forwarded to the backend, or ``None`` when
    the caller did not supply one (direct manager construction).  An omitted
    owner means "the only articulation"; it is legal only when the run
    declares exactly one.
    """
    if articulation_names is None:
        return None
    names = tuple(str(name) for name in articulation_names)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate articulation names declared by the run: {duplicates}")
    for spec in specs:
        if spec.articulation is None:
            if len(names) > 1:
                raise ValueError(
                    f"contact {spec.name!r}: the run declares articulations "
                    f"{list(names)}, so the group must name its owner with "
                    f"articulation: <name> (or "
                    f"sensor_bodies: {{contact_group: {{articulation: <name>, "
                    f"group: <group>}}}})"
                )
            continue
        if spec.articulation not in names:
            raise ValueError(
                f"contact {spec.name!r}: articulation {spec.articulation!r} "
                f"is not declared by the run (declared: {list(names)})"
            )
    return names


def build_contact_manager(
    cfg,
    backend: str,
    device: str = "cpu",
    contact_groups: Optional[Mapping[str, Mapping[str, tuple[str, ...]]]] = None,
    prop_names: Optional[list] = None,
    articulated_props: Optional[set] = None,
    contact_assets: Optional[Mapping[str, str]] = None,
    articulation_names: Optional[Sequence[str]] = None,
) -> Optional[ContactManager]:
    """collect_cli entry point.

    Args:
        cfg: yaml ``contacts:`` section (list of group dicts) or None.
            None/empty -> ``None`` (no manager, zero code paths).  Each
            entry's ``sensor_bodies``/``filter_bodies`` may use the
            declarative reference forms
            ``{contact_group: {articulation, group}}`` and
            ``{prop: <name>}`` -- resolved FIRST against
            ``contact_groups`` (the run layer's native-body inventory) and
            ``prop_names`` (the run's declared props), then validated as
            plain lists; see contacts.base.resolve_contact_entries.
        backend: env backend name (``isaaclab`` | ``mujoco``); anything else
            raises NotImplementedError BEFORE any sim state exists.
        device: env device the record tensors must live on.
        contact_groups: per-articulation native-body inventory backing the
            ``{contact_group: ...}`` references
            (``{articulation: {group: (native bodies)}}``).
        prop_names: names of the props declared in the same run config,
            backing the ``{prop: ...}`` filter references.
        articulated_props: names of the ARTICULATION props ;
            their ``{prop: ...}`` filters resolve to the whole subtree
            (``Prop_<name>_.*``) instead of a single body.
        contact_assets: per-articulation resolved selected asset path for
            ``contacts: all``.  Backends use it to scope the all-bodies
            expansion (isaaclab reads the pre-clone USD inventory);
            ``None``/absent means no ``all`` inventory was declared.
        articulation_names: the run's articulation declaration order.  When
            given, every explicitly owned group must name a declared
            articulation and every group in a multi-articulation run must
            name an owner (no global union and no cross-articulation body
            regex matching).  ``None`` keeps the historical direct-manager
            construction path.
    """
    if not cfg:
        return None
    from .base import build_contact_backend, resolve_contact_entries

    specs = [
        ContactGroupSpec.from_yaml(e)
        for e in resolve_contact_entries(
            cfg,
            backend,
            contact_groups=contact_groups,
            prop_names=prop_names,
            articulated_props=(frozenset(articulated_props or ())),
        )
    ]
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate contact group names in yaml: {dupes}")
    scope_names = _validate_articulation_scope(specs, articulation_names)
    be = build_contact_backend(
        backend,
        specs,
        contact_assets=contact_assets,
        articulation_names=scope_names,
    )
    return ContactManager(specs, be, device=device)
