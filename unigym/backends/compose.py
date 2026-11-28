"""Composition root: combine a backend's pure spec with its bindings.

:class:`BackendRuntime` is the composition-facing value object:
:class:`~unigym.capabilities.specs.BackendSpec` metadata plus the lazy
:class:`~unigym.backends.bindings.BackendBindings` construction callables.
Only composition code (``run.build``, ``env.backends``, CLI composition) may
import this module; neutral layers use the spec metadata directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from unigym.backends.bindings import BACKEND_BINDINGS, BackendBindings
from unigym.capabilities.selection import canonical_or_error
from unigym.capabilities.specs import BACKEND_SPECS, BackendSpec, validate_trait_specs
from unigym.capabilities.traits import Trait

__all__ = [
    "BackendRuntime",
    "backend_runtime",
    "validate_backend_runtime",
    "validate_backend_runtimes",
]


@dataclass(frozen=True)
class BackendRuntime:
    """One backend's pure spec plus its lazy construction bindings."""

    spec: BackendSpec
    bindings: BackendBindings


def backend_runtime(name: str) -> BackendRuntime:
    """Return the composition runtime for canonical backend *name*.

    Raises:
        BackendSelectionError: *name* is not a canonical backend name.
    """
    canonical = canonical_or_error(name)
    return BackendRuntime(spec=BACKEND_SPECS[canonical], bindings=BACKEND_BINDINGS[canonical])


def validate_backend_runtime(runtime: BackendRuntime) -> None:
    """Validate the spec <-> bindings declaration consistency of *runtime*.

    Enforces the cross-boundary invariants the pure spec cannot express:

    * ``MODEL_EDIT`` <-> ``model_edit_provider`` + matching field set;
    * ``RUNTIME_QUERY`` <-> ``query_backend``;
    * ``content_fill`` <-> ``content_router``.
    """
    spec = runtime.spec
    bindings = runtime.bindings
    traits = set(spec.traits)
    name = spec.name
    provider = bindings.model_edit_provider
    fields = set(spec.model_edit_fields)
    if Trait.MODEL_EDIT in traits:
        if provider is None:
            raise RuntimeError(
                f"backend {name!r}: MODEL_EDIT is declared but model_edit_provider is None"
            )
        if not fields:
            raise RuntimeError(
                f"backend {name!r}: MODEL_EDIT is declared but model_edit_fields is empty"
            )
        if set(provider.fields) != fields:
            raise RuntimeError(
                f"backend {name!r}: model_edit_provider fields "
                f"{sorted(field.value for field in provider.fields)} != "
                f"model_edit_fields {sorted(field.value for field in fields)}"
            )
    elif provider is not None or fields:
        raise RuntimeError(
            f"backend {name!r}: model-edit provider/fields declared without "
            f"the MODEL_EDIT trait"
        )
    declared = Trait.RUNTIME_QUERY in traits
    if declared and bindings.query_backend is None:
        raise RuntimeError(f"backend {name!r}: RUNTIME_QUERY is declared but query_backend is None")
    if not declared and bindings.query_backend is not None:
        raise RuntimeError(
            f"backend {name!r}: query_backend is declared without the RUNTIME_QUERY trait"
        )
    if spec.content_fill and bindings.content_router is None:
        raise RuntimeError(f"backend {name!r}: content_fill is declared but content_router is None")
    if not spec.content_fill and bindings.content_router is not None:
        raise RuntimeError(f"backend {name!r}: content_router is declared without content_fill")


def validate_backend_runtimes() -> None:
    """Validate every canonical spec + bindings pair (import-time self-check)."""
    validate_trait_specs()
    for name in BACKEND_SPECS:
        validate_backend_runtime(backend_runtime(name))


# import-time self-check: a bad declaration table must never reach a run
validate_backend_runtimes()
