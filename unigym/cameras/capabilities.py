"""Camera capability gate (v095): the single request-side trait check.

``CameraSpec.count × state`` (docs/designs/v095-camera-capability-model.md
§3) maps onto the flat ``Trait`` enum; the backend's own declaration table
is the only truth.  No second capability type, no second gate.
"""
from __future__ import annotations

from unigym.backends.registry import BackendInfo, Trait

__all__ = ["check", "check_backend", "required_traits"]

_LEGAL_COMBINATIONS = (
    "(global, shared), (per_env, shared), (per_env, per_instance)"
)


def required_traits(spec) -> tuple[Trait, ...]:
    """Traits a ``CameraSpec`` requests, in the doc §3 order."""
    if spec.count == "global":
        return (Trait.CAMERAS_GLOBAL,)
    traits = [Trait.CAMERAS_PER_ENV]
    if spec.state == "per_instance":
        traits.append(Trait.CAMERAS_PER_ENV_STATE)
    return tuple(traits)


def check_backend(backend_label: str, specs, num_envs: int | None = None) -> None:
    """Resolve the backend's ``BackendInfo`` and gate every spec.

    The manager calls this (rather than importing the registry itself):
    capability-side modules must not require backend-package imports.
    *num_envs* enables the v096 ``TraitScope`` check (``None`` = no check).
    """
    from unigym.backends.registry import backend_info

    info = backend_info(backend_label)
    for spec in specs:
        check(info, spec, num_envs)


def check(backend_info: BackendInfo, spec, num_envs: int | None = None) -> None:
    """Fail fast when *backend_info* cannot serve *spec*'s request.

    Reuses ``BackendInfo.missing`` / ``BackendInfo.blocked`` and the
    established ``{error, hint}`` message shape: the backend, the missing
    (or scope-blocked) traits and the legal request set are all named.
    """
    required = required_traits(spec)
    missing = backend_info.missing(*required)
    if missing:
        available = sorted(t.name for t in backend_info.traits)
        raise ValueError(
            f"backend '{backend_info.name}' does not implement camera "
            f"capability {[t.name for t in missing]} requested by camera "
            f"{spec.name!r} (count={spec.count!r}, state={spec.state!r}) "
            f"| hint: legal (count, state) combinations: "
            f"{_LEGAL_COMBINATIONS}; backend declares {available}")
    blocked = backend_info.blocked(required, num_envs)
    if blocked:
        trait, scope = blocked[0]
        raise ValueError(
            f"backend '{backend_info.name}' cannot serve camera capability "
            f"{trait.name} requested by camera {spec.name!r} at "
            f"num_envs={num_envs}: trait is scoped to "
            f"max_envs={scope.max_envs} (requested num_envs={num_envs}) "
            f"| hint: {scope.note or 'this capability is N=1 only'}; "
            f"legal (count, state) combinations: {_LEGAL_COMBINATIONS}")
