"""Command sampler registry (torch-only, unit-testable)."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module

from .base import CommandSampler
from .fixed import FixedSampler
from .lhs import LHSSampler
from .uniform import UniformSampler

COMMAND_REGISTRY = {
    "fixed": FixedSampler,
    "uniform": UniformSampler,
    "lhs": LHSSampler,
}

#: Fields a ``command.module`` section may declare: the factory reference, the
#: explicit command dimension and its keyword parameters.
_MODULE_FIELDS = frozenset({"module", "dim", "params"})


def build_sampler(section: dict, num_envs: int, device: str) -> CommandSampler:
    """Build the run's command sampler from its ``command:`` section.

    Exactly one of ``name`` (built-in registry) or ``module``
    (``"pkg.mod:factory"``) selects the sampler; declaring both or neither is
    a configuration error naming the field.  Built-in sections keep their
    arbitrary parameter pass-through.  A module section accepts only
    ``module`` / ``dim`` / optional ``params`` and calls
    ``factory(dim=..., device=..., num_envs=..., **params)``; the factory must
    return a :class:`CommandSampler`.
    """
    if not isinstance(section, Mapping):
        raise TypeError(f"command: section must be a mapping, got {type(section).__name__}")
    name = section.get("name")
    module = section.get("module")
    if (name is None) == (module is None):
        raise ValueError(
            "command: exactly one of 'name' (built-in sampler) or 'module' "
            "('pkg.mod:factory') is required; "
            f"got name={name!r}, module={module!r}"
        )
    if module is not None:
        return _build_module_sampler(section, num_envs=num_envs, device=device)
    params = {k: v for k, v in section.items() if k not in ("name", "dim")}
    return COMMAND_REGISTRY[section["name"]](
        dim=section["dim"], device=device, num_envs=num_envs, **params
    )


def _build_module_sampler(section: Mapping, *, num_envs: int, device: str) -> CommandSampler:
    """Build a custom sampler through a ``module:factory`` reference."""
    unknown = sorted(set(section) - _MODULE_FIELDS)
    if unknown:
        raise ValueError(
            "command: a 'module' section accepts only "
            f"{sorted(_MODULE_FIELDS)}, got unknown field(s) {unknown}"
        )
    if "dim" not in section:
        raise ValueError(
            "command.dim is required when command.module is declared "
            f"(section fields: {sorted(section)})"
        )
    dim = section["dim"]
    if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
        raise ValueError(f"command.dim must be a positive int with command.module, got {dim!r}")
    params = section.get("params", {})
    if not isinstance(params, Mapping):
        raise TypeError(
            "command.params must be a mapping of factory keyword arguments, "
            f"got {type(params).__name__}"
        )
    factory = _resolve_factory(section["module"])
    sampler = factory(dim=dim, device=device, num_envs=num_envs, **dict(params))
    if not isinstance(sampler, CommandSampler):
        raise TypeError(
            f"command.module {section['module']!r} returned "
            f"{type(sampler).__name__}, expected CommandSampler"
        )
    return sampler


def _resolve_factory(reference: object):
    """Import ``"pkg.mod:factory"`` and return the callable target."""
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError(f"command.module must be 'pkg.mod:factory', got {reference!r}")
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"command.module must be 'pkg.mod:factory', got {reference!r}")
    try:
        imported = import_module(module_name)
    except ImportError as err:
        raise ValueError(
            f"command.module {reference!r}: cannot import module " f"{module_name!r} ({err})"
        ) from err
    try:
        target = getattr(imported, attribute)
    except AttributeError as err:
        raise ValueError(
            f"command.module {reference!r}: module {module_name!r} has no "
            f"attribute {attribute!r}"
        ) from err
    if not callable(target):
        raise TypeError(f"command.module {reference!r} target {target!r} is not callable")
    return target
