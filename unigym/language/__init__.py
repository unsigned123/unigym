"""Language generator registry (torch-only, unit-testable)."""

from __future__ import annotations

from .base import LanguageGenerator
from .template import TemplateGenerator

LANGUAGE_REGISTRY = {
    "template": TemplateGenerator,
}


def build_generator(section: dict | None) -> LanguageGenerator | None:
    """Build a language generator from the yaml ``language`` section.

    Returns ``None`` when the section is absent or has no ``generator`` key,
    so callers can skip language generation entirely.
    """
    if section is None:
        return None
    name = section.get("generator")
    if name is None:
        return None
    if name not in LANGUAGE_REGISTRY:
        raise ValueError(
            f"unknown language generator '{name}'; " f"available: {sorted(LANGUAGE_REGISTRY)}"
        )
    params = dict(section.get("params", {}))
    return LANGUAGE_REGISTRY[name](**params)


__all__ = [
    "LanguageGenerator",
    "TemplateGenerator",
    "build_generator",
    "LANGUAGE_REGISTRY",
]
