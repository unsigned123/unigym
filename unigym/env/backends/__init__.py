"""Env backend registry (engine imports lazy via the central registry).

``unigym.backends.registry`` is the single backend truth: it owns the
canonical names, the build-name alias and every backend's lazy env factory.
This module keeps only the build face (``build_env_backend``) and the
PEP 562 materialization of ``BACKEND_REGISTRY``; importing it never imports
an engine (the factories perform those imports on first use).

Usage (collect.py):

    from unigym.env.backends import build_env_backend
    env = build_env_backend(cfg["env"].get("backend", "isaaclab"),
                            robot_profile=profile, adapter=adapter, ...)
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from unigym.backends.registry import backend_info, build_name
from unigym.backends.registry import known_backends as _registry_known


def known_backends() -> list[str]:
    """Registered canonical backend names (no engine import)."""
    return sorted(_registry_known())


def _env_class(name: str) -> type:
    """Resolve *name*'s env class through the registry factory (lazy)."""
    info = backend_info(name)
    if info.env_class is None:
        raise ValueError(f"env backend {name!r} declares no env_class factory")
    return info.env_class()


def __getattr__(name: str) -> Any:
    if name == "BACKEND_REGISTRY":
        # materialized {canonical name: env class} — imports every backend
        registry = {key: _env_class(key) for key in _registry_known()}
        globals()["BACKEND_REGISTRY"] = registry
        return registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + ["BACKEND_REGISTRY"])


def build_env_backend(name: str | None = None, **kwargs):
    """Build a collection backend by registry name and enforce its contract.

    Order of operations (before any sim state exists):
      1. registry lookup (unknown name -> ``ValueError``);
      2. capability gate: a yaml/adapter request for a capability the backend
         lacks -> ``NotImplementedError``;
      3. construct the backend env class with the caller's kwargs;
      4. :func:`unigym.env.backends.base.assert_backend_contract`
         structural validation (contract violation -> ``BackendContractError``).
    """
    from .base import assert_backend_contract, gate_capabilities

    canonical = build_name(name)
    env_cls = _env_class(canonical)
    mod = sys.modules.get(env_cls.__module__) or importlib.import_module(env_cls.__module__)
    caps = getattr(mod, "BACKEND_CAPABILITIES", None)
    if caps is not None:
        gate_capabilities(canonical, caps, kwargs, num_envs=kwargs.get("num_envs"))
    env = env_cls(**kwargs)
    assert_backend_contract(env)
    return env
