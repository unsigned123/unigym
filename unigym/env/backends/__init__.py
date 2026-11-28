"""Env backend composition (engine imports lazy via the bindings table).

Canonical names, the build-name spelling and the pure spec metadata live in
:mod:`unigym.capabilities`; the lazy env factory comes from
:mod:`unigym.backends.compose`.  This module keeps only the build face
(``build_env_backend``) and the PEP 562 materialization of
``BACKEND_REGISTRY``; importing it never imports an engine (the factories
perform those imports on first use).

Usage (collect.py):

    from unigym.env.backends import build_env_backend
    env = build_env_backend(cfg["env"]["backend"],  # required canonical name
                            robot_profile=profile, adapter=adapter, ...)

There is no default backend: the caller must pass one of the canonical
registry names (``mujoco`` / ``mjwarp`` / ``isaaclab2`` /
``isaaclab3_physx`` / ``isaaclab3_newton``).
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from unigym.backends.compose import backend_runtime
from unigym.capabilities.traits import build_name, known_backends as _runtime_known


def known_backends() -> list[str]:
    """Registered canonical backend names (no engine import)."""
    return sorted(_runtime_known())


def _env_class(name: str) -> type:
    """Resolve *name*'s env class through the composition bindings (lazy)."""
    env_class = backend_runtime(name).bindings.env_class
    if env_class is None:
        raise ValueError(f"env backend {name!r} declares no env_class factory")
    return env_class()


def __getattr__(name: str) -> Any:
    if name == "BACKEND_REGISTRY":
        # materialized {canonical name: env class} — imports every backend
        registry = {key: _env_class(key) for key in _runtime_known()}
        globals()["BACKEND_REGISTRY"] = registry
        return registry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + ["BACKEND_REGISTRY"])


def build_env_backend(name: str, **kwargs):
    """Build a collection backend by canonical registry name.

    *name* is required (no default).  The registry's :func:`build_name`
    validates the canonical spelling only; it performs no probing.

    Order of operations (before any sim state exists):
      1. canonical-name validation (unknown/non-canonical -> ``ValueError``);
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
