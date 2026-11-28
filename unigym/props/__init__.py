"""PropManager: interactive prop scene + state manager (neutral face).

``build_prop_manager(config, num_envs, env_spacing, device, base_dir,
backend)`` returns a :class:`PropManager` for a yaml ``props:``
section, or ``None`` when the section is absent / disabled / declares no
types -- so callers can treat the result uniformly.  ``backend`` is the
explicit canonical env backend (no alias and no default).

The handler implementations are backend-owned: the concrete
IsaacLab handlers live in ``unigym.backends.isaaclab2.props.handlers`` and
the MuJoCo side in ``unigym.backends.shared.mjfamily.props.props``.  This package keeps the
neutral vocabulary (:class:`PropSpec`, :class:`PropManager`, placement,
validation, geometry and the generated-URDF cache).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .base import (
    PropHandler,
    PropManager,
    PropPlacement,
    PropSpec,
    quat_rotate,
    quat_rotate_inverse,
)

# provider registration side effect (the closed template registry fills
# here; templates never imports providers, so the arrow is one-way)
from . import providers  # noqa: F401

if TYPE_CHECKING:
    from unigym.assets.plan import AssetPlan


def build_prop_manager(
    config: dict | None,
    num_envs: int,
    env_spacing: float,
    device: str = "cpu",
    base_dir: str | None = None,
    backend: str | None = None,
    asset_plan: AssetPlan | None = None,
    prop_backend: Callable[[], object] | None = None,
    mujoco_prop_backend: Callable[[], object] | None = None,
) -> PropManager | None:
    """Build a :class:`PropManager` from a yaml ``props:`` section.

    Returns ``None`` when the section is missing, ``enabled`` is false, or
    it declares no types -- so callers can treat the result uniformly.
    ``backend`` is the explicit canonical env backend (no alias and no
    default).  ``prop_backend`` / ``mujoco_prop_backend`` are the injected
    construction faces; ``asset_plan`` is the composition boundary that
    resolves template-emitted asset files.
    """
    if not config:
        return None
    if not config.get("enabled", True):
        return None
    if not config.get("types"):
        return None
    if backend is None or not str(backend).strip():
        from unigym.capabilities.traits import CANONICAL_BACKENDS

        raise ValueError(
            f"props: env backend is required; choose one of: " f"{', '.join(CANONICAL_BACKENDS)}"
        )
    return PropManager(
        config,
        num_envs=num_envs,
        env_spacing=env_spacing,
        device=device,
        base_dir=base_dir,
        backend=backend,
        asset_plan=asset_plan,
        prop_backend=prop_backend,
        mujoco_prop_backend=mujoco_prop_backend,
    )


__all__ = [
    "PropManager",
    "PropSpec",
    "PropPlacement",
    "PropHandler",
    "build_prop_manager",
    "quat_rotate",
    "quat_rotate_inverse",
]
