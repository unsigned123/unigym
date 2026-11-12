"""PropManager: interactive prop scene + state manager (neutral face).

``build_prop_manager(config, num_envs, env_spacing, device, base_dir,
backend=None)`` returns a :class:`PropManager` for a yaml ``props:``
section, or ``None`` when the section is absent / disabled / declares no
types -- so callers can treat the result uniformly.

The handler implementations are backend-owned: the concrete
IsaacLab handlers live in ``unigym.backends.isaaclab2.props.handlers`` and
the MuJoCo side in ``unigym.backends.shared.mjfamily.props.props``.  This package keeps the
neutral vocabulary (:class:`PropSpec`, :class:`PropManager`, placement,
validation, geometry and the generated-URDF cache) and lazy compatibility
re-exports for the long-standing public class names.
"""

from __future__ import annotations

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

#: legacy public class name -> backend handler module basename
_COMPAT_NAMES = frozenset(
    {
        "BallProp",
        "BoxProp",
        "FreeArticulationProp",
        "GenericProp",
        "PrimitiveProp",
        "PropArticulation",
        "UsdArticulationProp",
    }
)

if TYPE_CHECKING:
    from typing import Any

    from unigym.assets.plan import AssetPlan

    BallProp: Any
    BoxProp: Any
    FreeArticulationProp: Any
    GenericProp: Any
    PrimitiveProp: Any
    PropArticulation: Any
    UsdArticulationProp: Any


def __getattr__(name: str):
    """Lazy compatibility re-export of the backend handler classes (G4)."""
    if name in _COMPAT_NAMES:
        from .base import _compat_class

        return _compat_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _dispatch_handler(spec: PropSpec) -> type[PropHandler]:
    """Pick the handler class for a PropSpec through the backend registry."""
    from .base import _prop_backend

    return _prop_backend(None).handler_for(spec)


def build_prop_manager(
    config: dict | None,
    num_envs: int,
    env_spacing: float,
    device: str = "cpu",
    base_dir: str | None = None,
    backend: str | None = None,
    asset_plan: AssetPlan | None = None,
) -> PropManager | None:
    """Build a :class:`PropManager` from a yaml ``props:`` section.

    Returns ``None`` when the section is missing, ``enabled`` is false, or
    it declares no types -- so callers can treat the result uniformly.
    ``backend`` selects the handler registry (default: the isaaclab face,
    preserving the historical direct-construction behaviour).  ``asset_plan``
    is the composition boundary that resolves template-emitted asset files.
    """
    if not config:
        return None
    if not config.get("enabled", True):
        return None
    if not config.get("types"):
        return None
    return PropManager(
        config,
        num_envs=num_envs,
        env_spacing=env_spacing,
        device=device,
        base_dir=base_dir,
        backend=backend,
        asset_plan=asset_plan,
    )


__all__ = [
    "PropManager",
    "PropSpec",
    "PropPlacement",
    "PropHandler",
    "PropArticulation",
    "UsdArticulationProp",
    "FreeArticulationProp",
    "BallProp",
    "GenericProp",
    "BoxProp",
    "PrimitiveProp",
    "build_prop_manager",
    "quat_rotate",
    "quat_rotate_inverse",
]
