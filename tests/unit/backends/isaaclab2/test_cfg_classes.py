"""The IsaacLab cfg-class table declares the external opaque key set."""

from __future__ import annotations

from unigym.backends.isaaclab2.cfg_classes import IsaacLabCfgClasses
from unigym.backends.isaaclab2.static_shapes import build_static_field

_EXPECTED_KEYS = frozenset(
    {
        "ArticulationRootPropertiesCfg",
        "AssetBaseCfg",
        "CapsuleCfg",
        "CollisionPropertiesCfg",
        "CuboidCfg",
        "CylinderCfg",
        "GroundPlaneCfg",
        "MassPropertiesCfg",
        "PreviewSurfaceCfg",
        "RigidBodyPropertiesCfg",
        "RigidObjectCfg",
        "SphereCfg",
        "UrdfFileCfg",
        "UsdFileCfg",
    }
)


def test_cfg_classes_declares_the_external_key_set():
    """The typed table names every cfg class the asset/static seams resolve."""
    assert frozenset(IsaacLabCfgClasses.__annotations__) == _EXPECTED_KEYS
    assert IsaacLabCfgClasses.__required_keys__ == frozenset()


def test_injected_fake_table_still_builds_a_static_field():
    """A partial fake table satisfies the table contract for offline tests."""

    class _AssetBaseCfg:
        class InitialStateCfg:
            def __init__(self, **kwargs):
                self.kwargs = dict(kwargs)

        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    class _CuboidCfg:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    class _CollisionPropertiesCfg:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    class _PreviewSurfaceCfg:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    classes: IsaacLabCfgClasses = {
        "AssetBaseCfg": _AssetBaseCfg,
        "CuboidCfg": _CuboidCfg,
        "CollisionPropertiesCfg": _CollisionPropertiesCfg,
        "PreviewSurfaceCfg": _PreviewSurfaceCfg,
    }

    field = build_static_field(
        "crate",
        "box",
        {"size": [1.0, 2.0, 3.0]},
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        classes=classes,
    )

    assert isinstance(field, _AssetBaseCfg)
    assert field.kwargs["spawn"].kwargs["size"] == (1.0, 2.0, 3.0)
