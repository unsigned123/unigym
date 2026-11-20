"""Generated asset sites preserve body-local orientation through IR and scaling."""

from __future__ import annotations

import pytest

from unigym.props.asset_ir import AssetDefinition, Body, Site
from unigym.props.scaling import scale_asset_definition

_HALF = 2**-0.5


def _asset() -> AssetDefinition:
    return AssetDefinition(
        name="cup",
        root=Body(
            "cup",
            sites=(
                Site(
                    "handle",
                    pos=(0.0, 0.1, 0.2),
                    size=0.01,
                    quat=(_HALF, 0.0, 0.0, _HALF),
                ),
            ),
        ),
    )


def test_site_quat_is_normalized_at_construction() -> None:
    site = Site("handle", quat=(2.0, 0.0, 0.0, 0.0))

    assert site.quat == pytest.approx((1.0, 0.0, 0.0, 0.0))


def test_site_quat_rejects_zero_and_non_finite_orientations() -> None:
    with pytest.raises(ValueError, match="finite non-zero norm"):
        Site("handle", quat=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="4 finite numbers"):
        Site("handle", quat=(1.0, 0.0, 0.0, float("nan")))


def test_site_quat_normalizes_a_very_large_finite_orientation() -> None:
    """An overflow-scale finite quaternion normalizes without becoming zero."""
    site = Site("handle", quat=(1e308, 1e308, 0.0, 0.0))

    assert site.quat == pytest.approx((2**-0.5, 2**-0.5, 0.0, 0.0))


def test_site_quat_rejects_a_quaternion_whose_norm_overflows() -> None:
    """A finite quaternion with an infinite safe norm fails loudly."""
    with pytest.raises(ValueError, match="finite non-zero norm"):
        Site("handle", quat=(1e308, 1e308, 1e308, 1e308))


def test_to_model_ir_copies_the_site_quat() -> None:
    model = _asset().to_model_ir()

    assert tuple(model.sites["handle"].quat) == pytest.approx((_HALF, 0.0, 0.0, _HALF))


def test_scaling_preserves_rotation_and_scales_position_and_size() -> None:
    scaled = scale_asset_definition(_asset(), (2.0, 2.0, 2.0))

    site = scaled.root.sites[0]
    assert site.pos == pytest.approx((0.0, 0.2, 0.4))
    assert site.size == pytest.approx(0.02)
    assert site.quat == pytest.approx((_HALF, 0.0, 0.0, _HALF))


def test_site_specs_uses_payload_body_names() -> None:
    specs = _asset().site_specs()

    assert specs["handle"].body == "cup"
    assert specs["handle"].pos == (0.0, 0.1, 0.2)
    assert specs["handle"].quat_wxyz == pytest.approx((_HALF, 0.0, 0.0, _HALF))
