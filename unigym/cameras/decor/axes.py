"""axes generator: body-frame coordinate triad (probe1(c) recipe).

Three arrows along the body frame's X/Y/Z at the body origin -- the
classic EE-frame visualization.  All decor is injected (L1.5 scene
buffer); nothing here touches the model.
"""
from __future__ import annotations

import numpy as np

from . import (_check_keys, _pos_float, DecorContext, register_gen)

__all__ = ["Axes"]

AXIS_COLORS = ((1.00, 0.20, 0.20),   # X red
               (0.20, 1.00, 0.35),   # Y green
               (0.30, 0.50, 1.00))   # Z blue


@register_gen("axes")
class Axes:
    """Body-frame X/Y/Z triad at the target body origin."""

    @staticmethod
    def validate(params, label="axes"):
        _check_keys(params, ("size",), label)
        return {"size": _pos_float(params, "size", 0.03, label)}

    @staticmethod
    def generate(params, target, ctx: DecorContext):
        if target == "*":
            raise ValueError("axes: target must name a body "
                             "(no body prefix '*')")
        body, _ = ctx.target_geoms(target, collision_only=False)
        bid = ctx._body_lower.index(body.lower())
        pos = np.asarray(ctx.data.xpos[bid], dtype=np.float64)
        rot = np.asarray(ctx.data.xmat[bid]).reshape(3, 3)
        size = params["size"]
        out = []
        for i, col in enumerate(AXIS_COLORS):
            out.append({"type": "arrow", "from": pos,
                        "to": pos + rot[:, i] * size,
                        "radius": size * 0.06,
                        "rgba": np.array([col[0], col[1], col[2], 1.0])})
        return out
