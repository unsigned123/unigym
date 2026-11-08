"""Contact decor generators: dots / normals / mjVIS variant.

* ``contact_dots`` (probe4 verdict): small force-colored spheres
  (degenerate capsules), blue 0 N -> red ``force_scale`` N.  The v1
  coarse discs were rejected; r defaults to the probe's 3.5 mm.
* ``contact_normals``: thin arrow per contact along frame row 0 (the
  normal, pointing from ``geom[0]`` toward ``geom[1]`` -- MuJoCo
  convention).
* ``contact_mjvis``: the legacy flags ``contacts: true`` semantic folded in as
  a generator variant -- toggles ``mjVIS_CONTACTPOINT`` +
  ``mjVIS_CONTACTFORCE`` on the scene option (the "simple tier" of the
  flags migration map; the injected-geom tier above is the rich one).
"""
from __future__ import annotations

import numpy as np

from . import (_check_keys, _norm_contact_select, _pos_float,
               DecorContext, register_gen)

__all__ = ["ContactDots", "ContactNormals", "ContactMjvis"]

BLUE_F = np.array([0.15, 0.45, 1.00])   # 0 N
RED_F = np.array([1.00, 0.15, 0.10])    # force_scale N
GOLD = np.array([1.00, 0.78, 0.10])     # normal arrows


@register_gen("contact_dots")
class ContactDots:
    """Force-colored contact spheres (probe4 small-dot recipe)."""

    @staticmethod
    def validate(params, label="contact_dots"):
        _check_keys(params, ("r", "force_scale", "contact_select"), label)
        return {"r": _pos_float(params, "r", 0.0035, label),
                "force_scale": _pos_float(params, "force_scale", 1.2, label),
                "contact_select": _norm_contact_select(params, label)}

    @staticmethod
    def generate(params, target, ctx: DecorContext):
        sel = params["contact_select"]
        if sel is None and target != "*":
            sel = {"body_prefix": target}
        idx = ctx.contacts_where(sel)
        blue, red = BLUE_F, RED_F
        scale = params["force_scale"]
        out = []
        for i in idx:
            f = float(np.linalg.norm(ctx.contact_force[i]))
            t = float(np.clip(f / scale, 0, 1))
            col = blue * (1 - t) + red * t
            p = np.asarray(ctx.contact_pos[i], dtype=np.float64)
            out.append({"type": "capsule", "from": p, "to": p + 1e-5,
                        "radius": params["r"],
                        "rgba": np.array([col[0], col[1], col[2], 1.0])})
        return out


@register_gen("contact_normals")
class ContactNormals:
    """One thin arrow per contact along the contact-frame normal."""

    @staticmethod
    def validate(params, label="contact_normals"):
        _check_keys(params, ("len", "contact_select"), label)
        return {"len": _pos_float(params, "len", 0.02, label),
                "contact_select": _norm_contact_select(params, label)}

    @staticmethod
    def generate(params, target, ctx: DecorContext):
        sel = params["contact_select"]
        if sel is None and target != "*":
            sel = {"body_prefix": target}
        idx = ctx.contacts_where(sel)
        out = []
        for i in idx:
            p = np.asarray(ctx.contact_pos[i], dtype=np.float64)
            n = np.asarray(ctx.contact_normal[i], dtype=np.float64)
            out.append({"type": "arrow", "from": p, "to": p + n * params["len"],
                        "radius": 0.0022,
                        "rgba": np.array([GOLD[0], GOLD[1], GOLD[2], 1.0])})
        return out


@register_gen("contact_mjvis")
class ContactMjvis:
    """mjVIS_CONTACTPOINT/FORCE scene-option toggle (the flags simple tier)."""

    @staticmethod
    def validate(params, label="contact_mjvis"):
        _check_keys(params, (), label)
        return {}

    @staticmethod
    def generate(params, target, ctx: DecorContext):
        return [{"option_flags": {"mjVIS_CONTACTPOINT": True,
                                  "mjVIS_CONTACTFORCE": True}}]
