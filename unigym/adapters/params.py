"""Adapter PARAMS declaration: schema face + registry scan + validation.

Convention (the skills ``PARAMS`` shape): an adapter class may declare

    PARAMS = {name: (type, default, doc)}

and the spawn-params channel + build path then validate against that
declaration (types / required / unknown keys, teaching errors).  A
``default`` of the :data:`REQUIRED` sentinel marks a mandatory param.
Adapters WITHOUT a declaration keep the legacy behavior exactly: legal
keys are inferred from the constructor signature at overlay time, and
the WebUI serves them a raw-JSON box instead of per-param rows.

The declaration is the SPAWN-PARAM SURFACE, not the full constructor:
wiring/engine-handle keys (kinematics / ports / daemon_url / injected
transports) stay config-yaml-only -- they are construction details, not
per-spawn knobs.

This module must stay importable without torch (pure inspection), so
the registry scan imports the package lazily.
"""

from __future__ import annotations

import inspect


class _Required:
    """Sentinel for mandatory PARAMS defaults (``repr`` = the docs face)."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED = _Required()


#: the declared-type -> JSON-type table (skills ``params_schema`` shape;
#: one table so the endpoint / webui / session face can never drift)
_JSON_TYPES = {
    str: "string",
    float: "number",
    int: "integer",
    bool: "boolean",
    list: "array",
    dict: "object",
}

#: keys every adapter receives from the builder, never a spawn surface
_INJECTED = {"num_envs", "device", "name"}


def declared_params(cls) -> dict | None:
    """The adapter's PARAMS declaration, or None (= legacy signature
    inference)."""
    params = getattr(cls, "PARAMS", None)
    return dict(params) if params else None


def adapter_params_schema(cls) -> dict:
    """JSON-schema face of one adapter's declaration.

    ``{"declared": bool, "params": {name: {type, default?, required,
    doc}}}`` -- the payload GET /api/adapters/params serves per adapter
    (undeclared adapters carry ``declared: false`` and an empty param
    table)."""
    params = declared_params(cls)
    if params is None:
        return {"declared": False, "params": {}}
    props: dict = {}
    for pname, (typ, default, doc) in params.items():
        prop = {
            "type": _JSON_TYPES.get(typ, "string"),
            "required": default is REQUIRED,
            "doc": str(doc),
        }
        if default is not REQUIRED:
            prop["default"] = default
        props[pname] = prop
    return {"declared": True, "params": props}


def adapter_params_registry() -> dict:
    """Whole-registry scan: ``{name: adapter_params_schema(cls)}``
    (sorted; the /api/adapters/params payload and the session-face
    schema block share this one face)."""
    from . import ADAPTER_REGISTRY

    return {name: adapter_params_schema(cls) for name, cls in sorted(ADAPTER_REGISTRY.items())}


# --------------------------------------------------------------------------- #
# validation (teaching errors; the faces return .doc verbatim)
# --------------------------------------------------------------------------- #
def _fail(message: str, *, legal, example, hint: str) -> None:
    from . import AdapterParamsError, _teaching_doc

    raise AdapterParamsError(
        message, _teaching_doc(message, legal=legal, example=example, hint=hint)
    )


_SPAWN_EXAMPLE = {"name": "manual", "control_dt": 0.02}


def _check_value(adapter: str, key: str, typ, value) -> None:
    """Type gate ONE declared value (None = 'use the default', always
    legal; json ints coerce to float params, skills-style; bools never
    pass for int params -- isinstance(True, int) is a Python trap)."""
    if value is None:
        return
    if typ is float and isinstance(value, int) and not isinstance(value, bool):
        return  # json ints are legal floats
    if typ is int and isinstance(value, bool):
        _fail(
            f"adapter_params[{key!r}] for '{adapter}': bool is not an "
            f"int -- send a JSON integer",
            legal=sorted(_JSON_TYPES.values()),
            example={key: 4},
            hint=f"param {key!r} is declared int " f"(true/false are not integers)",
        )
    if isinstance(value, typ):
        return
    _fail(
        f"adapter_params[{key!r}] for '{adapter}': must be "
        f"{_JSON_TYPES.get(typ, typ)}, got {type(value).__name__}",
        legal=sorted(_JSON_TYPES.values()),
        example={key: value},
        hint=f"check the declaration via GET /api/adapters/params "
        f"(param {key!r} of adapter '{adapter}')",
    )


def check_spawn_values(name: str, params: dict) -> None:
    """Type gate the spawn overlay against the adapter's declaration
    (declaration-first: an adapter WITH a PARAMS table accepts only
    declared keys -- constructor kwargs not in the declaration are
    config-yaml surface, not spawn surface)."""
    from . import ADAPTER_REGISTRY

    declared = declared_params(ADAPTER_REGISTRY[name])
    if declared is None:
        return
    for key, value in params.items():
        if key == "name":
            continue
        if key not in declared:
            _fail(
                f"unknown adapter_params key {key!r} for '{name}' "
                f"(declared: {sorted(declared)})",
                legal=sorted(set(declared) | {"name"}),
                example=_SPAWN_EXAMPLE,
                hint="only declared params ride the spawn channel; "
                "constructor-only keys belong in the config yaml "
                "adapter section",
            )
        _check_value(name, key, declared[key][0], value)


def validate_effective_section(name: str, section: dict) -> None:
    """Build-time gate for a DECLARED adapter's effective section
    (config yaml merged with spawn params): required present, declared
    values type-legal, unknown keys rejected with the full legal set.
    Legacy (undeclared) adapters skip this entirely."""
    from . import ADAPTER_REGISTRY

    cls = ADAPTER_REGISTRY[name]
    declared = declared_params(cls)
    if declared is None:
        return
    params = {k: v for k, v in (section or {}).items() if k != "name"}
    missing = sorted(
        k for k, spec in declared.items() if spec[1] is REQUIRED and params.get(k) is None
    )
    if missing:
        docs = "; ".join(f"{k}: {declared[k][2]}" for k in missing)
        _fail(
            f"adapter '{name}' is missing required param(s) " f"{missing}: {docs}",
            legal=sorted(declared),
            example=dict(_SPAWN_EXAMPLE, **{k: "<value>" for k in missing}),
            hint="required params carry no default -- supply them in "
            "the yaml adapter section or via spawn adapter_params",
        )
    sig_params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    for key, value in params.items():
        if key in declared:
            _check_value(name, key, declared[key][0], value)
    unknown = sorted(set(params) - set(declared) - _INJECTED - sig_params)
    if unknown:
        _fail(
            f"unknown key(s) {unknown} for adapter '{name}' "
            f"(declared: {sorted(declared)}; "
            f"constructor: {sorted(sig_params)})",
            legal=sorted(set(declared) | sig_params | {"name"}),
            example=_SPAWN_EXAMPLE,
            hint="every yaml/spawn key must be a declared param or a " "constructor keyword",
        )
