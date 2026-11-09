"""Adapter registry. Imports must stay torch-only for local unit tests."""

from __future__ import annotations

import inspect
from collections.abc import Mapping

from unigym._typing import JsonValue

from .base import Adapter, RawObsRequest  # noqa: F401  (package-level re-export)
from .params import (  # noqa: F401  (package-level re-export)
    REQUIRED,
    adapter_params_registry,
    adapter_params_schema,
    declared_params,
    check_spawn_values,
    validate_effective_section,
)
from .manual import ManualAdapter

ADAPTER_REGISTRY = {
    # The zero-config / explicit-loop adapter: no policy forward; the
    # external driver feeds actions through ExternalRunner.step(actions).
    "manual": ManualAdapter,
}


class AdapterParamsError(ValueError):
    """Spawn-params channel contract violation (teaching payload).

    ``.doc`` carries the structured {error, legal, example, hint} dict the
    faces return verbatim -- the caller corrects itself without reading
    source (the teaching-error style).
    """

    def __init__(self, message: str, doc: dict[str, object]):
        super().__init__(message)
        self.doc = doc


def _teaching_doc(message: str, *, legal: object, example: object, hint: str) -> dict[str, object]:
    return {
        "ok": False,
        "error_code": "bad_adapter_params",
        "error": message,
        "legal": legal,
        "example": example,
        "hint": hint,
    }


#: the one scalar-type whitelist for spawn params (JSON-simple values)
_SCALARS = (bool, int, float, str)
_SPAWN_EXAMPLE = {"name": "manual", "control_dt": 0.02}
_SPAWN_HINT = (
    "spawn params overlay ONLY the yaml 'adapter:' section "
    "(never env/benchmark/cameras); values are scalars or "
    "scalar lists"
)


def validate_spawn_params(params: object) -> dict[str, JsonValue]:
    """Value-domain gate for the spawn-params channel (fail-fast).

    Legal: ``None`` / bool / int / float / str, or a flat list of those.
    Rejected: nested dicts and lists carrying dicts/lists -- the channel
    is a shallow overlay of scalar knobs, not a second config system.

    Returns *params* (as a plain dict) on success; raises
    :class:`AdapterParamsError` with the teaching payload otherwise.
    """
    if params is None:
        return {}
    if isinstance(params, str):
        raise AdapterParamsError(
            f"adapter_params must be an object, got a string: " f"{str(params)[:120]!r}",
            _teaching_doc(
                "send adapter_params as the structured JSON object, never " "as a stringified one",
                legal="object of scalar / scalar-list values",
                example=_SPAWN_EXAMPLE,
                hint=_SPAWN_HINT,
            ),
        )
    if not isinstance(params, dict):
        raise AdapterParamsError(
            f"adapter_params must be an object, got " f"{type(params).__name__}",
            _teaching_doc(
                "adapter_params must be a JSON object",
                legal="object of scalar / scalar-list values",
                example=_SPAWN_EXAMPLE,
                hint=_SPAWN_HINT,
            ),
        )
    for key, value in params.items():
        if value is None or isinstance(value, _SCALARS):
            continue
        if isinstance(value, (list, tuple)):
            bad = [v for v in value if not (v is None or isinstance(v, _SCALARS))]
            if bad:
                raise AdapterParamsError(
                    f"adapter_params[{key!r}]: lists must carry scalars "
                    f"only, found {type(bad[0]).__name__} entries",
                    _teaching_doc(
                        f"adapter_params[{key!r}] is not a scalar list",
                        legal="flat list of scalars (bool/int/float/str)",
                        example={key: [1, 2, 3]},
                        hint=_SPAWN_HINT,
                    ),
                )
            continue
        raise AdapterParamsError(
            f"adapter_params[{key!r}]: nested structures are not "
            f"accepted ({type(value).__name__})",
            _teaching_doc(
                f"adapter_params[{key!r}] must be a scalar or a flat "
                f"scalar list -- nested dicts are rejected (the channel "
                f"overlays the adapter section shallowly)",
                legal="scalar (bool/int/float/str/null) or flat scalar " "list",
                example=_SPAWN_EXAMPLE,
                hint=_SPAWN_HINT,
            ),
        )
    return dict(params)


def _legal_keys(name: str) -> set[str] | None:
    """Constructor keyword set of one registry adapter (None = **kwargs
    present, nothing to check against)."""
    sig = inspect.signature(ADAPTER_REGISTRY[name].__init__)
    params = list(sig.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        return None
    return {p.name for p in params} - {"self", "num_envs", "device"}


def apply_spawn_params(
    section: Mapping[str, JsonValue],
    params: object,
) -> tuple[dict[str, JsonValue], list[str]]:
    """Overlay spawn adapter_params onto the yaml adapter section (shallow).

    Priority chain: defaults < config yaml < spawn adapter_params (the
    most specific, per-spawn layer).  SCOPE LOCK: the overlay merges into
    the adapter SECTION ONLY -- env/benchmark/cameras/fuse-schedule are
    architecturally unreachable from here (the caller hands in one
    section, not the config tree).

    Semantics:
    * no ``name`` in *params* (or the config's own name) -> shallow merge:
      ``{**section, **params}``;
    * ``name`` OVERRIDDEN to a different adapter -> the section is
      REPLACED by the spawn params: the previous adapter's constructor
      keys would poison the new one (stale ``kinematics``/``control_port``-style
      keys are not the new adapter's business);
    * every *params* key must be a legal constructor keyword of the FINAL
      adapter (``name`` and the injected ``body_parts`` handled
      separately) -- unknown keys fail-fast with the teaching payload
      listing the legal set + an example;
    * DECLARATION-FIRST (the PARAMS convention): an adapter WITH a
      ``PARAMS`` table accepts only its declared keys (types gated too)
      -- constructor kwargs outside the declaration are config-yaml
      surface, not spawn surface; undeclared adapters keep the pure
      signature inference.

    Returns ``(effective_section, overridden_keys)``.
    """
    params = validate_spawn_params(params)
    if not params:
        return dict(section or {}), []
    base = dict(section or {})
    switching = "name" in params and params["name"] != base.get("name")
    if switching:
        merged = dict(params)
    else:
        merged = {**base, **params}
    final_name = merged.get("name")
    if final_name is None:
        raise AdapterParamsError(
            "adapter section has no 'name' (spawn params must carry it)",
            _teaching_doc(
                "the adapter section carries no 'name'",
                legal=f"one of {sorted(ADAPTER_REGISTRY)}",
                example=_SPAWN_EXAMPLE,
                hint="spawn params may set 'name' to switch the " "adapter",
            ),
        )
    if final_name not in ADAPTER_REGISTRY:
        raise AdapterParamsError(
            f"unknown adapter name {final_name!r}",
            _teaching_doc(
                f"unknown adapter name {final_name!r}",
                legal=f"one of {sorted(ADAPTER_REGISTRY)}",
                example=_SPAWN_EXAMPLE,
                hint="pick a registered adapter name",
            ),
        )
    declared = declared_params(ADAPTER_REGISTRY[final_name])
    if declared is not None:
        check_spawn_values(final_name, params)
        legal = set(declared)
    else:
        legal = _legal_keys(final_name)
    if legal is not None:
        unknown = set(params) - legal - {"name"}
        if "body_parts" in params:
            raise AdapterParamsError(
                "adapter_params cannot override 'body_parts' (injected "
                "from the yaml robot.body_parts section)",
                _teaching_doc(
                    "'body_parts' is not a spawn-param surface",
                    legal=sorted(legal | {"name"}),
                    example=_SPAWN_EXAMPLE,
                    hint="body parts come from the config robot section",
                ),
            )
        if unknown:
            raise AdapterParamsError(
                f"unknown adapter_params keys for '{final_name}': " f"{sorted(unknown)}",
                _teaching_doc(
                    f"unknown keys {sorted(unknown)} for adapter " f"'{final_name}'",
                    legal=sorted(legal | {"name"}),
                    example=_SPAWN_EXAMPLE,
                    hint="check spelling against the legal-key list; "
                    "config yaml keys stay legal too",
                ),
            )
    if switching:
        # the whole section was replaced: every spawn key is an override
        overridden = sorted(params)
    else:
        overridden = sorted(set(params) & set(base))
    return merged, overridden


def build_adapter(
    section: dict, num_envs: int, device: str, body_parts: dict | None = None
) -> Adapter:
    """Instantiate an adapter from a config dict entry ``{name: ..., **params}``.

    ``body_parts`` (from the yaml ``robot.body_parts`` section) is passed
    through to the adapter constructor when provided, so the adapter can
    use configurable joint/foot/hip names instead of hardcoded constants.

    DECLARED adapters (a ``PARAMS`` table on the class) are validated
    here on the EFFECTIVE section (yaml merged with spawn params):
    required present, declared values type-legal, every key either
    declared or a constructor keyword -- all teaching-style errors.
    Undeclared adapters keep the legacy behavior (the constructor's own
    TypeError is the unknown-key gate).
    """
    name = (section or {}).get("name")
    if name not in ADAPTER_REGISTRY:
        raise AdapterParamsError(
            f"unknown adapter name {name!r}",
            _teaching_doc(
                f"unknown adapter name {name!r}",
                legal=f"one of {sorted(ADAPTER_REGISTRY)}",
                example=_SPAWN_EXAMPLE,
                hint="pick a registered adapter name",
            ),
        )
    validate_effective_section(name, section)
    params = {k: v for k, v in section.items() if k != "name"}
    if body_parts is not None:
        params["body_parts"] = body_parts
    adapter = ADAPTER_REGISTRY[name](num_envs=num_envs, device=device, **params)
    # registry identity: the build-time controls[] partition and the
    # runner's group validation resolve ownership by this name (never by
    # joint-order / action-face inference).
    adapter._unigym_name = name
    return adapter
