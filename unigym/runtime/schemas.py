"""Control-plane JSON Schemas (ux-11, feat/agent-debug-ux).

One JSON Schema (draft 2020-12 subset) per control-plane body:
``control`` (POST /control), ``console`` (POST /console envelope),
``inspect`` / ``save_npz`` (the two bare-payload endpoints).  Served in
the GET /info document under ``"schemas"`` with a ``SCHEMA_VERSION``
counter (bump on BREAKING schema changes; additive changes do not bump).

Design rules:

* stdlib-only pure dicts -- importable from the bridge's HTTP thread
  before torch is warm;
* the schemas are the machine-checkable twin of ``unigym/stream/__init__.py``
  (control) and ``unigym/console/__init__.py::parse_console`` (console)
  -- those parsers remain the ENFORCERS; a schema mismatch with a parser
  is a bug caught by tests/test_schemas.py;
* semantic context the schema cannot express is stated in
  ``description`` fields (e.g. camera must be an INSTANTIATED name;
  pos+lookat pair -- expressed with if/then -- still requires a fixed
  rig, checked engine-side).
"""

from __future__ import annotations

from unigym._typing import JsonValue

__all__ = [
    "SCHEMA_VERSION",
    "CONTROL_SCHEMA",
    "CONSOLE_SCHEMA",
    "INSPECT_SCHEMA",
    "SAVE_NPZ_SCHEMA",
]

SCHEMA_VERSION = 1

_DRAFT = "https://json-schema.org/draft/2020-12/schema"

_CAMERA: dict[str, JsonValue] = {
    "type": "string",
    "minLength": 1,
    "maxLength": 63,
    "description": "an INSTANTIATED camera name (GET /info -> " "cameras.instantiated)",
}
_NON_NEG_INT: dict[str, JsonValue] = {"type": "integer", "minimum": 0}
_NUMBER: dict[str, JsonValue] = {"type": "number"}
_VEC3: dict[str, JsonValue] = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 3,
    "maxItems": 3,
}

_BARE: dict[str, JsonValue] = {
    "type": "object",
    "additionalProperties": False,
    "description": "no keys beyond 'type'",
}

_CONTROL_SNAPSHOT: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "type": {"const": "snapshot"},
        "camera": _CAMERA,
        "env_id": _NON_NEG_INT,
        "pos": _VEC3,
        "lookat": _VEC3,
    },
    "additionalProperties": False,
    "allOf": [
        {"if": {"required": ["pos"]}, "then": {"required": ["lookat"]}},
        {"if": {"required": ["lookat"]}, "then": {"required": ["pos"]}},
    ],
    "description": "camera optional (default 'free', the debug fly "
    "camera); pos+lookat TOGETHER = transient world-pose "
    "override for one frame (ux-7; fixed/free rigs only, "
    "body/chase -> 400 + hint; the rig pose is restored "
    "after the frame)",
}

CONTROL_SCHEMA: dict[str, JsonValue] = {
    "$schema": _DRAFT,
    "$id": "unigym:control",
    "title": "POST /control body",
    "version": SCHEMA_VERSION,
    "type": "object",
    "required": ["type"],
    "oneOf": [
        dict(_BARE, properties={"type": {"const": "pause"}}),
        dict(_BARE, properties={"type": {"const": "resume"}}),
        dict(_BARE, properties={"type": {"const": "shutdown"}}),
        _CONTROL_SNAPSHOT,
        {
            "type": "object",
            "properties": {
                "type": {"const": "camera_set_properties"},
                "camera": _CAMERA,
                "env_id": _NON_NEG_INT,
                "pose_offset": _VEC3,
                "yaw_deg": _NUMBER,
                "pitch_deg": _NUMBER,
                "interval": {"type": "integer", "minimum": 1},
                "target_env": _NON_NEG_INT,
            },
            "required": ["camera"],
            "additionalProperties": False,
            "description": "live whitelist : pose_offset / "
            "yaw_deg / pitch_deg / target_env / interval; "
            "frozen keys (resolution, hfov, types, ...) "
            "are rejected -- restart required",
        },
        {
            "type": "object",
            "properties": {
                "type": {"const": "stream_bind"},
                "camera": _CAMERA,
                "env_id": _NON_NEG_INT,
            },
            "required": ["camera"],
            "additionalProperties": False,
            "description": "rebind the broadcast producer (WebRTC L2)",
        },
    ],
    "description": "200 {ok, req_id, type, result}; snapshot 200 = "
    "image/jpeg bytes; 202 = no ack in time; 4xx carries "
    "an executable 'hint' (ux-8)",
}

_CONSOLE_INSPECT: dict[str, JsonValue] = {
    "type": "object",
    "properties": {
        "type": {"const": "inspect"},
        "target": {"enum": ["summary", "obs", "extra"]},
        "key": {"type": "string", "minLength": 1},
    },
    "required": ["target"],
    "additionalProperties": False,
    "allOf": [
        {
            "if": {"properties": {"target": {"const": "obs"}}, "required": ["target"]},
            "then": {"required": ["key"]},
        },
        {
            "if": {"properties": {"target": {"const": "extra"}}, "required": ["target"]},
            "then": {"required": ["key"]},
        },
    ],
}

_CONSOLE_EVAL: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"type": {"const": "eval"}, "expr": {"type": "string", "minLength": 1}},
    "required": ["expr"],
    "additionalProperties": False,
    "description": "PASS-THROUGH eval (the old "
    "restricted namespace is gone -- full builtins, "
    "imports, assignments, multi-line statements all "
    "execute; injected convenience names np torch "
    "obs_cache extra bundle adapter env; the last "
    "expression's value is returned; errors land in "
    "result.error, the run never dies)",
}

CONSOLE_SCHEMA: dict[str, JsonValue] = {
    "$schema": _DRAFT,
    "$id": "unigym:console",
    "title": "POST /console body",
    "version": SCHEMA_VERSION,
    "type": "object",
    "required": ["type"],
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "type": {"const": "step"},
                "n": {"type": "integer", "minimum": 1, "maximum": 100000},
            },
            "additionalProperties": False,
            "description": "policy steps, PAUSED only (no unit field "
            "exists in this contract); running -> "
            "ack error 'pause first'",
        },
        _CONSOLE_INSPECT,
        _CONSOLE_EVAL,
        {
            "type": "object",
            "properties": {
                "type": {"const": "save_npz"},
                "source": {"enum": ["obs", "extra", "all"]},
                "keys": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "path": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
            "description": "result.npz_path = the dump (relative -> run " "output_dir)",
        },
    ],
    "$comment": "envelope keys are whitelisted per type by "
    "unigym.console.parse_console -- this schema is its "
    "machine-checkable twin",
}

INSPECT_SCHEMA: dict[str, JsonValue] = {
    "$schema": _DRAFT,
    "$id": "unigym:inspect",
    "title": "POST /inspect body (bare console-inspect payload)",
    "version": SCHEMA_VERSION,
    "type": "object",
    "required": ["target"],
    "oneOf": [
        {
            "type": "object",
            "properties": {"target": {"const": "summary"}},
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"target": {"const": "obs"}, "key": {"type": "string", "minLength": 1}},
            "required": ["key"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"target": {"const": "extra"}, "key": {"type": "string", "minLength": 1}},
            "required": ["key"],
            "additionalProperties": False,
        },
        dict(
            _CONSOLE_EVAL,
            properties={"target": {"const": "eval"}, "expr": {"type": "string", "minLength": 1}},
            required=["expr"],
        ),
    ],
}

SAVE_NPZ_SCHEMA: dict[str, JsonValue] = {
    "$schema": _DRAFT,
    "$id": "unigym:save_npz",
    "title": "POST /save_npz body (bare console save_npz payload)",
    "version": SCHEMA_VERSION,
    "type": "object",
    "properties": {
        "source": {"enum": ["obs", "extra", "all"]},
        "keys": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "path": {"type": "string", "minLength": 1},
    },
    "additionalProperties": False,
}
