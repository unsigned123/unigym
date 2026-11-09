"""Robot-agnostic naming helpers (pure string logic, unit-testable).

Same style as :mod:`unigym.env.episode` /
:mod:`unigym.backends.isaaclab2.init_pose`: pure string / torch-free
functions, consumed by collect on the server side, unit-testable locally
without an isaaclab environment.

* :func:`foot_metric_label` — foot body name -> ``duty_<label>`` metric
  label: take the first segment of the foot name -- ``FL_foot`` -> ``FL``;
  ``left_ankle_roll_link`` -> ``left``.
* :func:`contact_sensor_prim_path` — ContactSensorCfg.prim_path builder.
  Default regex ``.*_foot``; when the profile declares
  ``body_parts["feet"]``, an explicit-set regex is used instead.
"""

from __future__ import annotations

from typing import Optional, Sequence

ENV_REGEX_NS = "{ENV_REGEX_NS}"
DEFAULT_CONTACT_PRIM_PATH = f"{ENV_REGEX_NS}/Robot/.*_foot"


def foot_metric_label(foot_name: str) -> str:
    """Short label for per-foot step metrics (``duty_<label>``).

    "FL_foot" -> "FL"; "left_ankle_roll_link" -> "left".  Unconventional
    names starting with a digit fall back to the full name, so key names
    never collide and are never empty.
    """
    head = foot_name.split("_")[0]
    return head if head and not head[0].isdigit() else foot_name


def contact_sensor_prim_path(feet: Optional[Sequence[str]]) -> str:
    """prim_path for the scene ContactSensorCfg.

    Empty/None ``feet`` keeps the legacy regex (behavior unchanged for the
    existing robot profiles); otherwise an explicit-set regex precisely
    matches the declared foot bodies.
    """
    if not feet:
        return DEFAULT_CONTACT_PRIM_PATH
    return f"{ENV_REGEX_NS}/Robot/({'|'.join(feet)})"
