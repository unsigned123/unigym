"""Canonical backend -> executor binding (builtin profile + custom bindings).

The builtin profile keeps the legacy executor *names* (``mujoco`` host,
``isaaclab`` docker) but binds every canonical backend key: ``mjwarp`` rides
the host mujoco executor and all three IsaacLab identities ride the docker
one.  Custom ``backends:`` keys are canonical-only.
"""

from __future__ import annotations

import pytest

from unigym.capabilities.traits import CANONICAL_BACKENDS
from unigym.executors import executor_for

_HOST = "/usr/bin/python3"

_ISAACLAB_FAMILY = ("isaaclab2", "isaaclab3_physx", "isaaclab3_newton")


def _base_cfg() -> dict:
    return {"venv_python": _HOST}


def test_every_canonical_backend_has_a_builtin_executor() -> None:
    cfg = _base_cfg()

    executors = {backend: executor_for(cfg, backend) for backend in CANONICAL_BACKENDS}

    assert set(executors) == set(CANONICAL_BACKENDS)
    for executor in executors.values():
        assert executor.kind in ("host", "docker")


def test_mujoco_family_uses_the_host_mujoco_executor() -> None:
    cfg = _base_cfg()

    mujoco = executor_for(cfg, "mujoco")
    mjwarp = executor_for(cfg, "mjwarp")

    assert (mujoco.name, mujoco.kind) == ("mujoco", "host")
    assert (mjwarp.name, mjwarp.kind) == ("mujoco", "host")
    assert mujoco.params["python"] == _HOST


@pytest.mark.parametrize("backend", _ISAACLAB_FAMILY)
def test_isaaclab_identities_use_the_docker_executor(backend: str) -> None:
    cfg = _base_cfg()

    executor = executor_for(cfg, backend)

    assert (executor.name, executor.kind) == ("isaaclab", "docker")


@pytest.mark.parametrize("backend", ["isaaclab", "isaaclab3", "not-a-backend"])
def test_custom_backends_reject_non_canonical_keys(backend: str) -> None:
    cfg = {**_base_cfg(), "backends": {backend: "mujoco"}}

    with pytest.raises(ValueError) as excinfo:
        executor_for(cfg, "mujoco")

    message = str(excinfo.value)
    assert "canonical" in message
    assert "aliases removed" in message
    assert "isaaclab3_physx" in message


def test_custom_backends_accept_a_canonical_key() -> None:
    cfg = {
        **_base_cfg(),
        "executors": {"hostpy": {"kind": "host", "python": _HOST}},
        "backends": {"isaaclab2": "hostpy"},
    }

    executor = executor_for(cfg, "isaaclab2")

    assert (executor.name, executor.kind) == ("hostpy", "host")


def test_custom_backends_missing_binding_still_fails_fast() -> None:
    cfg = {
        **_base_cfg(),
        "executors": {"hostpy": {"kind": "host", "python": _HOST}},
        "backends": {"mujoco": "hostpy"},
    }

    with pytest.raises(ValueError, match="no executor bound"):
        executor_for(cfg, "isaaclab3_physx")
