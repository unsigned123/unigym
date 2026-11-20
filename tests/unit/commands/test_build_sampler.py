"""command: authoring — built-in registry and the external module factory."""

from __future__ import annotations

import pytest

from tests._support.fakes.command_sampler import RecordingSampler
from unigym.commands import build_sampler
from unigym.commands.fixed import FixedSampler

_FACTORY = "tests._support.fakes.command_sampler:make_sampler"
_NOT_A_SAMPLER = "tests._support.fakes.command_sampler:make_not_a_sampler"


def test_build_sampler_builtin_name_keeps_the_registry_path():
    """A built-in section still builds through COMMAND_REGISTRY."""
    sampler = build_sampler(
        {"name": "fixed", "dim": 3, "values": [0.0, 1.0, 2.0]},
        num_envs=2,
        device="cpu",
    )

    assert isinstance(sampler, FixedSampler)


def test_build_sampler_module_factory_receives_dim_device_envs_and_params():
    """command.module calls factory(dim, device, num_envs, **params)."""
    sampler = build_sampler(
        {"module": _FACTORY, "dim": 4, "params": {"seats": 6}},
        num_envs=2,
        device="cpu",
    )

    assert isinstance(sampler, RecordingSampler)
    assert sampler.dim == 4
    assert sampler.num_envs == 2
    assert sampler.device == "cpu"
    assert sampler.seats == 6


def test_build_sampler_module_params_default_to_empty():
    """params is optional; the factory only receives the declared arguments."""
    sampler = build_sampler({"module": _FACTORY, "dim": 1}, num_envs=1, device="cpu")

    assert sampler.seats == 1


@pytest.mark.parametrize(
    ("section", "error", "match"),
    [
        (
            {"name": "fixed", "module": _FACTORY, "dim": 1},
            ValueError,
            "'name'.*'module'",
        ),
        ({"dim": 1}, ValueError, "'name'.*'module'"),
        (
            {"module": _FACTORY, "dim": 1, "values": [0.0]},
            ValueError,
            "unknown field",
        ),
        (
            {"module": _FACTORY, "dim": 1, "params": [("seats", 6)]},
            TypeError,
            "command.params",
        ),
        ({"module": _FACTORY, "dim": "4"}, ValueError, "command.dim"),
        ({"module": _FACTORY, "dim": 0}, ValueError, "command.dim"),
        ({"module": "not_a_reference", "dim": 1}, ValueError, "command.module"),
        (
            {"module": "tests._support.fakes.command_sampler:missing", "dim": 1},
            ValueError,
            "missing",
        ),
        (
            {"module": "tests.unit.commands.no_such_module:make", "dim": 1},
            ValueError,
            "cannot import",
        ),
        ({"module": _NOT_A_SAMPLER, "dim": 1}, TypeError, "expected CommandSampler"),
    ],
    ids=[
        "both-name-and-module",
        "neither-name-nor-module",
        "unknown-field",
        "non-mapping-params",
        "non-int-dim",
        "non-positive-dim",
        "malformed-reference",
        "missing-attribute",
        "missing-module",
        "non-sampler-return",
    ],
)
def test_build_sampler_module_section_fails_fast(section, error, match):
    """Invalid module authoring names the field and the expected shape."""
    with pytest.raises(error, match=match):
        build_sampler(section, num_envs=1, device="cpu")
