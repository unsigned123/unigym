"""IsaacLab 2 Cartpole construction, contract surface and compiled model IR."""

from __future__ import annotations

import pytest
import torch

from unigym.env.backends.base import assert_backend_contract

pytestmark = pytest.mark.sim

ARTICULATION = "cartpole"


def test_env_constructs_and_satisfies_the_backend_contract(cartpole_env):
    """The real Kit/PhysX Cartpole env satisfies the live EnvBackend contract."""
    assert_backend_contract(cartpole_env)


def test_single_env_fixture_exposes_one_env_on_the_requested_device(cartpole_env):
    """The one-env fixture builds one env on the configured CUDA device."""
    assert cartpole_env.num_envs == 1
    assert torch.device(cartpole_env.device) == torch.device("cuda:0")


def test_model_ir_reports_the_cartpole_body_and_joint_topology(cartpole_env):
    """The compiled IR exposes the Cartpole asset's bodies and joints."""
    ir = cartpole_env.model_set.articulations[ARTICULATION]

    assert [body.name for body in ir.bodies] == ["slider", "cart", "pole"]
    assert [joint.name for joint in ir.joints] == ["slider_to_cart", "cart_to_pole"]


def test_two_env_fixture_exposes_two_envs(two_env):
    """The two-env fixture builds a batch of two envs."""
    assert two_env.num_envs == 2


def test_partitioned_isaaclab_rejects_contact_manager_before_env(simulation_app):
    """Partitioned IsaacLab keeps its pre-env contact-manager fail-fast."""
    from unigym.backends.isaaclab2.articulation_runtime import validate_options

    class _Adapter:
        def get_obs_requests(self):
            return []

    with pytest.raises(NotImplementedError, match="contact_manager"):
        validate_options(
            _Adapter(),
            events_cfg=None,
            recorder=None,
            prop_manager=None,
            camera_manager=None,
            contact_manager=object(),
            director_cfg=None,
            init_pose_cfg=None,
        )
