import pytest
import torch
import torch.nn as nn

from rlinf.models.embodiment.flow_policy_taco.flow_taco_policy import (
    FlowPolicyTacoForRL,
)


class _TinyFlowPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_dim = 2
        self.Tp = 3
        self.To = 1
        self.obs_mode = "pc_qpos"
        self._param = nn.Parameter(torch.zeros(()))

    def forward(self, x_t, tau, obs_norm):
        return torch.zeros_like(x_t)


def _obs(batch_size: int = 4):
    return {
        "qpos": torch.zeros(batch_size, 1, 2),
        "pointcloud": torch.zeros(batch_size, 1, 1, 3),
    }


def test_flow_taco_can_fix_single_noise_injection_index():
    model = FlowPolicyTacoForRL(
        policy=_TinyFlowPolicy(),
        normalizer=nn.Identity(),
        num_action_chunks=2,
        action_env_dim=2,
        num_denoise_steps=5,
        noise_method="flow_sde",
        fix_noise_index=True,
        noise_index=2,
        add_value_head=False,
    )

    out = model.sample_actions(_obs(), mode="train", compute_values=False)

    assert out["denoise_inds"].shape == (4, 5)
    assert torch.equal(out["denoise_inds"], torch.full((4, 5), 2))


def test_flow_taco_rejects_invalid_fixed_noise_index():
    model = FlowPolicyTacoForRL(
        policy=_TinyFlowPolicy(),
        normalizer=nn.Identity(),
        num_action_chunks=2,
        action_env_dim=2,
        num_denoise_steps=5,
        noise_method="flow_sde",
        fix_noise_index=True,
        noise_index=5,
        add_value_head=False,
    )

    with pytest.raises(ValueError, match="noise_index"):
        model.sample_actions(_obs(), mode="train", compute_values=False)
