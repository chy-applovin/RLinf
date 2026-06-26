# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the critic-free ``ppo_return_as_adv`` advantage estimator.

The ``PPO_with_return_as_adv`` algorithm is identical to PPO except that the
per-step advantage is the discounted Monte-Carlo return-to-go ``adv_t = G_t``
and no value function is involved. These tests check the return-to-go math
(discounting + episode-boundary resets) and that the embodied dispatch keeps
the per-step structure (i.e. it does not collapse rewards into a single
per-trajectory score the way GRPO-style estimators do) and needs no values.
"""

import torch

from rlinf.algorithms.advantages import compute_return_as_advantages_and_returns
from rlinf.algorithms.registry import ADV_REGISTRY, calculate_adv_and_returns


def test_ppo_return_as_adv_is_registered():
    assert "ppo_return_as_adv" in ADV_REGISTRY


def test_return_to_go_discounting_no_boundary():
    gamma = 0.9
    rewards = torch.tensor([[1.0], [2.0], [3.0]])  # [T=3, bsz=1]
    dones = torch.zeros(4, 1, dtype=torch.bool)
    loss_mask = torch.ones_like(rewards).bool()

    advantages, returns = compute_return_as_advantages_and_returns(
        rewards=rewards,
        gamma=gamma,
        dones=dones,
        loss_mask=loss_mask,
        normalize_advantages=False,
    )

    # G2 = 3
    # G1 = 2 + 0.9 * 3            = 4.7
    # G0 = 1 + 0.9 * 4.7         = 5.23
    expected = torch.tensor([[5.23], [4.7], [3.0]])
    assert torch.allclose(returns, expected, atol=1e-5)
    # Without normalization the advantage is exactly the return-to-go.
    assert torch.allclose(advantages, expected, atol=1e-5)


def test_return_to_go_resets_at_episode_boundary():
    gamma = 0.9
    rewards = torch.tensor([[1.0], [2.0], [3.0]])  # [T=3, bsz=1]
    # dones[step + 1] cuts the return after `step`. dones[2]=True => episode ends
    # at step 1, and step 2 begins a fresh episode.
    dones = torch.tensor([[False], [False], [True], [False]])
    loss_mask = torch.ones_like(rewards).bool()

    _, returns = compute_return_as_advantages_and_returns(
        rewards=rewards,
        gamma=gamma,
        dones=dones,
        loss_mask=loss_mask,
        normalize_advantages=False,
    )

    # step 2 (new episode): G2 = 3
    # step 1 (episode end): G1 = 2 (no bootstrap across boundary)
    # step 0:               G0 = 1 + 0.9 * 2 = 2.8
    expected = torch.tensor([[2.8], [2.0], [3.0]])
    assert torch.allclose(returns, expected, atol=1e-5)


def test_normalization_whitens_advantages_only():
    gamma = 1.0
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    dones = torch.zeros(4, 1, dtype=torch.bool)
    loss_mask = torch.ones_like(rewards).bool()

    advantages, returns = compute_return_as_advantages_and_returns(
        rewards=rewards,
        gamma=gamma,
        dones=dones,
        loss_mask=loss_mask,
        normalize_advantages=True,
    )

    # Returns stay raw; advantages are whitened over the valid mask.
    raw = torch.tensor([[6.0], [5.0], [3.0]])  # gamma=1 cumulative return-to-go
    assert torch.allclose(returns, raw, atol=1e-5)
    valid = advantages[loss_mask]
    assert abs(valid.mean().item()) < 1e-5


def test_embodied_dispatch_keeps_per_step_returns_without_values():
    """End-to-end through ``calculate_adv_and_returns`` (embodied, chunk_level).

    Asserts: (a) it runs with values=None (no critic), (b) advantages vary across
    time within a trajectory (proving rewards were NOT collapsed to a single
    per-trajectory score), and (c) output shapes match the chunk layout.
    """
    gamma = 0.9
    num_chunk, bsz, chunk_size = 3, 2, 1
    rewards = torch.zeros(num_chunk, bsz, chunk_size)
    rewards[:, 0, 0] = torch.tensor([1.0, 2.0, 3.0])
    rewards[:, 1, 0] = torch.tensor([0.5, 0.5, 0.5])
    dones = torch.zeros(num_chunk + 1, bsz, chunk_size, dtype=torch.bool)
    loss_mask = torch.ones(num_chunk, bsz, chunk_size, dtype=torch.bool)

    out = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="ppo_return_as_adv",
        reward_type="chunk_level",
        rewards=rewards,
        dones=dones,
        values=None,  # critic-free: must not be required
        gamma=gamma,
        gae_lambda=0.95,
        group_size=1,
        loss_mask=loss_mask,
        loss_mask_sum=None,
        normalize_advantages=False,
    )

    advantages = out["advantages"]
    returns = out["returns"]
    assert advantages.shape == (num_chunk, bsz, chunk_size)
    assert returns.shape == (num_chunk, bsz, chunk_size)

    # env 0: [1, 2, 3] -> [5.23, 4.7, 3.0]; env 1: [0.5]*3 -> [1.355, 0.95, 0.5]
    expected = torch.tensor([[5.23, 1.355], [4.7, 0.95], [3.0, 0.5]]).reshape(
        num_chunk, bsz, chunk_size
    )
    assert torch.allclose(advantages, expected, atol=1e-5)

    # Per-step structure preserved: the three timesteps of env 0 are all distinct.
    env0 = advantages[:, 0, 0]
    assert env0[0] != env0[1] and env0[1] != env0[2]
