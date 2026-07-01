import torch

from rlinf.algorithms.registry import calculate_adv_and_returns


def test_raft_step_selects_top_reward_entries_and_masks_the_rest():
    rewards = torch.tensor(
        [
            [[1.0], [0.5]],
            [[3.0], [4.0]],
            [[2.0], [0.1]],
        ]
    )
    dones = torch.zeros(4, 2, 1, dtype=torch.bool)
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    out = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="raft_step",
        reward_type="chunk_level",
        rewards=rewards,
        dones=dones,
        values=None,
        group_size=1,
        loss_mask=loss_mask,
        loss_mask_sum=None,
        raft_type="top_k_perc_adv1",
        raft_top_k_percent=0.5,
    )

    expected = torch.tensor(
        [
            [[False], [False]],
            [[True], [True]],
            [[True], [False]],
        ]
    )
    assert torch.equal(out["loss_mask"], expected)
    assert torch.equal(out["advantages"], expected.float())
    assert "returns" not in out


def test_raft_episode_selects_top_return_episodes_and_keeps_all_their_steps():
    rewards = torch.tensor(
        [
            [[1.0], [0.5]],
            [[3.0], [4.0]],
            [[2.0], [0.1]],
        ]
    )
    dones = torch.zeros(4, 2, 1, dtype=torch.bool)
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)

    out = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="raft_episode",
        reward_type="chunk_level",
        rewards=rewards,
        dones=dones,
        values=None,
        group_size=1,
        loss_mask=loss_mask,
        loss_mask_sum=None,
        raft_type="top_k_perc_adv1",
        raft_top_k_percent=50,
    )

    # Episode returns are env0=6.0 and env1=4.6, so top 50% keeps env0 only.
    expected = torch.tensor(
        [
            [[True], [False]],
            [[True], [False]],
            [[True], [False]],
        ]
    )
    assert torch.equal(out["loss_mask"], expected)
    assert torch.equal(out["advantages"], expected.float())
    assert "returns" not in out
