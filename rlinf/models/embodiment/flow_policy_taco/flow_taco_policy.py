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

"""PPO-ready RL wrapper for the TACO flow-matching (rectified flow) policy.

The wrapped IL policy (``flow_policy.models.FlowMatchingTransformerPolicy``)
predicts a velocity field v_theta(a_tau, tau, obs) over a (Tp, 44) Allegro
qpos action chunk, with the rectified-flow convention

    a_tau = (1 - tau) * a0 + tau * a1,   a0 ~ N(0, I) (tau=0), a1 = data (tau=1).

RL fine-tuning follows the piRL recipe already used by RLinf's OpenPi0 model
(``rlinf/models/embodiment/openpi/openpi_action_model.py``):

* rollout: integrate the flow with Euler steps; at ONE randomly chosen
  denoising step inject stochastic noise (``flow_sde`` / ``flow_cps``) and
  record the per-element Gaussian log-prob of the realized transition. All
  other steps stay deterministic (``flow_ode``), which keeps the
  importance-sampling ratio well-defined for a single stochastic kernel.
* training: re-run the chosen denoising step from the stored ``chains`` /
  ``denoise_inds`` to recompute log-probs with gradients; PPO clipped loss +
  a value head trained with GAE returns.

Noise-method math is translated from OpenPi0's ``sample_mean_var_val`` to this
model's time convention via s := 1 - tau (s = remaining noise time, s=1 pure
noise, s=0 data):

    data_pred  = a + s * v
    noise_pred = a - (1 - s) * v
    flow_ode : mean = data_pred*(1-(s-d)) + noise_pred*(s-d)          == a + d*v
    flow_sde : sigma_i = eta * sqrt(s/(1-s));  std = sqrt(d) * sigma_i
               mean = data_pred*(1-(s-d)) + noise_pred*((s-d) - sigma_i^2*d/(2s))
    flow_cps : mean = data_pred*(1-(s-d)) + noise_pred*(s-d)*cos(pi*eta/2)
               std  = (s-d) * sin(pi*eta/2)

with d = 1/num_steps the Euler step. In eval mode every step is ``flow_ode``,
which reproduces the original IL policy's sampler exactly.

Everything (chains, log-probs) lives in the NORMALIZED action space; only the
env-facing actions are denormalized.
"""

from __future__ import annotations

import math
import random
from typing import Any, Literal

import torch
import torch.nn as nn
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.nested_dict_process import copy_dict_tensor

_NOISE_METHODS = ("flow_ode", "flow_sde", "flow_cps")


class FlowPolicyTacoForRL(nn.Module, BasePolicy):
    """RL wrapper: flow-matching transformer + normalizer + value head."""

    def __init__(
        self,
        policy: nn.Module,
        normalizer: nn.Module,
        *,
        num_action_chunks: int,
        action_env_dim: int = 44,
        num_denoise_steps: int = 16,
        noise_method: str = "flow_sde",
        noise_level: float = 0.5,
        normalize_obs: bool = True,
        add_value_head: bool = True,
        detach_critic_input: bool = True,
        value_head_hidden_sizes=(256, 128),
        safe_get_logprob: bool = False,
        ignore_last: bool = False,
    ):
        super().__init__()
        assert noise_method in _NOISE_METHODS, (
            f"noise_method must be one of {_NOISE_METHODS}, got {noise_method}"
        )
        self.policy = policy
        self.normalizer = normalizer
        # RL fine-tuning needs a deterministic policy forward: the PPO ratio
        # compares rollout-time and training-time log-probs of the SAME
        # transition, so IL-time dropout (p_drop=0.1) must be disabled or the
        # recomputed mean/std (and thus the ratio) would be stochastic.
        self._disable_dropout(self.policy)

        self.action_dim = int(policy.action_dim)  # model action dim (44)
        self.pred_horizon = int(policy.Tp)
        self.obs_horizon = int(policy.To)
        self.obs_mode = str(policy.obs_mode)
        assert self.obs_mode in ("pc_qpos", "pc2_qpos"), (
            f"Unsupported obs_mode {self.obs_mode}"
        )

        self.num_action_chunks = int(num_action_chunks)
        assert 0 < self.num_action_chunks <= self.pred_horizon, (
            f"num_action_chunks={num_action_chunks} must be in (0, Tp={self.pred_horizon}]"
        )
        self.action_env_dim = int(action_env_dim)
        self.num_denoise_steps = int(num_denoise_steps)
        self.noise_method = noise_method
        self.noise_level = float(noise_level)
        self.normalize_obs = bool(normalize_obs)
        self.safe_get_logprob = bool(safe_get_logprob)
        self.ignore_last = bool(ignore_last)
        self.detach_critic_input = bool(detach_critic_input)
        self.global_step = 0

        if add_value_head:
            # Name must contain "value_head": FSDPModelManager routes these
            # params to the optim.value_lr param group / critic warmup.
            n_emb = self.policy.cond_pos_emb.shape[-1]
            self.value_head = ValueHead(
                input_dim=n_emb,
                hidden_sizes=tuple(value_head_hidden_sizes),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

    # ------------------------------------------------------------ construction
    @classmethod
    def from_config(cls, cfg: DictConfig) -> "FlowPolicyTacoForRL":
        """Build from cfg.actor.model (model_path = flow-policy .pt ckpt).

        With ``random_init: True`` the checkpoint is used ONLY as the
        architecture + I/O spec: shape_meta / model hyperparameters / the
        dataset-fitted Normalizer stats (obs/action scaling is part of the
        input-output contract, not learned policy weights). The transformer's
        weights keep their fresh random initialization - RL from scratch.
        """
        from flow_policy.data.normalizer import Normalizer
        from flow_policy.models.transformer_policy import (
            FlowMatchingTransformerPolicy,
        )

        ckpt_path = cfg.model_path
        assert ckpt_path, "actor.model.model_path must point to a flow-policy .pt"
        random_init = bool(cfg.get("random_init", False))
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sm = state["shape_meta"]
        mcfg = {k: v for k, v in state["cfg"]["model"].items() if k != "_target_"}
        policy = FlowMatchingTransformerPolicy(shape_meta=sm, **mcfg)
        normalizer = Normalizer(
            sm["action_dim"],
            sm["obs_mode"],
            point_dim=sm.get("point_dim"),
            qpos_dim=sm.get("qpos_dim"),
        )
        if not random_init:
            policy.load_state_dict(state["model"])
        normalizer.load_state_dict(state["normalizer"])

        algo_cfg = state["cfg"].get("algo", {})
        num_denoise_steps = int(
            cfg.get("num_denoise_steps", 0)
            or algo_cfg.get("num_sample_steps", 16)
        )
        model = cls(
            policy=policy,
            normalizer=normalizer,
            num_action_chunks=int(cfg.num_action_chunks),
            action_env_dim=int(cfg.get("action_dim", 44)),
            num_denoise_steps=num_denoise_steps,
            noise_method=str(cfg.get("noise_method", "flow_sde")),
            noise_level=float(cfg.get("noise_level", 0.5)),
            normalize_obs=bool(algo_cfg.get("normalize_obs", True)),
            add_value_head=bool(cfg.get("add_value_head", True)),
            detach_critic_input=bool(cfg.get("detach_critic_input", True)),
            value_head_hidden_sizes=tuple(
                cfg.get("value_head_hidden_sizes", (256, 128))
            ),
            safe_get_logprob=bool(cfg.get("safe_get_logprob", False)),
            ignore_last=bool(cfg.get("ignore_last", False)),
        )
        return model

    def set_global_step(self, global_step: int) -> None:
        self.global_step = int(global_step)

    @staticmethod
    def _disable_dropout(module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
            elif isinstance(m, nn.MultiheadAttention):
                m.dropout = 0.0

    # -------------------------------------------------------------- obs handling
    def _build_normalized_obs(
        self, src: dict[str, torch.Tensor], device: torch.device
    ) -> dict[str, torch.Tensor]:
        """env/forward obs dict -> normalized policy obs dict (on device)."""

        def grab(key: str) -> torch.Tensor:
            value = src[key]
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            return value.to(device=device, dtype=torch.float32)

        states = grab("states")
        assert states.dim() == 3 and states.shape[1] == self.obs_horizon, (
            f"states must be (B, To={self.obs_horizon}, qpos), got {tuple(states.shape)}"
        )
        obs = {"pointcloud": grab("pointcloud"), "qpos": states}
        if self.obs_mode == "pc2_qpos":
            obs["tool_pointcloud"] = grab("tool_pointcloud")
        if self.normalize_obs:
            obs = self.normalizer.normalize_obs(obs)
        return obs

    # ----------------------------------------------------------------- denoising
    def _timesteps(self, device: torch.device) -> torch.Tensor:
        """Remaining-noise times s: linspace(1, 1/n, n) + [0]; s_i = 1 - tau_i."""
        n = self.num_denoise_steps
        ts = torch.linspace(1.0, 1.0 / n, n, device=device)
        return torch.cat([ts, torch.zeros(1, device=device)])

    def _denoise_mean_std(
        self,
        x_t: torch.Tensor,
        idx: torch.Tensor,
        obs_norm: dict[str, torch.Tensor],
        sample_method: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One Euler/SDE transition kernel N(mean, std^2) at denoise step idx.

        Args:
            x_t: (B, Tp, A) current normalized action chunk.
            idx: (B,) long denoise-step indices.
            obs_norm: normalized observation dict.
            sample_method: flow_ode | flow_sde | flow_cps.
        """
        device = x_t.device
        timesteps = self._timesteps(device)
        s = timesteps[idx]  # (B,) remaining-noise time
        delta = s - timesteps[idx + 1]  # (B,) Euler step size
        tau = 1.0 - s  # flow-policy convention

        v = self.policy(x_t, tau, obs_norm)  # (B, Tp, A) velocity toward data

        s_e = s[:, None, None].expand_as(x_t)
        d_e = delta[:, None, None].expand_as(x_t)
        data_pred = x_t + s_e * v
        noise_pred = x_t - (1.0 - s_e) * v

        if sample_method == "flow_ode":
            data_w = 1.0 - (s_e - d_e)
            noise_w = s_e - d_e
            std = torch.zeros_like(x_t)
        elif sample_method == "flow_sde":
            # sigma_i = eta * sqrt(s / (1 - s)); guard s == 1 with the next step.
            denom = torch.where(timesteps == 1.0, timesteps[1], timesteps)
            sigmas = self.noise_level * torch.sqrt(timesteps / (1.0 - denom))[:-1]
            sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
            data_w = 1.0 - (s_e - d_e)
            noise_w = (s_e - d_e) - sigma_i**2 * d_e / (2.0 * s_e)
            std = torch.sqrt(d_e) * sigma_i
        elif sample_method == "flow_cps":
            cos_term = math.cos(math.pi * self.noise_level / 2.0)
            sin_term = math.sin(math.pi * self.noise_level / 2.0)
            data_w = 1.0 - (s_e - d_e)
            noise_w = (s_e - d_e) * cos_term
            std = (s_e - d_e) * sin_term
        else:
            raise ValueError(f"Invalid noise method: {sample_method}")

        mean = data_pred * data_w + noise_pred * noise_w
        return mean, std

    def _gaussian_logprob(
        self, sample: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
    ) -> torch.Tensor:
        """Per-element Gaussian log-density; zero where sigma == 0 (ODE steps)."""
        if self.safe_get_logprob:
            return -torch.pow(sample - mu, 2)
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        log_prob = (
            -torch.log(sigma_safe)
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        )
        return torch.where(mask, torch.zeros_like(log_prob), log_prob)

    # ------------------------------------------------------------------- rollout
    @torch.no_grad()
    def sample_actions(
        self,
        obs_norm: dict[str, torch.Tensor],
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Full denoising rollout with single-step noise injection (train)."""
        device = next(self.parameters()).device
        bsize = obs_norm["qpos"].shape[0]
        n = self.num_denoise_steps

        x_t = torch.randn(
            bsize, self.pred_horizon, self.action_dim, device=device
        )
        chains = [x_t]
        log_probs = []

        if mode == "train":
            high = n - 2 if self.ignore_last else n - 1
            denoise_ind = random.randint(0, max(high, 0))
            denoise_inds = torch.full((bsize, n), denoise_ind, dtype=torch.long)
        else:
            denoise_inds = torch.full((bsize, n), -1, dtype=torch.long)

        for idx in range(n):
            sample_method = (
                self.noise_method if idx == int(denoise_inds[0, 0]) else "flow_ode"
            )
            idx_t = torch.full((bsize,), idx, device=device, dtype=torch.long)
            mean, std = self._denoise_mean_std(x_t, idx_t, obs_norm, sample_method)
            x_t = mean + torch.randn_like(x_t) * std
            log_probs.append(self._gaussian_logprob(x_t, mean, std))
            chains.append(x_t)

        chains = torch.stack(chains, dim=1)  # (B, n+1, Tp, A)
        log_probs = torch.stack(log_probs, dim=1)  # (B, n, Tp, A)
        # only the noisy step carries the policy's stochasticity
        log_probs = log_probs[torch.arange(bsize), denoise_inds[:, 0]]
        log_probs = log_probs[
            :, : self.num_action_chunks, : self.action_env_dim
        ]  # (B, chunk, A_env)

        if compute_values and hasattr(self, "value_head"):
            values = self.compute_values(obs_norm)[:, None]  # (B, 1)
        else:
            values = torch.zeros(bsize, 1, device=device)

        return {
            "actions": x_t,  # normalized (B, Tp, A)
            "chains": chains,
            "denoise_inds": denoise_inds,
            "prev_logprobs": log_probs,
            "prev_values": values,
        }

    def compute_values(self, obs_norm: dict[str, torch.Tensor]) -> torch.Tensor:
        """Value from mean-pooled observation tokens; (B,)."""
        feats = self.policy._encode_obs(obs_norm)  # (B, To*tokens, n_emb)
        if self.detach_critic_input:
            feats = feats.detach()
        pooled = feats.mean(dim=1).to(torch.float32)
        return self.value_head(pooled)[:, 0]

    # ------------------------------------------------------- BasePolicy interface
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(
            f"{type(self).__name__} does not support forward_type={forward_type}"
        )

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Rollout inference. Returns env actions + RL bookkeeping.

        env_obs: ``states`` (B, To, 44), ``pointcloud`` (B, To, K, 3) and, for
        pc2_qpos checkpoints, ``tool_pointcloud`` (B, To, K, 3).
        """
        device = next(self.parameters()).device
        obs_norm = self._build_normalized_obs(env_obs, device)

        outputs = self.sample_actions(obs_norm, mode=mode, compute_values=True)

        raw_chunk = self.normalizer.denormalize_action(outputs["actions"])
        actions = (
            raw_chunk[:, : self.num_action_chunks, : self.action_env_dim]
            .float()
            .cpu()
            .contiguous()
        )  # (B, chunk, 44) env-executable

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "action": actions.reshape(actions.shape[0], -1).contiguous(),
            "model_action": raw_chunk.reshape(raw_chunk.shape[0], -1)
            .float()
            .cpu()
            .contiguous(),
        }
        # store the raw observations needed to recompute log-probs in training
        obs_keys = ["states", "pointcloud"] + (
            ["tool_pointcloud"] if self.obs_mode == "pc2_qpos" else []
        )
        cloned_obs = copy_dict_tensor({k: env_obs[k] for k in obs_keys})
        forward_inputs.update(cloned_obs)

        result = {
            "prev_logprobs": outputs["prev_logprobs"].float().cpu(),
            "prev_values": outputs["prev_values"].float().cpu(),
            "forward_inputs": forward_inputs,
        }
        return actions, result

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Training forward: recompute log-probs (and values) with gradients."""
        compute_values = kwargs.get("compute_values", False)
        device = next(self.parameters()).device

        obs_norm = self._build_normalized_obs(forward_inputs, device)
        chains = forward_inputs["chains"].to(device=device, dtype=torch.float32)
        denoise_inds = forward_inputs["denoise_inds"].to(device).long()

        bsize = chains.shape[0]
        batch_idx = torch.arange(bsize, device=device)
        ind = denoise_inds[:, 0]
        x_prev = chains[batch_idx, ind]
        x_next = chains[batch_idx, ind + 1]

        mean, std = self._denoise_mean_std(x_prev, ind, obs_norm, self.noise_method)
        log_probs = self._gaussian_logprob(x_next, mean, std)
        log_probs = log_probs[
            :, : self.num_action_chunks, : self.action_env_dim
        ].float()

        if compute_values and hasattr(self, "value_head"):
            values = self.compute_values(obs_norm).float()
        else:
            values = torch.zeros(bsize, device=device)

        # flow_sde/cps std is state-independent => entropy carries no gradient
        # signal; report zeros (same convention as OpenPi0 for non-learned noise).
        entropy = torch.zeros(bsize, 1, device=device)

        return {
            "logprobs": log_probs,
            "values": values,
            "entropy": entropy,
        }
