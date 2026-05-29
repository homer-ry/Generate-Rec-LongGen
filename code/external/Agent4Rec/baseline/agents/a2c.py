from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .common import MLP


@dataclass
class A2CConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    actor_lr: float = 2e-4
    critic_lr: float = 5e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    update_epochs: int = 4


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.backbone = MLP(state_dim, 256, hidden_dims=(256,), activation=nn.Tanh)
        self.policy_mean = nn.Sequential(nn.Tanh(), nn.Linear(256, action_dim))
        self.value_head = nn.Sequential(nn.Tanh(), nn.Linear(256, 1))
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor):
        feat = self.backbone(state)
        mean = torch.tanh(self.policy_mean(feat))
        value = self.value_head(feat)
        std = torch.exp(self.log_std).clamp(1e-3, 1.0)
        return mean, std, value


class A2CAgent:
    """
    Lightweight continuous-control A2C baseline aligned with KuaiSim naming (model/agent/A2C.py).
    """

    def __init__(self, state_dim: int, action_dim: int, device: torch.device, cfg: A2CConfig | None = None):
        self.device = device
        self.cfg = cfg or A2CConfig()

        self.net = PolicyValueNet(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.cfg.actor_lr,
        )

    def sample_action(self, state: np.ndarray):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std, value = self.net(state_t)
            dist = Normal(mean, std)
            action = dist.sample().clamp(-1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)
        return (
            action.cpu().numpy()[0].astype(np.float32),
            float(log_prob.item()),
            float(value.squeeze(-1).item()),
        )

    def eval_action(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, _, _ = self.net(state_t)
        return mean.cpu().numpy()[0].astype(np.float32)

    def update(self, trajectory: dict) -> dict:
        states = torch.as_tensor(np.asarray(trajectory["states"], dtype=np.float32), device=self.device)
        actions = torch.as_tensor(np.asarray(trajectory["actions"], dtype=np.float32), device=self.device)
        rewards = np.asarray(trajectory["rewards"], dtype=np.float32)
        dones = np.asarray(trajectory["dones"], dtype=np.float32)
        values = np.asarray(trajectory["values"], dtype=np.float32)
        next_value = float(trajectory["last_value"])

        returns, advantages = self._compute_gae(rewards, dones, values, next_value)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-6)

        policy_loss_v = 0.0
        value_loss_v = 0.0
        entropy_v = 0.0

        for _ in range(self.cfg.update_epochs):
            mean, std, value_pred = self.net(states)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            policy_loss = -(log_prob * adv_t).mean()
            value_loss = F.mse_loss(value_pred.squeeze(-1), returns_t)
            total_loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

            policy_loss_v += float(policy_loss.item())
            value_loss_v += float(value_loss.item())
            entropy_v += float(entropy.item())

        k = float(self.cfg.update_epochs)
        return {
            "policy_loss": policy_loss_v / k,
            "value_loss": value_loss_v / k,
            "entropy": entropy_v / k,
        }

    def _compute_gae(self, rewards, dones, values, next_value):
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda

        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0

        vals = np.append(values, next_value)
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * vals[t + 1] * nonterminal - vals[t]
            gae = delta + gamma * lam * nonterminal * gae
            advantages[t] = gae

        returns = advantages + values
        return returns.astype(np.float32), advantages.astype(np.float32)
