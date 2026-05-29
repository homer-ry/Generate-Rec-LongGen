from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .common import MLP


@dataclass
class IQLConfig:
    gamma: float = 0.99
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    exploration_std: float = 0.05
    expectile: float = 0.7
    awr_beta: float = 3.0
    awr_max_weight: float = 20.0


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.backbone = MLP(state_dim, 256, hidden_dims=(256,), activation=nn.ReLU)
        self.mean_head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def dist(self, state: torch.Tensor):
        feat = self.backbone(state)
        mean = torch.tanh(self.mean_head(feat))
        std = torch.exp(self.log_std).clamp(1e-3, 1.0)
        return Normal(mean, std)

    def mean_action(self, state: torch.Tensor):
        feat = self.backbone(state)
        return torch.tanh(self.mean_head(feat))


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden_dims=(256, 256))
        self.q2 = MLP(state_dim + action_dim, 1, hidden_dims=(256, 256))

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class ValueNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.v = MLP(state_dim, 1, hidden_dims=(256, 256))

    def forward(self, state: torch.Tensor):
        return self.v(state)


class IQLAgent:
    """
    Lightweight IQL-style learner:
    - expectile value regression
    - TD critic on bootstrapped V
    - advantage-weighted behavior cloning actor
    """

    def __init__(self, state_dim: int, action_dim: int, device: torch.device, cfg: IQLConfig | None = None):
        self.device = device
        self.cfg = cfg or IQLConfig()
        self.actor = GaussianActor(state_dim, action_dim).to(self.device)
        self.critic = TwinCritic(state_dim, action_dim).to(self.device)
        self.value = ValueNet(state_dim).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.value_optim = torch.optim.Adam(self.value.parameters(), lr=self.cfg.value_lr)

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if explore:
                action = self.actor.dist(state_t).sample()
            else:
                action = self.actor.mean_action(state_t)
        action = action.cpu().numpy()[0]
        if explore:
            action = action + np.random.normal(0.0, self.cfg.exploration_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def eval_action(self, state: np.ndarray) -> np.ndarray:
        return self.select_action(state, explore=False)

    def train_step(self, batch: dict) -> dict:
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        not_done = batch["not_done"]

        with torch.no_grad():
            q1_data, q2_data = self.critic(state, action)
            q_data = torch.min(q1_data, q2_data)

        v_pred = self.value(state)
        diff = q_data - v_pred
        expectile = float(self.cfg.expectile)
        weight = torch.where(diff > 0, torch.full_like(diff, expectile), torch.full_like(diff, 1.0 - expectile))
        value_loss = (weight * diff.pow(2)).mean()
        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()

        with torch.no_grad():
            target_q = reward + self.cfg.gamma * not_done * self.value(next_state)
        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        with torch.no_grad():
            q1_det, q2_det = self.critic(state, action)
            q_det = torch.min(q1_det, q2_det)
            v_det = self.value(state)
            adv = q_det - v_det
            aw = torch.exp(self.cfg.awr_beta * adv).clamp(max=self.cfg.awr_max_weight)

        dist = self.actor.dist(state)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        actor_loss = -(aw * log_prob).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return {
            "value_loss": float(value_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "adv_mean": float(adv.mean().item()),
            "q": float(q_det.mean().item()),
        }

