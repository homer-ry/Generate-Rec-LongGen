from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP


@dataclass
class CQLConfig:
    gamma: float = 0.99
    tau: float = 0.01
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    actor_wd: float = 1e-5
    critic_wd: float = 1e-5
    exploration_std: float = 0.10
    conservative_alpha: float = 1.0
    bc_coef: float = 0.05


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.mlp = MLP(state_dim, action_dim, hidden_dims=(256, 256))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mlp(state))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.q1 = MLP(state_dim + action_dim, 1, hidden_dims=(256, 256))
        self.q2 = MLP(state_dim + action_dim, 1, hidden_dims=(256, 256))

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


class CQLAgent:
    """
    Lightweight CQL-style actor-critic:
    Bellman regression + conservative critic penalty + mild behavior cloning.
    """

    def __init__(self, state_dim: int, action_dim: int, device: torch.device, cfg: CQLConfig | None = None):
        self.device = device
        self.cfg = cfg or CQLConfig()

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=self.cfg.actor_lr, weight_decay=self.cfg.actor_wd
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.critic_lr, weight_decay=self.cfg.critic_wd
        )

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
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
            next_action = self.actor_target(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + self.cfg.gamma * not_done * torch.min(target_q1, target_q2)

        q1, q2 = self.critic(state, action)
        bellman_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        rand_action = torch.empty_like(action).uniform_(-1.0, 1.0)
        with torch.no_grad():
            pi_action_detach = self.actor(state)

        q1_rand, q2_rand = self.critic(state, rand_action)
        q1_pi, q2_pi = self.critic(state, pi_action_detach)

        q1_cat = torch.cat([q1_rand, q1_pi], dim=1)
        q2_cat = torch.cat([q2_rand, q2_pi], dim=1)
        cql1 = torch.logsumexp(q1_cat, dim=1).mean() - q1.mean()
        cql2 = torch.logsumexp(q2_cat, dim=1).mean() - q2.mean()
        conservative_loss = cql1 + cql2

        critic_loss = bellman_loss + self.cfg.conservative_alpha * conservative_loss
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        policy_action = self.actor(state)
        actor_loss = -self.critic.q1_value(state, policy_action).mean()
        bc_loss = F.mse_loss(policy_action, action)
        actor_total_loss = actor_loss + self.cfg.bc_coef * bc_loss

        self.actor_optim.zero_grad()
        actor_total_loss.backward()
        self.actor_optim.step()

        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)

        return {
            "critic_loss": float(critic_loss.item()),
            "bellman_loss": float(bellman_loss.item()),
            "conservative_loss": float(conservative_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "q": float(q1.mean().item()),
        }

    def _soft_update(self, src: nn.Module, tgt: nn.Module) -> None:
        tau = self.cfg.tau
        for p, p_tgt in zip(src.parameters(), tgt.parameters()):
            p_tgt.data.copy_(tau * p.data + (1.0 - tau) * p_tgt.data)

