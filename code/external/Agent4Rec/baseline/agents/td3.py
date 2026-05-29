from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP


@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.01
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    actor_wd: float = 1e-5
    critic_wd: float = 1e-5
    exploration_std: float = 0.10
    policy_noise: float = 0.10
    noise_clip: float = 0.25
    policy_delay: int = 2


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


class TD3Agent:
    """
    Lightweight TD3 baseline aligned with KuaiSim naming (model/agent/TD3.py).
    Used as "TD" baseline in this project.
    """

    def __init__(self, state_dim: int, action_dim: int, device: torch.device, cfg: TD3Config | None = None):
        self.device = device
        self.cfg = cfg or TD3Config()

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

        self.total_updates = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if explore:
            action = action + np.random.normal(0.0, self.cfg.exploration_std, size=action.shape)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def train_step(self, batch: dict) -> dict:
        self.total_updates += 1

        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        not_done = batch["not_done"]

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.cfg.policy_noise
            ).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = reward + self.cfg.gamma * not_done * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss_val = 0.0
        if self.total_updates % self.cfg.policy_delay == 0:
            actor_loss = -self.critic.q1_value(state, self.actor(state)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()
            actor_loss_val = float(actor_loss.item())

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_val,
            "q": float(current_q1.mean().item()),
        }

    def _soft_update(self, src: nn.Module, tgt: nn.Module) -> None:
        tau = self.cfg.tau
        for p, p_tgt in zip(src.parameters(), tgt.parameters()):
            p_tgt.data.copy_(tau * p.data + (1.0 - tau) * p_tgt.data)

    def eval_action(self, state: np.ndarray) -> np.ndarray:
        return self.select_action(state, explore=False)
