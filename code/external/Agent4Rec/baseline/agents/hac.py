from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP
from .ddpg import DDPGAgent, DDPGConfig


@dataclass
class HACConfig(DDPGConfig):
    behavior_lr: float = 1e-4
    behavior_coef: float = 0.20
    hyper_actor_coef: float = 0.10


class HACAgent(DDPGAgent):
    """
    Simplified HAC baseline aligned with KuaiSim HAC design:
    actor-critic + behavior-guidance + inverse hyper-action consistency.

    Reference source in KuaiSim:
    - code/model/agent/HAC.py
    """

    def __init__(self, state_dim: int, action_dim: int, effect_dim: int, device: torch.device, cfg: HACConfig | None = None):
        super().__init__(state_dim=state_dim, action_dim=action_dim, device=device, cfg=cfg or HACConfig())
        self.cfg: HACConfig = cfg or HACConfig()

        self.inverse_module = MLP(effect_dim, action_dim, hidden_dims=(128, 128)).to(self.device)
        self.inverse_optim = torch.optim.Adam(self.inverse_module.parameters(), lr=self.cfg.behavior_lr)

        self.behavior_optim = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.behavior_lr)

    def train_step(self, batch: dict) -> dict:
        state = batch["state"]
        action = batch["action"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        not_done = batch["not_done"]
        behavior_target = batch["behavior_target"]
        effect_action = batch["effect_action"]

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = reward + self.cfg.gamma * not_done * self.critic_target(next_state, next_action)

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Standard DDPG actor loss
        actor_action = self.actor(state)
        actor_loss_main = -self.critic(state, actor_action).mean()

        # KuaiSim-style inverse consistency: recover action from executed effect
        recovered_action = torch.tanh(self.inverse_module(effect_action))
        hyper_actor_loss = F.mse_loss(recovered_action, action)

        # KuaiSim-style behavior supervision
        behavior_loss = F.mse_loss(actor_action, behavior_target)

        actor_total_loss = (
            actor_loss_main
            + self.cfg.hyper_actor_coef * hyper_actor_loss
            + self.cfg.behavior_coef * behavior_loss
        )

        self.actor_optim.zero_grad()
        actor_total_loss.backward()
        self.actor_optim.step()

        # optimize inverse module separately for stable alignment
        recovered_action_detached = torch.tanh(self.inverse_module(effect_action))
        inv_loss = F.mse_loss(recovered_action_detached, action.detach())
        self.inverse_optim.zero_grad()
        inv_loss.backward()
        self.inverse_optim.step()

        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss_main.item()),
            "hyper_actor_loss": float(hyper_actor_loss.item()),
            "behavior_loss": float(behavior_loss.item()),
            "q": float(current_q.mean().item()),
        }

    def eval_action(self, state: np.ndarray) -> np.ndarray:
        return self.select_action(state, explore=False)
