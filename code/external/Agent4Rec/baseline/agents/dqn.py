from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import MLP


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 3e-4
    exploration_start: float = 1.0
    exploration_end: float = 0.05
    exploration_decay_steps: int = 5000
    target_update_interval: int = 200
    double_q: bool = True
    dueling: bool = True
    huber_delta: float = 1.0
    action_levels: tuple[float, ...] = (-1.0, 0.0, 1.0)


class QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.mlp = MLP(state_dim, n_actions, hidden_dims=(256, 256))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(state)


class DuelingQNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.backbone = MLP(state_dim, 256, hidden_dims=(256,), activation=nn.ReLU)
        self.adv = nn.Sequential(nn.ReLU(), nn.Linear(256, n_actions))
        self.val = nn.Sequential(nn.ReLU(), nn.Linear(256, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(state)
        adv = self.adv(feat)
        val = self.val(feat)
        return val + (adv - adv.mean(dim=-1, keepdim=True))


class DQNAgent:
    """
    DQN on a discretized 3-d continuous action space.
    The selected discrete action is mapped back to continuous weights action.
    """

    def __init__(self, state_dim: int, action_dim: int, device: torch.device, cfg: DQNConfig | None = None):
        self.device = device
        self.cfg = cfg or DQNConfig()

        self.action_table = self._build_action_table(action_dim, self.cfg.action_levels)
        self.n_actions = int(self.action_table.shape[0])

        q_cls = DuelingQNet if self.cfg.dueling else QNet
        self.q_net = q_cls(state_dim, self.n_actions).to(self.device)
        self.q_target = q_cls(state_dim, self.n_actions).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.cfg.lr)

        self.total_steps = 0
        self.total_updates = 0
        self.last_action_index = 0

    @staticmethod
    def _build_action_table(action_dim: int, levels: tuple[float, ...]) -> np.ndarray:
        table = np.array(list(itertools.product(levels, repeat=action_dim)), dtype=np.float32)
        return table

    def epsilon(self) -> float:
        t = min(float(self.total_steps), float(max(1, self.cfg.exploration_decay_steps)))
        frac = t / float(max(1, self.cfg.exploration_decay_steps))
        return float(self.cfg.exploration_start + frac * (self.cfg.exploration_end - self.cfg.exploration_start))

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        eps = self.epsilon() if explore else 0.0
        self.total_steps += 1

        if explore and np.random.rand() < eps:
            idx = int(np.random.randint(0, self.n_actions))
        else:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q = self.q_net(state_t)
                idx = int(torch.argmax(q, dim=-1).item())

        self.last_action_index = idx
        return self.action_table[idx].copy()

    def eval_action(self, state: np.ndarray) -> np.ndarray:
        return self.select_action(state, explore=False)

    def action_index_from_action(self, action: np.ndarray) -> int:
        action = np.asarray(action, dtype=np.float32)
        dist = np.square(self.action_table - action[None, :]).sum(axis=1)
        return int(np.argmin(dist))

    def train_step(self, batch: dict) -> dict:
        self.total_updates += 1
        state = batch["state"]
        reward = batch["reward"]
        next_state = batch["next_state"]
        not_done = batch["not_done"]
        action_idx = batch["action_idx"].long()
        if action_idx.ndim == 2:
            action_idx = action_idx.squeeze(-1)

        q_all = self.q_net(state)
        q_sa = q_all.gather(1, action_idx.unsqueeze(-1))

        with torch.no_grad():
            if self.cfg.double_q:
                next_actions = torch.argmax(self.q_net(next_state), dim=1, keepdim=True)
                next_q = self.q_target(next_state).gather(1, next_actions)
            else:
                next_q = self.q_target(next_state).max(dim=1, keepdim=True).values
            target_q = reward + self.cfg.gamma * not_done * next_q

        if self.cfg.huber_delta > 0:
            loss = F.smooth_l1_loss(q_sa, target_q)
        else:
            loss = F.mse_loss(q_sa, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        if self.total_updates % max(1, self.cfg.target_update_interval) == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        return {
            "q_loss": float(loss.item()),
            "q": float(q_sa.mean().item()),
            "epsilon": float(self.epsilon()),
        }


class RainbowAgent(DQNAgent):
    """
    Rainbow-lite variant:
    keeps double+dueling DQN structure with the same discrete action abstraction.
    """

    def __init__(self, state_dim: int, action_dim: int, device: torch.device):
        cfg = DQNConfig(
            double_q=True,
            dueling=True,
            target_update_interval=150,
            action_levels=(-1.0, -0.5, 0.0, 0.5, 1.0),
        )
        super().__init__(state_dim=state_dim, action_dim=action_dim, device=device, cfg=cfg)

