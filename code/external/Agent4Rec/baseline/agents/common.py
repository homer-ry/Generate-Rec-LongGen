import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims=(128, 128), activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, effect_dim: int, capacity: int = 200000):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.not_done = np.zeros((self.capacity, 1), dtype=np.float32)

        self.behavior_target = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.effect_action = np.zeros((self.capacity, effect_dim), dtype=np.float32)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        behavior_target: np.ndarray,
        effect_action: np.ndarray,
    ) -> int:
        idx = self.ptr
        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward
        self.next_state[idx] = next_state
        self.not_done[idx] = 0.0 if done else 1.0

        if behavior_target is not None:
            self.behavior_target[idx] = behavior_target
        if effect_action is not None:
            self.effect_action[idx] = effect_action

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def add_to_reward(self, idx: int, delta: float) -> None:
        self.reward[idx] += float(delta)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "state": torch.as_tensor(self.state[idx], device=device),
            "action": torch.as_tensor(self.action[idx], device=device),
            "reward": torch.as_tensor(self.reward[idx], device=device),
            "next_state": torch.as_tensor(self.next_state[idx], device=device),
            "not_done": torch.as_tensor(self.not_done[idx], device=device),
            "behavior_target": torch.as_tensor(self.behavior_target[idx], device=device),
            "effect_action": torch.as_tensor(self.effect_action[idx], device=device),
        }


@dataclass
class EpisodeStats:
    episode_return: float = 0.0
    length: int = 0
    likes: int = 0
    shown: int = 0
    repetition_sum: float = 0.0
    satisfaction_sum: float = 0.0

    def update(self, reward: float, info: dict) -> None:
        self.episode_return += float(reward)
        self.length += 1
        self.likes += int(info.get("likes", 0))
        self.shown += int(info.get("shown", 0))
        self.repetition_sum += float(info.get("genre_repetition", 0.0))
        self.satisfaction_sum += float(info.get("satisfaction", 0.0))

    def as_dict(self) -> dict:
        like_rate = self.likes / max(self.shown, 1)
        avg_repetition = self.repetition_sum / max(self.length, 1)
        avg_satisfaction = self.satisfaction_sum / max(self.length, 1)
        return {
            "return": self.episode_return,
            "length": self.length,
            "like_rate": like_rate,
            "avg_repetition": avg_repetition,
            "avg_satisfaction": avg_satisfaction,
        }
