from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class SeqBCConfig:
    max_seq_len: int = 20
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 8
    collect_episodes: int = 120


class SequenceDataset(Dataset):
    def __init__(self, seqs: np.ndarray, masks: np.ndarray, targets: np.ndarray):
        self.seqs = seqs.astype(np.float32)
        self.masks = masks.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return self.seqs.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.seqs[idx]),
            torch.from_numpy(self.masks[idx]),
            torch.from_numpy(self.targets[idx]),
        )


class GRU4RecPolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).long().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        last = h[-1]
        return self.head(last)


class SASRecPolicyNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.proj = nn.Linear(state_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, slen, _ = seq.shape
        x = self.proj(seq) + self.pos_emb[:, :slen, :]
        key_padding = mask < 0.5
        z = self.encoder(x, src_key_padding_mask=key_padding)
        z = self.ln(z)
        lengths = mask.sum(dim=1).long().clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, z.size(-1))
        last = z.gather(1, idx).squeeze(1)
        return self.head(last)


class TigerPolicyNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float,
    ):
        super().__init__()
        self.proj = nn.Linear(state_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)

        # Global context tower (bidirectional)
        global_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.global_encoder = nn.TransformerEncoder(global_layer, num_layers=n_layers)

        # Local autoregressive tower (causal)
        local_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.local_encoder = nn.TransformerEncoder(local_layer, num_layers=n_layers)

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, slen, _ = seq.shape
        x = self.proj(seq) + self.pos_emb[:, :slen, :]
        key_padding = mask < 0.5

        z_global = self.global_encoder(x, src_key_padding_mask=key_padding)

        causal_mask = torch.triu(
            torch.ones((slen, slen), device=seq.device, dtype=torch.bool),
            diagonal=1,
        )
        z_local = self.local_encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=key_padding,
        )

        lengths = mask.sum(dim=1).long().clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, z_global.size(-1))
        h_global = z_global.gather(1, idx).squeeze(1)
        h_local = z_local.gather(1, idx).squeeze(1)

        gate = self.fuse(torch.cat([h_global, h_local], dim=-1))
        h = gate * h_local + (1.0 - gate) * h_global
        h = self.ln(h)
        return self.head(h)


class OneRecPolicyNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float,
        user_ctx_dim: int,
    ):
        super().__init__()
        self.user_ctx_dim = user_ctx_dim
        self.user_proj = nn.Sequential(
            nn.Linear(user_ctx_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.seq_proj = nn.Linear(state_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, slen, state_dim = seq.shape
        user_ctx = seq[:, 0, : self.user_ctx_dim]
        user_h = self.user_proj(user_ctx)

        x = self.seq_proj(seq) + self.pos_emb[:, :slen, :]
        key_padding = mask < 0.5
        z = self.encoder(x, src_key_padding_mask=key_padding)

        lengths = mask.sum(dim=1).long().clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, z.size(-1))
        seq_h = z.gather(1, idx).squeeze(1)

        h = torch.cat([user_h, seq_h], dim=-1)
        return self.head(h)


class SequenceBCAgent:
    """
    Sequence baselines (SASRec / GRU4Rec / TIGER-style / OneRec-style) trained with behavior cloning
    on trajectories collected from the same long-session simulator.
    """

    def __init__(
        self,
        algo: str,
        state_dim: int,
        action_dim: int,
        device: torch.device,
        cfg: SeqBCConfig | None = None,
        user_ctx_dim: int = 21,
    ):
        self.algo = algo.upper()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.cfg = cfg or SeqBCConfig()
        self.max_seq_len = self.cfg.max_seq_len

        if self.algo == "GRU4REC":
            self.model = GRU4RecPolicyNet(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=self.cfg.hidden_dim,
                dropout=self.cfg.dropout,
            ).to(self.device)
        elif self.algo == "SASREC":
            self.model = SASRecPolicyNet(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=self.cfg.hidden_dim,
                n_heads=self.cfg.n_heads,
                n_layers=self.cfg.n_layers,
                max_seq_len=self.cfg.max_seq_len,
                dropout=self.cfg.dropout,
            ).to(self.device)
        elif self.algo == "TIGER":
            self.model = TigerPolicyNet(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=self.cfg.hidden_dim,
                n_heads=self.cfg.n_heads,
                n_layers=self.cfg.n_layers,
                max_seq_len=self.cfg.max_seq_len,
                dropout=self.cfg.dropout,
            ).to(self.device)
        elif self.algo == "ONEREC":
            self.model = OneRecPolicyNet(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=self.cfg.hidden_dim,
                n_heads=self.cfg.n_heads,
                n_layers=self.cfg.n_layers,
                max_seq_len=self.cfg.max_seq_len,
                dropout=self.cfg.dropout,
                user_ctx_dim=user_ctx_dim,
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported sequence algo: {self.algo}")

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.history: List[np.ndarray] = []

    def _build_sample(self, history: List[np.ndarray], state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seq = history + [state]
        seq = seq[-self.max_seq_len :]
        seq_arr = np.zeros((self.max_seq_len, self.state_dim), dtype=np.float32)
        mask = np.zeros((self.max_seq_len,), dtype=np.float32)
        seq_arr[: len(seq)] = np.stack(seq).astype(np.float32)
        mask[: len(seq)] = 1.0
        return seq_arr, mask

    def collect_imitation_data(self, env, collect_episodes: int, random_action_std: float = 0.35):
        seqs = []
        masks = []
        targets = []

        ep_lens = []
        ep_returns = []

        for _ in range(collect_episodes):
            state = env.reset()
            done = False
            history: List[np.ndarray] = []
            ep_len = 0
            ep_ret = 0.0
            while not done:
                seq_arr, mask = self._build_sample(history, state)

                # random + oracle-target mixture for broader coverage
                oracle = None
                if np.random.rand() < 0.20 and len(targets) > 0:
                    oracle = targets[np.random.randint(0, len(targets))]
                if oracle is None:
                    action = np.random.normal(0.0, random_action_std, size=(self.action_dim,)).astype(np.float32)
                    action = np.clip(action, -1.0, 1.0)
                else:
                    action = np.clip(oracle + np.random.normal(0.0, 0.10, size=oracle.shape), -1.0, 1.0)

                next_state, reward, done, info = env.step(action)
                target_action = info.get("behavior_target_action")

                seqs.append(seq_arr)
                masks.append(mask)
                targets.append(np.asarray(target_action, dtype=np.float32))

                history.append(state.astype(np.float32))
                if len(history) > self.max_seq_len:
                    history = history[-self.max_seq_len :]
                state = next_state
                ep_len += 1
                ep_ret += float(reward)

            ep_lens.append(ep_len)
            ep_returns.append(ep_ret)

        return (
            np.asarray(seqs, dtype=np.float32),
            np.asarray(masks, dtype=np.float32),
            np.asarray(targets, dtype=np.float32),
            {
                "collect_mean_length": float(np.mean(ep_lens)) if ep_lens else 0.0,
                "collect_mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
                "collect_samples": int(len(seqs)),
            },
        )

    def fit(self, env, collect_episodes: int | None = None, epochs: int | None = None) -> dict:
        collect_episodes = int(collect_episodes or self.cfg.collect_episodes)
        epochs = int(epochs or self.cfg.epochs)

        seqs, masks, targets, collect_stats = self.collect_imitation_data(env, collect_episodes=collect_episodes)
        ds = SequenceDataset(seqs, masks, targets)
        loader = DataLoader(ds, batch_size=self.cfg.batch_size, shuffle=True, drop_last=False)

        losses = []
        self.model.train()
        for _ in range(epochs):
            for seq_b, mask_b, target_b in loader:
                seq_b = seq_b.to(self.device)
                mask_b = mask_b.to(self.device)
                target_b = target_b.to(self.device)
                pred = self.model(seq_b, mask_b)
                loss = F.mse_loss(pred, target_b)

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                losses.append(float(loss.item()))

        return {
            "train_mean_bc_loss": float(np.mean(losses)) if losses else 0.0,
            "train_epochs": int(epochs),
            "train_episodes": int(collect_episodes),
            **collect_stats,
        }

    def reset_episode(self) -> None:
        self.history = []

    def update_history(self, state: np.ndarray) -> None:
        self.history.append(state.astype(np.float32))
        if len(self.history) > self.max_seq_len:
            self.history = self.history[-self.max_seq_len :]

    def eval_action(self, state: np.ndarray) -> np.ndarray:
        seq_arr, mask = self._build_sample(self.history, state)
        self.model.eval()
        with torch.no_grad():
            seq_t = torch.from_numpy(seq_arr).unsqueeze(0).to(self.device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
            action = self.model(seq_t, mask_t).cpu().numpy()[0]
        return np.clip(action, -1.0, 1.0).astype(np.float32)
