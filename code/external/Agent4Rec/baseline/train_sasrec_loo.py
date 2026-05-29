from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class LOOSplit:
    train_seq: Dict[int, List[int]]
    valid_item: Dict[int, int]
    test_item: Dict[int, int]
    user_item_set: Dict[int, set]
    n_users: int
    n_items: int


def load_sequences_from_raw(root: Path, min_rating: float) -> LOOSplit:
    raw_dir = root / "datasets" / "ml-1m" / "raw_data"
    user_map_path = raw_dir / "user_id_map.pkl"
    movie_map_path = raw_dir / "movie_id_map.pkl"
    ratings_path = raw_dir / "ratings.dat"

    with user_map_path.open("rb") as f:
        user_id_map = pickle.load(f)
    with movie_map_path.open("rb") as f:
        movie_id_map = pickle.load(f)

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )

    ratings = ratings[ratings["rating"] >= float(min_rating)]
    ratings = ratings[ratings["user_id"].isin(user_id_map.keys())]
    ratings = ratings[ratings["movie_id"].isin(movie_id_map.keys())]
    ratings = ratings.copy()

    ratings["uid"] = ratings["user_id"].map(user_id_map).astype(np.int32)
    ratings["iid"] = ratings["movie_id"].map(movie_id_map).astype(np.int32)
    ratings = ratings.sort_values(["uid", "timestamp"], ascending=[True, True]).reset_index(drop=True)

    n_users = int(max(user_id_map.values())) + 1
    n_items_raw = int(max(movie_id_map.values())) + 1

    seq_by_user: Dict[int, List[int]] = {u: [] for u in range(n_users)}
    for uid, iid in ratings[["uid", "iid"]].itertuples(index=False):
        # reserve 0 for padding index
        seq_by_user[int(uid)].append(int(iid) + 1)

    train_seq: Dict[int, List[int]] = {}
    valid_item: Dict[int, int] = {}
    test_item: Dict[int, int] = {}
    user_item_set: Dict[int, set] = {}

    for uid, seq in seq_by_user.items():
        if len(seq) < 3:
            continue
        train_seq[uid] = seq[:-2]
        valid_item[uid] = seq[-2]
        test_item[uid] = seq[-1]
        user_item_set[uid] = set(seq)

    # +1 since items were shifted by 1 (0 is pad)
    n_items = n_items_raw + 1
    return LOOSplit(
        train_seq=train_seq,
        valid_item=valid_item,
        test_item=test_item,
        user_item_set=user_item_set,
        n_users=n_users,
        n_items=n_items,
    )


def sample_negative(item_set: set, n_items: int) -> int:
    while True:
        item = random.randint(1, n_items - 1)
        if item not in item_set:
            return item


class SASRecTrainDataset(Dataset):
    def __init__(
        self,
        users: Sequence[int],
        train_seq: Dict[int, List[int]],
        user_item_set: Dict[int, set],
        max_len: int,
        n_items: int,
    ) -> None:
        self.users = list(users)
        self.train_seq = train_seq
        self.user_item_set = user_item_set
        self.max_len = int(max_len)
        self.n_items = int(n_items)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int):
        uid = self.users[idx]
        seq = self.train_seq[uid]

        seq_out = np.zeros(self.max_len, dtype=np.int64)
        pos_out = np.zeros(self.max_len, dtype=np.int64)
        neg_out = np.zeros(self.max_len, dtype=np.int64)

        if len(seq) >= 2:
            cursor = self.max_len - 1
            nxt = seq[-1]
            for item in reversed(seq[:-1]):
                seq_out[cursor] = item
                pos_out[cursor] = nxt
                neg_out[cursor] = sample_negative(self.user_item_set[uid], self.n_items)
                nxt = item
                cursor -= 1
                if cursor < 0:
                    break

        return (
            np.int64(uid),
            torch.from_numpy(seq_out),
            torch.from_numpy(pos_out),
            torch.from_numpy(neg_out),
        )


class SASRecModel(nn.Module):
    def __init__(
        self,
        n_items: int,
        max_len: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.n_items = int(n_items)
        self.max_len = int(max_len)
        self.hidden_dim = int(hidden_dim)

        self.item_emb = nn.Embedding(self.n_items, self.hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(self.max_len, self.hidden_dim)
        self.emb_dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=n_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(self.hidden_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        bsz, slen = seq.shape
        pos_idx = torch.arange(slen, device=seq.device).unsqueeze(0).expand(bsz, -1)
        x = self.item_emb(seq) * math.sqrt(self.hidden_dim)
        x = x + self.pos_emb(pos_idx)
        x = self.emb_dropout(x)

        padding_mask = seq.eq(0)
        causal_mask = torch.triu(
            torch.ones((slen, slen), dtype=torch.bool, device=seq.device),
            diagonal=1,
        )
        out = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        out = self.ln(out)
        return out

    def score_all_items(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.forward(seq)
        lengths = seq.ne(0).sum(dim=1).clamp(min=1) - 1
        final_h = h[torch.arange(seq.size(0), device=seq.device), lengths]
        item_matrix = self.item_emb.weight[1:]  # drop padding
        return torch.matmul(final_h, item_matrix.t())


def make_eval_seq(seq: Sequence[int], max_len: int) -> np.ndarray:
    arr = np.zeros(max_len, dtype=np.int64)
    trimmed = list(seq)[-max_len:]
    arr[-len(trimmed) :] = np.asarray(trimmed, dtype=np.int64)
    return arr


def evaluate_loo(
    model: SASRecModel,
    split: LOOSplit,
    users: Sequence[int],
    max_len: int,
    topk: int,
    mode: str,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[float, float]:
    assert mode in {"valid", "test"}

    eval_users = list(users)
    hr_list = []
    ndcg_list = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(eval_users), batch_size):
            batch_users = eval_users[start : start + batch_size]
            seq_batch = []
            targets = []
            seen_sets = []

            for uid in batch_users:
                train_seq = split.train_seq[uid]
                if mode == "valid":
                    seq = train_seq
                    target = split.valid_item[uid]
                    seen = set(train_seq)
                else:
                    seq = train_seq + [split.valid_item[uid]]
                    target = split.test_item[uid]
                    seen = set(seq)
                seq_batch.append(make_eval_seq(seq, max_len))
                targets.append(target)
                seen_sets.append(seen)

            seq_t = torch.from_numpy(np.stack(seq_batch)).to(device)
            scores = model.score_all_items(seq_t)  # [B, n_items-1], index 0 => item 1

            for i, uid in enumerate(batch_users):
                target = targets[i]
                seen = seen_sets[i]

                # remove seen items except target item
                for it in seen:
                    if it != target and it > 0:
                        scores[i, it - 1] = -1e9

                topk_idx = torch.topk(scores[i], k=topk).indices + 1
                topk_list = topk_idx.cpu().numpy().tolist()
                if target in topk_list:
                    hr_list.append(1.0)
                    rank = topk_list.index(target)
                    ndcg_list.append(1.0 / math.log2(rank + 2.0))
                else:
                    hr_list.append(0.0)
                    ndcg_list.append(0.0)

    return float(np.mean(hr_list)), float(np.mean(ndcg_list))


def train_one_epoch(
    model: SASRecModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []

    for _, seq, pos, neg in loader:
        seq = seq.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        h = model(seq)
        pos_emb = model.item_emb(pos)
        neg_emb = model.item_emb(neg)

        pos_logits = (h * pos_emb).sum(dim=-1)
        neg_logits = (h * neg_emb).sum(dim=-1)
        mask = pos.gt(0).float()

        if mask.sum().item() <= 0:
            continue

        loss_pos = F.binary_cross_entropy_with_logits(
            pos_logits, torch.ones_like(pos_logits), reduction="none"
        )
        loss_neg = F.binary_cross_entropy_with_logits(
            neg_logits, torch.zeros_like(neg_logits), reduction="none"
        )
        loss = ((loss_pos + loss_neg) * mask).sum() / mask.sum()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Train SASRec with LOO split on Agent4Rec ml-1m subset")
    p.add_argument("--root_dir", type=str, default=".")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--max_len", type=int, default=50)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--min_rating", type=float, default=4.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument(
        "--save_dir",
        type=str,
        default="recommenders/weights/ml-1m/SASRec/Saved",
    )
    p.add_argument(
        "--metrics_out",
        type=str,
        default="baseline/results/sasrec_loo_metrics.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    root = Path(args.root_dir).resolve()
    device = torch.device(args.device)

    split = load_sequences_from_raw(root, min_rating=args.min_rating)
    users = sorted(split.train_seq.keys())

    train_ds = SASRecTrainDataset(
        users=users,
        train_seq=split.train_seq,
        user_item_set=split.user_item_set,
        max_len=args.max_len,
        n_items=split.n_items,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = SASRecModel(
        n_items=split.n_items,
        max_len=args.max_len,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_dir = (root / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (root / args.metrics_out).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    best_valid_hr = -1.0
    best_epoch = -1
    patience_left = int(args.patience)
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_hr, valid_ndcg = evaluate_loo(
            model=model,
            split=split,
            users=users,
            max_len=args.max_len,
            topk=args.topk,
            mode="valid",
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_hr": valid_hr,
            "valid_ndcg": valid_ndcg,
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.5f} "
            f"valid_hr@{args.topk}={valid_hr:.5f} "
            f"valid_ndcg@{args.topk}={valid_ndcg:.5f}"
        )

        if valid_hr > best_valid_hr:
            best_valid_hr = valid_hr
            best_epoch = epoch
            patience_left = int(args.patience)
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": {
                    "n_items": split.n_items,
                    "max_len": args.max_len,
                    "hidden_dim": args.hidden_dim,
                    "n_heads": args.n_heads,
                    "n_layers": args.n_layers,
                    "dropout": args.dropout,
                },
            }
            torch.save(ckpt, save_dir / "epoch=best.loo.pth")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop at epoch={epoch}")
                break

    best_ckpt = torch.load(save_dir / "epoch=best.loo.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_hr, test_ndcg = evaluate_loo(
        model=model,
        split=split,
        users=users,
        max_len=args.max_len,
        topk=args.topk,
        mode="test",
        device=device,
    )
    print(f"test_hr@{args.topk}={test_hr:.5f} test_ndcg@{args.topk}={test_ndcg:.5f}")

    metrics = {
        "config": {
            "seed": args.seed,
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_len": args.max_len,
            "hidden_dim": args.hidden_dim,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "topk": args.topk,
            "min_rating": args.min_rating,
            "patience": args.patience,
            "n_users": len(users),
            "n_items_with_padding": split.n_items,
        },
        "best_epoch": int(best_epoch),
        "best_valid_hr": float(best_valid_hr),
        "test_hr": float(test_hr),
        "test_ndcg": float(test_ndcg),
        "history": history,
        "checkpoint": str((save_dir / "epoch=best.loo.pth").resolve()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved metrics: {metrics_path}")
    print(f"saved checkpoint: {(save_dir / 'epoch=best.loo.pth').resolve()}")


if __name__ == "__main__":
    main()
