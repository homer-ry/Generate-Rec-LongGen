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
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_negative(item_set: set, item_num: int, rng: random.Random) -> int:
    while True:
        t = rng.randint(1, item_num)
        if t not in item_set:
            return t


@dataclass
class SeqData:
    user_train: Dict[int, List[int]]
    user_valid: Dict[int, int]
    user_test: Dict[int, int]
    user_items_all: Dict[int, set]
    user_ids: List[int]
    item_num: int
    user_num: int


def load_seq_data(root: Path, min_rating: float) -> SeqData:
    raw_dir = root / "datasets" / "ml-1m" / "raw_data"
    with (raw_dir / "user_id_map.pkl").open("rb") as f:
        user_id_map = pickle.load(f)
    with (raw_dir / "movie_id_map.pkl").open("rb") as f:
        movie_id_map = pickle.load(f)

    ratings = pd.read_csv(
        raw_dir / "ratings.dat",
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    ratings = ratings[ratings["rating"] >= float(min_rating)]
    ratings = ratings[ratings["user_id"].isin(user_id_map.keys())]
    ratings = ratings[ratings["movie_id"].isin(movie_id_map.keys())].copy()
    ratings["uid"] = ratings["user_id"].map(user_id_map).astype(np.int32)
    ratings["iid"] = ratings["movie_id"].map(movie_id_map).astype(np.int32) + 1  # reserve 0 for padding
    ratings = ratings.sort_values(["uid", "timestamp"]).reset_index(drop=True)

    user_num = int(max(user_id_map.values())) + 1
    item_num = int(max(movie_id_map.values())) + 1

    seq_by_user: Dict[int, List[int]] = {u: [] for u in range(user_num)}
    for uid, iid in ratings[["uid", "iid"]].itertuples(index=False):
        seq_by_user[int(uid)].append(int(iid))

    user_train: Dict[int, List[int]] = {}
    user_valid: Dict[int, int] = {}
    user_test: Dict[int, int] = {}
    user_items_all: Dict[int, set] = {}

    for u, seq in seq_by_user.items():
        if len(seq) < 3:
            continue
        user_train[u] = seq[:-2]
        user_valid[u] = seq[-2]
        user_test[u] = seq[-1]
        user_items_all[u] = set(seq)

    user_ids = sorted(user_train.keys())
    return SeqData(
        user_train=user_train,
        user_valid=user_valid,
        user_test=user_test,
        user_items_all=user_items_all,
        user_ids=user_ids,
        item_num=item_num,
        user_num=user_num,
    )


class SASRecTrainDataset(Dataset):
    def __init__(self, data: SeqData, maxlen: int, seed: int) -> None:
        self.data = data
        self.maxlen = int(maxlen)
        self.user_ids = data.user_ids
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, index: int):
        u = self.user_ids[index]
        seq = np.zeros([self.maxlen], dtype=np.int64)
        pos = np.zeros([self.maxlen], dtype=np.int64)
        neg = np.zeros([self.maxlen], dtype=np.int64)

        user_seq = self.data.user_train[u]
        nxt = user_seq[-1]
        idx = self.maxlen - 1
        ts = self.data.user_items_all[u]

        for i in reversed(user_seq[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = sample_negative(ts, self.data.item_num, self.rng)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (
            np.int64(u),
            torch.from_numpy(seq),
            torch.from_numpy(pos),
            torch.from_numpy(neg),
        )


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units: int, dropout_rate: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRec(nn.Module):
    def __init__(
        self,
        item_num: int,
        maxlen: int,
        hidden_units: int,
        num_blocks: int,
        num_heads: int,
        dropout_rate: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.item_num = int(item_num)
        self.maxlen = int(maxlen)
        self.hidden_units = int(hidden_units)
        self.dev = device

        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.maxlen, self.hidden_units)
        self.emb_dropout = nn.Dropout(dropout_rate)

        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(self.hidden_units, eps=1e-8))
            self.attention_layers.append(
                nn.MultiheadAttention(
                    embed_dim=self.hidden_units,
                    num_heads=num_heads,
                    dropout=dropout_rate,
                    batch_first=True,
                )
            )
            self.forward_layernorms.append(nn.LayerNorm(self.hidden_units, eps=1e-8))
            self.forward_layers.append(PointWiseFeedForward(self.hidden_units, dropout_rate))

        self.last_layernorm = nn.LayerNorm(self.hidden_units, eps=1e-8)

    def log2feats(self, log_seqs: torch.Tensor) -> torch.Tensor:
        seqs = self.item_emb(log_seqs) * (self.hidden_units**0.5)
        poss = torch.arange(log_seqs.shape[1], device=self.dev).unsqueeze(0).repeat(log_seqs.shape[0], 1)
        seqs = seqs + self.pos_emb(poss)
        seqs = self.emb_dropout(seqs)

        timeline_mask = log_seqs.eq(0)
        seqs = seqs.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](
                Q,
                seqs,
                seqs,
                attn_mask=attention_mask,
                key_padding_mask=timeline_mask,
                need_weights=False,
            )
            seqs = Q + mha_outputs
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs = seqs.masked_fill(timeline_mask.unsqueeze(-1), 0.0)

        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, log_seqs: torch.Tensor, pos_seqs: torch.Tensor, neg_seqs: torch.Tensor):
        log_feats = self.log2feats(log_seqs)
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        return pos_logits, neg_logits

    def score_items(self, log_seqs: torch.Tensor, item_indices: torch.Tensor) -> torch.Tensor:
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(item_indices)
        logits = torch.bmm(item_embs, final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

    def score_all(self, log_seqs: torch.Tensor) -> torch.Tensor:
        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        all_items = self.item_emb.weight[1:]
        return torch.matmul(final_feat, all_items.t())


def build_eval_seq(user_train: Sequence[int], append_item: int, maxlen: int) -> np.ndarray:
    seq = np.zeros([maxlen], dtype=np.int64)
    idx = maxlen - 1
    if append_item > 0:
        seq[idx] = append_item
        idx -= 1
    for i in reversed(user_train):
        seq[idx] = i
        idx -= 1
        if idx == -1:
            break
    return seq


def evaluate_sampled(
    model: SASRec,
    data: SeqData,
    maxlen: int,
    topk: int,
    mode: str,
    num_neg: int,
    device: torch.device,
    seed: int,
) -> Tuple[float, float]:
    assert mode in {"valid", "test"}
    model.eval()
    rng = random.Random(seed)

    hrs = []
    ndcgs = []
    with torch.no_grad():
        for u in data.user_ids:
            if mode == "valid":
                seq = build_eval_seq(data.user_train[u], append_item=0, maxlen=maxlen)
                target = data.user_valid[u]
                rated = set(data.user_train[u])
            else:
                seq = build_eval_seq(data.user_train[u], append_item=data.user_valid[u], maxlen=maxlen)
                target = data.user_test[u]
                rated = set(data.user_train[u])
                rated.add(data.user_valid[u])

            rated.add(0)
            item_idx = [target]
            for _ in range(num_neg):
                item_idx.append(sample_negative(rated, data.item_num, rng))

            seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)
            item_t = torch.tensor(item_idx, dtype=torch.long, device=device).unsqueeze(0)
            logits = model.score_items(seq_t, item_t).squeeze(0).cpu().numpy()
            rank = int((-logits).argsort().tolist().index(0))

            if rank < topk:
                hrs.append(1.0)
                ndcgs.append(1.0 / math.log2(rank + 2.0))
            else:
                hrs.append(0.0)
                ndcgs.append(0.0)

    return float(np.mean(hrs)), float(np.mean(ndcgs))


def evaluate_full(
    model: SASRec,
    data: SeqData,
    maxlen: int,
    topk: int,
    mode: str,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[float, float]:
    assert mode in {"valid", "test"}
    model.eval()
    hrs = []
    ndcgs = []

    users = data.user_ids
    with torch.no_grad():
        for st in range(0, len(users), batch_size):
            batch = users[st : st + batch_size]
            seqs = []
            targets = []
            seen_sets = []

            for u in batch:
                if mode == "valid":
                    seqs.append(build_eval_seq(data.user_train[u], append_item=0, maxlen=maxlen))
                    targets.append(data.user_valid[u])
                    seen = set(data.user_train[u])
                else:
                    seqs.append(build_eval_seq(data.user_train[u], append_item=data.user_valid[u], maxlen=maxlen))
                    targets.append(data.user_test[u])
                    seen = set(data.user_train[u])
                    seen.add(data.user_valid[u])
                seen_sets.append(seen)

            seq_t = torch.from_numpy(np.stack(seqs)).to(device)
            scores = model.score_all(seq_t)  # [B, item_num], index j => item j+1

            for i, u in enumerate(batch):
                target = targets[i]
                seen = seen_sets[i]
                for it in seen:
                    if it != target and it > 0:
                        scores[i, it - 1] = -1e9

                top = torch.topk(scores[i], k=topk).indices + 1
                top_list = top.cpu().numpy().tolist()
                if target in top_list:
                    hrs.append(1.0)
                    rk = top_list.index(target)
                    ndcgs.append(1.0 / math.log2(rk + 2.0))
                else:
                    hrs.append(0.0)
                    ndcgs.append(0.0)

    return float(np.mean(hrs)), float(np.mean(ndcgs))


def parse_args():
    p = argparse.ArgumentParser(description="Standard SASRec training (LOO) on Agent4Rec ml-1m subset")
    p.add_argument("--root_dir", type=str, default=".")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--min_rating", type=float, default=4.0)
    p.add_argument("--maxlen", type=int, default=50)
    p.add_argument("--hidden_units", type=int, default=50)
    p.add_argument("--num_blocks", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=1)
    p.add_argument("--dropout_rate", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--l2_emb", type=float, default=0.0)
    p.add_argument("--train_objective", type=str, default="full_ce", choices=["bce", "full_ce"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--eval_neg", type=int, default=100)
    p.add_argument("--eval_mode", type=str, default="sampled", choices=["sampled", "full"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument(
        "--save_dir",
        type=str,
        default="recommenders/weights/ml-1m/SASRec/Saved",
    )
    p.add_argument(
        "--metrics_out",
        type=str,
        default="baseline/results/sasrec_loo_standard_metrics.json",
    )
    return p.parse_args()


def train_one_epoch_bce(
    model: SASRec,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    l2_emb: float,
    device: torch.device,
) -> float:
    bce_criterion = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    epoch_losses = []

    for _, seq, pos, neg in train_loader:
        seq = seq.to(device)
        pos = pos.to(device)
        neg = neg.to(device)

        pos_logits, neg_logits = model(seq, pos, neg)
        pos_labels = torch.ones_like(pos_logits, device=device)
        neg_labels = torch.zeros_like(neg_logits, device=device)

        mask = pos.ne(0).float()
        loss = (bce_criterion(pos_logits, pos_labels) + bce_criterion(neg_logits, neg_labels)) * mask
        loss = loss.sum() / torch.clamp(mask.sum(), min=1.0)

        if l2_emb > 0:
            loss = loss + l2_emb * model.item_emb.weight.norm(2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.item()))

    return float(np.mean(epoch_losses)) if epoch_losses else 0.0


def train_one_epoch_full_ce(
    model: SASRec,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    l2_emb: float,
    device: torch.device,
) -> float:
    model.train()
    epoch_losses = []

    for _, seq, pos, _ in train_loader:
        seq = seq.to(device)
        pos = pos.to(device)

        feats = model.log2feats(seq)  # [B, L, H]
        mask = pos.ne(0)
        if mask.sum().item() == 0:
            continue

        feat_flat = feats[mask]  # [M, H]
        target_flat = pos[mask] - 1  # [M], map item id [1..N] -> [0..N-1]
        logits = torch.matmul(feat_flat, model.item_emb.weight[1:].t())  # [M, item_num]
        loss = nn.functional.cross_entropy(logits, target_flat)

        if l2_emb > 0:
            loss = loss + l2_emb * model.item_emb.weight.norm(2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.item()))

    return float(np.mean(epoch_losses)) if epoch_losses else 0.0


def main():
    args = parse_args()
    set_seed(args.seed)
    root = Path(args.root_dir).resolve()
    device = torch.device(args.device)

    data = load_seq_data(root, min_rating=args.min_rating)
    train_ds = SASRecTrainDataset(data=data, maxlen=args.maxlen, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = SASRec(
        item_num=data.item_num,
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = (root / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (root / args.metrics_out).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    history = []
    best_epoch = 0
    best_valid_hr = -1.0
    patience_left = int(args.patience)

    for epoch in range(1, args.epochs + 1):
        if args.train_objective == "bce":
            train_loss = train_one_epoch_bce(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                l2_emb=args.l2_emb,
                device=device,
            )
        else:
            train_loss = train_one_epoch_full_ce(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                l2_emb=args.l2_emb,
                device=device,
            )

        if args.eval_mode == "sampled":
            valid_hr, valid_ndcg = evaluate_sampled(
                model=model,
                data=data,
                maxlen=args.maxlen,
                topk=args.topk,
                mode="valid",
                num_neg=args.eval_neg,
                device=device,
                seed=args.seed + epoch,
            )
        else:
            valid_hr, valid_ndcg = evaluate_full(
                model=model,
                data=data,
                maxlen=args.maxlen,
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
                    "item_num": data.item_num,
                    "maxlen": args.maxlen,
                    "hidden_units": args.hidden_units,
                    "num_blocks": args.num_blocks,
                    "num_heads": args.num_heads,
                    "dropout_rate": args.dropout_rate,
                },
            }
            torch.save(ckpt, save_dir / "epoch=best.loo.standard.pth")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"early stop at epoch={epoch}")
                break

    best_ckpt = torch.load(save_dir / "epoch=best.loo.standard.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    if args.eval_mode == "sampled":
        test_hr, test_ndcg = evaluate_sampled(
            model=model,
            data=data,
            maxlen=args.maxlen,
            topk=args.topk,
            mode="test",
            num_neg=args.eval_neg,
            device=device,
            seed=args.seed + 9999,
        )
    else:
        test_hr, test_ndcg = evaluate_full(
            model=model,
            data=data,
            maxlen=args.maxlen,
            topk=args.topk,
            mode="test",
            device=device,
        )

    print(f"test_hr@{args.topk}={test_hr:.5f} test_ndcg@{args.topk}={test_ndcg:.5f}")

    metrics = {
        "config": {
            "seed": args.seed,
            "device": args.device,
            "min_rating": args.min_rating,
            "maxlen": args.maxlen,
            "hidden_units": args.hidden_units,
            "num_blocks": args.num_blocks,
            "num_heads": args.num_heads,
            "dropout_rate": args.dropout_rate,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "l2_emb": args.l2_emb,
            "train_objective": args.train_objective,
            "epochs": args.epochs,
            "patience": args.patience,
            "topk": args.topk,
            "eval_neg": args.eval_neg,
            "eval_mode": args.eval_mode,
            "user_num": data.user_num,
            "item_num": data.item_num,
            "train_users": len(data.user_ids),
        },
        "best_epoch": int(best_epoch),
        "best_valid_hr": float(best_valid_hr),
        "test_hr": float(test_hr),
        "test_ndcg": float(test_ndcg),
        "history": history,
        "checkpoint": str((save_dir / "epoch=best.loo.standard.pth").resolve()),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved metrics: {metrics_path}")
    print(f"saved checkpoint: {(save_dir / 'epoch=best.loo.standard.pth').resolve()}")


if __name__ == "__main__":
    main()
