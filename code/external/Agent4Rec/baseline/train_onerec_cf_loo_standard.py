from __future__ import annotations

import argparse
import json
import math
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


def dedupe_keep_order(values: Sequence[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(int(v))
    return out


@dataclass
class SeqData:
    user_train: Dict[int, List[int]]
    user_valid: Dict[int, int]
    user_test: Dict[int, int]
    user_items_all: Dict[int, set]
    user_ids: List[int]
    item_num: int
    user_num: int


def _load_user_items_txt(path: Path) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            parts = raw.strip().split()
            if not parts:
                continue
            uid = int(parts[0])
            items = [int(x) for x in parts[1:]]
            out[uid] = items
    return out


def load_seq_data_cf(root: Path, dataset: str, cf_data_subdir: str = "cf_data") -> SeqData:
    cf_dir = root / "datasets" / dataset / str(cf_data_subdir)
    train = _load_user_items_txt(cf_dir / "train.txt")
    valid = _load_user_items_txt(cf_dir / "valid.txt")
    test = _load_user_items_txt(cf_dir / "test.txt")

    all_users = set(train.keys()) | set(valid.keys()) | set(test.keys())
    if not all_users:
        raise ValueError(f"Empty cf_data for dataset={dataset}")
    user_num = max(all_users) + 1

    max_item = -1
    for d in (train, valid, test):
        for items in d.values():
            if items:
                max_item = max(max_item, max(items))
    if max_item < 0:
        raise ValueError(f"No items found in cf_data for dataset={dataset}")
    item_num = max_item + 1

    user_train: Dict[int, List[int]] = {}
    user_valid: Dict[int, int] = {}
    user_test: Dict[int, int] = {}
    user_items_all: Dict[int, set] = {}

    for u in range(user_num):
        full0 = (train.get(u, []) or []) + (valid.get(u, []) or []) + (test.get(u, []) or [])
        full0 = dedupe_keep_order(full0)
        full = [int(i) + 1 for i in full0 if 0 <= int(i) < item_num]
        if len(full) < 3:
            continue

        user_train[u] = full[:-2]
        user_valid[u] = full[-2]
        user_test[u] = full[-1]
        user_items_all[u] = set(full)

    user_ids = sorted(user_train.keys())
    if not user_ids:
        raise ValueError(f"No users with >=3 interactions for dataset={dataset}")

    return SeqData(
        user_train=user_train,
        user_valid=user_valid,
        user_test=user_test,
        user_items_all=user_items_all,
        user_ids=user_ids,
        item_num=item_num,
        user_num=user_num,
    )


def _choose_ctx_columns(df: pd.DataFrame) -> List[str]:
    preferred = ["activity", "diversity", "conformity"]
    if all(c in df.columns for c in preferred):
        return preferred
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 3:
        return numeric_cols[:3]
    return preferred


def load_user_context(root: Path, dataset: str, user_num: int, ctx_dim: int = 3) -> np.ndarray:
    out = np.full((int(user_num), int(ctx_dim)), 0.5, dtype=np.float32)
    stat_path = root / "datasets" / dataset / "simulation" / "user_statistic.csv"
    if not stat_path.exists():
        return out

    df = pd.read_csv(stat_path)
    if "user_id" in df.columns:
        df = df.set_index("user_id")
    elif df.columns[0].lower().startswith("unnamed"):
        df = df.set_index(df.columns[0])
    try:
        df.index = df.index.astype(int)
    except Exception:
        pass

    cols = _choose_ctx_columns(df)
    for uid in range(int(user_num)):
        if uid not in df.index:
            continue
        row = df.loc[uid]
        vals = []
        for c in cols:
            if c in row.index:
                vals.append(float(row[c]))
            else:
                vals.append(2.0)
        vals = vals[:ctx_dim]
        if len(vals) < ctx_dim:
            vals += [2.0] * (ctx_dim - len(vals))
        out[uid] = np.asarray(vals, dtype=np.float32)

    # Normalize each feature to [0, 1] for stable conditioning.
    mins = out.min(axis=0, keepdims=True)
    maxs = out.max(axis=0, keepdims=True)
    denom = np.maximum(maxs - mins, 1e-6)
    out = (out - mins) / denom
    return out.astype(np.float32)


class OneRecTrainDataset(Dataset):
    """
    Train samples for next-item prediction with right-padded sequence.
    """

    def __init__(self, data: SeqData, maxlen: int) -> None:
        self.data = data
        self.maxlen = int(maxlen)
        self.user_ids = data.user_ids

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, index: int):
        u = self.user_ids[index]
        seq = np.zeros([self.maxlen], dtype=np.int64)
        pos = np.zeros([self.maxlen], dtype=np.int64)

        user_seq = self.data.user_train[u]
        if len(user_seq) >= 2:
            cut = user_seq[-(self.maxlen + 1) :]
            inp = cut[:-1]
            tgt = cut[1:]
            L = len(inp)
            seq[:L] = np.asarray(inp, dtype=np.int64)
            pos[:L] = np.asarray(tgt, dtype=np.int64)

        return np.int64(u), torch.from_numpy(seq), torch.from_numpy(pos)


class OneRec(nn.Module):
    def __init__(
        self,
        item_num: int,
        hidden_units: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        maxlen: int,
        user_ctx_dim: int = 3,
    ) -> None:
        super().__init__()
        self.item_num = int(item_num)
        self.hidden_units = int(hidden_units)
        self.maxlen = int(maxlen)
        self.user_ctx_dim = int(user_ctx_dim)

        self.item_emb = nn.Embedding(self.item_num + 1, self.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(self.maxlen, self.hidden_units)
        self.emb_dropout = nn.Dropout(float(dropout_rate))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_units,
            nhead=int(num_heads),
            dim_feedforward=self.hidden_units * 4,
            dropout=float(dropout_rate),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))
        self.ln = nn.LayerNorm(self.hidden_units)
        self.user_proj = nn.Sequential(
            nn.Linear(self.user_ctx_dim, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(self.hidden_units, self.hidden_units),
            nn.ReLU(),
        )

    def log2feats(self, log_seqs: torch.Tensor, user_ctx: torch.Tensor) -> torch.Tensor:
        bsz, slen = log_seqs.shape
        pos = torch.arange(slen, device=log_seqs.device).unsqueeze(0).expand(bsz, -1)

        x = self.item_emb(log_seqs) + self.pos_emb(pos)
        x = self.emb_dropout(x)
        key_padding = log_seqs.eq(0)
        z = self.encoder(x, src_key_padding_mask=key_padding)
        z = self.ln(z)

        user_h = self.user_proj(user_ctx).unsqueeze(1)  # [B,1,H]
        z = z + user_h
        z = z.masked_fill(key_padding.unsqueeze(-1), 0.0)
        return z

    def score_all(self, log_seqs: torch.Tensor, user_ctx: torch.Tensor) -> torch.Tensor:
        feats = self.log2feats(log_seqs, user_ctx)
        lengths = log_seqs.ne(0).sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, feats.size(-1))
        final_feat = feats.gather(1, idx).squeeze(1)  # [B,H]
        all_items = self.item_emb.weight[1:]
        return torch.matmul(final_feat, all_items.t())

    def score_items(self, log_seqs: torch.Tensor, item_indices: torch.Tensor, user_ctx: torch.Tensor) -> torch.Tensor:
        feats = self.log2feats(log_seqs, user_ctx)
        lengths = log_seqs.ne(0).sum(dim=1).clamp(min=1)
        idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, feats.size(-1))
        final_feat = feats.gather(1, idx).squeeze(1)
        item_embs = self.item_emb(item_indices)
        return torch.bmm(item_embs, final_feat.unsqueeze(-1)).squeeze(-1)


def build_eval_seq_right_padded(user_train: Sequence[int], append_item: int, maxlen: int) -> np.ndarray:
    seq = np.zeros([maxlen], dtype=np.int64)
    items = list(user_train)
    if append_item > 0:
        items.append(int(append_item))
    items = items[-maxlen:]
    if items:
        seq[: len(items)] = np.asarray(items, dtype=np.int64)
    return seq


@torch.no_grad()
def evaluate_sampled(
    model: OneRec,
    data: SeqData,
    user_ctx: np.ndarray,
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

    def sample_negative(item_set: set, item_num: int) -> int:
        while True:
            t = rng.randint(1, item_num)
            if t not in item_set:
                return t

    for u in data.user_ids:
        if mode == "valid":
            seq = build_eval_seq_right_padded(data.user_train[u], append_item=0, maxlen=maxlen)
            target = data.user_valid[u]
            rated = set(data.user_train[u])
        else:
            seq = build_eval_seq_right_padded(data.user_train[u], append_item=data.user_valid[u], maxlen=maxlen)
            target = data.user_test[u]
            rated = set(data.user_train[u])
            rated.add(data.user_valid[u])

        rated.add(0)
        item_idx = [target]
        for _ in range(int(num_neg)):
            item_idx.append(sample_negative(rated, data.item_num))

        seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)
        item_t = torch.tensor(item_idx, dtype=torch.long, device=device).unsqueeze(0)
        ctx_t = torch.from_numpy(user_ctx[int(u)]).unsqueeze(0).to(device)
        logits = model.score_items(seq_t, item_t, ctx_t).squeeze(0).cpu().numpy()
        rank = int((-logits).argsort().tolist().index(0))

        if rank < topk:
            hrs.append(1.0)
            ndcgs.append(1.0 / math.log2(rank + 2.0))
        else:
            hrs.append(0.0)
            ndcgs.append(0.0)

    return float(np.mean(hrs)), float(np.mean(ndcgs))


@torch.no_grad()
def evaluate_full(
    model: OneRec,
    data: SeqData,
    user_ctx: np.ndarray,
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
    for st in range(0, len(users), int(batch_size)):
        batch = users[st : st + int(batch_size)]
        seqs = []
        targets = []
        seen_sets = []
        ctxs = []
        for u in batch:
            if mode == "valid":
                seqs.append(build_eval_seq_right_padded(data.user_train[u], append_item=0, maxlen=maxlen))
                targets.append(data.user_valid[u])
                seen = set(data.user_train[u])
            else:
                seqs.append(build_eval_seq_right_padded(data.user_train[u], append_item=data.user_valid[u], maxlen=maxlen))
                targets.append(data.user_test[u])
                seen = set(data.user_train[u])
                seen.add(data.user_valid[u])
            seen_sets.append(seen)
            ctxs.append(user_ctx[int(u)])

        seq_t = torch.from_numpy(np.stack(seqs)).to(device)
        ctx_t = torch.from_numpy(np.stack(ctxs)).to(device)
        scores = model.score_all(seq_t, ctx_t)  # [B, item_num], index j => item j+1

        for i, _u in enumerate(batch):
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
    p = argparse.ArgumentParser(description="OneRec training (LOO) on Agent4Rec-format cf_data datasets")
    p.add_argument("--root_dir", type=str, default=".")
    p.add_argument("--dataset", type=str, default="all-beauty")
    p.add_argument(
        "--cf_data_subdir",
        type=str,
        default="cf_data",
        help="Which cf_data folder to read under datasets/<dataset>/ (e.g., cf_data or cf_data_min4).",
    )
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--maxlen", type=int, default=50)
    p.add_argument("--hidden_units", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout_rate", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--l2_emb", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--eval_neg", type=int, default=100)
    p.add_argument("--eval_mode", type=str, default="sampled", choices=["sampled", "full"])
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--metrics_out", type=str, default="")
    return p.parse_args()


def train_one_epoch_full_ce(
    model: OneRec,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    l2_emb: float,
    device: torch.device,
    user_ctx: np.ndarray,
) -> float:
    model.train()
    losses = []
    for u, seq, pos in train_loader:
        seq = seq.to(device)
        pos = pos.to(device)
        ctx = torch.from_numpy(user_ctx[u.numpy()]).to(device)
        feats = model.log2feats(seq, ctx)  # [B, L, H]
        mask = pos.ne(0)
        if mask.sum().item() == 0:
            continue
        feat_flat = feats[mask]  # [M, H]
        target_flat = pos[mask] - 1  # [M]
        logits = torch.matmul(feat_flat, model.item_emb.weight[1:].t())  # [M, item_num]
        loss = nn.functional.cross_entropy(logits, target_flat)
        if l2_emb > 0:
            loss = loss + float(l2_emb) * model.item_emb.weight.norm(2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def main():
    args = parse_args()
    set_seed(int(args.seed))

    root = Path(args.root_dir).resolve()
    device = torch.device(args.device)

    if not args.save_dir:
        args.save_dir = f"recommenders/weights/{args.dataset}/OneRec/Saved"
    if not args.metrics_out:
        args.metrics_out = f"baseline/results/onerec_{args.dataset}_loo_standard_metrics.json"

    data = load_seq_data_cf(root, dataset=args.dataset, cf_data_subdir=args.cf_data_subdir)
    user_ctx = load_user_context(root=root, dataset=args.dataset, user_num=data.user_num, ctx_dim=3)

    train_ds = OneRecTrainDataset(data=data, maxlen=int(args.maxlen))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        drop_last=False,
    )

    model = OneRec(
        item_num=int(data.item_num),
        hidden_units=int(args.hidden_units),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        dropout_rate=float(args.dropout_rate),
        maxlen=int(args.maxlen),
        user_ctx_dim=3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    best_valid_hr = -1.0
    best_epoch = 0
    patience_left = int(args.patience)
    history = []

    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_one_epoch_full_ce(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            l2_emb=float(args.l2_emb),
            device=device,
            user_ctx=user_ctx,
        )

        if args.eval_mode == "sampled":
            valid_hr, valid_ndcg = evaluate_sampled(
                model=model,
                data=data,
                user_ctx=user_ctx,
                maxlen=int(args.maxlen),
                topk=int(args.topk),
                mode="valid",
                num_neg=int(args.eval_neg),
                device=device,
                seed=int(args.seed) + 999,
            )
        else:
            valid_hr, valid_ndcg = evaluate_full(
                model=model,
                data=data,
                user_ctx=user_ctx,
                maxlen=int(args.maxlen),
                topk=int(args.topk),
                mode="valid",
                device=device,
            )

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "valid_hr": float(valid_hr),
                "valid_ndcg": float(valid_ndcg),
            }
        )
        print(
            f"epoch={epoch:03d} loss={train_loss:.5f} "
            f"valid_hr@{int(args.topk)}={valid_hr:.5f} valid_ndcg@{int(args.topk)}={valid_ndcg:.5f}"
        )

        if valid_hr > best_valid_hr:
            best_valid_hr = float(valid_hr)
            best_epoch = int(epoch)
            patience_left = int(args.patience)
            ckpt = {
                "epoch": int(epoch),
                "model_state_dict": model.state_dict(),
                "config": {
                    "item_num": int(data.item_num),
                    "maxlen": int(args.maxlen),
                    "hidden_units": int(args.hidden_units),
                    "num_layers": int(args.num_layers),
                    "num_heads": int(args.num_heads),
                    "dropout_rate": float(args.dropout_rate),
                    "user_ctx_dim": 3,
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
            user_ctx=user_ctx,
            maxlen=int(args.maxlen),
            topk=int(args.topk),
            mode="test",
            num_neg=int(args.eval_neg),
            device=device,
            seed=int(args.seed) + 9999,
        )
    else:
        test_hr, test_ndcg = evaluate_full(
            model=model,
            data=data,
            user_ctx=user_ctx,
            maxlen=int(args.maxlen),
            topk=int(args.topk),
            mode="test",
            device=device,
        )

    print(f"test_hr@{int(args.topk)}={test_hr:.5f} test_ndcg@{int(args.topk)}={test_ndcg:.5f}")

    metrics = {
        "config": {
            "seed": int(args.seed),
            "device": str(args.device),
            "maxlen": int(args.maxlen),
            "hidden_units": int(args.hidden_units),
            "num_layers": int(args.num_layers),
            "num_heads": int(args.num_heads),
            "dropout_rate": float(args.dropout_rate),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "l2_emb": float(args.l2_emb),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "topk": int(args.topk),
            "eval_neg": int(args.eval_neg),
            "eval_mode": str(args.eval_mode),
            "user_num": int(data.user_num),
            "item_num": int(data.item_num),
            "train_users": int(len(data.user_ids)),
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

    args_txt = save_dir / "args.txt"
    if not args_txt.exists():
        sim_args = {
            "vis": -1,
            "seed": int(args.seed),
            "clear_checkpoints": False,
            "candidate": False,
            "test_only": False,
            "data_path": "../datasets/",
            "dataset": str(args.dataset),
            "cf_data_subdir": str(args.cf_data_subdir),
            "embed_size": 64,
            "batch_size": 2048,
            "lr": 0.0005,
            "regs": 1e-05,
            "epoch": 1,
            "Ks": 20,
            "verbose": 5,
            "saveID": "Saved",
            "patience": 20,
            "checkpoint": "./",
            "cuda": 0,
            "IPStype": "cn",
            "n_layers": 0,
            "max2keep": 1,
            "infonce": 0,
            "neg_sample": 1,
            "num_workers": 0,
            "train_norm": False,
            "pred_norm": False,
            "nodrop": False,
            "no_wandb": True,
            "modeltype": "OneRec",
        }
        args_txt.write_text(json.dumps(sim_args, indent=2), encoding="utf-8")
        print(f"saved args: {args_txt}")


if __name__ == "__main__":
    main()
