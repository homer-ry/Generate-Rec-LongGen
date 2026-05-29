from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import T5Config, T5ForConditionalGeneration


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_cf_sequences(path: Path) -> Dict[int, List[int]]:
    seqs: Dict[int, List[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            arr = [int(x) for x in line.split()]
            if len(arr) < 2:
                continue
            seqs[int(arr[0])] = [int(x) for x in arr[1:]]
    return seqs


def infer_n_items(cf_paths: List[Path], movie_detail_path: Path) -> int:
    max_item = -1
    for path in cf_paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                arr = [int(x) for x in line.split()]
                if len(arr) > 1:
                    max_item = max(max_item, max(arr[1:]))
    if movie_detail_path.exists():
        md = pd.read_csv(movie_detail_path, usecols=["movie_id"])
        if not md.empty:
            max_item = max(max_item, int(md["movie_id"].max()))
    if max_item < 0:
        raise ValueError("failed to infer item count from cf_data/movie_detail")
    return max_item + 1


def build_content_item_features(
    movie_detail_path: Path,
    n_items: int,
    train_sequences: Dict[int, List[int]],
) -> np.ndarray:
    df = pd.read_csv(movie_detail_path)
    if "movie_id" not in df.columns:
        raise ValueError("movie_detail.csv must contain movie_id")

    genre_vocab = set()
    for g in df.get("genres", pd.Series(dtype=str)).fillna("").astype(str).tolist():
        for tok in g.split("|"):
            tok = tok.strip()
            if tok:
                genre_vocab.add(tok)
    genre_vocab = sorted(list(genre_vocab))
    genre_to_idx = {g: i for i, g in enumerate(genre_vocab)}

    rating_col = len(genre_vocab)
    pop_col = len(genre_vocab) + 1
    x = np.zeros((n_items, len(genre_vocab) + 2), dtype=np.float32)

    ratings = np.full((n_items,), np.nan, dtype=np.float32)
    for row in df.itertuples(index=False):
        iid = int(getattr(row, "movie_id"))
        if iid < 0 or iid >= n_items:
            continue
        genres = str(getattr(row, "genres", ""))
        for tok in genres.split("|"):
            tok = tok.strip()
            if tok in genre_to_idx:
                x[iid, genre_to_idx[tok]] = 1.0
        try:
            ratings[iid] = float(getattr(row, "rating", np.nan))
        except Exception:
            pass

    mean_rating = float(np.nanmean(ratings)) if np.isnan(ratings).sum() < len(ratings) else 0.0
    ratings = np.where(np.isnan(ratings), mean_rating, ratings)
    std_rating = float(np.std(ratings))
    if std_rating > 0:
        ratings = (ratings - float(np.mean(ratings))) / std_rating
    x[:, rating_col] = ratings.astype(np.float32)

    pop = np.zeros((n_items,), dtype=np.float32)
    for seq in train_sequences.values():
        for iid in seq:
            if 0 <= iid < n_items:
                pop[iid] += 1.0
    pop = np.log1p(pop)
    std_pop = float(np.std(pop))
    if std_pop > 0:
        pop = (pop - float(np.mean(pop))) / std_pop
    x[:, pop_col] = pop.astype(np.float32)

    mu = x.mean(axis=0, keepdims=True)
    sigma = x.std(axis=0, keepdims=True)
    sigma[sigma < 1e-6] = 1.0
    x = (x - mu) / sigma
    return x.astype(np.float32)


def build_interaction_item_features(
    n_items: int,
    train_sequences: Dict[int, List[int]],
    n_components: int,
    seed: int,
) -> np.ndarray:
    users = sorted(train_sequences.keys())
    uidx = {u: i for i, u in enumerate(users)}

    rows = []
    cols = []
    vals = []
    for u in users:
        seen = set()
        for iid in train_sequences[u]:
            if 0 <= iid < n_items and iid not in seen:
                rows.append(uidx[u])
                cols.append(iid)
                vals.append(1.0)
                seen.add(iid)

    if not rows:
        raise ValueError("interaction feature build failed: empty user-item matrix")

    mat = torch.sparse_coo_tensor(
        indices=torch.tensor([rows, cols], dtype=torch.long),
        values=torch.tensor(vals, dtype=torch.float32),
        size=(len(users), n_items),
    ).coalesce()
    dense = mat.to_dense().cpu().numpy().astype(np.float32)  # [U, I]

    x_item_user = dense.T  # [I, U]
    max_comp = max(2, min(x_item_user.shape[0], x_item_user.shape[1]) - 1)
    n_comp = int(min(max_comp, max(2, n_components)))

    svd = TruncatedSVD(n_components=n_comp, random_state=seed, n_iter=12)
    emb = svd.fit_transform(x_item_user).astype(np.float32)

    # Add normalized popularity as an explicit interaction signal.
    pop = np.asarray(dense.sum(axis=0), dtype=np.float32).reshape(-1, 1)
    pop = np.log1p(pop)
    pop = (pop - pop.mean(axis=0, keepdims=True)) / np.clip(pop.std(axis=0, keepdims=True), 1e-6, None)
    feat = np.concatenate([emb, pop], axis=1)
    feat = StandardScaler().fit_transform(feat).astype(np.float32)
    return feat


def compose_item_features(
    feature_source: str,
    content_feat: np.ndarray,
    interaction_feat: np.ndarray,
    content_weight: float,
    interaction_weight: float,
) -> np.ndarray:
    source = str(feature_source).lower().strip()
    if source == "content":
        return content_feat.astype(np.float32)
    if source == "interaction":
        return interaction_feat.astype(np.float32)
    if source != "hybrid":
        raise ValueError(f"unknown feature_source={feature_source}")

    c = content_feat.astype(np.float32)
    i = interaction_feat.astype(np.float32)
    c = StandardScaler().fit_transform(c).astype(np.float32)
    i = StandardScaler().fit_transform(i).astype(np.float32)
    c = float(content_weight) * c
    i = float(interaction_weight) * i
    return np.concatenate([c, i], axis=1).astype(np.float32)


def residual_kmeans_sid(
    features: np.ndarray,
    sid_depth: int,
    codebook_size: int,
    seed: int,
    batch_size: int,
) -> Tuple[np.ndarray, List[float]]:
    n_items = features.shape[0]
    residual = features.copy()
    codes = np.zeros((n_items, sid_depth), dtype=np.int32)
    layer_errors: List[float] = []

    for layer in range(sid_depth):
        km = MiniBatchKMeans(
            n_clusters=codebook_size,
            random_state=seed + layer,
            batch_size=batch_size,
            n_init=10,
            max_iter=200,
            reassignment_ratio=0.01,
        )
        labels = km.fit_predict(residual)
        centers = km.cluster_centers_.astype(np.float32)
        codes[:, layer] = labels.astype(np.int32)
        residual = residual - centers[labels]
        mse = float(np.mean(np.sum(residual * residual, axis=1)))
        layer_errors.append(mse)
        print(f"[sid] layer={layer + 1}/{sid_depth} residual_mse={mse:.6f}")

    return codes, layer_errors


def save_sid_mapping(codes: np.ndarray, out_path: Path) -> None:
    n_items, sid_depth = codes.shape
    data = {"item_id": np.arange(n_items, dtype=np.int32)}
    for i in range(sid_depth):
        data[f"sid_{i + 1}"] = codes[:, i].astype(np.int32)
    pd.DataFrame(data).to_csv(out_path, index=False)


def build_iid2sid_tokens(codes: np.ndarray) -> np.ndarray:
    n_items, sid_depth = codes.shape
    arr = np.zeros((n_items + 1, sid_depth), dtype=np.int32)
    arr[1:] = codes + 1
    return arr


def build_train_samples(
    train_sequences: Dict[int, List[int]],
    iid2sid_tok: np.ndarray,
    max_hist_items: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_items = iid2sid_tok.shape[0] - 1
    sid_depth = iid2sid_tok.shape[1]
    hist_len = max_hist_items * sid_depth

    total = 0
    for seq in train_sequences.values():
        seq = [i for i in seq if 0 <= i < n_items]
        if len(seq) >= 2:
            total += len(seq) - 1
    if total == 0:
        raise ValueError("no training samples from cf train.txt")

    histories = np.zeros((total, hist_len), dtype=np.uint16)
    masks = np.zeros((total, hist_len), dtype=np.uint8)
    labels = np.zeros((total, sid_depth), dtype=np.uint16)
    sample_uids = np.zeros((total,), dtype=np.int32)

    last_idx_by_user: Dict[int, int] = {}
    ptr = 0
    for uid in sorted(train_sequences.keys()):
        seq = [i for i in train_sequences[uid] if 0 <= i < n_items]
        if len(seq) < 2:
            continue
        for t in range(1, len(seq)):
            hist_items = seq[max(0, t - max_hist_items) : t]
            target = seq[t]

            hist_tokens = iid2sid_tok[np.asarray(hist_items, dtype=np.int32) + 1].reshape(-1)
            if hist_tokens.size > hist_len:
                hist_tokens = hist_tokens[-hist_len:]
            pad = hist_len - hist_tokens.size
            if hist_tokens.size > 0:
                histories[ptr, pad:] = hist_tokens.astype(np.uint16)
                masks[ptr, pad:] = 1
            labels[ptr] = iid2sid_tok[target + 1].astype(np.uint16)
            sample_uids[ptr] = int(uid)
            last_idx_by_user[int(uid)] = ptr
            ptr += 1

    if ptr != total:
        histories = histories[:ptr]
        masks = masks[:ptr]
        labels = labels[:ptr]
        sample_uids = sample_uids[:ptr]
        total = ptr

    val_indices = np.array(sorted(last_idx_by_user.values()), dtype=np.int64)
    train_mask = np.ones((total,), dtype=bool)
    train_mask[val_indices] = False
    train_indices = np.where(train_mask)[0].astype(np.int64)
    return histories, masks, labels, sample_uids, train_indices, val_indices


class TigerSIDDataset(Dataset):
    def __init__(self, histories: np.ndarray, masks: np.ndarray, labels: np.ndarray):
        self.histories = torch.from_numpy(histories)
        self.masks = torch.from_numpy(masks)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return int(self.histories.shape[0])

    def __getitem__(self, idx: int):
        return {
            "history": self.histories[idx],
            "attention_mask": self.masks[idx],
            "target": self.labels[idx],
        }


class TigerBackbone(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()
        t5_cfg = T5Config(
            num_layers=int(cfg["num_layers"]),
            num_decoder_layers=int(cfg["num_decoder_layers"]),
            d_model=int(cfg["d_model"]),
            d_ff=int(cfg["d_ff"]),
            num_heads=int(cfg["num_heads"]),
            d_kv=int(cfg["d_kv"]),
            dropout_rate=float(cfg["dropout_rate"]),
            vocab_size=int(cfg["vocab_size"]),
            pad_token_id=int(cfg["pad_token_id"]),
            eos_token_id=int(cfg["eos_token_id"]),
            decoder_start_token_id=int(cfg["pad_token_id"]),
            feed_forward_proj=str(cfg.get("feed_forward_proj", "relu")),
        )
        self.model = T5ForConditionalGeneration(t5_cfg)

    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss, out.logits

    def generate(self, input_ids, attention_mask=None, **kwargs):
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def train_one_epoch(
    model: TigerBackbone,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        input_ids = batch["history"].to(device=device, dtype=torch.long, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device=device, dtype=torch.long, non_blocking=True)
        labels = batch["target"].to(device=device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += float(loss.item())
        n += 1
    return total_loss / max(1, n)


@torch.no_grad()
def evaluate_recall_ndcg(
    model: TigerBackbone,
    loader: DataLoader,
    device: torch.device,
    sid_depth: int,
    topk: int,
    beam_size: int,
) -> Tuple[float, float]:
    model.eval()
    total_r = 0.0
    total_n = 0.0
    total_b = 0
    k = min(topk, beam_size)

    for batch in loader:
        input_ids = batch["history"].to(device=device, dtype=torch.long, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device=device, dtype=torch.long, non_blocking=True)
        labels = batch["target"].to(device=device, dtype=torch.long, non_blocking=True)
        bsz = int(labels.shape[0])

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=beam_size,
            num_return_sequences=beam_size,
            max_length=sid_depth + 1,
            early_stopping=True,
            do_sample=False,
        )
        gen = gen[:, 1 : 1 + sid_depth]
        gen = gen.view(bsz, beam_size, sid_depth)

        match = (gen == labels.unsqueeze(1)).all(dim=-1)[:, :k]
        hit = match.any(dim=1).float()
        total_r += float(hit.sum().item())

        ndcg = torch.zeros((bsz,), device=device, dtype=torch.float32)
        any_hit = match.any(dim=1)
        if any_hit.any():
            first = torch.argmax(match.float(), dim=1)
            ndcg[any_hit] = 1.0 / torch.log2(first[any_hit].float() + 2.0)
        total_n += float(ndcg.sum().item())
        total_b += bsz

    if total_b == 0:
        return 0.0, 0.0
    return total_r / total_b, total_n / total_b


def model_size_to_config(model_size: str) -> Dict:
    if model_size == "mini":
        return {
            "num_layers": 3,
            "num_decoder_layers": 3,
            "d_model": 128,
            "d_ff": 512,
            "num_heads": 4,
            "d_kv": 16,
            "dropout_rate": 0.1,
            "feed_forward_proj": "relu",
        }
    if model_size == "medium":
        return {
            "num_layers": 4,
            "num_decoder_layers": 4,
            "d_model": 128,
            "d_ff": 1024,
            "num_heads": 6,
            "d_kv": 64,
            "dropout_rate": 0.1,
            "feed_forward_proj": "relu",
        }
    if model_size == "large":
        return {
            "num_layers": 6,
            "num_decoder_layers": 6,
            "d_model": 192,
            "d_ff": 1536,
            "num_heads": 8,
            "d_kv": 24,
            "dropout_rate": 0.1,
            "feed_forward_proj": "relu",
        }
    raise ValueError(f"unknown model_size={model_size}")


def write_tiger_args(save_dir: Path) -> None:
    args_path = save_dir / "args.txt"
    if args_path.exists():
        data = json.loads(args_path.read_text(encoding="utf-8"))
    else:
        data = {
            "vis": -1,
            "seed": 101,
            "clear_checkpoints": True,
            "candidate": False,
            "test_only": False,
            "data_path": "../datasets/",
            "dataset": "ml-1m",
            "embed_size": 64,
            "batch_size": 2048,
            "lr": 5e-4,
            "regs": 1e-5,
            "epoch": 1,
            "Ks": 20,
            "verbose": 5,
            "saveID": "lgn",
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
        }
    data["modeltype"] = "TIGER"
    args_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args():
    p = argparse.ArgumentParser(description="Train native TIGER (T5 + SID) on Agent4Rec ml-1m")
    p.add_argument("--root_dir", type=str, default=".")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--sid_depth", type=int, default=4)
    p.add_argument("--codebook_size", type=int, default=32)
    p.add_argument("--kmeans_batch_size", type=int, default=2048)
    p.add_argument(
        "--feature_source",
        type=str,
        default="hybrid",
        choices=["content", "interaction", "hybrid"],
    )
    p.add_argument("--interaction_dim", type=int, default=128)
    p.add_argument("--content_weight", type=float, default=0.8)
    p.add_argument("--interaction_weight", type=float, default=1.2)
    p.add_argument("--max_hist_items", type=int, default=50)
    p.add_argument("--model_size", type=str, default="mini", choices=["mini", "medium", "large"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--beam_size", type=int, default=50)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument(
        "--save_dir",
        type=str,
        default="recommenders/weights/ml-1m/TIGER/Saved",
    )
    p.add_argument(
        "--metrics_out",
        type=str,
        default="baseline/results/tiger_native_ml1m_metrics.json",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    root = Path(args.root_dir).resolve()
    save_dir = (root / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (root / args.metrics_out).resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    train_path = root / "datasets" / "ml-1m" / "cf_data" / "train.txt"
    valid_path = root / "datasets" / "ml-1m" / "cf_data" / "valid.txt"
    test_path = root / "datasets" / "ml-1m" / "cf_data" / "test.txt"
    movie_detail_path = root / "datasets" / "ml-1m" / "simulation" / "movie_detail.csv"

    train_sequences = read_cf_sequences(train_path)
    n_items = infer_n_items([train_path, valid_path, test_path], movie_detail_path)
    print(f"[data] users={len(train_sequences)} n_items={n_items}")

    content_feat = build_content_item_features(movie_detail_path, n_items, train_sequences)
    interaction_feat = build_interaction_item_features(
        n_items=n_items,
        train_sequences=train_sequences,
        n_components=args.interaction_dim,
        seed=args.seed,
    )
    feat = compose_item_features(
        feature_source=args.feature_source,
        content_feat=content_feat,
        interaction_feat=interaction_feat,
        content_weight=args.content_weight,
        interaction_weight=args.interaction_weight,
    )
    print(
        f"[sid] feature_source={args.feature_source} "
        f"content_dim={content_feat.shape[1]} interaction_dim={interaction_feat.shape[1]} "
        f"final_dim={feat.shape[1]}"
    )
    codes, layer_errors = residual_kmeans_sid(
        features=feat,
        sid_depth=args.sid_depth,
        codebook_size=args.codebook_size,
        seed=args.seed,
        batch_size=args.kmeans_batch_size,
    )
    sid_map_path = save_dir / "sid_mapping_internal.csv"
    save_sid_mapping(codes, sid_map_path)
    (save_dir / "sid_config_internal.json").write_text(
        json.dumps(
            {
                "sid_depth": args.sid_depth,
                "codebook_size": args.codebook_size,
                "kmeans_batch_size": args.kmeans_batch_size,
                "feature_source": args.feature_source,
                "interaction_dim": args.interaction_dim,
                "content_weight": args.content_weight,
                "interaction_weight": args.interaction_weight,
                "content_dim": int(content_feat.shape[1]),
                "interaction_dim_real": int(interaction_feat.shape[1]),
                "feature_dim": int(feat.shape[1]),
                "layer_residual_mse": layer_errors,
                "n_items": n_items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[sid] saved mapping: {sid_map_path}")

    iid2sid_tok = build_iid2sid_tokens(codes)
    histories, masks, labels, sample_uids, train_indices, val_indices = build_train_samples(
        train_sequences=train_sequences,
        iid2sid_tok=iid2sid_tok,
        max_hist_items=args.max_hist_items,
    )
    print(
        f"[sample] total={len(histories)} "
        f"train={len(train_indices)} val={len(val_indices)} users={len(np.unique(sample_uids))}"
    )

    dataset = TigerSIDDataset(histories=histories, masks=masks, labels=labels)
    train_ds = Subset(dataset, train_indices.tolist())
    val_ds = Subset(dataset, val_indices.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    cfg = model_size_to_config(args.model_size)
    cfg["vocab_size"] = int(args.codebook_size + 1)
    cfg["pad_token_id"] = 0
    cfg["eos_token_id"] = 0

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = TigerBackbone(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_recall = -1.0
    best_epoch = 0
    patience_left = int(args.patience)
    history = []

    best_ckpt_path = save_dir / "epoch=best.tiger.native.pth"
    last_ckpt_path = save_dir / "tiger_native_last.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )
        val_recall, val_ndcg = evaluate_recall_ndcg(
            model=model,
            loader=val_loader,
            device=device,
            sid_depth=args.sid_depth,
            topk=args.topk,
            beam_size=args.beam_size,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_recall": float(val_recall),
                "val_ndcg": float(val_ndcg),
            }
        )
        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.5f} "
            f"val_recall@{args.topk}={val_recall:.5f} "
            f"val_ndcg@{args.topk}={val_ndcg:.5f}"
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "sid_depth": int(args.sid_depth),
            "codebook_size": int(args.codebook_size),
            "config": cfg,
            "val_recall_at_k": float(val_recall),
            "val_ndcg_at_k": float(val_ndcg),
            "topk": int(args.topk),
            "beam_size": int(args.beam_size),
        }
        torch.save(ckpt_payload, last_ckpt_path)

        if val_recall > best_recall:
            best_recall = float(val_recall)
            best_epoch = int(epoch)
            patience_left = int(args.patience)
            torch.save(ckpt_payload, best_ckpt_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[train] early stop at epoch={epoch}")
                break

    write_tiger_args(save_dir)

    metrics = {
        "config": {
            "seed": args.seed,
            "device": str(device),
            "sid_depth": args.sid_depth,
            "codebook_size": args.codebook_size,
            "feature_source": args.feature_source,
            "interaction_dim": args.interaction_dim,
            "content_weight": args.content_weight,
            "interaction_weight": args.interaction_weight,
            "max_hist_items": args.max_hist_items,
            "model_size": args.model_size,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
            "beam_size": args.beam_size,
            "topk": args.topk,
            "n_items": n_items,
            "n_users": len(train_sequences),
            "n_samples_total": len(histories),
            "n_samples_train": len(train_indices),
            "n_samples_val": len(val_indices),
        },
        "best_epoch": int(best_epoch),
        "best_val_recall": float(best_recall),
        "sid_mapping": str(sid_map_path),
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "history": history,
        "sid_layer_residual_mse": layer_errors,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[done] best checkpoint: {best_ckpt_path}")
    print(f"[done] sid mapping: {sid_map_path}")
    print(f"[done] metrics: {metrics_path}")


if __name__ == "__main__":
    main()
