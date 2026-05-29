from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.train_hazard_plan_reranker import (  # noqa: E402
    build_credit_reward,
    compute_discounted_returns,
    discover_run_dirs,
    load_tables,
    parse_interview_rating,
    parse_metrics_txt,
    train_credit_value_model,
)
from baseline.train_tiger_native_cf import infer_n_items, read_cf_sequences  # noqa: E402
from recommenders.models.TIGER import TIGER, TokenCreditTransportHead  # noqa: E402
from simulation.hazard_plan import (  # noqa: E402
    build_planner_feature_row,
    build_user_profile,
    heuristic_negative_increment,
    initial_rollout_state,
    parse_float,
    planner_row_to_vector,
    summarize_rollout_state,
    to_int_list,
    update_rollout_state,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_item_list(train_sequences: Dict[int, List[int]]) -> Dict[int, List[int]]:
    item_users: Dict[int, List[int]] = {}
    for uid, seq in train_sequences.items():
        for iid in seq:
            item_users.setdefault(int(iid), []).append(int(uid))
    return item_users


class TransportRecordDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[int(idx)]
        return (
            torch.tensor(rec["input_ids"], dtype=torch.long),
            torch.tensor(rec["attention_mask"], dtype=torch.long),
            torch.tensor(rec["target_tokens"], dtype=torch.long),
            torch.tensor(rec["block_credit"], dtype=torch.float32),
            torch.tensor(float(rec["page_credit"]), dtype=torch.float32),
        )


def collate_transport(batch):
    input_ids = torch.stack([row[0] for row in batch], dim=0)
    attention_mask = torch.stack([row[1] for row in batch], dim=0)
    target_tokens = torch.stack([row[2] for row in batch], dim=0)
    block_credit = torch.stack([row[3] for row in batch], dim=0)
    page_credit = torch.stack([row[4] for row in batch], dim=0)
    return input_ids, attention_mask, target_tokens, block_credit, page_credit


def build_transport_records(
    *,
    dataset: str,
    cf_data_subdir: str,
    run_dirs: Sequence[Path],
    tiger_model_path: str,
    device: str,
    only_items_per_page: int,
    credit_gamma: float,
    credit_reward_weights: Dict[str, float],
    seed: int,
    valid_ratio: float,
) -> Dict[str, Any]:
    persona_df, user_statistic, movie_detail = load_tables(dataset)
    cf_dir = REPO_ROOT / "datasets" / dataset / cf_data_subdir
    train_path = cf_dir / "train.txt"
    valid_path = cf_dir / "valid.txt"
    test_path = cf_dir / "test.txt"
    train_sequences = read_cf_sequences(train_path)
    n_items = infer_n_items([train_path, valid_path, test_path], REPO_ROOT / "datasets" / dataset / "simulation" / "movie_detail.csv")
    item_users = build_train_item_list(train_sequences)
    tiger_data = SimpleNamespace(
        n_items=n_items,
        train_user_list=train_sequences,
        train_item_list=item_users,
    )
    tiger_args = SimpleNamespace(
        dataset=str(dataset),
        cuda=str(device),
        model_path=str(tiger_model_path),
        tiger_attr_profile="default",
        tiger_transport_enabled="off",
    )
    tiger = TIGER(tiger_args, tiger_data)
    if tiger.backbone is None or tiger.iid2sid_tok is None or tiger.sid_depth is None:
        raise RuntimeError("Failed to load native TIGER backbone or SID mapping for transport training.")
    tiger.backbone.eval()
    for param in tiger.backbone.parameters():
        param.requires_grad = False

    raw_records: List[Dict[str, Any]] = []
    state_rows: List[np.ndarray] = []
    return_targets: List[float] = []
    value_groups: List[str] = []
    stats = {
        "runs_used": [str(p) for p in run_dirs],
        "users_total": 0,
        "pages_total": 0,
        "records_kept": 0,
    }

    for run_dir in run_dirs:
        behavior_dir = run_dir / "behavior"
        interview_dir = run_dir / "interview"
        if not behavior_dir.exists():
            continue
        metrics = parse_metrics_txt(run_dir / "metrics.txt")
        max_pages_cfg = int(parse_float(metrics.get("Maximum exit page"), 0.0))

        for pkl_path in sorted(behavior_dir.glob("*.pkl")):
            uid = int(pkl_path.stem)
            if uid >= len(persona_df) or uid not in user_statistic.index:
                continue
            with pkl_path.open("rb") as f:
                behavior = pickle.load(f)
            if not isinstance(behavior, dict):
                continue
            page_keys = sorted([k for k in behavior.keys() if isinstance(k, int)])
            if not page_keys:
                continue

            stats["users_total"] += 1
            user_profile = build_user_profile(persona_df.iloc[uid], user_statistic.loc[uid], uid)
            state = initial_rollout_state()
            history_items = list(train_sequences.get(uid, []))
            user_records: List[Dict[str, Any]] = []
            max_pages = max(max_pages_cfg, max(page_keys))
            interview_rating = 0.0
            interview_path = interview_dir / f"{uid}.pkl"
            if interview_path.exists():
                try:
                    with interview_path.open("rb") as f:
                        interview_rating = parse_interview_rating(pickle.load(f))
                except Exception:
                    interview_rating = 0.0

            for page_no in page_keys:
                stats["pages_total"] += 1
                info = behavior.get(page_no, {})
                rec_ids = [iid for iid in to_int_list(info.get("recommended_id")) if 0 <= int(iid) < n_items]
                state_summary = summarize_rollout_state(state, page_index=int(page_no), max_pages=max_pages)
                planner_row = build_planner_feature_row(user_profile, state_summary)

                align_ids = [iid for iid in to_int_list(info.get("align_id")) if 0 <= int(iid) < n_items]
                watch_ids = [iid for iid in to_int_list(info.get("watch_id")) if 0 <= int(iid) < n_items]
                ratings = [parse_float(v, 0.0) for v in info.get("rating", [])]
                ratings = [r for r in ratings if r > 0]
                avg_rating = float(np.mean(ratings)) if ratings else 0.0
                align_count = len(align_ids)
                watch_count = len(watch_ids)
                negative_increment = heuristic_negative_increment(watch_count, avg_rating)

                if (only_items_per_page <= 0 or len(rec_ids) == int(only_items_per_page)) and rec_ids:
                    exposed_iid = int(rec_ids[0])
                    immediate_reward = build_credit_reward(
                        watch_count=watch_count,
                        align_count=align_count,
                        avg_rating=avg_rating,
                        negative_increment=negative_increment,
                        continued=bool(int(page_no) < int(page_keys[-1])),
                        reward_weights=credit_reward_weights,
                    )
                    user_records.append(
                        {
                            "uid": int(uid),
                            "group": f"{run_dir.name}:{uid}",
                            "page_index": int(page_no),
                            "state_vec": planner_row_to_vector(planner_row),
                            "history_items": list(history_items),
                            "item_id": int(exposed_iid),
                            "reward": float(immediate_reward),
                        }
                    )

                update_rollout_state(
                    state,
                    align_count=align_count,
                    watch_count=watch_count,
                    avg_rating=avg_rating,
                    negative_increment=negative_increment,
                )
                if align_ids:
                    history_items.extend(int(iid) for iid in align_ids)

            terminal_reward = float(credit_reward_weights.get("terminal", 0.0)) * float(
                max(min(interview_rating / 10.0, 1.0), 0.0)
            )
            if not user_records:
                continue
            immediate_rewards = [float(rec["reward"]) for rec in user_records]
            discounted_returns = compute_discounted_returns(
                immediate_rewards,
                gamma=float(credit_gamma),
                terminal_reward=float(terminal_reward),
            )
            for rec_idx, rec in enumerate(user_records):
                record = dict(rec)
                record["return_target"] = float(discounted_returns[rec_idx])
                raw_records.append(record)
                state_rows.append(np.asarray(rec["state_vec"], dtype=np.float32))
                return_targets.append(float(discounted_returns[rec_idx]))
                value_groups.append(str(rec["group"]))

    if not raw_records:
        raise ValueError("No usable session records were extracted for transport training.")

    X_state = np.vstack(state_rows).astype(np.float32)
    y_return = np.asarray(return_targets, dtype=np.float32)
    value_model, value_metrics, value_pred = train_credit_value_model(
        X=X_state,
        y=y_return,
        groups=value_groups,
        seed=int(seed),
        valid_ratio=float(valid_ratio),
    )
    _ = value_model
    value_pred = np.asarray(value_pred, dtype=np.float32).reshape(-1)

    final_records: List[Dict[str, Any]] = []
    for idx, rec in enumerate(raw_records):
        history_items = [int(i) for i in rec["history_items"] if 0 <= int(i) < n_items]
        item_id = int(rec["item_id"])
        input_ids, attention_mask = tiger._build_history_tokens(history_items)
        if input_ids is None:
            continue
        target_tokens = tiger.iid2sid_tok[item_id + 1].astype(np.int64).tolist()
        page_credit = float(rec["return_target"] - value_pred[idx])
        transport = tiger.build_credit_transport_targets(history_items, item_id, page_credit)
        final_records.append(
            {
                "group": str(rec["group"]),
                "uid": int(rec["uid"]),
                "page_index": int(rec["page_index"]),
                "input_ids": list(input_ids),
                "attention_mask": list(attention_mask),
                "target_tokens": [int(v) for v in target_tokens],
                "page_credit": float(page_credit),
                "block_credit": [float(v) for v in transport.get("block_credit", [])],
                "transport": transport,
            }
        )

    stats["records_kept"] = int(len(final_records))
    return {
        "tiger": tiger,
        "records": final_records,
        "value_metrics": value_metrics,
        "stats": stats,
    }


def evaluate_head(
    *,
    tiger: TIGER,
    head: TokenCreditTransportHead,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    head.eval()
    total_rows = 0
    block_sse = 0.0
    page_sse = 0.0
    page_abs = 0.0
    gap_sum = 0.0
    with torch.no_grad():
        for input_ids, attention_mask, target_tokens, block_credit, page_credit in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_tokens = target_tokens.to(device)
            block_credit = block_credit.to(device)
            page_credit = page_credit.to(device)
            decoder_input_ids = torch.cat(
                [
                    torch.zeros((target_tokens.shape[0], 1), dtype=torch.long, device=device),
                    target_tokens[:, :-1],
                ],
                dim=1,
            )
            _, hidden = tiger.backbone.decode_with_hidden(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
            pred = head(hidden.detach(), target_tokens)
            total_rows += int(target_tokens.shape[0])
            block_sse += float(torch.sum((pred - block_credit) ** 2).item())
            page_pred = torch.sum(pred, dim=1)
            page_sse += float(torch.sum((page_pred - page_credit) ** 2).item())
            page_abs += float(torch.sum(torch.abs(page_pred - page_credit)).item())
            gap_sum += float(torch.sum(torch.abs(page_pred - torch.sum(pred, dim=1))).item())
    denom_block = max(total_rows * max(int(tiger.sid_depth or 1), 1), 1)
    denom_page = max(total_rows, 1)
    return {
        "block_mse": float(block_sse / float(denom_block)),
        "page_mse": float(page_sse / float(denom_page)),
        "page_mae": float(page_abs / float(denom_page)),
        "conservation_gap": float(gap_sum / float(denom_page)),
    }


def train_transport_head(
    *,
    tiger: TIGER,
    records: Sequence[Dict[str, Any]],
    batch_size: int,
    epochs: int,
    lr: float,
    valid_ratio: float,
    token_dim: int,
    mlp_dim: int,
    conservation_scale: float,
    sign_scale: float,
    seed: int,
    device: torch.device,
    patience: int,
) -> Dict[str, Any]:
    dataset = TransportRecordDataset(records)
    groups = [str(rec["group"]) for rec in records]
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=max(min(float(valid_ratio), 0.5), 0.05),
        random_state=int(seed),
    )
    indices = np.arange(len(records))
    train_idx, valid_idx = next(splitter.split(indices, groups=groups, y=np.zeros(len(records), dtype=np.int64)))
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_transport,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_idx.tolist()),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_transport,
    )

    head = TokenCreditTransportHead(
        hidden_size=int(tiger.backbone.model.config.d_model),
        vocab_size=int(tiger.backbone.model.config.vocab_size),
        token_dim=int(token_dim),
        mlp_dim=int(mlp_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(lr))

    best_metrics: Dict[str, float] = {}
    best_state = None
    best_key = float("inf")
    best_epoch = 0
    epochs_since_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        head.train()
        losses = []
        for input_ids, attention_mask, target_tokens, block_credit, page_credit in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_tokens = target_tokens.to(device)
            block_credit = block_credit.to(device)
            page_credit = page_credit.to(device)
            decoder_input_ids = torch.cat(
                [
                    torch.zeros((target_tokens.shape[0], 1), dtype=torch.long, device=device),
                    target_tokens[:, :-1],
                ],
                dim=1,
            )
            with torch.no_grad():
                _, hidden = tiger.backbone.decode_with_hidden(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
                hidden = hidden.detach()
            pred = head(hidden, target_tokens)
            page_pred = torch.sum(pred, dim=1)
            loss_block = F.smooth_l1_loss(pred, block_credit)
            loss_cons = F.mse_loss(page_pred, page_credit)
            target_sign = (block_credit >= 0.0).float()
            loss_sign = F.binary_cross_entropy_with_logits(pred, target_sign)
            loss = loss_block + float(conservation_scale) * loss_cons + float(sign_scale) * loss_sign
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        valid_metrics = evaluate_head(tiger=tiger, head=head, loader=valid_loader, device=device)
        valid_metrics["train_loss"] = float(np.mean(losses)) if losses else 0.0
        valid_metrics["epoch"] = float(epoch)
        history.append(dict(valid_metrics))
        key = float(valid_metrics["page_mse"] + 0.5 * valid_metrics["block_mse"])
        if key < best_key:
            best_key = key
            best_metrics = dict(valid_metrics)
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
            best_epoch = int(epoch)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if int(patience) > 0 and epochs_since_improve >= int(patience):
            break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
        best_metrics = evaluate_head(tiger=tiger, head=head, loader=valid_loader, device=device)

    return {
        "state_dict": best_state,
        "best_metrics": best_metrics,
        "history": history,
        "best_epoch": int(best_epoch) if best_epoch > 0 else int(len(history)),
        "split": {
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train a strict session-to-token credit transport head for native TIGER.")
    parser.add_argument("--dataset", type=str, default="all-beauty")
    parser.add_argument("--cf_data_subdir", type=str, default="cf_data")
    parser.add_argument("--run_dirs", nargs="*", default=[])
    parser.add_argument("--name_contains", type=str, default="")
    parser.add_argument("--tiger_model_path", type=str, default="Saved")
    parser.add_argument("--only_items_per_page", type=int, default=1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--conservation_scale", type=float, default=0.25)
    parser.add_argument("--sign_scale", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--credit_gamma", type=float, default=0.90)
    parser.add_argument("--credit_continue_weight", type=float, default=0.40)
    parser.add_argument("--credit_watch_weight", type=float, default=0.35)
    parser.add_argument("--credit_align_weight", type=float, default=0.20)
    parser.add_argument("--credit_rating_weight", type=float, default=0.25)
    parser.add_argument("--credit_negative_weight", type=float, default=0.55)
    parser.add_argument("--credit_terminal_weight", type=float, default=0.30)
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--metrics_out", type=str, default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))

    if args.run_dirs:
        run_dirs = [(REPO_ROOT / p).resolve() if not Path(p).is_absolute() else Path(p).resolve() for p in args.run_dirs]
    else:
        run_dirs = discover_run_dirs(args.dataset, name_contains=str(args.name_contains))
    if not run_dirs:
        raise ValueError("No session run directories found for transport training.")

    credit_reward_weights = {
        "continue": float(args.credit_continue_weight),
        "watch": float(args.credit_watch_weight),
        "align": float(args.credit_align_weight),
        "rating": float(args.credit_rating_weight),
        "negative": float(args.credit_negative_weight),
        "terminal": float(args.credit_terminal_weight),
    }
    build_out = build_transport_records(
        dataset=str(args.dataset),
        cf_data_subdir=str(args.cf_data_subdir),
        run_dirs=run_dirs,
        tiger_model_path=str(args.tiger_model_path),
        device=str(args.device),
        only_items_per_page=int(args.only_items_per_page),
        credit_gamma=float(args.credit_gamma),
        credit_reward_weights=credit_reward_weights,
        seed=int(args.seed),
        valid_ratio=float(args.valid_ratio),
    )
    tiger: TIGER = build_out["tiger"]
    records = build_out["records"]
    device = torch.device(str(args.device))
    train_out = train_transport_head(
        tiger=tiger,
        records=records,
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        valid_ratio=float(args.valid_ratio),
        token_dim=int(args.token_dim),
        mlp_dim=int(args.mlp_dim),
        conservation_scale=float(args.conservation_scale),
        sign_scale=float(args.sign_scale),
        seed=int(args.seed),
        device=device,
        patience=int(args.patience),
    )

    save_dir = Path(args.save_dir) if args.save_dir else REPO_ROOT / "recommenders" / "weights" / args.dataset / "TIGER" / args.tiger_model_path
    if not save_dir.is_absolute():
        save_dir = (REPO_ROOT / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    head_path = save_dir / "credit_transport_head.pt"
    meta_path = save_dir / "credit_transport_meta.json"
    torch.save({"model_state_dict": train_out["state_dict"]}, head_path)
    meta = {
        "dataset": str(args.dataset),
        "cf_data_subdir": str(args.cf_data_subdir),
        "model_path": str(args.tiger_model_path),
        "sid_depth": int(tiger.sid_depth or 0),
        "hidden_size": int(tiger.backbone.model.config.d_model),
        "vocab_size": int(tiger.backbone.model.config.vocab_size),
        "token_dim": int(args.token_dim),
        "mlp_dim": int(args.mlp_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "conservation_scale": float(args.conservation_scale),
        "sign_scale": float(args.sign_scale),
        "best_metrics": train_out["best_metrics"],
        "best_epoch": int(train_out["best_epoch"]),
        "split": train_out["split"],
        "value_metrics": build_out["value_metrics"],
        "extract_stats": build_out["stats"],
        "credit_gamma": float(args.credit_gamma),
        "credit_reward_weights": credit_reward_weights,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    metrics_out = Path(args.metrics_out) if args.metrics_out else REPO_ROOT / "baseline" / "results" / f"tiger_credit_transport_{args.dataset}.json"
    if not metrics_out.is_absolute():
        metrics_out = (REPO_ROOT / metrics_out).resolve()
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(
        json.dumps(
            {
                "head_path": str(head_path),
                "meta_path": str(meta_path),
                "best_metrics": train_out["best_metrics"],
                "best_epoch": int(train_out["best_epoch"]),
                "history": train_out["history"],
                "split": train_out["split"],
                "value_metrics": build_out["value_metrics"],
                "extract_stats": build_out["stats"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[transport] saved head to {head_path}")
    print(f"[transport] metrics written to {metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
