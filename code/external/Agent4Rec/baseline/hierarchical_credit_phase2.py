from __future__ import annotations

import json
import os
import pickle
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.hierarchical_credit_phase1 import (  # noqa: E402
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_WELFARE_WEIGHTS,
    _normalized_future_targets,
    compose_phase1_feature,
    parse_weight_overrides,
    set_seed,
    ScalarCritic,
)
from baseline.train_hazard_plan_reranker import (  # noqa: E402
    build_credit_reward,
    compute_discounted_returns,
    discover_run_dirs,
    load_tables,
    parse_interview_rating,
    parse_metrics_txt,
)
from baseline.train_tiger_native_cf import infer_n_items, read_cf_sequences  # noqa: E402
from recommenders.models.TIGER import TIGER, TokenCreditTransportHead  # noqa: E402
from simulation.hazard_plan import (  # noqa: E402
    build_feature_row,
    build_item_catalog,
    build_planner_feature_row,
    build_user_profile,
    choose_plan,
    heuristic_negative_increment,
    initial_rollout_state,
    parse_float,
    planner_row_to_vector,
    row_to_vector,
    summarize_rollout_state,
    to_int_list,
    update_rollout_state,
)


def build_train_item_list(train_sequences: Dict[int, List[int]]) -> Dict[int, List[int]]:
    item_users: Dict[int, List[int]] = {}
    for uid, seq in train_sequences.items():
        for iid in seq:
            item_users.setdefault(int(iid), []).append(int(uid))
    return item_users


def build_tiger_runtime(
    *,
    dataset: str,
    cf_data_subdir: str,
    tiger_model_path: str,
    device: str,
) -> Tuple[TIGER, Dict[int, List[int]], int]:
    cf_dir = REPO_ROOT / "datasets" / dataset / cf_data_subdir
    train_path = cf_dir / "train.txt"
    valid_path = cf_dir / "valid.txt"
    test_path = cf_dir / "test.txt"
    train_sequences = read_cf_sequences(train_path)
    n_items = infer_n_items(
        [train_path, valid_path, test_path],
        REPO_ROOT / "datasets" / dataset / "simulation" / "movie_detail.csv",
    )
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
    prev_cwd = os.getcwd()
    try:
        os.chdir(str(REPO_ROOT))
        tiger = TIGER(tiger_args, tiger_data)
    finally:
        os.chdir(prev_cwd)
    if tiger.backbone is None or tiger.iid2sid_tok is None or tiger.sid_depth is None:
        raise RuntimeError("Failed to load native TIGER backbone or SID mapping for Phase 2 segment transport.")
    tiger.backbone.eval()
    for param in tiger.backbone.parameters():
        param.requires_grad = False
    return tiger, train_sequences, int(n_items)


def build_phase2_segment_artifact(
    *,
    dataset: str,
    run_dirs: Sequence[Path],
    cf_data_subdir: str,
    tiger_model_path: str,
    device: str,
    only_items_per_page: int,
    future_horizon: int,
    credit_gamma: float,
    credit_reward_weights: Dict[str, float],
    welfare_weights: Dict[str, float],
    welfare_scale: float,
    max_history_items: int,
    item_credit_source: str = "manual_mix",
    phase1_item_critic_path: str = "",
    item_credit_blend_alpha: float = 0.5,
) -> Dict[str, Any]:
    persona_df, user_statistic, movie_detail = load_tables(dataset)
    item_catalog = build_item_catalog(movie_detail)
    tiger, train_sequences, n_items = build_tiger_runtime(
        dataset=dataset,
        cf_data_subdir=cf_data_subdir,
        tiger_model_path=tiger_model_path,
        device=device,
    )
    phase1_item_critic = None
    phase1_item_critic_device = torch.device("cpu")
    phase1_item_critic_meta: Dict[str, Any] | None = None
    source_name = str(item_credit_source or "manual_mix").strip().lower()
    if source_name in {"phase1_item_critic", "blend"}:
        critic_path = Path(str(phase1_item_critic_path or "")).expanduser()
        if not critic_path.is_absolute():
            critic_path = (REPO_ROOT / critic_path).resolve()
        if not critic_path.exists():
            raise FileNotFoundError(f"Phase 1 item critic not found: {critic_path}")
        critic_payload = torch.load(critic_path, map_location="cpu")
        critic_meta = dict(critic_payload.get("meta", {}))
        feature_dim = int(critic_meta.get("feature_dim", 0))
        state_dict = critic_payload["state_dict"]
        hidden_dim = int(state_dict["backbone.0.net.0.weight"].shape[0])
        depth = int(len([k for k in state_dict.keys() if k.endswith("net.0.weight")]))
        phase1_item_critic = ScalarCritic(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            dropout=0.0,
        )
        phase1_item_critic.load_state_dict(state_dict)
        phase1_item_critic.eval()
        phase1_item_critic_meta = {
            "path": str(critic_path),
            **critic_meta,
        }

    records: List[Dict[str, Any]] = []
    stats = {
        "runs_used": [str(p) for p in run_dirs],
        "users_total": 0,
        "pages_total": 0,
        "records_kept": 0,
        "sid_depth": int(tiger.sid_depth),
        "n_items": int(n_items),
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
            history_items = [int(i) for i in train_sequences.get(uid, []) if 0 <= int(i) < n_items]
            max_pages = max(max_pages_cfg, max(page_keys))
            interview_rating = 0.0
            interview_path = interview_dir / f"{uid}.pkl"
            if interview_path.exists():
                try:
                    with interview_path.open("rb") as f:
                        interview_rating = parse_interview_rating(pickle.load(f))
                except Exception:
                    interview_rating = 0.0
            interview_norm = float(np.clip(interview_rating / 10.0, 0.0, 1.0))

            page_records: List[Dict[str, Any]] = []
            immediate_rewards: List[float] = []

            for page_no in page_keys:
                stats["pages_total"] += 1
                info = behavior.get(page_no, {})
                rec_ids = [iid for iid in to_int_list(info.get("recommended_id")) if 0 <= int(iid) < n_items]
                align_ids = [iid for iid in to_int_list(info.get("align_id")) if 0 <= int(iid) < n_items]
                watch_ids = [iid for iid in to_int_list(info.get("watch_id")) if 0 <= int(iid) < n_items]
                ratings = [parse_float(v, 0.0) for v in info.get("rating", [])]
                ratings = [r for r in ratings if r > 0]
                avg_rating = float(np.mean(ratings)) if ratings else 0.0
                align_count = len(align_ids)
                watch_count = len(watch_ids)
                negative_increment = heuristic_negative_increment(watch_count, avg_rating)

                state_summary = summarize_rollout_state(state, page_index=int(page_no), max_pages=max_pages)
                planner_row = build_planner_feature_row(user_profile, state_summary)
                planner_vec = planner_row_to_vector(planner_row)
                plan_name = choose_plan(user_profile, state_summary, override="auto")

                selected_iid = int(rec_ids[0]) if rec_ids else -1
                item_row = None
                item_vec = None
                if selected_iid >= 0 and (only_items_per_page <= 0 or len(rec_ids) == int(only_items_per_page)):
                    item_info = item_catalog.get(selected_iid, {})
                    item_row = build_feature_row(user_profile, state_summary, item_info, plan_name)
                    item_vec = row_to_vector(item_row).astype(np.float32)

                rating_norm = float(np.clip(avg_rating / 5.0, 0.0, 1.0)) if watch_count > 0 else 0.0
                watch_flag = 1.0 if watch_count > 0 or avg_rating > 0 else 0.0
                negative_flag = 1.0 if int(negative_increment) > 0 else 0.0
                immediate_reward = build_credit_reward(
                    watch_count=watch_count,
                    align_count=align_count,
                    avg_rating=avg_rating,
                    negative_increment=negative_increment,
                    continued=bool(int(page_no) < int(page_keys[-1])),
                    reward_weights=credit_reward_weights,
                )
                page_records.append(
                    {
                        "page_no": int(page_no),
                        "uid": int(uid),
                        "run_name": str(run_dir.name),
                        "group": f"{run_dir.name}:{uid}",
                        "planner_vec": planner_vec.astype(np.float32),
                        "plan_name": str(plan_name),
                        "item_vec": item_vec,
                        "selected_item_id": int(selected_iid),
                        "history_items": list(history_items[-max(int(max_history_items), 1) :]),
                        "watch_flag": float(watch_flag),
                        "rating_norm": float(rating_norm),
                        "negative_flag": float(negative_flag),
                        "exit_page": int(max(page_keys)),
                        "interview_norm": float(interview_norm),
                    }
                )
                immediate_rewards.append(float(immediate_reward))

                update_rollout_state(
                    state,
                    align_count=align_count,
                    watch_count=watch_count,
                    avg_rating=avg_rating,
                    negative_increment=negative_increment,
                )
                if align_ids:
                    history_items.extend(int(iid) for iid in align_ids)

            if not page_records:
                continue

            terminal_reward = 0.20 * float(interview_norm)
            discounted_returns = compute_discounted_returns(
                immediate_rewards,
                gamma=float(credit_gamma),
                terminal_reward=float(terminal_reward),
            )

            for idx, row in enumerate(page_records):
                item_id = int(row["selected_item_id"])
                item_vec = row.get("item_vec")
                history_snapshot = [int(i) for i in row.get("history_items", []) if 0 <= int(i) < n_items]
                if item_id < 0 or item_vec is None:
                    continue
                input_ids, attention_mask = tiger._build_history_tokens(history_snapshot)
                if input_ids is None:
                    continue
                future_targets = _normalized_future_targets(
                    page_records,
                    index=idx,
                    future_horizon=future_horizon,
                    max_pages=max_pages,
                    interview_norm=interview_norm,
                    welfare_weights=welfare_weights,
                )
                discounted_return = float(discounted_returns[idx])
                future_welfare = float(future_targets["future_welfare"])
                manual_item_credit = discounted_return + float(welfare_scale) * future_welfare
                critic_item_credit = None
                if phase1_item_critic is not None:
                    item_feature = compose_phase1_feature(
                        planner_vec=np.asarray(row["planner_vec"], dtype=np.float32),
                        item_vec=np.asarray(item_vec, dtype=np.float32),
                        plan_name=str(row["plan_name"]),
                        level_name="item",
                    )
                    with torch.no_grad():
                        critic_item_credit = float(
                            phase1_item_critic(
                                torch.as_tensor(item_feature.reshape(1, -1), dtype=torch.float32, device=phase1_item_critic_device)
                            )
                            .detach()
                            .cpu()
                            .item()
                        )
                if source_name == "phase1_item_critic":
                    item_credit = float(critic_item_credit if critic_item_credit is not None else manual_item_credit)
                elif source_name == "blend":
                    alpha = float(np.clip(item_credit_blend_alpha, 0.0, 1.0))
                    critic_term = float(critic_item_credit if critic_item_credit is not None else manual_item_credit)
                    item_credit = float((1.0 - alpha) * manual_item_credit + alpha * critic_term)
                else:
                    item_credit = float(manual_item_credit)
                target_tokens = tiger.iid2sid_tok[item_id + 1].astype(np.int64).tolist()
                transport = tiger.build_credit_transport_targets(history_snapshot, item_id, item_credit)
                records.append(
                    {
                        "record_id": f"{run_dir.name}:{uid}:{row['page_no']}:{item_id}",
                        "group": str(row["group"]),
                        "run_name": str(run_dir.name),
                        "uid": int(uid),
                        "page_index": int(row["page_no"]),
                        "exit_page": int(row["exit_page"]),
                        "interview_norm": float(interview_norm),
                        "plan_name": str(row["plan_name"]),
                        "input_ids": [int(v) for v in list(input_ids)],
                        "attention_mask": [int(v) for v in list(attention_mask)],
                        "target_tokens": [int(v) for v in target_tokens],
                        "history_items": list(history_snapshot),
                        "selected_item_id": int(item_id),
                        "item_credit": float(item_credit),
                        "manual_item_credit": float(manual_item_credit),
                        "critic_item_credit": float(critic_item_credit) if critic_item_credit is not None else None,
                        "discounted_return": float(discounted_return),
                        "future_welfare": float(future_welfare),
                        "block_credit": [float(v) for v in transport.get("block_credit", [])],
                        "history_credit": [float(v) for v in transport.get("history_credit", [])],
                        "positive_mass": float(transport.get("positive_mass", 0.0)),
                        "negative_mass": float(transport.get("negative_mass", 0.0)),
                        "transport": transport,
                        "conservation_gap": float(transport.get("conservation_gap", 0.0)),
                    }
                )

    if not records:
        raise ValueError("No usable records were extracted for Phase 2 segment transport.")
    stats["records_kept"] = int(len(records))
    return {
        "meta": {
            "dataset": str(dataset),
            "run_dirs": [str(p) for p in run_dirs],
            "cf_data_subdir": str(cf_data_subdir),
            "tiger_model_path": str(tiger_model_path),
            "future_horizon": int(future_horizon),
            "credit_gamma": float(credit_gamma),
            "credit_reward_weights": dict(credit_reward_weights),
            "welfare_weights": dict(welfare_weights),
            "welfare_scale": float(welfare_scale),
            "only_items_per_page": int(only_items_per_page),
            "max_history_items": int(max_history_items),
            "item_credit_source": str(source_name),
            "phase1_item_critic": phase1_item_critic_meta,
            "item_credit_blend_alpha": float(item_credit_blend_alpha),
            "sid_depth": int(tiger.sid_depth),
            "n_items": int(n_items),
        },
        "stats": stats,
        "records": records,
    }


def save_phase2_segment_artifact(artifact: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def load_phase2_segment_artifact(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


def describe_phase2_segment_artifact(artifact: Dict[str, Any]) -> str:
    meta = artifact.get("meta", {})
    stats = artifact.get("stats", {})
    rows = int(len(artifact.get("records", [])))
    return "\n".join(
        [
            json.dumps(meta, ensure_ascii=False),
            json.dumps(stats, ensure_ascii=False),
            f"segment_records: n={rows} sid_depth={int(meta.get('sid_depth', 0))}",
        ]
    )


class SegmentTransportDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]]):
        self.records = list(records)

    def __len__(self) -> int:
        return int(len(self.records))

    def __getitem__(self, idx: int):
        rec = self.records[int(idx)]
        return (
            torch.tensor(rec["input_ids"], dtype=torch.long),
            torch.tensor(rec["attention_mask"], dtype=torch.long),
            torch.tensor(rec["target_tokens"], dtype=torch.long),
            torch.tensor(rec["block_credit"], dtype=torch.float32),
            torch.tensor(float(rec["item_credit"]), dtype=torch.float32),
        )


def collate_segment_transport(batch):
    input_ids = torch.stack([row[0] for row in batch], dim=0)
    attention_mask = torch.stack([row[1] for row in batch], dim=0)
    target_tokens = torch.stack([row[2] for row in batch], dim=0)
    block_credit = torch.stack([row[3] for row in batch], dim=0)
    item_credit = torch.stack([row[4] for row in batch], dim=0)
    return input_ids, attention_mask, target_tokens, block_credit, item_credit


def split_record_indices(groups: List[str], valid_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(groups) <= 1 or float(valid_ratio) <= 0.0:
        idx = np.arange(len(groups), dtype=np.int64)
        return idx, idx.copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(valid_ratio), random_state=int(seed))
    dummy = np.zeros(len(groups), dtype=np.float32)
    train_idx, valid_idx = next(splitter.split(dummy, groups=groups))
    return np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64)


def _decoder_input_ids(target_tokens: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros((target_tokens.shape[0], 1), dtype=torch.long, device=target_tokens.device),
            target_tokens[:, :-1],
        ],
        dim=1,
    )


def evaluate_segment_head(
    *,
    tiger: TIGER,
    head: TokenCreditTransportHead,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    head.eval()
    total_rows = 0
    total_blocks = 0
    block_sse = 0.0
    item_sse = 0.0
    item_abs = 0.0
    sign_correct = 0.0
    gap_sum = 0.0

    with torch.no_grad():
        for input_ids, attention_mask, target_tokens, block_credit, item_credit in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_tokens = target_tokens.to(device)
            block_credit = block_credit.to(device)
            item_credit = item_credit.to(device)
            _, hidden = tiger.backbone.decode_with_hidden(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=_decoder_input_ids(target_tokens),
            )
            pred = head(hidden.detach(), target_tokens)
            item_pred = torch.sum(pred, dim=1)

            total_rows += int(target_tokens.shape[0])
            total_blocks += int(np.prod(list(block_credit.shape)))
            block_sse += float(torch.sum((pred - block_credit) ** 2).item())
            item_sse += float(torch.sum((item_pred - item_credit) ** 2).item())
            item_abs += float(torch.sum(torch.abs(item_pred - item_credit)).item())
            sign_correct += float(torch.sum((torch.sign(pred) == torch.sign(block_credit)).float()).item())
            gap_sum += float(torch.sum(torch.abs(item_pred - torch.sum(pred, dim=1))).item())

    denom_rows = max(total_rows, 1)
    denom_blocks = max(total_blocks, 1)
    return {
        "block_mse": float(block_sse / float(denom_blocks)),
        "item_mse": float(item_sse / float(denom_rows)),
        "item_mae": float(item_abs / float(denom_rows)),
        "block_sign_acc": float(sign_correct / float(denom_blocks)),
        "conservation_gap": float(gap_sum / float(denom_rows)),
    }


def train_segment_transport_head(
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
    dataset = SegmentTransportDataset(records)
    groups = [str(rec["group"]) for rec in records]
    train_idx, valid_idx = split_record_indices(groups, valid_ratio, seed)

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_segment_transport,
    )
    valid_loader = DataLoader(
        Subset(dataset, valid_idx.tolist()),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_segment_transport,
    )

    head = TokenCreditTransportHead(
        hidden_size=int(tiger.backbone.model.config.d_model),
        vocab_size=int(tiger.backbone.model.config.vocab_size),
        token_dim=int(token_dim),
        mlp_dim=int(mlp_dim),
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(lr))

    best_state = None
    best_metrics: Dict[str, float] = {}
    best_key = float("inf")
    best_epoch = 0
    epochs_since_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        head.train()
        train_losses: List[float] = []
        for input_ids, attention_mask, target_tokens, block_credit, item_credit in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_tokens = target_tokens.to(device)
            block_credit = block_credit.to(device)
            item_credit = item_credit.to(device)
            with torch.no_grad():
                _, hidden = tiger.backbone.decode_with_hidden(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=_decoder_input_ids(target_tokens),
                )
            pred = head(hidden.detach(), target_tokens)
            pred_item_credit = torch.sum(pred, dim=1)
            loss_block = F.smooth_l1_loss(pred, block_credit)
            loss_cons = F.mse_loss(pred_item_credit, item_credit)
            target_sign = (block_credit >= 0.0).float()
            loss_sign = F.binary_cross_entropy_with_logits(pred, target_sign)
            loss = loss_block + float(conservation_scale) * loss_cons + float(sign_scale) * loss_sign
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        valid_metrics = evaluate_segment_head(tiger=tiger, head=head, loader=valid_loader, device=device)
        valid_metrics["epoch"] = float(epoch)
        valid_metrics["train_loss"] = float(np.mean(train_losses)) if train_losses else 0.0
        history.append(dict(valid_metrics))
        key = float(valid_metrics["item_mse"] + 0.5 * valid_metrics["block_mse"])
        if key < best_key:
            best_key = key
            best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
            best_metrics = dict(valid_metrics)
            best_epoch = int(epoch)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if int(patience) > 0 and epochs_since_improve >= int(patience):
            break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
        best_metrics = evaluate_segment_head(tiger=tiger, head=head, loader=valid_loader, device=device)

    return {
        "state_dict": best_state,
        "best_metrics": best_metrics,
        "best_epoch": int(best_epoch) if best_epoch > 0 else int(len(history)),
        "split": {
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
        },
        "history": history,
    }


def load_segment_head(
    *,
    head_path: Path,
    meta_path: Path,
    tiger: TIGER,
    device: torch.device,
) -> TokenCreditTransportHead:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    payload = torch.load(head_path, map_location="cpu")
    head = TokenCreditTransportHead(
        hidden_size=int(meta["hidden_size"]),
        vocab_size=int(meta["vocab_size"]),
        token_dim=int(meta["token_dim"]),
        mlp_dim=int(meta["mlp_dim"]),
    ).to(device)
    state_dict = payload.get("model_state_dict", payload.get("state_dict", payload))
    head.load_state_dict(state_dict)
    head.eval()
    _ = tiger
    return head


def discover_default_phase2_runs(dataset: str, name_contains: str = "") -> List[Path]:
    return discover_run_dirs(dataset=dataset, name_contains=name_contains)
