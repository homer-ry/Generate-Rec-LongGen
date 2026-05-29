from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

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
)
from simulation.hazard_plan import (  # noqa: E402
    FEATURE_NAMES,
    PLANNER_FEATURE_NAMES,
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

LEVEL_NAMES: Sequence[str] = ("slate", "plan", "item")
PLAN_INDEX: Dict[str, int] = {
    "safe_match": 0,
    "recover": 1,
    "explore": 2,
    "balanced": 3,
}
FUTURE_TARGET_NAMES: Sequence[str] = (
    "future_depth_norm",
    "future_continue_h",
    "future_rating_norm",
    "future_neg_rate",
    "future_click_rate",
    "future_interview_norm",
    "future_welfare",
)
DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "continue": 0.30,
    "watch": 1.00,
    "align": 0.30,
    "rating": 0.20,
    "negative": 0.80,
}
DEFAULT_WELFARE_WEIGHTS: Dict[str, float] = {
    "future_depth_norm": 0.25,
    "future_continue_h": 0.25,
    "future_rating_norm": 0.20,
    "future_neg_rate": -0.20,
    "future_click_rate": 0.10,
    "future_interview_norm": 0.20,
}
PHASE1_INPUT_DIM = len(PLANNER_FEATURE_NAMES) + len(FEATURE_NAMES) + len(PLAN_INDEX) + len(LEVEL_NAMES)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_weight_overrides(raw: str, defaults: Dict[str, float]) -> Dict[str, float]:
    weights = dict(defaults)
    text = str(raw or "").strip()
    if not text:
        return weights
    for piece in text.split(","):
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        key = key.strip()
        if key not in weights:
            continue
        try:
            weights[key] = float(value.strip())
        except Exception:
            continue
    return weights


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float32))) if values else 0.0


def _safe_plan_onehot(plan_name: str) -> np.ndarray:
    vec = np.zeros(len(PLAN_INDEX), dtype=np.float32)
    idx = PLAN_INDEX.get(str(plan_name), -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


def _level_onehot(level_name: str) -> np.ndarray:
    vec = np.zeros(len(LEVEL_NAMES), dtype=np.float32)
    try:
        vec[list(LEVEL_NAMES).index(str(level_name))] = 1.0
    except ValueError:
        pass
    return vec


def compose_phase1_feature(
    planner_vec: np.ndarray,
    item_vec: np.ndarray,
    plan_name: str,
    level_name: str,
) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(planner_vec, dtype=np.float32).reshape(-1),
            np.asarray(item_vec, dtype=np.float32).reshape(-1),
            _safe_plan_onehot(plan_name),
            _level_onehot(level_name),
        ],
        axis=0,
    ).astype(np.float32)


def _normalized_future_targets(
    page_records: Sequence[Dict[str, Any]],
    *,
    index: int,
    future_horizon: int,
    max_pages: int,
    interview_norm: float,
    welfare_weights: Dict[str, float],
) -> Dict[str, float]:
    horizon = max(int(future_horizon), 1)
    lo = int(index)
    hi = min(len(page_records), lo + horizon)
    future = list(page_records[lo:hi])
    remaining_pages = max(len(page_records) - lo, 0)
    future_depth_norm = float(remaining_pages / max(float(max_pages), 1.0))
    future_continue_h = float(min(max(remaining_pages - 1, 0), horizon) / float(horizon))
    future_rating_norm = _safe_mean([float(row["rating_norm"]) for row in future])
    future_neg_rate = _safe_mean([float(row["negative_flag"]) for row in future])
    future_click_rate = _safe_mean([float(row["watch_flag"]) for row in future])

    targets = {
        "future_depth_norm": float(np.clip(future_depth_norm, 0.0, 1.0)),
        "future_continue_h": float(np.clip(future_continue_h, 0.0, 1.0)),
        "future_rating_norm": float(np.clip(future_rating_norm, 0.0, 1.0)),
        "future_neg_rate": float(np.clip(future_neg_rate, 0.0, 1.0)),
        "future_click_rate": float(np.clip(future_click_rate, 0.0, 1.0)),
        "future_interview_norm": float(np.clip(interview_norm, 0.0, 1.0)),
    }
    welfare = 0.0
    for key, value in targets.items():
        welfare += float(welfare_weights.get(key, 0.0)) * float(value)
    targets["future_welfare"] = float(welfare)
    return targets


def build_phase1_artifact(
    *,
    dataset: str,
    run_dirs: Sequence[Path],
    future_horizon: int,
    credit_gamma: float,
    credit_reward_weights: Dict[str, float],
    welfare_weights: Dict[str, float],
) -> Dict[str, Any]:
    persona_df, user_statistic, movie_detail = load_tables(dataset)
    item_catalog = build_item_catalog(movie_detail)

    zeros_item = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    slate_features: List[np.ndarray] = []
    plan_features: List[np.ndarray] = []
    item_features: List[np.ndarray] = []
    slate_targets: List[np.ndarray] = []
    item_targets: List[np.ndarray] = []
    plan_targets: List[np.ndarray] = []
    slate_value_targets: List[float] = []
    item_value_targets: List[float] = []
    plan_value_targets: List[float] = []
    slate_groups: List[str] = []
    item_groups: List[str] = []
    plan_groups: List[str] = []
    slate_page_index: List[int] = []
    item_page_index: List[int] = []
    plan_page_index: List[int] = []
    plan_name_list: List[str] = []
    item_name_list: List[str] = []
    slate_record_ids: List[str] = []
    item_record_ids: List[str] = []
    plan_record_ids: List[str] = []
    stats = {
        "runs_used": [str(p) for p in run_dirs],
        "users_total": 0,
        "pages_total": 0,
        "slate_records": 0,
        "item_records": 0,
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
            max_pages = max(max_pages_cfg, max(page_keys))
            user_profile = build_user_profile(persona_df.iloc[uid], user_statistic.loc[uid], uid)
            state = initial_rollout_state()
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
                rec_ids = [iid for iid in to_int_list(info.get("recommended_id")) if iid in item_catalog]
                align_ids = [iid for iid in to_int_list(info.get("align_id")) if iid in item_catalog]
                watch_ids = [iid for iid in to_int_list(info.get("watch_id")) if iid in item_catalog]
                rating_values = [parse_float(v, 0.0) for v in info.get("rating", [])]
                rating_values = [v for v in rating_values if v > 0]
                avg_rating = float(np.mean(rating_values)) if rating_values else 0.0
                align_count = len(align_ids)
                watch_count = len(watch_ids)
                negative_increment = heuristic_negative_increment(watch_count, avg_rating)

                state_summary = summarize_rollout_state(state, page_index=int(page_no), max_pages=max_pages)
                planner_row = build_planner_feature_row(user_profile, state_summary)
                planner_vec = planner_row_to_vector(planner_row)
                plan_name = choose_plan(user_profile, state_summary, override="auto")
                selected_item = int(rec_ids[0]) if rec_ids else -1
                selected_info = item_catalog.get(selected_item, {})
                selected_row = (
                    build_feature_row(user_profile, state_summary, selected_info, plan_name)
                    if selected_item >= 0
                    else None
                )
                selected_vec = row_to_vector(selected_row) if selected_row is not None else zeros_item.copy()
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
                        "planner_vec": planner_vec.astype(np.float32),
                        "plan_name": str(plan_name),
                        "selected_item_vec": selected_vec.astype(np.float32),
                        "selected_item_id": int(selected_item),
                        "watch_flag": float(watch_flag),
                        "rating_norm": float(rating_norm),
                        "negative_flag": float(negative_flag),
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

            terminal_reward = 0.20 * float(interview_norm)
            discounted_returns = compute_discounted_returns(
                immediate_rewards,
                gamma=float(credit_gamma),
                terminal_reward=float(terminal_reward),
            )

            group_name = f"{run_dir.name}:{uid}"
            for idx, page_record in enumerate(page_records):
                targets = _normalized_future_targets(
                    page_records,
                    index=idx,
                    future_horizon=future_horizon,
                    max_pages=max_pages,
                    interview_norm=interview_norm,
                    welfare_weights=welfare_weights,
                )
                target_vec = np.asarray([float(targets[name]) for name in FUTURE_TARGET_NAMES], dtype=np.float32)
                value_target = float(discounted_returns[idx] + targets["future_welfare"])
                page_no = int(page_record["page_no"])
                plan_name = str(page_record["plan_name"])
                planner_vec = np.asarray(page_record["planner_vec"], dtype=np.float32)
                item_vec = np.asarray(page_record["selected_item_vec"], dtype=np.float32)
                record_id = f"{group_name}:{page_no}"

                slate_features.append(compose_phase1_feature(planner_vec, zeros_item, plan_name, "slate"))
                plan_features.append(compose_phase1_feature(planner_vec, zeros_item, plan_name, "plan"))
                item_features.append(compose_phase1_feature(planner_vec, item_vec, plan_name, "item"))

                slate_targets.append(target_vec)
                plan_targets.append(target_vec)
                item_targets.append(target_vec)
                slate_value_targets.append(value_target)
                plan_value_targets.append(value_target)
                item_value_targets.append(value_target)
                slate_groups.append(group_name)
                plan_groups.append(group_name)
                item_groups.append(group_name)
                slate_page_index.append(page_no)
                plan_page_index.append(page_no)
                item_page_index.append(page_no)
                plan_name_list.append(plan_name)
                item_name_list.append(plan_name)
                slate_record_ids.append(record_id)
                plan_record_ids.append(record_id)
                item_record_ids.append(record_id)

    stats["slate_records"] = int(len(slate_features))
    stats["item_records"] = int(len(item_features))
    artifact = {
        "meta": {
            "dataset": str(dataset),
            "future_horizon": int(future_horizon),
            "credit_gamma": float(credit_gamma),
            "credit_reward_weights": dict(credit_reward_weights),
            "welfare_weights": dict(welfare_weights),
            "input_dim": int(PHASE1_INPUT_DIM),
            "future_target_names": list(FUTURE_TARGET_NAMES),
            "level_names": list(LEVEL_NAMES),
            "plan_names": list(PLAN_INDEX.keys()),
        },
        "stats": stats,
        "slate": {
            "features": np.asarray(slate_features, dtype=np.float32),
            "future_targets": np.asarray(slate_targets, dtype=np.float32),
            "value_targets": np.asarray(slate_value_targets, dtype=np.float32),
            "groups": list(slate_groups),
            "page_index": np.asarray(slate_page_index, dtype=np.int64),
            "plan_name": list(plan_name_list),
            "record_id": list(slate_record_ids),
        },
        "plan": {
            "features": np.asarray(plan_features, dtype=np.float32),
            "future_targets": np.asarray(plan_targets, dtype=np.float32),
            "value_targets": np.asarray(plan_value_targets, dtype=np.float32),
            "groups": list(plan_groups),
            "page_index": np.asarray(plan_page_index, dtype=np.int64),
            "plan_name": list(plan_name_list),
            "record_id": list(plan_record_ids),
        },
        "item": {
            "features": np.asarray(item_features, dtype=np.float32),
            "future_targets": np.asarray(item_targets, dtype=np.float32),
            "value_targets": np.asarray(item_value_targets, dtype=np.float32),
            "groups": list(item_groups),
            "page_index": np.asarray(item_page_index, dtype=np.int64),
            "plan_name": list(item_name_list),
            "record_id": list(item_record_ids),
        },
    }
    return artifact


def save_phase1_artifact(artifact: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, path)


def load_phase1_artifact(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")


class ArrayDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        idx = int(index)
        return self.features[idx], self.targets[idx]


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FutureWelfareModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        dropout: float = 0.1,
        out_dim: int = len(FUTURE_TARGET_NAMES),
    ):
        super().__init__()
        layers: List[nn.Module] = []
        cur_dim = int(input_dim)
        for _ in range(max(int(depth), 1)):
            layers.append(MLPBlock(cur_dim, int(hidden_dim), float(dropout)))
            cur_dim = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(cur_dim, int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        return torch.sigmoid(self.head(hidden))


class ScalarCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        cur_dim = int(input_dim)
        for _ in range(max(int(depth), 1)):
            layers.append(MLPBlock(cur_dim, int(hidden_dim), float(dropout)))
            cur_dim = int(hidden_dim)
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(cur_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        return self.head(hidden).squeeze(-1)


def describe_phase1_artifact(artifact: Dict[str, Any]) -> str:
    lines = [
        json.dumps(artifact.get("meta", {}), ensure_ascii=False),
        json.dumps(artifact.get("stats", {}), ensure_ascii=False),
    ]
    for level in ("slate", "plan", "item"):
        entry = artifact.get(level, {})
        features = np.asarray(entry.get("features", np.zeros((0, PHASE1_INPUT_DIM), dtype=np.float32)))
        lines.append(f"{level}: n={features.shape[0]} dim={features.shape[1] if features.ndim == 2 else 0}")
    return "\n".join(lines)


def discover_default_runs(dataset: str, name_contains: str = "") -> List[Path]:
    return discover_run_dirs(dataset=dataset, name_contains=name_contains)
