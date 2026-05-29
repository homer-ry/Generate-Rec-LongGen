from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.hazard_plan import (
    FEATURE_NAMES,
    PLAN_OPTIONS,
    PLANNER_FEATURE_NAMES,
    build_feature_row,
    build_planner_feature_row,
    build_item_catalog,
    build_user_profile,
    choose_plan,
    compute_hindsight_plan_scores_v3,
    heuristic_negative_increment,
    infer_hindsight_plan_label,
    infer_hindsight_plan_label_v2,
    infer_hindsight_plan_label_v3,
    initial_rollout_state,
    mean_or_zero,
    parse_float,
    planner_row_to_vector,
    row_to_vector,
    summarize_rollout_state,
    to_int_list,
    update_rollout_state,
)


def parse_metrics_txt(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not path.exists():
        return out
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^([^:]+)\s*:\s*(.*)$", raw_line.strip())
        if not m:
            continue
        out[m.group(1).strip()] = m.group(2).strip()
    return out


def discover_run_dirs(dataset: str, name_contains: str = "") -> List[Path]:
    base = REPO_ROOT / "storage" / dataset
    out: List[Path] = []
    if not base.exists():
        return out
    needle = name_contains.strip().lower()
    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            if not (run_dir / "behavior").exists():
                continue
            if needle and needle not in run_dir.name.lower():
                continue
            out.append(run_dir)
    return out


def load_tables(dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = REPO_ROOT / "datasets" / dataset / "simulation"
    persona_df = pd.read_csv(base / "all_personas_like_modify.csv")
    user_statistic = pd.read_csv(base / "user_statistic.csv", index_col=0)
    try:
        user_statistic.index = user_statistic.index.astype(int)
    except Exception:
        pass
    movie_detail = pd.read_csv(base / "movie_detail.csv")
    return persona_df, user_statistic, movie_detail


def build_page_outcome(
    *,
    user_profile: Dict[str, Any],
    item_catalog: Dict[int, Dict[str, Any]],
    info: Dict[str, Any],
    page_no: int,
) -> Dict[str, float]:
    align_ids = to_int_list(info.get("align_id"))
    watch_ids = to_int_list(info.get("watch_id"))
    rec_ids = [iid for iid in to_int_list(info.get("recommended_id")) if iid in item_catalog]
    ratings = [parse_float(v, 0.0) for v in info.get("rating", [])]
    ratings = [r for r in ratings if r > 0]
    avg_rating = mean_or_zero(ratings)
    align_count = len(align_ids)
    watch_count = len(watch_ids)

    item_id = next(iter(watch_ids or align_ids or rec_ids), -1)
    item_info = item_catalog.get(int(item_id), {})
    item_tags = set(item_info.get("tags", set()))
    preferred_types = set(user_profile.get("preferred_types", []))
    focus_match = 1.0 if user_profile.get("focus", "") in item_tags else 0.0
    pref_match_any = 1.0 if preferred_types & item_tags else 0.0
    novelty_score = max(pref_match_any - focus_match, 0.0)
    rating_norm = max(min(avg_rating / 5.0, 1.0), 0.0) if watch_count > 0 else 0.0
    success_score = min(1.0, 0.55 * float(watch_count > 0) + 0.30 * rating_norm + 0.15 * float(align_count > 0))

    return {
        "page_no": float(page_no),
        "align_count": float(align_count),
        "watch_count": float(watch_count),
        "avg_rating": float(avg_rating),
        "focus_match": float(focus_match),
        "pref_match_any": float(pref_match_any),
        "novelty_score": float(novelty_score),
        "item_quality": float(item_info.get("quality", 0.0)),
        "success_score": float(success_score),
        "focus_score": float(success_score * (0.75 * focus_match + 0.25 * pref_match_any)),
        "quality_score": float(success_score * float(item_info.get("quality", 0.0))),
    }


def softmax_local(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.asarray([], dtype=np.float32)
    temp = max(float(temperature), 1e-6)
    arr = arr / temp
    arr = arr - float(np.max(arr))
    exp_arr = np.exp(arr)
    denom = float(np.sum(exp_arr))
    if denom <= 0:
        return np.full(arr.shape, 1.0 / float(arr.size), dtype=np.float32)
    return (exp_arr / denom).astype(np.float32)


def compute_future_plan_occupancy(
    score_targets: Sequence[np.ndarray],
    *,
    start_idx: int,
    horizon: int,
    discount: float = 0.85,
    temperature: float = 0.75,
) -> np.ndarray:
    target = np.zeros(len(PLAN_OPTIONS), dtype=np.float32)
    norm = 0.0
    max_horizon = max(int(horizon), 1)
    for offset in range(max_horizon):
        idx = int(start_idx) + int(offset)
        if idx >= len(score_targets):
            break
        probs = softmax_local(score_targets[idx], temperature=temperature)
        weight = float(discount) ** float(offset)
        target += float(weight) * probs
        norm += float(weight)
    if norm > 0:
        target = target / float(norm)
    return target.astype(np.float32)


def parse_interview_rating(interview_obj: Any) -> float:
    if isinstance(interview_obj, dict):
        raw_entries = interview_obj.get("interview", [])
    else:
        raw_entries = interview_obj
    entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
    for entry in entries:
        m = re.search(r"[-+]?\d+", str(entry))
        if m:
            try:
                rating = float(int(m.group(0)))
                return max(min(rating, 10.0), 1.0)
            except Exception:
                continue
    return 0.0


def build_credit_reward(
    *,
    watch_count: int,
    align_count: int,
    avg_rating: float,
    negative_increment: int,
    continued: bool,
    reward_weights: Dict[str, float],
) -> float:
    rating_norm = max(min(float(avg_rating) / 5.0, 1.0), 0.0) if watch_count > 0 else 0.0
    reward = 0.0
    reward += float(reward_weights.get("continue", 0.0)) * float(1 if continued else 0)
    reward += float(reward_weights.get("watch", 0.0)) * float(1 if watch_count > 0 else 0)
    reward += float(reward_weights.get("align", 0.0)) * float(1 if align_count > 0 else 0)
    reward += float(reward_weights.get("rating", 0.0)) * float(rating_norm)
    reward -= float(reward_weights.get("negative", 0.0)) * float(max(int(negative_increment), 0))
    return float(reward)


def compute_discounted_returns(
    immediate_rewards: Sequence[float],
    *,
    gamma: float,
    terminal_reward: float,
) -> np.ndarray:
    returns = np.zeros(len(immediate_rewards), dtype=np.float32)
    running = float(terminal_reward)
    for idx in range(len(immediate_rewards) - 1, -1, -1):
        running = float(immediate_rewards[idx]) + float(gamma) * running
        returns[idx] = float(running)
    return returns


def extract_examples(
    *,
    dataset: str,
    run_dirs: Sequence[Path],
    only_items_per_page: int,
    planner_label_mode: str,
    planner_future_horizon: int,
    credit_gamma: float,
    credit_reward_weights: Dict[str, float],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[str, Any],
]:
    persona_df, user_statistic, movie_detail = load_tables(dataset)
    item_catalog = build_item_catalog(movie_detail)

    rows: List[np.ndarray] = []
    exit_labels: List[int] = []
    enjoy_labels: List[int] = []
    item_groups: List[str] = []
    planner_rows: List[np.ndarray] = []
    planner_labels: List[str] = []
    planner_score_targets: List[np.ndarray] = []
    planner_dwell_targets: List[np.ndarray] = []
    planner_groups: List[str] = []
    stats = {
        "runs_used": [],
        "pages_total": 0,
        "pages_used": 0,
        "users_total": 0,
        "exit_positives": 0,
        "enjoy_positives": 0,
        "planner_label_hist": {},
        "planner_score_mean": {plan: 0.0 for plan in PLAN_OPTIONS},
        "planner_dwell_mean": {plan: 0.0 for plan in PLAN_OPTIONS},
        "credit_return_mean": 0.0,
        "credit_reward_mean": 0.0,
        "credit_terminal_mean": 0.0,
    }
    credit_state_rows: List[np.ndarray] = []
    credit_return_targets: List[float] = []
    credit_groups: List[str] = []

    for run_dir in run_dirs:
        behavior_dir = run_dir / "behavior"
        interview_dir = run_dir / "interview"
        if not behavior_dir.exists():
            continue
        metrics = parse_metrics_txt(run_dir / "metrics.txt")
        max_pages_cfg = int(parse_float(metrics.get("Maximum exit page"), 0.0))
        stats["runs_used"].append(str(run_dir))

        for pkl_path in sorted(behavior_dir.glob("*.pkl")):
            uid = int(pkl_path.stem)
            if uid >= len(persona_df):
                continue
            if uid not in user_statistic.index:
                continue
            with pkl_path.open("rb") as f:
                behavior = pickle.load(f)
            if not isinstance(behavior, dict):
                continue
            page_keys = sorted([k for k in behavior.keys() if isinstance(k, int)])
            if not page_keys:
                continue

            user_profile = build_user_profile(persona_df.iloc[uid], user_statistic.loc[uid], uid)
            state = initial_rollout_state()
            stats["users_total"] += 1
            max_pages = max(max_pages_cfg, max(page_keys))
            page_outcomes: List[Dict[str, float]] = []
            interview_rating = 0.0
            interview_path = interview_dir / f"{uid}.pkl"
            if interview_path.exists():
                try:
                    with interview_path.open("rb") as f:
                        interview_rating = parse_interview_rating(pickle.load(f))
                except Exception:
                    interview_rating = 0.0
            user_planner_rows: List[np.ndarray] = []
            user_planner_labels: List[str] = []
            user_planner_scores: List[np.ndarray] = []
            user_planner_groups: List[str] = []
            user_item_records: List[Dict[str, Any]] = []
            for page_no in page_keys:
                info = behavior.get(page_no, {})
                page_outcomes.append(
                    build_page_outcome(
                        user_profile=user_profile,
                        item_catalog=item_catalog,
                        info=info,
                        page_no=int(page_no),
                    )
                )

            for page_idx, page_no in enumerate(page_keys):
                info = behavior.get(page_no, {})
                stats["pages_total"] += 1
                candidate_ids = [iid for iid in to_int_list(info.get("recommended_id")) if iid in item_catalog]
                state_summary = summarize_rollout_state(state, page_index=int(page_no), max_pages=max_pages)
                plan = choose_plan(user_profile, state_summary, override="auto")
                planner_row = build_planner_feature_row(user_profile, state_summary)
                future_slice = page_outcomes[page_idx : min(page_idx + max(int(planner_future_horizon), 1), len(page_outcomes))]
                planner_mode = str(planner_label_mode).lower()
                if planner_mode in {"trajectory_v3", "hindsight_v3", "v3"}:
                    hindsight_label = infer_hindsight_plan_label_v3(
                        user_profile=user_profile,
                        state=state_summary,
                        current_outcome=page_outcomes[page_idx],
                        future_outcomes=future_slice,
                    )
                elif planner_mode in {"trajectory_v2", "hindsight_v2", "v2"}:
                    hindsight_label = infer_hindsight_plan_label_v2(
                        user_profile=user_profile,
                        state=state_summary,
                        current_outcome=page_outcomes[page_idx],
                        future_outcomes=future_slice,
                    )
                else:
                    hindsight_label = infer_hindsight_plan_label(
                        user_profile=user_profile,
                        state=state_summary,
                        current_outcome=page_outcomes[page_idx],
                        future_outcomes=future_slice,
                    )
                score_target = compute_hindsight_plan_scores_v3(
                    user_profile=user_profile,
                    state=state_summary,
                    current_outcome=page_outcomes[page_idx],
                    future_outcomes=future_slice,
                )
                score_target_vec = np.asarray(
                    [float(score_target.get(plan_name, 0.0)) for plan_name in PLAN_OPTIONS],
                    dtype=np.float32,
                )
                user_planner_rows.append(planner_row_to_vector(planner_row))
                user_planner_labels.append(hindsight_label)
                user_planner_scores.append(score_target_vec)
                user_planner_groups.append(f"{run_dir.name}:{uid}")
                stats["planner_label_hist"][hindsight_label] = int(stats["planner_label_hist"].get(hindsight_label, 0)) + 1
                for plan_name in PLAN_OPTIONS:
                    stats["planner_score_mean"][plan_name] = float(
                        stats["planner_score_mean"].get(plan_name, 0.0) + float(score_target.get(plan_name, 0.0))
                    )
                if int(only_items_per_page) <= 0 or len(candidate_ids) == int(only_items_per_page):
                    if candidate_ids:
                        item_info = item_catalog[candidate_ids[0]]
                        feature_row = build_feature_row(user_profile, state_summary, item_info, plan)
                        exit_label = 1 if int(page_no) == int(page_keys[-1]) else 0
                        align_count = len(to_int_list(info.get("align_id")))
                        watch_count = len(to_int_list(info.get("watch_id")))
                        ratings = [parse_float(v, 0.0) for v in info.get("rating", [])]
                        ratings = [r for r in ratings if r > 0]
                        avg_rating = mean_or_zero(ratings)
                        enjoy_label = 1 if (watch_count > 0 and avg_rating >= 4.0) else 0
                        negative_increment = heuristic_negative_increment(watch_count, avg_rating)
                        immediate_reward = build_credit_reward(
                            watch_count=watch_count,
                            align_count=align_count,
                            avg_rating=avg_rating,
                            negative_increment=negative_increment,
                            continued=bool(int(page_no) < int(page_keys[-1])),
                            reward_weights=credit_reward_weights,
                        )
                        user_item_records.append(
                            {
                                "row_vec": row_to_vector(feature_row),
                                "state_vec": planner_row_to_vector(planner_row),
                                "group": f"{run_dir.name}:{uid}",
                                "exit_label": int(exit_label),
                                "enjoy_label": int(enjoy_label),
                                "reward": float(immediate_reward),
                            }
                        )
                        stats["credit_reward_mean"] = float(stats["credit_reward_mean"] + float(immediate_reward))
                align_count = len(to_int_list(info.get("align_id")))
                watch_count = len(to_int_list(info.get("watch_id")))
                ratings = [parse_float(v, 0.0) for v in info.get("rating", [])]
                ratings = [r for r in ratings if r > 0]
                avg_rating = mean_or_zero(ratings)
                negative_increment = heuristic_negative_increment(watch_count, avg_rating)
                update_rollout_state(
                    state,
                    align_count=align_count,
                    watch_count=watch_count,
                    avg_rating=avg_rating,
                    negative_increment=negative_increment,
                )

            for planner_idx, planner_vec in enumerate(user_planner_rows):
                dwell_target = compute_future_plan_occupancy(
                    user_planner_scores,
                    start_idx=planner_idx,
                    horizon=max(int(planner_future_horizon), 1),
                )
                planner_rows.append(planner_vec)
                planner_labels.append(user_planner_labels[planner_idx])
                planner_score_targets.append(user_planner_scores[planner_idx])
                planner_dwell_targets.append(dwell_target)
                planner_groups.append(user_planner_groups[planner_idx])
                for plan_pos, plan_name in enumerate(PLAN_OPTIONS):
                    stats["planner_dwell_mean"][plan_name] = float(
                    stats["planner_dwell_mean"].get(plan_name, 0.0) + float(dwell_target[plan_pos])
                    )

            terminal_reward = float(credit_reward_weights.get("terminal", 0.0)) * float(
                max(min(interview_rating / 10.0, 1.0), 0.0)
            )
            if user_item_records:
                immediate_rewards = [float(rec["reward"]) for rec in user_item_records]
                discounted_returns = compute_discounted_returns(
                    immediate_rewards,
                    gamma=float(credit_gamma),
                    terminal_reward=float(terminal_reward),
                )
                stats["credit_terminal_mean"] = float(stats["credit_terminal_mean"] + float(terminal_reward))
                for rec_idx, rec in enumerate(user_item_records):
                    rows.append(rec["row_vec"])
                    exit_labels.append(int(rec["exit_label"]))
                    enjoy_labels.append(int(rec["enjoy_label"]))
                    item_groups.append(str(rec["group"]))
                    credit_state_rows.append(rec["state_vec"])
                    credit_return_targets.append(float(discounted_returns[rec_idx]))
                    credit_groups.append(str(rec["group"]))
                    stats["pages_used"] += 1
                    stats["exit_positives"] += int(rec["exit_label"])
                    stats["enjoy_positives"] += int(rec["enjoy_label"])
                    stats["credit_return_mean"] = float(
                        stats["credit_return_mean"] + float(discounted_returns[rec_idx])
                    )

    if not rows:
        raise ValueError("No usable training examples were extracted from the provided run directories.")
    X = np.vstack(rows).astype(np.float32)
    y_exit = np.asarray(exit_labels, dtype=np.int64)
    y_enjoy = np.asarray(enjoy_labels, dtype=np.int64)
    X_plan = np.vstack(planner_rows).astype(np.float32)
    y_plan = np.asarray(planner_labels, dtype=object)
    y_plan_scores = np.vstack(planner_score_targets).astype(np.float32)
    y_plan_dwell = np.vstack(planner_dwell_targets).astype(np.float32)
    n_plan_rows = max(int(len(planner_score_targets)), 1)
    stats["planner_score_mean"] = {
        plan_name: float(stats["planner_score_mean"].get(plan_name, 0.0) / n_plan_rows)
        for plan_name in PLAN_OPTIONS
    }
    stats["planner_dwell_mean"] = {
        plan_name: float(stats["planner_dwell_mean"].get(plan_name, 0.0) / n_plan_rows)
        for plan_name in PLAN_OPTIONS
    }
    n_credit_rows = max(int(len(credit_return_targets)), 1)
    stats["credit_return_mean"] = float(stats["credit_return_mean"] / n_credit_rows)
    stats["credit_reward_mean"] = float(stats["credit_reward_mean"] / n_credit_rows)
    stats["credit_terminal_mean"] = float(stats["credit_terminal_mean"] / max(int(stats["users_total"]), 1))
    X_credit_state = np.vstack(credit_state_rows).astype(np.float32)
    y_credit_return = np.asarray(credit_return_targets, dtype=np.float32)
    return (
        X,
        y_exit,
        y_enjoy,
        item_groups,
        X_plan,
        y_plan,
        y_plan_scores,
        y_plan_dwell,
        planner_groups,
        X_credit_state,
        y_credit_return,
        credit_groups,
        stats,
    )


def train_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    seed: int,
    valid_ratio: float,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=max(min(valid_ratio, 0.5), 0.05), random_state=seed)
    train_idx, valid_idx = next(splitter.split(X, y, groups=groups))

    model = RandomForestClassifier(
        n_estimators=240,
        max_depth=8,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X[train_idx], y[train_idx])
    valid_prob = model.predict_proba(X[valid_idx])[:, 1]
    valid_pred = (valid_prob >= 0.5).astype(np.int64)
    metrics = {
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "positive_rate_train": float(np.mean(y[train_idx])),
        "positive_rate_valid": float(np.mean(y[valid_idx])),
        "roc_auc": float(roc_auc_score(y[valid_idx], valid_prob)) if len(np.unique(y[valid_idx])) > 1 else 0.0,
        "avg_precision": float(average_precision_score(y[valid_idx], valid_prob)),
        "accuracy": float(accuracy_score(y[valid_idx], valid_pred)),
        "log_loss": float(log_loss(y[valid_idx], np.clip(valid_prob, 1e-6, 1.0 - 1e-6))),
    }
    return model, metrics


def train_planner_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    seed: int,
    valid_ratio: float,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=max(min(valid_ratio, 0.5), 0.05), random_state=seed)
    train_idx, valid_idx = next(splitter.split(X, y, groups=groups))
    model = RandomForestClassifier(
        n_estimators=220,
        max_depth=9,
        min_samples_leaf=6,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X[train_idx], y[train_idx])
    valid_pred = model.predict(X[valid_idx])
    metrics = {
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "accuracy": float(accuracy_score(y[valid_idx], valid_pred)),
        "label_hist_train": {str(k): int(v) for k, v in pd.Series(y[train_idx]).value_counts().to_dict().items()},
        "label_hist_valid": {str(k): int(v) for k, v in pd.Series(y[valid_idx]).value_counts().to_dict().items()},
    }
    return model, metrics


def train_planner_value_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    seed: int,
    valid_ratio: float,
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=max(min(valid_ratio, 0.5), 0.05), random_state=seed)
    train_idx, valid_idx = next(splitter.split(X, y[:, 0], groups=groups))
    model = RandomForestRegressor(
        n_estimators=260,
        max_depth=10,
        min_samples_leaf=6,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X[train_idx], y[train_idx])
    valid_pred = np.asarray(model.predict(X[valid_idx]), dtype=np.float32)
    if valid_pred.ndim == 1:
        valid_pred = valid_pred.reshape(-1, 1)
    teacher_top1 = np.argmax(y[valid_idx], axis=1)
    pred_top1 = np.argmax(valid_pred, axis=1)
    metrics = {
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "mse": float(mean_squared_error(y[valid_idx], valid_pred)),
        "top1_accuracy": float(accuracy_score(teacher_top1, pred_top1)),
        "target_mean": {
            plan_name: float(np.mean(y[:, idx])) for idx, plan_name in enumerate(PLAN_OPTIONS)
        },
    }
    return model, metrics


def train_planner_dwell_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    seed: int,
    valid_ratio: float,
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=max(min(valid_ratio, 0.5), 0.05), random_state=seed)
    train_idx, valid_idx = next(splitter.split(X, y[:, 0], groups=groups))
    model = RandomForestRegressor(
        n_estimators=220,
        max_depth=10,
        min_samples_leaf=6,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X[train_idx], y[train_idx])
    valid_pred = np.asarray(model.predict(X[valid_idx]), dtype=np.float32)
    if valid_pred.ndim == 1:
        valid_pred = valid_pred.reshape(-1, 1)
    teacher_top1 = np.argmax(y[valid_idx], axis=1)
    pred_top1 = np.argmax(valid_pred, axis=1)
    metrics = {
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "mse": float(mean_squared_error(y[valid_idx], valid_pred)),
        "top1_accuracy": float(accuracy_score(teacher_top1, pred_top1)),
        "target_mean": {
            plan_name: float(np.mean(y[:, idx])) for idx, plan_name in enumerate(PLAN_OPTIONS)
        },
    }
    return model, metrics


def train_credit_value_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    seed: int,
    valid_ratio: float,
) -> Tuple[RandomForestRegressor, Dict[str, Any], np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=max(min(valid_ratio, 0.5), 0.05), random_state=seed)
    train_idx, valid_idx = next(splitter.split(X, y, groups=groups))
    model = RandomForestRegressor(
        n_estimators=260,
        max_depth=10,
        min_samples_leaf=6,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X[train_idx], y[train_idx])
    valid_pred = np.asarray(model.predict(X[valid_idx]), dtype=np.float32).reshape(-1)
    all_pred = np.asarray(model.predict(X), dtype=np.float32).reshape(-1)
    metrics = {
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "mse": float(mean_squared_error(y[valid_idx], valid_pred)),
        "mae": float(np.mean(np.abs(y[valid_idx] - valid_pred))),
        "target_mean": float(np.mean(y)),
        "pred_mean_valid": float(np.mean(valid_pred)) if len(valid_pred) else 0.0,
    }
    return model, metrics, all_pred


def train_credit_model(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    seed: int,
    valid_ratio: float,
) -> Tuple[RandomForestRegressor, Dict[str, Any]]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=max(min(valid_ratio, 0.5), 0.05), random_state=seed)
    train_idx, valid_idx = next(splitter.split(X, y[:, 0], groups=groups))
    model = RandomForestRegressor(
        n_estimators=320,
        max_depth=11,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X[train_idx], y[train_idx])
    valid_pred = np.asarray(model.predict(X[valid_idx]), dtype=np.float32)
    if valid_pred.ndim == 1:
        valid_pred = valid_pred.reshape(-1, 1)
    valid_target = np.asarray(y[valid_idx], dtype=np.float32)
    adv_true = valid_target[:, 0]
    adv_pred = valid_pred[:, 0]
    sign_true = adv_true >= 0.0
    sign_pred = adv_pred >= 0.0
    metrics = {
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "mse": float(mean_squared_error(valid_target, valid_pred)),
        "adv_mse": float(mean_squared_error(adv_true, adv_pred)),
        "sign_accuracy": float(np.mean((sign_true == sign_pred).astype(np.float32))) if len(sign_true) else 0.0,
        "target_mean": {
            "adv": float(np.mean(y[:, 0])),
            "pos": float(np.mean(y[:, 1])) if y.shape[1] > 1 else 0.0,
            "neg": float(np.mean(y[:, 2])) if y.shape[1] > 2 else 0.0,
        },
    }
    return model, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a page-level hazard model for dynamic survival-aware reranking.")
    parser.add_argument("--dataset", type=str, default="all-beauty")
    parser.add_argument("--run_dirs", nargs="*", default=[], help="Optional simulation run directories to use as training data.")
    parser.add_argument("--name_contains", type=str, default="", help="Optional substring filter when auto-discovering runs.")
    parser.add_argument("--only_items_per_page", type=int, default=1, help="Keep only pages with this many shown items.")
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--planner_label_mode",
        type=str,
        default="hindsight_v1",
        help="Pseudo-label construction for planner training: hindsight_v1 or trajectory_v2.",
    )
    parser.add_argument(
        "--planner_future_horizon",
        type=int,
        default=3,
        help="How many pages of future outcomes to use when constructing planner pseudo-labels.",
    )
    parser.add_argument(
        "--planner_target_type",
        type=str,
        default="label",
        help="Planner training target: label or score.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Artifact output directory. Defaults to recommenders/weights/<dataset>/HazardPlan/Saved",
    )
    parser.add_argument(
        "--score_profile",
        type=str,
        default="default",
        help="Optional reranker scoring profile metadata, e.g. default or modeA_tiger.",
    )
    parser.add_argument("--credit_gamma", type=float, default=0.90, help="Discount factor for session-level credit return.")
    parser.add_argument("--credit_continue_weight", type=float, default=0.40)
    parser.add_argument("--credit_watch_weight", type=float, default=0.35)
    parser.add_argument("--credit_align_weight", type=float, default=0.20)
    parser.add_argument("--credit_rating_weight", type=float, default=0.25)
    parser.add_argument("--credit_negative_weight", type=float, default=0.55)
    parser.add_argument("--credit_terminal_weight", type=float, default=0.30)
    args = parser.parse_args()

    if args.run_dirs:
        run_dirs = [(REPO_ROOT / p).resolve() if not Path(p).is_absolute() else Path(p).resolve() for p in args.run_dirs]
    else:
        run_dirs = discover_run_dirs(args.dataset, name_contains=args.name_contains)
    if not run_dirs:
        raise ValueError("No run directories found for hazard-plan training.")

    credit_reward_weights = {
        "continue": float(args.credit_continue_weight),
        "watch": float(args.credit_watch_weight),
        "align": float(args.credit_align_weight),
        "rating": float(args.credit_rating_weight),
        "negative": float(args.credit_negative_weight),
        "terminal": float(args.credit_terminal_weight),
    }

    (
        X,
        y_exit,
        y_enjoy,
        item_groups,
        X_plan,
        y_plan,
        y_plan_scores,
        y_plan_dwell,
        planner_groups,
        X_credit_state,
        y_credit_return,
        credit_groups,
        extract_stats,
    ) = extract_examples(
        dataset=args.dataset,
        run_dirs=run_dirs,
        only_items_per_page=max(int(args.only_items_per_page), 0),
        planner_label_mode=str(args.planner_label_mode),
        planner_future_horizon=max(int(args.planner_future_horizon), 1),
        credit_gamma=float(args.credit_gamma),
        credit_reward_weights=credit_reward_weights,
    )
    hazard_model, hazard_metrics = train_model(
        X=X,
        y=y_exit,
        groups=item_groups,
        seed=int(args.seed),
        valid_ratio=float(args.valid_ratio),
    )
    enjoy_model, enjoy_metrics = train_model(
        X=X,
        y=y_enjoy,
        groups=item_groups,
        seed=int(args.seed),
        valid_ratio=float(args.valid_ratio),
    )
    planner_target_type = str(args.planner_target_type).lower()
    if planner_target_type == "score":
        planner_model, planner_metrics = train_planner_value_model(
            X=X_plan,
            y=y_plan_scores,
            groups=planner_groups,
            seed=int(args.seed),
            valid_ratio=float(args.valid_ratio),
        )
        planner_model_kind = "regressor_scores"
    else:
        planner_model, planner_metrics = train_planner_model(
            X=X_plan,
            y=y_plan,
            groups=planner_groups,
            seed=int(args.seed),
            valid_ratio=float(args.valid_ratio),
        )
        planner_model_kind = "classifier"
    planner_dwell_model, planner_dwell_metrics = train_planner_dwell_model(
        X=X_plan,
        y=y_plan_dwell,
        groups=planner_groups,
        seed=int(args.seed),
        valid_ratio=float(args.valid_ratio),
    )
    credit_value_model, credit_value_metrics, credit_value_pred = train_credit_value_model(
        X=X_credit_state,
        y=y_credit_return,
        groups=credit_groups,
        seed=int(args.seed),
        valid_ratio=float(args.valid_ratio),
    )
    credit_adv = np.asarray(y_credit_return, dtype=np.float32) - np.asarray(credit_value_pred, dtype=np.float32)
    credit_scale = max(float(np.quantile(np.abs(credit_adv), 0.90)), 1e-3)
    credit_adv_norm = np.clip(credit_adv / credit_scale, -1.5, 1.5)
    credit_pos_norm = np.clip(np.maximum(credit_adv, 0.0) / credit_scale, 0.0, 1.5)
    credit_neg_norm = np.clip(np.maximum(-credit_adv, 0.0) / credit_scale, 0.0, 1.5)
    y_credit = np.vstack([credit_adv_norm, credit_pos_norm, credit_neg_norm]).T.astype(np.float32)
    credit_model, credit_metrics = train_credit_model(
        X=X,
        y=y_credit,
        groups=item_groups,
        seed=int(args.seed),
        valid_ratio=float(args.valid_ratio),
    )

    save_dir = Path(args.save_dir) if args.save_dir else REPO_ROOT / "recommenders" / "weights" / args.dataset / "HazardPlan" / "Saved"
    if not save_dir.is_absolute():
        save_dir = (REPO_ROOT / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    with (save_dir / "hazard_model.pkl").open("wb") as f:
        pickle.dump(hazard_model, f)
    with (save_dir / "enjoy_model.pkl").open("wb") as f:
        pickle.dump(enjoy_model, f)
    with (save_dir / "planner_model.pkl").open("wb") as f:
        pickle.dump(planner_model, f)
    with (save_dir / "planner_dwell_model.pkl").open("wb") as f:
        pickle.dump(planner_dwell_model, f)
    with (save_dir / "credit_value_model.pkl").open("wb") as f:
        pickle.dump(credit_value_model, f)
    with (save_dir / "credit_model.pkl").open("wb") as f:
        pickle.dump(credit_model, f)
    metadata = {
        "feature_names": FEATURE_NAMES,
        "planner_feature_names": PLANNER_FEATURE_NAMES,
        "dataset": args.dataset,
        "score_profile": str(args.score_profile),
        "hazard_metrics": hazard_metrics,
        "enjoy_metrics": enjoy_metrics,
        "planner_metrics": planner_metrics,
        "planner_dwell_metrics": planner_dwell_metrics,
        "credit_value_metrics": credit_value_metrics,
        "credit_metrics": credit_metrics,
        "extract_stats": extract_stats,
        "run_dirs": [str(p) for p in run_dirs],
        "only_items_per_page": int(args.only_items_per_page),
        "planner_label_mode": str(args.planner_label_mode),
        "planner_future_horizon": int(args.planner_future_horizon),
        "planner_target_type": planner_target_type,
        "planner_model_kind": planner_model_kind,
        "planner_score_temperature": 0.75,
        "planner_dwell_discount": 0.85,
        "planner_dwell_temperature": 0.75,
        "credit_gamma": float(args.credit_gamma),
        "credit_reward_weights": credit_reward_weights,
        "credit_target_scale": float(credit_scale),
        "credit_output_names": ["adv", "pos", "neg"],
    }
    with (save_dir / "hazard_model_meta.pkl").open("wb") as f:
        pickle.dump(metadata, f)
    (save_dir / "hazard_model_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[saved] {save_dir / 'hazard_model.pkl'}")
    print(f"[saved] {save_dir / 'enjoy_model.pkl'}")
    print(f"[saved] {save_dir / 'planner_model.pkl'}")
    print(f"[saved] {save_dir / 'planner_dwell_model.pkl'}")
    print(f"[saved] {save_dir / 'credit_value_model.pkl'}")
    print(f"[saved] {save_dir / 'credit_model.pkl'}")
    print(f"[saved] {save_dir / 'hazard_model_meta.pkl'}")
    print(
        json.dumps(
            {
                "hazard_metrics": hazard_metrics,
                "enjoy_metrics": enjoy_metrics,
                "planner_metrics": planner_metrics,
                "planner_dwell_metrics": planner_dwell_metrics,
                "credit_value_metrics": credit_value_metrics,
                "credit_metrics": credit_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
