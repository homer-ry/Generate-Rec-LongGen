from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.hierarchical_credit_phase1 import (  # noqa: E402
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_WELFARE_WEIGHTS,
    FutureWelfareModel,
    ScalarCritic,
    build_phase1_artifact,
)
from baseline.train_hazard_plan_reranker import parse_interview_rating  # noqa: E402


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    if len(x) < 3 or len(y) < 3:
        return {"corr": 0.0, "pvalue": 1.0}
    corr, pvalue = spearmanr(x, y)
    return {"corr": float(corr), "pvalue": float(pvalue)}


def _load_future_model(path: Path) -> tuple[FutureWelfareModel, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    meta = dict(payload.get("meta", {}))
    target_names = list(meta.get("target_names", []))
    model = FutureWelfareModel(
        input_dim=int(meta.get("feature_dim", 0)),
        out_dim=len(target_names),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, meta


def _load_scalar_critic(path: Path) -> tuple[ScalarCritic, Dict[str, Any]]:
    payload = torch.load(path, map_location="cpu")
    meta = dict(payload.get("meta", {}))
    model = ScalarCritic(input_dim=int(meta.get("feature_dim", 0)))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, meta


def _collect_user_truth(run_dir: Path) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    behavior_dir = run_dir / "behavior"
    interview_dir = run_dir / "interview"
    for pkl_path in sorted(behavior_dir.glob("*.pkl")):
        uid = int(pkl_path.stem)
        with pkl_path.open("rb") as f:
            behavior = pickle.load(f)
        page_keys = sorted([k for k in behavior.keys() if isinstance(k, int)])
        exit_page = float(len(page_keys))
        interview_rating = 0.0
        iv_path = interview_dir / f"{uid}.pkl"
        if iv_path.exists():
            try:
                with iv_path.open("rb") as f:
                    interview_rating = float(parse_interview_rating(pickle.load(f)))
            except Exception:
                interview_rating = 0.0
        out[int(uid)] = {
            "exit_page": float(exit_page),
            "interview_rating": float(interview_rating),
        }
    return out


def score_single_run(
    *,
    dataset: str,
    run_dir: Path,
    future_model: FutureWelfareModel,
    future_meta: Dict[str, Any],
    slate_critic: ScalarCritic,
    credit_gamma: float,
    future_horizon: int,
    reward_weights: Dict[str, float],
    welfare_weights: Dict[str, float],
) -> Dict[str, Any]:
    artifact = build_phase1_artifact(
        dataset=dataset,
        run_dirs=[run_dir],
        future_horizon=int(future_horizon),
        credit_gamma=float(credit_gamma),
        credit_reward_weights=reward_weights,
        welfare_weights=welfare_weights,
    )
    entry = artifact["slate"]
    features = torch.as_tensor(np.asarray(entry["features"], dtype=np.float32))
    groups = list(entry["groups"])
    page_index = np.asarray(entry["page_index"], dtype=np.int64)

    with torch.no_grad():
        future_pred = future_model(features).cpu().numpy()
        critic_pred = slate_critic(features).cpu().numpy()

    user_truth = _collect_user_truth(run_dir)
    user_rows: Dict[int, Dict[str, List[float]]] = {}
    future_target_names = list(future_meta.get("target_names", []))
    welfare_idx = future_target_names.index("future_welfare")
    continue_idx = future_target_names.index("future_continue_h")
    depth_idx = future_target_names.index("future_depth_norm")

    for idx, group_name in enumerate(groups):
        uid = int(str(group_name).split(":")[-1])
        row = user_rows.setdefault(
            uid,
            {
                "page_index": [],
                "welfare": [],
                "critic": [],
                "future_continue": [],
                "future_depth": [],
            },
        )
        row["page_index"].append(int(page_index[idx]))
        row["welfare"].append(float(future_pred[idx, welfare_idx]))
        row["critic"].append(float(critic_pred[idx]))
        row["future_continue"].append(float(future_pred[idx, continue_idx]))
        row["future_depth"].append(float(future_pred[idx, depth_idx]))

    user_summary: List[Dict[str, float]] = []
    for uid, row in sorted(user_rows.items()):
        truth = user_truth.get(uid, {"exit_page": 0.0, "interview_rating": 0.0})
        user_summary.append(
            {
                "uid": float(uid),
                "exit_page": float(truth["exit_page"]),
                "interview_rating": float(truth["interview_rating"]),
                "welfare_mean": float(np.mean(row["welfare"])),
                "welfare_first": float(row["welfare"][0]),
                "critic_mean": float(np.mean(row["critic"])),
                "critic_first": float(row["critic"][0]),
                "future_continue_mean": float(np.mean(row["future_continue"])),
                "future_depth_mean": float(np.mean(row["future_depth"])),
            }
        )

    exit_vals = [row["exit_page"] for row in user_summary]
    interview_vals = [row["interview_rating"] for row in user_summary]
    out = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "n_users": int(len(user_summary)),
        "avg_exit_page": float(np.mean(exit_vals)) if exit_vals else 0.0,
        "avg_interview": float(np.mean(interview_vals)) if interview_vals else 0.0,
        "pred_welfare_mean": float(np.mean([row["welfare_mean"] for row in user_summary])) if user_summary else 0.0,
        "pred_critic_mean": float(np.mean([row["critic_mean"] for row in user_summary])) if user_summary else 0.0,
        "spearman": {
            "welfare_mean_vs_exit": _safe_spearman([row["welfare_mean"] for row in user_summary], exit_vals),
            "welfare_first_vs_exit": _safe_spearman([row["welfare_first"] for row in user_summary], exit_vals),
            "critic_mean_vs_exit": _safe_spearman([row["critic_mean"] for row in user_summary], exit_vals),
            "critic_first_vs_exit": _safe_spearman([row["critic_first"] for row in user_summary], exit_vals),
            "welfare_mean_vs_interview": _safe_spearman([row["welfare_mean"] for row in user_summary], interview_vals),
            "welfare_first_vs_interview": _safe_spearman([row["welfare_first"] for row in user_summary], interview_vals),
            "critic_mean_vs_interview": _safe_spearman([row["critic_mean"] for row in user_summary], interview_vals),
            "critic_first_vs_interview": _safe_spearman([row["critic_first"] for row in user_summary], interview_vals),
        },
        "user_summary": user_summary,
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Score Phase 1 prefix value signals on one or more runs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run_dirs", nargs="+", required=True)
    parser.add_argument("--future_model", required=True)
    parser.add_argument("--slate_critic", required=True)
    parser.add_argument("--future_horizon", type=int, default=5)
    parser.add_argument("--credit_gamma", type=float, default=0.88)
    parser.add_argument("--reward_weights", default="")
    parser.add_argument("--welfare_weights", default="")
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    reward_weights = dict(DEFAULT_REWARD_WEIGHTS)
    welfare_weights = dict(DEFAULT_WELFARE_WEIGHTS)
    if args.reward_weights:
        for piece in str(args.reward_weights).split(","):
            if "=" in piece:
                key, value = piece.split("=", 1)
                if key.strip() in reward_weights:
                    reward_weights[key.strip()] = float(value.strip())
    if args.welfare_weights:
        for piece in str(args.welfare_weights).split(","):
            if "=" in piece:
                key, value = piece.split("=", 1)
                if key.strip() in welfare_weights:
                    welfare_weights[key.strip()] = float(value.strip())

    future_model, future_meta = _load_future_model(Path(args.future_model))
    slate_critic, _ = _load_scalar_critic(Path(args.slate_critic))

    results = []
    for run_dir in [Path(p) for p in args.run_dirs]:
        results.append(
            score_single_run(
                dataset=args.dataset,
                run_dir=run_dir,
                future_model=future_model,
                future_meta=future_meta,
                slate_critic=slate_critic,
                credit_gamma=float(args.credit_gamma),
                future_horizon=int(args.future_horizon),
                reward_weights=reward_weights,
                welfare_weights=welfare_weights,
            )
        )

    report = {"dataset": args.dataset, "results": results}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
