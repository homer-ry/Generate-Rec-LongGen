from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.hierarchical_credit_phase2 import (  # noqa: E402
    SegmentTransportDataset,
    build_phase2_segment_artifact,
    build_tiger_runtime,
    collate_segment_transport,
    describe_phase2_segment_artifact,
    discover_default_phase2_runs,
    evaluate_segment_head,
    load_phase2_segment_artifact,
    load_segment_head,
)


def _spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if float(np.std(x)) < 1e-8 or float(np.std(y)) < 1e-8:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score Phase 2 SID-block transport head on one or more simulation runs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--head_path", required=True)
    parser.add_argument("--meta_path", required=True)
    parser.add_argument("--artifact_path", default="", help="Optional existing Phase 2 artifact path.")
    parser.add_argument("--run_name_contains", default="", help="Substring filter for simulation runs.")
    parser.add_argument("--run_dirs", nargs="*", default=[], help="Optional explicit run directories.")
    parser.add_argument("--cf_data_subdir", default="cf_data")
    parser.add_argument("--tiger_model_path", default="Saved")
    parser.add_argument("--only_items_per_page", type=int, default=1)
    parser.add_argument("--future_horizon", type=int, default=5)
    parser.add_argument("--credit_gamma", type=float, default=0.88)
    parser.add_argument("--welfare_scale", type=float, default=1.0)
    parser.add_argument("--item_credit_source", default="manual_mix", choices=["manual_mix", "phase1_item_critic", "blend"])
    parser.add_argument("--phase1_item_critic_path", default="")
    parser.add_argument("--item_credit_blend_alpha", type=float, default=0.5)
    parser.add_argument("--max_history_items", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_json", default="", help="Optional output json path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifact_path = Path(args.artifact_path) if args.artifact_path else None
    if artifact_path and artifact_path.exists():
        artifact = load_phase2_segment_artifact(artifact_path)
    else:
        run_dirs = [Path(p) for p in args.run_dirs] if args.run_dirs else discover_default_phase2_runs(args.dataset, args.run_name_contains)
        if not run_dirs:
            raise ValueError("No run dirs found for Phase 2 segment transport scoring.")
        artifact = build_phase2_segment_artifact(
            dataset=args.dataset,
            run_dirs=run_dirs,
            cf_data_subdir=str(args.cf_data_subdir),
            tiger_model_path=str(args.tiger_model_path),
            device=str(args.device),
            only_items_per_page=int(args.only_items_per_page),
            future_horizon=int(args.future_horizon),
            credit_gamma=float(args.credit_gamma),
            credit_reward_weights={
                "continue": 0.30,
                "watch": 1.00,
                "align": 0.30,
                "rating": 0.20,
                "negative": 0.80,
            },
            welfare_weights={
                "future_depth_norm": 0.25,
                "future_continue_h": 0.25,
                "future_rating_norm": 0.20,
                "future_neg_rate": -0.20,
                "future_click_rate": 0.10,
                "future_interview_norm": 0.20,
            },
            welfare_scale=float(args.welfare_scale),
            max_history_items=int(args.max_history_items),
            item_credit_source=str(args.item_credit_source),
            phase1_item_critic_path=str(args.phase1_item_critic_path),
            item_credit_blend_alpha=float(args.item_credit_blend_alpha),
        )

    print(describe_phase2_segment_artifact(artifact))
    tiger, _, _ = build_tiger_runtime(
        dataset=str(artifact["meta"]["dataset"]),
        cf_data_subdir=str(artifact["meta"]["cf_data_subdir"]),
        tiger_model_path=str(artifact["meta"]["tiger_model_path"]),
        device=str(args.device),
    )
    head = load_segment_head(
        head_path=Path(args.head_path),
        meta_path=Path(args.meta_path),
        tiger=tiger,
        device=torch.device(args.device),
    )

    dataset = SegmentTransportDataset(artifact["records"])
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_segment_transport,
    )
    overall_metrics = evaluate_segment_head(tiger=tiger, head=head, loader=loader, device=torch.device(args.device))

    user_pred: Dict[str, List[float]] = {}
    user_true: Dict[str, List[float]] = {}
    user_exit: Dict[str, float] = {}
    user_interview: Dict[str, float] = {}

    head.eval()
    with torch.no_grad():
        for start in range(0, len(artifact["records"]), int(args.batch_size)):
            batch_records = artifact["records"][start : start + int(args.batch_size)]
            batch = collate_segment_transport([dataset[i] for i in range(start, min(start + int(args.batch_size), len(dataset)))])
            input_ids, attention_mask, target_tokens, _, item_credit = batch
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)
            target_tokens = target_tokens.to(args.device)
            _, hidden = tiger.backbone.decode_with_hidden(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=torch.cat(
                    [
                        torch.zeros((target_tokens.shape[0], 1), dtype=torch.long, device=target_tokens.device),
                        target_tokens[:, :-1],
                    ],
                    dim=1,
                ),
            )
            pred = head(hidden.detach(), target_tokens)
            pred_credit = torch.sum(pred, dim=1).detach().cpu().numpy().reshape(-1)
            true_credit = item_credit.detach().cpu().numpy().reshape(-1)
            for rec, p, y in zip(batch_records, pred_credit.tolist(), true_credit.tolist()):
                key = str(rec["group"])
                user_pred.setdefault(key, []).append(float(p))
                user_true.setdefault(key, []).append(float(y))
                user_exit[key] = float(rec.get("exit_page", 0.0))
                user_interview[key] = float(rec.get("interview_norm", 0.0))

    pred_means = [float(np.mean(v)) for v in user_pred.values()]
    true_means = [float(np.mean(user_true[k])) for k in user_pred.keys()]
    exits = [float(user_exit[k]) for k in user_pred.keys()]
    interviews = [float(user_interview[k]) for k in user_pred.keys()]

    report = {
        "overall": overall_metrics,
        "user_level": {
            "n_users": int(len(user_pred)),
            "pred_mean_vs_true_credit": _spearman(pred_means, true_means),
            "pred_mean_vs_exit_page": _spearman(pred_means, exits),
            "pred_mean_vs_interview_norm": _spearman(pred_means, interviews),
        },
        "artifact_meta": artifact["meta"],
        "artifact_stats": artifact["stats"],
    }
    out_json = Path(args.out_json) if args.out_json else REPO_ROOT / "artifacts" / args.dataset / "phase2_sid_block_transport_score.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved Phase 2 score report to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
