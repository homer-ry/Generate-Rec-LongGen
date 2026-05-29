from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.hierarchical_credit_phase2 import (  # noqa: E402
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_WELFARE_WEIGHTS,
    build_phase2_segment_artifact,
    build_tiger_runtime,
    describe_phase2_segment_artifact,
    discover_default_phase2_runs,
    load_phase2_segment_artifact,
    parse_weight_overrides,
    save_phase2_segment_artifact,
    set_seed,
    train_segment_transport_head,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Phase 2 external item-to-SID-block transport head with conservation supervision."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run_name_contains", default="", help="Substring filter for simulation runs.")
    parser.add_argument("--run_dirs", nargs="*", default=[], help="Optional explicit run directories.")
    parser.add_argument("--artifact_path", default="", help="Optional existing Phase 2 artifact path.")
    parser.add_argument("--save_artifact_path", default="", help="Optional path to persist the built Phase 2 artifact.")
    parser.add_argument("--cf_data_subdir", default="cf_data")
    parser.add_argument("--tiger_model_path", default="Saved")
    parser.add_argument("--only_items_per_page", type=int, default=1)
    parser.add_argument("--future_horizon", type=int, default=5)
    parser.add_argument("--credit_gamma", type=float, default=0.88)
    parser.add_argument("--reward_weights", default="", help="Comma-separated reward weight overrides.")
    parser.add_argument("--welfare_weights", default="", help="Comma-separated welfare weight overrides.")
    parser.add_argument("--welfare_scale", type=float, default=1.0)
    parser.add_argument("--item_credit_source", default="manual_mix", choices=["manual_mix", "phase1_item_critic", "blend"])
    parser.add_argument("--phase1_item_critic_path", default="")
    parser.add_argument("--item_credit_blend_alpha", type=float, default=0.5)
    parser.add_argument("--max_history_items", type=int, default=50)
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--token_dim", type=int, default=32)
    parser.add_argument("--mlp_dim", type=int, default=128)
    parser.add_argument("--conservation_scale", type=float, default=0.25)
    parser.add_argument("--sign_scale", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default="", help="Directory to save head checkpoint and metrics.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(int(args.seed))
    reward_weights = parse_weight_overrides(args.reward_weights, DEFAULT_REWARD_WEIGHTS)
    welfare_weights = parse_weight_overrides(args.welfare_weights, DEFAULT_WELFARE_WEIGHTS)

    artifact_path = Path(args.artifact_path) if args.artifact_path else None
    if artifact_path and artifact_path.exists():
        artifact = load_phase2_segment_artifact(artifact_path)
    else:
        run_dirs = [Path(p) for p in args.run_dirs] if args.run_dirs else discover_default_phase2_runs(args.dataset, args.run_name_contains)
        if not run_dirs:
            raise ValueError("No run dirs found for Phase 2 segment transport training.")
        artifact = build_phase2_segment_artifact(
            dataset=args.dataset,
            run_dirs=run_dirs,
            cf_data_subdir=str(args.cf_data_subdir),
            tiger_model_path=str(args.tiger_model_path),
            device=str(args.device),
            only_items_per_page=int(args.only_items_per_page),
            future_horizon=int(args.future_horizon),
            credit_gamma=float(args.credit_gamma),
            credit_reward_weights=reward_weights,
            welfare_weights=welfare_weights,
            welfare_scale=float(args.welfare_scale),
            max_history_items=int(args.max_history_items),
            item_credit_source=str(args.item_credit_source),
            phase1_item_critic_path=str(args.phase1_item_critic_path),
            item_credit_blend_alpha=float(args.item_credit_blend_alpha),
        )
        if args.save_artifact_path:
            save_phase2_segment_artifact(artifact, Path(args.save_artifact_path))

    print(describe_phase2_segment_artifact(artifact))

    tiger, _, _ = build_tiger_runtime(
        dataset=str(artifact["meta"]["dataset"]),
        cf_data_subdir=str(artifact["meta"]["cf_data_subdir"]),
        tiger_model_path=str(artifact["meta"]["tiger_model_path"]),
        device=str(args.device),
    )
    train_out = train_segment_transport_head(
        tiger=tiger,
        records=artifact["records"],
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        lr=float(args.lr),
        valid_ratio=float(args.valid_ratio),
        token_dim=int(args.token_dim),
        mlp_dim=int(args.mlp_dim),
        conservation_scale=float(args.conservation_scale),
        sign_scale=float(args.sign_scale),
        seed=int(args.seed),
        device=torch.device(args.device),
        patience=int(args.patience),
    )

    save_dir = Path(args.save_dir) if args.save_dir else REPO_ROOT / "artifacts" / args.dataset / "phase2_sid_block_transport"
    save_dir.mkdir(parents=True, exist_ok=True)
    head_path = save_dir / "segment_transport_head.pt"
    meta_path = save_dir / "segment_transport_meta.json"
    metrics_path = save_dir / "segment_transport_metrics.json"

    torch.save({"model_state_dict": train_out["state_dict"]}, head_path)
    meta = {
        "dataset": str(artifact["meta"]["dataset"]),
        "cf_data_subdir": str(artifact["meta"]["cf_data_subdir"]),
        "tiger_model_path": str(artifact["meta"]["tiger_model_path"]),
        "sid_depth": int(artifact["meta"]["sid_depth"]),
        "hidden_size": int(tiger.backbone.model.config.d_model),
        "vocab_size": int(tiger.backbone.model.config.vocab_size),
        "token_dim": int(args.token_dim),
        "mlp_dim": int(args.mlp_dim),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "conservation_scale": float(args.conservation_scale),
        "sign_scale": float(args.sign_scale),
        "best_epoch": int(train_out["best_epoch"]),
        "best_metrics": train_out["best_metrics"],
        "split": train_out["split"],
        "artifact_meta": artifact["meta"],
        "artifact_stats": artifact["stats"],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "head_path": str(head_path),
                "meta_path": str(meta_path),
                "best_epoch": int(train_out["best_epoch"]),
                "best_metrics": train_out["best_metrics"],
                "split": train_out["split"],
                "history": train_out["history"],
                "artifact_meta": artifact["meta"],
                "artifact_stats": artifact["stats"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved Phase 2 head to {head_path}")
    print(f"saved Phase 2 metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
