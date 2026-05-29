from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline.hierarchical_credit_phase1 import (  # noqa: E402
    ArrayDataset,
    DEFAULT_REWARD_WEIGHTS,
    DEFAULT_WELFARE_WEIGHTS,
    FUTURE_TARGET_NAMES,
    FutureWelfareModel,
    build_phase1_artifact,
    describe_phase1_artifact,
    discover_default_runs,
    load_phase1_artifact,
    parse_weight_overrides,
    save_phase1_artifact,
    set_seed,
)


def split_indices(groups: List[str], valid_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if len(groups) <= 1 or float(valid_ratio) <= 0.0:
        idx = np.arange(len(groups), dtype=np.int64)
        return idx, idx.copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=float(valid_ratio), random_state=int(seed))
    dummy = np.zeros(len(groups), dtype=np.float32)
    train_idx, valid_idx = next(splitter.split(dummy, groups=groups))
    return np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64)


def run_epoch(
    *,
    model: FutureWelfareModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> Dict[str, float]:
    train_mode = optimizer is not None
    model.train(mode=train_mode)
    total_loss = 0.0
    total_count = 0
    mae_sum = np.zeros(len(FUTURE_TARGET_NAMES), dtype=np.float64)

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        pred = model(batch_x)
        loss = F.smooth_l1_loss(pred, batch_y)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_size = int(batch_x.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        mae_sum += torch.mean(torch.abs(pred - batch_y), dim=0).detach().cpu().numpy() * batch_size

    denom = max(total_count, 1)
    out = {"loss": float(total_loss / denom)}
    for idx, name in enumerate(FUTURE_TARGET_NAMES):
        out[f"mae_{name}"] = float(mae_sum[idx] / denom)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the external future-state / welfare model for Phase 1.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run_name_contains", default="", help="Substring filter for simulation runs.")
    parser.add_argument("--run_dirs", nargs="*", default=[], help="Optional explicit run directories.")
    parser.add_argument("--artifact_path", default="", help="Optional path to a prebuilt Phase 1 artifact.")
    parser.add_argument("--save_artifact_path", default="", help="Optional path to save the built Phase 1 artifact.")
    parser.add_argument("--future_horizon", type=int, default=5)
    parser.add_argument("--credit_gamma", type=float, default=0.88)
    parser.add_argument("--reward_weights", default="", help="Comma-separated overrides, e.g. continue=0.3,watch=1.0")
    parser.add_argument("--welfare_weights", default="", help="Comma-separated overrides for welfare weights.")
    parser.add_argument("--levels", nargs="+", default=["slate", "item"], choices=["slate", "plan", "item"])
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default="", help="Directory to save checkpoint and metrics.")
    args = parser.parse_args()

    set_seed(args.seed)
    reward_weights = parse_weight_overrides(args.reward_weights, DEFAULT_REWARD_WEIGHTS)
    welfare_weights = parse_weight_overrides(args.welfare_weights, DEFAULT_WELFARE_WEIGHTS)

    artifact_path = Path(args.artifact_path) if args.artifact_path else None
    if artifact_path and artifact_path.exists():
        artifact = load_phase1_artifact(artifact_path)
    else:
        run_dirs = [Path(p) for p in args.run_dirs] if args.run_dirs else discover_default_runs(args.dataset, args.run_name_contains)
        if not run_dirs:
            raise ValueError("No run dirs found for Phase 1 future-state training.")
        artifact = build_phase1_artifact(
            dataset=args.dataset,
            run_dirs=run_dirs,
            future_horizon=args.future_horizon,
            credit_gamma=args.credit_gamma,
            credit_reward_weights=reward_weights,
            welfare_weights=welfare_weights,
        )
        if args.save_artifact_path:
            save_phase1_artifact(artifact, Path(args.save_artifact_path))

    print(describe_phase1_artifact(artifact))

    feat_blocks = []
    target_blocks = []
    groups: List[str] = []
    for level in args.levels:
        entry = artifact[level]
        feat_blocks.append(np.asarray(entry["features"], dtype=np.float32))
        target_blocks.append(np.asarray(entry["future_targets"], dtype=np.float32))
        groups.extend(list(entry["groups"]))
    features = np.concatenate(feat_blocks, axis=0)
    targets = np.concatenate(target_blocks, axis=0)

    train_idx, valid_idx = split_indices(groups, args.valid_ratio, args.seed)
    dataset = ArrayDataset(features, targets)
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=int(args.batch_size), shuffle=True)
    valid_loader = DataLoader(Subset(dataset, valid_idx.tolist()), batch_size=int(args.batch_size), shuffle=False)

    device = torch.device(args.device)
    model = FutureWelfareModel(
        input_dim=int(features.shape[1]),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
        out_dim=int(targets.shape[1]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    history: List[Dict[str, float]] = []
    best_state = None
    best_valid = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = run_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device)
        valid_metrics = run_epoch(model=model, loader=valid_loader, optimizer=None, device=device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if valid_metrics["loss"] < best_valid:
            best_valid = float(valid_metrics["loss"])
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    save_dir = Path(args.save_dir) if args.save_dir else REPO_ROOT / "artifacts" / args.dataset / "phase1_future_welfare"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / "future_welfare_model.pt"
    metrics_path = save_dir / "future_welfare_metrics.json"
    payload = {
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "meta": {
            "dataset": args.dataset,
            "levels": list(args.levels),
            "feature_dim": int(features.shape[1]),
            "target_names": list(FUTURE_TARGET_NAMES),
            "future_horizon": int(artifact["meta"]["future_horizon"]),
            "credit_gamma": float(artifact["meta"]["credit_gamma"]),
            "reward_weights": reward_weights,
            "welfare_weights": welfare_weights,
        },
    }
    torch.save(payload, ckpt_path)
    metrics_path.write_text(
        json.dumps(
            {
                "best_valid_loss": float(best_valid),
                "history": history,
                "artifact_meta": artifact["meta"],
                "artifact_stats": artifact["stats"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved checkpoint to {ckpt_path}")
    print(f"saved metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
