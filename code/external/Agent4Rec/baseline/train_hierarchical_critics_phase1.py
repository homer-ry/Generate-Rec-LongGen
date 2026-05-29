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
    ScalarCritic,
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


def fit_single_critic(
    *,
    name: str,
    features: np.ndarray,
    targets: np.ndarray,
    groups: List[str],
    hidden_dim: int,
    depth: int,
    dropout: float,
    batch_size: int,
    epochs: int,
    lr: float,
    valid_ratio: float,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    train_idx, valid_idx = split_indices(groups, valid_ratio, seed)
    dataset = ArrayDataset(features, targets.reshape(-1, 1))
    train_loader = DataLoader(Subset(dataset, train_idx.tolist()), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(Subset(dataset, valid_idx.tolist()), batch_size=batch_size, shuffle=False)

    model = ScalarCritic(
        input_dim=int(features.shape[1]),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        dropout=float(dropout),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    best_valid = float("inf")
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).squeeze(-1)
            pred = model(batch_x)
            loss = F.smooth_l1_loss(pred, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size_cur = int(batch_x.shape[0])
            train_loss += float(loss.item()) * batch_size_cur
            train_count += batch_size_cur

        model.eval()
        valid_loss = 0.0
        valid_mae = 0.0
        valid_count = 0
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).squeeze(-1)
                pred = model(batch_x)
                loss = F.smooth_l1_loss(pred, batch_y)
                batch_size_cur = int(batch_x.shape[0])
                valid_loss += float(loss.item()) * batch_size_cur
                valid_mae += float(torch.mean(torch.abs(pred - batch_y)).item()) * batch_size_cur
                valid_count += batch_size_cur

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss / max(train_count, 1)),
            "valid_loss": float(valid_loss / max(valid_count, 1)),
            "valid_mae": float(valid_mae / max(valid_count, 1)),
        }
        history.append(row)
        print(json.dumps({"critic": name, **row}, ensure_ascii=False))
        if row["valid_loss"] < best_valid:
            best_valid = float(row["valid_loss"])
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    return {
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "history": history,
        "best_valid_loss": float(best_valid),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Phase 1 external hierarchical critics (slate/plan/item).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run_name_contains", default="", help="Substring filter for simulation runs.")
    parser.add_argument("--run_dirs", nargs="*", default=[], help="Optional explicit run directories.")
    parser.add_argument("--artifact_path", default="", help="Optional path to a prebuilt Phase 1 artifact.")
    parser.add_argument("--save_artifact_path", default="", help="Optional path to save the built Phase 1 artifact.")
    parser.add_argument("--future_horizon", type=int, default=5)
    parser.add_argument("--credit_gamma", type=float, default=0.88)
    parser.add_argument("--reward_weights", default="", help="Comma-separated reward weight overrides.")
    parser.add_argument("--welfare_weights", default="", help="Comma-separated welfare weight overrides.")
    parser.add_argument("--welfare_scale", type=float, default=1.0, help="Scale of welfare target added to MC return target.")
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default="", help="Directory to save critic checkpoints and metrics.")
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
            raise ValueError("No run dirs found for Phase 1 critic training.")
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
    device = torch.device(args.device)

    out: Dict[str, object] = {
        "artifact_meta": artifact["meta"],
        "artifact_stats": artifact["stats"],
        "welfare_scale": float(args.welfare_scale),
        "critics": {},
    }
    save_dir = Path(args.save_dir) if args.save_dir else REPO_ROOT / "artifacts" / args.dataset / "phase1_critics"
    save_dir.mkdir(parents=True, exist_ok=True)

    for level in ("slate", "plan", "item"):
        entry = artifact[level]
        features = np.asarray(entry["features"], dtype=np.float32)
        groups = list(entry["groups"])
        value_targets = np.asarray(entry["value_targets"], dtype=np.float32)
        welfare_targets = np.asarray(entry["future_targets"], dtype=np.float32)[:, -1]
        targets = value_targets + float(args.welfare_scale) * welfare_targets
        result = fit_single_critic(
            name=level,
            features=features,
            targets=targets,
            groups=groups,
            hidden_dim=int(args.hidden_dim),
            depth=int(args.depth),
            dropout=float(args.dropout),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            valid_ratio=float(args.valid_ratio),
            seed=int(args.seed),
            device=device,
        )
        ckpt_path = save_dir / f"{level}_critic.pt"
        torch.save(
            {
                "state_dict": result["state_dict"],
                "meta": {
                    "dataset": args.dataset,
                    "level": level,
                    "feature_dim": int(features.shape[1]),
                    "welfare_scale": float(args.welfare_scale),
                    "target_definition": "discounted_return + welfare_scale * future_welfare",
                },
            },
            ckpt_path,
        )
        out["critics"][level] = {
            "checkpoint": str(ckpt_path),
            "best_valid_loss": result["best_valid_loss"],
            "history": result["history"],
        }

    metrics_path = save_dir / "phase1_critics_metrics.json"
    metrics_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved critic metrics to {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
