from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from baseline.agents.a2c import A2CAgent, A2CConfig
from baseline.agents.cql import CQLAgent, CQLConfig
from baseline.agents.common import set_seed
from baseline.agents.ddpg import DDPGAgent, DDPGConfig
from baseline.agents.dqn import DQNAgent, DQNConfig, RainbowAgent
from baseline.agents.hac import HACAgent, HACConfig
from baseline.agents.iql import IQLAgent, IQLConfig
from baseline.agents.ppo import PPOAgent, PPOConfig
from baseline.agents.td3 import TD3Agent, TD3Config
from baseline.run_kuaisim_baselines import train_a2c, train_dqn, train_offpolicy, train_ppo
from baseline.session_env import LongSessionRecEnv


def make_env(args, seed_offset: int = 0):
    return LongSessionRecEnv(
        root_dir=args.root_dir,
        dataset=args.dataset,
        max_pages=args.max_pages,
        slate_size=args.slate_size,
        seed=args.seed + seed_offset,
        user_limit=args.user_limit,
        exit_threshold_scale=args.exit_threshold_scale,
        repetition_penalty_scale=args.repetition_penalty_scale,
        repeat_aversion_scale=args.repeat_aversion_scale,
        fatigue_scale=args.fatigue_scale,
        terminal_score_max=args.terminal_score_max,
    )


def write_model_args(save_dir: Path, modeltype: str, dataset: str, seed: int) -> None:
    data = {
        "vis": -1,
        "seed": int(seed),
        "clear_checkpoints": True,
        "candidate": False,
        "test_only": False,
        "data_path": "../datasets/",
        "dataset": dataset,
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
        "modeltype": modeltype,
    }
    (save_dir / "args.txt").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _cos_batch(item_genre: np.ndarray, recent_genre: np.ndarray) -> np.ndarray:
    nr = float(np.linalg.norm(recent_genre))
    if nr <= 1e-8:
        return np.zeros((item_genre.shape[0],), dtype=np.float32)
    dots = item_genre @ recent_genre
    ni = np.linalg.norm(item_genre, axis=1)
    denom = np.clip(ni * nr, 1e-8, None)
    return np.clip(dots / denom, -1.0, 1.0).astype(np.float32)


def _advance_deterministic(env: LongSessionRecEnv, uid: int, iid: int, rel: float, nov: float) -> None:
    traits = env._norm_traits(uid)
    activity_norm = float(traits[0])
    conformity_norm = float(traits[2])

    current_genre = env.item_genre[iid]
    genre_rep = env._cosine(current_genre, env.recent_genre)

    p_watch = (
        0.05
        + 0.35 * activity_norm
        + 0.55 * rel
        + 0.05 * nov
        - env.repetition_penalty_scale * env.repeat_aversion_scale * 0.35 * genre_rep
    )
    p_watch = float(np.clip(p_watch, 0.01, 0.99))

    p_like = 0.05 + 0.65 * rel + 0.10 * conformity_norm * float(env.item_quality[iid]) + 0.10 * nov
    p_like = float(np.clip(p_like, 0.01, 0.99))

    like_ratio = p_watch * p_like
    page_div = 1.0 - genre_rep

    satisfaction = (
        0.55 * rel
        + 0.25 * nov
        + 0.20 * page_div
        + 0.15 * like_ratio
        - env.repetition_penalty_scale * env.repeat_aversion_scale * 0.35 * genre_rep
        - 0.08 * (1.0 - p_watch)
    )
    satisfaction = float(np.clip(satisfaction, -1.0, 1.0))

    fatigue = env.fatigue_scale * (0.02 + 0.04 * (1.0 - activity_norm))
    env.dissatisfaction += max(0.0, 0.45 - satisfaction)
    env.dissatisfaction += env.repetition_penalty_scale * env.repeat_aversion_scale * 0.45 * genre_rep + fatigue
    env.dissatisfaction -= max(0.0, satisfaction - 0.55) * 0.20
    env.dissatisfaction = max(0.0, env.dissatisfaction)

    env.recent_genre = 0.65 * env.recent_genre + 0.35 * current_genre
    gsum = float(env.recent_genre.sum())
    if gsum > 1e-8:
        env.recent_genre = env.recent_genre / gsum
    env.last_genre_repetition = float(genre_rep)
    env.recent_satisfaction = 0.70 * env.recent_satisfaction + 0.30 * satisfaction
    env.page_idx = min(int(env.max_pages), int(env.page_idx) + 1)


def build_score_cache(agent, env: LongSessionRecEnv, rollout_topk: int) -> np.ndarray:
    n_users = int(max(env.user_ids.tolist()) + 1)
    n_items = int(env.n_items)
    out = np.full((n_users, n_items), -1e9, dtype=np.float32)

    item_genre_all = env.item_genre
    novelty_all = env.item_novelty
    popularity_all = env.item_popularity

    topk = max(1, int(min(rollout_topk, n_items)))

    # v1: keep ranking anchored to relevance/popularity and let RL only modulate.
    rel_coef = 1.00
    pop_coef = 0.35
    nov_coef = 0.05
    div_coef = 0.08
    rep_penalty = 0.10
    rl_coef = 0.22
    pop_floor = float(np.quantile(popularity_all, 0.15))
    sel_bonus_hi = 0.20
    sel_bonus_lo = 0.05

    for uid in env.user_ids.tolist():
        uid = int(uid)
        state = env.reset(user_id=uid)

        pref = env.user_pref[uid]
        relevance_all = (item_genre_all @ pref).astype(np.float32)
        base = rel_coef * relevance_all + pop_coef * popularity_all + nov_coef * novelty_all
        score = base.astype(np.float32).copy()

        chosen = np.zeros((n_items,), dtype=bool)
        for step in range(topk):
            avail = np.where(~chosen)[0]
            if avail.size == 0:
                break

            action = agent.eval_action(state)
            w = env._action_to_weights(action)

            rel = relevance_all[avail]
            nov = novelty_all[avail]
            pop = popularity_all[avail]
            rep = _cos_batch(item_genre_all[avail], env.recent_genre)
            if float(np.linalg.norm(env.recent_genre)) <= 1e-8:
                div = np.full((avail.size,), 0.5, dtype=np.float32)
            else:
                div = 1.0 - rep

            pop_gate = np.where(pop >= pop_floor, 0.0, -0.12).astype(np.float32)
            anchor = (
                rel_coef * rel
                + pop_coef * pop
                + nov_coef * nov
                + div_coef * div
                - rep_penalty * rep
                + pop_gate
            )
            rl_term = rl_coef * (w[0] * rel + 0.25 * w[1] * nov + 0.20 * w[2] * div)
            dyn = anchor + rl_term
            best_local = int(np.argmax(dyn))
            iid = int(avail[best_local])
            chosen[iid] = True

            decay = float(topk - step) / float(topk)
            bonus = sel_bonus_lo + (sel_bonus_hi - sel_bonus_lo) * decay
            score[iid] = max(float(score[iid]), float(anchor[best_local])) + float(bonus)

            _advance_deterministic(
                env=env,
                uid=uid,
                iid=iid,
                rel=float(rel[best_local]),
                nov=float(nov[best_local]),
            )
            state = env._get_state()

        out[uid] = score

    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Train RL baselines and export score cache for Agent4Rec main.py simulation"
    )
    p.add_argument("--root_dir", type=str, default=".")
    p.add_argument("--dataset", type=str, default="ml-1m")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--device", type=str, default="cpu")

    p.add_argument("--algos", type=str, default="TD,DDPG,HAC,A2C")
    p.add_argument("--train_episodes", type=int, default=180)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--warmup_steps", type=int, default=800)
    p.add_argument("--updates_per_step", type=int, default=1)
    p.add_argument("--buffer_capacity", type=int, default=120000)
    p.add_argument("--terminal_bonus_coef", type=float, default=0.10)
    p.add_argument("--terminal_bonus_spread", type=str, default="last", choices=["last", "all"])

    p.add_argument("--max_pages", type=int, default=20)
    p.add_argument("--slate_size", type=int, default=1)
    p.add_argument("--user_limit", type=int, default=300)
    p.add_argument("--exit_threshold_scale", type=float, default=1.35)
    p.add_argument("--repetition_penalty_scale", type=float, default=1.20)
    p.add_argument("--repeat_aversion_scale", type=float, default=1.50)
    p.add_argument("--fatigue_scale", type=float, default=1.80)
    p.add_argument("--terminal_score_max", type=float, default=10.0)

    p.add_argument("--rollout_topk", type=int, default=120)
    p.add_argument("--output", type=str, default="baseline/results/rl_mainpy_export_metrics.json")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    root = Path(args.root_dir).resolve()
    out_path = (root / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    algos = [a.strip().upper() for a in args.algos.split(",") if a.strip()]

    all_results: Dict[str, Dict] = {
        "config": {
            "dataset": args.dataset,
            "seed": args.seed,
            "train_episodes": args.train_episodes,
            "max_pages": args.max_pages,
            "slate_size": args.slate_size,
            "user_limit": args.user_limit,
            "rollout_topk": args.rollout_topk,
            "terminal_bonus_coef": args.terminal_bonus_coef,
            "terminal_bonus_spread": args.terminal_bonus_spread,
        },
        "results": {},
    }

    for idx, algo in enumerate(algos):
        t0 = time.time()
        env = make_env(args, seed_offset=100 + idx)

        if algo == "TD":
            agent = TD3Agent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                device=device,
                cfg=TD3Config(),
            )
            train_summary, steps = train_offpolicy(
                algo_name=algo,
                agent=agent,
                env=env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )
            policy_state = {"actor": agent.actor.state_dict(), "actor_target": agent.actor_target.state_dict()}
        elif algo == "DDPG":
            agent = DDPGAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                device=device,
                cfg=DDPGConfig(),
            )
            train_summary, steps = train_offpolicy(
                algo_name=algo,
                agent=agent,
                env=env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )
            policy_state = {"actor": agent.actor.state_dict(), "actor_target": agent.actor_target.state_dict()}
        elif algo == "HAC":
            agent = HACAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                effect_dim=env.effect_dim,
                device=device,
                cfg=HACConfig(),
            )
            train_summary, steps = train_offpolicy(
                algo_name=algo,
                agent=agent,
                env=env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )
            policy_state = {"actor": agent.actor.state_dict(), "actor_target": agent.actor_target.state_dict()}
        elif algo == "A2C":
            agent = A2CAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                device=device,
                cfg=A2CConfig(),
            )
            agent.terminal_bonus_coef = float(args.terminal_bonus_coef)
            agent.terminal_bonus_spread = str(args.terminal_bonus_spread)
            train_summary = train_a2c(agent=agent, env=env, train_episodes=args.train_episodes)
            steps = 0
            policy_state = {"net": agent.net.state_dict()}
        elif algo == "PPO":
            agent = PPOAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                device=device,
                cfg=PPOConfig(),
            )
            agent.terminal_bonus_coef = float(args.terminal_bonus_coef)
            agent.terminal_bonus_spread = str(args.terminal_bonus_spread)
            train_summary = train_ppo(agent=agent, env=env, train_episodes=args.train_episodes)
            steps = 0
            policy_state = {"net": agent.net.state_dict()}
        elif algo == "CQL":
            agent = CQLAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                device=device,
                cfg=CQLConfig(),
            )
            train_summary, steps = train_offpolicy(
                algo_name=algo,
                agent=agent,
                env=env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )
            policy_state = {
                "actor": agent.actor.state_dict(),
                "actor_target": agent.actor_target.state_dict(),
            }
        elif algo == "IQL":
            agent = IQLAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                device=device,
                cfg=IQLConfig(),
            )
            train_summary, steps = train_offpolicy(
                algo_name=algo,
                agent=agent,
                env=env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )
            policy_state = {
                "actor": agent.actor.state_dict(),
                "critic": agent.critic.state_dict(),
                "value": agent.value.state_dict(),
            }
        elif algo == "DQN":
            agent = DQNAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                device=device,
                cfg=DQNConfig(),
            )
            train_summary, steps = train_dqn(
                agent=agent,
                env=env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )
            policy_state = {
                "q_net": agent.q_net.state_dict(),
                "q_target": agent.q_target.state_dict(),
                "action_table": agent.action_table,
            }
        elif algo == "RAINBOW":
            agent = RainbowAgent(
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                device=device,
            )
            train_summary, steps = train_dqn(
                agent=agent,
                env=env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )
            policy_state = {
                "q_net": agent.q_net.state_dict(),
                "q_target": agent.q_target.state_dict(),
                "action_table": agent.action_table,
            }
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

        score_cache = build_score_cache(agent=agent, env=env, rollout_topk=args.rollout_topk)

        save_dir = root / "recommenders" / "weights" / args.dataset / algo / "Saved"
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "score_cache.npy", score_cache)
        torch.save(
            {
                "algo": algo,
                "state_dim": int(env.state_dim),
                "action_dim": int(env.action_dim),
                "policy_state": policy_state,
                "train_summary": train_summary,
            },
            save_dir / "policy_state.pth",
        )
        write_model_args(save_dir=save_dir, modeltype=algo, dataset=args.dataset, seed=args.seed)

        elapsed = time.time() - t0
        all_results["results"][algo] = {
            **train_summary,
            "train_steps": int(steps),
            "elapsed_sec": float(elapsed),
            "score_cache_path": str((save_dir / "score_cache.npy").resolve()),
            "policy_state_path": str((save_dir / "policy_state.pth").resolve()),
            "args_path": str((save_dir / "args.txt").resolve()),
            "score_cache_shape": [int(score_cache.shape[0]), int(score_cache.shape[1])],
        }
        print(
            f"{algo}: train_return={all_results['results'][algo].get('train_mean_return', 0.0):.4f}, "
            f"train_len={all_results['results'][algo].get('train_mean_length', 0.0):.4f}, "
            f"elapsed={elapsed:.1f}s"
        )

    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    csv_path = out_path.with_suffix(".csv")
    header = [
        "algo",
        "train_mean_return",
        "train_mean_length",
        "train_episodes",
        "train_steps",
        "elapsed_sec",
        "score_cache_path",
    ]
    lines: List[str] = [",".join(header)]
    for algo, m in all_results["results"].items():
        lines.append(
            ",".join(
                [
                    algo,
                    f"{m.get('train_mean_return', 0.0):.6f}",
                    f"{m.get('train_mean_length', 0.0):.6f}",
                    str(int(m.get("train_episodes", 0))),
                    str(int(m.get("train_steps", 0))),
                    f"{m.get('elapsed_sec', 0.0):.6f}",
                    m.get("score_cache_path", ""),
                ]
            )
        )
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {out_path}")
    print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
