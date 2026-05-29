from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

import sys

if __package__ is None or __package__ == "":
    # Allow direct script execution: python baseline/run_kuaisim_baselines.py
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from baseline.agents.a2c import A2CAgent, A2CConfig
from baseline.agents.cql import CQLAgent, CQLConfig
from baseline.agents.common import EpisodeStats, ReplayBuffer, set_seed
from baseline.agents.ddpg import DDPGAgent, DDPGConfig
from baseline.agents.dqn import DQNAgent, DQNConfig, RainbowAgent
from baseline.agents.hac import HACAgent, HACConfig
from baseline.agents.iql import IQLAgent, IQLConfig
from baseline.agents.ppo import PPOAgent, PPOConfig
from baseline.agents.sequence_bc import SeqBCConfig, SequenceBCAgent
from baseline.agents.td3 import TD3Agent, TD3Config
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


def evaluate_agent(agent, env, episodes: int, algo_name: str) -> Dict[str, float]:
    records = []
    reason_counter: Dict[str, int] = {}
    terminal_scores = []

    for _ in range(episodes):
        state = env.reset()
        if hasattr(agent, "reset_episode"):
            agent.reset_episode()
        done = False
        ep_stats = EpisodeStats()
        last_info = {}
        while not done:
            action = agent.eval_action(state)
            if hasattr(agent, "update_history"):
                agent.update_history(state)
            next_state, reward, done, info = env.step(action)
            ep_stats.update(reward, info)
            state = next_state
            last_info = info

        rec = ep_stats.as_dict()
        reason = str(last_info.get("done_reason", ""))
        rec["done_reason"] = reason
        reason_counter[reason] = reason_counter.get(reason, 0) + 1
        if last_info.get("terminal_score") is not None:
            terminal_scores.append(float(last_info["terminal_score"]))
        records.append(rec)

    returns = [r["return"] for r in records]
    lengths = [r["length"] for r in records]
    like_rates = [r["like_rate"] for r in records]
    reps = [r["avg_repetition"] for r in records]
    sats = [r["avg_satisfaction"] for r in records]
    retention_flags = [1.0 if r["length"] >= 5 else 0.0 for r in records]

    mean_return = float(np.mean(returns)) if returns else 0.0
    mean_len = float(np.mean(lengths)) if lengths else 0.0
    mean_like_rate = float(np.mean(like_rates)) if like_rates else 0.0
    mean_rep = float(np.mean(reps)) if reps else 0.0
    mean_sat = float(np.mean(sats)) if sats else 0.0
    retention_5 = float(np.mean(retention_flags)) if retention_flags else 0.0

    ltv_score = mean_return + 0.5 * mean_len

    out = {
        "algo": algo_name,
        "eval_episodes": int(episodes),
        "mean_return": mean_return,
        "mean_session_length": mean_len,
        "mean_like_rate": mean_like_rate,
        "mean_genre_repetition": mean_rep,
        "mean_satisfaction": mean_sat,
        "mean_terminal_score": float(np.mean(terminal_scores)) if terminal_scores else 0.0,
        "retention_5plus": retention_5,
        "ltv_score": ltv_score,
    }

    total = max(sum(reason_counter.values()), 1)
    for k, v in reason_counter.items():
        out[f"exit_reason::{k}"] = float(v / total)
    return out


def train_offpolicy(
    algo_name: str,
    agent,
    env,
    device: torch.device,
    train_episodes: int,
    batch_size: int,
    warmup_steps: int,
    updates_per_step: int,
    buffer_capacity: int,
    terminal_bonus_coef: float = 0.0,
    terminal_bonus_spread: str = "last",
) -> Tuple[Dict[str, float], int]:
    replay = ReplayBuffer(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        effect_dim=env.effect_dim,
        capacity=buffer_capacity,
    )

    global_step = 0
    train_returns = []
    train_lengths = []

    for ep in range(train_episodes):
        state = env.reset()
        done = False
        ep_stats = EpisodeStats()
        episode_buffer_indices = []

        while not done:
            if global_step < warmup_steps:
                action = np.random.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
            else:
                action = agent.select_action(state, explore=True)

            next_state, reward, done, info = env.step(action)

            buf_idx = replay.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                behavior_target=info.get("behavior_target_action"),
                effect_action=info.get("effect_action_feature"),
            )
            episode_buffer_indices.append(int(buf_idx))

            ep_stats.update(reward, info)
            state = next_state
            global_step += 1

            if done and terminal_bonus_coef > 0:
                terminal_score = info.get("terminal_score")
                if terminal_score is not None and len(episode_buffer_indices) > 0:
                    bonus = float(terminal_bonus_coef) * float(terminal_score)
                    if terminal_bonus_spread == "all":
                        delta = bonus / float(len(episode_buffer_indices))
                        for idx_ in episode_buffer_indices:
                            replay.add_to_reward(idx_, delta)
                    else:
                        replay.add_to_reward(episode_buffer_indices[-1], bonus)

            if replay.size >= max(batch_size, warmup_steps):
                for _ in range(updates_per_step):
                    batch = replay.sample(batch_size, device=device)
                    agent.train_step(batch)

        r = ep_stats.as_dict()
        train_returns.append(r["return"])
        train_lengths.append(r["length"])

    summary = {
        "train_mean_return": float(np.mean(train_returns)) if train_returns else 0.0,
        "train_mean_length": float(np.mean(train_lengths)) if train_lengths else 0.0,
        "train_episodes": int(train_episodes),
    }
    return summary, global_step


def train_a2c(agent: A2CAgent, env, train_episodes: int) -> Dict[str, float]:
    train_returns = []
    train_lengths = []

    for _ in range(train_episodes):
        state = env.reset()
        done = False

        traj = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }
        ep_stats = EpisodeStats()

        while not done:
            action, _, value = agent.sample_action(state)
            next_state, reward, done, info = env.step(action)

            traj["states"].append(state)
            traj["actions"].append(action)
            traj["rewards"].append(float(reward))
            traj["dones"].append(float(done))
            traj["values"].append(float(value))

            ep_stats.update(reward, info)
            state = next_state

        terminal_score = info.get("terminal_score")
        if terminal_score is not None and len(traj["rewards"]) > 0:
            bonus = float(getattr(agent, "terminal_bonus_coef", 0.0)) * float(terminal_score)
            spread = str(getattr(agent, "terminal_bonus_spread", "last")).lower()
            if bonus != 0.0:
                if spread == "all":
                    delta = bonus / float(len(traj["rewards"]))
                    traj["rewards"] = [float(r + delta) for r in traj["rewards"]]
                else:
                    traj["rewards"][-1] = float(traj["rewards"][-1] + bonus)

        traj["last_value"] = 0.0
        agent.update(traj)

        r = ep_stats.as_dict()
        train_returns.append(r["return"])
        train_lengths.append(r["length"])

    return {
        "train_mean_return": float(np.mean(train_returns)) if train_returns else 0.0,
        "train_mean_length": float(np.mean(train_lengths)) if train_lengths else 0.0,
        "train_episodes": int(train_episodes),
    }


def train_ppo(agent: PPOAgent, env, train_episodes: int) -> Dict[str, float]:
    train_returns = []
    train_lengths = []

    for _ in range(train_episodes):
        state = env.reset()
        done = False

        traj = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
            "values": [],
        }
        ep_stats = EpisodeStats()

        while not done:
            action, log_prob, value = agent.sample_action(state)
            next_state, reward, done, info = env.step(action)

            traj["states"].append(state)
            traj["actions"].append(action)
            traj["log_probs"].append(float(log_prob))
            traj["rewards"].append(float(reward))
            traj["dones"].append(float(done))
            traj["values"].append(float(value))

            ep_stats.update(reward, info)
            state = next_state

        terminal_score = info.get("terminal_score")
        if terminal_score is not None and len(traj["rewards"]) > 0:
            bonus = float(getattr(agent, "terminal_bonus_coef", 0.0)) * float(terminal_score)
            spread = str(getattr(agent, "terminal_bonus_spread", "last")).lower()
            if bonus != 0.0:
                if spread == "all":
                    delta = bonus / float(len(traj["rewards"]))
                    traj["rewards"] = [float(r + delta) for r in traj["rewards"]]
                else:
                    traj["rewards"][-1] = float(traj["rewards"][-1] + bonus)

        agent.update(traj)

        r = ep_stats.as_dict()
        train_returns.append(r["return"])
        train_lengths.append(r["length"])

    return {
        "train_mean_return": float(np.mean(train_returns)) if train_returns else 0.0,
        "train_mean_length": float(np.mean(train_lengths)) if train_lengths else 0.0,
        "train_episodes": int(train_episodes),
    }


class ReplayBufferDiscrete:
    def __init__(self, state_dim: int, capacity: int = 200000):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        self.not_done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.action_idx = np.zeros((self.capacity, 1), dtype=np.int64)

    def add(self, state: np.ndarray, action_idx: int, reward: float, next_state: np.ndarray, done: bool) -> int:
        idx = self.ptr
        self.state[idx] = state
        self.next_state[idx] = next_state
        self.reward[idx] = float(reward)
        self.not_done[idx] = 0.0 if done else 1.0
        self.action_idx[idx] = int(action_idx)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return idx

    def add_to_reward(self, idx: int, delta: float) -> None:
        self.reward[idx] += float(delta)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "state": torch.as_tensor(self.state[idx], device=device),
            "next_state": torch.as_tensor(self.next_state[idx], device=device),
            "reward": torch.as_tensor(self.reward[idx], device=device),
            "not_done": torch.as_tensor(self.not_done[idx], device=device),
            "action_idx": torch.as_tensor(self.action_idx[idx], device=device),
        }


def train_dqn(
    agent: DQNAgent,
    env,
    device: torch.device,
    train_episodes: int,
    batch_size: int,
    warmup_steps: int,
    updates_per_step: int,
    buffer_capacity: int,
    terminal_bonus_coef: float = 0.0,
    terminal_bonus_spread: str = "last",
) -> Tuple[Dict[str, float], int]:
    replay = ReplayBufferDiscrete(state_dim=env.state_dim, capacity=buffer_capacity)
    global_step = 0
    train_returns = []
    train_lengths = []

    for _ in range(train_episodes):
        state = env.reset()
        done = False
        ep_stats = EpisodeStats()
        episode_buffer_indices = []

        while not done:
            if global_step < warmup_steps:
                action_idx = int(np.random.randint(0, agent.n_actions))
                action = agent.action_table[action_idx].copy()
                agent.last_action_index = action_idx
            else:
                action = agent.select_action(state, explore=True)
                action_idx = int(agent.last_action_index)

            next_state, reward, done, info = env.step(action)
            buf_idx = replay.add(
                state=state,
                action_idx=action_idx,
                reward=float(reward),
                next_state=next_state,
                done=bool(done),
            )
            episode_buffer_indices.append(int(buf_idx))

            ep_stats.update(reward, info)
            state = next_state
            global_step += 1

            if done and terminal_bonus_coef > 0:
                terminal_score = info.get("terminal_score")
                if terminal_score is not None and len(episode_buffer_indices) > 0:
                    bonus = float(terminal_bonus_coef) * float(terminal_score)
                    if terminal_bonus_spread == "all":
                        delta = bonus / float(len(episode_buffer_indices))
                        for idx_ in episode_buffer_indices:
                            replay.add_to_reward(idx_, delta)
                    else:
                        replay.add_to_reward(episode_buffer_indices[-1], bonus)

            if replay.size >= max(batch_size, warmup_steps):
                for _ in range(updates_per_step):
                    batch = replay.sample(batch_size, device=device)
                    agent.train_step(batch)

        r = ep_stats.as_dict()
        train_returns.append(r["return"])
        train_lengths.append(r["length"])

    return {
        "train_mean_return": float(np.mean(train_returns)) if train_returns else 0.0,
        "train_mean_length": float(np.mean(train_lengths)) if train_lengths else 0.0,
        "train_episodes": int(train_episodes),
    }, global_step


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Legacy local simulator runner. "
            "For canonical KuaiSim evaluation, use scripts/run_kuaisim_wholesession_baselines.py "
            "with E:\\project\\Recommend\\KuaiSim-main\\KuaiSim-main."
        )
    )
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--train_episodes", type=int, default=180)
    parser.add_argument("--eval_episodes", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup_steps", type=int, default=800)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--buffer_capacity", type=int, default=120000)

    parser.add_argument("--max_pages", type=int, default=20)
    parser.add_argument("--slate_size", type=int, default=6)
    parser.add_argument("--user_limit", type=int, default=300)
    parser.add_argument("--exit_threshold_scale", type=float, default=1.35)
    parser.add_argument("--repetition_penalty_scale", type=float, default=1.20)
    parser.add_argument("--repeat_aversion_scale", type=float, default=1.50)
    parser.add_argument("--fatigue_scale", type=float, default=1.80)
    parser.add_argument("--terminal_score_max", type=float, default=10.0)

    parser.add_argument("--algos", type=str, default="TD,DDPG,HAC,A2C,SASREC,ONEREC,GRU4REC,TIGER")
    parser.add_argument("--seq_collect_episodes", type=int, default=130)
    parser.add_argument("--seq_epochs", type=int, default=8)
    parser.add_argument("--terminal_bonus_coef", type=float, default=0.10)
    parser.add_argument("--terminal_bonus_spread", type=str, default="last", choices=["last", "all"])
    parser.add_argument("--output", type=str, default="baseline/results/kuaisim_baselines_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        "[legacy] baseline/run_kuaisim_baselines.py uses LongSessionRecEnv, not the canonical KuaiSim clean evaluator. "
        "For paper-facing KuaiSim numbers, use scripts/run_kuaisim_wholesession_baselines.py."
    )
    set_seed(args.seed)

    device = torch.device(args.device)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    algos = [a.strip().upper() for a in args.algos.split(",") if a.strip()]

    all_results = {
        "config": {
            "dataset": args.dataset,
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "max_pages": args.max_pages,
            "slate_size": args.slate_size,
            "user_limit": args.user_limit,
            "exit_threshold_scale": args.exit_threshold_scale,
            "repetition_penalty_scale": args.repetition_penalty_scale,
            "repeat_aversion_scale": args.repeat_aversion_scale,
            "fatigue_scale": args.fatigue_scale,
            "terminal_score_max": args.terminal_score_max,
            "terminal_bonus_coef": args.terminal_bonus_coef,
            "terminal_bonus_spread": args.terminal_bonus_spread,
            "seed": args.seed,
        },
        "results": {},
    }

    for idx, algo in enumerate(algos):
        train_env = make_env(args, seed_offset=10 + idx)
        eval_env = make_env(args, seed_offset=100 + idx)

        t0 = time.time()
        print(f"\\n=== Training {algo} ===")

        if algo == "TD":
            agent = TD3Agent(
                state_dim=train_env.state_dim,
                action_dim=train_env.action_dim,
                device=device,
                cfg=TD3Config(),
            )
            train_summary, steps = train_offpolicy(
                algo_name=algo,
                agent=agent,
                env=train_env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )

        elif algo == "DDPG":
            agent = DDPGAgent(
                state_dim=train_env.state_dim,
                action_dim=train_env.action_dim,
                device=device,
                cfg=DDPGConfig(),
            )
            train_summary, steps = train_offpolicy(
                algo_name=algo,
                agent=agent,
                env=train_env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )

        elif algo == "HAC":
            agent = HACAgent(
                state_dim=train_env.state_dim,
                action_dim=train_env.action_dim,
                effect_dim=train_env.effect_dim,
                device=device,
                cfg=HACConfig(),
            )
            train_summary, steps = train_offpolicy(
                algo_name=algo,
                agent=agent,
                env=train_env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )

        elif algo == "A2C":
            agent = A2CAgent(
                state_dim=train_env.state_dim,
                action_dim=train_env.action_dim,
                device=device,
                cfg=A2CConfig(),
            )
            agent.terminal_bonus_coef = float(args.terminal_bonus_coef)
            agent.terminal_bonus_spread = str(args.terminal_bonus_spread)
            train_summary = train_a2c(agent=agent, env=train_env, train_episodes=args.train_episodes)
            steps = 0

        elif algo == "PPO":
            agent = PPOAgent(
                state_dim=train_env.state_dim,
                action_dim=train_env.action_dim,
                device=device,
                cfg=PPOConfig(),
            )
            agent.terminal_bonus_coef = float(args.terminal_bonus_coef)
            agent.terminal_bonus_spread = str(args.terminal_bonus_spread)
            train_summary = train_ppo(agent=agent, env=train_env, train_episodes=args.train_episodes)
            steps = 0

        elif algo == "CQL":
            agent = CQLAgent(
                state_dim=train_env.state_dim,
                action_dim=train_env.action_dim,
                device=device,
                cfg=CQLConfig(),
            )
            train_summary, steps = train_offpolicy(
                algo_name=algo,
                agent=agent,
                env=train_env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )

        elif algo == "IQL":
            agent = IQLAgent(
                state_dim=train_env.state_dim,
                action_dim=train_env.action_dim,
                device=device,
                cfg=IQLConfig(),
            )
            train_summary, steps = train_offpolicy(
                algo_name=algo,
                agent=agent,
                env=train_env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )

        elif algo == "DQN":
            agent = DQNAgent(
                state_dim=train_env.state_dim,
                action_dim=train_env.action_dim,
                device=device,
                cfg=DQNConfig(),
            )
            train_summary, steps = train_dqn(
                agent=agent,
                env=train_env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )

        elif algo == "RAINBOW":
            agent = RainbowAgent(
                state_dim=train_env.state_dim,
                action_dim=train_env.action_dim,
                device=device,
            )
            train_summary, steps = train_dqn(
                agent=agent,
                env=train_env,
                device=device,
                train_episodes=args.train_episodes,
                batch_size=args.batch_size,
                warmup_steps=args.warmup_steps,
                updates_per_step=args.updates_per_step,
                buffer_capacity=args.buffer_capacity,
                terminal_bonus_coef=args.terminal_bonus_coef,
                terminal_bonus_spread=args.terminal_bonus_spread,
            )

        elif algo in {"SASREC", "ONEREC", "GRU4REC", "TIGER"}:
            seq_cfg = SeqBCConfig(
                max_seq_len=args.max_pages,
                epochs=args.seq_epochs,
                collect_episodes=args.seq_collect_episodes,
            )
            agent = SequenceBCAgent(
                algo=algo,
                state_dim=train_env.state_dim,
                action_dim=train_env.action_dim,
                device=device,
                cfg=seq_cfg,
            )
            train_summary = agent.fit(
                env=train_env,
                collect_episodes=args.seq_collect_episodes,
                epochs=args.seq_epochs,
            )
            steps = 0

        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

        eval_summary = evaluate_agent(agent=agent, env=eval_env, episodes=args.eval_episodes, algo_name=algo)
        elapsed = time.time() - t0

        merged = {
            **train_summary,
            **eval_summary,
            "train_steps": int(steps),
            "elapsed_sec": float(elapsed),
        }
        all_results["results"][algo] = merged

        print(
            f"{algo}: len={merged['mean_session_length']:.3f}, "
            f"ret={merged['mean_return']:.3f}, "
            f"ltv={merged['ltv_score']:.3f}, "
            f"terminal={merged.get('mean_terminal_score', 0.0):.3f}, "
            f"retention5={merged['retention_5plus']:.3f}"
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # also export compact csv for quick viewing
    csv_path = output_path.with_suffix(".csv")
    header = [
        "algo",
        "mean_session_length",
        "mean_return",
        "ltv_score",
        "retention_5plus",
        "mean_like_rate",
        "mean_genre_repetition",
        "mean_satisfaction",
        "mean_terminal_score",
        "train_mean_return",
        "train_mean_length",
        "elapsed_sec",
    ]
    lines = [",".join(header)]
    for algo, m in all_results["results"].items():
        row = [
            algo,
            f"{m['mean_session_length']:.6f}",
            f"{m['mean_return']:.6f}",
            f"{m['ltv_score']:.6f}",
            f"{m['retention_5plus']:.6f}",
            f"{m['mean_like_rate']:.6f}",
            f"{m['mean_genre_repetition']:.6f}",
            f"{m['mean_satisfaction']:.6f}",
            f"{m.get('mean_terminal_score', 0.0):.6f}",
            f"{m.get('train_mean_return', m.get('collect_mean_return', 0.0)):.6f}",
            f"{m.get('train_mean_length', m.get('collect_mean_length', 0.0)):.6f}",
            f"{m['elapsed_sec']:.6f}",
        ]
        lines.append(",".join(row))
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\\nSaved metrics to {output_path}")
    print(f"Saved csv to {csv_path}")


if __name__ == "__main__":
    main()
