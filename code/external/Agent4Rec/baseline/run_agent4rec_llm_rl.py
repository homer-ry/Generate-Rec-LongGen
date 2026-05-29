from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from baseline.agent4rec_llm_env import Agent4RecLLMEnv
from baseline.agents.a2c import A2CAgent, A2CConfig
from baseline.agents.cql import CQLAgent, CQLConfig
from baseline.agents.common import set_seed
from baseline.agents.ddpg import DDPGAgent, DDPGConfig
from baseline.agents.dqn import DQNAgent, DQNConfig, RainbowAgent
from baseline.agents.hac import HACAgent, HACConfig
from baseline.agents.iql import IQLAgent, IQLConfig
from baseline.agents.ppo import PPOAgent, PPOConfig
from baseline.agents.td3 import TD3Agent, TD3Config
from baseline.run_kuaisim_baselines import evaluate_agent, train_a2c, train_dqn, train_offpolicy, train_ppo


def make_env(args, seed_offset: int = 0, phase: str = "train") -> Agent4RecLLMEnv:
    return Agent4RecLLMEnv(
        root_dir=args.root_dir,
        dataset=args.dataset,
        modeltype=args.modeltype,
        simulation_name=f"{args.simulation_name}_{phase}_{seed_offset}",
        llm_model=args.llm_model,
        llm_api_style=args.llm_api_style,
        use_wandb=args.use_wandb,
        max_pages=args.max_pages,
        slate_size=args.slate_size,
        seed=args.seed + seed_offset,
        user_limit=args.user_limit,
        exit_threshold_scale=args.exit_threshold_scale,
        repetition_penalty_scale=args.repetition_penalty_scale,
        repeat_aversion_scale=args.repeat_aversion_scale,
        fatigue_scale=args.fatigue_scale,
        terminal_score_max=args.terminal_score_max,
        terminal_from_interview=args.terminal_from_interview,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RL baselines on Agent4Rec LLM-avatar whole-session environment"
    )
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="ml-1m")
    parser.add_argument("--modeltype", type=str, default="SASRec")
    parser.add_argument("--simulation_name", type=str, default="rl_llm_env")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_wandb", action="store_true")

    parser.add_argument("--llm_model", type=str, default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument(
        "--llm_api_style",
        type=str,
        default=os.getenv("LLM_API_STYLE", "responses"),
        choices=["chat_completions", "responses"],
    )
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--openai_api_base", type=str, default=os.getenv("OPENAI_API_BASE", ""))

    parser.add_argument("--train_episodes", type=int, default=20)
    parser.add_argument("--eval_episodes", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=40)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--buffer_capacity", type=int, default=50000)

    parser.add_argument("--max_pages", type=int, default=8)
    parser.add_argument("--slate_size", type=int, default=1)
    parser.add_argument("--user_limit", type=int, default=50)
    parser.add_argument("--exit_threshold_scale", type=float, default=1.35)
    parser.add_argument("--repetition_penalty_scale", type=float, default=1.20)
    parser.add_argument("--repeat_aversion_scale", type=float, default=1.50)
    parser.add_argument("--fatigue_scale", type=float, default=1.80)
    parser.add_argument("--terminal_score_max", type=float, default=10.0)
    parser.add_argument("--terminal_from_interview", action="store_true")

    parser.add_argument("--algos", type=str, default="TD,DDPG,HAC,A2C")
    parser.add_argument("--terminal_bonus_coef", type=float, default=0.10)
    parser.add_argument("--terminal_bonus_spread", type=str, default="last", choices=["last", "all"])
    parser.add_argument("--output", type=str, default="baseline/results/agent4rec_llm_rl_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.openai_api_base:
        os.environ["OPENAI_API_BASE"] = args.openai_api_base
    set_seed(args.seed)
    device = torch.device(args.device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    algos = [a.strip().upper() for a in args.algos.split(",") if a.strip()]
    all_results = {
        "config": {
            "dataset": args.dataset,
            "modeltype": args.modeltype,
            "simulation_name": args.simulation_name,
            "llm_model": args.llm_model,
            "llm_api_style": args.llm_api_style,
            "openai_api_base": os.getenv("OPENAI_API_BASE", ""),
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
            "terminal_from_interview": args.terminal_from_interview,
            "terminal_bonus_coef": args.terminal_bonus_coef,
            "terminal_bonus_spread": args.terminal_bonus_spread,
            "seed": args.seed,
        },
        "results": {},
    }

    for idx, algo in enumerate(algos):
        train_env = make_env(args, seed_offset=10 + idx, phase="train")
        eval_env = make_env(args, seed_offset=100 + idx, phase="eval")

        t0 = time.time()
        print(f"\n=== Training {algo} on Agent4Rec LLM env ===")

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

        else:
            raise ValueError(f"Unsupported algorithm: {algo}")

        if args.eval_episodes > 0:
            eval_summary = evaluate_agent(agent=agent, env=eval_env, episodes=args.eval_episodes, algo_name=algo)
        else:
            eval_summary = {
                "algo": algo,
                "eval_episodes": 0,
                "mean_return": 0.0,
                "mean_session_length": 0.0,
                "mean_like_rate": 0.0,
                "mean_genre_repetition": 0.0,
                "mean_satisfaction": 0.0,
                "mean_terminal_score": 0.0,
                "retention_5plus": 0.0,
                "ltv_score": 0.0,
            }

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
            f"terminal={merged.get('mean_terminal_score', 0.0):.3f}"
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

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
            f"{m.get('train_mean_return', 0.0):.6f}",
            f"{m.get('train_mean_length', 0.0):.6f}",
            f"{m['elapsed_sec']:.6f}",
        ]
        lines.append(",".join(row))
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSaved metrics to {output_path}")
    print(f"Saved csv to {csv_path}")


if __name__ == "__main__":
    main()
