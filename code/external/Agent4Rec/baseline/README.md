# Baseline

This folder contains two different things:

- `baseline/run_kuaisim_baselines.py` and `baseline/session_env.py`: legacy Agent4Rec local simulator code kept for ablation/history.
- `scripts/run_kuaisim_wholesession_baselines.py` and `scripts/run_kuaisim_tiger_scope.py`: the canonical KuaiSim clean evaluator entrypoints.

If you want paper-facing KuaiSim numbers, do not use the legacy local simulator. Use the nested official repo only:

- `E:\project\Recommend\KuaiSim-main\KuaiSim-main\code`

The clean evaluator is shared by:

- `scripts/kuaisim_clean_common.py`

Legacy local baselines still available in this folder:

- `TD` (implemented as TD3, aligned with KuaiSim `TD3.py`)
- `DDPG`
- `HAC`
- `A2C`
- `PPO`
- `CQL`
- `IQL`
- `DQN`
- `RAINBOW` (rainbow-lite: double+dueling DQN with richer action discretization)

## Canonical KuaiSim Evaluation

Canonical evaluation root:

- `E:\project\Recommend\KuaiSim-main\KuaiSim-main\code`

Canonical scripts:

- `scripts/run_kuaisim_wholesession_baselines.py`
- `scripts/run_kuaisim_tiger_scope.py`
- `scripts/kuaisim_clean_common.py`

Canonical outputs:

- `baseline/results/kuaisim_clean_*.json`
- `baseline/results/kuaisim_clean_*.csv`

Example: Random/TD/DDPG/A2C/HAC

```bash
python scripts/run_kuaisim_wholesession_baselines.py \
  --algos Random TD DDPG A2C HAC \
  --kuaisim_root E:\project\Recommend\KuaiSim-main\KuaiSim-main \
  --slate_size 1 \
  --episode_batch_size 32 \
  --num_episodes 100 \
  --output_prefix baseline/results/kuaisim_clean_rl_seed11_eval100
```

Example: TIGER / TIGER+SCOPE

```bash
python scripts/run_kuaisim_tiger_scope.py \
  --algos TIGER TIGER_SCOPE \
  --kuaisim_root E:\project\Recommend\KuaiSim-main\KuaiSim-main \
  --slate_size 1 \
  --episode_batch_size 32 \
  --num_episodes 100 \
  --scope_candidate_pool 50 \
  --output_prefix baseline/results/kuaisim_clean_tiger_scope_seed11_eval100
```

## Legacy Local Simulator Files

- `session_env.py`: long-session simulator built from `datasets/ml-1m` data
- `agents/ddpg.py`, `agents/td3.py`, `agents/hac.py`, `agents/a2c.py`
- `agents/ppo.py`, `agents/cql.py`, `agents/iql.py`, `agents/dqn.py`
- `run_kuaisim_baselines.py`: legacy train/eval on local `LongSessionRecEnv`
- `run_agent4rec_llm_rl.py`: train RL baselines on Agent4Rec LLM-avatar environment (whole session)
- `kuaisim_source/`: copied original KuaiSim algorithm files (`A2C.py`, `DDPG.py`, `HAC.py`, `TD3.py`) for traceability

## Session Design

To evaluate long-term value, the simulator includes:

- page-wise interaction (`max_pages`)
- explicit dissatisfaction accumulation
- **higher exit threshold** (`exit_threshold_scale`) than earlier default behavior
- repetition penalty (genre-level repetition) to model "content gets repetitive"
- optional terminal score (`0-10`) at session end, which can be fed back into training

A user exits when dissatisfaction exceeds threshold or max pages reached.

## Legacy Local Run

```bash
python baseline/run_kuaisim_baselines.py \
  --train_episodes 180 \
  --eval_episodes 160 \
  --max_pages 20 \
  --user_limit 300 \
  --exit_threshold_scale 1.35 \
  --repetition_penalty_scale 1.20 \
  --terminal_bonus_coef 0.10 \
  --terminal_bonus_spread last
```

Legacy output:

- `baseline/results/kuaisim_baselines_metrics.json`
- `baseline/results/kuaisim_baselines_metrics.csv`

## LLM Avatar Training

```bash
python baseline/run_agent4rec_llm_rl.py \
  --algos DDPG \
  --train_episodes 10 \
  --eval_episodes 4 \
  --max_pages 6 \
  --slate_size 1 \
  --user_limit 20 \
  --terminal_bonus_coef 0.10 \
  --terminal_bonus_spread last \
  --llm_api_style responses \
  --openai_api_base <OPENAI_COMPATIBLE_BASE_URL> \
  --openai_api_key sk-xxx
```
