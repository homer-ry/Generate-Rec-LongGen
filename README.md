# SAGERec: Support-Aware Adaptive-Trust Generative Recommendation

面向生成式推荐长期价值后训练。核心问题：TIGER 在 SID token 级生成 item，但长期收益只在 page/slate/session 级可观测。SAGERec 通过 critic 差分将 page 级价值拆解为 item 级和 SID token 级 credit，再用 support-aware / uncertainty-aware 的 conservative GRPO 更新 actor。

## 主流程

```text
┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐
│ 1.下载原始数据   │▶│ 2.预处理       │▶│ 3.训练 URM     │▶│ 4.构建 SID     │▶│ 5.训练TIGER   │
│ KuaiRand-Pure  │  │ notebook→     │  │ 用户反馈模型    │  │ 物品→token    │  │ 基础策略      │
│ (3个log+特征)  │  │ log_session   │  │ KRMBUserResp  │  │ 32/64码本     │  │               │
│                │  │ + fillna      │  │               │  │               │  │               │
└────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘  └──────┬────────┘
                                                                               │
                     ┌──────────────────────────────────────────────────────────┘
                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                     6. SAGERec 闭环训练（迭代 3 轮）                        │
│                                                                           │
│  ┌──────────┐    ┌──────────┐    ┌────────────┐    ┌──────────┐          │
│  │ Rollout  │───▶│ Critic   │───▶│ Attribution│───▶│  Group   │          │
│  │ 收集轨迹  │    │ 训练Q网络 │    │ 构造归因链  │    │ 构造候选组 │          │
│  └──────────┘    └──────────┘    └────────────┘    └────┬─────┘          │
│                                                          │                │
│                    ┌─────────────────────────────────────┘                │
│                    ▼                                                      │
│              ┌──────────┐    ┌──────────┐                                 │
│              │  Actor   │───▶│   EMA    │──▶ 下一轮迭代                     │
│              │ GRPO更新  │    │ 同步策略  │                                 │
│              └──────────┘    └──────────┘                                 │
└───────────────────────────────────────────────────────────────────────────┘
```

**Step 2 预处理**（`code/preprocess/KuaiRandDataset.ipynb`）将原始 KuaiRand 日志转换为下游脚本所需的兼容文件：
- 合并 3 个原始 log → `log_session_4_08_to_5_08_Pure.csv`（20-core过滤 + session构建 + 排序）
- fillna 处理 → `user_features_Pure_fillna.csv`、`video_features_basic_Pure_fillna.csv`

**每轮迭代内部**：`rollout → train critic → build advantage chain → build groups → GRPO update actor → EMA sync`

## 目录结构

```
code/
├── preprocess/               # KuaiRandDataset.ipynb：原始数据→兼容文件预处理
├── reader/                  # 数据读取 (KRMBSeqReader)
├── model/
│   ├── simulator/           # 用户响应模型 URM (KRMBUserResponse)
│   ├── agent/               # RL 智能体 (A2C/DDPG/TD3)
│   └── *.py                 # TIGER/SASRec/GRU4Rec/BERT4Rec 等基线
├── tiger_page_sid_rl/       # ★ SAGERec 主线 (closed loop/critic/归因)
├── tiger_hcaa/              # 分层信用分配研究
├── tiger_hcla_rl/           # 长期 Actor-Critic 研究
├── train_*.py / eval_*.py   # 各模型训练/评估入口
└── run_*_strict_eval*.sh    # 严格评估套件
docs/                        # 消融实验报告
```

## 快速开始

### 1. 下载原始数据

从 [Zenodo](https://zenodo.org/records/10439422) 下载 KuaiRand-Pure：

```bash
mkdir -p dataset/kuairand && cd dataset/kuairand
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar -xzvf KuaiRand-Pure.tar.gz
mv KuaiRand-Pure kuairand-Pure   # 统一为小写，与代码内路径一致
```

原始文件包含：
- `log_standard_4_08_to_4_21_pure.csv`、`log_standard_4_22_to_5_08_pure.csv`、`log_random_4_22_to_5_08_pure.csv`
- `user_features_pure.csv`、`video_features_basic_pure.csv`、`video_features_statistic_pure.csv`

### 2. 预处理（生成兼容文件）

**方式 A**：运行 `code/preprocess/KuaiRandDataset.ipynb` notebook（完整流程：合并日志 → 20-core过滤 → session构建 → 排序 → fillna），产物为：
- `log_session_4_08_to_5_08_Pure.csv`
- `user_features_Pure_fillna.csv`
- `video_features_basic_Pure_fillna.csv`

```bash
# 交互式运行（推荐，可逐步检查中间结果）
jupyter notebook code/preprocess/KuaiRandDataset.ipynb

# 或命令行一键执行（需确保 notebook 内路径与本地一致）
jupyter nbconvert --to notebook --execute code/preprocess/KuaiRandDataset.ipynb
```

> notebook 内数据路径默认为 `dataset/kuairand/kuairand-Pure/data/`，如本地路径不同需修改 notebook 中对应变量。

**方式 B**：最小可用处理（适合跑通代码，不含 K-core过滤）：

```bash
python - <<'PY'
from pathlib import Path; import pandas as pd
root = Path("dataset/kuairand/kuairand-Pure/data")
logs = [root / f"log_standard_4_{s}_to_{e}_pure.csv" for s,e in [("08","4_21"),("22","5_08")]]
pd.concat([pd.read_csv(p) for p in logs], ignore_index=True) \
  .to_csv(root / "log_session_4_08_to_5_08_Pure.csv", index=False)
pd.read_csv(root / "user_features_pure.csv").fillna("unknown") \
  .to_csv(root / "user_features_Pure_fillna.csv", index=False)
pd.read_csv(root / "video_features_basic_pure.csv").fillna("unknown") \
  .to_csv(root / "video_features_basic_Pure_fillna.csv", index=False)
PY

# 复制到 code/dataset/（部分脚本读取此路径）
mkdir -p code/dataset/kuairand/kuairand-Pure/data
cp dataset/kuairand/kuairand-Pure/data/*_Pure*.csv code/dataset/kuairand/kuairand-Pure/data/
```

### 3. 训练 URM（用户反馈模拟器）

```bash
cd code
python train_multibehavior.py \
  --reader KRMBSeqReader --model KRMBUserResponse \
  --train_file dataset/kuairand/kuairand-Pure/data/log_session_4_08_to_5_08_Pure.csv \
  --user_meta_file dataset/kuairand/kuairand-Pure/data/user_features_Pure_fillna.csv \
  --item_meta_file dataset/kuairand/kuairand-Pure/data/video_features_basic_Pure_fillna.csv \
  --cuda 0 --lr 0.0001 --l2_coef 0 --epoch 10 \
  --model_path output/Kuairand_Pure/env/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model
```

产物：`output/Kuairand_Pure/env/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.checkpoint`

### 4. 构建 SID 映射 + 训练 TIGER

```bash
# SID 映射（已提供 32_mask，也可重新构建）
bash code/build_kuairand_sid.sh

# 训练 TIGER 基础策略
bash code/train_TIGER_krpure.sh
```

### 5. 运行 SAGERec 闭环

```bash
export TIGER_CKPT=output/KuaiRand_Pure/env/tiger_sid_krpure_mini.pth
export UIRM_LOG_PATH=code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.windows.log
export SID_MAPPING_PATH=code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv

bash code/tiger_page_sid_rl/run_sagerec_adaptive_grpo.sh
```

## 关键数据路径

```bash
# 输入数据
LOG_CSV=code/dataset/kuairand/kuairand-Pure/data/log_session_4_08_to_5_08_Pure.csv
USER_FEAT=code/dataset/kuairand/kuairand-Pure/data/user_features_Pure_fillna.csv
ITEM_FEAT=code/dataset/kuairand/kuairand-Pure/data/video_features_basic_Pure_fillna.csv
SID_MAP=code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv

# 中间产物（需本地运行生成）
URM_CKPT=code/output/Kuairand_Pure/env/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.checkpoint
UIRM_LOG=code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.windows.log
TIGER_CKPT=output/KuaiRand_Pure/env/tiger_sid_krpure_mini.pth
```

## Benchmark 与基线

| 类别 | 方法 | 入口脚本 |
|------|------|----------|
| **生成式推荐主线** | TIGER base、SAGERec (HCA-LCB-GRPO) | `run_sagerec_adaptive_grpo.sh` |
| TIGER post-training | DPO-style、SPRec、ReRe-GRPO | `run_tiger_base_posttrain_suite_*.sh` |
| 序列推荐 | SASRec、GRU4Rec、BERT4Rec、P5-style | `run_strict_seqrec_envmap_eval3_*.sh` |
| 在线 RL / DT | A2C、DDPG、TD3、HAC、DT | `train_*_krpure_wholesession.sh` |
| OneRec post-training | S-DPO、SPRec、ReRe-GRPO、LTV-GRPO | `run_posttrain_onerec_baselines_*.sh` |
| Slate rerank | SlateQ-like TIGER | `run_strict_slateq_like_eval3_*.sh` |

> 命名边界：P5-style ≠ OpenP5 复现，SlateQ-like ≠ canonical SlateQ。详见 `code/baseline_notes/`。

### 基线运行命令

**序列推荐基线**：

```bash
bash code/train_sasrec_baseline.sh          # SASRec
bash code/eval_sasrec_env.sh                # SASRec 环境评估
bash code/train_gru4rec_baseline.sh         # GRU4Rec
bash code/eval_gru4rec_env.sh               # GRU4Rec 环境评估
bash code/run_strict_bert4rec_eval3_20260512.sh    # BERT4Rec (严格评估)
bash code/run_strict_p5_style_eval3_20260512.sh    # P5-style (严格评估)
bash code/run_strict_seqrec_envmap_eval3_20260512.sh  # SASRec/GRU4Rec/BERT4Rec/P5 统一套件
```

**在线 RL / Decision Transformer**：

```bash
cd code
bash train_A2C_krpure_wholesession.sh       # A2C
bash train_ddpg_krpure_wholesession.sh      # DDPG
bash train_TD3_krpure_wholesession.sh       # TD3
bash train_HAC_krpure_wholesession.sh       # HAC
bash train_dt_log_session.sh                # DT (日志训练)
bash train_dt_env_click.sh                  # DT (环境训练)
bash eval_dt_policy_env.sh                  # DT 环境评估
bash eval_actor_critic.sh                   # RL 智能体评估
```

**TIGER post-training**：

```bash
bash code/run_tiger_base_posttrain_suite_strict_eval3_20260513.sh  # DPO/SPRec/ReRe-GRPO/SAGERec 统一套件
METHODS_STR="hca_lcb_grpo" bash code/run_tiger_base_posttrain_suite_strict_eval3_20260513.sh  # 只跑 SAGERec
SMOKE=1 bash code/run_tiger_base_posttrain_suite_strict_eval3_20260513.sh  # 烟测
bash code/train_tiger_phase2_blend.sh       # Phase2 blend
bash code/run_strict_slateq_like_eval3_20260512.sh  # SlateQ-like
```

**OneRec post-training**：

```bash
bash code/train_onerec_value.sh             # OneRec 基础训练
bash code/train_s_dpo.sh                    # S-DPO
bash code/train_sprec.sh                    # SPRec
bash code/train_rere_grpo.sh                # ReRe-style GRPO
bash code/run_posttrain_onerec_baselines_strict_eval3_20260512.sh   # OneRec 统一套件
SMOKE=1 bash code/run_posttrain_onerec_baselines_strict_eval3_20260512.sh  # 烟测
bash code/run_onerec_ltv_grpo_and_dpo_strict_eval3_20260512.sh      # LTV-GRPO + Plain DPO
bash code/run_onerec_fixed_posttrain_strict_eval3_20260512.sh       # Fixed posttrain
```

## 核心实现

| 模块 | 文件 | 功能 |
|------|------|------|
| URM | `model/simulator/KRMBUserResponse.py` | Transformer 编码用户历史 + 多反馈预测（7 类行为） |
| TIGER | `train_TIGER_krpure.py` | T5 seq2seq：历史 SID → 目标 SID |
| Closed loop | `tiger_page_sid_rl/run_page_sid_closed_loop.py` | 主 orchestrator：串联 rollout→critic→归因→group→actor |
| Critic | `tiger_page_sid_rl/train_page_critic.py` | Page/Item/SID 多粒度 Q 值估计 + ensemble |
| 归因链 | `tiger_page_sid_rl/build_sid_advantage_chain.py` | Q(full)-Q(without item) 差分 → item_adv；prefix Q 差分 → sid_adv |
| Group 构造 | `build_tiger_hca_grpo_groups.py` | Behavior+beam candidate，support gap + uncertainty 悲观估计 |
| Actor 更新 | `train_tiger_hca_grpo_actor.py` | HCA credit + clipped GRPO + adaptive KL/clip |

**SAGERec 三层 credit 归因**：

```text
page advantage   → 决定整条候选提升/打压
item advantage   → 决定 slate 内哪个 item 更关键   (Q(full) - Q(without i))
SID advantage    → 决定 item 的哪些 token 更关键    (Q(prefix k) - Q(prefix k-1))
```

**自适应信任机制**：support gap ↑ 或 critic uncertainty ↑ → KL 收紧、clip 缩小 → 更新更保守。

## 评估流程

所有评估均为**闭环环境交互评估**（KuaiSim 模拟器），非静态测试集。

### 评估环境

```text
TIGER 策略 → beam search 生成 slate=6 个物品
           → KuaiSim(URM) 返回 7 维用户反馈
           → 计算即时 reward → 重复直到用户退出或达到 20 步
```

### 评估协议

| 参数 | 值 |
|------|-----|
| slate_size | 6 |
| beam_width | 16 |
| eval_episodes | 200 |
| max_steps_per_episode | 20 |
| item_correlation | 0.2 |
| phase2_blend_scale | 0.20 |
| random_topk_sample | 10 |
| seed | 2026 |

### 评估时机

每轮迭代有 3 个评估节点，对应消融表的 4 列：

| 消融表列 | 含义 | 来源 |
|----------|------|------|
| Before | 本轮训练前的 rollout 策略得分 | `before_eval.log` |
| Rollout | 收集轨迹时的策略得分 | `rollout.log` |
| After Rollout | EMA 同步后的 rollout 策略得分 | `after_eval_rollout.log` |
| After Learner | GRPO 更新后的 learner 策略得分 | `after_eval_learner.log` |

### 评估指标

| 指标 | 含义 |
|------|------|
| **total_reward** | episode 累计奖励（主指标） |
| depth | 平均会话深度（步数） |
| avg_step_reward | 平均每步奖励 |
| coverage | 物品覆盖率 |
| ILD | 列表内多样性 |
| click / long_view | 点击率 / 长观看率 |
| is_like / is_comment / is_forward / is_follow / is_hate | 其他行为率 |

### 评估命令

```bash
# 单次策略评估
python code/eval_tiger_env.py \
  --tiger_ckpt $TIGER_CKPT \
  --sid_mapping_path code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv \
  --uirm_log_path $UIRM_LOG_PATH \
  --num_episodes 200 --beam_width 16 --slate_size 6 --seed 2026

# 统一评估套件（所有基线 + 消融）
bash code/run_tiger_base_posttrain_suite_strict_eval3_20260513.sh
SMOKE=1 bash code/run_tiger_base_posttrain_suite_strict_eval3_20260513.sh  # 烟测
```

## 消融实验结果

完整结果见 `docs/ablation_summary.md`。

| 方法 | Iter | After Learner | 结论 |
|------|:---:|--------------:|------|
| TIGER base | 0 | 4.5417 | 基准 (depth=12.26, click=37.06%, long_view=28.93%) |
| GRPO-only + EMA | 3 | 4.2017 | 多轮漂移 |
| GRPO + DPO-seq + EMA | 3 | 4.4142 | DPO 稳定学习器但偏弱 |
| DPO-only support-aware | 3 | 3.8825 | 纯 pairwise 不稳定 |
| **SAGERec adaptive-trust GRPO** | **2** | **4.7383** | **最佳 (+4.3%)** |
| SAGERec adaptive-trust GRPO | 3 | 4.3483 | iter3 过更新，建议 early stopping |



# dev
```
cd /share/rongyu03/onemodel/Generate-Rec-LongGen

```