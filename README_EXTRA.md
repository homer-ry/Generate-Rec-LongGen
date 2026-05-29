# SAGERec 全流程输入输出详解

每步标注 **→下一步输入**，确保上下游对齐。

---

## Step 1: 下载原始数据

### 输入
无（从 [Zenodo](https://zenodo.org/records/10439422) 下载）

### 输出 → Step 2 输入

```text
dataset/kuairand/kuairand-Pure/data/
├── log_standard_4_08_to_4_21_pure.csv    (交互日志，4月8日~4月21日)
├── log_standard_4_22_to_5_08_pure.csv    (交互日志，4月22日~5月8日)
├── log_random_4_22_to_5_08_pure.csv      (交互日志，随机曝光，4月22日~5月8日)
├── user_features_pure.csv                (用户特征)
├── video_features_basic_pure.csv         (物品基础特征)
└── video_features_statistic_pure.csv     (物品统计特征)
```

### 交互日志 demo

```
user_id,video_id,date,hourmin,time_ms,is_click,is_like,is_follow,is_comment,is_forward,is_hate,long_view,play_time_ms,duration_ms,profile_stay_time,comment_stay_time,is_profile_enter,is_rand,tab
3,1071,20220408,1600,1649408076860,1,0,0,0,0,0,1,175045,87433,0,0,0,0,1
```

解读：用户 3 在 2022-04-08 16:00 观看视频 1071，点击了(is_click=1)、长观看(long_view=1)，播放 175s，视频时长 87s。

---

## Step 2: 预处理 (KuaiRandDataset.ipynb)

### 输入 ← Step 1 输出
6 个原始 CSV 文件

### 处理逻辑

| 子步骤 | 操作 | 说明 |
|--------|------|------|
| ① 合并 | `pd.concat([df_1, df_2, df_3])` | 3 个 log → 1 个 DataFrame |
| ② 20-core | `run_multicore(n_core=20)` | 删除交互 <20 条的用户/物品 |
| ③ session | 按 user_id + date 划分 | 添加 `position` / `session` 列 |
| ④ 排序 | `sort_values(['user_id','time_ms'])` | 保证时间因果序 |
| ⑤ date | `time_ms → YYYYMMDD` | 毫秒时间戳 → 整数日期 |
| ⑥ fillna | user: `onehot_feat*` → `-1`；video: `tag`/`music_type` → `0` | 数值型填充 |

### 输出 → Step 3 + Step 5 输入

| 文件 | 行数 | 列数 | demo |
|------|------|------|------|
| `log_session_4_08_to_5_08_Pure.csv` | 1,436,609 | 19 | 见下方 |
| `user_features_Pure_fillna.csv` | 27,285 | 32 | 见下方 |
| `video_features_basic_Pure_fillna.csv` | 7,583 | 12 | 见下方 |

**log_session demo**：
```
user_id=0, video_id=1527, date=20220411, time_ms=1649675512388,
is_click=0, is_like=0, is_follow=0, is_comment=0, is_forward=0, is_hate=0, long_view=0,
play_time_ms=1385, duration_ms=209900, is_rand=0, tab=1
```

**user_features demo** (user_id=0)：
```
user_id=0, user_active_degree=full_active, is_live_streamer=1, is_video_author=1,
follow_user_num=514, fans_user_num=150, register_days=799,
onehot_feat0=1, onehot_feat1=29, ..., onehot_feat11=1.0
```

**video_features demo** (video_id=0)：
```
video_id=0, video_type=NORMAL, upload_type=LongImport,
video_duration=87433, server_width=720, server_height=1280,
music_type=9, tag=39
```

---

## Step 3: URM 训练数据构造 (KRMBSeqReader)

### 输入 ← Step 2 输出

| 文件 | 作用 |
|------|------|
| `log_session_4_08_to_5_08_Pure.csv` | 交互日志（1,436,609 行，19 列） |
| `user_features_Pure_fillna.csv` | 用户特征（27,285 行，32 列） |
| `video_features_basic_Pure_fillna.csv` | 物品特征（7,583 行，12 列） |

### 处理逻辑

**3a. 加载与词汇表构建**：

```text
log_data = pd.read_csv(log_session)          → DataFrame (1,436,609 × 19)
user_meta = pd.read_csv(user_feat)           → dict {uid: {feature: value}}
item_meta = pd.read_csv(item_feat)           → dict {iid: {feature: value}}

users = unique user_ids                       → 27,077 个用户
items = unique video_ids                      → 7,551 个物品

user_id_vocab = {0:1, 1:2, ...}               → 编号 1~27,077
item_id_vocab = {0:1, 1:2, ...}                → 编号 1~7,551

user_history = {uid: [row_id_list]}            → 每用户所有交互行号
```

**3b. 特征编码（one-hot / multi-hot）**：

```text
selected_user_features (13个):
  user_active_degree     → one-hot, dim=9    例: full_active → [1,0,0,...0]
  is_live_streamer       → one-hot, dim=2    例: 1 → [0,1]
  is_video_author        → one-hot, dim=2    例: 1 → [1,0]
  follow_user_num_range  → one-hot, dim=8    例: 500+ → [0,0,1,...0]
  fans_user_num_range    → one-hot, dim=9
  friend_user_num_range  → one-hot, dim=7
  register_days_range    → one-hot, dim=8
  onehot_feat0           → one-hot, dim=2    例: 1 → [1,0]
  onehot_feat1           → one-hot, dim=7    例: 29 → [0,0,1,...0]
  onehot_feat6           → one-hot, dim=3
  onehot_feat9           → one-hot, dim=7
  onehot_feat10          → one-hot, dim=5
  onehot_feat11          → one-hot, dim=5

  用户特征总维度: 9+2+2+8+9+7+8+2+7+3+7+5+5 = 70

selected_item_features (4个):
  video_type             → one-hot, dim=3    例: NORMAL → [1,0,0]
  music_type             → one-hot, dim=6    例: 9 → [1,0,0,...0]
  upload_type            → one-hot, dim=14   例: LongImport → [1,0,...0]
  tag                    → multi-hot, dim=47  例: tag=39 → 第39位=1,其余=0
                                             例: tag="2,39" → 第2位+第39位=1

  物品特征总维度: 3+6+14+47 = 70
```

**3c. 序列 Holdout 切分**：

```text
对每个用户按时间顺序切分:
  val_holdout=5, test_holdout=5

  例: 用户 0 有 10 条交互
  train = rows[0:0]       (0条,不够60% → 被跳过)
  例: 用户 3 有 100 条交互
  train = rows[0:90]      (90条)
  val    = rows[90:95]     (5条)
  test   = rows[95:100]    (5条)

  过滤条件: n_train ≥ 60% × total

最终:
  train: 1,144,773 条
  val:    89,765 条
  test:   89,765 条
```

**3d. 训练样本构造 (`__getitem__`)**：

每个样本 = **一条交互记录** + **该记录之前的用户历史**：

```text
Sample idx=6 (用户3的第7次交互):

  # 目标信息
  user_id:          3                   ← vocab 编码后的用户 ID
  item_id:          16                  ← vocab 编码后的物品 ID
  is_click:         0                   ← 7类反馈标签 (0/1)
  long_view:        0
  is_like:          0
  is_comment:       0
  is_forward:       0
  is_follow:        0
  is_hate:          0

  # 用户特征 (one-hot 向量)
  uf_user_active_degree:  (9,)    = [1,0,0,0,0,0,0,0,0]   ← full_active
  uf_is_live_streamer:    (2,)    = [0,1]                   ← 是直播者
  uf_onehot_feat0:        (2,)    = [1,0]                   ← 值=1
  ... (共 13 个向量, 总 dim=70)

  # 物品特征 (one-hot + multi-hot)
  if_video_type:     (3,)    = [1,0,0]                     ← NORMAL
  if_music_type:     (6,)    = [1,0,0,0,0,0]               ← music_type=9
  if_upload_type:    (14,)   = [...]                        ← upload_type
  if_tag:            (47,)   = [...]                        ← multi-hot

  # 用户历史 (因果: 只取 row_id < 当前行)
  history:               (100,)  = [0,0,0,...0]              ← 编码后的历史物品 ID (0=padding)
  history_length:        6                                    ← 实际历史长度
  history_if_video_type: (100,3)                            ← 历史物品特征
  history_if_tag:        (100,47)                           ← 历史物品标签
  history_is_click:      (100,)                             ← 历史每步反馈
  ... (7 类历史反馈 × (100,))

  # 损失权重 (正样本=1, 负样本=反馈比例, is_hate 取负)
  loss_weight:  (7,) = [0.851, 0.497, 0.019, 0.003, 0.001, 0.001, -0.0005]
```

### 输出 → Step 4 输入

Batch 形式 (batch_size=B):

```text
{
    'user_id':                  (B,)              ← 编码用户 ID
    'item_id':                  (B,)              ← 编码物品 ID
    'is_click':                 (B,)              ← 标签
    ... (7类反馈, 各 (B,))
    'uf_user_active_degree':    (B, 9)            ← 用户 one-hot 特征
    ... (13个用户特征, 总 (B, 70))
    'if_video_type':            (B, 3)            ← 物品 one-hot 特征
    ... (4个物品特征, 总 (B, 70))
    'history':                  (B, 100)          ← 历史物品序列
    'history_length':           (B,)              ← 历史实际长度
    'history_if_video_type':    (B, 100, 3)       ← 历史物品特征
    'history_if_tag':           (B, 100, 47)      ← 历史物品标签
    ... (4个历史物品特征)
    'history_is_click':         (B, 100)          ← 历史反馈
    ... (7个历史反馈)
    'loss_weight':              (B, 7)            ← 损失权重
}
```

---

## Step 4: 训练 URM (KRMBUserResponse)

### 输入 ← Step 3 输出
训练/val/test DataLoader batch

### 模型内部数据流

```text
feed_dict → KRMBUserResponse.do_forward_and_loss(feed_dict):

  User Encoder:
    user_id (B,) → uIDEmb → (B, 32)
    uf_* (B, feat_dim) × 13 → Linear → (B, 32) × 13 → concat → (B, 14, 32)
    → userEmbNorm → userFeatureKernel → sum(dim=1) → user_enc (B, 64)

  Item Encoder:
    item_id (B,) → iIDEmb → (B, 32)
    if_* (B, feat_dim) × 4 → Linear → (B, 32) × 4 → concat → (B, 5, 32)
    → itemEmbNorm → itemFeatureKernel → sum(dim=2) → item_enc (B, 64)

  History Encoder:
    history (B,100) → Item Encoder → history_item_enc (B, 100, 64)
    history_is_click 等 (B,100,7) → feedbackEncoder → (B, 100, 64)
    concat → (B, 100, 128) + posEmb → TransformerEncoder → output_seq (B, 100, 128)
    取最后一步 → hist_enc (B, 128)

  State:
    state = [hist_enc, user_enc] → (B, 192) = 3 × enc_dim

  Scorer:
    state (B, 192) → DNN(128, 7×64) → (B, 448) → reshape → (B, 1, 7, 64)
    × item_enc (B, 1, 1, 64) → mean(dim=-1) → behavior_scores (B, 1, 7)

  Loss:
    BCE loss per feedback type, weighted by loss_weight
```

### 输出 → Step 7 输入

| 文件 | 说明 |
|------|------|
| `user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.checkpoint` | URM 模型权重 |
| `user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.windows.log` | KuaiSim 环境日志 |

---

## Step 5: 构建 SID 映射 (RQ-KMeans)

### 输入 ← Step 1 + Step 2 输出

| 文件 | 作用 |
|------|------|
| `video_features_basic_pure.csv` | 物品基础特征 (7,583 行) |
| `video_features_statistic_pure.csv` | 物品统计特征 |
| `log_session_4_08_to_5_08_Pure.csv` | 过滤: 只对日志中出现的 video 构建 SID |

### 处理逻辑

```text
1. 加载特征, merge basic + statistic → (#vids, 原始列)
2. 过滤: 只保留 log_session 中的 video_id → 7,583 个视频
3. 构建特征矩阵:
   - 数值列 → log1p → StandardScaler          → X_num (7,583, ~80)
   - 分类列 → OneHotEncoder                    → X_cat (7,583, ~20)
   - tag 列 → Top-500 标签 → MultiLabelBinarizer → X_tag (7,583, 47)
   - 拼接: X = [X_num, X_cat, X_tag]          → (7,583, 127)

4. RQ-KMeans (4层, codebook_size=32):
   Layer 1: KMeans(32).fit(X)           → codes[:,0], centers1 (32, 127), residual1 = X - centers1[codes[:,0]]
   Layer 2: KMeans(32).fit(residual1)   → codes[:,1], centers2 (32, 127), residual2 = residual1 - centers2[codes[:,1]]
   Layer 3: KMeans(32).fit(residual2)   → codes[:,2], centers3 (32, 127)
   Layer 4: KMeans(32).fit(residual3)   → codes[:,3], centers4 (32, 127)
   最终 reconstruction MSE = 5.48
```

**SID 编码 demo**：

```text
video_id=0  → codes=[9,25,7,6]  → SID token 序列 = [10,26,8,7]   (raw_sid+1, 0保留给PAD)
video_id=1  → codes=[3,22,10,26] → SID token 序列 = [4,23,11,27]
video_id=2  → codes=[9,30,15,30] → SID token 序列 = [10,31,16,31]

同一个 video 的 SID token 依次编码:
  sid_1=9  → 粗粒度分类 (32个大类之一)
  sid_2=25 → 中粒度细分 (在大类9内的32个子类之一)
  sid_3=7  → 细粒度细分
  sid_4=6  → 最细粒度
```

### 输出 → Step 6 + Step 7 输入

| 文件 | Shape | demo |
|------|-------|------|
| `video_sid_mapping.csv` | (7,583, 5) | `video_id=0, sid_1=9, sid_2=25, sid_3=7, sid_4=6` |
| `codebook_layer1.npy` | (32, 127) | 第1层码本 |
| `codebook_layer2.npy` | (32, 127) | 第2层码本 |
| `codebook_layer3.npy` | (32, 127) | 第3层码本 |
| `codebook_layer4.npy` | (32, 127) | 第4层码本 |
| `sid_config.json` | — | `{n_layers:4, codebook_size:32, feature_dim:127, num_videos:7583}` |

---

## Step 6: 训练 TIGER 基础策略

### 输入 ← Step 2 + Step 5 输出

| 文件 | 作用 |
|------|------|
| `log_session_4_08_to_5_08_Pure.csv` | 交互日志 (用户-物品序列) |
| `video_sid_mapping.csv` | video_id → SID token 映射 |

### 处理逻辑

TIGER 基于 T5，将推荐转为 seq2seq 生成：

```text
样本构造:
  用户历史 items [item_1, item_2, ..., item_H]
  → 查 SID 映射表 → [[sid1_1,sid2_1,sid3_1,sid4_1], [sid1_2,...], ...]
  → flatten → input_ids = [sid1_1+1, sid2_1+1, sid3_1+1, sid4_1+1,
                            sid1_2+1, sid2_2+1, sid3_2+1, sid4_2+1, ...]
                 长度 = H × 4 (例: H=50 → input_ids 长度 200)

  目标 item [item_target]
  → 查 SID 映射表 → [sid1_t, sid2_t, sid3_t, sid4_t]
  → target_ids = [sid1_t+1, sid2_t+1, sid3_t+1, sid4_t+1]
                 长度 = 4

  例:
    用户3历史50个items → input_ids: shape (200), 值域 [1, 32]
    目标 video_id=1071 (sid=[?,?,?,?]) → target_ids: shape (4), 值域 [1, 32]

  生成: Beam Search (width=16) → 生成4个token → 映射回 video_id
```

### 输出 → Step 7 输入

| 文件 | 说明 |
|------|------|
| `tiger_sid_krpure_mini.pth` | TIGER 基础策略 checkpoint |

---

## Step 7: SAGERec 闭环训练

### 输入 ← Step 4 + Step 5 + Step 6 输出

| 文件 | 来源 |
|------|------|
| `tiger_sid_krpure_mini.pth` | ← Step 6 |
| `user_KRMBUserResponse_*.model.windows.log` | ← Step 4 |
| `video_sid_mapping.csv` | ← Step 5 |

### 每轮迭代 5 个子步骤

#### 7a. Rollout（收集轨迹）

```text
输入:  当前 TIGER policy + KuaiSim 环境(URM日志)
处理:  policy beam search 生成推荐列表(slate_size=6)
       → URM 返回7类反馈 → 计算reward → 记录轨迹
输出:  rollout_trace.jsonl (每行一条episode的step记录)
       → 7b 输入
```

#### 7b. Critic 训练

```text
输入:  rollout_trace.jsonl ← 7a + TIGER checkpoint ← Step 6
处理:  训练 ensemble(5个) PageSIDQCritic
       输入: (state, items) → Q_pess(state, slate)
       Q_pess = Q_mean - beta × Q_std  (悲观估计)
输出:  page_sid_qcritic_bundle.pt  → 5个critic权重
       page_sid_qcritic_meta.json  → 配置
       page_sid_qcritic_metrics.json → 训练指标
       → 7c 输入
```

#### 7c. 归因链构建

```text
输入:  rollout_trace.jsonl ← 7a + critic_bundle ← 7b + TIGER ← Step 6
处理:  分层优势计算:
       page_q      = Q_pess(full slate)                    → 页面级奖励
       item_adv[i] = Q_pess(with item_i) - Q_pess(without item_i) → 物品级优势
       sid_adv[k]  = Q_pess(prefix_k) - Q_pess(prefix_{k-1})     → token级优势

       例: slate=[item_a, item_b, item_c]
       page_q = Q(state, [a,b,c]) = 4.67
       item_adv[a] = Q(state,[a,b,c]) - Q(state,[b,c]) = 0.5
       item_adv[b] = Q(state,[a,b,c]) - Q(state,[a,c]) = 0.3
       item_adv[c] = Q(state,[a,b,c]) - Q(state,[a,b]) = 0.2

输出:  sid_advantage_chain.jsonl
       → 7d 输入
```

#### 7d. Group 构造

```text
输入:  sid_advantage_chain.jsonl ← 7c + critic_bundle ← 7b + TIGER ← Step 6
处理:  对每个上下文构造候选组:
       - behavior candidate: 当前策略生成的推荐
       - beam candidates:    beam search 生成的多个候选
       - 计算 support gap (策略分布 vs 训练数据分布)
       - 计算 adaptive_support_pess = group_reward - support_penalty

       例: 1个上下文 → group_size=8个候选 → 排序 → 选出最优/最差
输出:  hca_grpo_groups.jsonl
       → 7e 输入
```

#### 7e. Actor 更新 (GRPO)

```text
输入:  hca_grpo_groups.jsonl ← 7d + TIGER checkpoint ← Step 6
处理:  adaptive-trust GRPO:
       loss = -min(ratio × adv, clip(ratio, ε) × adv) + kl_scale × KL

       adaptive trust:
         kl_scale = 0.10 + 1.00 × support_gap + 0.25 × uncertainty
         clip_eps = 0.08 - 1.00 × support_gap - 0.25 × uncertainty

       support gap ↑ → KL 更重, clip 更小 → 更新更保守
       uncertainty ↑ → 同上

       train_scope = decoder_only (只更新 T5 decoder)
输出:  新 TIGER checkpoint → iter_01/grpo_actor/
       EMA 同步 rollout policy → 下一轮迭代使用 (sync_tau=0.20)
```

### 最终输出结构

```text
results/sagerec_adaptive_grpo_*/
├── closed_loop_summary.json
├── iter_01/
│   ├── rollout_trace.jsonl           ← 7a 输出
│   ├── page_qcritic/
│   │   ├── page_sid_qcritic_bundle.pt ← 7b 输出
│   │   ├── page_sid_qcritic_meta.json
│   │   └── page_sid_qcritic_metrics.json
│   ├── sid_advantage_chain.jsonl      ← 7c 输出
│   ├── hca_grpo_groups.jsonl          ← 7d 输出
│   ├── grpo_actor/                    ← 7e 输出
│   │   └── tiger_grpo_actor.pth
│   └── rollout_policy/
├── iter_02/                           ← 第2轮迭代
└── iter_03/                           ← 第3轮迭代
```

---

## Step 8: 闭环环境评估

### 输入 ← Step 4 + Step 5 + Step 6/7e 输出

| 文件 | 来源 | 作用 |
|------|------|------|
| TIGER checkpoint | ← Step 6 或 Step 7e | 待评估的策略 |
| URM windows.log | ← Step 4 | KuaiSim 环境日志 |
| `video_sid_mapping.csv` | ← Step 5 | SID 映射 |

### 处理逻辑

评估为**闭环交互评估**，非静态测试集。流程如下：

```text
1. 加载 TIGER checkpoint + SID mapping + KuaiSim 环境(URM)
2. 运行 N 个 episode (默认 200), 每个最多 20 步:
   for each step:
     TIGER policy → beam search(beam_width=16) → 生成 slate_size=6 个物品
     → KuaiSim(URM) 返回 7 维反馈 (is_click, long_view, ...)
     → get_immediate_reward() 计算即时奖励
     → 累加 reward, 更新用户历史
     → 用户退出(done)则结束 episode

3. 统计所有 episode 的平均指标
```

**评估协议参数**：

```text
slate_size=6           → 每步推荐 6 个物品
beam_width=16          → TIGER beam search 宽度
eval_episodes=200      → 跑 200 个 episode
max_steps_per_episode=20 → 每个 episode 最多 20 步
item_correlation=0.2   → 物品相似度惩罚
phase2_blend_scale=0.20 → 混合 phase-2 物品的概率
random_topk_sample=10  → 随机采样 top-k
seed=2026              → 固定随机种子
```

### SAGERec 闭环中的评估时机

每轮迭代有 3 个评估节点：

```text
Iter N:
  ┌─ Before Eval ─── 评估当前 rollout 策略 (本轮训练前)
  │                   输入: rollout_tiger_ckpt ← 上轮 7e 或 Step 6
  │                   输出: before_eval_metrics → {total_reward, depth, ...}
  │
  ├─ Rollout ──────── 收集轨迹 (64 episodes, phase2_blend)
  │                   输出: rollout_trace.jsonl → 7b 输入
  │
  ├─ Train Critic ── 7b
  ├─ Build Chain ─── 7c
  ├─ Build Groups ─── 7d
  ├─ GRPO Actor ──── 7e → learner_tiger_ckpt
  ├─ EMA Sync ────── rollout ← learner
  │
  └─ After Eval ──── 评估更新后的两个策略:
     ├─ after_eval_learner: 评估 learner_tiger_ckpt ← 7e
     └─ after_eval_rollout: 评估 rollout_tiger_ckpt ← EMA
     输出: after_eval_learner_metrics + after_eval_rollout_metrics
```

对应消融表的列含义：

| 消融表列 | 来源 | 含义 |
|----------|------|------|
| Before | before_eval.log | 训练前 rollout 策略得分 |
| Rollout | rollout.log | 收集轨迹时策略得分 |
| After Rollout | after_eval_rollout.log | EMA 同步后 rollout 策略得分 |
| After Learner | after_eval_learner.log | GRPO 更新后 learner 策略得分 |

### 评估指标输出

```text
eval_tiger_env.py 输出格式 (解析到 closed_loop_summary):

========================================
Total Reward: 4.5417
Depth: 12.26
Avg Step Reward: 0.3706
Coverage: 0.81
ILD: 0.9870
Table-style metrics:
Depth: 12.26
Average reward: 0.3706
Total reward: 4.5417
Coverage: 0.81
ILD: 0.9870
Behavior rates (count / impressions):
  is_click: 4717/12672 (37.0597%)
  long_view: 3666/12672 (28.9270%)
  is_like: 238/12672 (1.8826%)
  is_comment: 32/12672 (0.2552%)
  is_forward: 12/12672 (0.0963%)
  is_follow: 13/12672 (0.1075%)
  is_hate: 6/12672 (-0.0494%)    ← is_hate 为负反馈，计入惩罚
========================================
```

**parse_eval_metrics 从输出解析的字段**：

```text
{
    "total_reward": 4.5417,     ← 主指标 (消融表用此列)
    "depth": 12.26,             ← 平均 episode 长度
    "avg_reward": 0.3706,       ← avg_step_reward
    "coverage": 0.81,           ← 物品覆盖率
    "click": 37.0597,           ← click rate (%)
    "long_view": 28.9270,       ← long_view rate (%)
}
```

### 输出 → 消融表 / 论文

| 文件 | 内容 |
|------|------|
| `closed_loop_summary.json` | 每轮迭代的完整指标 |
| `closed_loop_summary.jsonl` | 每轮一行，方便解析 |
| `plots/*.png` | 可视化图表 |
| `iter_*/before_eval.log` | 训练前评估日志 |
| `iter_*/after_eval_learner.log` | learner 评估日志 |
| `iter_*/after_eval_rollout.log` | rollout 评估日志 |

### 评估命令

```bash
# 单次策略评估 (任何 TIGER checkpoint)
python code/eval_tiger_env.py \
  --tiger_ckpt output/KuaiRand_Pure/env/tiger_sid_krpure_mini.pth \
  --sid_mapping_path code/dataset/kuairand/kuairand-Pure/sid/32_mask/video_sid_mapping.csv \
  --uirm_log_path code/output/Kuairand_Pure/env/log/user_KRMBUserResponse_lr0.0001_reg0_nlayer2.model.windows.log \
  --num_episodes 200 --beam_width 16 --slate_size 6 --seed 2026

# 统一评估套件 (所有基线 + 消融)
bash code/run_tiger_base_posttrain_suite_strict_eval3_20260513.sh
SMOKE=1 bash code/run_tiger_base_posttrain_suite_strict_eval3_20260513.sh  # 烟测
```

---

## 流程数据对齐速查

```text
Step 1 输出 ─→ Step 2 输入 (6个原始CSV)
Step 2 输出 ─→ Step 3 输入 (log_session + user/item fillna)
Step 2 输出 ─→ Step 5 输入 (video_features + log_session 过滤)
Step 3 输出 ─→ Step 4 输入 (训练batch: B × {ids, feats, history, labels})
Step 4 输出 ─→ Step 7 输入 (URM checkpoint + windows.log)
Step 4 输出 ─→ Step 8 输入 (URM windows.log → KuaiSim 环境)
Step 5 输出 ─→ Step 6 输入 (SID mapping)
Step 5 输出 ─→ Step 7 输入 (SID mapping)
Step 5 输出 ─→ Step 8 输入 (SID mapping)
Step 6 输出 ─→ Step 7 输入 (TIGER checkpoint)
Step 6 输出 ─→ Step 8 输入 (TIGER checkpoint → 基础策略评估)
Step 7e 输出 ─→ Step 8 输入 (TIGER checkpoint → 每轮迭代后评估)

Step 7 内部:
  7a 输出 ─→ 7b 输入 (rollout_trace)
  7b 输出 ─→ 7c 输入 (critic_bundle)
  7c 输出 ─→ 7d 输入 (advantage_chain)
  7d 输出 ─→ 7e 输入 (groups)
  7e 输出 ─→ 7a 输入 (下一轮的新TIGER checkpoint)

Step 8 在 Step 7 内部调用:
  Before Eval ──→ before_eval_metrics → closed_loop_summary
  After Learner ──→ after_eval_learner_metrics → closed_loop_summary
  After Rollout ──→ after_eval_rollout_metrics → closed_loop_summary
```