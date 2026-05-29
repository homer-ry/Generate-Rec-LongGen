# TIGER HCA-GRPO 代码阅读报告

本文基于以下参考材料和仓库实现整理：

- [ref/ref1.md](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/ref/ref1.md)
- [ref/ref2.md](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/ref/ref2.md)

目标是快速说明这套长期价值生成式推荐代码在做什么、核心文件在哪、数据如何流动、训练和评估怎么串起来，以及阅读时最容易卡住的 tensor shape。

---

## 1. 一句话总览

这条主线可以概括成：

1. 用 KuaiRand 日志和 SID 映射预训练一个 TIGER 生成式基础策略。
2. 用基础策略在模拟环境里 rollout，得到 session trace。
3. 从 trace 构造 page 级长期价值训练样本，训练 page/item/SID 多粒度 critic。
4. 用 critic 对每个 page 和候选 item 做层级归因，得到 `page_q`、`item_advantage`、`sid_advantage`。
5. 再把行为候选和 beam 候选组成 group，构造 `group_advantage`。
6. 最后用 `train_tiger_hca_grpo_actor.py` 做一轮 conservative 的 GRPO-style actor 更新。

可以把它理解成：

- TIGER 负责“生成什么 item”
- Critic 负责“长期价值评估”
- HCA 负责“把 page 价值拆到账户到 item / SID token”
- GRPO 负责“把这些 credit 变成稳定的策略更新”

---

## 2. 代码主线地图

### 2.1 基础策略训练

- [code/train_TIGER_krpure.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_TIGER_krpure.py:1)
  - 用 KuaiRand 行为日志和 SID 映射训练 TIGER。
  - 核心数据集类：`KuaiRandSIDTigerDataset`
  - 核心模型类：`TIGER`

### 2.2 环境评估 / Rollout

- [code/eval_tiger_env.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/eval_tiger_env.py:1)
  - 把用户历史 item 编码成 SID token 序列。
  - TIGER beam search 生成候选 SID。
  - 映射回 item 后在 KuaiSim 环境里执行。

### 2.3 Critic 训练

- [code/tiger_page_sid_rl/train_page_critic.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:1)
  - 从 rollout trace 构造 page 样本。
  - 训练 page-level Q critic。
  - 可加 item / SID 前缀辅助损失，也支持 ensemble。

- [code/tiger_page_sid_rl/models.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/models.py:349)
  - `PageSIDQCriticV9Additive`
  - `PageSIDQCriticEnsemble`

### 2.4 归因链构造

- [code/tiger_page_sid_rl/build_sid_advantage_chain.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/build_sid_advantage_chain.py:1)
  - 计算 `page_q`、`item_advantage`、`sid_advantage`
  - 保存 advantage chain

### 2.5 Group 构造

- [code/build_tiger_hca_grpo_groups.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/build_tiger_hca_grpo_groups.py:522)
  - 为同一上下文构建一组行为候选 + beam 候选
  - 写出 `group_advantage`、`support_gap_scaled`、`uncertainty_ratio`

### 2.6 Actor 更新

- [code/train_tiger_hca_grpo_actor.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:18)
  - 用 grouped JSONL 做 GRPO-style actor 更新
  - 实际上是 “clipped PPO + HCA credit + adaptive trust”

---

## 3. 数据集和输入表示

### 3.1 原始数据

参考文档里提到的主数据源是 KuaiRand。这里代码真正消费的是：

- 行为日志 CSV
- `video_sid_mapping.csv`

SID 的作用是把一个 item 离散成定长 token 序列，供 TIGER 当作“生成目标序列”来预测。

### 3.2 TIGER 训练样本是怎么构造的

`KuaiRandSIDTigerDataset` 在 [train_TIGER_krpure.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_TIGER_krpure.py:78) 里完成了基础样本构造。

每条样本大致是：

- `history`: 历史 item 的 SID token 展平序列
- `attention_mask`
- `target`: 当前正样本 item 的 SID token
- `sample_weight`

核心规则：

- 每个 item SID 长度固定为 `sid_depth`
- 原始 SID code 是 `0..K-1`
- 模型 token id 使用 `sid_raw + 1`
- `0` 保留给 PAD / EOS

### 3.3 基础策略训练的关键 shape

设：

- `B` = batch size
- `H` = `max_hist_items`
- `D` = `sid_depth`
- `V` = `vocab_size`

则基础 TIGER 训练时：

- `history`: `[B, H * D]`
- `attention_mask`: `[B, H * D]`
- `target`: `[B, D]`
- `logits`: `[B, D, V]`
- token CE loss reshape 前：`[B * D, V]`

对应代码：

- 样本组织：[train_TIGER_krpure.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_TIGER_krpure.py:225)
- 训练一步：[train_TIGER_krpure.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_TIGER_krpure.py:332)

### 3.4 验证集切分方式

基础 TIGER 不是随机按样本切，而是按每个用户最后一个样本作为验证集：

- `uid_last_idx[uid] = idx`
- 所有用户最后一个样本进 `val_indices`

实现见 [train_TIGER_krpure.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_TIGER_krpure.py:719)

这能减少同一用户相邻行为泄漏到 train/val 两边。

---

## 4. TIGER 模型本体

### 4.1 模型本质

TIGER 在这个仓库里本质是一个 T5 encoder-decoder：

- encoder 输入：用户历史对应的 SID token 序列
- decoder 输出：目标 item 的 SID token 序列

定义见：

- [code/train_TIGER_krpure.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_TIGER_krpure.py:17)
- [code/tiger_phase2_blend_common.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_phase2_blend_common.py:13)

### 4.2 历史输入如何组织

历史 item 序列先映射成 `[B, H, D]` 的 SID token，再 flatten 成 `[B, H * D]`。

实现见：

- [build_history_tokens()](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_phase2_blend_common.py:367)
- [TigerSIDPolicy._build_history_tokens()](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/eval_tiger_env.py:168)

这是阅读本仓库最重要的 shape 之一：

- item history 不是直接喂 item id
- 而是先展开成 SID token 平铺序列

---

## 5. Rollout 和评估流程

### 5.1 `eval_tiger_env.py` 做了什么

评估主线：

1. 环境返回用户历史 `history`
2. 历史 item 转成 SID token 输入
3. TIGER beam search 生成若干 SID 序列
4. 用 `sid2iid_map` 把 SID 序列映射回 item
5. 过滤重复和历史里出现过的 item
6. 在环境中执行 slate
7. 累积 reward、depth、coverage、ILD、行为率

### 5.2 评估时的关键 shape

在 [TigerSIDPolicy.act()](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/eval_tiger_env.py:198) 里：

- `observation["user_history"]["history"]`: `[B, H_env]`
- 编码后 `input_ids`: `[B, H * D]`
- `generate()` 输出原始 shape：`[B * W, D + 1]`
- 去掉 decoder start token 后：`[B * W, D]`
- reshape 后：`[B, W, D]`

其中：

- `B` = 环境 batch size
- `W` = beam width
- `D` = sid depth

这是理解 beam 候选构造和后续 group 构造的基础。

### 5.3 环境指标

最终会打印：

- `Total Reward`
- `Depth`
- `Avg Step Reward`
- `Coverage`
- `ILD`
- 每个行为类型的触发率

实现见 [eval_tiger_env.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/eval_tiger_env.py:347)

---

## 6. Critic 训练逻辑

### 6.1 critic 训练样本从哪里来

`train_page_critic.py` 并不直接读取原始日志，而是读取 rollout trace，然后用：

- `build_page_samples()`

把每个 page 转成一个监督样本。

实现见：

- [train_page_critic.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:129)
- [build_page_samples()](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/common.py:301)

### 6.2 page sample 里包含什么

每条 page sample 主要字段：

- `pre_input_ids`, `pre_attention_mask`
- `token_ids`
- `page_features`
- `user_features`
- `q_target`
- `item_share_target`
- `item_adv_target`
- `sid_adv_target`

其中：

- `q_target` = page 级 discounted return
- `item_adv_target` = item 级 credit
- `sid_adv_target` = SID token 级 credit

### 6.3 page sample 的关键 shape

`collate_trace_pages()` 在 [common.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/common.py:479) 里把样本拼成 batch。

设：

- `B` = page batch size
- `S` = page 内 slate size 上界
- `D` = sid depth
- `F_page` = page feature dim，当前固定为 `3`
- `F_user` = user feature dim
- `L_hist` = `max_hist_items * sid_depth`

则批次张量 shape 如下：

- `pre_input_ids`: `[B, L_hist]`
- `pre_attention_mask`: `[B, L_hist]`
- `page_features`: `[B, 3]`
- `user_features`: `[B, F_user]`
- `token_ids`: `[B, S, D]`
- `item_mask`: `[B, S]`
- `q_target`: `[B]`
- `item_share_target`: `[B, S]`
- `item_adv_target`: `[B, S]`
- `sid_share_target`: `[B, S, D]`
- `sid_adv_target`: `[B, S, D]`

这是后面 critic forward、归因链和 group builder 一直复用的核心 shape。

### 6.4 critic 的输入输出

在 [forward_batch()](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:243) 里，先用 TIGER encoder 对历史做 pooling：

- `pre_summary = pooled_history_summary(tiger, pre_input_ids, pre_attention_mask)`

输出 shape：

- `pre_summary`: `[B, hidden_size]`

然后喂给 critic：

- `pre_summary`: `[B, Hc]`
- `token_ids`: `[B, S, D]`
- `item_mask`: `[B, S]`
- `page_features`: `[B, F_page]`
- `user_features`: `[B, F_user]`

critic 输出：

- 单模型：
  - `q_value`: `[B]`
- ensemble：
  - `q_values`: `[B, M]`
  - `q_mean`: `[B]`
  - `q_std`: `[B]`

对应代码：

- `extract_q_outputs()` [train_page_critic.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:90)
- `PageSIDQCriticEnsemble` [models.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/models.py:616)

### 6.5 多 Critic / Ensemble 设计

参考 `ref1.md` 提到“多个独立 head 共享 backbone”，从当前代码实现看，更准确地说是：

- ensemble 外层用 `PageSIDQCriticEnsemble`
- 里面是多个完整 critic member
- forward 时把每个 member 的 `q_value` 堆叠成 `q_values`
- 再计算：
  - `q_mean = mean(q_values)`
  - `q_std = std(q_values)`

实现见 [models.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/models.py:623)

这一步直接决定后面：

- pessimistic score
- uncertainty penalty
- adaptive trust

### 6.6 当前最关键的 critic 结构

参考材料里重点点名的是 `PageSIDQCriticV9Additive`，它也是这条主线最值得看的版本。

位置：

- [PageSIDQCriticV9Additive](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/models.py:349)

可以直观理解成 5 步：

1. `token_ids [B,S,D]` 先做 token embedding
2. item 内部 token 聚合，得到 item 表示
3. item 间做 self-attention
4. page context 对 item 做 cross-attention
5. page base value + 每个 item additive contribution 相加，得到最终 `q_value`

这也是名字里 `Additive` 的来源：

- `q_value = base_value + item_contrib.sum(dim=1)`

见 [models.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/models.py:585)

### 6.7 critic 的辅助损失

如果打开辅助项，`train_page_critic.py` 还会做：

- page loss
- item delta loss
- SID prefix delta loss
- pairwise rank loss
- monotonic loss

实现见 [train_page_critic.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:260)

其中最重要的是 variant batch：

- 对每个 item 构造“删掉该 item”的 null variant
- 对每个 SID token 构造 prefix variant

构造函数：

- [build_variant_batch()](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:172)

如果一个 batch 有 `B` 个 page，每个 page 平均 `S` 个 item，每个 item 有 `D` 个 SID token，那么变体数近似是：

- 每个 item 1 个 null variant
- 每个 item 再加 `valid_len` 个 prefix variant
- 总体大约 `B * S * (1 + D)`

这就是 critic 训练显存/耗时会显著上升的主要原因。

---

## 7. 归因链：item_advantage 和 sid_advantage 怎么来的

### 7.1 主文件

- [code/tiger_page_sid_rl/build_sid_advantage_chain.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/build_sid_advantage_chain.py:1)

### 7.2 归因基本思路

对一个 page 中的每个 item：

1. 先算完整 slate 的 `Q(x, y)`
2. 再算去掉该 item 后的 `Q(x, y \\ i)`
3. item advantage:
   - `item_adv = Q(x, y) - Q(x, y \\ i)`
4. 对该 item 的 SID token 逐段做 prefix：
   - `A_sid,k = Q(prefix到k) - Q(prefix到k-1)`

实现就在：

- [build_page_variants()](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/build_sid_advantage_chain.py:80)
- [main 循环里的差分计算](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/build_sid_advantage_chain.py:169)

### 7.3 归因时的关键 shape

对单个 page：

- `token_ids`: `[S, D]`
- 完整 page variant + null variants + prefix variants 组成：
  - `variants`: `[N_var, S, D]`

其中：

- 第 0 个 variant 是完整 slate
- 接下来每个 item 一个 null variant
- 再接着是每个 token prefix variant

critic 批量评估后得到：

- `q_mean`: `[N_var]`
- `q_std`: `[N_var]`
- `q_pess`: `[N_var]`

然后再回填成：

- `item_advantage`: `[S]`
- `sid_advantage`: `[S, D]`

### 7.4 pessimistic 版本

归因链同时保留：

- `item_advantage`
- `item_advantage_pess`
- `sid_advantage`
- `sid_advantage_pess`

其中 pessimistic 版本基于：

- `q_pess = q_mean - beta * q_std`

这一步把 ensemble 不确定性显式融入 credit。

---

## 8. Group 构造逻辑

### 8.1 主文件

- [code/build_tiger_hca_grpo_groups.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/build_tiger_hca_grpo_groups.py:522)

### 8.2 它做了什么

对于同一个 `(episode_id, page_index, slot_index)`：

1. 保留行为 item 作为一个候选
2. 用当前 TIGER 生成 beam 候选
3. 对每个候选计算 critic 价值、support、uncertainty
4. 把 reward 做 group 内标准化
5. 产出训练 actor 的 grouped JSONL

### 8.3 输出字段重点

每条 group record 大致包含：

- `input_ids`, `attention_mask`
- `target_tokens`
- `page_q_value`, `page_q_std`, `page_q_pess`
- `item_advantage`, `sid_advantage`
- `support_logprob_mean`
- `support_gap_scaled`
- `uncertainty_ratio`
- `reward_raw`
- `group_advantage`

实现见 [build_tiger_hca_grpo_groups.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/build_tiger_hca_grpo_groups.py:565)

### 8.4 trust 信号怎么构造

最关键的是两项：

- `support_gap_scaled`
  - 候选相对 behavior 的支持度差距
- `uncertainty_ratio`
  - 当前候选的不确定性相对组均值

它们会进一步影响：

- `adaptive_support_pess`
- actor 阶段的 adaptive KL / adaptive clip

实现见 [build_tiger_hca_grpo_groups.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/build_tiger_hca_grpo_groups.py:602)

---

## 9. Actor 更新逻辑

### 9.1 主文件

- [code/train_tiger_hca_grpo_actor.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:18)

### 9.2 输入数据 shape

`collate_rows()` 后的 batch：

- `input_ids`: `[B, L_hist]`
- `attention_mask`: `[B, L_hist]`
- `target_tokens`: `[B, D]`
- `token_adv`: `[B, D]`
- `item_adv`: `[B]`
- `group_adv`: `[B]`
- `page_reward`: `[B]`
- `trust_support`: `[B]`
- `trust_unc`: `[B]`

实现见 [train_tiger_hca_grpo_actor.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:102)

### 9.3 训练时 forward 的关键 shape

在 `forward_actor()` 中：

- `decoder_input_ids = decoder_input_ids_from_targets(target_tokens)`
- 若 `target_tokens` 是 `[B, D]`
- 则 `decoder_input_ids` 也是 `[B, D]`

old policy 和 actor policy 都输出：

- `logits`: `[B, D, V]`

再提取目标 token log-prob：

- `actor_target_logp`: `[B, D]`
- `old_target_logp`: `[B, D]`

### 9.4 HCA + GRPO 的真正结合点

最关键的是 [build_effective_advantages()](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:230)。

它把：

- group 级 `group_adv`
- item 级 `item_adv`
- token 级 `token_adv`
- page 级 `page_reward`

合成为：

- `effective_adv`: `[B, D]`
- `weights`: `[B, D]`

阅读时建议牢记一句话：

- `group_adv` 决定正向还是负向更新
- `item_adv` / `token_adv` 决定更新主要落在哪些 SID token 上
- `page_reward` 决定整条样本 gate 大小

### 9.5 两种 advantage 融合模式

#### `multiplicative`

逻辑：

- 先根据 group advantage 正负，选正 token 或负 token
- 稀疏 top-k 归一化后得到 token 权重
- 最终：
  - `effective_adv = group_adv * page_gate * weights`

#### `additive_zero_sum`

逻辑：

- 先把 group advantage 平均分给激活 token
- 再加一个 token/item attribution 的零和 residual
- 保持总 group credit 不变，只做组内重分配

核心代码在：

- [train_tiger_hca_grpo_actor.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:279)

这是整个仓库最值得精读的几行之一。

### 9.6 实际 loss 不是纯 PPO

`compute_grpo_loss()` 最终组合了：

- clipped policy loss
- KL penalty
- entropy bonus
- optional SFT CE loss

即：

- `loss = policy_loss + kl_scale * kl_loss - entropy_scale * entropy_bonus + sft_scale * ce_loss`

实现见 [train_tiger_hca_grpo_actor.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:315)

所以更准确地说，这是：

- GRPO-style clipped PPO
- 加上 HCA credit
- 再加 adaptive trust

### 9.7 trust 如何影响更新

train 阶段读取：

- `support_gap_scaled`
- `uncertainty_ratio`

变成：

- `trust_multiplier`
- `clip_multiplier`

效果：

- 不可信样本的 KL 更重
- 不可信样本的 clip 更小

实现见 [train_tiger_hca_grpo_actor.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:373)

这正对应参考材料里强调的 conservative update 思路。

---

## 10. 训练、验证、评估各自关注什么

### 10.1 基础 TIGER 训练

训练目标：

- teacher forcing token CE

验证指标：

- `Recall@K`
- `NDCG@K`
- `teacher forcing loss`

实现见：

- [train_TIGER_krpure.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_TIGER_krpure.py:332)
- [train_TIGER_krpure.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_TIGER_krpure.py:383)

### 10.2 Critic 训练

训练目标：

- `page_loss`
- 可选 `item_loss`
- 可选 `prefix_loss`
- 可选 `rank_loss`
- 可选 `monotonic_loss`

验证指标：

- `loss`
- `q_mae`
- `q_corr`
- `item_delta_mae`
- `prefix_delta_mae`

### 10.3 环境评估

关心：

- total reward
- avg step reward
- depth
- coverage
- ILD
- behavior rates

### 10.4 Actor 更新

关心：

- `loss`
- `policy_loss`
- `kl_loss`
- `target_gain`
- `approx_kl`
- `clip_frac`
- `page_gate_mean`
- `active_frac`

这些统计能帮助判断：

- 更新是否太猛
- 是否只在极少 token 上起作用
- page gate 是否过强
- adaptive clip / adaptive KL 是否生效

---

## 11. 最建议优先精读的函数

如果时间有限，建议按这个顺序读：

1. [KuaiRandSIDTigerDataset.__init__](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_TIGER_krpure.py:78)
   - 看清 item 是怎么变成 SID token 的。
2. [TigerSIDPolicy.act](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/eval_tiger_env.py:198)
   - 看清评估时 beam 是怎么生成、去重、映射回 item 的。
3. [build_page_samples](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/common.py:301)
   - 看清 critic 监督信号从 trace 怎么来。
4. [PageSIDQCriticV9Additive.forward_from_item_repr](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/models.py:483)
   - 看清 critic 结构。
5. [build_sid_advantage_chain.main](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/build_sid_advantage_chain.py:101)
   - 看清 item / SID 差分归因。
6. [build_tiger_hca_grpo_groups.py 主循环](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/build_tiger_hca_grpo_groups.py:522)
   - 看清 group advantage 和 trust 字段。
7. [build_effective_advantages](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:230)
   - 看清 HCA + GRPO 真正耦合点。
8. [compute_grpo_loss](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:315)
   - 看清最终优化目标。

---

## 12. 阅读时最容易混淆的点

### 12.1 item id 和 SID token 不是一回事

- 环境里动作是 item id
- TIGER 生成的是 SID token 序列
- 中间必须经过 `sid2iid_map`

### 12.2 page / item / SID 三层 credit 是不同粒度

- `page_q` 是整个 slate 的长期价值
- `item_advantage` 是“这个 item 放进页面值不值”
- `sid_advantage` 是“这个 item 的哪几个 SID token 更关键”

### 12.3 actor 更新不是直接模仿最优候选

它不是简单 SFT，也不是 DPO，而是：

- 先做相对优势
- 再按 token 做稀疏 credit
- 最后做 clipped PPO 风格 update

### 12.4 ensemble 的 `q_std` 很重要

它不只是日志指标，而是真正进入了：

- pessimistic score
- support-aware reward
- adaptive trust update

---

## 13. 一个最小 shape 调试清单

如果后面要跑代码或加 debug，最值得先打印这些 shape：

### 13.1 基础 TIGER

- `history.shape`
- `attention_mask.shape`
- `target.shape`
- `logits.shape`

预期：

- `[B, H*D]`
- `[B, H*D]`
- `[B, D]`
- `[B, D, V]`

### 13.2 Critic batch

- `pre_input_ids.shape`
- `token_ids.shape`
- `item_mask.shape`
- `page_features.shape`
- `user_features.shape`
- `q_target.shape`

预期：

- `[B, L_hist]`
- `[B, S, D]`
- `[B, S]`
- `[B, 3]`
- `[B, F_user]`
- `[B]`

### 13.3 Critic variant batch

- `variant_token_ids.shape`
- `variant_item_mask.shape`
- `variant_owner_idx.shape`

预期：

- `[N_var, S, D]`
- `[N_var, S]`
- `[N_var]`

### 13.4 Actor batch

- `target_tokens.shape`
- `token_adv.shape`
- `group_adv.shape`
- `effective_adv.shape`
- `actor_logits.shape`
- `old_logits.shape`

预期：

- `[B, D]`
- `[B, D]`
- `[B]`
- `[B, D]`
- `[B, D, V]`
- `[B, D, V]`

---

## 14. 总结

这套代码的核心创新点，可以很直观地归纳成三件事：

1. 用 TIGER 把推荐变成“生成 SID token 序列”问题。
2. 用 page/item/SID 多粒度 critic，把长期价值拆成可训练的层级 credit。
3. 用带 support 和 uncertainty 约束的 GRPO-style actor update，尽量稳定地把 critic 信号反哺给策略。

如果只抓主线，最值得盯住的是三段代码：

- [PageSIDQCriticV9Additive](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/models.py:349)
- [build_sid_advantage_chain.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/build_sid_advantage_chain.py:101)
- [train_tiger_hca_grpo_actor.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:230)

它们基本对应：

- critic 怎么看页面
- credit 怎么拆到 item / token
- credit 怎么变成策略更新

---

## 15. 按执行顺序读代码

这一节专门服务“打开文件后应该先看哪几个函数”的需求。建议直接对着函数顺序读，不要一开始就从中间 loss 公式硬啃。

### 15.1 `train_page_critic.py` 阅读顺序

建议顺序：

1. `parse_args()`
2. `build_samples_from_trace()`
3. `build_loaders()`
4. `forward_batch()`
5. `train_one_epoch()`
6. `evaluate()`
7. `run_training()`

#### 第一步：看输入是什么

`parse_args()` 在 [train_page_critic.py:46](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:46)。

最关键的参数分 4 类：

- 数据来源：
  - `trace_path`
  - `uirm_log_path`
  - `sid_mapping_path`
- 目标构造：
  - `gamma`
  - `hazard_lambda`
  - `critic_target_heuristic_mix`
  - `critic_target_support_mix`
  - `critic_target_response_mix`
- critic 架构：
  - `critic_arch`
  - `ensemble_size`
  - `critic_num_heads`
  - `critic_num_layers`
- loss 组成：
  - `critic_page_loss_scale`
  - `critic_item_loss_scale`
  - `critic_prefix_loss_scale`
  - `critic_rank_loss_scale`
  - `critic_monotonic_loss_scale`

如果第一次读，只要先记住：

- `trace_path` 决定样本
- `critic_arch` 决定结构
- 一堆 `*_loss_scale` 决定这次训不训 item / prefix 辅助目标

#### 第二步：看 trace 怎么变成训练样本

`build_samples_from_trace()` 在 [train_page_critic.py:134](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:134)。

它做 3 件事：

1. 读取 rollout trace JSONL
2. 从 `uirm_log_path` 里恢复 reader
3. 调 `build_page_samples()` 产出 page 级样本

这里真正的“数据工程核心”在 [build_page_samples()](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/common.py:301)。

读这个函数时建议把每条 sample 理解为：

- 一个 page
- 这个 page 有一组 item，每个 item 是 SID token 序列
- 监督目标不是点击率，而是长期回报和 credit 拆分结果

#### 第三步：看 batch 进模型前长什么样

`build_loaders()` 在 [train_page_critic.py:102](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:102)，实际拼 batch 的是 [collate_trace_pages()](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/common.py:479)。

最值得直接打出来的 shape：

```python
print("pre_input_ids", batch["pre_input_ids"].shape)
print("page_features", batch["page_features"].shape)
print("user_features", batch["user_features"].shape)
print("token_ids", batch["token_ids"].shape)
print("item_mask", batch["item_mask"].shape)
print("q_target", batch["q_target"].shape)
print("item_adv_target", batch["item_adv_target"].shape)
print("sid_adv_target", batch["sid_adv_target"].shape)
```

预期：

- `pre_input_ids`: `[B, L_hist]`
- `page_features`: `[B, 3]`
- `user_features`: `[B, F_user]`
- `token_ids`: `[B, S, D]`
- `item_mask`: `[B, S]`
- `q_target`: `[B]`
- `item_adv_target`: `[B, S]`
- `sid_adv_target`: `[B, S, D]`

#### 第四步：看 `forward_batch()` 真正在优化什么

`forward_batch()` 在 [train_page_critic.py:244](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:244)。

建议按这个顺序读：

1. 先看输入张量取出部分
2. 看 `pre_summary = pooled_history_summary(...)`
3. 看 `outputs = model(...)`
4. 看 `page_loss`
5. 再决定是否继续看辅助变体逻辑

最核心的 3 行是：

- `pre_summary = pooled_history_summary(...)` [train_page_critic.py:263](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:263)
- `outputs = model(...)` [train_page_critic.py:265](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:265)
- `page_loss = masked_huber_loss(...)` [train_page_critic.py:273](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:273)

这一步 shape 关系最重要：

- `pre_summary`: `[B, hidden_size]`
- `token_ids`: `[B, S, D]`
- `q_values`: `[B, M]`，`M=ensemble_size`
- `q_target_all`: `[B, M]`

如果没有开辅助 loss，本质上这里就是一个 page-level value regression。

#### 第五步：看辅助变体是怎么做的

如果 `critic_item_loss_scale` 或 `critic_prefix_loss_scale` 大于 0，就会进 [has_auxiliary_losses()](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:167) 分支。

这一段最值得先读 [build_variant_batch()](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:179)。

它会把 `[B,S,D]` 的原始 page batch 展开成一个更大的变体 batch：

- item null variant：删掉某个 item
- prefix variant：保留某个 item 的前 k 个 token

建议直接打印：

```python
variant_pack = build_variant_batch(token_ids, item_mask)
if variant_pack is not None:
    variant_token_ids, variant_item_mask, variant_owner_idx, item_null_indices, prefix_indices = variant_pack
    print("variant_token_ids", variant_token_ids.shape)
    print("variant_item_mask", variant_item_mask.shape)
    print("variant_owner_idx", variant_owner_idx.shape)
    print("item_null_indices", item_null_indices.shape)
    print("prefix_indices", prefix_indices.shape)
```

预期：

- `variant_token_ids`: `[N_var, S, D]`
- `variant_item_mask`: `[N_var, S]`
- `variant_owner_idx`: `[N_var]`
- `item_null_indices`: `[B, S]`
- `prefix_indices`: `[B, S, D]`

然后再看：

- `item_delta_mean = q_pred.unsqueeze(1) - item_null_q_mean` [train_page_critic.py:333](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:333)
- `sid_delta_mean[:, :, sid_idx] = cur_q_mean - prev_q_mean` [train_page_critic.py:349](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:349)

这两行就是后面 item advantage / SID prefix advantage 差分学习的训练版。

#### 第六步：看训练主循环

训练主循环在 [run_training()](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:480)。

建议只抓 4 个节点：

1. build samples
2. load frozen TIGER encoder
3. build critic
4. epoch train + valid + save best

其中一个重要细节是：

- critic 自己训练
- TIGER 只拿来做 `pooled_history_summary`
- TIGER 参数被冻结不更新

对应 [train_page_critic.py:507](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:507)

### 15.2 `build_sid_advantage_chain.py` 阅读顺序

建议顺序：

1. `parse_args()`
2. `build_page_variants()`
3. `evaluate_q_variants()`
4. `main()`

#### 第一步：先看变体怎么构造

`build_page_variants()` 在 [build_sid_advantage_chain.py:86](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/build_sid_advantage_chain.py:86)。

它对单个 page 的 `token_ids [S,D]` 生成：

- 原始完整 slate 1 个
- 每个 item 一个 null variant
- 每个 item 的每个有效 SID prefix 一个 prefix variant

建议直接打印：

```python
variants, item_null_indices, sid_prefix_indices, valid_lengths = build_page_variants(token_ids)
print("token_ids", token_ids.shape)
print("variants", variants.shape)
print("len(item_null_indices)", len(item_null_indices))
print("valid_lengths", valid_lengths)
```

这里是理解归因最核心的一步。

#### 第二步：看批量 critic 评估

`evaluate_q_variants()` 在 [build_sid_advantage_chain.py:51](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/build_sid_advantage_chain.py:51)。

要点：

- 输入不是单条 token 序列，而是一批 `variant_token_ids [N_var,S,D]`
- `pre_summary`、`page_features`、`user_features` 用 `expand` 复制到每个变体

建议打：

```python
print("pre_summary", pre_summary.shape)
print("page_features", page_features.shape)
print("user_features", user_features.shape)
print("variant_token_ids", variant_token_ids.shape)
```

扩展后喂 critic 的逻辑非常直观：

- page context 固定
- 改变 item/token 组合
- 观察 `Q` 怎么变

#### 第三步：看差分归因主循环

主循环在 [build_sid_advantage_chain.py:105](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/build_sid_advantage_chain.py:105)。

推荐按这个顺序看：

1. 载入 trace 并复用 `build_page_samples()`
2. 加载 frozen TIGER 和 frozen critic
3. 对每个 page 构造 variants
4. 取 `full_q`
5. 对每个 item 取 `q_without_item`
6. 对每个 prefix 取 `prefix_q`
7. 用差分得到 `item_adv` 和 `sid_adv`
8. 写 JSONL

最关键的公式都可以直接在代码里看到：

- `item_adv = full_q - q_without_item` [build_sid_advantage_chain.py:204](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/build_sid_advantage_chain.py:204)
- `sid_adv[sid_idx] = prefix_q - prev_q` [build_sid_advantage_chain.py:221](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/build_sid_advantage_chain.py:221)

如果只想验证这一步没读错，最有用的 debug 是：

```python
print("full_q", full_q)
print("q_without_item", q_without_item)
print("item_adv", item_adv)
print("sid_adv", sid_adv)
print("sid_sum", sum(sid_adv))
```

正常情况下：

- `sum(sid_adv)` 应该接近 `item_adv`

代码里已经有这个一致性统计：

- `sid_item_cons_mae`
- `sid_item_cons_pess_mae`

### 15.3 `train_tiger_hca_grpo_actor.py` 阅读顺序

建议顺序：

1. `parse_args()`
2. `load_group_rows()`
3. `collate_rows()`
4. `build_effective_advantages()`
5. `compute_grpo_loss()`
6. `forward_actor()`
7. `main()`

#### 第一步：先看 actor 吃什么数据

`load_group_rows()` 在 [train_tiger_hca_grpo_actor.py:133](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:133)。

这里有一个重要认知：

- actor 已经不看原始 trace 了
- actor 直接看 group builder 产出的监督 JSONL

读取的最关键字段：

- `input_ids`
- `attention_mask`
- `target_tokens`
- `token_adv`
- `item_adv`
- `group_adv`
- `page_reward`
- `trust_support`
- `trust_unc`

#### 第二步：先看 batch shape，再看 loss

`collate_rows()` 在 [train_tiger_hca_grpo_actor.py:102](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:102)。

建议先打：

```python
print("input_ids", batch["input_ids"].shape)
print("target_tokens", batch["target_tokens"].shape)
print("token_adv", batch["token_adv"].shape)
print("item_adv", batch["item_adv"].shape)
print("group_adv", batch["group_adv"].shape)
print("page_reward", batch["page_reward"].shape)
```

预期：

- `input_ids`: `[B, L_hist]`
- `target_tokens`: `[B, D]`
- `token_adv`: `[B, D]`
- `item_adv`: `[B]`
- `group_adv`: `[B]`
- `page_reward`: `[B]`

#### 第三步：优先精读 `build_effective_advantages()`

这是 actor 文件里最重要的函数，位置在 [train_tiger_hca_grpo_actor.py:230](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:230)。

建议读法：

1. 看 clip + renorm
2. 看正负 token top-k mask
3. 看 `page_gate`
4. 分别看 `additive_zero_sum` 和 `multiplicative`

最适合加的 debug：

```python
print("token_adv", token_adv.shape)
print("item_adv", item_adv.shape)
print("group_adv", group_adv.shape)
print("pos_mask", pos_mask.shape, pos_mask.sum(dim=-1))
print("neg_mask", neg_mask.shape, neg_mask.sum(dim=-1))
print("page_gate", page_gate.shape, page_gate[:5].view(-1))
print("effective_adv", effective_adv.shape)
print("weights", weights.shape)
```

关注点：

- `page_gate` 是否经常打到上限或下限
- `pos_mask` / `neg_mask` 是否几乎全空
- `effective_adv` 是否过稀或过大

#### 第四步：再看 PPO/GRPO 损失

`compute_grpo_loss()` 在 [train_tiger_hca_grpo_actor.py:315](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:315)。

建议先只盯以下几行：

- `actor_target_logp` / `old_target_logp`
- `ratio`
- `clipped_ratio`
- `policy_loss`
- `kl_loss`
- `loss`

最适合打印：

```python
print("actor_logits", actor_logits.shape)
print("old_logits", old_logits.shape)
print("actor_target_logp", actor_target_logp.shape)
print("ratio", ratio.shape, ratio.mean().item())
print("effective_clip_eps", effective_clip_eps.shape)
print("active_mask", active_mask.shape, active_mask.float().mean().item())
```

预期：

- `actor_logits`: `[B, D, V]`
- `old_logits`: `[B, D, V]`
- `actor_target_logp`: `[B, D]`
- `ratio`: `[B, D]`
- `effective_clip_eps`: `[B, D]`
- `active_mask`: `[B, D]`

#### 第五步：最后再看主训练循环

`main()` 在 [train_tiger_hca_grpo_actor.py:522](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:522)。

建议只抓住这 5 步：

1. load grouped rows
2. load old policy
3. load actor init policy
4. set train scope
5. epoch train + valid + save best

一个阅读上很重要的点是：

- `old_tiger` 是冻结的 rollout / reference policy
- `actor_tiger` 是当前要更新的 learner

这也是为什么这里要同时 forward 两个 TIGER。

---

## 16. 最小化插桩建议

如果你下一步准备自己加 debug，我建议优先在下面 3 个点插桩，收益最高。

### 16.1 Critic 训练

位置：

- [train_page_critic.py:251](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py:251)

建议打印：

```python
print("pre_input_ids", pre_input_ids.shape)
print("token_ids", token_ids.shape)
print("item_mask valid avg", item_mask.float().sum(dim=-1).mean().item())
print("q_target", q_target.shape, q_target[:4])
```

### 16.2 Advantage Chain

位置：

- [build_sid_advantage_chain.py:175](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/build_sid_advantage_chain.py:175)

建议打印：

```python
print("pre_summary", pre_summary.shape)
print("variants", variants.shape)
print("full_q", full_q, "full_q_pess", full_q_pess)
```

### 16.3 Actor 更新

位置：

- [train_tiger_hca_grpo_actor.py:354](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py:354)

建议打印：

```python
print("effective_adv", effective_adv.shape, effective_adv.abs().mean().item())
print("weights", weights.shape, weights.sum(dim=-1)[:4])
print("page_gate", page_gate[:4].view(-1))
print("trust_support", trust_support[:4].view(-1))
print("trust_unc", trust_unc[:4].view(-1))
```

如果这几组数先看顺了，后面再读更复杂的 loss 细节会轻松很多。
