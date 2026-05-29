# Motivation

## 核心问题

生成式推荐里，actor 在做的是 **token 级生成**，但长期价值通常只能在 **page/slate 级** 观测到。

这会带来一个直接问题：

- critic 告诉我们一整页推荐“值不值”
- actor 真正更新的却是每一个 SID token 的生成概率
- 如果把整页 reward 直接平均分给所有 token，监督会非常粗，难以知道到底是哪个 item、哪个 token 真正在提升或伤害长期价值

所以需要做 **层级 credit assignment**，把 page 级长期价值拆解成更细粒度的训练信号。

## 核心想法

这套方法的本质可以概括成一句话：

**用 critic 的差值，来近似 item 和 token 的边际贡献。**

也就是说，不直接凭启发式给 token 打分，而是看：

- 去掉某个 item 之后，整页 Q 值下降了多少
- 增加一个 SID token 前后，prefix Q 值变化了多少

这些“前后差值”就是 credit assignment 的来源。

## 为什么要分层

这里的 credit 被拆成三层：

- `page/group advantage`：决定这条样本整体应该被提升还是打压
- `item advantage`：决定这一页里哪个 item 更值得被强化
- `sid/token advantage`：决定一个 item 的 SID 序列中，哪些 token 更关键

三层信号各司其职：

- page 级信号负责更新方向
- item/token 级信号负责更新位置和幅度

这样可以避免“整页好坏”被粗暴地均摊到所有 token 上。

## 直观理解

可以把 critic 看成一个长期价值打分器：

- `Q(full slate)` 表示整页候选的长期价值
- `Q(without item i)` 表示去掉 item `i` 后的长期价值
- `Q(prefix k)` 表示只生成到第 `k` 个 SID token 时的长期价值

于是就有：

- item 边际贡献 ≈ `Q(full slate) - Q(without item i)`
- token 边际贡献 ≈ `Q(prefix k) - Q(prefix k-1)`

所以文档里说“其实就相当于 critic 差值”，本质上就是把 critic 从“总分器”变成“边际贡献估计器”。

## 想解决的训练痛点

这套设计主要在解决三个问题：

1. 长期 reward 稀疏，只在 session 后面才显现
2. actor 的动作空间很细，是 SID token 序列而不是单个 item id
3. 同一页里 item 的贡献不同，同一 item 内不同 token 的贡献也不同

如果没有层级归因，actor 很容易只收到一个非常粗的 page 级回报，训练信号会弱且不稳定。

## 和代码的对应关系

这份动机在项目里的落地链路是：

1. 先训练 critic，对 page / item / prefix 提供长期价值估计  
   [train_page_critic.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/train_page_critic.py)
2. 再用 critic 差分构造 `item_advantage` 和 `sid_advantage`  
   [build_sid_advantage_chain.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/tiger_page_sid_rl/build_sid_advantage_chain.py)
3. 最后把 page/item/token 三层信号融合成 actor 使用的 `effective_advantage`，做 GRPO-style 更新  
   [train_tiger_hca_grpo_actor.py](/Users/rongyu/Documents/develop/model/long-term-Generate-Recommendation/code/train_tiger_hca_grpo_actor.py)

## 一句话总结

**因为 actor 在 token 级生成，而长期价值只在 page 级显现，所以必须用 critic 差分把总价值拆成 item 级和 token 级边际贡献，才能把长期奖励有效传回生成过程。**
