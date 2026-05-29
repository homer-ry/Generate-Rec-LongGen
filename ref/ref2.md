生成式推荐长期价值技术细节

训练流程

预处理：
1.数据：
数据集介绍https://kuairand.com/
数据集可以直接在https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/dataset/kuairand/kuairand-Pure/data/video_features_basic_Pure_fillna.csv
https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/dataset/kuairand/kuairand-Pure/data/user_features_Pure_fillna.csv使用，与kuaisim框架同口径

2.用kuairand数据集来训练一个模拟环境，具体代码：
主训练脚本(通用入口)
https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/train_multibehavior.py
响应模型 (URM) 训练:多行为(点击/长播/关注/…)联合预测
https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/train_general_model.py
留存模型可以不用训练，这个版本并没有用到
模拟环境的模型类：
https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/model/simulator/KRMBUserResponse.py
多行为头预测 click/long_view/follow/：
https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/model/simulator/KRMBUserResponseWithBias.py

3.构建sid用于TIGER训练/推理：https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/build_pure_sid.py以及启动sid的sh：https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/build_kuairand_sid.sh
或者直接用：https://github.com/kaifengGuo/long-term-Generate-Recommendation/tree/main/code/dataset/kuairand/kuairand-Pure/sid/32_mask我已经跑完了sid构建


正式训练流程

1.Rollout
先用当前 TIGER/EMA rollout policy 在模拟环境里跑用户 session，收集 trace：
history -> slate -> user response -> reward -> next state
2.构造 critic 训练样本
从 rollout trace 里算每个 page/slate 的长期回报 target：
\[ Q_{\text{target}}(x,y)=\sum_{t'\ge t}\gamma^{t'-t}r_{t'} \]
同时构造 item 级、SID token 级辅助 target。
3.训练 ensemble critic
用 PageSIDQCriticV9Additive 训练多个 critic head/ensemble。输入是：\[ x=\text{user history/context},\quad y=\text{generated slate/SID tokens} \]

输出：
\[ \mu_Q(x,y),\quad \sigma_Q(x,y) \]

第一项是均值，第二项是方差。


4.构造 GRPO group
对同一个上下文生成一组候选：
\[ \{y_1,\dots,y_G\} \]

每个候选有一个 critic 分数 \(s_i\)，组内做 advantage

5.做层级归因
用 critic 做差分：
\[ A_{\text{item}} = Q(x,y)-Q(x,y_{\setminus i}) \]

对 SID/token 做 prefix 差分：
/
\[ A_{\text{sid},k} = Q(x,y_{\setminus i}+y_{i,1:k}) - Q(x,y_{\setminus i}+y_{i,1:k-1}) \]




长期价值模型代码
框架图：
 

模块1基础模型预训练：
https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/train_TIGER_krpure.py 
最核心的基础策略文件：
● 定义 TIGER 模型类
● 用 T5ForConditionalGeneration 作为生成 backbone
● 把 item 表示成 SID token sequence
● 构造训练样本：历史序列 → 目标 SID sequence
● 完成基础策略 π0​ 的训练

https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/eval_tiger_env.py
推理/评估时的实现：
● 加载基础 TIGER checkpoint
● 构造用户历史对应的 SID token 输入
● 调用 TIGER 进行 beam search 解码
● 把生成的 SID token 序列映射回 item
● 在环境里执行推荐动作

模块2候选构造代码细节：
https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/build_tiger_hca_grpo_groups.py
模块2在 GRPO 主线下的核心实现。
它负责：
● 给每个用户状态生成一个候选池
● 从中选出 group_size 个候选
● 把每条候选展开成 SID token 序列
● 输出一个分组 jsonl 文件供后续 actor 训练使用


模块3 多critic设计

https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/tiger_page_sid_rl/train_page_critic.py
模块3 最核心文件。
负责：
● 训练 page/item/SID 级 critic
● 支持 ensemble（pessimistic value + uncertainty）
● 多 loss 组合（page_loss / item_loss / prefix_loss / rank_loss / monotonic_loss）
	目前用的是v9add函数


模块4critic结构和训练，归因计算：


https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/tiger_page_sid_rl/build_sid_advantage_chain.py
层次化 attribution 的主实现。
负责：
● 给每个候选组算 page_q / item_adv / sid_advantage
● 输出 pessimistic / raw / mean 等不同口径的 advantage 字段
● 产出一条 "advantage chain" 供 actor 消费

https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/tiger_page_sid_rl/models.py
Page/Item/SID critic 的模型类定义（和 train_page_critic 配套）。

https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/tiger_page_sid_rl/common.py
Critic / attribution 的公共工具函数（特征组装、reward 聚合、support/uncertainty 等）。

代码片段



模块5 GRPO更新
https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/train_tiger_hca_grpo_actor.py
GRPO actor 更新代码


模拟环境KuaiSIM
介绍
kuaisim介绍：
https://zhuanlan.zhihu.com/p/660470451

细节代码位置：https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/env/KREnvironment_WholeSession_GPU.py
代码介绍
KREnvironment_WholeSession_GPU.py 是整个推荐闭环里的核心模拟环境实现。 它的职责不是训练模型，而是充当一个 GPU 上运行的推荐环境（environment），让策略模型可以像在线系统一样：
1. 读取当前用户状态
2. 给出一页推荐列表（slate）
3. 调用用户即时反馈模型（URM, user immediate response model）生成反馈
4. 更新用户历史
5. 计算用户是否离开会话
6. 返回新的 observation、反馈结果和辅助信息

实验效果

方法、指标、复现流程


模拟环境Agent4rec
介绍
agent4rec细节

● 论文：On Generative Agents in Recommendation（SIGIR 2024, Zhang 等）
● GitHub：LehengTHU/Agent4Rec
● 核心思路：用 LLM 当用户 agent，每个 agent 配一份 profile（taste / mood / activity），让 LLM 在推荐 slate 上自主决策"看 / 喜欢 / 不喜欢 / 退出"，再把这些行为当 simulator 的反馈


修改后的代码，复现需要调用大模型api
https://github.com/kaifengGuo/long-term-Generate-Recommendation/tree/main/code/external/Agent4Rec


实验效果
方法、指标、复现流程




