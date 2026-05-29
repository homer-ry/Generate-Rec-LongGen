
优化1 多Critic设计
ensemble critic 由 M 个独立 head 组成，每个 head 共享同一个backbone
具体代码位置:
PageSIDQCriticV9Additive（critic类定义）+ PageSIDQCriticEnsemble（把critic给集成）
code/tiger_page_sid_rl/models.py
● PageSIDQCriticV9Additive
● PageSIDQCriticEnsemble


优化2 归因

核心是三层 credit assignment：

具体归因代码位置：https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/tiger_page_sid_rl/build_sid_advantage_chain.py

归因信号计算一共有两个版本，：
sid_adv[sid_idx] = prefix_q - prev_q
sid_adv_pess[sid_idx] = prefix_q_pess - prev_q_pess


实际 actor 用的 token 优势是三层信号的合成：
 
具体整合归因代码位置：
code/tiger_page_sid_rl/build_sid_advantage_chain.py
函数位置：build_effective_advantages()
page的优势决定更新方向（打压或者提升），item和token的优势决定更新幅度
核心代码：
attr_scores = token_adv + item_adv_scale * item_adv.unsqueeze(-1)
attr_residual = (attr_scores - attr_mean) * weight_mask
list_component = group_adv.unsqueeze(-1) / active_count
effective_adv = page_gate * (
    list_component + hca_residual_scale * attr_residual
)


优化3 ppo训练
当前 repo 里实际是 GRPO-style clipped PPO
具体代码位置：https://github.com/kaifengGuo/long-term-Generate-Recommendation/blob/main/code/train_tiger_hca_grpo_actor.py

