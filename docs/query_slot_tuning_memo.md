# Query-Slot 调参备忘（2026-03-01）

本文档记录最近关于 `query-slot` 分支的设计决策、已实现项、风险与调参建议，避免后续反复讨论或遗漏。

## 1. 目标与问题背景

- 目标：让 `entity/action` 分支真正对检索有稳定边际增益，而非融合时被 `global` 完全压制。
- 核心想法：
  - `entity` 对齐对象槽位（近似 box/tube 概念）；
  - `action` 对齐对象时序变化（diff 轨迹特征）。
- 关键担心：三个文本分支共用同一文本编码器，若不做分支特化会导致对齐对象漂移。

## 2. 当前实现状态（已完成）

### 2.1 Query-slot 主分支

- 已支持开关：`--enable_query_slot_branch`
- 已支持参数：
  - `--slot_num_queries`
  - `--slot_topk_entity`
  - `--slot_topk_action`
  - `--slot_branch_layers`
  - `--lambda_slot_diversity`
  - `--lambda_slot_balance`

### 2.2 规范性增强（结构完整性）

- 已加 `slot attention dropout`：`--slot_attn_dropout`
- 已加 `post-FFN residual block`（可关闭）：`--disable_slot_post_ffn`
- 现有结构满足：归一化、FFN、dropout、残差（slot 后处理与分支 text adapter 均有）。

### 2.3 文本分支特化（共享编码器风险缓解）

- 已加分支专属文本适配器（默认启用）：
  - `entity_text_adapter`
  - `action_text_adapter`
- 开关：`--disable_branch_text_adapter`

### 2.4 分支温度（logit scale）

- 当前为三路独立温度：
  - `global`: `clip.logit_scale`
  - `entity`: `entity_logit_scale`
  - `action`: `action_logit_scale`
- 训练时均已做 `clamp(max=log(100))` 防发散。

## 3. 当前相似度计算（query-slot 路径）

给定文本向量 $t_a$、视频 $b$ 的 slot 特征 $s_{b,m}$：

1) 先算 pair-wise slot 相似度：
$$
\mathrm{sim}_{a,b,m}=\langle t_a,s_{b,m}\rangle
$$

2) 取 top-k 槽位并 softmax 权重：
$$
w_{a,b,m}=\mathrm{softmax}(\mathrm{topk}(\mathrm{sim}_{a,b,:}))
$$

3) 融合得到 pair-specific video 向量：
$$
\hat v_{a,b}=\sum_{m\in\mathrm{topk}} w_{a,b,m}s_{b,m}
$$

4) 最终 logits：
$$
\ell^{entity}_{a,b}=\tau_e\langle t^{entity}_a,\hat v^{entity}_{a,b}\rangle,\quad
\ell^{action}_{a,b}=\tau_a\langle t^{action}_a,\hat v^{action}_{a,b}\rangle
$$

其中 $\tau_e,\tau_a$ 分别为 entity/action 独立温度。

## 4. 风险与缓解

### 风险 A：slot 坍缩（多个 slot 学到同一区域）
- 缓解：保留 `slot_diversity` + `slot_balance` 正则；注意不要把权重设太小。

### 风险 B：共享文本编码导致分支语义混叠
- 缓解：保持分支 text adapter 开启；必要时增大 adapter dropout 到 0.15。

### 风险 C：action 分支波动大
- 缓解：独立 `action_logit_scale` 已启用；必要时降低 `slot_topk_action`（2 -> 1）稳定训练。

### 风险 D：算力增加
- 缓解：优先控制 `slot_num_queries`（建议 6~8）与 `slot_branch_layers`（建议 1）。

## 5. 建议默认配置（首轮）

- `--enable_query_slot_branch`
- `--slot_num_queries 8`
- `--slot_topk_entity 3`
- `--slot_topk_action 2`
- `--slot_branch_layers 1`
- `--slot_attn_dropout 0.1`
- `--lambda_slot_diversity 0.02`
- `--lambda_slot_balance 0.01`
- 保持 text adapter 开启（不要加 `--disable_branch_text_adapter`）
- 保持 slot post-FFN 开启（不要加 `--disable_slot_post_ffn`）

## 6. 建议消融矩阵（最小必要）

1) `baseline`：无 query-slot（现有 xpool 路径）  
2) `+slot`：开 query-slot，不开 text adapter（验证混叠风险）  
3) `+slot+adapter`：开 query-slot + text adapter  
4) `+slot+adapter+attn_dropout+post_ffn`（当前推荐主配置）

每组至少记录：
- 四分支 `R@1/R@5/R@10`
- 四分支融合最优权重（wg/we/wa/ws）
- 训练日志中的 `Lsd/Lsb`（slot 正则是否有效）

## 7. 下一步未完成事项（重要）

- 尚未实现 `entity-guided action mask`：
  - 目标：action 仅在 entity 命中槽位上计算，进一步缓解对齐对象漂移。
  - 这是下一步优先级最高的结构改进。
