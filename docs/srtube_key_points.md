# SRTube 关键内容速记（CVPR 2024）

对应论文：`Lee_SRTube_Video-Language_Pre-Training_with_Action-Centric_Video_Tube_Features_and_Semantic_CVPR_2024_paper.pdf`

本文档用于“防遗忘”与“实现对照”，重点记录 SRTube 为什么有效、关键模块、关键公式与可迁移点。

## 1) 核心思想（一句话）

SRTube 的提升不是单靠 `box 检测`，而是将**视频侧动作轨迹（tube）**与**文本侧语义角色（SRL）**同时结构化，再通过专门的跨模态注意力与动作代理任务进行对齐。

---

## 2) 两条特征流

### 视频侧：Tube features

- 目标：建模“同一对象跨时间的轨迹”，而不是仅单帧区域。
- 论文记号：
  - $U \in \mathbb{R}^{N \times K \times D}$：tube 特征序列（沿对象轨迹的时序表征）。
  - 其中 $N$ 对应 tube/query 数（可理解为可跟踪对象槽位数），$K$ 与时间/片段相关，$D$ 为特征维。
- Tube builder 先做动作相关预训练（文中使用 AVA v2），学习更稳定的动作区域表征。

### 文本侧：SRL semantic phrases

- 目标：把句子拆成动作相关语义结构，而非只用全句 embedding。
- SRL 将词/短语打上语义角色：
  - `VERB`：动作词
  - `ARG0`：施事（谁做）
  - `ARG1`：受事（对谁/对什么做）
- 由 SRL 构建动作语义短语（典型是 `ARG0-VERB-ARG1`），形成语义短语特征 $S$。

---

## 3) Cross Fusion Transformer (CFT) 的两种注意力

SRTube 在 CFT 中做两类融合：全局对齐 + 动作对齐。

### (1) Visual-to-Text Attention (VTA)

- 用全局视频特征 $V$ 与全局文本特征 $T$ 生成全局融合特征 $M_G$：

$$
M_G = \mathrm{MHA}(Q=V,\ K=T,\ V=T) \tag{1}
$$

### (2) Tube-to-SRL Attention (TSA)

- 用 tube 特征 $U$ 与 SRL 语义特征 $S$ 生成动作融合特征 $M_L$：

$$
M_L = \mathrm{MHA}(Q=U,\ K=S,\ V=S) \tag{2}
$$

这一步是 SRTube 与普通全局 VidLP 的关键差异：显式做“动作轨迹 ↔ 语义角色”对齐。

---

## 4) 预训练目标（5 个 proxy tasks）

论文总共使用 5 个任务：

- 通用任务：`MLM`、`VTM`、`VTC`
- 新增动作任务：`MAM`、`ANM`

### (1) Masked Action Modeling (MAM)

- 只对 SRL 中 `VERB` 相关 token 做 mask，迫使模型用 $U$ 与掩码后的语义短语 $S_m$ 恢复动作词。
- 论文公式：

$$
\mathcal{L}_{\mathrm{MAM}} = \mathbb{E}_{(U,S_m)\sim \mathcal{D}}\,\mathcal{H}\big(y_v\mid p_m(U,S_m)\big) \tag{3}
$$

其中 $\mathcal{H}$ 为交叉熵，$y_v$ 为被 mask 的动词标签。

### (2) Action Numbering Modeling (ANM)

- 用 SRL 统计句子动作数（动词个数）作为伪标签 $y_n$，预测动作强度/复杂度。
- 论文公式：

$$
\mathcal{L}_{\mathrm{ANM}} = \mathrm{MSE}(y_n, p_a) \tag{4}
$$

该项可缓解“只偏好强运动场景”的 bias，对静态场景（$y_n=0$）有约束作用。

### (3) 总损失

$$
\mathcal{L}_{\mathrm{all}}=\mathcal{L}_{\mathrm{MLM}}+\mathcal{L}_{\mathrm{VTM}}+\mathcal{L}_{\mathrm{VTC}}+\mathcal{L}_{\mathrm{MAM}}+\mathcal{L}_{\mathrm{ANM}} \tag{5}
$$

---

## 5) 为什么 SRL 能增强（结论）

不是因为“用了 box 检测器就必然变强”，而是因为：

1. **文本可对齐性更强**：SRL 显式给出谁-做什么-作用于谁，减少 caption 全句 embedding 的语义混叠。  
2. **视频时序一致性更强**：tube 关注对象轨迹与动作区域，不是单帧局部。  
3. **对齐机制更匹配**：TSA 直接对齐 $U$ 与 $S$，不是只做 global pooling。  
4. **训练信号更动作化**：MAM/ANM 明确把“动作词”和“动作证据”绑定。

---

## 6) 对本仓库实现的对照提醒

若要接近 SRTube 的收益，优先检查这 4 点是否齐全：

1. 是否有“tube 级时序对象特征”（而非仅 frame/patch 平均）。
2. 是否有“SRL 语义角色短语”（而非仅 entity/action 关键字拼接）。
3. 是否有“Tube-to-SRL 专门注意力”路径（等价 TSA）。
4. 训练是否包含动作代理损失（至少类 MAM、ANM）。

若缺其中任意两项，通常会出现：分支单独指标可升，但融合最优权重仍偏向 global。
