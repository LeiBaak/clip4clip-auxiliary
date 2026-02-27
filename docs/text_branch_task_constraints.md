# Text Branch Task Constraints (Single Source of Truth)

## 不可违反清单
- 只处理文本分支抽取任务：entity_text 与 action_text。
- 唯一实现文件是 dataloaders/text_branch_utils.py。
- 必须 SRL-first。
- 默认 SRL 后端为 AllenNLP SRL。
- 动作短语禁止开放词表模板拼接，必须由 SRL 核心角色聚合。
- ARGM（时间、地点、方式等状语）不得进入动作短语。
- 无可用 SRL frame 时，action_fallback 必须为 1，action_text 回退原句。
- entity 优先来自 SRL 核心 ARG，缺失时可用轻量通用名词规则兜底。
- 在线 dataloader 与离线缓存构建必须共用同一代码路径。
- 产生的文件只能写入工作区内路径（不得写到工作区外）。

## 环境约束
- 文本分支抽取与离线构建必须在 conda allennlp-srl 环境执行。
- 尽可能多使用 GPU；可用 GPU 固定为 0,1,6,7，离线构建默认应优先占满这 4 张卡（在资源冲突时再降级）。

## 动作短语聚合规则
- 主组合顺序：ARG0 + VERB + ARG1 + 可选 ARG2。
- 短语必须简洁、去重。

## 术语说明
- ARG0：通常是施事/执行者（谁在做）。
- ARG1：通常是受事/作用对象（做了什么到谁/什么）。
- ARG2：谓词相关的补足核心论元（因谓词而异，常见为工具、受益者、结果、第二对象等）。
- ARGM：附加状语成分（时间、地点、方式、原因等），当前任务中不纳入动作短语。

## 唯一入口
- 继续只在 dataloaders/text_branch_utils.py 上迭代。
- 离线构建入口：python dataloaders/text_branch_utils.py --data_path /data/jzw/msvd --subsets val --output_dir /data/jzw/CLIP4Clip-auxiliary/text_branches