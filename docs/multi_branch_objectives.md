# Multi-Branch Retrieval: 任务目标与约束

## 目标
- 在原始 CLIP4Clip 全局匹配分支之外，加入 `entity` 与 `action` 两个匹配分支。
- 两个新增分支都可独立开关控制：
  - `--disable_entity_branch`
  - `--disable_action_branch`
- 当两个开关都关闭时，训练与评测行为应退化为原始 CLIP4Clip 的全局分支逻辑。

## 约束
- 环境统一使用 `conda clip`。
- `entity/action` 文本只能使用离线抽取缓存（JSON），不允许在线抽取兜底。
- 评测仍保留固定权重融合网格搜索，但分支关闭时其相似度矩阵置零，不影响全局分支排名。

## 数据集文本来源（与当前 dataloader 一致）
- `msvd`
  - 文本来源: `msvd_<subset>.json` 的 `caption` 字段（字符串或字符串列表）。
- `msrvtt`
  - 训练: `--data_path` 指向的 train json 中 `sentences[*].caption`（可按 train csv 的 `video_id` 过滤）。
  - 验证/测试: 对应 csv 的 `sentence` 列。
- `lsmdc`
  - 文本来源: `LSMDC16_annos_training.csv` / `LSMDC16_annos_val.csv` / `LSMDC16_challenge_1000_publictect.csv` 的第 6 列句子。
- `activity`
  - 文本来源: `train.json` / `val_1.json` 的 `sentences` 列表拼接成单句。
- `didemo`
  - 文本来源: `train_data.json` / `val_data.json` / `test_data.json`，按 `video` 聚合后将 `description` 拼接成单句。

## 离线抽取输出格式
每个文件名：`<dataset>_<subset>_text_branches.json`

```json
{
  "meta": {
    "dataset": "msvd",
    "subset": "val",
    "ordered": true,
    "num_records": 4290,
    "source_script": "dataloaders/build_msvd_ordered_text_branches.py"
  },
  "records": [
    {
      "idx": 0,
      "video_id": "video_xxx",
      "caption": "a man is playing guitar",
      "entity_text": "man and guitar",
      "action_text": "man playing guitar",
      "entity_fallback": 0,
      "action_fallback": 0
    }
  ]
}
```

- `records[idx]` 必须与 dataloader 的 `(video_id, caption)` 顺序严格对齐，保证训练/评测配对关系不被打乱。

## 运行方式
- 离线抽取（示例）
  - `conda activate allennlp-srl`
  - `PYTHONPATH=/data/jzw/CLIP4Clip-auxiliary python -m dataloaders.build_msvd_ordered_text_branches --data_path /path/to/msvd --subsets train,val,test --output_dir /path/to/msvd --output_suffix _text_branches.json --allennlp_srl_cuda_device 0,1,6,7`
- 训练/评测使用离线缓存目录
  - `--text_branch_cache_path /path/to/branch_cache`
  - 系统会自动按 `<dataset>_<subset>_text_branches.json` 读取。
  - 可在启动前执行对齐检查：`python -m dataloaders.verify_msvd_ordered_text_branches --data_path /path/to/msvd --cache_dir /path/to/msvd --subsets train,val,test`

## Entity 分支（ProST风格实现要点）
- 文本侧：对 `entity_text` 的 token hidden 使用可学习 `word weight`，聚合为多组 `word prototypes`。
- 视频侧：对每帧 patch tokens 使用可学习 `patch weight`，形成每帧 `patch prototypes`（含 CLS 原型）。
- 相似度：采用 ProST 风格原型匹配聚合
  - token-原型与patch-原型两两相似；
  - 对 `word prototype` 维度取 max；
  - 对时间帧维度取 max（考虑 `video_mask`）；
  - `patch prototype` 维度不取 max，而是求和/平均聚合（与 ProST 聚合逻辑一致）；
  - 因此并不是 `word` 和 `patch` 两个维度都取 max。
