prost entity patch（含 CLS 输入权重、再拼 CLS、无 decoder 参与 word-patch 匹配）
2026-02-28 12:42:05,885:INFO: Branch Text-to-Video:
2026-02-28 12:42:05,885:INFO: 	Global >>> R@1: 45.3 - R@5: 75.2 - R@10: 84.2
2026-02-28 12:42:05,885:INFO: 	Entity >>> R@1: 28.7 - R@5: 57.1 - R@10: 69.8
2026-02-28 12:42:05,885:INFO: 	Action >>> R@1: 1.0 - R@5: 2.8 - R@10: 4.1
2026-02-28 12:45:13,223:INFO: Fixed-weight Fusion:
2026-02-28 12:45:13,223:INFO: 	Best Weights (R@1 first): wg=0.90, we=0.10, wa=0.00
2026-02-28 12:45:13,223:INFO: 	>>>  R@1: 45.6 - R@5: 75.4 - R@10: 84.4 - Median R: 2.0 - Mean R: 10.2
2026-02-28 12:45:13,223:INFO: Video-to-Text (Fusion):
2026-02-28 12:45:13,223:INFO: 	>>>  V2T$R@1: 41.8 - V2T$R@5: 62.8 - V2T$R@10: 67.6 - V2T$Median R: 2.0 - V2T$Mean R: 15.0
2026-02-28 12:45:13,223:INFO: Eval fallback rate: entity=0.0042, action=0.0613
2026-02-28 12:45:13,251:INFO: Final-epoch Eval R@1: 45.5606

entity单独编码
2026-02-28 14:07:04,272:INFO: Branch Text-to-Video:
2026-02-28 14:07:04,273:INFO: 	Global >>> R@1: 44.8 - R@5: 74.8 - R@10: 83.8
2026-02-28 14:07:04,273:INFO: 	Entity >>> R@1: 19.0 - R@5: 46.1 - R@10: 59.0
2026-02-28 14:07:04,273:INFO: 	Action >>> R@1: 0.9 - R@5: 2.7 - R@10: 4.2
2026-02-28 14:10:12,320:INFO: Fixed-weight Fusion:
2026-02-28 14:10:12,320:INFO: 	Best Weights (R@1 first): wg=0.80, we=0.10, wa=0.10
2026-02-28 14:10:12,320:INFO: 	>>>  R@1: 45.0 - R@5: 75.1 - R@10: 83.8 - Median R: 2.0 - Mean R: 10.4
2026-02-28 14:10:12,321:INFO: Video-to-Text (Fusion):
2026-02-28 14:10:12,321:INFO: 	>>>  V2T$R@1: 53.4 - V2T$R@5: 78.0 - V2T$R@10: 85.2 - V2T$Median R: 1.0 - V2T$Mean R: 7.9
2026-02-28 14:10:12,321:INFO: Eval fallback rate: entity=0.0042, action=0.0613
2026-02-28 14:10:12,352:INFO: Final-epoch Eval R@1: 44.9915

entity + caption CLS 融合
2026-02-28 15:14:55,691:INFO: Branch Text-to-Video:
2026-02-28 15:14:55,691:INFO: 	Global >>> R@1: 45.7 - R@5: 75.5 - R@10: 84.8
2026-02-28 15:14:55,692:INFO: 	Entity >>> R@1: 27.3 - R@5: 59.9 - R@10: 73.0
2026-02-28 15:14:55,692:INFO: 	Action >>> R@1: 0.8 - R@5: 2.0 - R@10: 3.5
2026-02-28 15:18:02,685:INFO: Fixed-weight Fusion:
2026-02-28 15:18:02,686:INFO: 	Best Weights (R@1 first): wg=0.80, we=0.10, wa=0.10
2026-02-28 15:18:02,686:INFO: 	>>>  R@1: 46.0 - R@5: 75.9 - R@10: 84.9 - Median R: 2.0 - Mean R: 9.9
2026-02-28 15:18:02,686:INFO: Video-to-Text (Fusion):
2026-02-28 15:18:02,686:INFO: 	>>>  V2T$R@1: 48.1 - V2T$R@5: 71.9 - V2T$R@10: 78.6 - V2T$Median R: 2.0 - V2T$Mean R: 12.5
2026-02-28 15:18:02,686:INFO: Eval fallback rate: entity=0.0042, action=0.0613
2026-02-28 15:18:02,717:INFO: Final-epoch Eval R@1: 45.9893

entity template单独编码
2026-02-28 15:31:14,702:INFO: Branch Text-to-Video:
2026-02-28 15:31:14,702:INFO: 	Global >>> R@1: 45.6 - R@5: 74.9 - R@10: 84.1
2026-02-28 15:31:14,702:INFO: 	Entity >>> R@1: 23.2 - R@5: 49.8 - R@10: 63.2
2026-02-28 15:31:14,702:INFO: 	Action >>> R@1: 0.9 - R@5: 2.2 - R@10: 3.3
2026-02-28 15:34:21,628:INFO: Fixed-weight Fusion:
2026-02-28 15:34:21,628:INFO: 	Best Weights (R@1 first): wg=0.90, we=0.10, wa=0.00
2026-02-28 15:34:21,628:INFO: 	>>>  R@1: 45.7 - R@5: 75.2 - R@10: 84.3 - Median R: 2.0 - Mean R: 10.4
2026-02-28 15:34:21,628:INFO: Video-to-Text (Fusion):
2026-02-28 15:34:21,629:INFO: 	>>>  V2T$R@1: 51.3 - V2T$R@5: 75.7 - V2T$R@10: 81.2 - V2T$Median R: 1.0 - V2T$Mean R: 10.3
2026-02-28 15:34:21,629:INFO: Eval fallback rate: entity=0.0042, action=0.0613
2026-02-28 15:34:21,660:INFO: Final-epoch Eval R@1: 45.7371

按 and 分 entity + CLS 融合 + CLS拼接到proto序列首位
2026-02-28 17:05:52,179:INFO: Branch Text-to-Video:
2026-02-28 17:05:52,180:INFO: 	Global >>> R@1: 43.7 - R@5: 73.6 - R@10: 82.7
2026-02-28 17:05:52,180:INFO: 	Entity >>> R@1: 33.7 - R@5: 65.1 - R@10: 76.5
2026-02-28 17:05:52,180:INFO: 	Action >>> R@1: 1.7 - R@5: 4.6 - R@10: 7.1
2026-02-28 17:08:59,117:INFO: Fixed-weight Fusion:
2026-02-28 17:08:59,117:INFO: 	Best Weights (R@1 first): wg=0.30, we=0.70, wa=0.00
2026-02-28 17:08:59,117:INFO: 	>>>  R@1: 44.1 - R@5: 73.9 - R@10: 82.9 - Median R: 2.0 - Mean R: 11.0
2026-02-28 17:08:59,117:INFO: Video-to-Text (Fusion):
2026-02-28 17:08:59,117:INFO: 	>>>  V2T$R@1: 38.4 - V2T$R@5: 58.8 - V2T$R@10: 63.3 - V2T$Median R: 3.0 - V2T$Mean R: 21.3
2026-02-28 17:08:59,117:INFO: Eval fallback rate: entity=0.0042, action=0.0613
2026-02-28 17:08:59,148:INFO: Final-epoch Eval R@1: 44.0550

用 global/entity/action/struct 四分支联合检索，其中 entity 与 action 采用 text-conditioned xpool 纯算子（相似度、softmax、加权求和）视频对齐、struct 用文本侧与视频侧各自两层 Transformer 做分支级结构对齐（global entity action各占一个槽位），
2026-03-01 01:15:08,555:INFO: Branch Text-to-Video:
2026-03-01 01:15:08,555:INFO: 	Global >>> R@1: 44.9 - R@5: 74.8 - R@10: 83.6
2026-03-01 01:15:08,556:INFO: 	Entity >>> R@1: 30.8 - R@5: 60.2 - R@10: 71.7
2026-03-01 01:15:08,556:INFO: 	Action >>> R@1: 33.5 - R@5: 63.3 - R@10: 74.3
2026-03-01 01:15:08,556:INFO: 	Struct >>> R@1: 36.5 - R@5: 69.1 - R@10: 80.2
2026-03-01 01:29:11,639:INFO: Fixed-weight Fusion:
2026-03-01 01:29:11,639:INFO: 	Best Weights (R@1 first): wg=0.70, we=0.20, wa=0.00, ws=0.10
2026-03-01 01:29:11,639:INFO: 	>>>  R@1: 45.5 - R@5: 75.6 - R@10: 84.2 - Median R: 2.0 - Mean R: 10.9
2026-03-01 01:29:11,639:INFO: Video-to-Text (Fusion):
2026-03-01 01:29:11,639:INFO: 	>>>  V2T$R@1: 39.4 - V2T$R@5: 67.0 - V2T$R@10: 74.5 - V2T$Median R: 2.0 - V2T$Mean R: 14.9
2026-03-01 01:29:11,639:INFO: Eval fallback rate: entity=0.0042, action=0.0613
2026-03-01 01:29:11,663:INFO: Final-epoch Eval R@1: 45.4634
