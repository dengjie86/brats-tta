# GLI→PED TTA研究证据

主路线已选定为 **union2补充teacher + soft_extent八步IN学习 + no_come输出**。算法、参数和正式结果统一在[方法文档](GLI_PED_TTA_METHOD.md)；任务运行状态和未完成事项只维护在[HANDOFF](../HANDOFF.md)。

## 最新适用范围限制（2026-10-22）

Senior188同66例中，用户方法平均Dice58.72396%，Source58.70143%，TENT62.67784%；方法队列已按用户要求暂停。teacher没有提供足够内部组织纠错，不能把历史IN收益推广到此BN模型。六例BN统计探测也没有建立稳定修复；见[诊断](../outputs/senior188_bn_diagnosis_20261022/REPORT.md)。这与旧native BN同24例的轻微收益是不同实验。

## 已建立的开发证据

- 98组主方法相对Source：WT/TC/ET Dice为−0.13202/+9.35708/+0.13691pp；WT仍有代价。相对旧soft_stable TC+0.96119pp。
- 匹配相同输出规则，学习本身带来TC+3.00925pp，新teacher进入学习额外+0.29916pp。见[最终控制](../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/INTERPRETATION.md)。
- 移除COME和冗余末端stable后的98份输出完成独立核验及缓存重放。见[交付](../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/delivery/README.md)。
- 跨Source使用相同24例、固定方法，按各自Source比较；已完成的BN阶段Dice见[阶段报告](../outputs/ped2023_cross_model_transfer_20261015/interim_readout/REPORT.md)，完整复核状态以HANDOFF为准。

## 已完成但未成为主路线的方向

| 方向 | 目前判断 | 结果入口 |
|---|---|---|
| 固定Source统计、停止统计梯度 | 改变收益与边界风险取舍，不能单因果归为统计漂移 | [统计回放](../outputs/ped2023_source_moment_replay_20261013/INTERPRETATION.md) |
| 原生view可靠性 | 未超过同数量confidence选择，不继续追加该门控 | [可靠性](../outputs/ped2023_context_view_reliability_20261014/INTERPRETATION.md) |
| encoder/decoder IN分区 | 两臂主要TC收益下降 | [分区](../outputs/ped2023_in_subspace_20261014/INTERPRETATION.md) |
| KL替换MSE | 同输出下TC下降 | [概率蒸馏](../outputs/ped2023_probability_distillation_20261014/INTERPRETATION.md) |
| 短步数与最低loss选回 | WT/ET小收益伴TC代价，未成为替代方案 | [loss/步数](../outputs/ped2023_loss_checkpoint_20261014/INTERPRETATION.md) |
| 分模态均值teacher | 直接均值损失旧正确编辑，后续转为保留旧编辑再补充 | [背景观察](../outputs/ped2023_modality_context_20261014/INTERPRETATION.md) |

完整实验和前序路线见[实验索引](EXPERIMENT_INDEX.md)；按时间记录的原始判断见[历史归档](archive/research_navigation_20261015/RESEARCH_STATUS_history.md)。保留负结果，不重新包装为成功方向。

## 结论边界

所有98组均已开发暴露；old24/new24/remaining50不构成独立测试。跨Source是两个训练分支的checkpoint比较，IN/BN差异还混有训练轨迹和epoch差异。WT/ET稳定收益、跨架构普适性、独立泛化和创新性尚未确认。

比较时固定病例集合、Source、区域定义和适应预算。三区域平均Dice按WT/TC/ET等权，HD95按共同有效病例配对并报告n；不能用IN98组提升对比BN24组。空预测、极端病例和开发期多次选择均需保留说明。原八家族基线资料按IN/BN及预算分别呈现，不宣称未完成的TEGDA结果。
