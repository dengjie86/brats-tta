# 本地实验目录索引

- [Senior BN迁移失效诊断](../outputs/senior188_bn_diagnosis_20261022/REPORT.md)：66例teacher/raw/final分解、同66例历史IN teacher比较、6例无学习BN统计干预已完成。teacher未提供有效TC修正；统计方式对部分病例影响明显但并非普遍改善。原评测保持暂停，没有接续剩余病例。

- [Senior best188 用户方法：评测入口](../outputs/senior188_ped2023_ours_20261021/README.md)：**10月22日01:09按用户要求暂停，评测进程已退出**，保留66/98例；平均Dice Source58.70143%、TENT62.67784%、用户方法58.72396%。union2 + single-view8 soft_extent + no_come的Senior无镜像概率融合迁移版本；见paused.json，不自动恢复，原ETA失效。

缓存可用性与占用见[当前存储说明](STORAGE_STATUS.md)。下列实验有新增`CACHE_STATUS.md`，读取旧数组前先查看；旧成绩与回执保留。

- [ped2023_decision_consensus_anchor_20261012](../outputs/ped2023_decision_consensus_anchor_20261012/CACHE_STATUS.md)
- [ped2023_extent_factor_learning_20261013](../outputs/ped2023_extent_factor_learning_20261013/CACHE_STATUS.md)
- [ped2023_extent_preserving_targets_20261012](../outputs/ped2023_extent_preserving_targets_20261012/CACHE_STATUS.md)
- [ped2023_in_context_decomposition_20261008](../outputs/ped2023_in_context_decomposition_20261008/CACHE_STATUS.md)
- [ped2023_in_subspace_20261014](../outputs/ped2023_in_subspace_20261014/CACHE_STATUS.md)
- [ped2023_loss_checkpoint_20261014](../outputs/ped2023_loss_checkpoint_20261014/CACHE_STATUS.md)
- [ped2023_probability_distillation_20261014](../outputs/ped2023_probability_distillation_20261014/CACHE_STATUS.md)
- [ped2023_source_moment_replay_20261013](../outputs/ped2023_source_moment_replay_20261013/CACHE_STATUS.md)
- [ped2023_stable_consensus_anchor_20261012](../outputs/ped2023_stable_consensus_anchor_20261012/CACHE_STATUS.md)
- [ped2023_structure_response_20261008](../outputs/ped2023_structure_response_20261008/CACHE_STATUS.md)
- [ped2023_teacher_complement_learning_20261015](../outputs/ped2023_teacher_complement_learning_20261015/CACHE_STATUS.md)

2026-10-15整理时覆盖`outputs/`45个一级实验目录；10月16日追加计算核算、效率先导及完整98组效率验证三目录。按研究用途分类，目录和冻结依赖未搬动。最新执行状态只看[HANDOFF](../HANDOFF.md)，冻结主方法看[方法说明](GLI_PED_TTA_METHOD.md)。更早材料见[历史实验索引](HISTORICAL_EXPERIMENT_INDEX.md)。

## 直接入口

| 内容 | 入口 |
|---|---|
| Senior best188：本地PED2023全98组Source/TENT，原包预处理与无镜像 | [协议、运行与结果入口](../outputs/senior188_ped2023_source_tent_20261021/README.md) |
| 第二台AutoDL：所给Senior包仅改128³，原训练协议 | [senior_unet128_source_20261020：协议与进度页面](../outputs/senior_unet128_source_20261020/README.md) |
| 用户BN U-Net epoch125：本地PED2023全98组Source/TENT | [运行与阶段结果入口](../outputs/qq_unet125_ped2023_source_tent_20261019/README.md) |
| 第一台AutoDL新Source训练：用户提供的五层BN U-Net | [qq_unet_source_20261019：协议、验证与实时监控](../outputs/qq_unet_source_20261019/README.md) |
| 原完整方法只取消学习镜像，98组已完成 | [union2 teacher＋单视角＋8步结果入口](../outputs/ped2023_efficiency_full98_20261016/union2_single_view8/README.md) |
| 轻量方案完整98组验证 | [base2_v1_s4完成结果](../outputs/ped2023_efficiency_full98_20261016/REPORT.md) |
| 计算效率优化先导 | [固定6例入口及失败记录](../outputs/ped2023_efficiency_pilot_20261016/README.md)；[attempt03完成结果](../outputs/ped2023_efficiency_pilot_20261016/attempt03/REPORT.md) |
| 实际网络核心FLOPs核算 | [meta追踪及逐层公式报告](../outputs/ped2023_compute_audit_20261016/REPORT.md) |
| 主输出选择与98份mask | [decoder_pruning](../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/INTERPRETATION.md)；`cases/<id>/no_come.nii.gz` |
| 最终学习贡献与基线汇总 | [final_controls](../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/INTERPRETATION.md) |
| 98份缓存交付重放 | [delivery](../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/delivery/README.md) |
| 跨模型阶段Dice | [interim_readout](../outputs/ped2023_cross_model_transfer_20261015/interim_readout/REPORT.md) |
| 旧soft_stable参照 | [extent_factor/decoder_transfer](../outputs/ped2023_extent_factor_full98_20261013/decoder_transfer/INTERPRETATION.md) |
| union2补充teacher来源 | [proposal_union](../outputs/ped2023_modality_context_20261014/proposal_union/INTERPRETATION.md)及[输出迁移](../outputs/ped2023_modality_context_20261014/proposal_union/decoder_transfer/INTERPRETATION.md) |
| 原统计回放的独立修复 | [review_fix01](../outputs/ped2023_source_moment_replay_20261013/review_fix01/README.md) |
| IN分区正式尝试 | [attempt02](../outputs/ped2023_in_subspace_20261014/attempt02/) |

## 主方法、效率与跨模型验证（6）

| 目录 | 用途与结果定位 | 阅读入口 |
|---|---|---|
| [ped2023_efficiency_full98_20261016](../outputs/ped2023_efficiency_full98_20261016/) | 两轮98组均完成：base2_v1_s4降算85.15%、均值低0.847pp；union2_single_view8保留完整teacher和8步，降算59.20%、均值低0.082pp，均相对main | [README.md](../outputs/ped2023_efficiency_full98_20261016/README.md) |
| [ped2023_efficiency_pilot_20261016](../outputs/ped2023_efficiency_pilot_20261016/) | attempt03完成6例/36份候选及108项Dice检查；双观察单视角4步成本降85.1%、均值低0.15pp，尚未扩量 | [README.md](../outputs/ped2023_efficiency_pilot_20261016/README.md) |
| [ped2023_compute_audit_20261016](../outputs/ped2023_compute_audit_20261016/) | 实际native前/反向卷积FLOPs；98例从头执行预算核算，无GPU推理 | [REPORT.md](../outputs/ped2023_compute_audit_20261016/REPORT.md) |
| [ped2023_cross_model_transfer_20261015](../outputs/ped2023_cross_model_transfer_20261015/) | BN225/BN250各24例Dice已读；IN200中断，完整复核未完成 | [README.md](../outputs/ped2023_cross_model_transfer_20261015/README.md) |
| [ped2023_teacher_complement_full98_20261015](../outputs/ped2023_teacher_complement_full98_20261015/) | union2完整98组；子目录包含no_come选择、最终控制和交付 | [INTERPRETATION.md](../outputs/ped2023_teacher_complement_full98_20261015/INTERPRETATION.md) |
| [ped2023_teacher_complement_learning_20261015](../outputs/ped2023_teacher_complement_learning_20261015/) | 48组补充teacher学习；union2/confidence及匹配输出控制 | [INTERPRETATION.md](../outputs/ped2023_teacher_complement_learning_20261015/INTERPRETATION.md) |

## 主路线的前序阶段（10）

| 目录 | 用途与结果定位 | 阅读入口 |
|---|---|---|
| [ped2023_context_probe_20261008](../outputs/ped2023_context_probe_20261008/) | 原12例双背景观察、共识teacher与早期学习 | [REPORT.md](../outputs/ped2023_context_probe_20261008/REPORT.md) |
| [ped2023_context_full99_20261008](../outputs/ped2023_context_full99_20261008/) | 双背景共识完整99记录；98唯一组分析须去重复 | [REPORT.md](../outputs/ped2023_context_full99_20261008/REPORT.md) |
| [ped2023_native_objective_20261008](../outputs/ped2023_native_objective_20261008/) | 原12例native fused/viewwise IN学习 | [REPORT.md](../outputs/ped2023_native_objective_20261008/REPORT.md) |
| [ped2023_native_objective_extension_20261008](../outputs/ped2023_native_objective_extension_20261008/) | 额外12例固定fused设置，组成原24例 | [REPORT.md](../outputs/ped2023_native_objective_extension_20261008/REPORT.md) |
| [ped2023_native_objective_full99_20261011](../outputs/ped2023_native_objective_full99_20261011/) | plain IN学习完整开发队列 | [REPORT.md](../outputs/ped2023_native_objective_full99_20261011/REPORT.md) |
| [ped2023_pairwise_distillation_20261011](../outputs/ped2023_pairwise_distillation_20261011/) | 类别对蒸馏及独立子研究extent_projection前景校正 | [INTERPRETATION.md](../outputs/ped2023_pairwise_distillation_20261011/INTERPRETATION.md) |
| [ped2023_teacher_retention_20261011](../outputs/ped2023_teacher_retention_20261011/) | teacher编辑/完整共识保留、种子传播及支持诊断 | [INTERPRETATION.md](../outputs/ped2023_teacher_retention_20261011/INTERPRETATION.md) |
| [ped2023_mechanism_merge_20261012](../outputs/ped2023_mechanism_merge_20261012/) | COME合并、范围池化、margin/presence及stable保护前序 | [INTERPRETATION.md](../outputs/ped2023_mechanism_merge_20261012/INTERPRETATION.md) |
| [ped2023_extent_factor_learning_20261013](../outputs/ped2023_extent_factor_learning_20261013/) | 固定24例soft/hard范围投影学习和匹配输出对照 | [INTERPRETATION.md](../outputs/ped2023_extent_factor_learning_20261013/INTERPRETATION.md) |
| [ped2023_extent_factor_full98_20261013](../outputs/ped2023_extent_factor_full98_20261013/) | soft扩展98组；decoder_transfer形成旧soft_stable参照 | [INTERPRETATION.md](../outputs/ped2023_extent_factor_full98_20261013/INTERPRETATION.md) |

## 外部基线与BN分支（7）

| 目录 | 用途与结果定位 | 阅读入口 |
|---|---|---|
| [ped2023_literature_baselines_20261011](../outputs/ped2023_literature_baselines_20261011/) | 文献核查与5条新增配置98组；IN/预算须分开比较 | [INTERPRETATION.md](../outputs/ped2023_literature_baselines_20261011/INTERPRETATION.md) |
| [ped2023_expanded_baselines_20261011](../outputs/ped2023_expanded_baselines_20261011/) | BN的InTEnt/VPTTA与IN的MEMO/CoTTA等同24组对照 | [INTERPRETATION.md](../outputs/ped2023_expanded_baselines_20261011/INTERPRETATION.md) |
| [ped2023_bn_resource_review_20261008](../outputs/ped2023_bn_resource_review_20261008/) | BN权重和统计缓冲区资源核查 | [目录](../outputs/ped2023_bn_resource_review_20261008/) |
| [ped2023_bn_statistics_pilot_20261008](../outputs/ped2023_bn_statistics_pilot_20261008/) | 固定24例BN统计替换；WT收益伴TC/ET下降 | [REPORT.md](../outputs/ped2023_bn_statistics_pilot_20261008/REPORT.md) |
| [ped2023_bn_context_20261011](../outputs/ped2023_bn_context_20261011/) | 原双背景teacher在BN225上的24例负对照 | [INTERPRETATION.md](../outputs/ped2023_bn_context_20261011/INTERPRETATION.md) |
| [ped2023_bn_complement_audit_20261008](../outputs/ped2023_bn_complement_audit_20261008/) | BN/IN缓存互补诊断；科学review已有，final回执缺失 | [REPORT.md](../outputs/ped2023_bn_complement_audit_20261008/REPORT.md) |
| [ped2023_bn_input_style_20261008](../outputs/ped2023_bn_input_style_20261008/) | 仅协议、实现及CPU准备，没有患者性能报告 | [PROTOCOL.md](../outputs/ped2023_bn_input_style_20261008/PROTOCOL.md) |

## 学习目标与组件消融（14）

| 目录 | 用途与结果定位 | 阅读入口 |
|---|---|---|
| [ped2023_in_affine_anchor_20261008](../outputs/ped2023_in_affine_anchor_20261008/) | IN affine锚定与背景学习机制检查 | [REPORT.md](../outputs/ped2023_in_affine_anchor_20261008/REPORT.md) |
| [ped2023_in_context_decomposition_20261008](../outputs/ped2023_in_context_decomposition_20261008/) | IN与上下文学习作用拆解 | [REPORT.md](../outputs/ped2023_in_context_decomposition_20261008/REPORT.md) |
| [ped2023_hierarchical_anchor_20261008](../outputs/ped2023_hierarchical_anchor_20261008/) | 分层保护目标对照 | [REPORT.md](../outputs/ped2023_hierarchical_anchor_20261008/REPORT.md) |
| [ped2023_fisher_anchor_20261008](../outputs/ped2023_fisher_anchor_20261008/) | Fisher软锚点权重消融 | [REPORT.md](../outputs/ped2023_fisher_anchor_20261008/REPORT.md) |
| [ped2023_deletion_ablation_20261008](../outputs/ped2023_deletion_ablation_20261008/) | teacher前景到背景删除监督消融 | [REPORT.md](../outputs/ped2023_deletion_ablation_20261008/REPORT.md) |
| [ped2023_student_moment_calibration_20261011](../outputs/ped2023_student_moment_calibration_20261011/) | 学习后四类矩校准；未改善单前景偏置 | [INTERPRETATION.md](../outputs/ped2023_student_moment_calibration_20261011/INTERPRETATION.md) |
| [ped2023_stable_consensus_anchor_20261012](../outputs/ped2023_stable_consensus_anchor_20261012/) | 稳定共识学习锚点与转移诊断；保留负结果 | [INTERPRETATION.md](../outputs/ped2023_stable_consensus_anchor_20261012/INTERPRETATION.md) |
| [ped2023_extent_preserving_targets_20261012](../outputs/ped2023_extent_preserving_targets_20261012/) | 范围保护监督目标及梯度检查；有WT/TC取舍 | [INTERPRETATION.md](../outputs/ped2023_extent_preserving_targets_20261012/INTERPRETATION.md) |
| [ped2023_decision_consensus_anchor_20261012](../outputs/ped2023_decision_consensus_anchor_20261012/) | 可信类别顺序锚点；相对plain几乎不变 | [INTERPRETATION.md](../outputs/ped2023_decision_consensus_anchor_20261012/INTERPRETATION.md) |
| [ped2023_source_moment_replay_20261013](../outputs/ped2023_source_moment_replay_20261013/) | 固定Source统计/停止统计梯度；收益与距离风险取舍 | [INTERPRETATION.md](../outputs/ped2023_source_moment_replay_20261013/INTERPRETATION.md) |
| [ped2023_decoder_ablation_20261014](../outputs/ped2023_decoder_ablation_20261014/) | 98组输出组件删除；简化与WT/ET尾部取舍 | [INTERPRETATION.md](../outputs/ped2023_decoder_ablation_20261014/INTERPRETATION.md) |
| [ped2023_in_subspace_20261014](../outputs/ped2023_in_subspace_20261014/) | encoder/decoder IN分区；正式结果在attempt02，两臂TC下降 | [INTERPRETATION.md](../outputs/ped2023_in_subspace_20261014/INTERPRETATION.md) |
| [ped2023_probability_distillation_20261014](../outputs/ped2023_probability_distillation_20261014/) | 同目标KL替换MSE；TC下降，保留负对照 | [INTERPRETATION.md](../outputs/ped2023_probability_distillation_20261014/INTERPRETATION.md) |
| [ped2023_loss_checkpoint_20261014](../outputs/ped2023_loss_checkpoint_20261014/) | 固定4步与最低无标签loss选回；未改善主要TC收益 | [INTERPRETATION.md](../outputs/ped2023_loss_checkpoint_20261014/INTERPRETATION.md) |

## 输入观察、可靠性与事后诊断（11）

| 目录 | 用途与结果定位 | 阅读入口 |
|---|---|---|
| [ped2023_candidate_reassessment_20261008](../outputs/ped2023_candidate_reassessment_20261008/) | 10月8日候选重评快照，不是最新主方法选择 | [目录](../outputs/ped2023_candidate_reassessment_20261008/) |
| [ped2023_context_budget_audit_20261008](../outputs/ped2023_context_budget_audit_20261008/) | 背景编辑数量和匹配预算诊断 | [REPORT.md](../outputs/ped2023_context_budget_audit_20261008/REPORT.md) |
| [ped2023_context_decoder_geometry_audit_20261008](../outputs/ped2023_context_decoder_geometry_audit_20261008/) | 上下文/decoder变化的空间几何诊断 | [REPORT.md](../outputs/ped2023_context_decoder_geometry_audit_20261008/REPORT.md) |
| [ped2023_contrast_probe_20261008](../outputs/ped2023_contrast_probe_20261008/) | MRI对比度响应观察 | [REPORT.md](../outputs/ped2023_contrast_probe_20261008/REPORT.md) |
| [ped2023_input_scale_probe_20261008](../outputs/ped2023_input_scale_probe_20261008/) | 输入缩放及往返插值对照 | [REPORT.md](../outputs/ped2023_input_scale_probe_20261008/REPORT.md) |
| [ped2023_intersequence_probe_20261008](../outputs/ped2023_intersequence_probe_20261008/) | 跨序列频谱观察，不等于完整VISTA复现 | [REPORT.md](../outputs/ped2023_intersequence_probe_20261008/REPORT.md) |
| [ped2023_joint_observation_audit_20261008](../outputs/ped2023_joint_observation_audit_20261008/) | 背景、尺度与跨序列观察的缓存互补性 | [REPORT.md](../outputs/ped2023_joint_observation_audit_20261008/REPORT.md) |
| [ped2023_native_view_audit_20261008](../outputs/ped2023_native_view_audit_20261008/) | 原生patch/mirror与聚合阶段的变化审计 | [REPORT.md](../outputs/ped2023_native_view_audit_20261008/REPORT.md) |
| [ped2023_structure_response_20261008](../outputs/ped2023_structure_response_20261008/) | 结构响应筛选，固定方案未通过 | [REPORT.md](../outputs/ped2023_structure_response_20261008/REPORT.md) |
| [ped2023_context_view_reliability_20261014](../outputs/ped2023_context_view_reliability_20261014/) | 原生视图可靠性未超过同数量confidence控制 | [INTERPRETATION.md](../outputs/ped2023_context_view_reliability_20261014/INTERPRETATION.md) |
| [ped2023_modality_context_20261014](../outputs/ped2023_modality_context_20261014/) | 分模态均值teacher负结果；proposal_union子研究保留旧编辑并补充候选 | [INTERPRETATION.md](../outputs/ped2023_modality_context_20261014/INTERPRETATION.md) |

## 子研究与历史详情

旧索引中更细的子目录说明已保留在[历史详细索引](archive/research_navigation_20261015/EXPERIMENT_INDEX_history.md)。其中的启动时间、PID、候选选择和待办均为历史快照。目录名体现来源，不能据此推断当前方法使用父目录的全部算法。

实验内文件类型看[项目文件说明](PROJECT_FILE_GUIDE.md)。`preparation.json`、冻结代码/配置、学习状态、预测与指标按原样保存；修复目录保留原失败记录。新实验不覆盖原目录，导航更新不回写旧回执哈希。统计容量和checkpoint数量须注明盘点日期，不能把历史快照当作实时磁盘占用。
