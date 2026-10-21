# 历史实验索引与子研究详情

2026-10-15整理前的历史快照。文中的“当前”“运行中”“下一步”属于记录当时，不能作为执行指令。当前状态只读[根目录HANDOFF](../../../HANDOFF.md)。链接已调整为归档位置；逐字原文保存在[维护备份](../../../artifacts/maintenance/navigation_20261015/before.zip)。

---

# 本地实验目录索引

新增[跨Source检查](../../../outputs/ped2023_cross_model_transfer_20261015/README.md)：10月15日10:38:25启动，native BN225/BN250/IN200同24组；固定主算法，BN保持eval统计。CPU端口检查通过，GPU先做IN25000003精确复现，随后72状态/360mask/1080区域自动队列。无新结果，不重复运行。

模型名称、完整 checkpoint 路径、文件重复关系及 2026-10-09 清理记录另见[模型与权重索引](../../MODEL_INDEX.md)。

E 盘数据集、解压/预处理版本、HD_TTA 目录命名来历，以及 2026-10-10 已完成的派生数据/旧概率清理见[数据与磁盘说明](../../DATA_STORAGE_INDEX.md)。

[返回仓库入口](../../../README.md) · [文档分类](../../README.md) · [当前候选状态](../../../HANDOFF.md) · [10月8日判断快照](../../ped2023_candidate_reassessment_20261008.md)

2026-10-15最终入口：[确定的主方法与复现说明](../../GLI_PED_TTA_METHOD.md)。完整98组、输出删减、同输出流程的学习控制和既有八家族基线汇总均完成；09:50缓存重放98份主mask逐值一致，无COME/GT/GPU。下面带日期的阶段判断按当时记录理解，勿据旧启动信息重复执行。

更新于2026-10-13：Source统计学习固定24例及全部参照于09:37北京完成，TC Dice有增益但WT/TC HD95有尾部风险；停止统计梯度改善WT却削弱TC，两臂尚不替换主候选，不自动扩98。可信类别顺序约束近乎plain，保留负消融。整体路线与已确定/待验证事项见[研究进度](../../RESEARCH_STATUS.md)。此前索引记录`outputs/`共33个顶层目录；27目录约21.01 GiB的容量是更早统计，不含后续新增产物。机制合并及后续输出对照完成98组，plain、前景校正、teacher修改E保留和完整共识C保留也已完成完整队列评价、独立核验和报告；主要98唯一组，99记录另列。类别对蒸馏、BN背景共识、输出偏置对照和学习后四类矩校准各有24例结果，效果判断见下表。C保留等嵌套子实验未另计为顶层目录；不是所有目录都完成了患者实验。43个历史目录另见[历史实验索引](../../HISTORICAL_EXPERIMENT_INDEX.md)，结构与容量统计时点见[文件总说明](../../PROJECT_FILE_GUIDE.md)。

`outputs/`被`.gitignore`整体忽略，旧22目录的189个Python文件保留产生结果时的代码，新实验另加runner。公共实现位于应纳入Git的 [src/brats_tta/native](../../../src/brats_tta/native/README.md)，当前新增代码仍未提交；旧runner不自动切换，10月11日类别对实验和plain完整队列均显式使用公共包。下列结果链接仍需要本机产物。

## 当前候选及其直接前序

2026-10-14新增[context_view_reliability](../../../outputs/ped2023_context_view_reliability_20261014/INTERPRETATION.md)：10:42北京完成392状态GPU只读推断，旧全概率逐值重现。view未建立超过confidence的同数量可靠性，停止该门控方向。26代码/8记录冻结，无优化、新分割方法或成绩；不要重复执行。

新增[in_subspace](../../../outputs/ped2023_in_subspace_20261014/PROTOCOL.md)：48组encoder/decoder IN-only学习两臂已启动，CPU自然分区/别名核对通过，4480/3200参数，原teacher/soft/8Adam.001不变。自动8步全网旧soft复现后新96状态、384mask、1152区域复核。尚无性能，不自动扩98，不重复prepare/execute。

18:33更新：[IN分区解释](../../../outputs/ped2023_in_subspace_20261014/INTERPRETATION.md)，正式结果位于attempt02，原启动目录冲突失败保留；96状态/384mask/1152区域通过，两臂TC损失，保留消融。[probability_distillation](../../../outputs/ped2023_probability_distillation_20261014/INTERPRETATION.md)18:21:58完成48状态/192mask/576区域，42代码/10记录冻结；同spatial TC−1.12042pp、HD95+.23746mm(n46)，保留负对照，不扩98。

[loss_checkpoint](../../../outputs/ped2023_loss_checkpoint_20261014/INTERPRETATION.md)20:53:57完成同48组：原MSE固定4步与无GT最小total loss状态best8。39状态复用、57前缀精确对照，96状态/384mask/1152区域通过；44代码/15记录冻结。best8/short4 matched spatial TC−.41317/−1.87931pp，WT/ET小收益伴误报及尾部取舍；保留消融不扩98，不重复执行。

[modality_context](../../../outputs/ped2023_modality_context_20261014/INTERPRETATION.md)22:26:14完成同48组528完整推断/144mask/480输入重建/432区域，41代码/6记录冻结。均值/全一致性/平均输入对旧teacher TC−2.49656/−2.90928/−.75327pp，保留负对照不扩98。

其子研究[proposal_union](../../../outputs/ped2023_modality_context_20261014/proposal_union/INTERPRETATION.md)23:47:20完成192mask/576独立区域，43代码/13记录冻结；union2对旧teacher TC+.41769pp，但confidence+.40637pp。再下一级[decoder_transfer](../../../outputs/ped2023_modality_context_20261014/proposal_union/decoder_transfer/INTERPRETATION.md)10月15日00:05:47完成96旧mask复现/192新mask/576区域，45代码/21记录冻结；union2对匹配spatial/stable TC+.18309/+.20284pp，但均值受00002影响。

[teacher_complement_learning](../../../outputs/ped2023_teacher_complement_learning_20261015/INTERPRETATION.md)05:00:41完成96状态/384mask/1152区域，47代码/30记录冻结。union2完整对旧soft_stable TC+.56553pp，学习贡献+.36270的区间跨零；两组表现不同，保留全部对照。

[teacher_complement_full98](../../../outputs/ped2023_teacher_complement_full98_20261015/INTERPRETATION.md)08:46:35完成：复用48、新增50全部实际学习，450前向/300新mask/900区域/400输入通过，49代码/37记录冻结。子研究[decoder_pruning](../../../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/INTERPRETATION.md)09:06:07完成196新mask/588区域/196实际保护恒等，52代码/44记录冻结；选择no_come主输出，对旧soft_stable TC+.96119pp并保留WT代价。其[final_controls](../../../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/INTERPRETATION.md)09:40:25完成98主mask恒等/196控制mask/588独立区域及八家族既有基线汇总，54代码/66记录冻结、4阶段exit0/7终态hash通过。同输出学习TC+3.00925pp，新teacher进入学习+.29916pp。[交付重放](../../../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/delivery/README.md)09:50:25完成98份无COME输入精确复现。全部队列终态；禁止重复prepare/execute或改冻结文件。

2026-10-14新增[decoder_ablation](../../../outputs/ped2023_decoder_ablation_20261014/INTERPRETATION.md)：98组×10臂08:39北京全部完成，无新模型forward/优化。98旧mask/980新mask独立重建/2940区域覆盖通过，15代码/6记录冻结，不要重复执行。spatial_context保留为简单并行候选，TC额外+.63307pp但WT/ET距离有取舍；Goal active，最终方法未确定。

[范围与组织分解学习](../../../outputs/ped2023_extent_factor_learning_20261013/INTERPRETATION.md)固定24例两臂及matched输出控制已完成，48状态/216mask/648独立区域核验通过，23代码/162记录保持冻结。完整解读与readout统计补充完成：soft matched TC+.75086pp且描述性区间不跨零，WT/ET仍有取舍；hard matched ET−.20200pp。经此解读选择[soft完整98扩展](../../../outputs/ped2023_extent_factor_full98_20261013/README.md)，16:26北京启动，复用24、仅新增74，32代码/18记录冻结，尚无新增性能；不要重复启动或修改冻结依赖。普通队列自动完成评价/独立复核/报告，预计4小时，等待规则见HANDOFF。池化已有CPU输出先例，不宣称本轮首次提出。

20:54更新：上述soft完整98已经20:17完成，74新状态/296mask/888区域核验通过；matched TC+.60491pp但仍不及stable组合，见[完整解释](../../../outputs/ped2023_extent_factor_full98_20261013/INTERPRETATION.md)。新[decoder_transfer](../../../outputs/ped2023_extent_factor_full98_20261013/decoder_transfer/INTERPRETATION.md)已于21:14完成：旧98mask逐值复现、新98mask/294区域独立复核。固定stable流水线只替换学习器，TC+.46236pp、WT/ET均值基本维持，作为优先开发候选；尚无独立泛化。父/子队列均不要重复执行。

统计回放的[review_fix01](../../../outputs/ped2023_source_moment_replay_20261013/review_fix01/README.md)已按冻结条件完成接续，原失败日志保留；48状态/192mask/576区域核验通过，主25/19不变，没有重跑学习或预测。完整效果先读[INTERPRETATION](../../../outputs/ped2023_source_moment_replay_20261013/INTERPRETATION.md)，停止旧队列身份的重复启动。

[IN平移方向检查](../../../outputs/ped2023_source_moment_replay_20261013/invariance_audit/REPORT.md)是成绩完成前的冻结数学记录。固定统计改变前向及有效参数方向，停止统计梯度使用替代反向；不能把TC增益全部归因于减少统计漂移。

| 目录 | 实际内容与状态 | 先读 / 代码入口 |
|---|---|---|
| [ped2023_source_moment_replay_20261013](../../../outputs/ped2023_source_moment_replay_20261013) | 固定24例两臂及七方法参照于09:37北京完成，48状态/192mask/576区域通过。fixed_source对plain TC Dice+1.1376pp，WT/TC配对HD95+2.6887/+2.1662mm；detached_current WT+0.4512pp、TC−2.7463pp。保留机制证据，不自动扩98。主25/19、修复2/9、参照2/25冻结 | `INTERPRETATION.md`、`REPORT.md`、`verification.json`、`queue_complete.json`、`common_case_influence.json`；`review_fix01/complete.json`与`reference_comparison/final.json`；不要重启或重复补充 |
| [ped2023_decision_consensus_anchor_20261012](../../../outputs/ped2023_decision_consensus_anchor_20261012) | 固定24例两臂及全部补充于10月13日00:46北京完成，48状态/192mask/576区域核验通过。decision对plain Dice−0.00016/+0.00078/−0.00027pp，几乎不变；保留负消融，不扩98。主27代码/12记录、参照2/25冻结 | `INTERPRETATION.md`、`REPORT.md`、`verification.json`、`common_case_influence.json`、`study_complete.json`；`reference_comparison/REPORT.md`及`final.json`；勿重启或重复补充 |
| [ped2023_extent_preserving_targets_20261012](../../../outputs/ped2023_extent_preserving_targets_20261012) | 固定24例于21:30完成，48状态/192mask/576区域核验通过；全24梯度及七方法同病例参照完成。对plain Dice+0.1130/−0.1350/−0.0288pp，WT小收益伴TC代价，未优于已有范围校正；保留折中对照，不自动扩98。主24代码/11记录冻结 | `INTERPRETATION.md`、`REPORT.md`、`common_case_influence.json`、`queue_complete.json`；`gradient_geometry/final.json`、`reference_comparison/REPORT.md`及`final.json`；不要重跑队列或补充 |
| [ped2023_stable_consensus_anchor_20261012](../../../outputs/ped2023_stable_consensus_anchor_20261012) | 固定24例stable/all_anchors学习于15:28完成；48状态/192mask/576区域指标复核通过。stable对plain Dice−0.0064/−0.1841/+0.0197pp，三项配对HD95略差；保留消融，不扩98。恢复复用18例/36臂、新跑12臂，20代码/8记录冻结 | `INTERPRETATION.md`、`REPORT.md`、`common_case_influence.json`、`pipeline_complete.json`、`queue_complete.json`；`recovery_preflight.json`及`recovery_01/`保存恢复来源，勿重启 |
| [stable_consensus_anchor/transfer_diagnosis](../../../outputs/ped2023_stable_consensus_anchor_20261012/transfer_diagnosis) | 已有24例输出诊断完成，360绝对区域/1,264路径/216精确Dice分解通过；TC下降主要是未accepted位置NC变回ED。GT事后分析，无新推断/训练 | `INTERPRETATION.md`、`results.json`、`verification.json`；`analyze.py`、`review.py`及原失败/修复日志 |
| [local_support_filter/consensus_preservation](../../../outputs/ped2023_mechanism_merge_20261012/local_support_filter/consensus_preservation) | 完成98×2、196mask/588区域独立核验；stable共识保护对Source Dice+0.0825/+7.9335/+0.1148pp，HD95−0.7298/−2.2782/−0.1393mm，仍ET尾部敏感。全部teacher恢复臂有新增空预测，完整保留 | `INTERPRETATION.md`、`REPORT.md`、`CONTROL_REPORT.md`、`pipeline_complete.json`；`rules.py`、`study.py`、`ORIGIN.json` |
| [mechanism_merge/residual_transition_audit](../../../outputs/ped2023_mechanism_merge_20261012/residual_transition_audit) | 完成98组、2,058项绝对计数、8,706路径、588项Dice分解；另完成teacher类别图诊断，不支持硬门控。诊断无新预测 | `INTERPRETATION.md`、`REPORT.md`、`GRAPH_REPORT.md`、`complete.json`、`graph_verification.json`；`audit.py`、`class_graph.py` |
| [local_support_filter/presence_guard](../../../outputs/ped2023_mechanism_merge_20261012/local_support_filter/presence_guard) | 98×2类别存在性保护完成，196份mask/588项独立指标核验；margin6_guard对Source Dice+0.1014/+7.8452/+0.1111pp，HD95−0.7527/−2.2672/−0.1431mm，n=98/91/84。WT/ET收益仍不稳定，全部开发暴露 | `INTERPRETATION.md`、`REPORT.md`、`CONTROL_REPORT.md`、`pipeline_complete.json`；`presence.py`、`study.py` |
| [mechanism_merge/local_support_filter](../../../outputs/ped2023_mechanism_merge_20261012/local_support_filter) | 正式attempt02完成98×3、294份mask/882项独立指标；opening6与同预算margin6接近，后者TC/ET HD95更好。整类空预测副作用已记录，后续保护另立实验，未覆盖原结果 | `INTERPRETATION.md`、`README.md`、`attempt02/REPORT.md`；`run_v2.py`、`filtering_v2.py`；原失败与布局回归保留 |
| [mechanism_merge/component_support_audit](../../../outputs/ped2023_mechanism_merge_20261012/component_support_audit) | 完成98组、22,279个分量、1,176项Dice分解、882项绝对区域计数；不支持Source同类连接作为通用门控，无新预测 | `INTERPRETATION.md`、`REPORT.md`、`complete.json`、`absolute_verification.json`；`audit.py`、`check_absolute.py` |
| [mechanism_merge/spatial_extent_pool](../../../outputs/ped2023_mechanism_merge_20261012/spatial_extent_pool) | 98组空间前景约束、硬Source WT控制及COME组合完成；294份mask/882区域核验。组合对Source Dice+0.0911/+8.2473/+0.2607pp，ET HD95+0.5416mm仍差；soft与hard的TC接近 | `INTERPRETATION.md`、`REPORT.md`、`pipeline_complete.json`；`pooling.py`、`study.py`；`et_tail_components.json`与`tail_verification.json`定位ET尾部 |
| [mechanism_merge/source_trust_pool](../../../outputs/ped2023_mechanism_merge_20261012/source_trust_pool) | 四类等权几何/算术池化及COME组合完成98组；正式产物在`attempt02/`。几何池化对Source TC仅+2.9613pp、ET HD95恶化，保留负对照；初次缓存key失败的冻结记录保留 | `README.md`、`INTERPRETATION.md`；`run_v2.py`、`attempt02/REPORT.md`与`pipeline_complete.json`；小型`evidence.npz`被空间范围实验复用，不清理 |
| [mechanism_merge/provenance_audit](../../../outputs/ped2023_mechanism_merge_20261012/provenance_audit) | 98组teacher/额外学习/COME来源记账完成；112种合法标签组合和2,688次类别置换复核。没有新预测或训练 | `INTERPRETATION.md`、`REPORT.md`、`complete.json`；`audit.py`、`check_supplement.py`与`supplementary_verification.json` |
| [ped2023_mechanism_merge_20261012](../../../outputs/ped2023_mechanism_merge_20261012) | COME与seeded6的非冲突union及一致intersection均完成98组；196份mask/588项区域指标核验通过。union对Source Dice−0.1312/+7.2053/+0.4123pp，ET HD95略差；保留开发与双路径成本限制 | `README.md`、`PROTOCOL.md`、`REPORT.md`、`INTERPRETATION.md`、`pipeline_complete.json`、`behavior_audit.json`；`merge.py`、`study.py` |
| [teacher_retention/seed_support_audit](../../../outputs/ped2023_teacher_retention_20261011/seed_support_audit) | 10月12日完成98组只读分区诊断和独立复核；稳定高置信邻接同时覆盖收益/损伤，00085缺少NC/ET支持，未据此生成新预测 | `INTERPRETATION.md`、`REPORT.md`、`complete.json`；`audit.py`、`check_extended.py` |
| [ped2023_expanded_baselines_20261011](../../../outputs/ped2023_expanded_baselines_20261011) | BN225的TENT/InTEnt/VPTTA与IN250的MEMO/CoTTA均已完成同24组，576项独立区域指标通过。BN方法WT涨但TC未改善，MEMO/CoTTA接近Source，CoTTA增强门控0/172；四个新方法家族不与变体混数，TEGDA未运行 | `REPORT.md`、`INTERPRETATION.md`、`NORMALIZATION_REVIEW.md`、`comparison.png`、`delivery_verification.json`；`bn_controls/`及`in_controls/`分别保存冻结协议引用、结果与回执 |
| [ped2023_literature_baselines_20261011](../../../outputs/ped2023_literature_baselines_20261011) | 文献检索与5条新增配置均完成98组评价、独立核验及组织审计。COME三域Dice小幅正增益、TC HD95略差；SAR关闭恢复与重置VISTA近乎不变；连续VISTA和TENT十步版TC恶化。后续CoTTA及BN方法24组另见expanded目录，TEGDA仍未完成 | `README.md`、`LITERATURE_REVIEW.md`、`REPORT.md`、`INTERPRETATION.md`、`comparison.png/pdf`；各分支`PROTOCOL.md`及完成回执；小型作者快照在`references/` |
| [ped2023_teacher_retention_20261011](../../../outputs/ped2023_teacher_retention_20261011) | 在Source→teacher修改位置E保留teacher，其余位置用plain加前景偏置；固定98组CPU预测/评价/独立核验完成，TC进一步改善、WT/ET相对Source仍有代价 | `INTERPRETATION.md`、`REPORT.md`、`verification.json`、`pipeline_complete.json`；`retention.py`、`study.py` |
| [teacher_retention/accepted_support](../../../outputs/ped2023_teacher_retention_20261011/accepted_support) | 保留整个原高置信共识集合C，包括支持Source不变标签的位置；98组完成。相对Source Dice−0.6586/+8.2119/−0.1546pp；相对E的差异很小，不能称解决WT/ET损伤 | `INTERPRETATION.md`、`REPORT.md`、`verification.json`、`pipeline_complete.json`；独立冻结`retention.py`、`study.py`及来源差异记录 |
| [teacher_retention/seeded_propagation](../../../outputs/ped2023_teacher_retention_20261011/seeded_propagation) | 6邻接主规则、26邻接敏感性、同修改数距离对照各完成98组；相对C减轻WT/ET平均代价但损失TC；距离对照几乎相同，未建立独特连通性优势。294份mask及882项区域指标独立复核通过 | `INTERPRETATION.md`、`REPORT.md`、`verification.json`、`pipeline_complete.json`；`propagation.py`、`study.py`；`decision_audit.json`记录尾部敏感性与失败 |
| [ped2023_student_moment_calibration_20261011](../../../outputs/ped2023_student_moment_calibration_20261011) | 学习后以原teacher的四类soft总量校准plain，固定24例CPU预测、评价与独立复核完成；相对单前景偏置Dice−0.0876/−2.4986/−0.2968pp，保留负对照，不扩展 | `INTERPRETATION.md`、`REPORT.md`、`verification.json`、`pipeline_complete.json`；`study.py`复用原`calibration.py` |
| [ped2023_native_objective_full99_20261011](../../../outputs/ped2023_native_objective_full99_20261011) | 固定旧plain完整队列研究：复用24例、新增74组，主要98唯一MRI组、另列99记录；8步等价预检及完整评价/独立核验通过；相对Source TC提升、WT/ET下降 | `REPORT.md`、`evaluation.json`、`verification.json`、`pipeline_complete.json`；原执行器已结束 |
| [extent_projection/full_cohort](../../../outputs/ped2023_pairwise_distillation_20261011/extent_projection/full_cohort) | 同一前景偏置规则扩至完整队列：旧24例复用，新74组仅CPU校正；完整评价/独立核验及事后诊断完成；优于plain三项均值，相对Source仍有WT/ET代价 | `INTERPRETATION.md`、`REPORT.md`、`verification.json`、`pipeline_complete.json`、`posthoc_receipt.json`；原执行器已结束 |
| [ped2023_pairwise_distillation_20261011](../../../outputs/ped2023_pairwise_distillation_20261011) | 固定24例IN学习、评价和独立核验均已完成；相对旧plain的WT/TC/ET Dice为−0.0093/−0.8912/−0.4027pp，未改善原方法，保留作消融；`transfer_audit/`另分解已完成24例的teacher域内/外Dice变化 | `REPORT.md`、`evaluation.json`、`verification.json`、`pipeline_complete.json`；[传播诊断](../../../outputs/ped2023_pairwise_distillation_20261011/transfer_audit/INTERPRETATION.md)；`run.py`、`evaluate.py`、`review.py` |
| [ped2023_candidate_reassessment_20261008](../../../outputs/ped2023_candidate_reassessment_20261008) | 只读汇总旧评价，按用户新标准重新定位候选；已完成，无新推断 | [解释](../../ped2023_candidate_reassessment_20261008.md)；`summarize.py`、`summary.json`、`verification.json` |
| [ped2023_context_probe_20261008](../../../outputs/ped2023_context_probe_20261008) | 原 12 例背景观察、直接共识教师、早期 decoder 学习；均已完成。decoder 学习不是后来的 IN-only 方法 | `PROTOCOL.md` / `REPORT.md`；`common.py` 的 `make_contexts` / `teacher`；`predict.py`；学习另看 `ADAPT_PROTOCOL.md` / `ADAPT_REPORT.md` / `adapt.py` |
| [ped2023_context_full99_20261008](../../../outputs/ped2023_context_full99_20261008) | 固定背景教师的 99 例开发结果，含同修改量几何对照；当前主要效果证据，无参数更新 | `REPORT.md`、`evaluation.json`、`geometry_evaluation.json`；`predict.py`、`geometry_predict.py` |
| [ped2023_native_objective_20261008](../../../outputs/ped2023_native_objective_20261008) | 原 12 例 IN affine 学习：native 聚合后 fused 与逐视图 viewwise 目标；已完成 | `PROTOCOL.md` / `REPORT.md`；`settings.py`、`replay.py`、`adapt.py` |
| [ped2023_native_objective_extension_20261008](../../../outputs/ped2023_native_objective_extension_20261008) | 冻结 fused 配置的额外 12 例；已完成。与上行合为 24 例开发结果，WT 有代价 | `REPORT.md`；`prepare.py`、`run.py`、`equivalence.json` |
| [ped2023_fisher_anchor_20261008](../../../outputs/ped2023_fisher_anchor_20261008) | 同 24 例将 source 软锚点换为 Fisher 权重；已完成，作为消融保留 | `REPORT.md`、`case_risk_audit.json`；`common.py`、`run.py`。这里结果键 `fused` 指 Fisher，`plain` 才是旧均匀锚点 |

现在可用完整队列的同病例Source/teacher/plain/前景校正/E保留/C保留结果比较，主要口径98唯一组；旧99例teacher与旧24例学习仍不能直接相减。C报告中的`edit_retention`指父实验E，`retained`才指C，不能混用结果键。完整结论看[HANDOFF](../../../HANDOFF.md)；[重新评估](../../ped2023_candidate_reassessment_20261008.md)及旧联合门槛/False记录保留当时状态。

## 机制、对照与 WT 损伤诊断

| 目录 | 实际内容与状态 | 入口 |
|---|---|---|
| [pairwise/extent_projection](../../../outputs/ped2023_pairwise_distillation_20261011/extent_projection) | 学习后只校正一个前景偏置，固定24例预测/评价/独立复核完成；相对plain的WT/TC/ET Dice +0.3540/+0.0483/−0.0564pp，减轻WT代价，ET仍不稳定；完整队列验证另已完成，完整结论见上表 | `INTERPRETATION.md`、`PROTOCOL.md`、`REPORT.md`、`verification.json`、`pipeline_complete.json` |
| [pairwise/transfer_audit](../../../outputs/ped2023_pairwise_distillation_20261011/transfer_audit) | 旧24例Source/teacher/plain/pair输出的精确Dice分解；核验完成。主要TC收益与WT损失同在teacher未编辑位置，ET收益分布不均；无新学习或分割 | `INTERPRETATION.md`、`REPORT.md`、`analysis.json`、`review_receipt.json` |
| [pairwise/directional_support_audit](../../../outputs/ped2023_pairwise_distillation_20261011/directional_support_audit) | 旧24例检查两种背景观察是否同向支持plain变化；独立复核完成。相对同数量Source margin没有统一可靠性优势，暂不加入门控；不读取新增74例GT | `REPORT.md`、`analysis.json`、`review_verification.json`、`complete.json` |
| [pairwise/moment_bias_control](../../../outputs/ped2023_pairwise_distillation_20261011/moment_bias_control) | 背景teacher的病例级soft概率总量→三个自由度的输出校准；固定24例预测、评价、独立复核完成。Source相对Dice/配对HD95均值略改善，TC低于plain且ET不稳定，保留为校准对照 | `INTERPRETATION.md`、`PROTOCOL.md`、`REPORT.md`、`verification.json`、`pipeline_complete.json` |
| [ped2023_context_decoder_geometry_audit_20261008](../../../outputs/ped2023_context_decoder_geometry_audit_20261008) | 对旧 12 例 decoder 输出补做同方向/数量的几何对照；已完成，无新网络前向 | `PROTOCOL.md` / `REPORT.md`、`audit.py` |
| [ped2023_context_budget_audit_20261008](../../../outputs/ped2023_context_budget_audit_20261008) | 去重 98 文件组的病例修改量审计、均匀/随机数量分配、WT 删除可靠性；已完成，无训练/推断 | `REPORT.md`、`common.py`、`prepare.py`、`evaluate.py`、`*_permutations.npz` |
| [ped2023_in_context_decomposition_20261008](../../../outputs/ped2023_in_context_decomposition_20261008) | 12 例 IN 统计与输入内容交换；已完成诊断，不是可加的因果分解 | `PROTOCOL.md`、`INTERPRETATION_RULES.md`、`REPORT.md`、`instrument.py`、`predict.py` |
| [ped2023_in_affine_anchor_20261008](../../../outputs/ped2023_in_affine_anchor_20261008) | 12 例 IN affine 与匹配 patch 的 source 软锚点；已完成，参数差值及 WT 审计保留；旧 patch 参照数组已清理 | `REPORT.md`、`common.py`、`adapt.py`、`wt_post_result_audit.json` |
| [ped2023_hierarchical_anchor_20261008](../../../outputs/ped2023_hierarchical_anchor_20261008) | 12 例拆分 WT decision margin 与内部组织锚点；已完成，附训练上下文诊断 | `REPORT.md`、`common.py`、`adapt.py`、`context_audit/` |
| [ped2023_deletion_ablation_20261008](../../../outputs/ped2023_deletion_ablation_20261008) | 12 例删除方向消融：只保留或去掉 FG→BG 教师修正；已完成 | `REPORT.md`、`adapt.py`、`audit_errors.py`、`error_audit.json` |
| [ped2023_native_view_audit_20261008](../../../outputs/ped2023_native_view_audit_20261008) | 12 例 source/full/without_delete 状态重放，检查 patch/mirror 与融合；已完成只读网络前向，无新学习候选 | `REPORT.md`、`trace.py`、`prepare.py`、`compare_training_views.py` |

上述`extent_projection/`和`moment_bias_control/`嵌套在pairwise目录是研究过程形成的路径，不表示它们使用类别对蒸馏：前者校正plain输出的一个前景偏置，后者从teacher概率总量拟合三个有效类别偏置。两者的代码、输入、协议和评价分别冻结，现有路径保留。

## 其他观察与可靠性探针

| 目录 | 实际内容与状态 | 入口 |
|---|---|---|
| [ped2023_contrast_probe_20261008](../../../outputs/ped2023_contrast_probe_20261008) | 12 例 FLAIR/T2 gamma 观察，检查高置信 NC→ED 能否纠正；已完成，纠错信号不足 | `PROTOCOL.md` / `REPORT.md` / `RESEARCH_NOTES.md`、`common.py`、`predict.py` |
| [ped2023_structure_response_20261008](../../../outputs/ped2023_structure_response_20261008) | 12 例病灶区域块重排，用结构响应筛选背景教师；已完成，本配置未通过 | `REPORT.md`、`common.py`、`predict.py` |
| [ped2023_input_scale_probe_20261008](../../../outputs/ped2023_input_scale_probe_20261008) | 24 例 0.8/1.25 尺度与往返插值对照，附候选信息审计；已完成，效果不足 | `README.md` / `REPORT.md`、`resampling.py`、`runtime.py`、`run.py`、`candidate_audit.json` |
| [ped2023_intersequence_probe_20261008](../../../outputs/ped2023_intersequence_probe_20261008) | 24 例六个模态对的低频幅度交换；已完成，固定方案未通过。含参考代码及许可证 | `README.md` / `REPORT.md`、`interventions.py`、`diagnostics.py`、`run.py`、`reference/` |
| [ped2023_joint_observation_audit_20261008](../../../outputs/ped2023_joint_observation_audit_20261008) | 24 例旧 IN Source + 十个观察的互补性与凸混合限制；GT 辅助事后诊断，无新预测 | `REPORT.md`、`audit.py`、`audit.json`、`review.py` |

## BN 资源与分支

| 目录 | 实际内容与状态 | 入口 |
|---|---|---|
| [ped2023_bn_resource_review_20261008](../../../outputs/ped2023_bn_resource_review_20261008) | 只在 CPU 检查既有 BN225 权重、22 层 BN、训练配置与 IN 的区别；已完成，无 PED 推断 | `check_resource.py`、`inspection.json` |
| [ped2023_bn_context_20261011](../../../outputs/ped2023_bn_context_20261011) | 固定BN225及running统计，只执行与IN同参数的背景观察/共识；24例预测/评价/独立复核完成，Dice变化−0.0642/+0.0128/−0.2703pp，无整体收益 | `INTERPRETATION.md`、`REPORT.md`、`verification.json`、`review_verification.json` |
| [ped2023_bn_statistics_pilot_20261008](../../../outputs/ped2023_bn_statistics_pilot_20261008) | 24 例 BN source running statistics 与当前 patch statistics；已完成全部预测、评价和核验，非参数学习 | `README.md` / `REPORT.md` / `EXECUTION_SCOPE.md`、`bn_adapter.py`、`runtime.py`、`run.py` |
| [ped2023_bn_complement_audit_20261008](../../../outputs/ped2023_bn_complement_audit_20261008) | 24 例 BN 与已有 IN 观察互补/置信选择诊断；审计及科学 review 已完成，无新分割 | `REPORT.md`、`audit.py`、`audit.json`、`review_verification.json`；**没有 `final_verification.json`**，报告中的预期文件不能当现存回执 |
| [ped2023_bn_input_style_20261008](../../../outputs/ped2023_bn_input_style_20261008) | 仅有 BN 输入仿射变换/统计目标协议、实现和 CPU 检查；当前备选，**未做真实模型/PED 前向** | `PROTOCOL.md`、`style.py`、`check_cpu.py`、`cpu_check.json`；没有性能报告 |

BN 统计目录的冻结 `PROTOCOL.md` 还保留历史等待措辞；其后执行范围更正、完成回执及 HANDOFF 已覆盖该状态。本次不改冻结协议，也不因读到旧“未启动”文字重跑实验。

## 实验内部文件和子目录怎么读

| 类型 | 实际用途 |
|---|---|
| `PROTOCOL.md`、`ADAPT_PROTOCOL.md` | 当次实验执行前的设计、资源、规则和判定条件；不要当成最新任务清单 |
| `REPORT.md`、`ADAPT_REPORT.md`、`RESEARCH_NOTES.md` | 结果解释与后续假设；与最新 HANDOFF/候选判断一起读 |
| `prepare.py` / `predict.py` / `run.py` / `adapt.py` | 输入清单准备、冻结预测或实际参数学习；根据该目录协议判断，不按文件名猜测 |
| `common.py` / `settings.py` / `runtime.py` / `replay.py` | 本轮方法、超参数、模型适配或梯度重放；可能动态导入其他实验目录 |
| `evaluate.py` / `audit.py` / `review.py` / `check*.py` | 评价、事后诊断、复核和数值检查；部分 audit 会前向模型，部分只读缓存 |
| `inputs.json` / `cohorts.json` / `preparation*.json` | 输入身份、队列选择、依赖代码和资源指纹 |
| `prediction_complete.json` / `finish_status.json` | 当次阶段完成状态；文件不存在就不能声称该阶段完成 |
| `evaluation.json` / `audit.json` / `summary.json` | 机器可读指标或诊断，含义随实验而变 |
| `verification.json` / `review_verification.json` / `final_verification.json` | 不同层次的工程/科学/交付核验；并非每轮都有全部三种，不等于方法有效 |
| `cases/<case_id>/` | 每例结果；预测类实验含 NIfTI、概率/几何/参数 `.npz` 和 JSON；缓存审计可能只有计数/诊断数组 |
| `*_margin` / `*_geometry` / `*_random` | 使用匹配编辑方向、数量和可编辑区域的对照，不是额外训练模型 |
| `adapt_attempt01/` | context probe 初次学习尝试的代码、协议和日志归档 |
| `preflight_marginal/` | hierarchical anchor 中早期 marginal 形式的数值预检归档 |
| `context_audit/cases/` | hierarchical anchor 的训练上下文与 source 参照诊断 |
| `equivalence/<case_id>/` | native extension 对旧实现的单病例等价性核验产物 |
| `reference/` | intersequence 中参考的 `interventions.py` / `views.py` 及 `LICENSE`；不是患者数据或最终模型 |
| `.log`、图像/PDF、`__pycache__/` | 日志、报告图和 Python 缓存；失败/恢复日志也作为实验沿革保留 |

## 集中存放的模型与历史实验

以下目录已于 2026-10-10 从 E 盘迁入本项目，逐文件校验后移除旧位置；没有重新评估或重训。完整 checkpoint 路径与哈希见[模型索引](../../MODEL_INDEX.md)，迁移沿革见[存储说明](../../STORAGE_LAYOUT.md)。这些本机模型/实验产物仍被 Git 忽略。

| 路径 | 用途 |
|---|---|
| [D:/Workspace/deeplearn/3dunet/artifacts/models/in_dicece](D:/Workspace/deeplearn/3dunet/artifacts/models/in_dicece/) | 已完成 native IN 二分类/四分类源模型、source bundle、下载与模型核验；四分类 IN250 为当前主 source |
| [D:/Workspace/deeplearn/3dunet/artifacts/models/bn_dicece](D:/Workspace/deeplearn/3dunet/artifacts/models/bn_dicece/) | native BN norm-only 对照、训练日志、plans、checkpoint 与下载核验；当前诊断用 best225 |
| [D:/Workspace/deeplearn/3dunet/artifacts/models/wt_pure_dice](D:/Workspace/deeplearn/3dunet/artifacts/models/wt_pure_dice/) | pure-Dice WT 源模型备份及运行材料；不是当前四分类模型 |
| [D:/Workspace/deeplearn/3dunet/artifacts/experiments/nnUNet_PED2023_IN_analysis_20261005](D:/Workspace/deeplearn/3dunet/artifacts/experiments/nnUNet_PED2023_IN_analysis_20261005/) | IN250 的 99 例 Source 完整预测、manifest、误差分析和原始浏览页 `outputs/full/index.html` |
| [D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_TC_shape_review_20261006](D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_TC_shape_review_20261006/) | Source 病例可视化、形状分析和模型卡；推荐浏览 `easy_view/index.html` |
| [D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_norm_TTA_20261006](D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_norm_TTA_20261006/) | native TENT/SAR 实验，以及当前多轮复用的 `native_adapter.py` 和 `independent_metrics.py` |
| [D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_axis_probe_20261006](D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_axis_probe_20261006/) | 跨轴观察/学习结果及部分后续实验复用的 source 缓存 |
| [D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_snapshot_probe_20261006](D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_snapshot_probe_20261006/) | checkpoint 快照探针；其 `common.py` 被当前 context 代码动态复用 |
| [D:/Workspace/deeplearn/3dunet/artifacts/models/legacy_custom_bn](D:/Workspace/deeplearn/3dunet/artifacts/models/legacy_custom_bn/) | 九月份自定义训练恢复、历史原始/HD-BET 数据诊断；不能与 native PED2023 协议混用 |

全部 43 个历史目录的用途、入口及复用状态见[历史实验索引](../../HISTORICAL_EXPERIMENT_INDEX.md)；研究文字摘要另见[文档分类](../../README.md)。

## 路径迁移与复用关系

`context_probe/common.py` 复用历史 adapter/helper；`context_full99` 复用 `context_probe`；native 学习通过 `common.py` 连续复用 deletion / hierarchical / affine 等前序目录。迁移已修正绝对路径与工作副本的文本哈希引用，原始记录另存归档；当前导入和依赖校验仍须以实际验证为准。

10月10日最初的集中存放和路径适配没有改变算法；随后已把公共实现提取到 `src/brats_tta/native/`，10月11日的新类别对实验显式使用该包。产生旧结果的 runner 仍保留原导入链与快照。旧缓存清理后，部分历史核验需先重建输入；已有指标和最终分割保留。原始文本、清理清单和迁移校验均见[存储说明](../../STORAGE_LAYOUT.md)。
