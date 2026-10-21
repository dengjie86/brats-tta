# 历史研究专题导航

这里只索引设计演变、机制诊断与历史报告。报告中的待办以[当前交接](../../HANDOFF.md)为准；专题报告原路径和内容保留。

## 前序路线文档（10月8日至13日）

| 文档 | 内容与当前定位 |
|---|---|
| [当前交接](../../HANDOFF.md) | 最新资源、研究任务和完成范围；优先于旧报告中的待办措辞 |
| [用户提供的四份训练代码对比](../qq_source_code_comparison_20261013.md) | 外部 BN U-Net 与当前 native Source 的结构、训练及指标差异；CPU 复现 batch 汇总错误，实际 YAML/loader/loss/权重配置仍未知 |
| [范围约束相近工作](../ped2023_extent_related_work_20261013.md) | 10月13日补充BAPA、Shape Moments、ReGA、PIE及其协议边界；ReGA已做公开代码接口核对，没有新增native性能成绩 |
| [种子约束传播](../../outputs/ped2023_teacher_retention_20261011/seeded_propagation/INTERPRETATION.md) | 6邻接、26邻接与同修改数距离对照各完成98组；WT/ET平均代价减轻但TC回落，距离对照几乎相同；保留更保守候选与尾部失败说明 |
| [teacher修改E保留](../../outputs/ped2023_teacher_retention_20261011/INTERPRETATION.md) / [完整共识C保留](../../outputs/ped2023_teacher_retention_20261011/accepted_support/INTERPRETATION.md) | 两项98组CPU实验均完成预测、评价和独立核验；进一步提高TC，C与E差别很小，WT/ET损伤仍未解决；实现仍在各自实验目录 |
| [相近文献与贡献边界](../ped2023_context_related_work_20261011.md) | 10月11日核对TRACE-Seg3D、MuVi及IN滑窗机制：区分已有概念、具体实现差异、已有证据和未确认的新颖性；不改变当前实验 |
| [plain IN完整开发队列](../../outputs/ped2023_native_objective_full99_20261011/README.md) | 8步等价预检通过，复用24例并新增74组；完整评价、独立核验和报告已完成 |
| [前景偏置校正完整开发队列](../../outputs/ped2023_pairwise_distillation_20261011/extent_projection/full_cohort/README.md) | 复用24例、新增74组的完整评价和独立核验已完成；优于plain三项均值，相对Source仍有WT/ET代价，另有完整队列解释与诊断 |
| [BN背景共识适用性检验](../../outputs/ped2023_bn_context_20261011/INTERPRETATION.md) | 24例评价和独立核验完成，无自身BN Source相对整体收益；保留负结果与适用范围说明 |
| [学习后四类矩校准](../../outputs/ped2023_student_moment_calibration_20261011/INTERPRETATION.md) | 10月11日新24例CPU对照及独立复核完成；损失较多TC收益，未改善单前景偏置，不扩展该固定方案 |
| [类别对蒸馏结果](../../outputs/ped2023_pairwise_distillation_20261011/REPORT.md) | 10月11日24例评价及独立核验已完成；未超过旧plain IN学习，保留作消融 |
| [候选重新评估](../ped2023_candidate_reassessment_20261008.md) | 对10月8日已有结果的判断：区分 Source 相对收益、旧联合门槛、机制证据；保留背景共识与轻量 IN 学习候选 |
| [背景上下文教师](../ped2023_context_teacher_validation_20261008.md) | 原 12 例的背景干预、共识教师及早期 decoder 学习；99 例的原始报告另在实验索引 |
| [Native 聚合目标](../ped2023_native_objective_20261008.md) | 原 12 例 fused 与 viewwise IN 学习配对实验 |
| [Native 目标额外 12 例](../ped2023_native_objective_extension_20261008.md) | 冻结 fused 配置复验；NC/TC 有收益，WT 未通过当时保护门槛，仍属开发数据 |
| [Fisher 锚点](../ped2023_fisher_anchor_20261008.md) | 同 24 例改变软锚点权重；作为消融保留，尚无稳定优于 plain 的证据 |

## 上下文主线的机制与学习诊断

| 文档 | 要回答的问题 |
|---|---|
| [旧24例学习传播分解](../../outputs/ped2023_pairwise_distillation_20261011/transfer_audit/INTERPRETATION.md) | 10月11日已完成：plain的大部分TC收益和WT代价都发生在teacher未编辑的位置；ET均值受少数病例影响。这是输出分解，不是新算法或因果证明 |
| [双观察同向支持诊断](../../outputs/ped2023_pairwise_distillation_20261011/directional_support_audit/REPORT.md) | 10月11日已完成：相对同数量Source margin，WT贡献更差、TC略低、ET较好，暂不据此加入通用筛选或门控 |
| [病例级输出偏置对照](../../outputs/ped2023_pairwise_distillation_20261011/moment_bias_control/INTERPRETATION.md) | 10月11日24例已完成：只保留teacher的soft概率总量也有Source相对收益，但TC低于plain、ET病例波动明显；不是参数数量的严格消融或新颖性证明 |
| [学习后前景偏置校正](../../outputs/ped2023_pairwise_distillation_20261011/extent_projection/INTERPRETATION.md) | 10月11日24例已完成：保留plain内部类别关系，仅恢复Source的二元前景代理均值；WT代价减小、TC收益保留，ET仍有波动。完整队列验证已另完成，原解释中的“下一轮”是完成时快照 |
| [IN与冻结BN的算子机制](../../outputs/ped2023_bn_context_20261011/NORMALIZATION_MECHANISM.md) | 单层归一化推导和CPU Jacobian检查；支持理解上下文依赖，不能证明整个模型的唯一因果机制或BN上的方法有效性 |
| [背景修改量审计](../ped2023_context_budget_audit_20261008.md) | 收益来自修改位置还是每个病例的修改数量？同时检查 WT 有害删除 |
| [IN 统计与上下文分解](../ped2023_in_context_decomposition_20261008.md) | 交换 IN 统计与图像上下文，附旧 decoder 同数量几何对照；不能当严格因果分解 |
| [IN affine 与 patch 锚点](../ped2023_in_affine_anchor_20261008.md) | 从 decoder 学习转为 IN affine；校准训练 patch 的 source 参照和 self 对照 |
| [WT 与内部组织锚点拆分](../ped2023_hierarchical_anchor_20261008.md) | 区分前景范围保留与内部类别保留的作用 |
| [前景删除方向消融](../ped2023_deletion_ablation_20261008.md) | 检验 teacher 的 FG→BG 监督是否造成 WT 损失 |
| [Native 视图审计](../ped2023_native_view_audit_20261008.md) | WT 损失在单个 patch/mirror 预测还是融合阶段出现 |
| [结构响应筛选](../ped2023_structure_response_20261008.md) | 用病灶块重排后的响应筛选背景教师；本配置未通过 |

## 其他观察与 BN 分支

| 文档 | 内容与状态 |
|---|---|
| [MRI 对比度响应](../ped2023_contrast_response_20261008.md) | FLAIR/T2 gamma 观察的纠错能力；是提出背景探针的前序依据 |
| [输入尺度观察](../ped2023_input_scale_probe_20261008.md) | 0.8/1.25 尺度及 image/probability 往返插值对照；24 例已完成 |
| [跨序列观察](../ped2023_intersequence_probe_20261008.md) | 六个 MRI 模态对的低频幅度交换；24 例已完成，并非 VISTA 完整复现 |
| [观察联合审计](../ped2023_joint_observation_audit_20261008.md) | 缓存背景、尺度、跨序列观察的互补性与概率混合限制；无新预测 |
| [BN 统计诊断原方案](../ped2023_bn_statistics_pilot_proposal_20261008.md) | 历史准备/范围说明；顶部已注明完成，不能按正文旧待办重复启动 |
| [BN 统计诊断结果](../ped2023_bn_statistics_pilot_20261008.md) | native BN225 的 source running statistics 与当前 patch statistics；WT 改善、TC/ET 退化 |
| [BN 与 IN 的互补审计](../ped2023_bn_complement_audit_20261008.md) | 两种源模型与已有观察的缓存诊断；未得到可靠融合规则。科学 review 已有，final 回执缺失 |
| [BN背景共识检查](../../outputs/ped2023_bn_context_20261011/README.md) | 同24例、固定BN225参数和running统计；24例完成，TC基本持平、WT/ET下降，无整体收益，区别于上面的统计替换实验 |

BN input-style 只有协议、实现和 CPU 检查，没有对应性能报告；具体入口见[实验索引](../EXPERIMENT_INDEX.md)。

## 10 月 6 日已完成的探索

这些文件的详细产物已从 E 盘迁入本项目的 `artifacts/experiments/` 和 `artifacts/models/`，不是缺失了同名 `outputs/` 目录。

| 文档 | 内容 |
|---|---|
| [Source TC 形状与病例浏览](../ped2023_source_tc_shape_review.md) | 复用 99 例 source 结果做可视化、好坏病例与形状分析 |
| [边界与 native TENT/SAR](../ped2023_boundary_and_norm_tta_20261006.md) | 99 例 native IN TENT/SAR 结果及边界验证；SAR 解释需连同下篇阅读 |
| [SAR 恢复审计](../ped2023_sar_audit_20261006.md) | 738 次恢复、阈值行为、真实 patch 重放和无恢复诊断 |
| [局部边界可行性](../ped2023_boundary_feasibility_20261006.md) | MRI/特征最小瓶颈路径，固定 WT/ET 的局部机制检验 |
| [99 例连续性](../ped2023_continuity_full99_20261006.md) | 固定 WT/ET、ED→NC 连续性扩张；不作为完整四分类适应 |
| [MRI 同修改量对照](../ped2023_mri_matched_controls_20261006.md) | 连续性方法与 margin/geometry/random 等强对照，审查扩张解释 |
| [留出模态边界](../ped2023_heldout_boundary_20261006.md) | 用未参与候选生成的 MRI 模态检查 NC/ED 界面；已完成机制检验 |
| [完整四分类边界](../ped2023_fourclass_boundary_20261006.md) | 开放 BG/ET 后的边界选择器，检查 WT/TC/ET 共同变化 |
| [跨轴观察与学习](../ped2023_axis_tta_validation_20261006.md) | 3D 轴置换、完整体积推断和 decoder 参数适应 |
| [已有 checkpoint 教师](../ped2023_snapshot_teacher_validation_20261006.md) | 175/200/225 轮快照提供的额外类别证据；明确多 checkpoint 资源 |
| [对侧结构与 T2/FLAIR](../ped2023_anatomy_contrast_validation_20261006.md) | 正常参照、病例内信号先验的两种固定检验 |
| [N4 偏场校正](../ped2023_n4_input_validation_20261006.md) | 输入偏场拟合和冻结模型推断；无参数 TTA |

## 设计演变与早期局部实验

| 文档 | 内容与边界 |
|---|---|
| [HIR-TTA 设计总记](../gli_to_ped2023_hir_tta_design.md) | 早期机制假设、验证要求及后续修订入口；不是当前最终算法 |
| [v2：源组织参照](../gli_to_ped2023_tta_revision_v2.md) | 需要 GLI 数据的提案，已撤回，仅保留设计历史 |
| [v3：单模型多尺度](../gli_to_ped2023_tta_revision_v3.md) | 改为 checkpoint 与当前病例，12 例局部 pilot 已完成 |
| [v4：文献与监督粒度](../gli_to_ped2023_tta_literature_revision_v4.md) | 多视图、细类/区域伪监督的设计与梯度核查 |
| [v5：先验实验](../gli_to_ped2023_tta_prior_experiments_v5.md) | 多轮先验/校准实验；有 TC 校准收益，独特先验贡献未建立 |
| [v6：组织分群](../gli_to_ped2023_tta_semantic_support_v6.md) | 组织分群、语义拒绝及局部验证 |
| [v7：三种教师](../gli_to_ped2023_three_teacher_experiments_v7.md) | GMM、图传播、多裁剪教师的实际模型更新实验 |
| [v8：冻结 head](../gli_to_ped2023_frozen_head_candidates_v8.md) | IN-only 学习和局部候选选择，12 例局部结果 |
| [从几何转向语义证据](../gli_to_ped2023_geometry_tta_rethink_20261006.md) | 解释为何固定 WT/ET 的 TC 涨分不够，后续跨轴检验已完成 |

## 九月份自定义框架与拓扑对照

| 文档 | 内容 |
|---|---|
| [AutoDL 自定义源训练](../autodl_source_training.md) | 六阶段 BN、300 epochs、FP32；不是 native IN250/BN225 的训练协议 |
| [高级 TTA 兼容实现](../advanced_tta_protocol.md) | 自定义 BN 框架的 SAR/VPTTA/TEGDA、状态和复现边界 |
| [Original U-Net 对照](../original_unet_ablation.md) | 五层原始拓扑的 3D 适配与 BN 对照 |
| [Cicek 3D U-Net 对照](../cicek_unet3d_ablation.md) | 2016 架构、计算/显存对照和 Kaggle T4 探针 |

## 阅读结果时如何判断状态

`verification.json` 的 `passed` 通常表示实现、文件或指标核验通过，不能代替方法效果判断。旧 `REPORT.md` 的联合门槛失败与当前“相对 Source 有收益、值得作为候选”可以同时成立。旧报告和核验文件保留原文；导航只注明它们现在的用途。

部分带日期的`docs/*.md`也已被实验核验文件记录哈希，例如10月8日候选重新评估。遇到旧的“当前”“下一步”先看本页和HANDOFF中的后续状态；不要为更新导航而重写这些历史报告或修改其核验哈希。
