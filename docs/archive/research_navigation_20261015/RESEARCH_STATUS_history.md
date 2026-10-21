# 历史研究判断与阶段记录

2026-10-15整理前的历史快照。文中的“当前”“运行中”“下一步”属于记录当时，不能作为执行指令。当前状态只读[根目录HANDOFF](../../../HANDOFF.md)。链接已调整为归档位置；逐字原文保存在[维护备份](../../../artifacts/maintenance/navigation_20261015/before.zip)。

---

# GLI→PED TTA研究进度

跨Source阶段读数：[已完成BN225/BN250各24例Dice](../../../outputs/ped2023_cross_model_transfer_20261015/interim_readout/REPORT.md)。相对各自Source，WT/TC/ET分别−.02271/+2.37210/−.08118pp和−.10188/+1.08397/−.05765pp；TC区间BN225高于零、BN250跨零。720项独立计数核验通过，HD95及完整复核尚待主队列，IN200成绩未读。用户要求非Goal不持续等待：后台队列继续，已停止工具侧长等待。

新增任务（10月15日10:38北京）：用户要求检验其他模型。[跨Source固定24组检查](../../../outputs/ped2023_cross_model_transfer_20261015/PROTOCOL.md)已启动BN225/BN250/IN200，先IN250精确复现预检，再自动预测、独立评价与报告。BN冻结running statistics，仅适应affine；无阈值或步骤调整。属于两个训练分支的checkpoint检查，无独立架构/泛化结论，尚无新成绩。原Goal已完成的状态只对应上一轮方法收敛，不代表这个新增任务已结束。

Goal工具已确认`complete`，对应本轮主方法收敛与交付完成；科学结论的限制仍如下保留。

更新：2026-10-15。开发主方法已确定为union2补充teacher + soft_extent八步IN学习 + no_come输出。本轮路线选择、组件删除、98组改进验证、必要学习对照、既有基线汇总及缓存交付均完成，无运行队列。完整算法、证据与复现命令见[主方法说明](../../GLI_PED_TTA_METHOD.md)。独立泛化、WT/ET稳定收益、BN迁移和创新性仍未确认。Goal状态以实际工具记录为准。

## 本轮探索经过（按阶段记录）

以下保留当时判断，其中的启动、下一步与未定方法均为历史状态；当前选择见上方方法说明。

新[组件删除与简化研究](../../../outputs/ped2023_decoder_ablation_20261014/INTERPRETATION.md)98组×10臂已于08:39北京完成，980新mask独立重建及2940区域覆盖通过。spatial_context去掉COME和末端筛选/保护，相对soft_stable Dice−.24711/+.63307/−.03135pp；TC更好、WT小幅成本、ET距离尾部恶化，保留为较简单并行开发候选。删除context学习TC−3.05772pp，学习仍必要。下一步检查teacher视图可靠性，未定最终方法。

[原生视图可靠性诊断](../../../outputs/ped2023_context_view_reliability_20261014/INTERPRETATION.md)已于10:42北京完成392状态/23360次网络调用的精确复现。context视图支持未超过置信度的同数量选择，不追加门控。转向[IN自然分区学习](../../../outputs/ped2023_in_subspace_20261014/PROTOCOL.md)：48组encoder/decoder两臂已启动，其余teacher、soft目标及8步预算不变，尚无新性能；不自动扩98。

18:33更新：[IN分区](../../../outputs/ped2023_in_subspace_20261014/INTERPRETATION.md)及[概率KL](../../../outputs/ped2023_probability_distillation_20261014/INTERPRETATION.md)均已完成并保留负对照，不扩98。KL同48组相同spatial输出TC−1.12042pp、HD95+.23746mm(n46)，old24/new24均下降，去最大损失后仍−.58679pp。48状态/192mask/576区域通过；WT/ET没有稳定补偿收益。

[loss checkpoint研究](../../../outputs/ped2023_loss_checkpoint_20261014/INTERPRETATION.md)已在20:53:57完成，96状态/384mask/1152区域核验通过。同48组best8/short4相对原spatial TC Dice−.41317/−1.87931pp，TC HD95+.12369/+.44282mm(n46)；WT/ET小幅改善但ET距离主要由00032推动，无ET GT误报增加。保留预算/选择消融，不扩98或扫patience。best8不是验证早停，部署仍需8步。

[分模态背景观察](../../../outputs/ped2023_modality_context_20261014/INTERPRETATION.md)22:26:14完成同48组528完整推断/144mask/480输入重建/432区域。均值teacher对旧teacher TC−2.49656pp，CI[−4.42732,−.86957]；全一致性及平均输入也下降，三臂不扩展。缓存GT事后分析发现大量旧正确编辑被撤回，单观察仍有额外正确类别候选；候选覆盖不能当成部署收益。

[保留旧编辑的建议补充](../../../outputs/ped2023_modality_context_20261014/proposal_union/INTERPRETATION.md)23:47:20完成192mask/576区域。union2对旧teacher TC+.41769pp，CI[+.13348,+.78902]、两分组均正；WT−.15245、ET+.08567且ET尾部敏感。confidence也TC+.40637，未建立两票独有优势。

10月15日00:05:47，[固定旧学习器接入对照](../../../outputs/ped2023_modality_context_20261014/proposal_union/decoder_transfer/INTERPRETATION.md)完成96旧mask复现/192新mask/576区域。union2对旧spatial/stable TC+.18309/+.20284pp，去最大收益后+.05574/+.06720；WT小幅下降、ET仍有取舍，保留为新学习的output-only对照。

[补充teacher学习](../../../outputs/ped2023_teacher_complement_learning_20261015/INTERPRETATION.md)05:00:41完成96状态/384mask/1152区域。union2完整对旧soft_stable TC+.56553pp，CI[+.07381,+1.24321]，去最大收益仍+.31002；WT−.02290、ET+.07353。old24/new24 TC−.03233/+1.16340；学习对同teacher仅输出的+.36270及对confidence的+.11258区间均跨零，不称机制已证实。

[固定union2完整98](../../../outputs/ped2023_teacher_complement_full98_20261015/INTERPRETATION.md)08:46:35完成：复用48，新增50全部学习；450完整前向、300新mask/900区域/400输入复核通过，49代码/37记录冻结。完整对旧soft_stable TC+.78169pp，去最大收益仍+.66012；对同teacher仅输出学习贡献+.29444pp且CI高于零。完整对Source TC+9.17759，简单spatial+9.80229，但后者ET距离有尾部成本。WT/ET平均增益仍不稳定，confidence学习仅48组。

[输出组件删除](../../../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/INTERPRETATION.md)09:06:07完成196新mask/588区域及末端保护恒等。开发主路线确定为union2+soft_extent+no_come，删除COME和冗余stable，保留margin/presence。对Source TC+9.35708pp、WT−.13202；对旧soft_stable TC+.96119，去最大收益仍+.84011。WT/ET取舍和开发暴露仍明确保留；删presence增加两例ET全漏，故不删除。

[最终必要对照与基线汇总](../../../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/INTERPRETATION.md)09:40:25完成，98主mask恒等/196控制mask/588独立区域及4阶段exit0/7终态hash通过。同输出下IN学习TC+3.00925pp，新teacher进入学习再增.29916pp。八外部家族分开IN/BN及98/24汇总。09:50:25交付重放98份mask逐值一致，无COME/GT/model/CUDA；所有队列终态，不再追加规则或训练。

## 已确定的基础

- 主Source是native四分类nnU-Net，IN、Dice+CE、GLI训练epoch250；BN best225作为单独对照。模型身份和路径见[MODEL_INDEX](../../MODEL_INDEX.md)。
- PED2023主开发队列是98个唯一MRI/标签组，99条记录；固定24例研究单独报告，不能混用两种均值。
- TENT、SAR、COME、VISTA有98组结果；MEMO、CoTTA、InTEnt、VPTTA完成固定24例。BN和IN各自相对自己的Source比较，移植限制和预算不同，尚不是统一预算的最终排名。见[首轮比较](../../../outputs/ped2023_literature_baselines_20261011/INTERPRETATION.md)及[扩展比较](../../../outputs/ped2023_expanded_baselines_20261011/INTERPRETATION.md)。

## 当前主方法

固定union2+soft_extent+no_come，保留margin/presence，删除独立COME分支和冗余末端stable。完整union2_stable与纯union2_spatial保留为取舍参照，不按病例选分支。

| 98组开发结果 | WT | TC | ET |
|---|---:|---:|---:|
| Source Dice% | 89.27230 | 48.84463 | 54.32808 |
| 旧soft_stable Dice% | 89.34488 | 57.24052 | 54.44104 |
| 主方法 Dice% | 89.14027 | 58.20171 | 54.46499 |
| 主方法−Source Dice，pp | −.13202 | +9.35708 | +.13691 |
| 主方法−Source配对HD95，mm | −.57316 | −2.68657 | −.16936 |
| HD95共同有效n | 98 | 91 | 84 |

对旧soft_stable TC+.96119pp，95%CI[+.49263,+1.51049]、去最大收益仍+.84011；WT−.20461pp及HD95+.15206mm是真实取舍。WT/ET区间与尾部不支持稳定全面改善。相同最终输出下学习TC+3.00925pp，新增teacher进入学习贡献+.29916pp；后者中位数0、old24为负，不能说各组和多数病例均改善。

TC/ET全漏2/5，Source7/7；无GT ET误报4955体素。全部98组已开发暴露；配对bootstrap区间为描述且未经多重比较校正。完整证据和冻结产物见[方法说明](../../GLI_PED_TTA_METHOD.md)。

## 前序机制与范围学习证据（历史）

动机是：模型对病灶组织的判断会受到周围输入上下文影响。保护Source预测病灶及邻近输入，改变远处背景；从两种观察的一致性与置信度产生teacher，再学习其建议。保护的是预测范围，不能把它称作真实病灶边界或已识别的因果干预。

TC收益主要表现为ED→NC重分类。学习会把少量teacher建议传播到其他位置；这种传播既贡献TC收益，也造成WT/ET损伤。现有证据支持研究这种取舍，尚不能证明儿童生物学差异就是原因，也不支持“只保留teacher修改点就足够”。

最近的学习约束并非都有效：类别对蒸馏、稳定组织锚点、可信类别顺序约束均未改善原plain；范围监督仅带来小幅WT收益和TC代价。负结果已保留，不能把原plain继承的TC收益归功于新增约束。

最新[Source统计回放学习](../../../outputs/ped2023_source_moment_replay_20261013/INTERPRETATION.md)于09:37北京完成24例预测、评价、独立复核和参照：

- `fixed_source`相对plain的TC Dice再增1.1376pp，但WT/TC配对HD95分别恶化2.6887/2.1662mm，尚不能替换主候选。
- `detached_current`相对plain的WT Dice增0.4512pp，但TC Dice下降2.7463pp；相对Source虽有较温和收益，仍未优于同24例的较强候选。
- 固定统计还改变归一化前向及有效参数方向，不能把结果全部归因于减少统计漂移。这两臂未自动扩展98组。

复核脚本的argmax赋值遗漏已通过独立`review_fix01`接续修复；48个状态、192份mask、576项独立区域指标通过。原失败日志保留，学习和预测没有重跑。

## 10月13日阶段路线与限制（历史）

1. 保留上下文学习中有益的组织重分类，同时控制额外误改和远处假阳性；先根据已完成队列的传播及归一化证据定义可检验机制，再做新的小规模学习实验。
2. 简化当前输出组合，用必要消融区分teacher、学习及保护各自贡献，形成能讲清楚的理论动机和方法。
3. 完善公平比较、BN可用性及独立验证。当前BN背景共识未成功，不能宣称迁移；跨版本MRI审计发现大量2023/2024重叠，现有PED2024还已有历史Source评分暴露，不能直接称独立测试集。见[数据审计](../../../artifacts/maintenance/source_moment_replay_20261013/target_cross_version_identity/REPORT.md)。

创新性、WT/ET稳定收益和独立泛化仍未确认。统计回放与MRI审计已结束；新[范围/组织学习队列](../../../outputs/ped2023_extent_factor_learning_20261013/README.md)于15:07北京完成固定24例软/硬两臂、匹配输出对照、评价、独立复核、报告及influence。23代码/162记录继续冻结，9项完成回执哈希及7阶段exit0已核对；不重复运行。

新报告的soft_extent对Source Dice为−0.04707/+4.58838/+0.38419pp，配对HD95为−2.27985/−1.02494/−0.21381mm（n24/23/22）。[完整解读](../../../outputs/ped2023_extent_factor_learning_20261013/INTERPRETATION.md)现已完成：对匹配plain_soft的TC再增0.75086pp，描述性95%CI[+.28423,+1.37449]，去最大收益仍+.54672；raw对plain TC+.77216pp。同24例soft对stable Dice−.35534/+1.16943/+.05892pp，有明确WT代价；WT HD95与ET均值对Source仍尾部敏感。hard matched ET−.20200pp的区间不跨零，不扩hard。

[固定soft完整98开发队列](../../../outputs/ped2023_extent_factor_full98_20261013/INTERPRETATION.md)于20:17完成，74新增状态/296mask/888区域独立复核通过。soft−plain_soft TC Dice+.60491pp（CI[+.34700,+.89120]）、HD95−.17027mm(n92)，new74单独+.55758pp，新增学习收益保留。但soft−stable三项Dice−.73375/−.41879/−.37852pp，WT明显较差，独立soft流程不替换原主候选。

[固定解码流水线学习器替换](../../../outputs/ped2023_extent_factor_full98_20261013/decoder_transfer/INTERPRETATION.md)21:14完成：原98 stable全流程重建逐值一致，新98mask/294区域独立复核通过。soft_stable对旧stable TC+.46236pp（旧24+.57982/new74+.42426），75例改善、18例下降、5例相同；去最大收益仍+.39670。WT/ET均值几乎维持，但不是等效性证明；TC/ET全漏病例集合2/5不变，无ET GT的FP体素6524→6363。保留它为优先后续验证的开发候选，不能声称更简单、独立泛化或概念创新。

本轮所有队列完成；后续重点是预设组件删除消融以简化流程、teacher纠错信息来源及BG/条件FG可靠性，以及公平预算/BN/数据身份核查后的外部验证。未启动新队列，不继续范围anchor或rho扫描。

本轮采用两次可被消息打断的长等待，16:30→20:45、20:55→21:35，没有中途模型进度查询。普通Python自动接续全部评价与复核；未创建桌面Scheduled或启用CLI原型。接续入口[HANDOFF](../../../HANDOFF.md)，Goal最终状态以工具为准。
