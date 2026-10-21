# GLI→PED TTA：确定的开发主路线

2026-10-15。当前主方法固定为 **union2补充teacher + soft_extent IN学习 + no_come输出**。它取代本轮开始时优先保留的`soft_stable`路线；完整`union2_stable`和纯`union2_spatial`保留为取舍对照，不按病例切换。停止本轮规则和参数搜索。

本文件给出已实现、已评估的研究方法与复现入口，不代表独立临床验证、BN迁移或创新性已获证明。

## 1. 固定协议

| 项目 | 固定值 |
|---|---|
| Source | native Dataset503四分类nnU-Net，IN，Dice+CE，epoch250；身份见[MODEL_INDEX](MODEL_INDEX.md) |
| 目标数据 | PED2023，98唯一MRI/label组；99记录中的121/137重复组只计一次 |
| 预处理 | native preprocessing，无额外HD-BET；不使用旧custom-model YAML载入native权重 |
| 通道 | `t2f,t1n,t1c,t2w` |
| 标签 | BG=0、NC=1、ED=2、ET=3；WT={1,2,3}、TC={1,3}、ET={3} |
| 更新 | 每病例从同一Source重置，只更新7,680个IN affine标量；8步Adam，lr=.001；无teacher编辑时遵循原流程0步 |
| 损失/推理 | 原native full Jacobian、编辑MSE及extent锚；`soft_extent` rho=.5；无KL替换、统计detach、loss选回或提前停止 |
| 评价 | both-empty Dice=1/HD95=0；one-empty Dice=0/HD95=null；HD95按双方共同有效病例配对并报告n |

98组均已有开发暴露，分组old24/new24/remaining50不是独立测试。置信度学习对照完成48组，BN与扩展基线完成原24组；不能混用样本数和Source。

## 2. 实际计算流程

1. **背景观察。** 固定Source预测并保护其WT及邻域，使用原`make_contexts`构造blur/neutral背景观察。再对每种背景分别保留一个模态原值、处理其余三个，得到8个补充观察；保护区内四通道及原零值保持。包括原2个背景观察，共10个context观察，网络权重相同。
2. **补充teacher。** 原双背景teacher使用同类、top≥.8、margin≥.2。保留其全部编辑；仅在旧teacher等于Source的位置，接纳至少两对模态观察支持且不存在其他修改类别冲突的新建议。每对内部仍用原blur/neutral门槛。新增概率取所有赞成该类的模态对等权均值，旧位置概率保持。旧accepted且与Source一致的位置允许被新增建议改变。所有四类同规则，无GT、病例ID或TC专属门控。
3. **IN学习。** 仅在新增索引处替换原目标构造的概率，其他位置目标逐值保持；使用原`soft_extent`八步学习。训练目标按原FP32计算，输出teacher概率使用冻结FP64均值。原Source概率和原生logits须逐值复现，非IN参数冻结，病例结束恢复Source。
4. **范围与teacher输出。** 对学习后raw概率按原规则拟合全局FG偏置，使平均max-FG/BG sigmoid量匹配Source；在accepted位置使用teacher概率。对与Source不一致的位置执行原Source/当前预测max-FG/BG空间池化，保留当前预测的FG类别选择。
5. **筛选与类别存在保护。** 原6邻接opening只确定每个Source→目标类别方向的保留数量，再按Source margin排序选取同数量体素。保留原四类presence guard，避免筛选额外抹去proposal已有类别。输出NIfTI标签。

已删除独立COME分支及末端stable-consensus保护。后者在本流程中是冗余操作，已通过24项独立合成检查和196份实际输出的恒等核验；删除presence则额外造成00021/00055两例ET全空，因此保留。

这不是更强raw-network extent正则的证明：原extent-factor几何审计表明raw损失/梯度缩放及交叉项仍存在。方法的主要开发证据是补充的类别纠错建议与后续学习，而不是“多个观察相互独立”或“更高一致性必然更正确”。

## 3. 完整98组结果

| Dice% / 差值pp | WT | TC | ET |
|---|---:|---:|---:|
| Source | 89.27230 | 48.84463 | 54.32808 |
| 旧soft_stable | 89.34488 | 57.24052 | 54.44104 |
| 主方法 | **89.14027** | **58.20171** | **54.46499** |
| 主方法−Source | −0.13202 | **+9.35708** | +0.13691 |
| 主方法−旧soft_stable | −0.20461 | **+0.96119** | +0.02396 |

主方法对Source配对HD95为WT−.57316、TC−2.68657、ET−.16936mm，n=98/91/84。对旧soft_stable为+.15206/−.22245/−.07679mm，n=98/96/86。**WT距离相对旧流程的代价真实存在**；选择较短流程没有以“三域全部显著提升”为前提。

TC相对Source的描述性95%CI为[+6.30683,+12.83677]pp，77改善/18下降/3不变；去最大收益仍+8.59460pp。相对旧soft_stable的CI为[+.49263,+1.51049]，去最大收益仍+.84011pp。所有病例保留；剔除单例仅作敏感性分析。

WT/ET Dice区间跨零。ET Dice去最大收益00104后相对Source为−.11751pp；WT/ET HD95的均值也受尾部影响。主方法TC/ET全漏2/5，Source7/7，旧soft_stable2/5；无GT ET误报体素4955，Source8968、旧soft_stable6363。这不是已解决所有漏检或稳定全面优势。

## 4. 必要对照支持什么

在完全相同的最终输出流程下：

| 对照差值 | TC Dice pp | 95%CI | TC paired HD95 mm |
|---|---:|---|---:|
| 主方法−新teacher但不学习 | +3.00925 | [+1.84012,+4.31910] | −.68567，n96 |
| 主方法−新teacher配旧学习器 | +.29916 | [+.04001,+.65497] | −.07763，n96 |
| 主方法−旧teacher/旧学习器的no_come | +.77971 | [+.33822,+1.31224] | −.23720，n96 |

新teacher仅用于输出贡献+.48055pp，进入学习再增加+.29916pp；这是固定路径的差分分解，不能把全部9.35708pp归给新增模块。额外学习贡献去最大收益仍+.17420pp，但中位数为0，old24为−.13522pp，不能说各组或多数病例都一致受益。

学习前的均值/全一致性新teacher曾显著下降；保留旧编辑才得到补充收益。KL、IN分区、最小loss选回等失败/取舍结果均保存。48组同数量confidence学习接近union2，不能声称投票有普遍独有优势。对应解释见[48组学习](../outputs/ped2023_teacher_complement_learning_20261015/INTERPRETATION.md)、[完整98](../outputs/ped2023_teacher_complement_full98_20261015/INTERPRETATION.md)、[删除对照](../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/INTERPRETATION.md)、[最终学习控制](../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/INTERPRETATION.md)。

## 5. 外部基线与成本边界

[基线汇总](../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/BASELINES.md)保留已完成的八个外部方法家族：TENT、SAR、COME、VISTA、MEMO、CoTTA、InTEnt、VPTTA；变体不重复计数。

98组IN主方法TC58.20171，COME48.97041，TENT10后输出45.33306，SAR关闭恢复48.84226，VISTA重置48.83595/连续46.97958。COME的WT/ET为89.54947/54.69935，有自身优势。同原24组IN主方法TC63.56226、Source59.41157、MEMO59.40904、CoTTA59.39519。BN225独立按其Source报告，不能跨Source排绝对分数名次。

历史TENT/SAR配自身Source；VISTA连续流、CoTTA增强门控未激活及各3D移植限制保留。计算预算、输出时机、病例历史不同，不能称等算力SOTA比较。TEGDA未完成native运行，未填成绩。新方法增加8个背景前向，但移除了独立COME适应；已有阶段墙钟包含缓存复用、复核和不同硬件状态，不能直接当作部署加速比。

## 6. 冻结结果与复现入口

主98份mask位于：

`outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/cases/<case_id>/no_come.nii.gz`

最终统计和外部方法汇总位于其`final_controls/`，当前方法使用的学习状态分别保存在48组学习目录及完整98研究的新增50目录。所有checkpoint、原始数据、历史失败和负对照保持原样；不要向冻结目录重复执行`prepare/execute`。

[最小缓存重放脚本](../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/delivery/replay_main.py)只加载Source/原teacher概率、补充索引概率、学习后logits、geometry、Source图像头及bias；**不读取COME预测文件，不读取GT，不做GPU前向或优化**。输入及代码身份在同目录`manifest.json`，交付核验在`replay_check.json`。

在仓库根目录、已有冻结缓存的本机执行单例只读复现：

```powershell
& 'D:\Environment\Miniforge3\envs\cv311\python.exe' `
  'outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/delivery/replay_main.py' `
  verify --case BraTS-PED-00003-000
```

省略`--case`核对全部98组。2026-10-15 09:50:25北京，完整交付核验已通过：98份mask逐值一致，GT/model forward/CUDA/COME input均为false，进程exit0。需要另存一份mask时使用`decode --case <id> --output <新的nii.gz路径>`，脚本拒绝覆盖已有文件。这是**已完成研究状态的缓存重放入口**，不是陌生MRI的一键部署CLI；使用原冻结研究Python依赖，不需要新Conda环境。

## 7. 结论边界

本轮已完成路线选择、组件删除、必要对照、98组改进验证及可复现交付。继续保留完整和纯spatial结果供复查，不按GT病例挑分支，不再追加规则。

全部病例为开发队列，区间为未作多重比较校正的配对bootstrap描述。PED2023/PED2024存在MRI重叠，不能仅按年份宣称独立测试。独立患者泛化、BN迁移、唯一机制和论文创新性仍未确立；这些是后续另行设计的研究问题，不属于本轮已证明的结论。
