# BN与既有IN观察的互补性：固定24例缓存诊断

2026-10-08。结论：新增BN提供少量新候选，新增12例的NC/ED软排序有改善；但跨两组的预设排序证据、全类别分歧选择和前景内部选择均未通过。**没有得到可部署的融合规则，GLI→PED域适应仍未解决。**

这是明确的事后、GT辅助开发诊断。同一原12/新增12，已有GT暴露；只读取既有概率和掩膜，没有网络前向、参数更新、新分割、阈值/权重拟合或新增Dice/HD95成绩。不能称独立测试或TTA成功。

读取Source IN、两背景观察、两尺度、六跨序列观察，以及独立BN Source和BN patch统计，共13份概率。BN是GLI预先选择的best225；不是通过本轮PED成绩选择模型。所有NC→ED分母均以**IN Source**为准，不与上一轮BN自身错误分母混用。

## 1. BN补充了多少此前没有的候选

至少一个观察argmax为真类，只是GT已知时的候选可用性；不是算法恢复率。旧十个IN观察已包含背景变化、尺度和跨序列变化。

| IN Source NC→ED错误 | 原12 | 新增12 |
|---|---:|---:|
| 错误体素 | 216,148 | 361,699 |
| 旧观察已有正确NC候选 | 75,566 | 48,519 |
| BN两观察独有新增正确NC候选 | 4,569 | 98 |
| 旧概率混合严格受阻 | 128,182 | 300,998 |
| 加入BN后仍严格受阻 | 124,453 | 300,877 |
| 新增候选/错误体素，池化 | 2.1138% | 0.0271% |
| 新增候选/错误体素，病例均值 | 1.5513% | 0.0204% |
| 加入BN后受阻比例，池化 | 57.5777% | 83.1844% |

受阻定义：Source及所有观察在同一位置均严格满足 `P(truth)<P(IN_Source_class)`。任意非负、类别共享且和为1的概率混合仍满足该不等式。它是特定缓存与混合族的充分限制，**不否定类别校准、特征学习、输入适应或新的观察**；ties不算受阻。仅argmax缺席本身不能推出同样结论。

| 以GT组织划分的IN错误 | 原12：错误/BN独有新增 | 新增12：错误/BN独有新增 |
|---|---:|---:|
| BG | 65,823 / 507 | 160,649 / 1,918 |
| NC | 388,027 / 4,935 | 474,509 / 294 |
| ED | 9,678 / 23 | 11,982 / 11 |
| ET | 16,407 / 183 | 47,490 / 1,547 |

高置信错误采用IN Source `P(ED)>=.9`：

| 项目 | 原12 | 新增12 |
|---|---:|---:|
| 错误数 | 192,986 | 329,564 |
| BN独有新增候选 | 4,280 | 82 |
| 扩展后严格受阻 | 114,249 | 283,116 |

## 2. 排序信息与硬候选覆盖不是一回事

查询固定为IN Source判ED、GT属于NC/ED，正类NC。下面是病例内AUROC的等权均值；单真类病例不能计算，有效数为6/9。没有用GT拟合任何分界值。

| score | 原12，n=6 | 新增12，n=9 |
|---|---:|---:|
| in_margin | 0.737547 | 0.698933 |
| bn_margin | 0.737009 | 0.816039 |
| bn_patch_margin | 0.715292 | 0.739631 |
| two_source_mean_margin | 0.731765 | 0.787284 |
| bn_minus_in_margin | 0.671344 | 0.744127 |

BN Source margin减IN margin的配对差及10000次病例bootstrap95%区间：

- original12：-0.000538 [-0.112846, +0.086185]；该组信号=False。
- additional12：+0.117105 [+0.006257, +0.279183]；该组信号=True。

新增组8/9有效病例BN排序更好，但00099从.2105到.9286，对平均增量贡献较大。原组3/6改善，00009从.9188降至.6645。逐例表如下；不按这些结果剔除病例。

两组池化AUROC IN为.7513/.4121、BN为.8731/.7082，与病例均值明显不同；病例大小、类别量和跨病例score偏移都可能影响池化结果，不能据此证明每例更可靠或完成概率校准。

## 3. 有分歧时，能否识别哪一个源模型正确

表中的“仅BN对/仅IN对/两者都错”覆盖所有四类分歧。后面的AUROC仅在**恰有一个正确**的GT条件子集中计算，正类为BN正确；它不能识别两者都错，也不是部署选择准确率。

| 范围 | 原12：仅BN对/仅IN对/都错 | 新增12：仅BN对/仅IN对/都错 |
|---|---:|---:|
| all_disagreements | 41,183 / 27,001 / 22,818 | 64,258 / 36,135 / 31,204 |
| foreground_disagreements | 19,845 / 4,992 / 4,087 | 19,320 / 15,510 / 13,316 |
| boundary_disagreements | 21,338 / 22,009 / 18,731 | 44,938 / 20,625 / 17,888 |

| 条件查询与score | 原组AUROC病例均值(n) | 新增组AUROC病例均值(n) |
|---|---:|---:|
| all_disagreements: bn_confidence | 0.652134 (12) | 0.528442 (12) |
| all_disagreements: in_error_confidence | 0.452281 (12) | 0.585463 (12) |
| all_disagreements: confidence_difference | 0.572832 (12) | 0.565526 (12) |
| all_disagreements: entropy_difference | 0.561184 (12) | 0.581138 (12) |
| all_disagreements: pair_log_odds_sum | 0.566804 (12) | 0.584201 (12) |
| foreground_disagreements: bn_confidence | 0.621817 (11) | 0.529068 (12) |
| foreground_disagreements: in_error_confidence | 0.391480 (11) | 0.492121 (12) |
| foreground_disagreements: confidence_difference | 0.519717 (11) | 0.513819 (12) |
| foreground_disagreements: entropy_difference | 0.507586 (11) | 0.520043 (12) |
| foreground_disagreements: pair_log_odds_sum | 0.459004 (11) | 0.505915 (12) |
| boundary_disagreements: bn_confidence | 0.615360 (12) | 0.575442 (11) |
| boundary_disagreements: in_error_confidence | 0.519677 (12) | 0.581152 (11) |
| boundary_disagreements: confidence_difference | 0.585815 (12) | 0.599169 (11) |
| boundary_disagreements: entropy_difference | 0.576748 (12) | 0.617533 (11) |
| boundary_disagreements: pair_log_odds_sum | 0.611887 (12) | 0.644875 (11) |

前景内部的confidence_difference只有.5197/.5138，不能支持按两个模型的最大概率差选择组织标签。它在原组比单BN置信度低.1021（区间[−.2034,−.0108]）；新增组差−.0152，区间跨0。熵差和pair log-odds也未表现出稳定的前景选择信号，不事后将其替换成主要方法。

两源模型在前景分歧中“仅BN对”总量较多，但这种池化计数不等于每例Dice优势，也不能为“在所有前景冲突选BN”提供独立验证。所有4×4×4类别/真值计数保留在audit.json；不只展示ED→NC的有利方向。

## 4. 逐例NC/ED排序

| case | cohort | IN | BN | BN patch |
|---|---|---:|---:|---:|
| BraTS-PED-00003-000 | original12 | 0.688692 | 0.674406 | 0.670446 |
| BraTS-PED-00004-000 | original12 | 0.947064 | 0.924879 | 0.952456 |
| BraTS-PED-00008-000 | original12 | null | null | null |
| BraTS-PED-00009-000 | original12 | 0.918802 | 0.664469 | 0.661115 |
| BraTS-PED-00042-000 | additional12 | 0.812138 | 0.828744 | 0.845012 |
| BraTS-PED-00044-000 | additional12 | 0.717534 | 0.803406 | 0.832902 |
| BraTS-PED-00046-000 | additional12 | 0.764234 | 0.796970 | 0.830970 |
| BraTS-PED-00055-000 | original12 | null | null | null |
| BraTS-PED-00057-000 | additional12 | 0.726222 | 0.827512 | 0.808939 |
| BraTS-PED-00061-000 | additional12 | 0.920905 | 0.976942 | 0.993330 |
| BraTS-PED-00062-000 | original12 | 0.717703 | 0.817979 | 0.762658 |
| BraTS-PED-00063-000 | original12 | null | null | null |
| BraTS-PED-00070-000 | additional12 | null | null | null |
| BraTS-PED-00072-000 | additional12 | null | null | null |
| BraTS-PED-00081-000 | original12 | null | null | null |
| BraTS-PED-00099-000 | additional12 | 0.210483 | 0.928617 | 0.297485 |
| BraTS-PED-00101-000 | original12 | 0.557281 | 0.687577 | 0.593107 |
| BraTS-PED-00112-000 | additional12 | 0.481543 | 0.600319 | 0.572758 |
| BraTS-PED-00120-000 | original12 | null | null | null |
| BraTS-PED-00121-000 | additional12 | 0.834530 | 0.873654 | 0.705501 |
| BraTS-PED-00129-000 | original12 | null | null | null |
| BraTS-PED-00140-000 | additional12 | null | null | null |
| BraTS-PED-00167-000 | original12 | 0.595742 | 0.652747 | 0.651967 |
| BraTS-PED-00170-000 | additional12 | 0.822811 | 0.708182 | 0.769781 |

## 5. 验证与后续决策

审计50.81s，独立复核40.81s，均exit0。24例输入/概率/空间/crop轴/旧科学文件hash核验通过；旧联合计数和IN AUROC精确回归。独立复核用直接不等式及显式八交集重建混合限制，用完整NIfTI重建分歧表，重算全部score/GT查询，并用独立ties累计法核对AUROC；六体素四类合成例覆盖两者都错与空正负类。

没有新增网络推断或分割，checkpoint和旧结果不变。review_verification.json是科学复核回执，final_verification.json另外记录交付文档。

停止把BN/IN固定概率融合或最大置信度选专家当作已得到支持的路线。软排序仍有局部信息，不能说BN完全无价值；但如何在无GT条件下确定可信语义与病例内决策界面仍未解决。

下一候选可利用BN源running statistics监督输入风格适应：固定源分类器，通过可逆、小参数输入变换产生旧概率集合之外的观察。其统计目标是否会把儿童组织组成错误拉向成人同样未知，需要单独固定协议、完整四类、两组验证和同预算控制。**本报告未实施或验证该候选，也未拟合类别先验。**
