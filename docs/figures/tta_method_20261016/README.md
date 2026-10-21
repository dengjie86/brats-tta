# 当前方法图解

2026-10-16。按当前冻结实现绘制，图示基于IN250主方法。

| 图 | PNG | 可缩放SVG |
|---|---|---|
| 完整架构 | [architecture.png](architecture.png) | [architecture.svg](architecture.svg) |
| 实际背景处理 | [background_example.png](background_example.png) | [background_example.svg](background_example.svg) |
| 10个观察的模态组合 | [modality_pairs.png](modality_pairs.png) | [modality_pairs.svg](modality_pairs.svg) |

背景图使用已有病例BraTS-PED-00003-000的四模态原始MRI和已保存Source分割，经同一native预处理后，调用当前`make_contexts`作CPU变换。仅展示FLAIR的同一切片，所有图使用相同显示窗；粉色轮廓为Source预测WT。没有读取GT，没有加载模型或重新推断，没有修改原实验数组。

## 背景变换

以Source预测WT为起点，在原生1 mm网格上计算三维欧氏距离d：

- d≤10 mm：保护区，四模态像素逐值不变。
- 10<d<20 mm：原图与处理背景平滑混合。
- d≥20 mm：使用处理后的背景。

令`t=clip((d−10)/10,0,1)`，处理权重`w=t²(3−2t)`。每模态新图为`(1−w)x+wB`，原始零值和保护区再次强制恢复。

对于blur，非零掩膜m定义为`x != 0`，先算`b=Gσ(x·m)/max(Gσ(m),1e−6)`，其中三维Gaussian σ=4体素。然后在`m & (d≥20)`内，将b的均值和标准差重新匹配到原图x，得到B。对于neutral，B直接取同一区域原图的中位数。无Source WT或任何模态远处非零体素少于1,000时，代码返回原图副本。

修改的是远处背景，teacher候选在保护区内筛选。学习阶段仍输入原始MRI，以teacher目标更新少量IN affine参数；背景观察只负责产生纠错证据。一致性是筛选条件，不保证建议正确。

## 实现与检查

- [背景处理与双观察teacher](../../../src/brats_tta/native/context.py)
- [保留单模态输入组合](../../../outputs/ped2023_modality_context_20261014/study_modality.py)
- [union2规则](../../../outputs/ped2023_modality_context_20261014/proposal_union/PROTOCOL.md)
- [完整方法与结果](../../GLI_PED_TTA_METHOD.md)
- [绘图脚本](draw_method.py)与[单例输入检查](verification.json)

本次核对native crop/轴序与旧geometry一致、四模态保护区逐值不变、原始零值不变，并人工检查图形布局。没有运行新的性能实验。
