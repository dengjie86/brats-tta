# 范围约束路线的相近工作与比较边界

2026-10-13。补充核对原论文、会议摘要和部分作者代码，服务于当前extent-factor实验解释。
不是新增性能结果，也不是穷尽性创新检索；原背景teacher定位见
[已有相关工作](ped2023_context_related_work_20261011.md)。

| 工作 | 实际相关内容 | 对本项目的影响 |
|---|---|---|
| [BAPA，2026预印本](https://arxiv.org/html/2609.21743v1) | 二分类dense VLM；类内高置信选择、FG/BG等总权、动态锚点，只更新文本残差。共享logit位移下证明一种熵塌缩充分条件 | 类别平衡和限制更新方向已有依据；该定理不自动适用于我们四类teacher平方损失与空间异质IN梯度 |
| [Shape Moments，MICCAI 2022](https://conferences.miccai.org/2022/papers/502-Paper2231.html) | 单病例分割，形状先验引导熵最小化，更新BN scale/bias | 结构先验与有限参数TTA是已有路线；我们当前每体素Source竞争保留不是训练得到的解剖形状先验 |
| [ReGA，MICCAI 2026](https://papers.miccai.org/miccai-2026/1043-Paper1573.html) | 骨盆CT，dropout区域/边界一致性、跨病例特征库、区域对比、EMA teacher | 与本研究的可靠性问题相关，不能称相同机制。公开代码使用IN；[本地接口审计](../artifacts/experiments/ReGA_interface_audit_20261013/AUDIT.md)记录移植差异，目前没有native结果 |
| [PIE，2026预印本](https://arxiv.org/html/2609.21412v1) | 冻结权重与统计，尺度/翻转后逆变换并均匀平均概率 | 支持保留冻结推断对照这一实验设计；论文§4明确缺少split、seed、适应超参数及延迟等复现信息，暂不作为稳定排名或理论保证 |

PIE正文Table 1的最终均值为0.7786，Table 2标作最终A1的均值为0.7816，二者
关系未充分交代。其表注关于不劣于Source的说法不能当作任意分割Dice保证。
作为我们自己的数学反例：一个真实FG体素的两个视图FG概率为.6和.1，均匀
平均后变成.35，argmax即可把原本正确的体素变错。这只是反驳一般保证，不是
重算该文数据。[论文原文](https://arxiv.org/html/2609.21412v1)

本项目Source已包含八镜像。此前[输入尺度24例](ped2023_input_scale_probe_20261008.md)
也保留了Source加0.8/1.25观察的均匀概率平均和插值往返控制，TC改善有限。
它不是PIE的28视图复现，但已有证据不足以支持仅为增加方法数量再扫描尺度。

当前更可检验的问题是：在相同上下文teacher、相同更新预算下，改变范围/组织
表示，是否比对原plain仅做相同输出变换更有效。已推导的
[损失几何](../outputs/ped2023_extent_factor_learning_20261013/loss_geometry/README.md)
表明soft使原始未编辑anchor变为1/4，编辑项还有交叉项，因此不能讲成单纯
加强范围正则，或把三个前景类别的分数差保留称为全网络解耦。

新增结果只在完整固定24例评价和独立复核后解释；相同输出控制、raw预测、强
组合参照及空预测均需保留。若新增学习没有实际贡献，结论就是该学习变换未
带来收益，不能把继承自teacher/plain的TC提升换名归功于新模块。
