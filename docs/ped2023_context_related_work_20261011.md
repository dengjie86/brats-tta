# 背景上下文路线：相关工作与可支持的贡献

2026-10-11。核对三篇原论文和TRACE-Seg3D的相关作者代码后记录；这是方法定位与后续实验判断，不是新增性能结果。当前完整队列保持原plain配置，新增74组全部预测固定后才评价。本轮未读取这些病例的GT。

**当前结论：背景干预、上下文共识、单病例多视图学习、IN的空间耦合均有已有研究依据。我们的具体流程存在差异，但尚不能确认创新性。** 后续需要证明这些差异提供了什么额外信息和实际收益，而不是把通用机制重新命名。

## 已核对的相近工作

| 工作 | 与本研究有关的内容 | 对当前定位的影响 |
|---|---|---|
| [TRACE-Seg3D，2026年7月8日预印本](https://arxiv.org/html/2607.07038v1) | §3.2–3.3专门学习病灶/上下文潜变量，用源训练病例的context bank替换上下文并形成共识/不稳定性图；§3.3明确这些主要是审计输出，部署分割另有结构校准 | “保留病灶证据、改变上下文、观察稳定性”的宽泛概念已有直接前作；不能称首次提出 |
| [Single Image Test-Time Adaptation via Multi-View Co-Training（MuVi），2025](https://arxiv.org/html/2506.23705v1) | §2使用当前病例的轴向置换、伪标签及预测/特征一致性进行适应，每病例重置；原任务为乳腺MRI分割，主配置BN并另报IN实验 | “只用当前病例、多观察产生监督、再更新网络”也不是独立的新贡献；视图带来的纠错信息须单独证明 |
| [Tiling artifacts and trade-offs of feature normalization in the segmentation of large biological images，2025](https://arxiv.org/html/2503.19545v1) | 引言及§3、§5说明tile内IN统计让局部输出依赖整个tile；研究滑窗拼接和归一化取舍，实验证据来自显微图像 | IN的非局部依赖有直接理论背景；它不能证明PED错误由背景造成，或保证本项目干预有效 |

另外核对了TRACE作者代码的[固定版本](https://github.com/danleneurocom/Counterfactual-Representation-Network/blob/ce160a375e77518a5ebc187014b1e1d43edd0312/src/causal_mednext/causal_model.py#L681)：`backdoor_adjusted_logits`保留传入features和`z_d`，遍历context bank调用decoder后平均logits。此处只作实现结构确认，没有运行该仓库，也没有复现论文结果。该提交是本次读取时的main，不据此声称所有默认配置都等于论文配置。

## 我们的具体设定及差异

本项目使用已有IN250四分类Source，不训练额外的病灶/上下文编码器，不建立源域context bank。保护区来自**Source预测WT及10 mm邻域**，在更远位置构造blur/neutral两种输入，再按固定共识与置信条件生成纠错。代码见[context.py](../src/brats_tta/native/context.py)。

必须准确区分“预测病灶附近输入保持不变”和“真实病灶证据完全保持不变”：Source会漏检，保护区未必覆盖全部真实病灶；即使输入局部逐值相同，网络的中间特征仍会变化。我们没有得到可识别的因果变量，不能把这一输入干预直接解释成真实成像机制的反事实。

直接teacher冻结模型；可选学习扩展仅更新22层IN的7680个affine标量，每病例重置，在原图native融合输出上学习teacher修正，同时用Source extent作软锚点。四类输出开放，WT/ET没有硬锁。两者应分别报告：前者是测试时预测纠错，后者才包含参数适应。[当前实现](../src/brats_tta/native/README.md)、[完整学习协议](../outputs/ped2023_native_objective_full99_20261011/PROTOCOL.md)。

这些是资源约束和实现上的区别，尚不能独立证明方法新颖、具有独特机制或优于相近方法。不同数据集、模型和评价口径下的论文绝对Dice不作横向排名。

## 当前证据允许讲到哪一步

| 拟讲的内容 | 已有证据 | 尚不能推出的结论 |
|---|---|---|
| 当前病例的背景响应包含纠错信息 | [99例直接共识](../outputs/ped2023_context_full99_20261008/REPORT.md)相对匹配Source有TC收益，WT有代价；配对指标见[候选重评](ped2023_candidate_reassessment_20261008.md) | 不能保证全部组织和所有病例都改善 |
| 病例相关的修改程度有作用 | [修改预算审计](../outputs/ped2023_context_budget_audit_20261008/REPORT.md)对照相同总量的统一分配；借用teacher预算的几何对照已接近TC收益 | 尚未证明独特的精细体素选点能力；事后关联不是独立泛化或因果效应 |
| 学习会把修正传播到teacher未编辑位置 | [24例精确分解](../outputs/ped2023_pairwise_distillation_20261011/transfer_audit/INTERPRETATION.md)显示TC收益和WT损伤主要都在这部分 | 不能把全部传播当错误，也不能由贡献分解推断关闭某部分后的实际新分割成绩 |
| 双观察能筛选传播可靠性 | [同向支持检查](../outputs/ped2023_pairwise_distillation_20261011/directional_support_audit/REPORT.md)已完成，相对同数量margin没有统一优势 | 当前不据此构造新的通用门控或宣称可靠性已解决 |
| 归一化参与上下文响应 | [算子推导与CPU检查](../outputs/ped2023_bn_context_20261011/NORMALIZATION_MECHANISM.md)，以及已有统计交换诊断 | 不等于IN是唯一原因；[BN固定背景实验](../outputs/ped2023_bn_context_20261011/INTERPRETATION.md)24例无整体收益，不能宣称BN已适用 |

因此当前更准确的研究问题是：**在只有已有模型和当前病例的条件下，能否利用预测病灶邻域保留的上下文响应估计纠错方向与程度，并让可选参数学习带来额外收益？** 这是待检验的问题，不是新颖性声明。方法必须同时解释WT代价和ET病例波动，不能只围绕TC均值组织故事。

## 对接下来的实际决策

1. 完成已固定plain完整队列，与同病例Source、直接teacher和匹配预算对照比较；分别呈现旧24、新74和98唯一MRI组。全部99记录已有开发暴露，不通过重新分组把它们称为独立测试。
2. 若参数学习未增加有意义收益，就保留为消融，主结果采用直接预测纠错，并如实交代不更新模型参数。若有收益，则同时报告学习成本、WT/ET分布和共同有效HD95，不能只报平均Dice。
3. 新机制需要提供现有观察/Source margin之外的可检验信息。当前不因为简单门控或类别对投影具有数学性质就继续堆叠模块；也不在完整队列运行中修改协议。
4. 关于“首次”“可泛化到任意归一化”“已解释成人到儿童错误的根因”，现有证据均不足。此处只是针对最接近工作的核对，不是穷尽性新颖性检索。

本文不修改旧设计文档和论文引用，不追溯改写当时的灵感来源。新发现的前作从本次开始纳入比较，后续状态以[HANDOFF](../HANDOFF.md)及实验回执为准。
