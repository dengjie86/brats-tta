# 完整背景共识 TTA 流水线

[工程架构](../../../docs/reference/architecture.md) · [运行指南](../../../docs/guides/reproducibility.md)

`run_case(backend, image, geometry)` 接收明确的模型后端、预处理 MRI 和坐标信息，完成 Source → 10 次背景观察 → union2 teacher → 8 步 soft_extent → no_come。不读取 GT、不查找历史实验、不计算评价指标。`image` 是后端约定的四通道数组，返回原始 MRI 网格的预测、Source、teacher、affine 副本和学习记录。

## 模块职责

| 文件 | 职责 |
|---|---|
| [core.py](core.py) | 组织整条单病例流程 |
| [teacher.py](teacher.py) / [teacher_rules.py](teacher_rules.py) | 两种背景变换及各自四种保留模态观察；共识、支持数和冲突筛选；构造学习目标 |
| [backends.py](backends.py) | Native IN 和 Senior BN 两套更新实现，保持原数值协议 |
| [probability_fusion.py](probability_fusion.py) / [probability_replay.py](probability_replay.py) | Senior Gaussian 概率融合与对应 Jacobian/逐 patch 反向重放 |
| [projection.py](projection.py) | 匹配 Source 前景范围代理量的全局 bias |
| [decoder.py](decoder.py) / [pooling.py](pooling.py) | no_come：teacher 覆盖、范围融合，再筛选变化 |
| [support.py](support.py) / [presence.py](presence.py) | 六邻域形态支持、按类别转移数量匹配 Source margin、消失类别保护 |
| [state.py](state.py) | 检查非 affine 状态冻结；成功/异常退出均恢复调用前参数、buffer、梯度和训练标志 |
| [io.py](io.py) / [senior_model.py](senior_model.py) | 明确加载模型、输入和预处理；Senior 原网络拓扑 |
| [ORIGIN.json](ORIGIN.json) | 提取来源与产生时 SHA-256；不是运行时依赖 |

## 固定协议

| Profile | 模态顺序 | 观察/最终推断 | 学习 | 可更新参数 |
|---|---|---|---|---|
| `native-in250-single-view8` | t2f,t1n,t1c,t2w | native 八镜像、FP16 logit 融合 | 单视角、8 步、自己的单视角 Source reference | 7,680 IN affine 标量 |
| `native-in250-eight-view8` | 同上 | 同上 | 八镜像、8 步 | 同上 |
| `senior188-single-view8` | t1c,t2w,t2f,t1n | 无镜像、BF16 网络、FP32 概率融合 | 单视角、8 步、log(probability) 目标及概率 Jacobian | 5,888 BN affine 标量；冻结 Source running stats |

共同采用 Adam lr=.001、soft_extent gain=.5，128³ patch、50% 重叠。每例独立重置；没有有效编辑时跳过优化。Native 使用 plans 指定的非零裁剪/标准化；Senior 使用原全网格各模态非零 z-score。当前 IO 适配器限定对齐的 1 mm MRI；不会静默替用户重采样。

teacher 基础候选保留原阈值（置信度 ≥.8、margin ≥.2、保护区内双观察一致）。union2 新增候选要求至少两个保留模态分支支持且无类别冲突，并保留原基础修改。学习目标使用 FP32；decoder 的 teacher 合成保留 FP64 边界，不能随意统一 dtype。背景保护与变换复用 [native/context.py](../native/context.py)。

Native 融合 logits；Senior 融合 softmax 概率。二者反向公式不同，因此只共享目标函数与编排，不共用错误的融合梯度。decoder 保留只在分歧体素上计算的历史顺序。

## 运行与维护边界

Windows 用 `scripts/project.ps1 adapt --help`，安装后命令为 `brats-adapt-context`。CLI 明确接收四个 MRI 文件及模型路径；只写全新的 `outputs/` 子目录，保存输入/代码/权重哈希、预测、affine、teacher 增补及完成/失败记录。

这三个 profile 对应特定已核对的 Source 协议，不是任意 IN/BN 模型的自动适配器。CLI 不批量接续、不自动评价、不恢复暂停队列。Senior 的 CPU 模式仅供检查，不能用其 FP32 结果冒充历史 BF16 GPU 协议。

旧 runner 仍指向自己的冻结实现；本包没有反向改写旧哈希或替换旧实验。新增 CPU 测试和真实单例回归的范围、数值差异见[维护记录](../../../docs/reference/refactor-validation.md)。单例重构验证不构成全数据集复现或新的效果证据。
