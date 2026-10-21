# 用户提供的四份代码与当前 native Source 对比

核对日期：2026-10-13。已阅读 `D:/TencentData/qqData/` 中的 `metrics.py`、`unet3d.py`、`train (1).py`、`trainer (1).py`，并对照保留的 native plans、trainer 和实际训练 debug 记录。四份外部源码没有修改；逐字节副本、SHA256 和 CPU 验证结果保存在 [核对目录](../artifacts/maintenance/qq_code_comparison_20261013/README.md)。

结论：这是一套不同的 BN U-Net 训练实现，不能当作只替换归一化层的对照。它的默认训练配置、模型结构和部分指标汇总方式均不同。单靠四份代码无法判断哪套实际分割效果更好，也不能据此替换目前的 Source。

下表中的“外部默认”只指这四份文件的代码默认值。入口还会读取 dataset/experiment YAML，并接受 CLI 覆盖；本次没有获得对应 YAML、数据加载器、loss 实现或 checkpoint，不能将默认值写成某次真实训练的配置。

## 模型结构

| 项目 | 外部默认 UNet3D | 当前 native 四分类 IN Source |
|---|---|---|
| 分辨率层级 / 通道数 | 5 层：32/64/128/256/512 | 6 层：32/64/128/256/320/320 |
| 参数量，CPU 实例化核对 | 21,365,348 | 31,199,796 |
| 归一化 | 18 个 BatchNorm3d | 22 个 InstanceNorm3d |
| 归一化 affine | 36 个 tensor，5,888 个标量 | 44 个 tensor，7,680 个标量 |
| 下采样 | MaxPool3d | stride=2 的卷积 |
| 上采样 | trilinear 插值 + 1×1×1 卷积 | ConvTranspose3d |
| 激活函数 | ReLU | LeakyReLU |
| Dropout | 默认 9 个 Dropout3d，p=0.3 | plans 中为 null |
| 深监督 | 单一最终输出 | native 训练使用深监督，推断关闭 |
| 输出尺寸处理 | 中心裁剪 skip，末端插值回输入尺寸 | native plans、滑窗和导出流程 |

外部代码的瓶颈 `forward_feat` 接口可以供其他方法调用，但四份文件本身不包含该域适应方法。`train (1).py` 还支持 `model_type=unet3d_smart`；其实现未提供，不能假定已有权重必定来自默认 BN 模型。

我们已经有与 IN Source 保持架构及训练协议一致的 native BN225 对照。要判断 IN/BN 对 TTA 的影响，应优先用这一配对，而不是同时改变网络、优化器和数据处理。外部默认有 Dropout，只说明具备随机 Dropout 模块，不能据此宣称某种依赖 MC dropout 的 TTA 已经兼容或有效。

## 训练与验证

| 项目 | 外部代码默认 / 可见行为 | 当前 native Source 已保存的配置 |
|---|---|---|
| 输入模态顺序 | 入口传给 loader：t1c/t2w/t2f/t1n | t2f/t1n/t1c/t2w |
| 优化器 | AdamW，AMSGrad=True | SGD，momentum=0.99，Nesterov=True |
| 初始学习率 / weight decay | 5e-5 / 1e-4 | 1e-2 / 3e-5 |
| 学习率调度 | CosineAnnealingLR，eta_min=1e-7 | PolyLRScheduler |
| loss | 传入 dice_weight=0.6、ce_weight=0.4、class_weights=[0.1,1.5,1,2] | Dice+CE 等权；Dice 排除 BG；深监督 |
| batch / patch | batch 默认 1；crop_size 可配置，loader 未提供 | batch 2，128³ |
| 训练长度 | 默认 300 epochs，每轮遍历 loader；patience=50 | 250 epochs × 250 iterations，不使用该早停规则 |
| 精度 | 可见循环没有 autocast/GradScaler；实际 dtype 依赖外部数据与模型 | CUDA AMP / GradScaler |
| 梯度裁剪 | norm=1 | norm=12 |
| 随机种子 | 默认 42 | 2025 |
| 数据划分 | 传入 train_ratio=0.8、seed；是否按患者划分未知 | 患者不重叠的 998/253 划分 |
| 每轮验证 | 直接调用 model(images)，具体是 crop 还是全图由未知 loader 决定 | 每轮 patch 验证；每 25 epochs 对全 253 例做 native 全卷验证 |
| best 选择 | NC/ED/ET 类别 Dice 的均值，经下述 batch 汇总 | 周期全卷验证的 native foreground class mean Dice；IN best/final=250 |

`WeightedCombinedLoss` 来自未提供的 `trainers.losses`。目前只能确认传入参数，无法确认背景是否参与 Dice、平滑项以及类别权重究竟作用于哪一项。预处理、强度归一化、增强、实际划分和裁剪也不能从 `create_dataloaders` 的调用推断出来。

当前 native trainer 专门将父类的 patch-EMA best 保存为 `checkpoint_best_patch.pth`，另按每 25 epochs 的全卷验证更新正式 `checkpoint_best.pth`。因此不能仅阅读原生父类就把当前模型的选择口径说成 patch EMA。组织类别平均 Dice 与 WT/TC/ET 区域平均 Dice也不是同一个指标。

输入通道顺序必须与各自训练时一致。把我们预处理的四通道 tensor 原样交给外部模型，会与它的默认入口顺序不同；对应训练配置确认前不应混用权重和输入。

## 已用实际函数复现的指标问题

### 1. batch 大于 1 时，外部 validate 会低报 Dice

`trainer (1).py:159–178` 累计的是每个 batch 的类别 Dice，以及每个 batch 的 WT/TC/ET 均值，最后却除以病例数。CPU 检查直接从 AST 提取并运行原 `Trainer.validate`，只替换 loader、模型和无关的 loss；没有执行训练入口。

| 同样 4 个完全正确的合成预测 | 正确 Dice | validate 输出类别及 WT/TC/ET Dice |
|---|---:|---:|
| batch=1，4 个 batch | 1.0 | 1.0 |
| batch=2，2 个 batch | 1.0 | 0.5 |

默认 batch=1 不触发这个低报问题。若实际验证使用 batch>1，应重算指标后再比较。类别 Dice 还先跨 batch 内体素汇总，不等价于逐病例平均；因此仅给最终结果乘以 batch size 不是一般情况下的完整修复。没有实际配置和预测，不能推断既有某次训练的成绩或 best 选择已经受影响。

### 2. 一个 helper 把类别平均误命名为区域 Dice

`metrics.py:245–252` 的 `calculate_tumor_metrics` 将 `(Dice_NC+Dice_ED+Dice_ET)/3` 命名为 WT，将 `(Dice_NC+Dice_ET)/2` 命名为 TC。这不等于对标签取并集后计算 Dice。

合成例中只交换 NC/ED，整个肿瘤范围完全不变：真正的 WT Dice=1，而这个 helper 的 WT≈0.33333336。外部 `softmax_output_dice_class4` 正确使用 WT={1,2,3}、TC={1,3}、ET={3}；Trainer 打印的区域分数来自这个正确函数，但仍有上述 batch 汇总问题。用于选 best 的 `tumor_mean` 实际是类别均值，它作为类别指标本身有意义，只是不能称为 WT/TC/ET 区域均值。

### 3. HD95 的空预测约定不同

| 情况 | 外部 metrics | 当前研究的 native 评价约定 |
|---|---|---|
| 预测和 GT 都为空 | 0 | 0 |
| 仅一方为空 | 最后三维 shape 的欧氏范数，未乘 spacing | null，比较共同有效病例并报告 n |
| 两方非空 | 双向表面距离合并后取 95 分位 | 同一表面距离定义；本次两组合成形状/spacing 结果一致 |

形状为 5×6×7 的一空一非空例，外部 HD95=√110≈10.4881，当前实现为 null。外部区域函数没有传递 spacing，默认单位体素；底层 `_hd95_binary` 虽接收 spacing，但其空区域惩罚仍不按 spacing 缩放。因此不能直接将两套 HD95 均值排在同一表中。源码注释中关于“官方等价”的表述，本次没有独立验证。

### 4. ET 的 500 体素规则只是另一个输出列

外部函数在整例预测 ET 总量小于 500 体素时，将 `ET_postpro` 对应的 ET 二值预测清空。一个完全正确的单体素 ET 合成例，原始 ET Dice=1，后处理列约为 1e-8。

Trainer 实际使用 `evaluate[:,2]` 的原始 ET，未使用第 4 列 `ET_postpro`。因此不能把这份 Trainer 的 ET 成绩误说成已经经过 500 体素处理。该规则也不是逐连通分量筛选。当前研究没有这一固定 ET 清除规则。

## 对当前研究的处理

- 保留当前 IN250 Source 和配对 BN225，四份外部代码作为结构/协议参考，不据此重新训练或替换 Source。
- 若后续比较外部模型，先取得与 checkpoint 配套的实际 YAML、数据加载器、loss 及划分记录，再统一标签、模态顺序、推断方式和评价口径；现有材料不足以比较真实性能。
- 不把外部默认的类别加权或 ET 阈值直接加入当前 TTA：前者属于源训练条件，后者会改变评价输出，需要独立动机与对照。

[CPU 检查](../artifacts/maintenance/qq_code_comparison_20261013/check.json)记录了两套结构计数、原 validate 的 batch 反例、类别/区域反例、ET 阈值例和 HD95 对照。全部使用 cv311 和合成数组，没有加载 checkpoint、真实病例或初始化 CUDA；四份原文件哈希保持不变。首次检查的参数量预期更正记录单独保留。当前 extent-factor GPU 队列、冻结代码与历史成绩均未修改。
