# 源域基线设计与实验约定

## 1. 网络形状

对 `128³` patch，主干中的特征形状为：

| 位置 | 通道 | 空间尺寸 | 操作 |
|---|---:|---:|---|
| Encoder 0 | 32 | `128³` | 两个 `3³` 卷积 |
| Encoder 1 | 64 | `64³` | 首卷积 stride 2，再一个卷积 |
| Encoder 2 | 128 | `32³` | 同上 |
| Encoder 3 | 256 | `16³` | 同上 |
| Bottleneck | 320 | `8³` | 同上 |
| Decoder 3 | 256 | `16³` | `2³` 转置卷积、concat skip、两个卷积 |
| Decoder 2 | 128 | `32³` | 同上 |
| Decoder 1 | 64 | `64³` | 同上 |
| Decoder 0 | 32 | `128³` | 同上 |

所有 `3³` 卷积使用 padding 1。每个卷积后接 `InstanceNorm3d(eps=1e-5, affine=True, track_running_stats=False)` 和 `LeakyReLU(0.01)`。转置卷积之后不单独归一化，其输出先与 skip 拼接，再进入卷积块。卷积和转置卷积使用 Kaiming normal 初始化；归一化层的 gamma 初始化为 1、beta 初始化为 0。

输入的三个空间维必须能被 16 整除。训练 patch 和滑窗 patch 均由配置校验这一约束。

## 2. 区域定义和损失

输出是三个可重叠的二值区域，而不是四类互斥分割：

```text
ET ⊆ TC ⊆ WT
```

模型内部不加 sigmoid。每个深监督输出使用：

```text
L_region = L_soft-dice + L_BCE-with-logits
```

配置中的 `[1, 0.5, 0.25, 0]` 是相对权重，实际归一化为约 `[0.5714, 0.2857, 0.1429, 0]`。最低分辨率 head 保留在模型中但不参与默认损失，使网络结构与后续消融保持一致。

`hierarchy_weight` 默认是 0。也就是说源模型训练不额外加入层级正则，避免将实验变量偷偷放进 baseline；推理导出时可以进行确定性的集合并运算，保证最终 label map 合法。

## 3. 训练协议

默认配置采用：

- 每个 epoch 250 次参数更新，共 1000 epoch；
- SGD，初始学习率 0.01，momentum 0.99，Nesterov，weight decay `3e-5`；
- polynomial learning-rate decay，指数 0.9；
- 梯度范数裁剪 12；
- CUDA 上自动混合精度；
- foreground oversampling 0.33，随机翻转、强度缩放/平移和高斯噪声；
- validation 使用 50% overlap 的 Gaussian-weighted sliding window。

这是项目固定的第一版源域协议。任何改变都应另建配置和输出目录，不应覆盖这条 baseline。

## 4. InstanceNorm 对后续 TTA 的边界

`track_running_stats=False` 意味着训练和测试都会用当前病例、当前通道的空间统计量；模型中不存在可更新的 source/target running mean 和 running variance。因此：

- batch size 1 或 2 不改变 InstanceNorm 的统计定义；
- 逐病例强度偏移和尺度变化会被部分抵消，这通常提高小 batch 3D 分割的稳定性；
- 依赖 BatchNorm running-stat 重估或 AdaBN 的 TTA 不能原样使用，因为这里没有这类状态；
- `affine=True` 仍保留每通道可学习的 gamma/beta，因此熵最小化等方法可以选择更新 InstanceNorm affine 参数；
- 也可以更新卷积参数、adapter 参数、输入归一化或输出伪标签，InstanceNorm 并不会禁止这些 TTA 路线。

后续对 TTA 做比较时，需要明确报告更新参数集合、episodic/continual 协议、每病例步数、学习率和是否在每个病例前恢复源模型。仅写“更新归一化层”是不充分的。

## 5. 公平比较约定

建议至少保留以下不变量：

- 同一个 source checkpoint；
- 同一目标域病例顺序和 manifest；
- 同一预处理、滑窗 patch、overlap 和阈值；
- source-only 和所有 TTA 方法使用相同的后处理；
- 同时保存适应前与适应后的概率，区分模型提升和后处理提升；
- SSA 与 PED 分开报告，不把两个目标域混合求均值。

