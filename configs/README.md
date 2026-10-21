# 配置目录

[返回仓库入口](../README.md) · [启动脚本](../scripts/README.md)

这里维护 **5 份自定义训练 YAML、路径配置和实验登记表**。[paths.yaml](paths.yaml) 集中记录当前模型、数据和结果位置，由 `brats_tta.paths.resource_paths` 读取；项目内路径相对仓库解析，可用 `scripts/project.ps1 check` 核对。

[experiments.yaml](experiments.yaml)登记5个主要实验的稳定ID、模型、状态快照和协议/报告/代码引用。`project.ps1 experiments list|show|check`读取它；`check --local`额外核对本地引用是否存在。它不是训练或适应参数配置，不复制科学阈值，也不自动启动旧runner。新实验如何组织参数和产物见[开发指南](../docs/guides/development.md)。

下面 5 份训练 YAML 由 `src/brats_tta/config.py` 校验、`src/brats_tta/cli/train_source.py` 使用。它们不是 native nnU-Net IN250 / BN225 的 plans，也不能用于直接重建其 checkpoint。

| 配置 | 实际用途 | 主要区别 |
|---|---|---|
| [source_brats_gli_4class_bn.yaml](source_brats_gli_4class_bn.yaml) | 九月份六阶段 plain 3D U-Net 源模型，以及后续拓扑对照 | `[32,64,128,256,320,320]`、BN、LeakyReLU、deep supervision；自定义 Dice+CE 包含背景 |
| [source_brats_gli_original_unet3d.yaml](source_brats_gli_original_unet3d.yaml) | 2015 U-Net 拓扑的 3D 无归一化对照 | 五层 `[32,64,128,256,512]`，ReLU，无 deep supervision |
| [source_brats_gli_original_unet3d_bn_control.yaml](source_brats_gli_original_unet3d_bn_control.yaml) | 与上一份配对的 BN 对照 | 保持五层拓扑，加入 BN；不是 Cicek 网络 |
| [source_brats_gli_cicek_unet3d_bn.yaml](source_brats_gli_cicek_unet3d_bn.yaml) | Cicek 2016 3D U-Net 的 BraTS 适配 | 四层 `[64,128,256,512]`、17 层 BN、ReLU；BN `track_running_stats=False` |
| [source_brats_gli_cicek_unet3d_no_bn.yaml](source_brats_gli_cicek_unet3d_no_bn.yaml) | Cicek 网络的无 BN 消融 | 同一拓扑，`norm: none` |

这 5 份配置均声明四模态、四分类、`128³` patch、300 epochs、250 iterations/epoch 和 FP32。具体验证周期、保存周期、BN 行为以各 YAML 和脚本覆盖项为准；存在配置不代表对应正式训练已经完成。

`data/manifests/...` 和 `outputs/source_...` 是配置中的输入/输出位置，不表示这些目录当前已经存在。运行脚本会按参数生成 manifest、覆盖配置并另存运行快照。

当前 native 模型的 `plans.json`、`dataset.json`、trainer 和 checkpoint 位于项目内 `artifacts/models/`，入口见[模型索引](../docs/MODEL_INDEX.md)。模态顺序、损失背景处理、预处理和 HD95 协议不能从这几份历史 YAML 套到 native 实验。
