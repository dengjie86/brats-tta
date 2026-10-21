# 历史自定义框架说明

本页归档原 README 的历史命令。当前研究与维护入口见 [仓库首页](../../README.md)。

# BraTS GLI -> SSA / PED Source + TTA

本仓库保留当前实验主线：使用 BraTS 2023 GLI 训练四分类源域模型，并在 SSA 和 PED 目标域上比较 Source-only、TENT、SAR、VPTTA 和 TEGDA 3D 兼容实现。

## 当前标准模型

| 项目 | 设置 |
|---|---|
| 输入 | 4 个模态，顺序为 `t1n, t1c, t2w, t2f` |
| 网络 | 六阶段 plain 3D U-Net |
| 特征通道 | `[32, 64, 128, 256, 320, 320]` |
| 参数量 | `31,199,796` |
| patch | `128 x 128 x 128` |
| 输出 | `background, NCR/NET, edema, ET` 四个互斥类别 |
| 归一化 | `BatchNorm3d`，affine 开启 |
| 训练精度 | FP32，AMP/TF32 关闭 |
| 训练计划 | 300 epochs，每轮 250 iterations，batch size 2 |
| 优化器 | SGD + Nesterov，初始学习率 0.01，poly decay |
| 保存 | 每 20 轮、best、latest、last |

唯一源模型配置是 [configs/source_brats_gli_4class_bn.yaml](../../configs/source_brats_gli_4class_bn.yaml)。旧三通道 InstanceNorm 基线、旧 continual/multi-round TENT 和无法追溯的早期兼容试验已经移除；当前高级 TTA 统一由独立入口运行。

## 安装

建议使用 Python 3.10-3.12，并先安装与显卡匹配的 PyTorch：

```powershell
python -m pip install -e ".[dev]"
```

HD-BET 建议放在独立环境中：

```powershell
python -m pip install -e ".[brain-extraction]"
```

## 标签和评估

- GLI/SSA 使用 `brats_modern`：原始值 `0,1,2,3`。
- PED 2024 使用 `brats_ped_2024`：ET=1、NET=2、CC=3、ED=4。
- 模型输出始终先转换为 ET/TC/WT 区域，再与目标域标签计算 Dice。
- PED 的五值标签不会被有损改写为成人四分类训练标签。

## 源域训练

AutoDL 启动脚本会生成 manifest、记录完整日志、监控 GPU，并保存周期权重：

```bash
DATA_ROOT=/root/autodl-fs/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
WORK_ROOT=/root/autodl-tmp/brats2023_gli_4class_bn_31m_6stage_300_fp32 \
NUM_GPUS=1 \
bash scripts/autodl_train_source.sh
```

完整说明见 [docs/autodl_source_training.md](../autodl_source_training.md)。

## 目标域 manifest

SSA：

```powershell
brats-prepare-manifest `  --root E:\dataset\ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2 `  --train-output D:\results\manifests\ssa_raw.json `  --val-fraction 0 `  --label-schema brats_modern
```

PED Kaggle 版本：

```powershell
brats-prepare-manifest `  --root E:\dataset\BraTS-PEDs2024_Training `  --train-output D:\results\manifests\ped_raw.json `  --val-fraction 0 `  --label-schema brats_ped_2024 `  --skip-incomplete
```

`--skip-incomplete` 只排除缺少模态或标签的病例，不修改原始数据。当前 PED Kaggle 数据中缺失 T1c 的病例应保持排除，不再从其他来源补文件。

## PED 脑提取

PED 原始图像未去颅骨，而 GLI 源模型使用脑提取输入。正式 Source/TENT 必须共用同一份 HD-BET 派生脑掩膜：

```powershell
brats-extract-brain-masks `  --manifest D:\results\manifests\ped_raw.json `  --output-root D:\results\preprocessed\ped_brain_masks `  --output-manifest D:\results\manifests\ped_brain.json `  --weights C:\Users\Yuuhi\hd-bet_params\release_2.0.0
```

命令只生成掩膜和派生 manifest，不改原图或分割标签。SSA 已经脑提取，不额外运行 HD-BET。

## Source-only

目标域 Source-only 使用统一评估入口，支持逐病例 JSONL、断点续跑、设置指纹、显存和耗时记录：

```powershell
brats-evaluate-tta `  --checkpoint D:\models\best.pt `  --manifest D:\results\manifests\ssa_raw.json `  --output-dir D:\results\source\ssa `  --methods source `  --device cuda `  --patch-size 128 128 128 `  --overlap 0.5 `  --sw-batch-size 1 `  --no-amp
```

## TENT

当前只保留与四分类 BatchNorm 模型匹配的标准 TENT：

- 每个病例开始前恢复源模型 affine 参数和 Adam 状态；
- 禁用 BatchNorm running statistics，使用当前 patch 统计；
- 只更新 BatchNorm affine scale/shift；
- 对 softmax 四分类输出最小化 categorical entropy；
- Adam，学习率 `1e-3`，weight decay 0；
- 每个 patch batch 更新 1 次；
- 目标标签只在预测完成后计算 Dice，不参与适配。

```powershell
brats-evaluate-tta `  --checkpoint D:\models\best.pt `  --manifest D:\results\manifests\ssa_raw.json `  --output-dir D:\results\tent\ssa `  --methods tent `  --device cuda `  --patch-size 128 128 128 `  --overlap 0.5 `  --sw-batch-size 1 `  --tent-lr 0.001 `  --tent-steps 1 `  --no-amp
```

每个方法会保存 `*_cases.jsonl`、`*_summary.json`、`*_run_settings.json` 和 `progress.json`。更换 checkpoint、manifest、patch 或 TENT 参数时必须使用新输出目录，避免混合旧结果。

## SAR、VPTTA 和 TEGDA

高级方法使用独立入口，每次只运行一种方法。默认采用 continual 状态，并保存固定病例顺序、逐病例指标、运行设置和可恢复适配状态：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m brats_tta.cli.evaluate_advanced_tta `
  --checkpoint D:\models\best.pt `
  --manifest D:\results\manifests\ssa_raw.json `
  --output-dir D:\results\advanced\sar\ssa `
  --method sar `
  --device cuda `
  --patch-size 128 128 128 `
  --overlap 0.5 `
  --sw-batch-size 1 `
  --hd95
```

- SAR：四分类 voxel-wise reliable entropy、SAM-SGD、只更新 BatchNorm affine。
- VPTTA：四模态 3D Fourier prompt、AdaBN、容量 40 的 continual memory bank。
- TEGDA：ADIC、bottleneck feature bank 和质量引导 mean teacher 的 3D compatibility port；正式默认采用论文的全参数更新、Adam `1e-4`、dropout `0.5`、10 次随机推理、长度 10 的 feature bank 和 90 分位筛选。本地 pilot 会显式使用较省显存的 `--tegda-scope norm`。

TEGDA 不是官方网络上的逐行复现，因为当前源模型没有其训练期 dropout，且官方 3D 发布代码混用了 2D feature-update 模块。论文将整例缩放到 `128^3` 后适配一次；本项目保留原空间并在 `128^3`、overlap `0.5` 的每个滑窗 patch 上适配，以便和现有 Source/TENT 公平比较，因此不能直接复现论文表格中的绝对数值。完整差异、推荐参数和正式实验要求见 [docs/advanced_tta_protocol.md](../advanced_tta_protocol.md)。固定小样本试跑脚本为 [scripts/run_advanced_tta_pilot.ps1](../../scripts/run_advanced_tta_pilot.ps1)。

AutoDL 全量运行使用 [scripts/autodl_evaluate_advanced_tta.sh](../../scripts/autodl_evaluate_advanced_tta.sh)。脚本先自动执行 1 例 TEGDA 全参数显存测试，再分别保存 SAR 官方恢复版、SAR 无恢复消融、VPTTA 和 TEGDA 3D compatibility port；支持直接在原输出目录断点续跑，不应给不同参数共用同一目录。

## 验证

```powershell
& 'D:\Environment\miniforge3\envs\cv311\python.exe' -m pytest -q
& 'D:\Environment\miniforge3\envs\cv311\python.exe' -m compileall -q src
```

## 参考

- Isensee et al., nnU-Net, Nature Methods, 2021.
- Wang et al., Tent: Fully Test-Time Adaptation by Entropy Minimization, ICLR 2021.
- Niu et al., SAR: Towards Stable Test-time Adaptation in Dynamic Wild World, ICLR 2023.
- Chen et al., Each Test Image Deserves A Specific Prompt, CVPR 2024.
- Zhou et al., TEGDA: Test-time Entropy-Guided Domain Adaptation for Medical Image Segmentation.

