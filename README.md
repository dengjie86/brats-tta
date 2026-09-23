# BraTS GLI Source Model and TENT

本仓库当前提交的实验主线只有两部分：

1. 在 AutoDL 上训练 BraTS 2023 GLI 四分类源域模型；
2. 在 SSA/PED 目标域上比较 Source-only 与 episodic TENT。

后续讨论的其他 3D U-Net 拓扑和高级 TTA 方法不属于本次提交。

## 源域模型

| 项目 | 设置 |
|---|---|
| 输入 | `t1n, t1c, t2w, t2f` 四个模态 |
| 网络 | 六阶段 plain 3D U-Net |
| 通道 | `[32, 64, 128, 256, 320, 320]` |
| 参数量 | `31,199,796` |
| 归一化 | `BatchNorm3d`，affine 开启 |
| 输出 | `background, NCR/NET, edema, ET` 四个互斥类别 |
| Patch | `128 x 128 x 128` |
| Batch size | 单卡 `2` |
| 训练 | 300 epochs，每轮 250 iterations |
| 优化器 | SGD + Nesterov，初始学习率 `0.01`，poly decay |
| 精度 | 严格 FP32，AMP 和 TF32 关闭 |
| 保存 | 每 20 轮保存，同时保留 `best.pt`、`latest.pt` 和 `last.pt` |

固定配置为 [`configs/source_brats_gli_4class_bn.yaml`](configs/source_brats_gli_4class_bn.yaml)，AutoDL 启动入口为 [`scripts/autodl_train_source.sh`](scripts/autodl_train_source.sh)。

```bash
DATA_ROOT=/root/autodl-fs/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
WORK_ROOT=/root/autodl-tmp/brats2023_gli_4class_bn_31m_6stage_300_fp32 \
NUM_GPUS=1 \
bash scripts/autodl_train_source.sh
```

脚本生成 manifest，保存最终配置、完整训练日志、环境信息、GPU 监控、每轮历史和 checkpoint。断点续训时 `EPOCHS` 表示目标总轮数：

```bash
RESUME_CHECKPOINT=/path/to/latest.pt \
EPOCHS=300 \
bash scripts/autodl_train_source.sh
```

详细说明见 [`docs/autodl_source_training.md`](docs/autodl_source_training.md)。

## 数据和标签

- GLI/SSA 使用 `brats_modern`：原始标签值为 `0,1,2,3`。
- PED 2024 使用 `brats_ped_2024`：ET=1、NET=2、CC=3、ED=4。
- 四分类源模型只使用 GLI 训练。
- 目标域评估统一把标签转换为 ET/TC/WT 区域，PED 标签不会被有损映射成成人四分类标签。
- PED 原始数据需要使用同一套脑提取预处理后再公平比较 Source 与 TENT。

建立目标域 manifest：

```powershell
brats-prepare-manifest `
  --root E:\dataset\ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2 `
  --train-output outputs\manifests\ssa_raw.json `
  --val-fraction 0 `
  --label-schema brats_modern

brats-prepare-manifest `
  --root E:\dataset\BraTS-PEDs2024_Training `
  --train-output outputs\manifests\ped_raw.json `
  --val-fraction 0 `
  --label-schema brats_ped_2024 `
  --skip-incomplete
```

## TENT

本实现与四分类 BatchNorm 源模型配套：

- 每个病例开始前恢复源模型的 BN affine 参数、Adam 状态和 GradScaler 状态；
- 所有网络参数冻结，只更新选定 `BatchNorm3d` 的 scale/shift；
- 禁用 BN running statistics，使用当前目标 patch 的统计量；
- 对四分类 softmax 输出最小化 categorical entropy；
- 默认 Adam，学习率 `1e-3`，weight decay 为 0；
- 默认每个滑窗 patch batch 更新一次；
- 目标标签只在预测完成后计算指标，不参与 TENT 更新。

Source-only 与 TENT 使用相同的滑窗、patch、重叠率和精度：

```powershell
brats-evaluate-tta `
  --checkpoint D:\models\best.pt `
  --manifest outputs\manifests\ssa_raw.json `
  --output-dir outputs\ssa_source_tent `
  --methods source tent `
  --device cuda `
  --patch-size 128 128 128 `
  --overlap 0.5 `
  --sw-batch-size 1 `
  --tent-lr 0.001 `
  --tent-steps 1 `
  --no-amp
```

可选指标：

```powershell
brats-evaluate-tta `
  --checkpoint D:\models\best.pt `
  --manifest outputs\manifests\ssa_raw.json `
  --output-dir outputs\ssa_source_tent_metrics `
  --methods source tent `
  --no-amp `
  --hd95 `
  --lesion-wise
```

每种方法分别保存逐病例 JSONL、汇总 JSON、运行设置和进度文件。设置文件包含 checkpoint/manifest 哈希，防止不同实验配置混入同一结果目录。

## 安装与测试

先安装与 CUDA 环境匹配的 PyTorch，然后安装项目：

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q src
```

## 主要入口

- `brats-prepare-manifest`：扫描 BraTS 数据并生成 manifest。
- `brats-train-source`：训练或恢复源域模型。
- `brats-evaluate-source`：验证源域 checkpoint。
- `brats-evaluate-tta`：运行 Source-only 和 episodic TENT。
- `brats-infer`：导出分割预测。

## 参考

- Isensee et al., [nnU-Net](https://doi.org/10.1038/s41592-020-01008-z), *Nature Methods*, 2021.
- Wang et al., [Tent: Fully Test-Time Adaptation by Entropy Minimization](https://openreview.net/forum?id=uXl3bZLkr3c), *ICLR*, 2021.
