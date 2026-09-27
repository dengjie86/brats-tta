# AutoDL 源域模型训练

本文记录当前唯一源域训练配置：`configs/source_brats_gli_4class_bn.yaml`。

## 固定设置

- 四分类：`background, NCR/NET, edema, ET`
- 六阶段 plain 3D U-Net：`[32, 64, 128, 256, 320, 320]`
- 参数量：`31,199,796`
- `BatchNorm3d`，多卡时转换为 `SyncBatchNorm`
- patch：`128 x 128 x 128`
- 每卡 batch size：2
- 300 epochs，每轮 250 iterations
- FP32，AMP 和 TF32 关闭
- SGD + Nesterov，初始学习率 0.01，poly decay
- 训练 12 workers，验证 2 workers
- 按患者编号划分训练/验证，纵向扫描不会跨集合
- 每 25 轮在完整验证集上评估并选择 best checkpoint
- 每 20 轮保存周期 checkpoint，同时保存 best/latest/last
- 增强：前景采样、左右翻转、仿射、Gamma、弱高斯噪声
- 线性强度 scale/shift 关闭

PED 五值标签不能直接用于该四分类 head。源模型只使用 GLI `brats_modern` 标签训练。

## 环境检查

```bash
cd /path/to/brats-tta
python -m pip install -e '.[dev]'
python -m pytest -q
python -m compileall -q src
```

AutoDL 镜像通常已经包含与驱动匹配的 PyTorch，不要在未核对 CUDA 版本时覆盖它。

## 启动

`DATA_ROOT` 可以指向包含病例目录的任意上级目录，扫描器会递归寻找 GLI 病例。

```bash
cd /path/to/brats-tta
DATA_ROOT=/root/autodl-fs/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
WORK_ROOT=/root/autodl-tmp/brats2023_gli_4class_bn_31m_6stage_300_fp32_patient_split \
NUM_GPUS=1 \
bash scripts/autodl_train_source.sh
```

脚本默认每 25 轮使用完整验证集。只有快速调试时才应限制验证病例数，例如：

```bash
VALIDATION_CASES=64 VALIDATE_EVERY=25 \
DATA_ROOT=/root/autodl-fs/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
bash scripts/autodl_train_source.sh
```

正式训练不要复用旧的 scan-level manifests。脚本的新默认工作目录带有
`_patient_split` 后缀，会重新生成包含 `split_unit: patient` 元数据的清单。

## 断点续训

`EPOCHS` 表示目标总轮数，不是追加轮数：

```bash
DATA_ROOT=/root/autodl-fs/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData \
WORK_ROOT=/root/autodl-tmp/brats2023_gli_4class_bn_31m_6stage_300_fp32_patient_split \
RESUME_CHECKPOINT=/root/autodl-tmp/brats2023_gli_4class_bn_31m_6stage_300_fp32_patient_split/run/checkpoints/latest.pt \
EPOCHS=300 \
bash scripts/autodl_train_source.sh
```

## 输出

```text
WORK_ROOT/
  launch_environment.txt
  pip_freeze.txt
  nvidia_smi.txt
  gpu_utilization.csv
  console.log
  manifests/gli_train_raw.json
  manifests/gli_val_raw.json
  run/config.yaml
  run/training.log
  run/history.jsonl
  run/checkpoints/latest.pt
  run/checkpoints/best.pt
  run/checkpoints/last.pt
  run/checkpoints/epoch_0020.pt
  ...
  run/checkpoints/epoch_0300.pt
```

`training.log` 记录每个 iteration、验证病例、学习率、精度模式、worker 设置和峰值显存。`history.jsonl` 每轮同步写入训练/验证指标与本轮 checkpoint 列表。

单个包含 SGD 动量的 checkpoint 约 0.25 GiB。300 轮按每 20 轮保存，加上 best/latest/last，建议至少预留 8 GiB 输出空间。
