# BraTS GLI → SSA / PED：3D U-Net 源域基线

本项目先固定并实现一个可复现的 BraTS GLI 源域模型，再在同一数据、推理和评估接口上扩展 SSA / PED 的测试时自适应（TTA）。当前版本已经包含源域训练所需的完整闭环：NIfTI 扫描、manifest、预处理、patch 训练、全体积滑窗验证、断点恢复、评估和 NIfTI 导出。

## 已固定的源模型

| 项目 | 设置 |
|---|---|
| 输入 | `4 × 128 × 128 × 128`，顺序固定为 `t1n, t1c, t2w, t2f` |
| 编码器 | `32, 64, 128, 256, 320` |
| 每个 stage | `2 × Conv3d(3³)`；后四个编码 stage 的首个卷积 `stride=2` |
| 解码器 | bottleneck `320`，随后 `256, 128, 64, 32`；转置卷积上采样和 skip concat |
| 归一化 | `InstanceNorm3d(affine=True, track_running_stats=False)` |
| 激活 | `LeakyReLU(negative_slope=0.01)` |
| 输出 | 3 个独立 logits，顺序 `ET, TC, WT`，使用 sigmoid 而不是 softmax |
| Dropout | `0` |
| 训练损失 | soft Dice + BCEWithLogits，带深监督 |
| 参数量 | `16,550,668` |

训练时四个 decoder 输出的空间尺寸是 `128³, 64³, 32³, 16³`，相对深监督权重为 `[1, 0.5, 0.25, 0]`；权重会在损失内部归一化。验证和推理只返回最高分辨率输出。

这个实现采用 nnU-Net 风格的 plain 3D U-Net、深监督、Dice+BCE、SGD+Nesterov 和 polynomial decay，但它是一个透明、可控的研究基线，不声称复刻 nnU-Net 的自动规划器和全部数据增强。详细约定见 [源模型设计](docs/source_baseline.md)。

## 项目结构

```text
configs/                       实验配置
src/brats_tta/
  cli/                         命令行入口
  data/                        manifest、NIfTI 预处理、dataset、增强
  models/                      PlainUNet3D
  losses/                      Dice、BCE、层级约束、深监督
  metrics/                     ET/TC/WT Dice 和层级违规率
  engine/                      训练器和滑窗推理
  utils/                       随机性、设备和原子 checkpoint
tests/                         单元与端到端冒烟测试
```

源模型、数据管线和推理引擎相互独立。后续 TTA 方法应放在单独的 `tta/` 包中，复用同一 checkpoint、target manifest 和滑窗推理接口，避免把源训练逻辑与目标域在线更新混在一起。

## 安装

建议使用 Python 3.10–3.12 和带 CUDA 的 PyTorch。先按显卡环境从 PyTorch 官方渠道安装 PyTorch，再安装本项目：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

当前机器即使只有 CPU 也能运行测试和小尺寸冒烟验证，但正式 `128³` 训练应使用 CUDA。

## 数据目录和标签

脚本同时识别现代 BraTS 命名和旧命名。例如：

```text
data/raw/BraTS-GLI-00001/
  BraTS-GLI-00001-t1n.nii.gz
  BraTS-GLI-00001-t1c.nii.gz
  BraTS-GLI-00001-t2w.nii.gz
  BraTS-GLI-00001-t2f.nii.gz
  BraTS-GLI-00001-seg.nii.gz
```

- `brats_modern`：标签值 `1, 2, 3`，其中 ET=`3`、TC=`1∪3`、WT=`1∪2∪3`。
- `brats_legacy`：标签值 `1, 2, 4`，其中 ET=`4`、TC=`1∪4`、WT=`1∪2∪4`。

不要仅凭文件名猜标签规范；第一次运行前应检查每个数据集 segmentation 的唯一值。四个模态必须已经共配准，shape 和 affine 不一致时预处理会立即失败。

## 从原始数据到训练

1. 扫描 GLI 数据并固定 train/validation split：

```powershell
brats-prepare-manifest `
  --root data/raw/BraTS-GLI `
  --train-output data/manifests/gli_train_raw.json `
  --val-output data/manifests/gli_val_raw.json `
  --val-fraction 0.2 `
  --seed 2025
```

2. 对每个模态在非零区域做逐病例 z-score，并缓存为 mmap 友好的 `.npy`：

```powershell
brats-preprocess `
  --manifest data/manifests/gli_train_raw.json `
  --output-root data/preprocessed/gli `
  --output-manifest data/manifests/gli_train_preprocessed.json `
  --label-schema brats_modern

brats-preprocess `
  --manifest data/manifests/gli_val_raw.json `
  --output-root data/preprocessed/gli `
  --output-manifest data/manifests/gli_val_preprocessed.json `
  --label-schema brats_modern
```

3. 检查网络，再训练：

```powershell
brats-inspect-model --config configs/source_brats_gli.yaml --forward-shape 32 32 32
brats-train-source --config configs/source_brats_gli.yaml
```

在 Kaggle 或其他临时运行环境中，可以直接从命令行覆盖路径和常用参数，不需要为每次运行复制 YAML：

```bash
brats-train-source \
  --config configs/source_brats_gli.yaml \
  --train-manifest /kaggle/input/brats/gli_train_preprocessed.json \
  --val-manifest /kaggle/input/brats/gli_val_preprocessed.json \
  --output-dir /kaggle/working/source_run \
  --batch-size 2 \
  --epochs 1000 \
  --iterations-per-epoch 250 \
  --learning-rate 0.01 \
  --num-workers 4 \
  --patch-size 128 128 128 \
  --amp
```

其他已有配置项使用可重复的 `--set` 覆盖；值按照 YAML 类型解析：

```bash
brats-train-source \
  --config configs/source_brats_gli.yaml \
  --set training.save_every=25 \
  --set training.validation_cases=20 \
  --set data.augmentation.noise_probability=0.1 \
  --set inference.overlap=0.625
```

优先级为：YAML < `--set` < 显式参数。拼错或不存在的 `--set` key 会直接报错，覆盖后的最终配置仍会写入输出目录的 `config.yaml`。

`configs/source_brats_gli.yaml` 默认 batch size 为 2。如果显存不足，改为 1 即可；InstanceNorm 不依赖 batch 维统计，因此不会出现 BatchNorm 在小 batch 下的统计退化。训练产物位于配置中的 `experiment.output_dir`：

```text
config.yaml
history.jsonl
checkpoints/latest.pt
checkpoints/best.pt
checkpoints/epoch_XXXX.pt
```

### Kaggle 双 T4 和 BraTS 2021 嵌套目录

项目支持 `torchrun` 启动的 DistributedDataParallel。`--batch-size` 表示每个进程、每张 GPU 的 batch；两张 T4 使用每卡 1，得到全局 batch 2。训练集由 `DistributedSampler` 划分，两张卡不会重复读取同一批样本：

```bash
torchrun --standalone --nproc-per-node=2 \
  -m brats_tta.cli.train_source \
  --config configs/source_brats_gli.yaml \
  --train-manifest /kaggle/working/brats2021_source_ddp/manifests/gli_train_raw.json \
  --val-manifest /kaggle/working/brats2021_source_ddp/manifests/gli_val_raw.json \
  --output-dir /kaggle/working/brats2021_source_ddp/run \
  --batch-size 1 \
  --device cuda \
  --amp \
  --set data.label_schema=brats_legacy
```

截图所示的 Kaggle 数据把每个模态放在名为 `*.nii` 的子目录，真实 NIfTI 再位于下一层。manifest 扫描器会递归聚合到共同的 `BraTS2021_XXXXX` 病例目录，并同时识别该镜像中的 `*_final_seg.nii` 与 `*_seg_new.nii` 标签，因此可以直接运行一键脚本：

```bash
DATA_ROOT=/kaggle/input/datasets/vietanh21/brats-2021-task-1-dataset \
  bash scripts/kaggle_train_2gpu.sh
```

如果 Kaggle 实际挂载为常见的 `/kaggle/input/brats-2021-task-1-dataset`，脚本也会自动回退到该位置。DDP 会把验证病例无重复地分配到两张卡，指标汇总、记录日志和保存 checkpoint 则只由 rank 0 执行；保存的模型没有 `module.` 前缀，可以直接用于单卡评估和推理。训练阶段默认每 10 个 iteration 输出一条普通文本日志。

双卡任务从 checkpoint 续训时，`EPOCHS` 仍表示目标总 epoch 数：

```bash
EPOCHS=1000 \
RESUME_CHECKPOINT=/kaggle/input/your-checkpoint/latest.pt \
  bash scripts/kaggle_train_2gpu.sh
```

断点续训：

```powershell
brats-train-source `
  --config configs/source_brats_gli.yaml `
  --resume outputs/source_brats_gli_plain_unet3d/checkpoints/latest.pt
```

## 评估和目标域 source-only 推理

```powershell
brats-evaluate-source `
  --checkpoint outputs/source_brats_gli_plain_unet3d/checkpoints/best.pt `
  --manifest data/manifests/gli_val_preprocessed.json `
  --output outputs/source_brats_gli_plain_unet3d/metrics.json
```

SSA / PED 无标签数据也使用 `brats-prepare-manifest --allow-missing-label --val-fraction 0` 和 `brats-preprocess` 建立 manifest。source-only 基线推理为：

```powershell
brats-infer `
  --checkpoint outputs/source_brats_gli_plain_unet3d/checkpoints/best.pt `
  --manifest data/manifests/ssa_preprocessed.json `
  --output-dir outputs/source_only_ssa `
  --save-probabilities
```

导出的 label map 会按 `ET ⊆ TC ⊆ WT` 做确定性层级修复；原始概率仍可通过 `--save-probabilities` 保存，便于后续公平比较 TTA 方法。

## 验证代码

```powershell
python -m pytest
python -m compileall -q src
```

测试覆盖网络规格、深监督反向传播、标签转换、NIfTI 预处理、无标签 batch、滑窗拼接、预测导出、checkpoint 和一次完整训练/验证/恢复流程。

## 研究依据

- Isensee et al., [nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation](https://doi.org/10.1038/s41592-020-01008-z), *Nature Methods*, 2021.
- 官方开源实现：[MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)。
