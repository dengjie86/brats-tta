# AutoDL 源域模型训练

本文对应当前的四互斥类源域实验，不是 README 中原先的三通道区域输出基线。

## 固定实验设置

- 配置：`configs/source_brats_gli_4class_bn.yaml`
- 标签：`background, NCR/NET, edema, ET`，原始值必须是 `0, 1, 2, 3`
- 输出：4 通道 softmax logits
- 网络：nnU-Net-style plain six-stage 3D U-Net，features 为 `[32, 64, 128, 256, 320, 320]`
- 参数量：`31,199,796`，约为原 16.55M 五阶段模型的 1.88 倍
- 下采样：5 次，`128³` patch 的 bottleneck 为 `4³`
- 归一化：`BatchNorm3d`；多卡时启用 `SyncBatchNorm`
- patch：`128 x 128 x 128`
- 每卡 batch：2
- 训练：300 epochs，每个 epoch 250 iterations，严格 FP32（AMP 和 TF32 均关闭）
- 数据加载：训练 12 个 worker、验证 2 个 worker；训练预取 4 个 batch/worker、验证预取 1 个 batch/worker，保持 worker 常驻
- CPU：16 核配额下每个 PyTorch 进程默认 1 个计算线程，避免 worker 和底层线程过度抢占
- 源域增强：0.33 前景采样、LR 翻转、`0.8--1.2`/`±15°` 仿射、Gamma 和弱高斯噪声
- 线性强度 scale/shift 关闭

不要把 BraTS PED 的五值标签直接交给这个四类 head，否则模型定义和本次实验不一致。

## 环境检查

在服务器上执行：

```bash
cd /path/to/brats-tta
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
python -m compileall -q src
```

AutoDL 镜像通常已经带匹配驱动的 PyTorch。不要在未核对 CUDA 版本时直接覆盖它。

## 数据和输出位置

把 BraTS 2023 GLI ZIP 解压到数据盘，例如：

```text
/root/autodl-tmp/datasets/brats2023_gli/
  ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
    BraTS-GLI-00000-000/
    ...
```

`DATA_ROOT` 可以指向包含病例目录的任意上级目录，manifest 扫描器会递归查找病例。
训练输出默认写入
`/root/autodl-tmp/brats2023_gli_4class_bn_31m_6stage_300_fp32`，不写进仓库。

本配置在 AutoDL RTX 4080 SUPER 32 GiB 上完成过一次 `128³`、batch 2、严格 FP32 的完整
前向、反向和 SGD step，PyTorch peak allocated 约为 7.95 GiB，peak reserved 约为
9.24 GiB。每轮训练还会记录实际峰值显存，启动脚本每 5 秒将 GPU 利用率、显存、功耗
和温度写入 `gpu_utilization.csv`。

## 启动训练

单卡启动：

```bash
cd /path/to/brats-tta
DATA_ROOT=/root/autodl-tmp/datasets/brats2023_gli \
  WORK_ROOT=/root/autodl-tmp/brats2023_gli_4class_bn_31m_6stage_300_fp32 \
NUM_GPUS=1 \
bash scripts/autodl_train_source.sh
```

脚本默认使用 64 个固定验证病例，每 25 轮验证一次，以控制租用时长；训练完成后应使用
`best.pt` 对完整验证集单独评估。若要训练期间验证全部病例：

```bash
VALIDATION_CASES=all VALIDATE_EVERY=10 \
DATA_ROOT=/root/autodl-tmp/datasets/brats2023_gli \
bash scripts/autodl_train_source.sh
```

多卡服务器使用：

```bash
NUM_GPUS=2 DATA_ROOT=/root/autodl-tmp/datasets/brats2023_gli \
bash scripts/autodl_train_source.sh
```

脚本通过 `python -m torch.distributed.run` 启动 DDP。`BATCH_SIZE` 是每卡 batch，默认 2。

## 断点续训

`EPOCHS` 始终表示目标总轮数，不是追加轮数：

```bash
DATA_ROOT=/root/autodl-tmp/datasets/brats2023_gli \
  WORK_ROOT=/root/autodl-tmp/brats2023_gli_4class_bn_31m_6stage_300_fp32 \
  RESUME_CHECKPOINT=/root/autodl-tmp/brats2023_gli_4class_bn_31m_6stage_300_fp32/run/checkpoints/latest.pt \
EPOCHS=300 \
bash scripts/autodl_train_source.sh
```

建议在 `tmux` 中运行，关闭浏览器后任务仍会继续。关键文件包括：

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
  run/checkpoints/epoch_0040.pt
  ...
  run/checkpoints/epoch_0300.pt
```

`launch_environment.txt` 保存最终命令、配置哈希、CPU/内存配额和 Git 状态；`pip_freeze.txt` 与
`nvidia_smi.txt` 保存软件及 GPU 环境。`gpu_utilization.csv` 用于检查数据供给导致的利用率波动。
`console.log` 保留进度条和异常输出；
`run/training.log` 保留完整结构化训练日志；
`history.jsonl` 每轮同步落盘训练指标、验证指标和本轮保存的权重文件。训练迭代和验证病例
均逐条记录。周期权重每 20 轮保存一次，同时始终保留可恢复的 `latest.pt`、验证最优
`best.pt` 和训练正常结束时的 `last.pt`。`training.log` 会记录实际精度、TF32 开关、
worker/预取设置和每轮峰值显存。

单个 31.20M checkpoint 含 SGD 动量时约 0.25 GiB。300 轮按每 20 轮保存会留下 15 个
周期 checkpoint，再加 `best.pt`、`last.pt` 和 `latest.pt`，权重总量约 4.5 GiB；建议为
训练输出至少预留 8 GiB。若服务器环境中仍出现 OOM，应先记录实际峰值和
cuDNN 配置；为了保持本次源域对照，不要静默缩小 patch 或替换网络结构。
