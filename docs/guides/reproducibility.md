# 复现指南

先区分“查看结果”“重放保存的预测”“从 MRI 重新推理/适应”。它们需要的资源和计算量不同。

## 找到对应协议

```powershell
.\scripts\project.ps1 experiments show native-in250-single-view8
.\scripts\project.ps1 experiments show senior188-source-tent
```

输出包含模型身份、记录状态、入口、协议和报告位置。登记仅覆盖 5 个主要入口；其他研究从[实验索引](../EXPERIMENT_INDEX.md)进入。存在 `study.py` 不意味着可以直接执行，其初始化/prepare 可能写入原目录；先读协议和原 CLI。

## 必须配套的材料

| 材料 | 用途与位置 |
|---|---|
| Source checkpoint、模型结构/元数据 | [模型索引](../MODEL_INDEX.md)，native 还需要 plans/dataset/trainer |
| 模态顺序、标签映射、预处理与滑窗 | 各实验 `PROTOCOL.md`；IN native 和 Senior 不相同 |
| 源码版本与依赖身份 | `ORIGIN.json`、`preparation.json`、运行快照；以该实验实际文件为准 |
| 固定患者集合和顺序 | 原 manifest/队列文件；98、66、24 例不能混比 |
| 优化、统计量和状态重置方式 | 协议与 runner；BN stats 冻结和 TENT 当前 patch stats 不同 |
| 缓存、适应参数和最终 mask | 原实验目录；读取前检查 `CACHE_STATUS.md` |
| 指标定义和空区域约定 | 原评价代码/协议；训练选模指标和 WT/TC/ET 平均不能混用 |

## 三种操作

**查看已有结果**：直接读 REPORT、summary 和已保存 mask，无需 GPU；需要事后分析时优先复用已有结果。最新完整 Source/TENT 和已暂停方法的结果入口见[HANDOFF](../../HANDOFF.md)。

**缓存重放**：原 IN 主方法的[方法文档](../GLI_PED_TTA_METHOD.md)提供 `replay_main.py verify --case ...`。该入口依赖已有缓存，历史数组清理可能使其他实验缺少必要输入。路径检查通过不代表所有数组完整或哈希一致；旧 passed 回执也不能代替当前依赖存在性。

**新病例**：使用新增的 `project.ps1 adapt`，完整流程位于[公共 pipeline](../../src/brats_tta/pipeline/README.md)。它不依赖旧病例缓存或日期脚本，只处理明确指定的一例，输出目录必须全新。历史批量 runner 的接续仍需遵循各自协议。

先查看参数：

```powershell
.\scripts\project.ps1 adapt --help
```

以下是命令模板；尖括号中的模型目录与四个文件需要替换成真实路径。IN model directory 包含 `plans.json`、`dataset.json` 和 `fold_0/checkpoint_final.pth`：

```powershell
.\scripts\project.ps1 adapt --profile native-in250-single-view8 `
  --model '<IN250 model directory>' `
  --t1c '<case-t1c.nii.gz>' --t1n '<case-t1n.nii.gz>' `
  --t2w '<case-t2w.nii.gz>' --t2f '<case-t2f.nii.gz>' `
  --output 'outputs/my_new_run/case001'
```

原八视角学习用 `native-in250-eight-view8`；Senior best188 用 `senior188-single-view8` 且 `--model` 指向该 `best.pth` 文件。各 profile 的预处理、模态顺序与融合精度见公共包说明。GPU 协议使用默认 `--device cuda`；Senior 的 CPU 模式仅供检查。

输出包括 `prediction.nii.gz`、`source.nii.gz`、`teacher.nii.gz`、`affine.npz`、`teacher_additions.npz`、输入/代码/权重身份和完成/失败记录。预测入口不接收 GT；评价应在预测冻结后另行执行。运行失败会保留目录和错误记录，再次尝试必须使用新的目录。

本轮没有给暂停实验增加恢复命令。Senior188 原方法仍暂停在 66/98，只有用户明确要求继续才安排隔离接续。

## 环境和备份的边界

当前本机使用已有 cv311。本次没有创建环境或重新下载模型。仓库 `pyproject.toml` 是依赖范围，CI 的 CPU 依赖只保证其选择的测试用途，均不是所有历史 GPU 实验的环境锁。

源码 Git 可见、已提交、已推送是不同状态。本分支发布公共源码、测试及文档；大量权重、训练工程、manifest 和结果仍为本地资产。保存完整研究需保留 Git 工作树及本地资产备份，不能只复制 `.git` 或只克隆远端。不要原地覆写可能共享 NTFS 硬链接的 checkpoint。
