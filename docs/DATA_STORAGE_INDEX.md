# 数据与磁盘占用说明

2026-10-16：本仓库新增清理21.17 GiB已结束实验的概率/logits与工具缓存，项目由89.37降至68.21 GiB（按路径累计）。原始数据、权重、学习状态、mask与指标保留；完整范围见[当前存储说明](STORAGE_STATUS.md)。下面的历史数据集数字保持其原盘点日期。

2026-10-13：[跨版本MRI审计](../artifacts/maintenance/source_moment_replay_20261013/target_cross_version_identity/REPORT.md)于07:53北京完成99条Source及518条目标。主/alternate分别82/80条目标的四MRI在Source非零支持区域内精确对应，其中79/77条存在异ID匹配；全图hash或mask不同不证明独立影像。保守剩余完整记录129/127条，其中129/126已有历史Source评分；alternate255是唯一完整未评分ID，仍未证明患者独立。原数据集未改，没有新GT数组、评分或推断，不重复audit。

2026-10-10 后续维护额外清理项目旧环境/工具缓存约0.65 GiB，并通过只读权重硬链接减少约1.86 GiB重复存储；全部75个权重路径保留。原始数据与既有实验数组没有在这轮删除，详见[维护回执](../artifacts/maintenance/maintainability_20261010/)。下方99.17 GiB是此前清理轮次的累计值，不含这次去重。

**当前位置：**项目文件已集中到 `D:/Workspace/deeplearn/3dunet`，原始数据集仍在原处。此前集中清理新增删除 32.742276 GiB，连同更早两轮累计约 99.17 GiB；之后才进行了上方0.65 GiB清理和1.86 GiB去重。E盘原项目目录已移入 `artifacts/`，详见[项目存储位置](STORAGE_LAYOUT.md)。本次文档更新将6个过时安装元数据文件原样归档，未删除文件。下文旧 E 盘占用、83份权重和“未执行”均对应历史阶段；现保留75个权重路径，旧249项候选已在集中清理时核验并执行。

[返回仓库入口](../README.md) · [实验目录](EXPERIMENT_INDEX.md) · [模型权重](MODEL_INDEX.md)

盘点日期：2026-10-10。原始数据集、原始 ZIP 与 `E:/dataset` 全部保留；用户随后允许清理不再复用的派生数据，执行范围和结果见下文。以下为文件逻辑大小，非逐文件物理分配大小。

## 为什么当前方法还引用 HD_TTA 目录

`HD_TTA_source_*` 是训练和备份时沿用的目录名称，不是模型架构名。当前背景共识读取 `artifacts/models/in_dicece/` 中的 native 四分类 IN nnU-Net Source（原 `HD_TTA_source_models_completed_20261002`）；BN 对照读取 `artifacts/models/bn_dicece/`（原 `HD_TTA_source_4class_bn_completed_20261004`）。它们是在 GLI 上训练的源模型，不是经过 HD-TTA 优化后的模型。

当前公共实现维护在 [native/context.py](../src/brats_tta/native/context.py)；产生既有实验结果的实现冻结在 [common.py](../outputs/ped2023_context_probe_20261008/common.py)：保护源预测 WT 及邻近输入，改变远处背景，再用两种观察的一致性和置信度生成教师。后续 IN 参数适应学习该教师。此前本地 HD-TTA 实验使用 compact/diffuse 概率假设、gate 和 selector；当前背景教师没有调用这套优化。两者共享部分源模型、native 推理工具和研究背景，文件夹名称不表示算法继承，也不构成新颖性证明。

用户于 2026-10-10 进一步明确：不再反复使用、可从保留原始数据重建的派生数据可以清理。原始 MRI/标签、正式模型和重建所需配置仍保留；转换后的训练副本、预处理数组及完成实验的概率缓存按具体依赖核查。预测概率需要原始图像、对应模型、代码和配置才能重算，不能仅凭原始数据集恢复；模型权重更不能当作普通预处理缓存。

## E:/dataset 为什么有 142.06 GiB

这里同时保存多个年份/队列，以及原始文件、解压版本、下载的预处理版本和本地生成的预处理版本。不能将整个目录理解为一份原始压缩包的解压结果。

| 目录 | 大小 | 已核实内容 |
|---|---:|---|
| [TEGDA_official_eval](E:/dataset/TEGDA_official_eval/) | 39.505 GiB | 本地生成的 TEGDA 预处理版本：GLI 26.965、PED2024 11.174、SSA 1.365 GiB；记录为 128³、图像范围 [0,1] |
| [ped_Training](E:/dataset/ped_Training/) | 24.586 GiB | 另一套 PED MRI/标签；PED2024 预处理记录将其作为 fallback，并实际补用了一个病例的 t1c |
| [BraTS-PEDs2024_Training](E:/dataset/BraTS-PEDs2024_Training/) | 22.869 GiB | PED2024 MRI/标签，作为上述 PED 预处理的主要输入 |
| [BraTS_PED_2023_Kaggle](E:/dataset/BraTS_PED_2023_Kaggle/) | 21.646 GiB | 当前 PED2023 的 1.890 GiB ZIP 与 19.756 GiB 解压影像同时保留，另有少量记录 |
| [TEGDA_official_raw](E:/dataset/TEGDA_official_raw/) | 12.470 GiB | GLI2023 原始 MRI/标签，是上方 GLI 预处理的输入 |
| [BraTS2021_Training_Data](E:/dataset/BraTS2021_Training_Data/) | 12.464 GiB | BraTS2021 MRI/标签，独立目录 |
| [TEGDA_BraTS2023_downloaded](E:/dataset/TEGDA_BraTS2023_downloaded/) | 6.309 GiB | GLI/PED 的 `preprocessed_data`、少量 `raw_recovery` 和相关记录；与本地生成的 official_eval 不应仅凭名称视为相同内容 |
| [BraTS2023_GLI_independent159](E:/dataset/BraTS2023_GLI_independent159/) | 1.475 GiB | GLI 独立159扫描子集，用于历史 Source 核验 |
| [SSA2023](E:/dataset/ASNR-MICCAI-BraTS2023-SSA-Challenge-TrainingData_V2/) | 0.740 GiB | SSA MRI/标签和少量临时产物；整个数据目录保留 |
| **合计** | **142.064 GiB** | 31,414 个文件 |

### 压缩包很小，但解压后可能很大

实际读取 [PED2023 ZIP](E:/dataset/BraTS_PED_2023_Kaggle/2023-brats-pediatric-full.zip) 的 ZIP 目录：文件大小 1.889823 GiB，包含 495 个文件，未压缩总大小 19.756241 GiB，与已解压 `.nii` 总量相符，约为压缩包的 10.45 倍。这里保留的是未压缩 `.nii`，不是 `.nii.gz`；ZIP 和解压内容同时存在，所以目录约为 21.65 GiB。

### 预处理版本是额外产物

已读取三个 `preprocessing_report.json`，确认以下输入输出关系：

- [GLI 报告](E:/dataset/TEGDA_official_eval/BraTS_GLI2023/preprocessing_report.json)：读取 `TEGDA_official_raw/GLI2023`，输出 GLI 的 TEGDA 预处理版本，记录 1,251 例。
- [PED 报告](E:/dataset/TEGDA_official_eval/BraTS_PED2024/preprocessing_report.json)：主要读取 `BraTS-PEDs2024_Training`，使用 `ped_Training` 作为备用；记录 261 例，并补用了 `BraTS-PED-00255-000` 的 t1c。
- [SSA 报告](E:/dataset/TEGDA_official_eval/BraTS_SSA2023/preprocessing_report.json)：读取 SSA2023，记录 60 例。

这些报告记录了输出 128³ 和 [0,1] 强度范围。本次没有重新预处理，也没有逐体素比较不同 PED 目录；不能声称 `ped_Training` 和 `BraTS-PEDs2024_Training` 完全重复。预处理会改变数组和压缩效果，体积更小的网格也不保证压缩文件更小。

## 历史盘点：集中清理前 dataset 之外的占用

首次清理前，E 盘另有 48 个 BraTS / HD_TTA / nnUNet 项目目录，共约 100.833 GiB：其中 `.npz` 64.870 GiB、独立 checkpoint 19.892 GiB，其余 16.072 GiB。这里不含 E:/dataset；也不含 D 盘仓库约 35.33 GiB 的 outputs。

例如 [停止的本地纯 Dice 训练](D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_source_dice_local_20261003/) 原约 28.50 GiB，包括转换后的 `nnUNet_raw` 12.48 GiB、`nnUNet_preprocessed` 15.52 GiB，以及权重和记录。后续核实来源后，已删除其中约 28.02 GiB 的可重建派生文件，余约 0.47 GiB 的权重、代码和记录。

## 历史第二轮：清理可重建训练副本与三组评测概率

在用户明确派生数据可以删除后，该轮删除 10,610 个文件、38.695795 GiB。此前两组概率另已删除 27.728517 GiB，两次累计 66.424312 GiB；当时249个小范围候选尚未执行，之后已执行。该轮结束时相关 E 盘项目目录估算约34.41 GiB，加 `E:/dataset` 约176.47 GiB，其中22个 `HD_TTA*` 目录约12.16 GiB。这些是迁入仓库之前的阶段值，不是现在的E盘项目占用。

| 位置 | 本次删除 | 保留 |
|---|---:|---|
| [旧本地训练](D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_source_dice_local_20261003/) | 28.02 GiB，10,016 文件：转换后的 MRI/二值标签、预处理数组/病例属性、GT 副本、少量 probe/partial 数据 | 权重、代码、plans、fingerprint、split、转换记录、日志 |
| [二分类 Dice+CE HD-TTA](D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_native_nnunet_20261002/) | 6.134 GiB，198 个 initial/refined 概率 | 全部代码、模型引用、指标、最终 mask、协议 |
| [四分类合并 WT HD-TTA](D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_4class_nnunet_20261003/) | 3.116 GiB，198 个 initial/refined 概率 | 同上 |
| [纯 Dice WT HD-TTA](D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_wt_pure_dice_20261004/) | 1.422 GiB，198 个 initial/refined 概率 | 同上 |

删除前读取原始 GLI ZIP 全部 6,255 个成员并完成 CRC 校验；5,004 份 MRI 与转换副本 SHA-256 一致，1,251 份 WT 标签逐体素等于原始 `seg > 0`，预处理 GT 副本也逐文件一致。原始 ZIP、转换代码、固定 plans 和患者划分均保留，没有重新训练或推理。

删除后核验：所有 10,610 个目标文件消失、执行日志与清单一致；4,159 个保留文件 SHA-256 不变（包含全部 83 份 checkpoint），原始 ZIP SHA-256 不变；`E:/dataset` 31,414 文件及当前仓库 `outputs` 6,708 文件的大小和 mtime 不变。没有对整个数据目录重新计算内容哈希。完整清单见 [本次核验](archive/storage_records/derived_data_cleanup_20261010.json)。

历史依赖：已完成的 [BN 源域验证](D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_GLI_BN_regions_20261005/) 读取过该训练缓存；[论文公式对照](D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_wt_paper_20261004/) 读取过纯 Dice initial 概率。以后重新推理/优化前须重建相应输入；既有分割和指标不受影响。保留的历史完成回执只证明当时完成，不能据此断言所有缓存仍存在。

重建训练数据见 [重建说明](D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_source_dice_local_20261003/DERIVED_DATA_CLEANUP_20261010.md)。不要直接启动旧训练入口来补缓存；原转换记录仍在，会检查已删除文件，而训练入口之后还会启动训练。当前 PED 背景共识与 IN 学习的代码/缓存没有清理。

## 历史第一轮：两组 HD-TTA 概率清理

用户明确指定后，删除以下两个实验中的 `initial_probabilities.npz` 和 `refined_probabilities.npz`，各 99 例 × 2 文件，总计 396 文件、27.728517 GiB。两个目录的代码、协议、指标、日志和最终分割图保留，分别剩余约 35.45 / 35.69 MiB。

- [2026-09-30 旧 HD-TTA](D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_20260930/)：删除 14.224284 GiB。
- [2026-10-01 axis0 对照](D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_axis0_20261001/)：删除 13.504233 GiB。

这一步操作前后 E 盘可用空间增加约 27.7293 GiB，与删除量相符；文件系统分配和同期系统活动会造成小差异。这一步完成时相关项目目录约 73.10 GiB，加未改动的数据集约 215.17 GiB；后续清理后的占用见上节。

已核验 1,770 个保留实验文件 SHA-256 不变、数据集 31,414 文件大小/mtime 不变；没有重算数据集全部内容哈希。完整删除文件清单及哈希见 [清理核验](archive/storage_records/legacy_hdtta_probability_cleanup_20261010.json)。

旧 diagnostics、params、params_round2、params_round3 的部分脚本读取这些概率，之后重跑须先重新生成概率；历史结果和脚本仍保留。没有查到当前仓库代码对这两组目录的直接引用，不将静态搜索宣称为覆盖一切动态依赖。

第一轮结束时，249个小范围候选因当时审批拒绝未执行，[旧清单](archive/storage_records/storage_cleanup_20261010.json)记录的是该时点。随后集中清理已重新核验并执行，当前以[迁移核验](../artifacts/maintenance/consolidation_20261010/verification.json)为准；不将旧阻塞状态作为现在的待办。
