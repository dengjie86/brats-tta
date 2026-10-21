# 模型与权重索引

2026-10-11历史盘点为 **75 份 checkpoint**：正式 Source 备份在 `artifacts/models/`，运行副本、HD-BET 和旧 adapter/resume 状态随历史实验保存在 `artifacts/experiments/`。75份文件包含标准别名和训练快照，不等于75个独立模型。本次核对路径、大小和修改时间，未重新加载或重算全部权重哈希；完整身份以既有下载/模型核验为依据。

后续维护已将其中 **8 个字节完全相同的副本改成只读 NTFS 硬链接**，保留全部文件名、加载路径和 SHA-256，节省约 1.86 GiB 实际存储。相同组共享底层文件；续训保存到新的输出目录。逐项记录在 [去重回执](../artifacts/maintenance/maintainability_20261010/cleanup_events.jsonl)。下表大小是各路径逻辑大小之和，不能直接作为去重后的实际占用。

[项目文件总说明](PROJECT_FILE_GUIDE.md) · [当前逐文件清单与 SHA-256](model_inventory_current_20261010.json) · [模型目录说明](../artifacts/models/README.md) · [迁移和清理记录](STORAGE_LAYOUT.md)

当前背景共识使用 **native 四分类 IN / Dice+CE / final250**，BN 诊断使用 **native 四分类 BN / best225**。正式模型保留；已终止的本地 7 epoch、Kaggle 74 epoch 和三步测试共 8 份 checkpoint 已删除，不能再按旧清单寻找它们。

10月19日新增[用户五层BN U-Net第125轮正式备份](../artifacts/models/qq_unet_source_20261019/epoch_125/README.md)，与远程14份文件哈希一致。原75份为历史盘点数；此次未重新盘点全项目。新模型尚未进行PED/TTA评价，不替换当前TTA Source。

10月20日已增加[用户五层BN U-Net final300与best275正式备份](../artifacts/models/qq_unet_source_20261019/completed_300/README.md)，两份完整checkpoint及配套29文件校验通过。最新模型的PED未评价；125轮PED98例Source/TENT已完成。

10月21日新增[第二台Senior U-Net 128³完整工程备份](../artifacts/models/senior_unet128_source_20261020/README.md)：第238轮正常早停，区域最佳178（89.08984%）、组织最佳188（83.11838%）。含17个checkpoint路径及全部远程代码/配置/日志/验证记录，排除数据数组。已下载并核验远程全部 3,378 个文件（5.55 GiB 逻辑大小，去重后 4.56 GiB），三份主要权重 CPU 严格加载及 optimizer/scheduler/RNG 检查通过；详见 `backup_complete.json`。该模型的PED评测进展见下文，不替换现有TTA Source。

## 当前应使用哪个权重

10月21日21:43 Senior **best188**同98例PED2023 Source/TENT已完成，exit0；平均Dice55.71318%→61.67384%，+5.96066pp，588项运行时Dice检查通过，完成回执及汇总复核通过。[本次评测入口](../outputs/senior188_ped2023_source_tent_20261021/README.md)。原包预处理/无镜像与QQ125不同，不能当作仅换权重的严格消融。

| 模型 | 用途 | 实际入口 |
|---|---|---|
| **native 四分类 IN / Dice+CE / epoch 250** | **当前 GLI→PED2023 背景共识、IN 参数适应的 Source** | [checkpoint_final.pth](D:/Workspace/deeplearn/3dunet/artifacts/models/in_dicece/run/nnUNet_results/Dataset503_BraTS_Classes/nnUNetTrainerBraTS250__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth) |
| native 四分类 BN / Dice+CE / best epoch 225 | norm-only 对照；已完成的 BN PED 24 例诊断使用它 | [checkpoint_best.pth](D:/Workspace/deeplearn/3dunet/artifacts/models/bn_dicece/run/nnUNet_results/Dataset503_BraTS_Classes/nnUNetTrainerBraTSBN250__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth) |
| native WT 二分类 IN / Dice+CE / epoch 250 | 早期二分类 WT / HD-TTA 对照；不能直接输出 NC、ED、ET | [checkpoint_final.pth](D:/Workspace/deeplearn/3dunet/artifacts/models/in_dicece/run/nnUNet_results/Dataset502_BraTS_Binary/nnUNetTrainerBraTS250__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth) |
| native WT 二分类 IN / pure Dice / epoch 250 | pure-Dice HD-TTA 评价；best 为 225，但既有 PED 实验使用 final 250 | [checkpoint_final.pth](D:/Workspace/deeplearn/3dunet/artifacts/models/wt_pure_dice/run/nnUNet_results/Dataset502_BraTS_Binary/nnUNetTrainerBraTSDice250__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth) |
| 九月份自定义四分类 BN / epoch 300 | 历史 Source、TENT、PED2024 探索性比较；存在旧划分和选模限制 | [best.pt](D:/Workspace/deeplearn/3dunet/artifacts/models/legacy_custom_bn/source_training/run/checkpoints/best.pt) |
| HD-BET 脑提取模型 | 历史脑提取流程；当前 PED2023 主线不额外执行 HD-BET | [checkpoint_final.pth](D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED_TTA_20260928/weights_dataset/hdbet/release_2.0.0/fold_all/checkpoint_final.pth) |

此前已将四个完成的 native 训练分支 best/final 共 8 份 checkpoint 在 CPU 读取，核对 epoch、trainer、optimizer/scaler 字段；未初始化 CUDA、未运行推断。四分类 IN 主权重 SHA-256 为 `2b92812a25d4842d34a1d6ade2c60594f8d8bed30eb80af1b2b00bba5eeb6875`；BN best225 为 `2c3917ae7a41a13f04c2825de407ecf398dde4cd6ae3a9b6dede4e50fc78ea64`。

## 为什么有这么多文件

| 存储根目录 | 现有 checkpoint 数 / 大小 | 内容及处理 |
|---|---:|---|
| [IN 模型正式备份](D:/Workspace/deeplearn/3dunet/artifacts/models/in_dicece/) | 26 / 6.048 GiB | WT、四分类各 13 份，包含每 25 epoch 快照；全部保留 |
| [BN 模型正式备份](D:/Workspace/deeplearn/3dunet/artifacts/models/bn_dicece/) | 13 / 3.025 GiB | best225、final250、patch-best 及周期快照；全部保留 |
| [pure-Dice 正式备份](D:/Workspace/deeplearn/3dunet/artifacts/models/wt_pure_dice/) | 2 / 0.465 GiB | best225 和 final250；保留 |
| [九月份恢复备份](D:/Workspace/deeplearn/3dunet/artifacts/models/legacy_custom_bn/source_training/) | 18 / 4.186 GiB | 自定义 BN 的 best、last、latest 和周期快照；保留 |
| [native TTA bundle](D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_norm_TTA_20261006/bundle/model/) | 1 / 0.233 GiB | 四分类 IN250 的运行副本，字节与正式备份相同；现有流程使用该结构，保留 |
| [2026-09-28 TTA](D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED_TTA_20260928/) | 10 / 2.211 GiB | 旧 Source 副本、HD-BET 和各方法 adapter state；不是十个源模型，保留 |
| [2026-09-29 TTA suite](D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED_TTA_20260929_suite/) | 5 / 1.863 GiB | CoTTA、IST、TEGDA 的 resume 状态；含未完成和已取消运行，不等同完整 baseline，保留 |

`best` 按对应实验的选模指标保存，`final` 是训练终点，`best_patch` 是 patch 验证指标的最好点，不能互换。`checkpoint_epoch_*` 是训练过程快照；当前四分类 IN 的 175/200/225 已用于快照教师对照，不能因不是 final 就判定无用。

现存权重中有 5 组字节重复，例如 IN250 的 best/final/epoch250 和运行副本。现已共享这5组的底层存储，所有标准路径继续有效。早期83份盘点中的重复组数量是清理前状态。

## 模型定义和 TTA 后的权重在哪里

- native 网络由同目录上一级的 `plans.json`、`dataset.json` 以及 checkpoint 中的 trainer/init args 共同确定。保留完整 `nnUNet_results` 结构；[正式 source bundle](D:/Workspace/deeplearn/3dunet/artifacts/models/in_dicece/source_bundle/) 含 trainer、源码和运行信息。仅复制一个 `.pth` 不等于完整运行包。
- [src/brats_tta/models](../src/brats_tta/models/) 是历史自定义 U-Net、Original、Cicek 拓扑代码，不应直接拿其 YAML 加载当前 native 权重。本次已扫描范围内没有发现独立 Original/Cicek 正式训练 checkpoint；不据此断言电脑其他位置绝不存在。
- **当前背景共识直接方法冻结 Source，因此没有单独训练一个新的源模型 checkpoint。** 保存的是不同观察的概率和合成预测。
- 轻量 IN 学习按病例重置到同一个 Source，实验目录中 `*_affine.npz` / `*_delta.npz` 等保存各病例的参数或增量。这些也是重建适应状态的材料，不能当普通临时数组一并清理；旧 decoder 等实验的状态结构应按各自代码读取。
- `adapter_state.pt` / `resume.pt` 可能包含 teacher、optimizer、memory bank、随机状态，不是可随意替换为普通 Source 的权重文件。

## 已删除的旧权重与历史记录

| 历史材料 | 现在的状态 |
|---|---|
| pure-Dice 本地 7 epoch 训练的 2 份权重 | 已删除；代码、配置、患者划分、转换记录与日志仍在对应实验目录 |
| pure-Dice Kaggle 74 epoch 的 5 份权重 | 已删除；由后续 AutoDL 从头训练的正式250轮模型取代 |
| 三步显存测试的 1 份 checkpoint | 曾隔离，之后已删除；不属于正式 Source |
| 10 月 9 日隔离目录 | 现仅留 README / RESTORE_MAP 等记录，不能再从中恢复已删内容 |
| 旧 `source_references.npz` 等 patch 参照 | 在后续授权清理中已删；当前 native 主线所需缓存保留，重跑旧核验需重建 |

原 [83 份权重盘点](archive/storage_records/model_inventory_20261009.json)、[10 月 9 日隔离记录](archive/storage_records/storage_cleanup_20261009.json)、[曾被拒绝的清理计划](archive/storage_records/storage_cleanup_20261010.json)都是发生时的历史证据。后来授权执行的事实见[最终迁移核验](../artifacts/maintenance/consolidation_20261010/verification.json)。本页当前位置与75份清单优先于那些旧状态；没有改写原始回执。

前次导航整理的完整旧索引保存在 [navigation_before.zip](../artifacts/maintenance/organization_20261010/navigation_before.zip)。后续维护的只读硬链接与CPU加载核验另记在 `maintainability_20261010/`，没有改写模型张量或重算研究成绩。
