# 项目存储位置

所有目录/文件类型的用途见[项目文件总说明](PROJECT_FILE_GUIDE.md)，43 个历史实验逐项见[历史实验索引](HISTORICAL_EXPERIMENT_INDEX.md)。本页主要记录存储位置和已完成的清理迁移。

**维护更新：**公共算法进入 `src/brats_tta/native/`，资源路径集中到 `configs/paths.yaml`，旧长交接移入 `docs/archive/`。清理未使用的项目 `.venv`、缓存和空 `output/` 约0.65 GiB；将8个完全相同的权重副本改为只读硬链接，额外节省约1.86 GiB，全部75个路径与SHA保持有效。本轮没有再删实验概率、模型内容或数据集，详见[维护回执](../artifacts/maintenance/maintainability_20261010/)。

2026-10-10 起，项目文件统一放在 `D:/Workspace/deeplearn/3dunet`。E 盘根目录原有的 BraTS / HD_TTA / nnUNet 项目目录已迁入本项目；原始数据集仍在原位置，不改数据协议。

| 目录 | 用途 |
|---|---|
| [src](../src/) / [scripts](../scripts/) / [configs](../configs/) | 主代码、运行入口、配置 |
| [outputs](../outputs/) | 当前背景共识、IN 学习、BN 诊断及相关消融；含代码、结果和仍在复用的缓存 |
| [artifacts/models](../artifacts/models/) | 正式 Source 模型、训练快照、plans、训练与下载记录 |
| [artifacts/experiments](../artifacts/experiments/) | 从 E 盘迁入的旧实验、可复用 helper、病例浏览页和历史报告 |
| [artifacts/maintenance](../artifacts/maintenance/) | 清理/迁移清单、校验记录、迁移前原文备份；不是实验数据集 |
| [E:/dataset](E:/dataset/) | 已有数据集；本轮没有删除或搬动其中的文件 |

原始 GLI ZIP 仍在 `D:/Downloads/Zip/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip`。不额外复制数据集进项目。`outputs/` 和 `artifacts/` 都被 Git 忽略，仅克隆仓库不会得到本机模型和实验产物。

## 模型入口

| 子目录 | 内容 |
|---|---|
| [in_dicece](../artifacts/models/in_dicece/) | native IN 的 WT 二分类和四分类 Dice+CE；当前 Source 为四分类 epoch250 |
| [bn_dicece](../artifacts/models/bn_dicece/) | native 四分类 BN 对照，当前诊断使用 best225 |
| [wt_pure_dice](../artifacts/models/wt_pure_dice/) | pure-Dice WT 二分类 best225 / final250 |
| [legacy_custom_bn](../artifacts/models/legacy_custom_bn/) | 九月份自定义 BN 训练恢复和历史诊断，不能与 native 协议混用 |

完整 checkpoint 入口见[模型索引](MODEL_INDEX.md)。模型目录整体保留 nnU-Net 的内部层级；`HD_TTA` 曾是训练阶段的目录前缀，不表示当前方法调用 HD-TTA 优化。

## 此前集中清理与迁移（2026-10-10，代码重构之前）

- 删除 3,122 个文件、32.742276 GiB：旧尺度/跨序列/对比度/N4 等探针缓存、早期消融 logits/概率、未继续使用的 7/74 epoch 训练权重、测试 checkpoint、重复上传包、下载残留和工具缓存。代码、最终分割、指标及保留的适应参数未作为缓存删除。
- 将 49 个散落目录中的 18,908 个保留文件、21.518253 GiB 收入本项目；其中包含 48 个 E 盘项目目录和一个 D 盘旧缓存隔离目录。跨盘操作先复制、逐文件校验 SHA-256，再移除旧位置。迁移量不计作删除释放量。
- 75 份 checkpoint 保留；删除的 8 份是已核实的中断训练/三步测试权重，正式 Source 模型未删。现用背景教师、Source 概率及 native IN 学习需要的输入保留。
- 加此前两次删除的 66.424312 GiB，该阶段结束后累计删除约 99.17 GiB。各次记录分别保留，旧记录中曾被审批拒绝的 249 项在该阶段重新核验后已执行。后续环境清理/权重去重的约2.51 GiB另计，见页首维护更新。

## 路径与历史记录

迁移会改变代码文件中的绝对路径，从而改变文本 SHA-256。工作副本只更新路径和依赖这些文本的哈希引用，不重算或修改实验指标。修改前的文本逐字节归档；旧科学记录与迁移记录不能混淆。

- [迁移前后路径及逐文件清单](../artifacts/maintenance/consolidation_20261010/plan.json)
- [路径修改与原文备份记录](../artifacts/maintenance/consolidation_20261010/path_rewrite_receipt.json)
- [原始文本归档](../artifacts/maintenance/consolidation_20261010/original_text.zip)
- [最终文件完整性核验](../artifacts/maintenance/consolidation_20261010/verification.json)：25,250 个保留文件核验通过，75 份 checkpoint 和原始 GLI ZIP 的 SHA-256 不变，数据集 31,414 文件的大小/mtime 不变，旧外部项目根目录已消失。
- [当前流程 CPU 检查](../artifacts/maintenance/consolidation_20261010/current_workflow_verification.json)：四个模块导入、21 项依赖、396 个 MRI 路径、36 份复用概率、背景教师小检查及 IN250/BN225 加载通过；未运行真实推理或训练。
- [早期维护记录的新位置](../artifacts/maintenance/consolidation_20261010/temporary_audit_migration.json)：12 个本项目临时核验文件已收进 `artifacts/maintenance/earlier_audits/`。

早期 `docs/*cleanup*.json`、原模型盘点和旧导航快照仍保存发生操作时的路径，作为历史证据，不代表文件仍在旧位置。定位现有文件应使用本页和上述迁移表。旧实验的缓存被删除后，重跑需按原始数据、模型及配置重建；不能把历史 `complete` 字段理解为所有中间文件还存在。

以后新实验写入 `outputs/<实验名>/`，模型备份写入 `artifacts/models/`，旧实验归入 `artifacts/experiments/`，维护记录写入 `artifacts/maintenance/`。不再在盘符根目录或桌面创建项目结果目录。
