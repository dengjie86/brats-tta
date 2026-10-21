# 当前磁盘占用与保留规则

2026-10-22整理前文件长度盘点约 **74.83 GiB**：outputs约46.590 GiB、artifacts约28.210 GiB。未去除NTFS硬链接重复，也不是磁盘分配大小。本轮归档和源码可见性调整没有删除模型、数据或实验缓存。

## 2026-10-16清理记录（历史快照）

2026-10-16，范围仅为`D:/Workspace/deeplearn/3dunet`。本次清理973份已结束消融实验的概率/logits和工具缓存，释放 **21.17 GiB**；45个实验目录继续保留原路径，并按[用途索引](EXPERIMENT_INDEX.md)查找。

| 指标 | 清理前 | 清理后 |
|---|---:|---:|
| 项目按文件路径累计大小 | 89.37 GiB | 68.21 GiB |
| 按文件ID去除硬链接重复后的逻辑大小 | 87.51 GiB | 66.35 GiB |
| D盘可用空间（盘点时） | 8.24 GiB | 29.40 GiB |

目录大小不是逐文件磁盘分配空间。原有模型硬链接仍保留，卷空闲空间也会受项目外写入影响；本页数字是本次盘点时点，不是实时监控。维护清单自身占用少量空间，原始数据集和项目外目录未清理。

## 文件保留层次

| 类别 | 处理 |
|---|---|
| Source checkpoint、学习后affine/delta状态 | 全部保留 |
| 最终mask、指标、报告、协议、代码、配置、日志及回执 | 全部保留原路径 |
| 主方法98组缓存重放与未完成跨Source检查 | 保留相关分支和清单中的1,070个直接依赖文件 |
| 已结束、未采用方案的稠密概率/logits | 仅删除下表明确列入清单的数组；重做数组级复核需要重建 |
| Python及测试工具缓存 | 删除清单中的可再生文件和空缓存目录 |

## 本次清理范围

| 实验与缓存说明 | 数组文件数 | 释放GiB |
|---|---:|---:|
| [ped2023_decision_consensus_anchor_20261012](../outputs/ped2023_decision_consensus_anchor_20261012/CACHE_STATUS.md) | 48 | 1.383 |
| [ped2023_extent_factor_learning_20261013](../outputs/ped2023_extent_factor_learning_20261013/CACHE_STATUS.md) | 48 | 1.133 |
| [ped2023_extent_preserving_targets_20261012](../outputs/ped2023_extent_preserving_targets_20261012/CACHE_STATUS.md) | 48 | 1.381 |
| [ped2023_in_context_decomposition_20261008](../outputs/ped2023_in_context_decomposition_20261008/CACHE_STATUS.md) | 48 | 1.356 |
| [ped2023_in_subspace_20261014](../outputs/ped2023_in_subspace_20261014/CACHE_STATUS.md) | 192 | 4.636 |
| [ped2023_loss_checkpoint_20261014](../outputs/ped2023_loss_checkpoint_20261014/CACHE_STATUS.md) | 114 | 2.755 |
| [ped2023_probability_distillation_20261014](../outputs/ped2023_probability_distillation_20261014/CACHE_STATUS.md) | 96 | 2.318 |
| [ped2023_source_moment_replay_20261013](../outputs/ped2023_source_moment_replay_20261013/CACHE_STATUS.md) | 48 | 1.388 |
| [ped2023_stable_consensus_anchor_20261012](../outputs/ped2023_stable_consensus_anchor_20261012/CACHE_STATUS.md) | 48 | 1.382 |
| [ped2023_structure_response_20261008](../outputs/ped2023_structure_response_20261008/CACHE_STATUS.md) | 36 | 1.112 |
| [ped2023_teacher_complement_learning_20261015](../outputs/ped2023_teacher_complement_learning_20261015/CACHE_STATUS.md) | 96 | 2.323 |

此外清理少量工具缓存。每个受影响目录新增`CACHE_STATUS.md`，原冻结报告与回执不改写，不把历史核验改成当前文件仍完整的证明。来源、删除前SHA-256和保留文件核对见[维护记录](../artifacts/maintenance/storage_cleanup_20261016/README.md)。

## 清理后占用最大的目录

| 目录 | GiB |
|---|---:|
| `outputs/ped2023_cross_model_transfer_20261015/` | 12.42 |
| `outputs/ped2023_context_full99_20261008/` | 7.63 |
| `artifacts/models/in_dicece/` | 6.07 |
| `outputs/ped2023_extent_factor_full98_20261013/` | 4.81 |
| `artifacts/models/legacy_custom_bn/` | 4.23 |
| `outputs/ped2023_native_objective_full99_20261011/` | 3.40 |
| `outputs/ped2023_teacher_complement_full98_20261015/` | 3.38 |
| `artifacts/models/bn_dicece/` | 3.03 |
| `artifacts/experiments/BraTS_PED2023_axis_probe_20261006/` | 2.40 |
| `outputs/ped2023_teacher_complement_learning_20261015/` | 2.37 |

当前仍占空间的主要是跨Source待完成任务、主方法及前序复用缓存、模型与学习状态。目录按用途在索引中组织，实际路径不搬动，以保留冻结依赖。后续新实验只保存必要中间状态；结束后明确哪些概率可重建，避免把每次完整概率长期重复保留。旧数据迁移与更早清理历史见[数据存储索引](DATA_STORAGE_INDEX.md)。
