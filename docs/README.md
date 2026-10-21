# 文档导航

第一次使用从[快速开始](guides/getting-started.md)进入；查研究状态看[当前交接](../HANDOFF.md)。

## 工程文档

| 目的 | 文档 |
|---|---|
| 环境、日常命令、查找实验 | [快速开始](guides/getting-started.md) |
| 修改代码、新增实验、测试与 CI | [开发指南](guides/development.md) |
| 查看结果、缓存重放、完整重跑的区别 | [复现指南](guides/reproducibility.md) |
| 完整方法公共实现与固定profile | [pipeline说明](../src/brats_tta/pipeline/README.md) |
| 模块职责、依赖方向和目录结构 | [架构说明](reference/architecture.md) |

## 研究与资料

| 内容 | 入口 |
|---|---|
| 当前任务、暂停状态与结果 | [HANDOFF](../HANDOFF.md) |
| 方法、参数、图与证据 | [GLI_PED_TTA_METHOD](GLI_PED_TTA_METHOD.md) · [架构图](figures/tta_method_20261016/README.md) · [RESEARCH_STATUS](RESEARCH_STATUS.md) |
| Senior188 结果与负结果诊断 | [Source/TENT](../outputs/senior188_ped2023_source_tent_20261021/REPORT.md) · [暂停方法](../outputs/senior188_ped2023_ours_20261021/README.md) · [诊断](../outputs/senior188_bn_diagnosis_20261022/REPORT.md) |
| 所有近期实验 | [EXPERIMENT_INDEX](EXPERIMENT_INDEX.md) |
| 代码职责、Git 边界与未解决依赖 | [PROJECT_FILE_GUIDE](PROJECT_FILE_GUIDE.md) |
| 权重、数据与容量 | [MODEL_INDEX](MODEL_INDEX.md) · [DATA_STORAGE_INDEX](DATA_STORAGE_INDEX.md) · [STORAGE_STATUS](STORAGE_STATUS.md) |
| 历史专题、早期设计与文献讨论 | [历史专题导航](archive/research_topics.md) |
| 历史实验及训练协议 | [历史实验索引](HISTORICAL_EXPERIMENT_INDEX.md) · [Source 协议](archive/source_protocols.md) |
| 旧时间线与存储操作记录 | [归档入口](archive/README.md) |

根目录下部分带日期的研究报告被旧核验脚本按路径读取并记录哈希，因此保留原路径和正文。它们是冻结资料；其中的“当前”“待启动”以根目录 HANDOFF 为准。
