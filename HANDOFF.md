# 当前交接

更新：2026-10-22。这里只保留当前结论、资源入口和接续约束；逐次启动、ETA、GPU 优化记录已收进[历史交接](docs/archive/HANDOFF_before_organization_20261022.md)。

## 当前任务与运行约束

- 本轮整理当前仓库；未另建独立工程、Goal 或 Scheduled。
- **Senior188 用户方法评测保持暂停**：用户于 10 月 22 日明确叫停，完成 66/98 例；暂停记录见[paused.json](outputs/senior188_ped2023_ours_20261021/paused.json)。原进程停止时保留全部完成产物及第 67 例现场；旧 progress、PID、ETA 不能证明现在存活。
- 不自动恢复实验或巡检。用户已于 10 月 21 日取消原 40 分钟巡检；非 Goal 模式启动确认后返回，不持续占对话等待。
- 只有用户要求继续时，才读原协议和接续条件、隔离剩余病例；不能覆盖冻结代码、结果或重复原 prepare。中断瞬间的内存 optimizer 状态未保存。

## 最新结果：Senior best188

采用用户指定的第 188 轮 `best.pth`，PED2023 去重后 98 例。完整 Source/TENT 于 10 月 21 日完成；用户方法只有相同子集 66 例，不能将两行跨集合相减。

| 病例集合 | Source 平均 Dice% | TENT 平均 Dice% | 用户方法平均 Dice% |
|---|---:|---:|---:|
| 完整 98 例 | 55.71318 | 61.67384 | 未完成 |
| 相同已完成 66 例 | 58.70143 | 62.67784 | 58.72396 |

完整 [Source/TENT 报告](outputs/senior188_ped2023_source_tent_20261021/REPORT.md)：TENT +5.96066pp；WT/TC/ET 为 84.28050/45.95262/54.78840%。WT 的配对 HD95 变差，不能称所有指标改善。

[用户方法](outputs/senior188_ped2023_ours_20261021/README.md)为 union2 + 单视角 8 步 soft_extent + no_come，Adam lr=.001、rho=.5；仅更新 5,888 个 BN affine 标量，冻结 running stats。Senior 使用原包全网格预处理、128³/50%重叠、Gaussian **概率融合**、无镜像、BF16，区别于 native 的八镜像/logit 融合。

[诊断](outputs/senior188_bn_diagnosis_20261022/REPORT.md)显示同 66 例 Source/teacher/raw student/final 为 58.70143/58.66164/58.55671/58.72396%。61 例有更新且记录损失下降；teacher 修改的 98.33% 是背景↔肿瘤，ED→NC 仅 86 体素，补回 TC 漏分的 0.255%。主要直接证据是 teacher 没有提供所需的内部组织纠错，不能概括成优化器完全没运行。

历史 IN 同 66 例 teacher 的 ED→NC 为 175,219 体素、TC 从 49.7993% 到 57.2892%，但两模型不是 norm-only 对照。6 例 BN 当前 patch 统计探测仅 2 例提高、3 例下降、1 例持平，去最大增益病例后平均 −1.28171pp；没有建立通用修复。诊断已经结束，没有接续原队列。

## 已有 IN 与迁移证据

| 实验 | 结果及入口 |
|---|---|
| IN250 原主方法，98 组 | 平均 Dice 67.26899%，Source 64.14833%；WT/TC/ET 89.14027/58.20171/54.46499%。[方法及控制](docs/GLI_PED_TTA_METHOD.md) |
| union2 单视角 8 步，98 组 | 平均 67.18677%，相对原主方法 −0.08222pp；卷积估算成本下降 59.197%，不是端到端实测加速。[报告](outputs/ped2023_efficiency_full98_20261016/union2_single_view8/REPORT.md) |
| 双观察单视角 4 步，98 组 | 平均 66.42193%，原主方法 −0.84706pp；卷积成本下降 85.150%。[报告](outputs/ped2023_efficiency_full98_20261016/REPORT.md) |
| BN225/BN250 固定 24 例 | 相对各自 Source 平均 +0.75607/+0.30815pp；两分支 checkpoint 比较，非多架构普适证明。[阶段报告](outputs/ped2023_cross_model_transfer_20261015/interim_readout/REPORT.md) |
| IN200 固定 24 例 | 10 月 15 日记录 22/24 后退出，exit1073807364，原因未知；无完整评价，不自动接续。[入口](outputs/ped2023_cross_model_transfer_20261015/README.md) |
| QQ125 Source/TENT，98 例 | 平均 47.80466→60.80041%，+12.99575pp。[报告](outputs/qq_unet125_ped2023_source_tent_20261019/REPORT.md) |

上述均为开发证据。固定 24、完整 98、Senior 已完成 66 不可混比；IN/BN 差异混有训练轨迹、模型及协议差异。PED2023/2024 存在大量 MRI 重叠，跨年份不是独立患者泛化。指标采用 WT/TC/ET 等权平均；空区域处理按各实验协议，HD95 比较报告共同有效 n。

## Source 备份与资源

- [Senior 训练工程备份](artifacts/models/senior_unet128_source_20261020/README.md)：238 轮 patience50 正常早停；组织指标 best188=83.11838%，区域指标 best178=89.08984%。此前 3,378 个非数据文件下载哈希核验、三份主要权重 CPU 加载通过。3378 文件/5.55 GiB 是当时备份记录，本轮没有重新联网核验。
- [QQ 完成 300 轮备份](artifacts/models/qq_unet_source_20261019/completed_300/README.md)：best275/final300 及 125 轮独立备份保留，275/300 尚未建立 PED 结果。
- native IN250、BN225/250 与其他 Source 见[模型索引](docs/MODEL_INDEX.md)；数据见[数据索引](docs/DATA_STORAGE_INDEX.md)，路径配置见[paths.yaml](configs/paths.yaml)。
- 本地 Python：`D:/Environment/miniforge3/envs/cv311/python.exe`；RTX 4060 Laptop 8 GiB。远程当前开关机状态未查，不按历史日志判断。

## 仓库维护与剩余问题

工程整理（2026-10-22）：沿用当前仓库和环境，新增[工程文档入口](docs/README.md)、5项只读实验登记及`project.ps1 experiments/lint`。新增CI配置尚未提交或远端执行。登记不替代冻结协议，也不会启动实验。此前整理见[记录](artifacts/maintenance/repository_modernization_20261022/README.md)。

本轮已将完整 union2 → soft_extent → no_come 抽取到[公共 pipeline](src/brats_tta/pipeline/README.md)，新增`project.ps1 adapt`单病例入口。IN250单/八视角学习与Senior188概率融合分别保留；新入口不动态导入日期实验、不读GT、只写全新输出目录。原native文件、冻结runner及暂停队列保留原样。

验证：7模块45项CPU测试通过；IN250真实00002病例复用teacher缓存，8步损失/affine/bias/最终mask与原单视角结果完全一致；原八视角结果的缓存decoder重放也完全一致。Senior188同一例从MRI重新生成10次观察并完成学习，Source/teacher/最终mask及teacher增补一致，但affine最大绝对差0.00091769、最大逐步loss差0.00102448，不能声称逐位一致。128³隔离探测确认旧GPU重放自身也有浮点变化。未重跑98例或新验证完整八视角学习；详见[重构与数值边界](docs/reference/refactor-validation.md)。

本次已精简当前入口、归档旧时间线和六份历史存储回执，并放行指定 PED 研究源码/协议的 Git 可见性。未提交代码，未搬动实验、数据、权重、mask 或数组；历史核验哈希不回填。范围、验证和剩余限制见[文件说明](docs/PROJECT_FILE_GUIDE.md)及[维护回执](artifacts/maintenance/repository_organization_20261022/README.md)。

继续开发从公共pipeline进入；新实验明确代码、配置和产物边界，冻结后保持原样。当前仍未提供任意checkpoint适配、完整批量接续和自动评价；这些不影响新入口运行明确指定的单病例。
