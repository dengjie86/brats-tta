# 文件与维护说明

2026-10-22 整理。公共源码集中在`src/brats_tta/`；工程入口见[快速开始](guides/getting-started.md)，模块依赖见[架构说明](reference/architecture.md)。难点集中在历史实验依赖、版本管理边界和积累的状态文档。

## 日常工作与冻结资产

| 位置 | 职责 | 修改边界 |
|---|---|---|
| [.github](../.github/workflows/ci.yml) / [CONTRIBUTING](../CONTRIBUTING.md) | CPU CI 配置与开发入口 | CI尚未远端执行；本机沿用既有cv311 |
| [实验登记](../configs/experiments.yaml) / [工程指南](guides/getting-started.md) | 5个主要实验的只读查询及使用文档 | 状态为记录快照，不复制科学参数、不执行runner |
| [README](../README.md) / [HANDOFF](../HANDOFF.md) / [AGENTS](../AGENTS.md) | 入口 / 唯一当前状态 / 规则与环境 | 替换过时状态，历史记录归档，不持续前插旧日志 |
| [src/brats_tta/native](../src/brats_tta/native/README.md) | 背景处理、学习目标、梯度重放等公共算法 | 新研究优先复用；禁止依赖带日期的实验目录 |
| [src/brats_tta/pipeline](../src/brats_tta/pipeline/README.md) | 完整 union2/soft_extent/no_come 单病例方法 | IN 与 Senior 后端分别维护；CLI 显式指定模型/MRI，不查找历史 runner |
| [其他 src 子包](../src/brats_tta/README.md) | 自定义模型、训练、数据、指标和 TTA 框架 | 保持原协议边界，不能等同于 native 训练流程 |
| [configs](../configs/README.md) / [scripts](../scripts/README.md) / [tests](../tests/README.md) | 配置 / 启动工具 / 公共代码测试 | 与本次修改匹配验证；不默认执行历史训练脚本 |
| [outputs](../outputs/README.md) | 54 个实验目录，代码快照、协议及产物 | 冻结代码和结果保留路径/内容；接续须隔离新产物 |
| [artifacts/models](../artifacts/models/) | 7 个 Source 备份目录 | 不原地覆盖权重；部分 checkpoint 是只读硬链接 |
| [artifacts/experiments](../artifacts/experiments/) | 45 个历史实验/helper 目录 | 部分仍被动态导入，不按“旧实验”整体删除 |
| [docs/archive](archive/README.md) | 历史交接、协议、研究导航及存储回执 | 仅供追溯；旧状态不能覆盖 HANDOFF |
| [artifacts/maintenance](../artifacts/maintenance/README.md) | 操作清单、原文备份和核验 | 本地保留，不是实验有效性证据 |

目录数量为本次盘点值，不是实时计数。文件累计约 74.83 GiB，主要为结果和模型；未按硬链接去重，不能视为实际物理占用。本次没有释放模型/数据空间。

## Git 边界已经调整

此前 `outputs/`、`artifacts/` 整体忽略，研究代码也无法自然进入版本管理。本次仅放行以下研究目录的 `.py`、`.ps1`、指定 `.sh` 和 `PROTOCOL.md`：

- `outputs/ped2023_*/`、`outputs/senior188_*/`、`outputs/qq_unet125_ped2023_source_tent_20261019/`。
- `artifacts/experiments/BraTS_PED2023*/`、`artifacts/experiments/HD_TTA_PED2023*/`。

`startup_failures/`、字节码、病例数组、权重、mask、日志、JSON 结果及其他本地备份继续忽略。已核查新增可见文件的种类及明显凭据赋值模式；这不是远程提交，本次没有 `git add/commit/push`。既有未提交改动原样保留，新增可见快照多不意味着这些代码都需要日常维护。

这是一条有限的源码保全规则，不是可复现资产的完整清单。Source 训练代码备份、配置/manifest、模型和数据仍需本地备份；克隆当前仓库无法直接重跑所有实验。未来新增不同命名的实验时，必须明确哪些源码/协议应纳入 Git，不能假定已自动放行。

## 方法的实际入口与剩余技术债

| 内容 | 实际入口 |
|---|---|
| 新病例公共入口 | `scripts/project.ps1 adapt` → [adapt_context.py](../src/brats_tta/cli/adapt_context.py) → [pipeline](../src/brats_tta/pipeline/README.md) |
| 原主方法算法和冻结结果 | [方法说明](GLI_PED_TTA_METHOD.md) · [98 组研究](../outputs/ped2023_teacher_complement_full98_20261015/) |
| 单视角 8 步 IN 版本 | [single_view.py](../outputs/ped2023_efficiency_full98_20261016/union2_single_view8/single_view.py) |
| Senior Source/TENT | [study.py](../outputs/senior188_ped2023_source_tent_20261021/study.py) |
| Senior 用户方法，保持暂停 | [study.py](../outputs/senior188_ped2023_ours_20261021/study.py)及同目录 teacher/learning/decoder |
| 当前负结果诊断 | [诊断报告](../outputs/senior188_bn_diagnosis_20261022/REPORT.md) |

旧 IN 单视角 runner 仍动态加载 `ped2023_efficiency_pilot_20261016/attempt03/study.py`；旧 Senior runner 也保持原依赖。这些冻结入口服务历史结果。新公共 pipeline 已提取 union2、学习和 no_come，运行时不加载日期实验；提取来源和回归范围见[重构记录](reference/refactor-validation.md)。

后续方法开发从公共 pipeline 开始，旧源码用于追溯与对照。尚未实现通用批量编排、自动接续和任意 checkpoint 适配；不能把三个固定 profile 解释为所有模型都已支持或验证。

## 本次归档与验证

当前交接和文档入口已缩短，根 AGENTS 的历史训练细节移到[Source 协议](archive/source_protocols.md)；六份旧存储 JSON 回执原字节移到[storage_records](archive/storage_records/)。`model_inventory_current_20261010.json` 保留原位，因为历史维护工具仍引用它。冻结研究报告和实验目录未移动。

原文、移动清单及验证结果保存在[维护回执](../artifacts/maintenance/repository_organization_20261022/README.md)。历史哈希仍属于产生时快照，不为新导航回填。公共测试通过仅说明对应实现行为保持，不表示方法效果或独立泛化已经成立。
