# BraTS GLI → PED2023 TTA

本分支提供公共方法代码、测试和文档。数据、权重、历史实验与运行结果保留在本地；文档中的 `outputs/`、`artifacts/` 链接可能仅在原工作区有效。验证范围见[重构验证](docs/reference/refactor-validation.md)。

本仓库保留公共代码、冻结实验和本地研究资产。日常修改从 `src/` 开始；历史 runner 继续保留在原实验目录，不能随意移动或改写。

**当前状态（2026-10-22）：Senior188 上的用户方法已按要求暂停，完成 66/98 例，平均 Dice 基本无提升。** 原 IN250 上的开发收益仍成立，但不能推广到该 BN 模型。后续任务尚未启动，详见[当前交接](HANDOFF.md)。

| 要做什么 | 入口 |
|---|---|
| 第一次使用与日常命令 | [快速开始](docs/guides/getting-started.md) |
| 开发代码、复现实验、理解架构 | [开发指南](docs/guides/development.md) · [复现指南](docs/guides/reproducibility.md) · [架构说明](docs/reference/architecture.md) |
| 了解当前结果、暂停状态和待解决问题 | [HANDOFF](HANDOFF.md) |
| 找可维护代码、了解仓库边界 | [文件与维护说明](docs/PROJECT_FILE_GUIDE.md) · [公共源码](src/brats_tta/README.md) |
| 看方法、架构图和证据限制 | [方法](docs/GLI_PED_TTA_METHOD.md) · [图](docs/figures/tta_method_20261016/README.md) · [证据](docs/RESEARCH_STATUS.md) |
| 找具体实验与对照 | [实验索引](docs/EXPERIMENT_INDEX.md) |
| 找模型、数据及历史资料 | [模型](docs/MODEL_INDEX.md) · [数据](docs/DATA_STORAGE_INDEX.md) · [文档导航](docs/README.md) |

```text
.github/             CPU CI 及测试依赖
src/                 公共代码：完整 pipeline + native 原语 + 自定义框架
configs/             资源路径和训练配置
scripts/             本机入口、历史训练与工具脚本
tests/               公共代码测试
docs/                guides/ 使用与开发；reference/ 架构；archive/ 历史
outputs/             实验代码快照、协议、结果与缓存
artifacts/models/    Source 权重和训练备份
artifacts/experiments/ 历史实验和仍被复用的 helper
artifacts/maintenance/ 文件整理与核验记录
```

在仓库根目录使用已有 Conda `cv311`：

```powershell
.\scripts\project.ps1 check
.\scripts\project.ps1 experiments list
.\scripts\project.ps1 experiments show senior188-ours
.\scripts\project.ps1 lint
.\scripts\project.ps1 test
.\scripts\project.ps1 teacher --help
.\scripts\project.ps1 adapt --help
```

`check` 检查资源路径；`experiments` 查询登记的 5 个主要实验；`lint` 检查关键代码错误；`test` 运行 45 项 CPU 测试；`teacher` 处理缓存概率；`adapt` 从 MRI 运行完整单病例方法，见[公共 pipeline](src/brats_tta/pipeline/README.md)。旧 runner 保留原样，登记查询不会启动它们。

开发规则见 [CONTRIBUTING](CONTRIBUTING.md)。继续沿用现有 Conda 环境和 `pyproject.toml`，本次没有升级依赖、改动模型实现或恢复实验。

指定 PED 研究目录的源码和 `PROTOCOL.md` 已从 Git 忽略规则中放行，本分支未上传这些历史 runner。其余本地结果、数据、权重及训练备份仍被忽略；克隆仓库不能代替本地资产备份。原实验内容和路径保持，维护边界及剩余问题见[文件说明](docs/PROJECT_FILE_GUIDE.md)。
