# 开发指南

## 代码与产物放在哪里

沿用当前 `src` 包布局和 `pyproject.toml`，新增公共逻辑放在对应 `src/brats_tta/` 子包。CLI 只解析参数与组织调用；可复用算法不通过 `sys.path` 或 `spec_from_file_location` 加载日期实验。新运行产物仍放 `outputs/<experiment>/`，Source 正式备份放 `artifacts/models/`。

| 内容 | 维护位置 |
|---|---|
| 可复用算法、指标、数据处理 | `src/brats_tta/<module>/` |
| 稳定命令行接口 | `src/brats_tta/cli/`；安装命令登记到 `pyproject.toml` |
| 公共资源位置 | `configs/paths.yaml`；模型专用路径可由运行配置传入 |
| 新实验参数 | `configs/experiments/<name>.yaml`，由对应 runner 明确解析；当前尚未创建通用科学参数 schema |
| 主要实验导航记录 | `configs/experiments.yaml`，仅登记身份、状态和引用路径 |
| 测试 | `tests/test_<module>.py`；实验专用核验与该实验一同保存 |
| 运行日志、mask、指标和配置快照 | `outputs/<experiment>/`，保留冻结快照与原哈希 |

`configs/experiments.yaml` **不是训练/适应参数配置**。科学参数继续以冻结 `PROTOCOL.md`、实际代码及 preparation 为准，避免同一阈值在导航与 runner 各写一份。

## 一个修改的最小流程

1. 查看 `git status`，读取本模块和实验协议；确定影响的是公共代码还是冻结实现。
2. 在公共模块中做本任务需要的修改。历史 runner 不自动换成新包，避免改变旧结果的来源。
3. 运行相关测试；本机日常入口是 `project.ps1 lint` 和 `project.ps1 test`。算法或适应状态变更补数值、梯度、重置/冻结等有意义的回归。
4. 新实验用独立输出目录保存代码身份、完整配置、病例顺序、checkpoint 身份和失败/完成状态。不能从旧目录原地覆盖结果来“接续”。
5. 状态变化更新根 HANDOFF；主要入口变化同步登记表和实验索引。完成记录留在实验目录，交接不持续追加全部训练日志。

是否提交由用户决定；提交前按文件检查新增可见范围，不使用 `git add .` 来跳过审阅。现有 Git 放行规则只覆盖指定历史研究目录；新配置在 configs 下可见，新输出中的代码/协议是否可见需用 `git check-ignore` 核实。权重、MRI、大数组不进入普通 Git。

## 检查范围与 CI

`scripts/project.ps1 test` 是本地和 CI 共用的 7 模块测试清单，共 45 项；全套 `python -m pytest -q` 仍包括历史训练/DDP等测试，不是日常默认入口。`lint` 选择 Ruff 的 `E9,F63,F7,F82`，用于已有代码的关键错误检查，不声称全仓库通过全部风格规则。

[CI 配置](../../.github/workflows/ci.yml)在 GitHub 的 Linux CPU runner 执行 lint、相同测试和无本地资产的登记表校验。CPU 测试依赖版本见[requirements-cpu.txt](../../.github/requirements-cpu.txt)，不安装 nnU-Net、不获取数据、不训练、不连接 AutoDL。工作流仅在相关代码变更或手动触发时运行，配置随本分支发布，运行状态以 GitHub Actions 为准。

Actions 接口依据官方 [checkout](https://github.com/actions/checkout) 与 [setup-python](https://github.com/actions/setup-python) 文档配置。配置存在不代表云端执行已通过；本次实际验证的是本机 Windows/cv311。

## 文档分工

README 是入口；HANDOFF 是唯一当前状态；AGENTS 是操作规则；本目录是工程指南；`docs/reference/` 解释架构；方法、结果及证据仍由既有专题和实验报告负责。`docs/archive/` 只存历史，不另建第二份“当前交接”。
