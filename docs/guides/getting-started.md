# 快速开始

本指南面向当前 Windows 工作区 `D:/Workspace/deeplearn/3dunet`。先了解结果看[当前交接](../../HANDOFF.md)；想了解方法看[方法说明](../GLI_PED_TTA_METHOD.md)和[架构图](../figures/tta_method_20261016/README.md)。

## 环境与检查

本机使用现有 `D:/Environment/miniforge3/envs/cv311/python.exe`，本轮核对为 Python 3.11.15、PyTorch 2.11.0+cu128。以下命令不需要安装项目或新建环境；`project.ps1` 临时设置 `PYTHONPATH`，结束后恢复，并使用 Python UTF-8 模式显示中文。

```powershell
Set-Location 'D:\Workspace\deeplearn\3dunet'
& 'D:\Environment\miniforge3\envs\cv311\python.exe' -c "import sys; print(sys.executable)"
.\scripts\project.ps1 check
.\scripts\project.ps1 lint
.\scripts\project.ps1 test
```

| 命令 | 实际检查 |
|---|---|
| `check` | `configs/paths.yaml` 的模型、数据及缓存位置是否存在；不加载权重 |
| `lint` | 公共源码和测试的关键语法/错误规则；不格式化冻结实验 |
| `test` | 7 个模块、45 项 CPU 小测试，覆盖完整方法、背景处理、梯度重放、路径及实验登记 |

新克隆没有本地资产时，`check` 可能报告路径缺失；这不同于代码测试失败。不要为让路径检查变绿而创建空权重/数据目录。

## 查找实验

```powershell
.\scripts\project.ps1 experiments list
.\scripts\project.ps1 experiments show senior188-ours
.\scripts\project.ps1 experiments check
.\scripts\project.ps1 experiments check --local
```

登记表目前收录 5 个主要入口，全部历史实验仍看[实验索引](../EXPERIMENT_INDEX.md)。`list/show` 只展示记录；`check` 验证字段和仓库内相对路径，`--local` 再检查引用文件是否存在。它们不导入 runner、不启动 GPU、不巡检、不恢复已暂停任务，也不重新核验全量权重/数组哈希。

`senior188-ours` 的 `paused 66/98` 是已记录状态；不是实时进程探测。`native-in250-main` 指向缓存重放，其余入口指向冻结 runner，二者不能混作新病例的一键推理命令。

## 单病例方法入口

`project.ps1 adapt --help`查看新MRI输入要求；完整命令与输出见[复现指南](reproducibility.md)。它通过公共pipeline运行，旧实验登记仍指向原冻结实现。

## 下一步读什么

| 目的 | 文档 |
|---|---|
| 修改代码或新增研究 | [开发指南](development.md) |
| 理解模块和调用边界 | [架构说明](../reference/architecture.md) |
| 重建旧实验环境或检查结果 | [复现指南](reproducibility.md) |
| 找 Source 权重和数据 | [模型](../MODEL_INDEX.md) · [数据](../DATA_STORAGE_INDEX.md) |

Linux/其他机器可使用已有兼容环境设置 `PYTHONPATH=src` 后运行 `python -m brats_tta.cli.experiments check`。`pyproject.toml` 定义安装入口和依赖范围；它不是当前 Conda 环境的完整锁文件，旧 `uv.lock` 也未经核对为该 Conda 环境的精确快照。本轮没有安装或升级任何本地核心依赖。
