# 开发入口

本机沿用 Conda `cv311`，先读[快速开始](docs/guides/getting-started.md)，代码职责见[架构说明](docs/reference/architecture.md)。具体修改、测试和实验记录规则在[开发指南](docs/guides/development.md)。

修改前查看 `git status` 并阅读 [AGENTS](AGENTS.md)。公共代码放 `src/brats_tta/`，配置放 `configs/`，测试放 `tests/`；实验运行产物放 `outputs/<experiment>/`。新公共实现不导入旧日期实验目录。

```powershell
.\scripts\project.ps1 lint
.\scripts\project.ps1 test
.\scripts\project.ps1 experiments check
```

算法改动增加与实际风险匹配的数值/状态回归；目录、文档调整不启动完整训练或推理。冻结实验、权重和历史核验回执保留原样。旧实验复现前读[复现指南](docs/guides/reproducibility.md)。

本仓库已有大量未提交工作；不得整树格式化、覆盖用户改动或自行提交/推送。CI 实际状态以 GitHub Actions 为准，本地测试通过不代表云端通过。
