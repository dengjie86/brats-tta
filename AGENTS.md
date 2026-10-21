# Project environment notes

This file records the local runtime details that are easy to lose between sessions. Keep it up to date when the machine or the training workflow changes.

## 长任务等待与额度使用

- 禁止频繁轮询而无故消耗大量 token / 额度。长时间训练、推理、评价正常运行时，不按秒、按分钟或逐病例反复唤起模型查询进度，也不通过连续短等待绕过此规则。
- 启动后确认一次任务已正常运行；后续优先由普通脚本或已配置的完成/失败通知负责等待。只有完成、失败、异常超时、需要决策的阶段变化或用户主动询问时，才做有目的的检查。确有定时检查必要时采用与任务时长相称的低频间隔；“每 30 分钟”仅是此前讨论方案，不代表已启用调度。
- 无新证据时，不反复读取相同日志、整份交接或进度文件，不重复已通过的哈希核验、测试和审计，不为填补等待时间新增无关检查、反复改写进度文档或调用子代理盯任务。优先读取新增输出，核验范围应与本次变化相称。
- 当前环境没有自动唤醒能力时，如实说明限制，不用高频模型调用伪装全自动；不要仅因等待就擅自暂停 Goal、将其标为完成/阻塞或创建调度。Goal 与 Scheduled 的状态只按实际工具结果记录，交接文档不构成启动授权。
- 当前交接统一维护在根目录 `HANDOFF.md`，不另建日期版当前交接。更新时同步相关导航；只在状态有实质变化或用户要求交接时记录，历史结果和冻结实验文件保持原样。

## Current protocol and status (October 22, 2026)

- 当前交接只维护在根目录[HANDOFF.md](HANDOFF.md)；方法细节见[docs/GLI_PED_TTA_METHOD.md](docs/GLI_PED_TTA_METHOD.md)，所有近期实验见[docs/EXPERIMENT_INDEX.md](docs/EXPERIMENT_INDEX.md)。原逐次启动/完成记录已归档到[历史研究记录](docs/archive/research_navigation_20261015/AGENTS_history.md)，不要把历史PID或待办当作现在状态。
- 主路线已选定：union2 complementary teacher + soft_extent（8 Adam steps, lr=.001, 7,680 IN affine scalars）+ no_come decoder。98组结果、最终学习控制和缓存交付重放完成；此前Goal已complete，本次跨Source及目录整理未另建Goal/Scheduled。
- 用户10月15日追加其他模型验证：`outputs/ped2023_cross_model_transfer_20261015/`，BN225/BN250/IN200同原24例，复用IN250。两个训练分支/checkpoint，不是三种架构。BN affine only，原eval running stats与非affine参数冻结；不调teacher、步数或decoder。
- 10月15日历史核对：BN225/BN250各24例预测exit0；720项独立Dice计数通过，结果在`interim_readout/`。IN200于14:55:24 exit1073807364，日志记载22/24完成、00167第5步后中断；无完整队列/评价回执，未见匹配进程。原因未知，整理时未重启。接续须保护已完成结果与失败现场，禁止重复原prepare/execute或覆盖冻结文件。
- 非Goal模式不持续占对话等待长实验；启动确认后返回，脚本后台接续，用户询问或实质异常时再检查。停止等待不等于停止实验，但旧的存活确认也不证明进程现在仍运行。
- 当前主结果全为开发证据；98组/固定24例不可混比。WT/ET稳定性、BN普适性、独立泛化和新颖性未证明。原24均GT含ET；2023/2024大量MRI重叠，跨年份不等于独立患者。
- 冻结实验代码、协议、preparation、模型状态、mask、指标和旧回执保持原样；导航可更新，旧文档核验哈希保留产生时记录，不能回填为新哈希。新修复/新科学研究另建子目录。

- 最新Senior188方法评测按用户要求暂停，完成66/98；诊断已结束。不得自动恢复或巡检。详细结果只维护在HANDOFF。10月21日取消的40分钟巡检授权不得沿用。
- 本轮整理不改变算法：指定研究源码/协议放行Git，模型、数组、结果继续忽略。公共代码禁止导入outputs/artifacts中的日期实验链；冻结runner保持原样。

## Workspace

- Repository: `D:\Workspace\deeplearn\3dunet`
- Git remote: `https://github.com/dengjie86/brats-tta.git`
- Current work must preserve the user's uncommitted changes. Do not use destructive reset/checkout commands.

## Engineering workflow (October 22, 2026)

- 完整单病例方法已提取到`src/brats_tta/pipeline/`；`project.ps1 adapt`显式接收模型和四个MRI文件，只写全新outputs子目录。Native IN与Senior BN融合/反向各自保留；旧native原语和冻结runner本轮不改。具体回归范围见[重构记录](docs/reference/refactor-validation.md)。
- 工程说明统一从[快速开始](docs/guides/getting-started.md)、[开发指南](docs/guides/development.md)、[复现指南](docs/guides/reproducibility.md)和[架构说明](docs/reference/architecture.md)进入。
- `src/brats_tta/`放公共实现，`cli/`放薄入口，`configs/`放配置，`tests/`放公共验证。新公共代码不得导入日期实验；现有冻结runner及其依赖保持原样。
- 新实验运行产物仍放`outputs/<experiment>/`；新维护源码/参数优先放公共包和`configs/experiments/<name>.yaml`，以独立运行快照绑定，不能把输出目录当作唯一源码备份。
- `configs/experiments.yaml`只登记主要实验身份、状态快照与文件引用，不是科学参数配置。`project.ps1 experiments list|show|check`均只读；登记paused不构成恢复授权，登记completed不证明文件完整或进程实时状态。
- `project.ps1 lint`检查公共src/tests关键错误；`test`现覆盖7模块45项CPU测试。`.github/workflows/ci.yml`提供同范围CPU配置，已纳入发布分支；远端状态以GitHub Actions为准，不得把本地通过说成CI通过。

## Project storage (updated October 16, 2026)

- 用户10月16日授权实际清理占用：删除973份明确清单内的可重建缓存，共21.17 GiB。11组历史实验新增`CACHE_STATUS.md`；旧数组级核验须先重建相应缓存，不能把旧passed回执当作当前数组仍完整。模型、affine/delta、mask、指标、代码和协议保持；主方法缓存重放及未完成跨Source依赖保留。清单见`artifacts/maintenance/storage_cleanup_20261016/`，当前占用见[docs/STORAGE_STATUS.md](docs/STORAGE_STATUS.md)。

- Maintain the current native context/IN algorithms in `src/brats_tta/native/`; `ORIGIN.json` records their frozen experiment sources. Historical experiment runners remain byte-preserved snapshots with their own hashes, and do not automatically import the new package. New shared code must not depend on dated experiment import chains.
- Resource locations are centralized in `configs/paths.yaml`; `scripts/project.ps1 check|test|lint|teacher|experiments|adapt` uses the existing cv311 interpreter with a temporary source import path. The unused project `.venv` and empty singular `output/` were removed; do not recreate them as alternate working locations.
- Identical archived checkpoint copies share NTFS hardlinks and are read-only. All 75 checkpoint paths and contents remain valid. Never write checkpoint backups in place; training/resume outputs belong in a new experiment directory. Dedupe records are in `artifacts/maintenance/maintainability_20261010/`.
- `HANDOFF.md` is the concise current handoff. The full previous timeline and custom-framework README are archived in `docs/archive/`. Read current handoff before interpreting historical pending/cleanup states.

- Keep all project-owned files under this repository. New experiments go in `outputs/<experiment>/`, formal source backups in `artifacts/models/`, historical experiments/helpers in `artifacts/experiments/`, and maintenance receipts in `artifacts/maintenance/`. Do not create new experiment folders at a drive root or on the Desktop.
- All former E-drive BraTS/HD_TTA/nnUNet project directories have been consolidated here. Use `docs/STORAGE_LAYOUT.md` and `docs/MODEL_INDEX.md` for current locations. Most outputs/artifacts payloads remain Git-ignored. Selected PED research source files and PROTOCOL.md are visible to Git; see .gitignore and docs/PROJECT_FILE_GUIDE.md. Visibility does not mean committed or fully reproducible from a clone.
- Preserve original datasets at their existing locations (`E:/dataset` and the original GLI ZIP). Do not duplicate them into experiment folders. The user authorizes cleaning non-reused or infrequently used reconstructible project caches; preserve source models, learned adaptation parameters needed for reproduction, code, configuration, final masks and metrics. Check actual contents/dependencies before deletion.
- Migration changed only absolute paths and dependent text-file digest references in working copies. Original records are archived at `artifacts/maintenance/consolidation_20261010/original_text.zip`. Old `docs/*cleanup*.json` and historical inventory snapshots intentionally retain original operation-time paths. Do not treat them as current locations or invent a new experiment run from migrated receipts.

## Miniforge environments

- Miniforge root: `D:\Environment\miniforge3`
- Main development/training environment: `D:\Environment\miniforge3\envs\cv311`
  - Python 3.11
  - PyTorch 2.11.0+cu128
  - CUDA 12.8
  - Use `D:\Environment\miniforge3\envs\cv311\python.exe` for tests and local model checks.
- Legacy TTA environment: `D:\Environment\miniforge3\envs\TTA`
  - Python 3.8
  - PyTorch 2.1.2 / CUDA 11.8
- Local GPU: one `NVIDIA GeForce RTX 4060 Laptop GPU` with approximately 8 GiB VRAM.

## Kaggle

- Kaggle CLI: `D:\Environment\miniforge3\envs\cv311\Scripts\kaggle.exe`
- CLI version recorded at the previous verification: 2.2.4 (not rechecked online during the October 10 documentation refresh).
- Credentials file: `C:\Users\Yuuhi\.kaggle\credentials.json` (never print or expose its token).
- Kaggle account: `nagisa0v0`
- Public training dataset: `nagisa0v0/brats2023-gli-training-data`
- Local dataset source ZIP: `D:\Downloads\Zip\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip`
- Historical upload staging directory: `outputs\kaggle\brats2023_gli_training_public`; it does not exist locally as of October 10. This is not the retained original dataset location.

When submitting a kernel, use the Kaggle CLI from the `cv311` environment and include the current source tree in the submission. Do not clone an old Git commit when the working tree contains required changes. Kaggle kernels normally expose the GLI dataset as a ZIP below `/kaggle/input`; extraction to `/kaggle/working` and raw-manifest generation are part of the training script.

## Historical source protocols

原自定义、Original/Cicek、native IN/BN训练细节、标签与历史诊断完整保存在[历史Source协议](docs/archive/source_protocols.md)。操作这些模型前读取对应章节及各模型协议，不以旧远程状态判断当前机器或任务。

## Verification commands

```powershell
.\scripts\project.ps1 check
.\scripts\project.ps1 experiments check
.\scripts\project.ps1 lint
.\scripts\project.ps1 test
```

These check local resources, experiment metadata, critical code errors, and 45 focused CPU tests. Use `experiments check --local` to require local catalog references. Run broader tests only when relevant; Kaggle status is an optional online query, not a requirement for documentation edits:

```powershell
& 'D:\Environment\miniforge3\envs\cv311\python.exe' -m pytest -q
& 'D:\Environment\miniforge3\envs\cv311\Scripts\kaggle.exe' datasets status nagisa0v0/brats2023-gli-training-data
```
