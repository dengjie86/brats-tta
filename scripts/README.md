# 脚本目录

[返回仓库入口](../README.md) · [配置说明](../configs/README.md) · [当前实验索引](../docs/EXPERIMENT_INDEX.md)

当前统一入口是 [project.ps1](project.ps1)：`check` 检查资源路径，`test` 运行当前 native 公共模块测试，`teacher` 从已保存概率重建背景共识。脚本使用已有 Conda cv311，并临时配置 `src` 导入；`-Python` 可指定其他已有解释器。

新增`experiments list|show <id>|check [--local]`读取实验登记，`lint`检查公共src/tests的关键错误。登记查询不会运行旧脚本或恢复暂停任务。完整用法见[快速开始](../docs/guides/getting-started.md)。

`test`当前包含native context/adaptation/efficiency、transition projection、project paths、experiments和context pipeline七个模块，共45项CPU测试；本机和CI调用同一清单。不是完整训练/DDP测试套件。见[测试说明](../tests/README.md)。

其余 15 个历史脚本/元数据文件主要服务九月份的自定义模型、拓扑对照和 Source/TENT/SAR/VPTTA/TEGDA 流程。完整单病例方法入口为`project.ps1 adapt`，调用`src/brats_tta/pipeline/`；背景与IN原语在 `src/brats_tta/native/`；历史实验 runner 仍在 `outputs/ped2023_*`。

## AutoDL / Linux

| 文件 | 实际执行内容 |
|---|---|
| [autodl_train_source.sh](autodl_train_source.sh) | 检查环境，按患者生成 GLI train/val manifest，启动自定义 `train_source`，记录日志和 GPU 状态；默认六阶段 BN 配置 |
| [autodl_original_unet_ablation.sh](autodl_original_unet_ablation.sh) | 顺序运行五层原始拓扑无 BN 与 BN 对照 |
| [autodl_cicek_unet3d_ablation.sh](autodl_cicek_unet3d_ablation.sh) | 先做显存检查，再运行 Cicek BN、六阶段 BN 对照和可选 Cicek 无 BN |
| [autodl_evaluate_advanced_tta.sh](autodl_evaluate_advanced_tta.sh) | 对指定 checkpoint/SSA/PED manifest 运行高级 TTA；含 TEGDA 显存预检及 SAR 恢复消融 |

## Windows 本地流程

| 文件 | 实际执行内容 |
|---|---|
| [run_local_target_pipeline.ps1](run_local_target_pipeline.ps1) | 历史 PED 脑提取、PED Source、SSA/PED TENT 流程；使用旧数据 manifest 和独立 HD-BET 环境 |
| [run_source_tent_lesionwise_local.ps1](run_source_tent_lesionwise_local.ps1) | Source/TENT × SSA/PED 四组 lesion-wise Dice 评价 |
| [run_tent_bn_ablation_local.ps1](run_tent_bn_ablation_local.ps1) | shallow/deep BN affine、仅 batch statistics、保留 source statistics 四种 TENT 对照 |
| [run_advanced_tta_pilot.ps1](run_advanced_tta_pilot.ps1) | 小样本执行/显存检查；默认 1 例、`96³`、TEGDA norm scope，不能当正式队列成绩 |
| [run_advanced_tta_full_local.ps1](run_advanced_tta_full_local.ps1) | SAR、SAR no-recovery、VPTTA、TEGDA × SSA/PED 的八组本地运行及状态记录 |

这些脚本的结果输出默认值已在上次集中整理时改到项目 `outputs/`；部分历史 checkpoint / manifest 输入仍写着旧桌面位置，使用前需要显式提供现存输入。它们不是当前 PED2023 上下文主线的启动器；本次文件导航整理没有执行或修改脚本。

## Kaggle / Cicek

| 文件 | 实际执行内容 |
|---|---|
| [kaggle_cicek_memory_probe.py](kaggle_cicek_memory_probe.py) | 定位上传的源码和配置，做 FP32 显存探针及真实数据 pilot，保存 JSON |
| [kaggle_cicek_train.py](kaggle_cicek_train.py) | 定位/解压 GLI 数据、准备患者划分、恢复已有进度、生成运行配置并启动分段训练 |
| [kaggle_cicek_chain.ps1](kaggle_cicek_chain.ps1) | 轮询 kernel、下载输出、打包续训 dataset 并提交下一段 kernel；是执行器，不是只读查看器 |
| [kaggle_cicek_kernel_metadata.json](kaggle_cicek_kernel_metadata.json) | 显存探针 kernel 的 code file、数据依赖及机器设置 |
| [kaggle_cicek_train_kernel_metadata.json](kaggle_cicek_train_kernel_metadata.json) | 分段训练 kernel 的元数据 |
| [kaggle_cicek_source_dataset_metadata.json](kaggle_cicek_source_dataset_metadata.json) | 打包当前源码供 Kaggle 使用的私有依赖 dataset 元数据 |

`__pycache__/` 是 Python 自动生成的字节码缓存；其中可能保留已经删除脚本的缓存文件，不能按缓存名判断当前还有该功能。
