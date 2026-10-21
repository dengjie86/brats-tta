# Python 包目录

[返回仓库入口](../../README.md) · [配置](../../configs/README.md) · [测试](../../tests/README.md)

当前 PED2023 背景共识与 IN 学习的公共算法已整合到 [native/](native/README.md)，资源位置由 [paths.py](paths.py) 和 `configs/paths.yaml` 管理。`cli/project.py` 检查本地资源，`cli/context_teacher.py` 重建已有缓存的共识；Windows 可通过 `scripts/project.ps1` 调用。

[experiments.py](experiments.py)和[cli/experiments.py](cli/experiments.py)只读实验登记、核对路径；不会导入或执行历史runner。工程分层见[架构说明](../../docs/reference/architecture.md)。

其他子目录保留自定义 3D U-Net、训练、预处理、评价及 TTA 实现。名称中的 nnU-Net-style 表示架构风格；这里的 `SourceTrainer` 不等于备份模型所用的 native nnU-Net trainer。旧实验代码保留为冻结快照，完整方法的新开发入口是[pipeline/](pipeline/README.md)，底层原语复用`native/`。

## 各子目录

| 目录 | 用途 | 文件职责 |
|---|---|---|
| [pipeline/](pipeline/README.md) | 完整union2/soft_extent/no_come | 单病例编排、IN/Senior后端、teacher、decoder、Source状态恢复及显式输入 |
| [native/](native/README.md) | 当前 native 公共算法 | 背景观察/共识、坐标、匹配对照、IN affine/损失/梯度重放、IN250 模型加载；不依赖旧实验导入链 |
| [cli/](cli/) | 命令行入口 | 解析参数，组织数据、模型、推断和结果写入；入口映射见根目录 `pyproject.toml` |
| [data/](data/) | 数据与训练输入 | `manifest.py` 发现病例和患者划分；`preprocessing.py` 处理归一化、脑掩膜和标签映射；`brats.py` 提供 Dataset/DataLoader；`transforms.py` 做 patch 采样与增强 |
| [models/](models/) | 三种自定义网络 | `unet3d.py` 六阶段 plain 网络与模型工厂；`original_unet3d.py` 五层拓扑对照；`cicek_unet3d.py` 四层 Cicek 2016 适配；均保留特征接口供 TTA 使用 |
| [losses/](losses/) | 监督训练损失 | `segmentation.py` 包含区域 Dice+BCE、四分类 Dice+CE、层级约束及 deep supervision 包装；配置决定具体使用哪一种 |
| [engine/](engine/) | 训练和普通推断 | `trainer.py` 管理训练、验证、优化器、调度器、checkpoint/恢复；`inference.py` 做滑窗融合和 NIfTI 输出 |
| [tta/](tta/) | 自定义框架的测试时适应 | `tent.py` 为 BN affine 熵最小化；`sar.py` 为 reliable entropy + SAM；`vptta.py` 为 3D Fourier prompt/统计适应/记忆库；`tegda.py` 为质量、特征库、teacher 的兼容实现；`inference.py` 组织逐 patch 适应及 online/two-pass 推断 |
| [metrics/](metrics/) | 指标计算 | `segmentation.py` 提供区域 Dice、层级违例、HD95、lesion-wise Dice/TP/FP/FN 和汇总 |
| [utils/](utils/) | 公共支撑 | `atomic_io.py` 原子写入；`checkpoint.py` 权重读写；`distributed.py` DDP/进程通信；`reproducibility.py` 随机种子与设备选择 |

根部 `config.py` 负责 YAML 读取、环境变量展开、校验、严格 dotted-key 覆盖、配置快照和指纹；`__init__.py` 导出网络和版本。各子包的 `__init__.py` 提供导入接口。

## CLI 文件

| 文件 | 用途 |
|---|---|
| [adapt_context.py](cli/adapt_context.py) | 完整单病例方法入口：MRI/模型显式输入，独立输出目录，记录身份与完成状态 |
| [project.py](cli/project.py) | 当前入口：解析统一资源配置并检查路径是否存在 |
| [experiments.py](cli/experiments.py) | 主要实验的list/show/check，只读身份、记录状态和本地引用 |
| [context_teacher.py](cli/context_teacher.py) | 当前入口：用已有 Source/blur/neutral 概率及保护区生成共识数组；不执行 MRI 推理 |
| [prepare_manifest.py](cli/prepare_manifest.py) | 扫描原始数据，按指定标签 schema 生成病例清单与 train/val 划分 |
| [preprocess.py](cli/preprocess.py) | 将 manifest 对应的原始数据写成预处理缓存 |
| [extract_brain_masks.py](cli/extract_brain_masks.py) | 调用 HD-BET 生成派生脑掩膜与 manifest；保留原图和原标签 |
| [train_source.py](cli/train_source.py) | 组装配置、训练数据、模型、损失和 SourceTrainer，支持运行参数覆盖及分布式训练 |
| [evaluate_source.py](cli/evaluate_source.py) | 普通 source checkpoint 验证 |
| [evaluate_tta.py](cli/evaluate_tta.py) | Source/TENT 评价、逐例记录、指纹核验、恢复与可选 HD95/lesion-wise Dice |
| [evaluate_advanced_tta.py](cli/evaluate_advanced_tta.py) | SAR/VPTTA/TEGDA 单方法评价、病例顺序和适应状态保存 |
| [evaluate_legacy_unet.py](cli/evaluate_legacy_unet.py) | 已发布 TEGDA plain U-Net 的架构对照；处理其 Z/Y/X 轴序，不替代当前 source evaluator |
| [infer.py](cli/infer.py) | 普通模型分割并保存 NIfTI |
| [inspect_model.py](cli/inspect_model.py) | 根据配置检查模型结构、参数量和输入输出 |
| [common.py](cli/common.py) | 日志设置和自定义 checkpoint 模型加载 |

## 与 native PED2023 主线的边界

- 自定义数据代码的模态顺序是 `t1n,t1c,t2w,t2f`；当前 native 实验使用 `t2f,t1n,t1c,t2w`。应跟随对应模型元数据。
- `tta/tent.py` 面向 BN；当前背景/IN 公共实现位于 `native/`。产生旧结果的 runner 仍使用冻结实验代码，见[依赖边界](../../docs/reference/architecture.md)。
- 本包 `hd95_binary` 使用 surface-area-weighted HD95，单边空默认惩罚 `374.0`；当前 native PED2023 实验用六邻域表面、双向距离合并的 95 分位，单边空为 `null` 并报告有效病例数。两套汇总不能直接混用。
- 旧 `src/brats_tta.egg-info/` 的6个生成文件已原样归档到 `artifacts/maintenance/status_refresh_20261010/obsolete_egg_info/`，其中保留着过时三输出模型说明和仅6项 CLI。当前14项命令入口以 `pyproject.toml` 为准；安装/构建会再生元数据，本次未安装项目或修改 Conda 环境。
- 各层 `__pycache__/` 为可再生字节码缓存；10月10日曾清理，之后导入已重新生成，不能把此前清理记录当作当前目录为空。
