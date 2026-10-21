# 测试目录

[返回仓库入口](../README.md) · [被测 Python 包](../src/brats_tta/README.md)

这里有`conftest.py`和22个`test_*.py`文件：15个自定义框架模块、5个native/路径模块、1个实验登记模块，以及1个完整pipeline模块。`conftest.py`配置本地源码导入。实验自带的检查脚本与核验JSON仍在各自目录；工程测试通过不代表科学结论成立。

| 文件 | 验证范围 |
|---|---|
| [test_context_pipeline.py](test_context_pipeline.py) | 冻结teacher/decoder回归、概率融合梯度与8步更新、Source恢复、CLI防覆盖、预处理和无历史导入依赖 |
| [test_experiments.py](test_experiments.py) | 无资产登记校验、只读查询不执行runner、外部路径拒绝、重复ID与错误完成状态拒绝 |
| [test_native_context.py](test_native_context.py) | 背景保护/零填充、全类别共识门控、匹配对照、坐标和缓存 CLI 防覆盖 |
| [test_native_adaptation.py](test_native_adaptation.py) | logit 平移不变性、概率目标、八视图梯度重放与完整计算图一致性 |
| [test_native_efficiency.py](test_native_efficiency.py) | 单视角/多视角效率路径的梯度重放一致性 |
| [test_transition_projection.py](test_transition_projection.py) | 保留类别对contrast的最小范数投影、全部类别置换对称性、无编辑及无效输入 |
| [test_project_paths.py](test_project_paths.py) | 路径随仓库解析、工作目录变化、配置校验与缺失资源返回码 |
| [test_atomic_io.py](test_atomic_io.py) | 临时文件锁重试、不可恢复写入错误、旧文件保护 |
| [test_brain_masks.py](test_brain_masks.py) | 掩膜几何/二值/非空检查、先掩膜后归一化、GT 不变、旧结果设置隔离 |
| [test_config.py](test_config.py) | 五份源配置、网络选择、参数覆盖和错误配置拒绝 |
| [test_data_pipeline.py](test_data_pipeline.py) | 病例发现、患者划分、预处理和 Dataset、分布式验证无重复、训练增强 |
| [test_distributed_training.py](test_distributed_training.py) | 两进程 DDP 的训练、验证及 checkpoint |
| [test_hd95.py](test_hd95.py) | 物理间距、空掩膜约定及 area-weighted 实现一致性 |
| [test_inference.py](test_inference.py) | 滑窗拼接、padding 裁回、NIfTI 几何保存 |
| [test_lesion_wise_dice.py](test_lesion_wise_dice.py) | 病灶检测、误检/漏检惩罚、体积阈值和区域汇总 |
| [test_model_and_loss.py](test_model_and_loss.py) | 网络拓扑、归一化、参数量、deep supervision 与损失梯度 |
| [test_preprocessing.py](test_preprocessing.py) | 非零归一化、标签 schema、类别/区域转换和层级修复 |
| [test_tent.py](test_tent.py) | BN 更新与重置、scope/统计消融、online/two-pass、AMP 与更新次数 |
| [test_sar.py](test_sar.py) | BN 更新、可靠性筛选、状态保存/恢复 |
| [test_vptta.py](test_vptta.py) | Fourier identity、prompt 更新、memory bank 与状态恢复 |
| [test_tegda.py](test_tegda.py) | ADIC、feature bank、梯度、更新范围及状态恢复 |
| [test_trainer.py](test_trainer.py) | 小型训练—验证—checkpoint—恢复完整流程 |

在仓库根目录使用已配置的 `cv311` 环境：

```powershell
& 'D:\Environment\miniforge3\envs\cv311\python.exe' -m pytest -q
```

`scripts/project.ps1 test`现运行native context/adaptation/efficiency、transition projection、project paths、experiments和context pipeline七个模块，共45项CPU小测试。CI使用同一入口，不依赖真实数据或权重。完整pytest命令包含历史训练/DDP等模块，按任务需要运行。历史冻结回归见 `artifacts/maintenance/maintainability_20261010/regression.json`；GPU复现与研究状态看根目录HANDOFF。`__pycache__/`是可再生字节码。
