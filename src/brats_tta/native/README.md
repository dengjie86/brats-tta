# 当前 native 方法的维护入口

这里维护 GLI→PED2023 背景共识和轻量 IN 学习的公共算法，属于应纳入 Git 的源码范围；当前新增文件尚未提交。没有通过 `sys.path` 或动态 import 依赖 `outputs/`、旧实验目录，也没有硬编码本机模型路径。

| 模块 | 用途 |
|---|---|
| `context.py` | `make_contexts` 保护病灶邻域、构造 blur/neutral 背景；`teacher` 按双观察一致性和置信度生成预测 |
| `coordinates.py` | native crop / NIfTI 数组轴序转换与 patch 裁取 |
| `controls.py` | 保持类别转换数量一致的 margin、random、geometry 对照 |
| `adaptation.py` | IN affine 参数选择、中心化 logit 目标、Source extent 锚点；另有显式调用的类别转换投影，保留 hierarchical 对照 |
| `replay.py` | native 聚合目标与逐视图对照、FP32 adjoint、单个网络图的梯度重放 |
| `efficient.py` | 单/八视角学习布局、soft_extent 目标及梯度重放；公共实现不读取实验目录 |
| `predictor.py` | 严格核验并加载 IN250 四分类 Source；设备可选，默认 CUDA |
| `ORIGIN.json` | 提取来源、原文件 SHA-256、函数及改动边界 |

10月10日提取时阈值、通道顺序、损失和计算精度沿用原实验。10月11日新增`project_transition_delta`供[类别对蒸馏实验](../../../outputs/ped2023_pairwise_distillation_20261011/PROTOCOL.md)显式调用，旧默认loss与目标函数不变。`predictor.py`限定当前IN250协议，不能直接用于BN225。调用前先使用 [configs/paths.yaml](../../../configs/paths.yaml) 定位资源。

已验证公共模块测试、冻结实现CPU数值/梯度回归、已有共识逐体素比对、IN250 CPU加载；10月11日又在00003完成旧plain八步GPU学习复现，概率、首梯度、state逐值相同。新类别对候选24例已完成评价和独立核验，但没有改善旧plain方案。各项性能证据属于对应冻结实验；工程验证不代表方法已定稿或已经适用于BN。

[学习后前景偏置校正](../../../outputs/ped2023_pairwise_distillation_20261011/extent_projection/README.md)目前仍在实验目录的`projection.py`实现，尚未并入本包。它复用plain预测，不使用类别对投影；其完整队列预测、评价和独立核验已完成，主要98唯一组；WT/ET相对Source仍有代价。不要将本包的`project_transition_delta`与该单个前景偏置校正混为一谈。

随后完成的[E修改保留](../../../outputs/ped2023_teacher_retention_20261011/README.md)和[C完整共识保留](../../../outputs/ped2023_teacher_retention_20261011/accepted_support/README.md)也仍在各自实验目录的`retention.py`，各有98组评价与独立核验。它们使用原teacher标签及前景校正结果做最终标签组合，C另使用原`teacher_accepted`集合；没有加入本包默认学习目标或生成新的网络权重。最新效果与取舍看HANDOFF。

```python
from brats_tta.paths import resource_paths
from brats_tta.native import make_contexts, teacher
from brats_tta.native.predictor import load_predictor

model_dir = resource_paths()["source_in"]
predictor = load_predictor(model_dir)
# x: native-preprocessed (4,Z,Y,X), 1 mm; source: (Z,Y,X) argmax labels
# contexts, protected, info = make_contexts(x, source)
# prediction, accepted = teacher(source, [blur_probability, neutral_probability], protected)
```

该示例的模型加载需要已经安装的 nnunetv2。只用背景数组处理、共识、坐标或损失函数，不需要 nnunetv2；模块导入不会加载模型或初始化 CUDA。

`outputs/` 与 `artifacts/experiments/` 中产生旧结果的代码作为冻结快照保留，连同原 hash 核验链继续使用，**不会自动改用新包**。10月11日类别对实验和[plain完整队列](../../../outputs/ped2023_native_objective_full99_20261011/README.md)显式调用本包，各自记录队列和冻结依赖。完整队列已通过旧plain八步逐值复现并完成评价、独立核验；其`preparation.json`记录本包全部7个Python文件的哈希。复现应匹配该版本，未来修改须记录新的实验依赖。公共算法以本目录为维护入口；修改前核对现有实验依赖，复现旧结果仍须匹配当时源码。最新研究状态见[HANDOFF](../../../HANDOFF.md)。

本机用 `scripts/project.ps1 teacher --help` 查看缓存共识入口；脚本自动配置 `src` 导入路径。已安装此包或设置好 `PYTHONPATH` 时，也可使用 `python -m brats_tta.cli.context_teacher` 及上面的 Python 示例。输出是 native crop 数组；完整体积导出用 `coordinates.restore` 并携带原图空间元数据。缓存命令不加载模型、不读取标签，也不覆盖已有输出。
