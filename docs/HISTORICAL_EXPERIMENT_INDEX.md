# 历史实验与复用代码索引

[文件总说明](PROJECT_FILE_GUIDE.md) · [当前研究目录](EXPERIMENT_INDEX.md) · [模型](MODEL_INDEX.md)

以下 **43 个目录** 均在 `artifacts/experiments/`。根据实际协议、报告、代码入口和产物整理；有些历史代码仍由当前流程调用，有些只保留研究证据。目录名中的 `HD_TTA` 是当时实验命名，不能据此判断当前方法来自 HD-TTA。

旧概率、预处理数组等部分已清理，代码、最终分割、指标和保留的参数状态仍在。下面的“已完成”指历史实验确实执行过，不保证如今不重建缓存就能直接重跑。运行状态以最新交接为准，旧 README 中的 PID、排队和监控状态不代表现在仍有任务运行。

## 当前仍直接使用的材料

| 目录 | 做什么 / 现在为什么保留 | 阅读入口 |
|---|---|---|
| [nnUNet_PED2023_IN_analysis_20261005](../artifacts/experiments/nnUNet_PED2023_IN_analysis_20261005/) | IN250 的 99 例冻结 Source 基线：分割、reference/error、Dice/HD95、图和 HTML | `outputs/full/ANALYSIS.md`、`outputs/full/index.html` |
| [BraTS_PED2023_TC_shape_review_20261006](../artifacts/experiments/BraTS_PED2023_TC_shape_review_20261006/) | 从 Source 结果分析 TC 形状，整理模型卡和好坏病例 | `easy_view/index.html`、`MODEL_CARD.md`、`REPORT.md` |
| [BraTS_PED2023_norm_TTA_20261006](../artifacts/experiments/BraTS_PED2023_norm_TTA_20261006/) | native IN TENT/SAR 的 99 例结果、Kaggle 包与下载；其中 adapter/指标实现被当前流程复用 | `REPORT.md`、`native_adapter.py`、`independent_metrics.py` |
| [BraTS_PED2023_axis_probe_20261006](../artifacts/experiments/BraTS_PED2023_axis_probe_20261006/) | 12 例轴置换观察及 decoder 学习；其 Source 概率仍供背景实验复用 | `FINAL_REPORT.md`、`common.py`、`cases/` |
| [BraTS_PED2023_snapshot_probe_20261006](../artifacts/experiments/BraTS_PED2023_snapshot_probe_20261006/) | 175/200/225 轮快照教师诊断，未解决 NC→ED；`common.py` 仍提供匹配对照函数 | `REPORT.md`、`common.py`；旧快照概率已清理 |

## 10 月 6 日机制探索与早期候选

| 目录 | 实际内容与用途 | 阅读入口 |
|---|---|---|
| [BraTS_PED2023_TTA_hypothesis_20261006](../artifacts/experiments/BraTS_PED2023_TTA_hypothesis_20261006/) | 早期组织分布与自预测锚点假设检验；固定实现未通过当时门槛 | `PROTOCOL.md`、`REPORT.md`、`predict_evidence.py` |
| [BraTS_PED2023_multiscale_probe_20261006](../artifacts/experiments/BraTS_PED2023_multiscale_probe_20261006/) | 辅助分类头/多尺度与两个真实适配器的局部试验；不是后来的输入缩放实验 | `REPORT.md`、`predict_probe.py`、`config.json` |
| [BraTS_PED2023_prior_TTA_20261006](../artifacts/experiments/BraTS_PED2023_prior_TTA_20261006/) | 多轮先验、NC/ED 分类头偏置校准及完整体积验证；有 TC 收益但独特先验贡献不足 | `FINAL_ASSESSMENT.md`、`REPORT.md`、`full_volume/` |
| [BraTS_PED2023_support_probe_20261006](../artifacts/experiments/BraTS_PED2023_support_probe_20261006/) | 局部特征/自动参照及 GT 辅助读出诊断；oracle 是分析上限，不是可部署方法 | `REPORT.md`、`ORACLE_PROTOCOL.md`、`probe.py` |
| [BraTS_PED2023_group_TTA_20261006](../artifacts/experiments/BraTS_PED2023_group_TTA_20261006/) | 12 例局部组织分群与语义拒绝；未超过同修改量的强对照 | `REPORT.md`、`group_method.py` |
| [BraTS_PED2023_three_teachers_20261006](../artifacts/experiments/BraTS_PED2023_three_teachers_20261006/) | GMM、影像图传播、多上下文互教；含真实分类头/末端 IN 更新，局部开发结果 | `ASSESSMENT.md`、`teachers.py`、`adaptation.py` |
| [BraTS_PED2023_frozen_head_candidates_20261006](../artifacts/experiments/BraTS_PED2023_frozen_head_candidates_20261006/) | 冻结 head、IN-only 和候选筛选的三个补试；局部结果未改善 TC | `ASSESSMENT.md`、`in_only.py`、`candidates.py` |
| [BraTS_PED2023_boundary_probe_20261006](../artifacts/experiments/BraTS_PED2023_boundary_probe_20261006/) | 12 例局部稀疏边界候选；覆盖很小，另补算 ROI HD95 | `REPORT.md`、`HD95_SUPPLEMENT.md`、`run.py` |
| [BraTS_PED2023_boundary_feasibility_20261006](../artifacts/experiments/BraTS_PED2023_boundary_feasibility_20261006/) | 12 例局部 MRI/特征连续性，固定 WT/ET 的机制验证 | `REPORT.md`、`run.py` |
| [BraTS_PED2023_continuity_full99_20261006](../artifacts/experiments/BraTS_PED2023_continuity_full99_20261006/) | 将连续性 ED→NC 扩张检验扩展至 99 例完整体积；固定 WT/ET、模型冻结 | `REPORT.md`、`run.py`、`per_case_metrics.csv` |
| [BraTS_PED2023_MRI_matched_controls_20261006](../artifacts/experiments/BraTS_PED2023_MRI_matched_controls_20261006/) | 给上述方法补同修改量的 margin/geometry/random 对照，审查 TC 涨分原因 | `REPORT.md`、`LITERATURE.md`、`controls.py` |
| [BraTS_PED2023_heldout_boundary_20261006](../artifacts/experiments/BraTS_PED2023_heldout_boundary_20261006/) | 用未参与候选生成的 MRI 模态检查边界；99 例、WT/ET 固定 | `REPORT.md`、`probe.py` |
| [BraTS_PED2023_fourclass_boundary_20261006](../artifacts/experiments/BraTS_PED2023_fourclass_boundary_20261006/) | 开放背景/ET 变化，验证完整四分类边界选择器与区域变化 | `REPORT.md`、`predict.py`、`direction_diagnosis.json` |
| [BraTS_PED2023_anatomy_contrast_20261006](../artifacts/experiments/BraTS_PED2023_anatomy_contrast_20261006/) | 对侧正常结构、T2/FLAIR 先验的两种固定观察；12 例完整体积，未进入学习 | `REPORT.md`、`priors.py` |
| [BraTS_PED2023_n4_probe_20261006](../artifacts/experiments/BraTS_PED2023_n4_probe_20261006/) | 用 N4 校正 MRI 低频偏场后再做冻结 Source 推理；是已有预处理机制 | `REPORT.md`、`correct.py`、`predict.py` |
| [BraTS_PED2023_SAR_audit_20261006](../artifacts/experiments/BraTS_PED2023_SAR_audit_20261006/) | 解释 native SAR 的 738 次恢复，检查 EMA/阈值、真实 patch 更新与无恢复分支 | `REPORT.md`、`log_audit.py`、`real_patch_audit.py` |
| [BraTS_TTA_literature_design_20261006](../artifacts/experiments/BraTS_TTA_literature_design_20261006/) | 文献网页、提取文本与阅读材料；支撑设计演变，不是患者数据或效果结果 | `.html` / `.txt` 与对应的 docs 设计 v4 等文档 |

## HD-TTA 比较、诊断和汇报

| 目录 | 实际内容与用途 | 阅读入口 |
|---|---|---|
| [HD_TTA_PED2023_20260930](../artifacts/experiments/HD_TTA_PED2023_20260930/) | 历史源模型上的 PED2023 Source/Classical/HD-TTA 与消融；旧八镜像配置 | `PROTOCOL.md`、`run_experiment.py`、`outputs/` |
| [HD_TTA_PED2023_axis0_20261001](../artifacts/experiments/HD_TTA_PED2023_axis0_20261001/) | 将该历史源模型的镜像轴修正为 axis0 后的对照 | `PROTOCOL.md`、`outputs/`；不是当前 native 八镜像协议 |
| [HD_TTA_PED2023_diagnostics_20261001](../artifacts/experiments/HD_TTA_PED2023_diagnostics_20261001/) | 检查镜像轴错配、compactness 收缩和强度引导失效 | `AUDIT.md`、`diagnose.py`、`inspect_guidance.py` |
| [HD_TTA_PED2023_params_20261001](../artifacts/experiments/HD_TTA_PED2023_params_20261001/) | 六例旧 HD-TTA 参数探索；只评价 WT | `RESULTS.md`、`protocol.json`、`sweep.py` |
| [HD_TTA_PED2023_params_round2](../artifacts/experiments/HD_TTA_PED2023_params_round2/) | 六例调参后额外十二例固定方案比较 | `RESULTS.md`、`selection.json`、`experiment.py` |
| [HD_TTA_PED2023_params_round3](../artifacts/experiments/HD_TTA_PED2023_params_round3/) | 18 例探索、额外 30 例固定方案评价；均为已接触开发数据 | `RESULTS.md`、`PROTOCOL.md`、`study.py` |
| [HD_TTA_PED2023_native_nnunet_20261002](../artifacts/experiments/HD_TTA_PED2023_native_nnunet_20261002/) | native IN WT 二分类 Dice+CE 模型上的 99 例 HD-TTA/消融 | `README.md`、`PROTOCOL.md`、`outputs/` |
| [HD_TTA_PED2023_4class_nnunet_20261003](../artifacts/experiments/HD_TTA_PED2023_4class_nnunet_20261003/) | native IN 四分类模型的 WT 概率合并后做 HD-TTA；不是当前四分类 argmax 教师 | `README.md`、`PROTOCOL.md`、`outputs/` |
| [HD_TTA_PED2023_wt_pure_dice_20261004](../artifacts/experiments/HD_TTA_PED2023_wt_pure_dice_20261004/) | pure-Dice WT final250 的 99 例官方代码比较，含 gate 触发审计 | `outputs/full/RESULTS.md`、`gate_audit.json` |
| [HD_TTA_PED2023_wt_paper_20261004](../artifacts/experiments/HD_TTA_PED2023_wt_paper_20261004/) | 与上述相同模型/病例的论文公式 sum/mean 约定比较 | `PROTOCOL.md`、`paper_methods.py`、`outputs/` |
| [HD_TTA_GLI_BN_regions_20261005](../artifacts/experiments/HD_TTA_GLI_BN_regions_20261005/) | BN best225 在 GLI 253 例上的区域 Dice 复算；是源域验证，不是 PED | `PROTOCOL.md`、`outputs/`；重推理需重建已删训练缓存 |
| [HD_TTA_group_meeting_20261006](../artifacts/experiments/HD_TTA_group_meeting_20261006/) | 组会报告、对比表、消融图、病例图、讲稿与浏览检查 | `report.html`、`speaking_notes.txt`、`build_report.py` |

## 源模型训练包与中断尝试

这里保存“如何训练、部署、监控”的材料；已经完成的正式权重归在 `artifacts/models/`。

| 目录 | 实际内容与当前状态 | 阅读入口 |
|---|---|---|
| [HD_TTA_source_nnunet_20261002](../artifacts/experiments/HD_TTA_source_nnunet_20261002/) | 最初仅 WT 的 native 源训练准备包，后来被双分支方案替代 | `README.md`、`prepare_dataset.py`、`nnUNetTrainerHDTTA250.py` |
| [HD_TTA_source_dual_nnunet_20261002](../artifacts/experiments/HD_TTA_source_dual_nnunet_20261002/) | 已完成 IN WT/四分类 Dice+CE 的准备、患者划分、trainer、启动和控制核验 | `README.md`、`nnUNetTrainerBraTS250.py`、`splits_final.json` |
| [HD_TTA_source_dice_kaggle_20261003](../artifacts/experiments/HD_TTA_source_dice_kaggle_20261003/) | pure-Dice 初次 Kaggle 提交、额度/启动失败及分段调度记录 | `README.md`、`chain_state.json`、`kernel.py` |
| [HD_TTA_source_dice_local_20261003](../artifacts/experiments/HD_TTA_source_dice_local_20261003/) | pure-Dice 本地 7 epoch 中断尝试；转换代码、plans、split、日志保留，旧数组和权重已删 | `DERIVED_DATA_CLEANUP_20261010.md`、`local_run.py`；旧说明中的“权重保留”已被后续清理覆盖 |
| [HD_TTA_source_dice_kaggle_retry_20261003](../artifacts/experiments/HD_TTA_source_dice_kaggle_retry_20261003/) | Kaggle 重试停在 74 epoch，后来由 AutoDL 从头训练取代；代码日志保留、权重已删 | `README.md`、`chain_state.json`、`schedule_verification.json` |
| [HD_TTA_source_dice_autodl_20261003](../artifacts/experiments/HD_TTA_source_dice_autodl_20261003/) | 完成 pure-Dice250 的部署、重新预处理、CPU/I/O 性能诊断和下载工具 | `README.md`、`CACHE_OPTIMIZATION.md`、`fresh_pipeline.py` |
| [HD_TTA_source_4class_bn_20261004](../artifacts/experiments/HD_TTA_source_4class_bn_20261004/) | native IN→BN norm-only 控制的训练包、排队/监控/下载工具；正式权重在 models/bn_dicece | `README.md`、`bundle/`、`verify_downloaded.py` |

这些入口可能连接远端、安装依赖、启动训练或上传数据。此索引只说明用途，本轮没有执行它们。

## 九月份 PED2024 旧框架实验

| 目录 | 内容与边界 | 阅读入口 |
|---|---|---|
| [BraTS_PED_TTA_20260928](../artifacts/experiments/BraTS_PED_TTA_20260928/) | 历史自定义 BN checkpoint、PED2024 脑提取/manifest、Source/TENT/SAR/VPTTA 等 260 例材料、HD-BET 权重和 adapter 状态 | `RESULTS.md`、`manifests/`、`weights_inventory.json` |
| [BraTS_PED_TTA_20260929_suite](../artifacts/experiments/BraTS_PED_TTA_20260929_suite/) | 继续比较 InTEnt、IST、CoTTA、TEGDA、HD-TTA 等；包含复用结果及取消/不完整运行 | `EXPERIMENT_STATUS.md`、`FINAL_RESULTS.md`、`DATASET_AUDIT.md` |

这两组不是当前 PED2023/native/no-HD-BET 协议。IST/CoTTA 的部分结果不能当完整 baseline；历史参考 `references/IST_PED.py` 存在原有语法错误，作为参考快照保留，不是本轮验证通过的执行入口。
