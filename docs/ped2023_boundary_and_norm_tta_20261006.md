# PED2023：边界验证及当前模型TENT/SAR

> 2026-10-06补充审计：SAR的738/738恢复属实，但当前源模型更新前熵已远低于0.2，绝对分类阈值在四类密集分割中退化，不能将此表解释为充分适配后的通用SAR性能。作者核心SAM真实patch对照通过；另有base-SGD momentum恢复状态差异。3例关闭恢复诊断保留更新但仅改变1/1/104体素，未读GT、不代表改进成绩。详见[ped2023_sar_audit_20261006.md](ped2023_sar_audit_20261006.md)和`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_SAR_audit_20261006/REPORT.md`。

## 最终Kaggle结果（已独立核验）

两个账号各两块T4均完成99例；`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_norm_TTA_20261006/REPORT.md`、`evaluation_verified.json`、`verification.json`已生成。396个全体积mask核验通过，Dice和HD95以独立KD-tree对照上传EDT复算。以下覆盖后文运行中状态。

| 方法 | WT Dice(%) | TC Dice(%) | ET Dice(%) | WT HD95(mm,n) | TC HD95(mm,n) | ET HD95(mm,n) |
|---|---:|---:|---:|---|---|---|
| source | 89.3333 | 48.8627 | 54.6715 | 5.2682,99 | 14.7467,92 | 9.3231,85 |
| TENT-IN | 89.5657 | 47.8587 | 54.8304 | 4.4156,99 | 15.0171,91 | 9.1098,84 |
| SAR-IN | 89.3333 | 48.8627 | 54.6715 | 5.2682,99 | 14.7467,92 | 9.3231,85 |

TENT738次有限梯度更新、无恢复；TC完全漏检7→8例，ET完全漏检7→8例。SAR738次SAM更新后全部触发EMA<0.2恢复，99例mask与source完全一致；不能写成SAR未执行更新。HD95有效数变化要用同病例配对结果解释，不能直接比较剔除不同漏检病例后的均值。

北京时间：TENT11:10:16提交，11:43:09计算结束，总32.88分钟（病例阶段24.48分钟）；SAR11:12:34→11:45:00，总32.43分钟（病例阶段25.66分钟）。两账号并行总34.74分钟。每病例适应推理均值TENT8.33秒、SAR10.39秒；均不包含source/IO/GT评价。结果下载后本地独立核验另4.81分钟。

两账号source逐体素相同；与本机既有source相比70例共194体素差异，报告用各任务匹配source。NumPy环境变化见原说明。原99与去重98表、空区域计数、逐例CSV均在完整报告中。

2026-10-06，用户授权本地验证边界思路，并使用两个Kaggle账号各两块GPU运行TENT/SAR；随后要求除Dice外报告HD95。

## 本地边界验证已完成

- 目录：`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_boundary_probe_20261006`。
- 既有12个开发病例，80³局部ROI、128³网络上下文；冻结模型，固定WT/ET，只允许源ED→NC。
- 四模态MRI与冻结32D特征共同检查相邻组织差异；可靠源NC作为语义锚点，单步接受，无递归扩张、无新增教师训练。
- 仅00062一例修改84体素，全部恢复NC，损害0；覆盖212816个NC→ED错误中的0.0395%。TC Dice58.1153%→58.1304%，WT86.9499%、ET50.5789%硬保持。
- 局部TC HD95保持11.2513mm，有效11/12；WT4.8086mm（12/12）、ET8.2608mm（10/12）均不变。ROI被裁剪，不能当全体积HD95。
- 预设严格源NC/ED交界对没有留下真实NC/ED分界，主要AUROC不可计算。事后全部相邻区域诊断仅1例可评价，不能支持跨病例泛化。
- 当前规则覆盖不足，未继续启动参数学习。报告与回执：`REPORT.md`、`HD95_SUPPLEMENT.md`、`evaluation.json`、`hd95_evaluation.json`、`verification.json`。

## TENT/SAR全体积运行

- 目录：`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_norm_TTA_20261006`；运行协议`PROTOCOL.md`，HD95约定`HD95_PROTOCOL_ADDENDUM.md`。
- 源模型：native nnU-Net四类IN、Dice+CE、epoch250；checkpoint SHA256 `2b92812a25d4842d34a1d6ade2c60594f8d8bed30eb80af1b2b00bba5eeb6875`。
- 全99例PED2023，另报去重98组文件；原生预处理、无HD-BET。均为已有开发证据。
- 逐病例重置源权重、优化器、scaler和EMA；病例内patch保留更新，无CTTA。
- 这是dense3D/IN移植：更新全部22层IN仿射参数7680标量；卷积和分类头冻结，TENT/SAR不固定WT/ET。
- 128³patch、0.5滑窗、Gaussian、8镜像；每patch先输出更新前8镜像结果，再对identity熵做一步更新，不额外用最终参数做第二轮推理。
- TENT：Adam1e-3，全体素四类熵。SAR：两次entropy<0.4ln4，SAM-SGD1e-3、momentum0.9、rho0.05、EMA0.9、恢复阈值0.2。
- 两个私有任务：`nagisa0v0/ped2023-native-in-tent-episodic-20261006`，`djnagisa/ped2023-native-in-sar-episodic-20261006`；各两个独立T4 worker，按病例分片。
- 本地机制测试和一例全体积smoke通过；源输出与既有源mask一致。SAR smoke每patch都恢复，不能把实际反向传播误写成未运行。
- Kaggle v1依赖构建将镜像NumPy2.1.3替换为1.26.4；任务随后成功进入适配。本机核心环境未改。`ENVIRONMENT_NOTE.md`记录了元数据约束不足及未提交的--no-deps备选启动器。必须使用各任务匹配source评价，不声称原始云环境不变。
- 2026-10-06 11:29北京时间仍在运行。最终状态与成绩以实验目录中的`REPORT.md`、`evaluation_verified.json`、`verification.json`为准；未生成时不能称已完成。

## 指标与核验

WT/TC/ET Dice和HD95均报告。HD95使用6邻域表面、双向距离合并95分位、单位mm；双方空为0，单方空为NA，报告有效n和完全漏检/空GT误报数，并给出双方有效的同病例配对变化。

下载后逐mask核验SHA、坐标、GT哈希与独立Dice/HD95（KD-tree对照上传EDT）。模型冻结/每病例重置在worker内断言；保留逐patch梯度、delta、恢复、跳过日志。源mask额外对照本机原始99例与另一账号。

报告实测提交到启动、环境准备、两GPU病例阶段、适应推理时间和总历时；双账号并行总时间不能把两方法历时相加。没有基于目标GT调参或改变SAR恢复规则。
