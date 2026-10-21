# GLI -> BraTS-PED 2023 的 TTA 设计与机制检验

更新时间：2026-10-06（北京时间）

最新补充实验见 [冻结分类头的IN适应与局部候选选择](gli_to_ped2023_frozen_head_candidates_v8.md)。
按新附件只补试末端IN、全部IN与局部候选选择，未运行额外文献基线。
同12例局部三类TC为58.0598/57.9832/57.4016%，源58.1153%；均未改善TC。
候选中存在较好修改，但当前无标签组织评分选择不准；未扩到99例全体积。

上一轮实际实验见 [三种固定教师TTA的真实参数更新试验](gli_to_ped2023_three_teacher_experiments_v7.md)。
已按新附件分别运行组织GMM、随机游走图和多裁剪IN/head适应；12例局部启动配置均未通过。
其中上下文学习与同gate冻结集成逐体素相同，不能归功于学习。全量和独立验证均未开展。

最新候选设计见 [病例内组织分群与语义拒绝](gli_to_ped2023_tta_semantic_support_v6.md)。
结合用户参考资料，新完成12例自动支持/同编辑量margin诊断与明确使用GT的离线表示读出。
自动支持版本仍未超过margin；离线读出提示表示存在信息，但不能作为TTA成绩。
新组织分群候选随后已在12例局部pilot中实际运行，未通过推进标准：TC仅+0.2932pp，
较同编辑量margin为−0.0038pp；未扩到99例全体积。见v6文档及
`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_group_TTA_20261006/REPORT.md`。离线监督读出分数仍不得当TTA性能。

最新实际先验实验见 [脑肿瘤先验辅助 TTA 的连续实验](gli_to_ped2023_tta_prior_experiments_v5.md)。
已运行真实模型更新与独立目录内的多轮检验。大幅 TC 增益尚不能证明先验独特贡献，
必须结合简单对照、真实 ED 代价与该文最终状态解读。
此前的 [多视图与集合伪监督文献方案](gli_to_ped2023_tta_literature_revision_v4.md)
仅完成文献、官方实现和梯度核查，不代表已实施患者实验。
此前的 [仅使用模型与当前病例的 TTA 方案](gli_to_ped2023_tta_revision_v3.md) 保留为失败实验记录。
v2 源特征参考库因增加 GLI 数据依赖已撤回。v3 的 12 例实际多头/小适配器 pilot 也未通过：
留出上下文 C 上适配器区域平均 Dice 仅 +0.000982 pp；最终头的 NC→ED 错误体素只有 1.55%
被至少一个辅助头正确分类。结果见 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_multiscale_probe_20261006/REPORT.md`。
这是匹配局部 cube 结果，不是全体积 benchmark；v3 当前实现不再作为主方案推荐。
下面保留已完成的失败检验及更早设计，不能将历史方案当作当前推荐。

## 当前结论：组织证据假设尚未通过检验

已完成独立目录 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_TTA_hypothesis_20261006` 中的 24 例机制检验。
完整规则、代码、逐例预测、哈希、评价和验证见该目录的 `PROTOCOL.md`、`REPORT.md`、
`summary.json`、`verification.json`。本节优先于后面的历史设计草稿。

固定 ID 哈希选取 4 例开发病例和 20 例本次检验留出病例。此前的 99 例分析已影响设计，
所以这不是完全未见的外部验证。所有 24 例无标签预测保存后才开始标签评价，未用结果调阈值。
每例独立拟合、无跨病例记忆，属于 TTA 的病例独立设定，不是 CTTA。

实际测试：用源预测区域内部的空间种子，拟合四模态强度和局部统计的组织分布，再按固定
似然比修改标签。这是检验学习信号的诊断掩膜调整，尚未训练真实 decoder 特征适配器。

| 20 例逐病例平均 Dice | WT | TC | ET |
|---|---:|---:|---:|
| 原 nnU-Net | 88.9926% | 55.1293% | 52.8614% |
| 只调 NC/ED，固定 WT/ET | 88.9926% | 55.7645% | 52.8614% |
| 调三种前景，固定 WT | 88.9926% | 56.5154% | 51.6370% |
| 允许 ROI 内四分类调整 | 86.6715% | 56.3903% | 51.7072% |

- 主对照 TC 变化 +0.6352 pp，95% 配对 bootstrap CI [-0.5838, +2.1491] pp；
  6 例提升、8 例下降、6 例不变。全部 24 例的 TC 变化只有 +0.0185 pp。
- 实际 ED 拟合种子逐病例平均纯度仅 18.75%；NC/ET 只能在 14/20、15/20 例建立模型。
- 主对照纠正 18,694 个体素，同时破坏 18,636 个原本正确的体素。不能据此声称监督方向可靠。
- 三种前景一起调整时 ET 均值下降；加入背景调整后 WT 也下降，必须同时关注三种区域。

预先写下的推进条件要求主对照 TC 区间下界大于零，当前未达到，因而没有继续真实适配器训练。
本次结果不支持“预测内部原型足以指导纠错”的强表述；它不否定所有适配器或更强的 MRI 表征。

当前讨论的最小适配器是最后一层 decoder 特征上的两层 `1x1x1` 映射，输出三个条件 logit
残差；第一层正常随机初始化，最后一层权重和 bias 为零。对源四分类 logits `z`，定义
`a=logsumexp(z_NC,z_ED,z_ET)-z_BG`、`b=logsumexp(z_NC,z_ET)-z_ED`、`c=z_ET-z_NC`。
令 `r=sigmoid(a+delta_a)`、`s=sigmoid(b+delta_b)`、`t=sigmoid(c+delta_c)`，输出
`[1-r, r*s*(1-t), r*(1-s), r*s*t]`，顺序为 BG/NC/ED/ET。
合成数据已核验零更新概率一致、可反向传播和重置，但这不是患者收益的证据。
软层级不能保证四分类 argmax 后的 WT/ET 掩膜不变；上述固定掩膜对照使用显式标签限制。

下面第 1--7 节保留为早期、未执行的设计记录：双速率适配器、模态删除一致性和候选路由
均未在本次实现或验证，不能据此声称已证实有效。尤其移除 T1ce 可能合理地改变 ET 预测，
不可直接把模态删除下的不变性当作正确监督。后续设计应先解决可靠证据与类别缺失，
而不是默认增加适配器就能解决问题。

---

本文提出一个面向当前项目的研究方案。方法暂命名为 **HIR-TTA**（Hierarchical
Interventional Reliability-guided Test-Time Adaptation）。名称描述的是
设计假设，不表示已经完成代码实现或证明了文献首创性。

## 1. 设计依据

当前可复用的源模型是 native nnU-Net v2 四分类 InstanceNorm、Dice+CE、epoch 250
checkpoint。PED2023 的本地 99 例评估使用原生 preprocessing、四分类 argmax 和八视图
镜像集成，得到 WT/TC/ET Dice **89.3334% / 48.8627% / 54.6713%**。已有逐例分析显示：

- 45/99 例 WT Dice >= 0.8 但 TC Dice < 0.5；
- 88/99 例 TC 预测体积低于标签 10% 以上；
- pooled confusion matrix 中，标签 NC -> 预测 ED 是主要内部混淆；
- 约 85.0% 的 pooled TC 漏分体素其实位于源模型的 WT 预测内（这是基于标签的诊断，
  不是方法结果）。

因此，主要问题不是简单的“找不到肿瘤”，而是 WT 已经提供了相当好的空间支架，源模型在
这个支架内没有可靠地区分 NC、ED 和 ET。全体素 entropy minimization、BN 统计量更新或
无条件 pseudo-label self-training 都可能把这种错误进一步强化。

BraTS-PED 2023 论文公布的是 228 例多机构儿童高级别胶质瘤，输入为 T1、T1ce、T2、
FLAIR，发布标签为 NC、ED、ET。当前 Kaggle 镜像的 99 例标签使用项目约定
`0=background, 1=NC, 2=ED, 3=ET`；镜像与官方文件的逐体素一致性仍需单独说明。

## 2. 与现有工作的边界

近期工作已经覆盖了许多单独部件：

- TEGDA 使用 dropout agreement、feature fusion 和 quality-weighted updating；
  TEGDA+ 增加 prototype pool 与 Fisher-guided restoration。
- HD-TTA 使用 gatekeeper、compact/diffuse hypotheses 和 representation selector，
  已在 GLI -> PED/MEN 的二分类设定中研究安全回退。
- SmaRT 使用 style modulation、EMA 双分支以及 integrity/connectivity 结构约束。
- VISTA 直接研究多序列 MRI 的跨模态 intervention 和 variance-gated pseudo-label，
  但其公开设定是 GLI-Pre -> PED2024，输入 resize 到固定体积，不能直接作为本项目
  PED2023 的数值结果。
- SPEGC、SicTTA、FAWA 等 CTTA 方法分别使用 graph clustering、source-friendly
  memory 或 feature-statistics memory；它们只作为背景边界，不属于本方案的主比较。

所以 HIR-TTA 的可主张贡献应限定为一个针对 **nnU-Net 四分类 GLI -> PED2023** 的组合
假设：把适应范围从“所有体素、所有类别”改为“由反事实模态可靠性控制的嵌套肿瘤区域和
内部类别”，并用候选输出路由器决定是否接受更新。它不是把已有 TENT、TEGDA 或 VISTA
换一个名字。

## 3. HIR-TTA 的核心流程

### 3.1 冻结的 nnU-Net 和两个适配器

令已训练的 nnU-Net 为 `f_theta0`。主干卷积、decoder、InstanceNorm 统计和原始 head
全部保留；在 decoder 的最后三个尺度后插入 zero-initialized、低秩的 3D `1x1x1`
residual adapter。只更新 adapter，不更新 `theta0`。

适配器分为两个更新尺度，但两者都只在当前病例内存在：

1. **fast style adapter**：只表示当前病例的 intensity/style 偏移；
2. **slow semantic adapter**：只作用于肿瘤内部的条件类别 logits。

病例结束后同时重置两个 adapter，并从同一个 `theta0` 开始下一个病例。这里的“slow”
只表示更小的病例内学习率和更强的 anchor，不表示向后续病例传递参数。

这使 InstanceNorm 源模型也能进行参数高效适应，不把“有 BN 才能更新 affine”误当成
   普适 TTA。新四分类 BN 模型可以作为额外的 norm-only 对照，但不能和 IN 的结果混为
   同一个主实验。

### 3.2 分层区域和安全支架

从 source/EMA teacher 得到四分类概率，并定义

\[
p_{WT}=p_1+p_2+p_3,\quad p_{TC}=p_1+p_3,\quad p_{ET}=p_3,
\]

以及 WT 内的条件分布

\[
q_c=\frac{p_c}{p_1+p_2+p_3+\epsilon},\quad c\in\{NC,ED,ET\}.
\]

每个病例产生三个区域：

- `safe core`：高置信 WT 且在多视图下稳定的体素；
- `recovery band`：safe core 的 3--5 mm 形态膨胀带；
- `outside`：其余脑区。

在 safe core 内，更新只针对 `q_NC/q_ED/q_ET` 的相对关系；在 recovery band 内，只有
   通过模态证据和跨视图稳定性检查的体素才能参与 WT 恢复；outside 的高置信背景作为
   anchor，禁止适配器把肿瘤大面积推向脑外或远处脑区。

这一步不是预测后处理：最终仍然由四分类 argmax 生成标签。支架只限制无标签更新的
   梯度来源，避免用形态学规则直接改写结果。

### 3.3 反事实模态可靠性

对每个 target volume 生成以下无标签 views：

- 原始 view `x0`；
- 八个空间镜像 view；
- 四个 leave-one-modality-out views，将被移除模态替换为该病例非零区域的中性统计；
- 一个低频 style probe 和一个轻微局部 intensity probe。

对 voxel `v` 和 class `c` 计算 teacher 在这些 views 上的 JS/KL disagreement：

\[
D_{v,c}=\operatorname{Var}_k[p^{(k)}_c(v)]
       +\alpha\operatorname{JS}(p^{(0)}_c(v),p^{(-m)}_c(v)).
\]

再与 source-only calibration 得到的 class margin 相乘，形成可靠性权重

\[
r_{v,c}=\operatorname{sigmoid}(m_{v,c}/\tau_m)\exp(-D_{v,c}/\tau_D).
\]

`r` 高的体素才产生 soft pseudo-target。关键点是 leave-one-modality-out 不是简单地
   判断“哪个模态最重要”，而是判断一个类别结论是否依赖脆弱的单模态证据；阈值和每类
   正常敏感度只从 GLI source split 校准，不读取 PED 标签。

### 3.4 分层无标签目标

建议的目标函数为：

\[
\mathcal L = \mathcal L_{region}^{r}
 + \lambda_{mv}\mathcal L_{intervention}
 + \lambda_{bg}\mathcal L_{background}
 + \lambda_{anc}\|\phi-\phi_0\|_2^2.
\]

- `L_region^r`：对 WT、TC、ET 做 region-balanced soft Dice+CE；在 WT 内再对
  `q_NC/q_ED/q_ET` 做可靠性加权的 conditional CE。它不让 background voxels 淹没
  NC/ED/ET 的学习信号。
- `L_intervention`：safe core 中，原始 view 与通过可靠性筛选的 intervention views
  保持分层概率一致；recovery band 只要求 WT 一致，不强迫不可靠的内部类别一致。
- `L_background`：outside 的高置信背景 anchor 和 safe core 的 WT anchor，抑制远处
  false positive 与无界 inflation。
- `L_anchor`：约束 fast/slow adapter 偏离 zero initialization，防止单病例更新吞掉
  源模型。

不把全体素 entropy minimization 作为主损失。它可以作为一个明确的 ablation，因为当前
   TC/ET 的错误不是低置信度这一种形式，盲目压低 entropy 可能只会产生更自信的 ED。

### 3.5 双速率 TTA 和候选输出路由

每个病例先得到 frozen-source candidate `P0`，再得到三个候选：

1. `P0`：不适应的源输出；
2. `Pstyle`：仅 fast style adapter；
3. `Phier`：fast style + safe-core conditional semantic adapter。

每个候选只用无标签信号打分：

\[
Q(P)=-U_{view}(P)-\eta F(P_0,P)-\zeta L_{outside}(P),
\]

其中 `U_view` 是 intervention disagreement，`F` 是候选与 source 的分层 mask
   fragmentation，`L_outside` 是 recovery band 外新增前景量。若 `Phier` 没有超过
   `P0` 的安全 margin，则输出 `P0`；`Pstyle` 和 `Phier` 的接受率单独记录。

候选路由只决定当前病例是否接受适应结果。无论接受哪个候选，病例结束后都丢弃两个
adapter 和临时 pseudo-target；下一个病例重新加载 `theta0`。因此主实验是 episodic
TTA，而不是依赖病例顺序的 CTTA，也不存在跨病例 feature bank 或长期状态。

## 4. 训练和评估协议

### 固定设置

- 源模型：`D:/Workspace/deeplearn/3dunet/artifacts/models/in_dicece` 的四分类 IN Dice+CE epoch 250；
  不重新训练 source model。
- target：当前已核验的 BraTS-PED 2023 99 例；每例独立适应，病例顺序不参与预测。
- preprocessing：严格复用 native nnU-Net manifest、spacing、normalization 和八视图
  推理；不额外添加 HD-BET。Source、所有 baseline 和 HIR-TTA 必须使用同一输入。
- adaptation：target labels 只在预测完成后计算指标；所有阈值只从 GLI source split
  或预先锁定的无标签工程配置得到。
- 运行模式：先用一个病例做 memory/VRAM smoke，再运行完整 99 例。每个病例的 adapter
  从零初始化；每个方法使用全新 output directory、独立 state 和可恢复 manifest。

### 对比方法

至少报告：

1. frozen nnU-Net + 八视图 source baseline；
2. entropy/adapter baseline（对应 IN 模型，不能假装是 BN-TENT）；
3. TENT、SAR、VPTTA、TEGDA compatibility port 中适用于当前 3D 协议的实现；
4. VISTA 的公开设定复现或明确标为 reimplementation，不能引用其 PED2024 分数；
5. HIR-TTA full method。

HD-TTA 可作为 WT safety baseline，但它原论文是 binary hypothesis refinement，四分类
   NC/ED/ET 结果应标成 adaptation/extension，而不是 exact reproduction。

### 必须做的消融

- 去掉 leave-one-modality-out，只保留普通 view variance；
- 去掉 hierarchical conditional loss，改成全体素 loss；
- 去掉 recovery band；
- 去掉 fast/slow 双速率，只保留单一 adapter；
- 去掉候选路由，强制接受每次更新；
- 仅更新 style adapter、仅更新 semantic adapter、两者同时更新；
- IN 主模型与 BN norm-only 控制模型分别运行，不能合并统计。

## 5. 报告指标和成功标准

除 WT/TC/ET Dice 外，固定报告 physical HD95、ASSD、precision、recall、volume bias、
lesion-wise Dice 和有效 HD95 样本数。TTA 特别报告：

- per-case `Delta Dice` 的中位数、IQR、paired bootstrap CI；
- harmful rate：任一关键区域 Dice 下降超过 0.02 的病例比例；
- helpful rate 和 harmful accepted area；
- 候选接受率、平均 adapter norm、每例更新步数和 early-stop 比例；
- all-background、远处 false-positive component 和 TC/ET 完全漏检数。

主成功标准应是：在 WT 不显著下降的前提下，提高 TC/ET Dice 和 recall，同时降低
   harmful rate 与 HD95。只报告三种区域的平均 Dice 不足以证明 TTA 安全。

## 6. 可证伪假设和风险

1. **H1：** source WT 内的可靠支架足以为大部分 TC/ET 恢复提供空间。若 recovery band
   外的漏分占主导，分层更新会受限，需要扩大 candidate，但不能事后用标签决定半径。
2. **H2：** leave-one-modality-out disagreement 能识别 PED 中不可靠的跨序列证据。若
   模态缺失 probe 反而误伤真实的单模态 ET 证据，应改成 source-calibrated class-
   conditional gate，而不是取消安全门。
3. **H3：** 候选路由可减少负迁移。若 source candidate 在多数病例已最好，HIR-TTA 的
   贡献应体现为低 harmful rate 和低 drift，而不是强行追求平均 Dice 增益。
4. 当前 99 例标签已经用于既有分析。正式方法选择前应锁定代码和阈值；任何用这些标签
   挑超参数的结果只能称为 exploratory，不能称为无标签 TTA。
5. VISTA、SmaRT、HD-TTA 和 2026 年新的 CTTA 预印本仍在快速更新，它们只用于背景比较。
   提交论文前应重新检索并做 component-level novelty check，不能仅凭本设计文档声明“首次”。

## 7. 参考入口

- [BraTS-PEDs 2023 challenge report](https://arxiv.org/html/2305.17033v4)
- [TEGDA official repository](https://github.com/HiLab-git/TEGDA)
- [TEGDA+ official repository](https://github.com/HiLab-git/TEGDAplus)
- [HD-TTA, arXiv:2602.19454](https://arxiv.org/html/2602.19454v1)
- [SmaRT, arXiv:2509.17925](https://arxiv.org/html/2509.17925v1)
- [VISTA, arXiv:2605.17433](https://arxiv.org/html/2605.17433v2)
- [SPEGC, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Du_SPEGC_Continual_Test-Time_Adaptation_via_Semantic-Prompt-Enhanced_Graph_Clustering_for_Medical_CVPR_2026_paper.html)
- [MedSeg-TTA benchmark, arXiv:2512.02497](https://arxiv.org/abs/2512.02497)
- [Single Image Test-Time Adaptation via Multi-View Co-Training](https://arxiv.org/abs/2506.23705)
- [Should This Case Be Adapted? Prediction Fragmentation Controls Test-Time Adaptation](https://arxiv.org/abs/2609.20700)

