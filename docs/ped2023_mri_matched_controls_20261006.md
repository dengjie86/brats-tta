# MRI连续性的同修改量核验与新颖性

2026-10-06用户质疑“源NC大量被分为ED，随便扩张也会提高TC”，要求实测并核对研究前例。已完成99例全体积、45候选/例（4455组稀疏编辑）及独立核验，无新模型推断或训练。

产物：`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_MRI_matched_controls_20261006`，包含`PROTOCOL.md`、`controls.py`、`evaluate.py`、`check_methods.py`、`report.py`、`REPORT.md`、`LITERATURE.md`、`evaluation.json`、`verification.json`、`hd95_paired.json`、`diagnosis.json`、`per_case_metrics.csv`、图和各例编辑索引/评价回执。

## 对照设计

每例修改量K完全固定为上一轮MRI方法自动产生的ED→NC数量；只改变位置。78例K>0，其余21例回退仍进入主均值。候选含全ED随机20次、同种子无权图可达域随机20次、P(NC)排序、可达域P(NC)排序、距任意sourceNC的欧氏距离、从同一高置信NC种子出发的无权6邻域路径距离。无权路径不使用MRI/feature。可达域没有使用MRI阈值，否则会泄露选择答案。

所有99修改位置先保存锁定，再读取GT评价。只复用保存的P(NC)，不是NC-ED margin。随机Dice/编辑组成有超几何精确期望；20次实现用于HD95及核对随机均值。完全匹配K隔离的是位置选择，不检验MRI的K是否更优；不能说任意扩张量都有效。

## 99例结果

| 方法 | TC Dice% | HD95 mm（同92例） | 真ED存在57例ED Dice% |
|---|---:|---:|---:|
| source | 48.8627 | 14.7466 | 24.3881 |
| MRI | 65.5944 | 9.4948 | 16.1813 |
| 同K全ED随机 | 64.1401 | 9.7765 | 16.9042 |
| 同K可达ED随机 | 64.2161 | 9.6526 | 16.9042 |
| 源P(NC) | 65.1389 | 9.9506 | 19.2048 |
| 可达域P(NC) | 65.2096 | 9.7971 | 19.1935 |
| 距源NC最近 | 65.3519 | 10.2993 | 17.5464 |
| 同种子无权路径 | 65.1817 | 10.2871 | 16.7610 |

随机Dice/ED Dice为精确条件期望，HD95为20次均值。WT89.3334%、ET54.6713%与其HD95均硬固定，不是学习增益；TC全漏仍7例。单方空HD95=NA、Dice0；双方空HD950、Dice1。

随机同K达到MRI相对源平均Dice增量的91.31%，仅为条件对照的数值比值，不是因果分解。MRI比随机多约1.45pp，但比源概率、邻近/无权路径仅约0.24–0.46pp。

## 去重98组描述性区间

去除00137重复文件组，仍非独立测试。MRI减对照TC Dice：随机+1.4467pp[1.0190,1.9233]；P(NC)+0.4302pp[-0.0558,0.9832]；邻近+0.2303pp[-0.2276,0.6807]；同种子无权路径+0.4008pp[-0.0765,0.8636]。

HD95仍有额外信号：MRI减P(NC)−0.4765mm[-0.8927,−0.1263]；减邻近−0.7927mm[-1.2757,−0.3390]；减同种子路径−0.7828mm[-1.1985,−0.3909]。随机HD95的配对区间跨0。所有区间均为事后开发样本10000次配对bootstrap，不作独立测试显著性声明。

因此不能称MRI完全无效，也不能把16.73pp全部归于MRI组织边界：明显扩大TC是主要混杂，简单位置规则已非常接近；MRI有有限边界收益。MRI误吞340632真ED，源P(NC)误吞312157；MRI新增TC合并正确率64.8435%，略低于P(NC)64.9426%。有ED子集MRI TC66.5950%、邻近66.8525%，没有稳定的NC/ED语义分离证据。

## 新颖性核对

- 当前`min_path max_edge difference`经亲和度`exp(-difference)`变换，等价于fuzzy-connectedness的`max_path min_edge affinity`。2002年Moonis等已用于脑肿瘤MRI；2013年Huo等原文明确描述该算子。边权/种子不同，但核心路径原理不新。[2002](https://pmc.ncbi.nlm.nih.gov/articles/PMC7975291/)、[2013](https://aapm.onlinelibrary.wiley.com/doi/10.1118/1.4817475)
- DeepIGeoS已有测地距离与深度脑肿瘤分割结合，但需要人工交互，不能与无标签方案等同。[原论文](https://arxiv.org/abs/1707.00652)
- HD-TTA MICCAI2026已有影像梯度geodesic barrier约束扩张和强度一致性选择；官方回复也讨论class-conditional selector及modality-specific barrier多类扩展。后者是设想，不是已完成四分类验证。[方法](https://arxiv.org/html/2602.19454v1)、[会议及作者回复](https://papers.miccai.org/miccai-2026/0449-Paper3549.html)
- LAME CVPR2022证明输出适配可以不更新模型参数。此前笼统说“没有参数更新就不是TTA”不准确；应区分parameter-free TTA、参数学习和当前方法有效性。[LAME](https://openaccess.thecvf.com/content/CVPR2022/html/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.html)

结论：当前MRI规则可保留为候选生成或对照，尚不足以独立承担强方法创新。解除WT/ET保护、加适配器/蒸馏不能自动解决证据问题。没有本轮结果后调参或重跑。

## 核验

K/唯一性/源ED范围、可达域、随机与排序重放、坐标、原MRI完整mask重建、源和MRI所有Dice/HD95精确复现均通过。精确随机期望及方差以枚举小例独立验证；6种坐标转置与裁剪表面等价测试通过。图已目视检查。稀疏编辑可无损重建全部完整mask。

生成对照1.14分钟，四CPU进程评价1.56分钟；未使用GPU、未改checkpoint/旧预测或本机依赖。99例均开发证据、98文件组不保证独立患者、镜像官方等价未核实等限制保留。
