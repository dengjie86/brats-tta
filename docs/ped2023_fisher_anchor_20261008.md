# 源margin Fisher权重的native软锚点实验

2026-10-08。当前状态：固定24例预测、评价和核验均已完成，**两个12例组均未通过联合条件**。Fisher权重没有稳定解决WT损伤，TC仍与同预算geometry几乎相同；停止扩大本配置，没有自动扩到99例。数值与产物核验通过不代表方法有效。

根目录：`D:/Workspace/deeplearn/3dunet/outputs/ped2023_fisher_anchor_20261008`。上轮[冻结配置复验](ped2023_native_objective_extension_20261008.md)已完成且WT失败，本轮只改变锚点的组内权重。

## 动机和唯一变化

上轮额外12例的source-BG锚点中，|g0|<2仅占.2030%体素，却包含87.1903%的WT换类和仅.7998%的最终cost；|g0|>=4占98.3862%cost。这是池化loss能量，不能当作参数梯度的因果贡献。新的假设是：在同一native输出上，把软保留约束更多分配给源模型的决策边界。

令`g=max(FG logits)-BG`，g0为冻结源native gap，`q0=sigmoid(g0)`。q0是最高前景类对背景的Bernoulli代理，**不是四类概率的前景和，也不是已校准的真实WT后验**。取代理的logit Fisher曲率`w=q0*(1-q0)`，用`exp(-abs(g0))/(1+exp(-abs(g0)))²`稳定计算并停止梯度。

对E外冻结source-native BG/FG每组，将原`mean((3/16)*(g-g0)²)`替换成`sum(w*(3/16)*(g-g0)²)/sum(w)`，存在的两组仍等权平均。保留原teacher编辑目标、edit/anchor两个系数1、3/16尺度，没有额外指数、距离阈值、网格搜索或WT/ET贴回。若整组权重数值下溢为0，退回该组均匀权重并记录；当前24例源权重均无此情况。

沿用native IN四分类epoch250、原预处理/context teacher/编辑域/相对中心化log-odds修正，包含FG→BG监督。只更新22层IN的44个affine张量/7,680标量；每例重置源权重、Adam与scale。固定8次全native更新、Adam lr1e-3/defaults、AMP初始scale1024与原非有限梯度回退策略。四类输出均可改变。

native128³/.5滑窗、八镜像、Gaussian/FP16前向及FP32线性权重直通反向不变。replay.py与旧版字节相同，原extension的逐例执行函数直接复用；通过common模块注入新loss，并在启动时核对函数绑定。为兼容旧验收接口，文件内部仍用`fused`作为新候选key，报告中明确称Fisher；`plain`才是缓存的旧均匀锚点结果，不能混淆。

## 固定队列和比较条件

只使用已固定的original12与additional12，共24个不同四模态文件组，不重新挑病例。输入及分组在inputs.json/cohorts.json。全部24新预测固定后才开始本轮评价；这些病例均已有开发GT暴露，不是独立测试。

两个12例组分别判定，合并24只作描述。保留Source、直接teacher、同病例/同8步的plain及其geometry。对Fisher自身结果，独立重建同可修改域、逐源→目标数量匹配的margin/geometry/random控制；geometry借用候选预算，不是独立部署算法。

沿用每组原条件：NC/TC相对Source至少+.5pp、三域均值上升；WT/ET/ED-present损失不超过.2pp；净NC纠错；全漏/零重叠不差于Source和teacher；NC/TC高于margin与geometry；NC/TC/三域均值高于teacher。本轮权重主张另需WT高于plain，且TC/NC/ET/ED-present相对plain损失均不超过.2pp。**两个组都必须满足**，不能只保留通过的一组或用合并均值代替。

报告Dice、共同有效HD95、组织改对/破坏、WT编辑内外误删/误增、全漏/零重叠、病例风险及10,000次固定seed20261008的配对bootstrap。两侧空Dice1/HD950，单侧空Dice0/HD95NA。原组ED存在7、新增组9，两个组ET缺失均0，该场景仍未验证。小正差且区间跨零不能声称稳定优势，不自动扩大99例。

## 已完成的预检

- 解析权重与独立Bernoulli Hessian、稳定极端margin、独立加权cost及零质量回退检查通过。最初合成下溢fixture未同时降低其他前景logits，修正fixture后通过；未改loss实现，原失败日志保留在gradient_check_initial_test_failure.log。
- 小型网络、两个重叠patch、八镜像、FP32/AMP及参数扰动后，完整图和重算梯度最大相对误差约1.06e-7；self loss/gradient为0。
- 真实00003预检通过，23.39s，峰值GPU allocated2.610GiB。源完整概率和前后native logits逐值一致，32个raw源参照逐值一致；self Adam调用没有改变state，纠错首梯度与原均匀锚点逐值相同。预检无真实context参数更新、无GT读取，session19804已exit0。
- 全24例源权重清单通过，48个BG/FG组没有下溢回退。按病例平均，|g0|<2的BG位置占.1628%体素但获得61.0161%权重；FG相应5.1529%/65.6967%。这是权重分布，不是最终loss/梯度份额，也不是效果指标。
- 旧两组指标回归通过；把候选设成与plain完全相同时，配对WT改善条件正确返回False。该预检阶段未读取本轮候选GT指标。所有脚本AST检查通过。
- 全24例新输出均已逐值核对与plain相同的源logits、geometry及首梯度；这验证了初始条件一致，不意味着后续结果应相同。

## 完整结果

以下为Dice百分数。plain为旧均匀锚点，Fisher_geometry借用本轮候选的逐病例/逐源→目标修改数量，不是独立可部署方法。

### original12

| Method | WT | TC | ET | NC | ED-present (7) |
|---|---:|---:|---:|---:|---:|
| Source | 87.0109 | 57.9758 | 50.9932 | 30.1266 | 27.7336 |
| Teacher | 86.9463 | 60.2818 | 51.1056 | 33.0682 | 27.9593 |
| plain | 86.8295 | 63.0608 | 51.1585 | 35.9406 | 28.8029 |
| Fisher | 86.7479 | 63.0930 | 51.2788 | 35.9656 | 28.5268 |
| Fisher_geometry | 86.8944 | 63.1020 | 50.4378 | 35.5616 | 28.4346 |

WT相对Source−.2631pp，95%配对bootstrap区间[−.8148,+.3488]，超过预设.2pp损失限制。相对plain的WT−.0816pp，区间[−.5182,+.2413]；真实ED宏平均Dice也比plain低.2761pp，故两个配对保护条件失败。TC相对Source+5.1172pp、NC+5.8390pp，但相对自身geometry的TC−.0090pp，区间[−.3260,+.2461]，不能声称精细选点优势。几何差异的符号很小，主要问题不是这几个末位小数。

### additional12

| Method | WT | TC | ET | NC | ED-present (9) |
|---|---:|---:|---:|---:|---:|
| Source | 91.3024 | 60.8474 | 71.8222 | 38.1867 | 27.7839 |
| Teacher | 91.2373 | 61.5090 | 71.8616 | 38.9362 | 27.8482 |
| plain | 90.7300 | 63.4023 | 72.6066 | 40.4035 | 28.2776 |
| Fisher | 90.9650 | 63.4429 | 72.6455 | 40.3937 | 28.0819 |
| Fisher_geometry | 90.9529 | 63.4429 | 71.1260 | 40.0369 | 28.0044 |

WT相对plain+.2350pp，区间[−.1970,+.7960]，仍比Source低.3374pp，区间[−.7431,+.0046]；未达到WT保护条件。TC相对Source+2.5955pp、NC+2.2070pp；相对自身geometry的TC差为−.0000258pp，区间[−.1243,+.1323]，实质持平。相对geometry的NC+.3568pp、ET+1.5195pp有利，不能据此说完全没有组织信息，也不能将其描述为TC定位优势。

本组的四项plain配对条件均通过，但原工程条件仍失败；original12也失败，所以主结论为False。合并24 Source WT/TC/ET=89.1567/59.4116/61.4077%，Fisher=88.8564/63.2679/61.9622%，WT仍下降.3002pp。合并结果不能挽救任何单组。

## 病例损伤与作用边界

两组均有8/12例WT低于Source、2/12例下降超过1pp。原组最差00009为−2.2366pp；00063相对plain进一步下降2.0341pp。新增组最差00121为−1.7351pp，00170为−1.5799pp。

只读[病例敏感性检查](../outputs/ped2023_fisher_anchor_20261008/case_risk_audit.json)未重新推断或读取GT文件，也未改病例/条件。两组相对plain均为WT 7升5降；新增组均值+.2350pp、中位仅+.0279pp，去掉00099作描述性检查后其余11例为−.0012pp。原组去掉00063后其余11例为+.0959pp。它们说明病例构成敏感，不能据此剔除病例或重新挑成功队列。新增组ET相对Source均值+.8233pp，但7例下降，中位−.5576pp；00140单例+15.9627pp，去掉该例后其余均值−.5531pp。

软锚点确实同时减少了部分错误和正确改动。原组WT误删10,279→7,971、新增假WT4,190→3,157，但正确清除假WT15,648→11,080、补回真WT3,200→2,675也减少；WT总错误反而由plain的140,279升至142,031。新增组相应误删9,940→6,075、新增假WT25,871→18,270，WT总错误235,250→228,195，但仍高于Source215,661。这是保存掩膜的对比，不能直接当作参数梯度的因果归因。

Fisher的NC改对/破坏分别31,538/81与17,690/5,583；ED为27/810与508/1,156；ET为1,043/2,109与5,531/3,330。原NC→ED恢复27,374与14,855，新增0与14。ED的池化受损体素比plain少，不等于每病例宏平均ED Dice提高，两种统计都保留。

Fisher WT/TC/ET HD95：原组5.0350/13.9214/7.3049mm，有效12/12/10；新增组4.0070/12.7396/5.3286mm，有效均12。原组共同有效11例TC为Source11.3177→10.1100mm；新增组TC比Source改善.5620mm。不过TC HD95相对自身geometry分别差3.6579/3.1033mm。全漏/零重叠原组0/0/2与0/0/3，新增组均0/0/0；没有新增全漏。两个组都没有真实ET缺失病例，不能确认该场景安全。

## 执行核验与接续

主进程PID25448、session30446于12:16北京完成并exit0；24预测全部固定后才启动评价。24个适应后state全部精确CPU重建，144mask/24概率/24组affine及delta通过核验；192次真实更新、AMP回退0、权重下溢回退0、共1,376个source native raw视图。源checkpoint与冻结参数未改变。

预测4008.49s，评价104.81s，CPU状态审计8.90s，GPU峰值2.747GiB；预测耗时不含缓存teacher生成。监督session30992先完成audit_state/evaluate，随后因report.py把fallbacks写成fallback而exit1。已保留report_initial_failure.log与原failed状态，修正报告变量后只重跑report/review，恢复命令exit0，finish_status=complete；prediction_complete/state_audit/evaluation/verification的SHA均未改变。report_recovery.json记录修复前后hash。不是训练或评价重跑，也未隐藏失败日志。新增风险审计纳入最终review。

[完整报告](../outputs/ped2023_fisher_anchor_20261008/REPORT.md)和[配对差值图](../outputs/ped2023_fisher_anchor_20261008/comparison.png)已生成，图已目视检查。最终交接检查由final_review.py执行，工程核验与experiment_primary_pass分开记录。无本轮运行中任务，勿按旧PID重复启动。

当前证据不支持继续扩大此Fisher配置。下一步先检验输入尺度观察能否提供超出插值/roundtrip对照的真实纠错信息；[准备目录](../outputs/ped2023_input_scale_probe_20261008/README.md)只有36项合成CPU坐标检查通过，真实协议仍需冻结，尚无真实缩放推断、性能结果或新参数适应。这个观察已有文献前例，不是创新性结论。

未改核心项目代码、依赖、源checkpoint或已有实验产物，未启动云任务、commit或push。目标仍未完成。
