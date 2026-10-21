# 2026-10-22 整理前历史交接

仅供追溯；当前状态以[根目录交接](../../HANDOFF.md)为准。旧 ETA、启动记录不构成运行状态。原文字节备份保存在本次维护目录；本阅读副本仅调整相对链接。

# 当前交接

诊断补充（2026-10-22，原实验仍暂停）：完成66例保存mask的分阶段诊断、同66例历史IN teacher比较及固定6例BN统计量干预，见[诊断报告](../../outputs/senior188_bn_diagnosis_20261022/REPORT.md)。66例Source/teacher/raw student/最终平均Dice为58.70143/58.66164/58.55671/58.72396%；61例有更新，记录损失全部下降（第1步前到第8步前，中位69.08%），无溢出重试。主要直接证据是teacher纠错方向不匹配：98.33%修改是背景↔肿瘤，ED→NC仅86体素，补回Source TC漏分的0.255%；历史IN teacher同66例ED→NC为175,219体素，TC均值49.7993%→57.2892%，但不是norm-only对照。6例单独切换当前patch BN统计、不学习，使平均56.60464%→60.16564%、TC34.05831%→48.59738%；仅2例提高、3下降、1持平，去掉最大增益病例BraTS-PED-00078-000后，其余5例平均−1.28171pp。不能据此宣称全面切换统计量已解决。6例短探测161.4秒完成退出，Source及基础teacher重放逐元素一致、参数未更新；原98例队列没有接续。新诊断GT仅用于事后分析，未调参。

最新暂停：2026-10-22 01:09:49用户明确要求暂停Senior188用户方法评测。已核对命令行身份并停止评测Python，监督进程也已退出，GPU计算列表无该PID；完成66/98例，66份病例result与汇总条目数一致。相同66例平均Dice：Source58.70143%、TENT62.67784%、用户方法58.72396%（仅+0.02253pp）。暂停发生在第67例00113的Source预测阶段，未完成病例不计入结果；所有已完成mask、affine、指标、原权重与未完成现场保留。详见[暂停回执](../../outputs/senior188_ped2023_ours_20261021/paused.json)。process_exit=-1是此次用户停止产生，不是全量完成；旧progress及ETA为暂停前记录，已失效。不得自动重启、巡检或接续；将来只有用户要求继续时，保留现有目录并隔离接续剩余病例。没有保存中断瞬间的内存optimizer状态。

以下为暂停前启动记录，不能作为当前仍在运行的依据。

最新：2026-10-21 22:50:47按用户要求在本机启动Senior best188的用户方法全98例评测，目录[senior188_ped2023_ours_20261021](../../outputs/senior188_ped2023_ours_20261021/README.md)。union2 + 单视角8步soft_extent（Adam .001/rho=.5）+ no_come；BN running stats冻结，仅5888 affine标量更新。为匹配同权重Source/TENT，保留Senior无镜像、128³/50%重叠、Gaussian概率融合和原始全网格预处理；属于Senior迁移版本，不能声称照搬native八镜像/logit融合。概率融合Jacobian回放与直接自动求导最大差9.31e-10，8组teacher/decoder独立对照通过。22:53:03首例00002完成、8步有效、Source mask完全一致、BN buffers及非affine未变、每例源状态恢复；132.86秒/例，GPU峰值allocated 2.595GiB，已接续00003。初估剩余3小时35分、10月22日02:28左右，非保证；后续以progress/complete/process_exit为准，不用旧PID推断存活。一次启动前模型代码换行符哈希问题已用原字节复制修复，失败现场保留startup_failures/001，没有改权重。普通脚本后台运行，首例确认后返回；未启动Goal/Scheduled或巡检。

前次完成：2026-10-21 21:43:05本机Senior best188的PED2023 Source/TENT完成98/98，exit0，总耗时16分40秒。Source WT/TC/ET=78.69835/32.93539/55.50581%，平均55.71318%；TENT=84.28050/45.95262/54.78840%，平均61.67384%，+5.96066pp，配对95%区间[3.37866,8.72485]pp；68例改善、28下降、2持平。1764次更新全部有效，源状态每例恢复；588项运行时Dice检查通过，本次完成/summary哈希、98例顺序及汇总均值复核通过。共同有效病例WT HD95+2.66421mm(n93)，TC−0.50821(n70)，ET−0.27247(n73)，不能宣称所有指标改善。按用户选定best188，原包预处理、128³/50%重叠/Gaussian概率融合、无镜像、BF16；TENT每patch一步Adam .001，仅5888 BN affine标量，每例重置。与QQ125并非仅换权重的严格对照。完整结果见[本次报告](../../outputs/senior188_ped2023_source_tent_20261021/REPORT.md)，复核见同目录completion_review.json。未启动新Goal/Scheduled或巡检。

更新：2026-10-21 18:28核实第二台Senior训练已于18:11:16完成238轮及最终同步，因组织均值连续50轮未刷新而正常早停。区域最佳epoch178=89.08984%，组织最佳epoch188=83.11838%；最终238区域88.44874%、组织82.51625%。用户要求下载数据集之外的全部工程，正式备份位于[Senior工程备份](../../artifacts/models/senior_unet128_source_20261020/README.md)，远程3378文件、逻辑5.55GiB、内容去重4.56GiB；包括远程本地盘checkpoint和日志，不遗漏v9恢复状态。已下载并核验远程全部 3,378 个文件（5.55 GiB 逻辑大小，去重后 4.56 GiB），三份主要权重 CPU 严格加载及 optimizer/scheduler/RNG 检查通过；详见 `backup_complete.json`。下载/检查由普通脚本接续，未新建Goal/Scheduled，40分钟guardian仍保持停止。此文件是唯一当前交接。

**新恢复路径必须保留**：远程`/root/autodl-tmp/senior_unet128_source_20261020/checkpoints_v9/`内latest/best/best_region是正式完整状态，不可当作缓存删除；网络副本可暂时落后。`control/supervise_v9.py`先同步本地新状态再加载网络断点，正常结束先排空复制。下载最新权重/关机前检查复制状态；页面显示待同步文件。来源、验证及新控制入口详见实验README。旧运行记录和指标保留，不以日志优化短测宣称全部GPU波动消除。

20:01:26补充：v8后台跨轮采样已完成，430秒/860点、最大间隔0.50029秒，GPU均值65.47%。训练28段0读数中25段在checkpoint快照日志后4秒内开始，代表段主线程处于`wait_on_page_bit_common`，指向尚未隔离的checkpoint相关I/O/页等待；具体阻塞文件/系统调用未确认。最长名义0段7秒在验证收尾。v8完整epoch14/15为5.923/6.520分钟，不能宣称全部0%消除或稳定加速。旧v7采样有网络读阻塞，不与其零值时长直接比较；显存快照约12.6/32GiB，逐batch显存预取尚未实施。详见实验README和`v8_short_drop_analysis.json`。

21:14页面/接口确认epoch28验证99/252，日志复制前缀字节一致、guardian和单次效果采样任务存活；新日志路径下的一分钟窗口仍有最长约6秒连续0读数，**尚未根治，不能把日志隔离测试当作生产提速证明**。后续优先读取`local_log_performance_confirmation.json`，勿重新启动已在后台等待/采样的任务。

训练代码已按用户要求交付至桌面`D:/PersonalData/Desktop/GLI_UNet3D_训练代码_20261019/`；[项目内同版及说明](../../artifacts/experiments/qq_unet_source_code_20261019/README.md)，[复制核验](../../outputs/qq_unet_source_20261019/continue_to_300/desktop_delivery.json)。

## 用户目标和偏好

- 研究GLI→PED的TTA，允许当前切入点无效时转向其他方向；10月16日完成双观察/单视角/4步全98组后，用户要求不再对全部镜像进行适应。当前保留原union2及8步，只取消学习镜像，在同98组验证。
- 中文简洁交流，保留负结果，不把实现核验等同于方法有效。
- 长实验不频繁轮询。非Goal模式启动确认后返回，由后台脚本接续，用户询问或出现实质变化再检查；不擅自创建Goal或调度。
- 10月21日用户撤销巡检需求：原40分钟`healthcheck_40min.py`已停止，保留stop标志，勿根据10月20日旧授权重启。训练正常会话续训与页面只读采集保留；未创建模型/Scheduled唤醒。详见实验README和停止回执。
- 不覆盖未提交工作、冻结实验、checkpoint或数据；不升级环境，不提交Git。

## 当前运行状态

| 工作 | 最新可核实状态 | 证据入口 |
|---|---|---|
| Senior best188：用户方法全98例 | **10月22日01:09用户要求暂停，评测与监督进程已退出**。保留66例；Source58.70143%、TENT62.67784%、用户方法58.72396%，仅+0.02253pp。第67例00113未完成；不自动恢复，旧ETA失效 | [暂停回执](../../outputs/senior188_ped2023_ours_20261021/paused.json)、[运行说明](../../outputs/senior188_ped2023_ours_20261021/README.md) |
| Senior best188：本地PED2023全98例Source/TENT | 10月21日21:43完成98例、exit0；平均55.71318%→61.67384%，+5.96066pp，588项运行时Dice检查通过，全部1764次更新有效。WT共同有效病例HD95变差2.66421mm；详见报告及复核 | [完整报告](../../outputs/senior188_ped2023_source_tent_20261021/REPORT.md) |
| 第二台AutoDL：Senior原包科学配置仅改128³ | 10月21日18:11:16完成238轮，patience50正常早停且最终复制stopped_synced。区域最佳178=89.08984%，组织最佳188=83.11838%。全部非数据工程下载校验及三份主要权重CPU加载检查通过，详见备份目录backup_complete.json。40分钟guardian保持停止 | [正式工程备份](../../artifacts/models/senior_unet128_source_20261020/README.md)、[协议与记录](../../outputs/senior_unet128_source_20261020/README.md) |
| 新用户模型125轮PED Source/TENT | 10月20日00:08:35完成98/98，exit0；588项Dice计数检查通过，完成回执/summary哈希及98条记录核对通过。Source平均47.80466%，TENT60.80041%，+12.99575pp，配对95%区间[9.25762,16.79995]pp。无新Goal/Scheduled | [完整结果](../../outputs/qq_unet125_ped2023_source_tent_20261019/REPORT.md) |
| 用户提供的五层BN U-Net Source | 10月20日00:32:13完成300轮、12次全253例验证；best275组织均值79.96953%，final300为79.04828%。01:23下载final300/best275及配置/代码/记录，29文件哈希一致、CPU严格加载通过。125轮独立备份保留；275/300尚未跑PED。远程训练进程退出，GPU无计算进程，可关机 | [最新正式备份](../../artifacts/models/qq_unet_source_20261019/completed_300/README.md)；[续训记录](../../outputs/qq_unet_source_20261019/continue_to_300/README.md) |
| 原方法取消学习镜像 | union2_v1_s8于10月16日23:54:32完成98组，294项独立Dice检查通过；代码、完成/预测/评价/桥接及98份病例回执哈希匹配，进程已退出。相对main卷积降算59.197%、平均Dice低0.08222pp、TC低0.28263pp | [结果](../../outputs/ped2023_efficiency_full98_20261016/union2_single_view8/REPORT.md)、[回执入口](../../outputs/ped2023_efficiency_full98_20261016/union2_single_view8/README.md) |
| 轻量方案完整98组 | 19:29:49完成98组（新增92、复用6），294项独立Dice检查通过；代码及完成回执哈希匹配，进程已退出。卷积降算85.150%，相对main平均Dice低0.84706pp、TC低2.66515pp | [全量结果](../../outputs/ped2023_efficiency_full98_20261016/REPORT.md)、[回执入口](../../outputs/ped2023_efficiency_full98_20261016/README.md) |
| 计算效率先导 | attempt03于18:35:13完成6例/36份候选；首例8步数值等价通过，108项独立Dice检查通过；先导进程已退出 | [先导结果](../../outputs/ped2023_efficiency_pilot_20261016/attempt03/REPORT.md)、[回执入口](../../outputs/ped2023_efficiency_pilot_20261016/attempt03/README.md) |
| 核心计算量核算 | 完成meta/逐层公式交叉核验；98例从头执行估计平均2.107 PFLOPs/例，仅卷积口径 | [核算报告](../../outputs/ped2023_compute_audit_20261016/REPORT.md) |
| 主方法选择与98组交付 | 已完成；此前Goal已完成，本轮未新建Goal | [方法说明](../../docs/GLI_PED_TTA_METHOD.md)、[交付重放](../../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/delivery/README.md) |
| BN225、BN250迁移 | 各24例预测完成，阶段Dice及720项独立计数检查完成 | [阶段报告](../../outputs/ped2023_cross_model_transfer_20261015/interim_readout/REPORT.md) |
| IN200迁移 | **14:55:24退出，exit_code=1073807364**；日志记载22/24例完成，随后00167在第5步后中断 | [阶段回执](../../outputs/ped2023_cross_model_transfer_20261015/in200_stage.json)、[日志](../../outputs/ped2023_cross_model_transfer_20261015/in200.log) |
| 跨模型完整评价/复核 | 未完成；未见queue_complete或evaluate_review阶段回执，整理时未见该队列Python进程 | [实验入口](../../outputs/ped2023_cross_model_transfer_20261015/README.md) |

中断原因尚未确认。10月15日整理时未恢复或重启队列；本次磁盘清理也未接续实验。不要使用旧PID判断存活，也不要重复原prepare/execute；接续前应读取原脚本、回执与未完成病例，保留22例产物及失败现场。此前关闭等待时进程仍存活，这不证明后续持续运行。

效率先导固定00003/00044/00061/00081/00120/00170：union2八视角2/4步、union2单视角2/4步、原双观察teacher单视角2/4步；最终仍8镜像/no_come。attempt03已完成；00003的8步首梯度/affine/raw/mask均逐元素一致，完成回执的评价哈希及等价性回执哈希已核对。原失败现场和attempt02的CPU入口导入冲突记录均保留，不重写原progress或冻结结果。

同6例平均WT/TC/ET Dice：Source 73.61968%，main 74.03836%；base2_v1_s4为73.88787%，比main低0.15049pp、比Source高0.26819pp。从头生成teacher计入后，卷积成本由平均2089.20降至311.14 TFLOPs/例，下降85.107%；不是端到端实测加速。该候选TC平均比main低0.71449pp，00170的TC低3.52143pp，无新增空区域预测。base2_v1_s2平均73.81668%、成本降87.131%。这些是IN250六例开发证据，不能据此替换98组主方法。

完整98组实验位于`outputs/ped2023_efficiency_full98_20261016/`，已完成。配置固定为base2_v1_s4，6例保留原产物并复用；92例只生成第4步终点。00003桥接通过，所有预测完成后评价；无溢出重试，294项独立Dice计数检查通过。代码与preparation哈希、预测/评价与complete哈希、桥接回执哈希匹配。未新增Goal或调度。

全98组轻量版WT/TC/ET为89.15792/55.53656/54.57131%，平均66.42193%；Source平均64.14833%，main平均67.26899%。相对Source平均+2.27360pp、TC+6.69193pp；相对main平均−0.84706pp、TC−2.66515pp（配对bootstrap95%区间[−3.78184,−1.72202]）。TC在74/98组下降、41组下降超过1pp；00085的TC从main52.41758%降至19.97825%（Source0.04450%），不能把先导0.15pp损失外推全量。无新增空区域预测；相对main的TC HD95在共同有效96组平均差+0.73524mm。

完整计入teacher的平均卷积成本由2107.087降至312.906 TFLOPs/例，减少85.150%；不是端到端实测加速。WT/ET相对main的小幅均值变化区间均跨零；BN及独立泛化未验证。轻量版暂不替代原主方法，减少步数/镜像和删补充teacher各自的全量贡献尚未分离。

取消学习镜像的实验在上述目录的`union2_single_view8/`，已完成。固定原union2、8步、lr=.001、soft_extent及no_come，只有原始方向参与学习；单方向Source reference保持自身零损失，teacher/decoder仍用原8镜像Source。00003第4步与已有union2_v1_s4桥接通过后继续第8步，全部98组新执行。缓存复用和零步分支保持，梯度视图数强制为1，冷启动计算量计入完整teacher，桥接额外预测单列。

union2_v1_s8的WT/TC/ET为89.15009/57.91907/54.49115%，平均67.18677%；相对Source平均+3.03844pp、TC+9.07445pp。相对main平均−0.08222pp、TC−0.28263pp（配对bootstrap95%区间[−0.54389,−0.04950]）。TC有9组下降超过1pp，最差00032下降9.48623pp；无新增空区域预测。完整计入teacher的平均卷积成本为859.754 TFLOPs/例，比main减少59.197%，不是端到端实测加速。支持继续采用单方向学习作为效率候选，但不能称无损，也尚未验证BN或独立泛化；不自动替换原冻结主结果。10月17日仅核对完成产物及更新导航，未启动新实验。

术语修正：“8个方向”应称8种轴翻转组合（原图、X/Y/Z、XY/XZ/YZ、XYZ）。nnU-Net内置的是无梯度镜像预测；把8种组合全部纳入学习梯度是原实验的额外设计，TENT等不要求如此。本地TENT/SAR/COME原方向适应，镜像仅融合输出；多增强teacher与对多增强反向应区分。

## 当前主方法及98组结果

固定路线：union2补充teacher → soft_extent学习（8步Adam，lr=.001，7,680个IN affine标量）→ no_come输出。保留范围池化、margin筛选和presence保护，已删除COME及冗余末端stable。详细算法、依赖与命令只维护在[方法文档](../../docs/GLI_PED_TTA_METHOD.md)。

| 98组开发结果 | WT | TC | ET |
|---|---:|---:|---:|
| Source Dice% | 89.27230 | 48.84463 | 54.32808 |
| 主方法 Dice% | 89.14027 | 58.20171 | 54.46499 |
| 相对Source，pp | −0.13202 | +9.35708 | +0.13691 |
| 配对HD95变化，mm | −0.57316 | −2.68657 | −0.16936 |
| HD95共同有效n | 98 | 91 | 84 |

对旧soft_stable：TC+0.96119pp，WT−0.20461pp。同输出的学习贡献TC+3.00925pp；新teacher进入学习额外+0.29916pp。最终控制及98份无GT/model/CUDA/COME输入的缓存重放已完成，勿重复运行。

## 同24组的跨模型结果

使用BN225、BN250、IN200三份额外权重，复用原IN250同24组；这是两个训练分支的checkpoint比较，不是三个独立架构。BN只更新affine，原eval running statistics冻结，其余方法不调参。IN200尚无完整成绩。

| 模型 | Source平均Dice% | 主方法平均Dice% | 提升，pp | TC提升，pp |
|---|---:|---:|---:|---:|
| IN250 | 69.99198 | 71.52986 | +1.53788 | +4.15069 |
| BN225 | 70.82468 | 71.58075 | +0.75607 | +2.37210 |
| BN250 | 69.71812 | 70.02627 | +0.30815 | +1.08397 |

平均Dice是WT/TC/ET等权平均，均为相同24组。BN结果是阶段Dice，尚无完整HD95与状态复核；BN250的TC描述性区间跨零。不能拿IN98组+9.35708pp与BN24组直接比较，也不能把跨训练分支的差异全归因于IN/BN。

## 固定协议与环境

- 10月20日Senior包新Source位于`outputs/senior_unet128_source_20261020/`，远程第二台`/root/autodl-fs/senior_unet128_source_20261020`。训练科学配置来自压缩包，仅CLI改patch128；运行时已作经核验的效率修改，不与10月19日native数据/损失版本混淆。原10小时会话到期由后台supervisor自动续同一断点，异常停止；19:31新增用户授权的40分钟巡检，对明确临时故障限次恢复。本地8767页面每60秒采集，完成后停止采集，失败后继续读取以显示恢复结果。原始数据视图复用5004份逐文件CRC一致的影像，另从原ZIP提取1251份四分类标签；不可清理其所依赖的第二台旧`Dataset502_BraTS_Binary/imagesTr`影像。未创建Goal/Codex Scheduled，不持续模型轮询。

- 10月19日Source实验独立位于`outputs/qq_unet_source_20261019/`，已在第一台完成300轮，不替换当前IN250 TTA Source。原外部`unet3d.py`原样复制，21,365,348参数、18BN、Dropout .3；数据沿用native GLI 998/253患者划分与预处理。损失Dice不含BG、CE含BG，1:1无类别权重；仅最终输出。原8765页面属于这个已完成的运行；第二台10月20日新实验用8767页面，不重复旧首次启动脚本。详细协议和完成备份见上表。

- 仓库 `D:/Workspace/deeplearn/3dunet`；已有Python `D:/Environment/Miniforge3/envs/cv311/python.exe`；RTX4060 Laptop约8GiB。完整环境说明在[AGENTS](../../AGENTS.md)。
- Source为native四分类nnU-Net IN Dice+CE epoch250；权重见[模型索引](../../docs/MODEL_INDEX.md)，资源见[paths.yaml](../../configs/paths.yaml)。
- PED2023 native预处理，无额外HD-BET；通道`t2f,t1n,t1c,t2w`；BG=0、NC=1、ED=2、ET=3；WT={1,2,3}、TC={1,3}、ET={3}。
- 99条记录含121/137重复，主结果按98唯一MRI/label组。固定24例属开发子集，GT均含ET。
- both-empty Dice=1/HD95=0；one-empty Dice=0/HD95=null；距离必须报告共同有效病例数。
- 2023/2024存在大量MRI重叠，不能按年份宣称独立测试，见[跨版本审计](../../artifacts/maintenance/source_moment_replay_20261013/target_cross_version_identity/REPORT.md)。

## 哪些已经确定，哪些仍待验证

主方法和开发结果已确定；WT/ET稳定优势、独立泛化、跨架构适用性和创新性未确认。IN分区、KL、loss选回、view可靠性等已完成负结果均保留，入口见[研究状态](../../docs/RESEARCH_STATUS.md)与[实验索引](../../docs/EXPERIMENT_INDEX.md)。

`scripts/research_dispatch.py`仍是未启用的CLI原型，不管理Goal开关；没有新建Scheduled。后续先解决跨模型任务的中断与完整复核，不根据未完整评价的IN200调参或自动扩98。

## 文件维护

当前入口保持简短；过程记录进入[历史归档](../../docs/archive/research_navigation_20261015/README.md)，新实验写`outputs/<experiment>/`。10月15日仅整理导航和历史时间线，未清理缓存；10月16日的实际缓存清理见下段。备份和核验见[维护记录](../../artifacts/maintenance/navigation_20261015/README.md)。

2026-10-16按用户要求实际清理973份缓存，释放21.17 GiB，项目约68.21 GiB。11个完成研究新增`CACHE_STATUS.md`，数组级历史review需先重建所删缓存；原始数据、模型、学习状态、mask及指标保留，跨Source中断状态未处理。详见[当前存储说明](../../docs/STORAGE_STATUS.md)与[清理核验](../../artifacts/maintenance/storage_cleanup_20261016/README.md)。
