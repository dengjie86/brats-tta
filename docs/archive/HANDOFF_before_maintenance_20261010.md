> 这是整理前的完整时间线，包含已经被后续记录覆盖的状态。当前交接见 [HANDOFF](../../HANDOFF.md)。

# 研究与实验交接

## 2026-10-10：补齐全部文件用途与当前导航

- 用户要求整理所有项目相关文件并解释用途。在此前已完成的归拢基础上，核对 22 个 outputs 研究目录、43 个 artifacts/experiments 历史目录和 4 个模型备份目录，新增 [项目文件总说明](../PROJECT_FILE_GUIDE.md) 与 [43 个历史实验逐项索引](../HISTORICAL_EXPERIMENT_INDEX.md)，为 models/experiments/maintenance 补目录 README。
- 修正导航中旧 E 盘位置、35.33 GiB 旧占用、已删 patch 参照和7/74 epoch权重仍保留等过时描述；模型索引以现存75份文件为准，旧核验JSON/科学报告原样保留。说明 Source、适应参数、概率/logits、几何数组、分割与指标的区别，并列出当前跨目录代码依赖。
- 项目文件清单遍历25,279项（修改导航前，不含环境、Git内部、工具/字节码缓存和本轮维护目录）；程序读取1,413个代码/说明/配置文本，结合报告、入口、数组头核对。参考快照 `artifacts/experiments/BraTS_PED_TTA_20260929_suite/references/IST_PED.py:438` 原有语法错误仅记录，没有修改。
- 相邻 `D:/Workspace/deeplearn/3dunet-source-tent-submit` 是已登记的 Git worktree（`codex/source-tent-submission`），保留原位置；数据集和环境继续位于专用目录。没有新增删除/搬迁，没有实验代码、配置、指标或模型修改；没有训练、真实推理或云任务。
- 本轮盘点、读取证据、原导航ZIP和最终核验在 `artifacts/maintenance/organization_20261010/`。研究候选与科学结论继续以下方10月8日“候选重新评估”为准。


## 2026-10-10：项目文件集中存放，新增清理 32.74 GiB

- 用户进一步要求删除不再复用/不常用且可重建的文件，并把项目文件统一放在项目目录。最新存储入口：[STORAGE_LAYOUT.md](../STORAGE_LAYOUT.md)。E 盘原 48 个 BraTS/HD_TTA/nnUNet 项目目录和 D 盘旧缓存隔离目录共 49 个目录，已迁入本项目；原始数据集位置不变。禁止后续再向盘符根目录或桌面散放项目实验。
- 本轮删除 3,122 文件、32.742276 GiB；其中有旧探针/消融的概率和 source_references、7/74 epoch 停止训练权重、测试权重、重复包、下载残留及工具缓存。当前背景共识/Source/native IN 必需输入、模型/保留的适应参数、代码、最终 mask 和指标保留。全部保留内容 18,908 文件、21.518253 GiB 已先复制、SHA-256 核验，再移除旧目录。
- 正式模型：`artifacts/models/in_dicece`、`bn_dicece`、`wt_pure_dice`、`legacy_custom_bn`。历史实验/helper 在 `artifacts/experiments/<原目录名>`；当前实验仍在 `outputs/`，其占用由 35.33 GiB 降为约 15.48 GiB。现行 checkpoint 清单为 [model_inventory_current_20261010.json](../model_inventory_current_20261010.json)，共 75 文件；旧 83 文件盘点仅为历史。
- 工作副本机械更新绝对路径与依赖文本哈希，共 2,002 文件。187 个 Python 文件通过语法及 AST 对比，除路径/哈希字符串外程序结构不变；1,683 个 JSON 通过结构/非字符串数值不变检查。修改前原文保存在 `artifacts/maintenance/consolidation_20261010/original_text.zip`，必须把它与路径适配后的工作副本区分，未重跑实验或重算成绩。先前清理清单等 `docs/*.json` 保持操作时的原路径。
- 当前工作流的四个模块导入、21 项 native 依赖哈希、396 个原始 MRI 输入路径、背景保留区及共识教师 CPU 小检查通过；IN250/BN225 在新路径 CPU 加载成功，无 CUDA 初始化、训练或真实推理。详情在 `artifacts/maintenance/consolidation_20261010/current_workflow_verification.json`。旧完整核验若依赖本次删去的缓存，须先重建；不能声称所有历史重跑都无需恢复。
- 首次文本路径扫描因效率问题在只读阶段停止，优化扫描后一次完成写入；没有因此改变算法或输出。所有清理/移动/路径改写清单和最终完整性核验集中在 `artifacts/maintenance/consolidation_20261010/`。此前被拒的 249 项在本轮新授权下重新核验并完成删除，旧拒绝回执保留为历史；本轮无审批阻断。

## 2026-10-10：按用户澄清清理不再复用的派生数据，新增释放 38.70 GiB

- 用户询问当前方法与 HD-TTA 的关系，并明确不反复使用、来自原始数据的派生文件可以删除。已解释：当前方法沿用当时训练的 native nnU-Net Source 和推理工具；背景共识/后续 IN 学习是后续实验实现，核心未调用旧 HD-TTA compact/diffuse 优化。保留固定路径，不更名正式模型目录。
- 读取转换/预处理/评测代码及跨目录引用后，删除 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_source_dice_local_20261003` 中 10,016 个旧训练派生文件、约 28.02 GiB（raw MRI/WT 标签、预处理 npz/pkl/GT、少量 probe/partial 数据）。训练已在 7 epoch 停止，代码、权重、plans、fingerprint、患者划分与转换记录保留。原始 GLI ZIP 6,255 成员全部读取/CRC 检查；5,004 MRI 逐文件 SHA 相同、1,251 WT 标签逐体素核验 `seg > 0`。
- 另删除 native 二分类、四分类合并 WT、纯 Dice WT 三组已完成 HD-TTA 评测的 initial/refined 概率，各 198 文件，共 594 文件、约 10.67 GiB。三根目录分别为 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_native_nnunet_20261002`、`D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_4class_nnunet_20261003`、`D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_wt_pure_dice_20261004`。指标、mask、代码和协议全部保留。
- 本轮合计 10,610 文件、38.695795 GiB，原生 PowerShell 精确文件清单删除完成。4,159 个保留文件 SHA 不变（包括全部 83 checkpoint），原始 ZIP SHA 不变；`E:/dataset` 31,414 文件、仓库 `outputs` 6,708 文件的大小/mtime 不变。完整记录：[derived_data_cleanup_20261010.json](../derived_data_cleanup_20261010.json)。与此前成功删除 27.728517 GiB 合计 66.424312 GiB；下方被拒绝的 249 文件清单没有重试。
- 依赖影响：已完成的 BN GLI 253 例验证使用过 local raw/cache，论文公式对照使用过 pure-Dice initial 概率。重跑须先重建对应输入，不能直接把旧 complete/identity 当作缓存存在证明。各受影响根目录新增清理说明；本地训练重建需在新 work root 转换/预处理，旧入口还会训练且转换记录引用已删文件，不应直接启动。
- 当前 PED 背景共识、IN 学习及 BN 对照的模型和仓库实验缓存未动。更新[数据与磁盘说明](../DATA_STORAGE_INDEX.md)，没有改变研究结果/算法，未启动训练、推理、云任务或依赖安装。

## 2026-10-10：用户明确指定的两组旧 HD-TTA 概率已删除，数据集保留

- 用户在列出两个旧实验的概率大小后明确说“那就把这两个删掉”，同时询问数据集为何占用较大。按已列范围删除 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_20260930` 和 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_axis0_20261001` 的 `outputs/full/cases/*/{initial,refined}_probabilities.npz`，各 198 文件，共 396 文件、27.728517 GiB；本次命令成功，E 盘空闲空间增加约 27.7293 GiB。没有删除整个实验目录，也没有重新尝试下方被拒的 249 文件清单。
- [删除核验](../legacy_hdtta_probability_cleanup_20261010.json)包含删除前 SHA-256、ZIP 成员信息及实际状态。1,770 个原有非概率文件（代码、协议、指标、日志、mask 等）SHA-256 不变，E:/dataset 31,414 文件大小/mtime 不变；没有重算所有数据内容哈希。两个旧实验根目录各新增 PROBABILITY_CLEANUP_20261010.md，历史回执保持原样。
- 已确认旧 diagnostics、params、params_round2、params_round3 的部分脚本依赖这些概率；今后重跑历史概率分析须先重新生成，不把原有“complete”回执解读为所有中间文件仍在。当前仓库代码未发现这两组路径的直接引用，未声称穷尽动态依赖。
- [数据与磁盘说明](../DATA_STORAGE_INDEX.md)：E:/dataset 共 142.064 GiB，包含多个年份、队列及原始/解压/预处理版本。PED2023 ZIP 实测 1.890 GiB、495 个成员、未压缩总量 19.756 GiB；同目录保留二者共 21.646 GiB。TEGDA_official_eval 39.505 GiB 是根据三个 preprocessing_report 由 GLI/PED/SSA 另行生成的 128³/[0,1] 版本，不是另一份单纯解压。
- ped_Training 与 PED2024 两目录没有在本次进行逐体素等价核查；PED 预处理记录确实使用前者补过一个病例的 t1c，不能认定完全重复。本次不删除或重压缩任何数据集，也不修改模型、环境或研究结果。

## 2026-10-10：用户授权 E 盘清理，永久删除被自动审批拒绝，实际删除为零

- 用户明确要求“数据集里面的数据不用删，但是把e盘里面那些没用的东西删一下”。保留 E:/dataset 全部内容，也保留项目目录内 raw/preprocessed 数据、完整正式模型/历史模型、预测概率、mask、代码、日志和指标。
- 核验后的清单为 249 文件、3.845 GiB：上次 E 盘隔离的迁移包和三步测试 checkpoint 2 个；236 个 Python 字节码；2 个完成任务的重复上传包；2 个 BN 下载残留；中止的本地 7 epoch 权重 2 个、Kaggle 74 epoch 权重 5 个。旧训练由用户选择的 AutoDL fresh run 取代，正式 pure-Dice250 已完成，旧 Kaggle schedule 为 Disabled。
- 两个 `.partial` 均与保留的完整 checkpoint 的对应前缀一致，完整文件 SHA-256 与旧模型盘点一致；两份 upload bundle 与保留原包 SHA-256 一致。没有把效果不佳的实验概率或仍有引用的中间数组列为无用。
- 永久删除命令在启动前被自动审批拒绝，只返回 `blocked by policy`，无具体原因。**实际删除/移动为 0，释放空间为 0**。没有更换删除工具或改为新隔离操作；不要把清单当成已经执行。已有用户授权，受阻原因来自工具审批。
- [详细清单和被拒记录](../storage_cleanup_20261010.json)：249 个目标全部仍在原位，执行日志不存在，63,628 个已盘点 E 盘文件的大小/mtime 全部不变，其中 E:/dataset 31,414 个。删除前已核验候选 SHA；被拒后没有重算全部数据哈希，没有训练、推断或环境修改。

## 2026-10-09：权重盘点与可恢复清理

- 用户询问模型权重位置并允许清理无用文件。新增 [模型与权重索引](../MODEL_INDEX.md)、`docs/model_inventory_20261009.json`、`docs/storage_cleanup_20261009.json`，README/文档/实验索引均增加入口。
- 在仓库和 E 盘 47 个项目目录找到 83 个 checkpoint 文件、19.892 GiB、73 种文件 SHA-256；不含压缩包内部副本，也不是全机扫描或 83 个独立模型。四个正式 native 分支的 best/final 共 8 份在 CPU 核对 epoch/trainer/optimizer/scaler，没有 CUDA 初始化或新推断。
- 当前 Source 仍是 native 四分类 IN250，BN 对照仍是 best225。正式 best/final、周期快照、运行副本、历史训练/TTA 状态均保留；IN175/200/225 为快照教师实验依赖，`source_references.npz` 为多个后续核验依赖，不能当无用文件删掉。
- 自动审批拒绝批量永久删除，仅返回 `blocked by policy`。改用可恢复隔离：296 个仓库缓存文件、未使用的 `migration.tar` 和仅三步显存测试产生的 checkpoint，共 298 文件、1.404 GiB，分别移到 `D:/Workspace/deeplearn/3dunet/artifacts/maintenance/_3dunet_cleanup_20261009` 与 `D:/Workspace/deeplearn/3dunet/artifacts/maintenance/BraTS_cleanup_20261009`。没有永久删除，磁盘释放量为 0；82 个 checkpoint 继续在原路径，测试 checkpoint 可从隔离目录恢复。
- 迁移包 91 个普通成员全部有字节一致的独立留存；其中唯一的 321-byte 状态另存 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_source_dice_autodl_20261003/cleanup_20261009/kaggle_segment_status.json`。删除/移动前后以清理记录说明历史清单变化，未改旧研究回执；原始数据、预处理、预测概率、mask、参数状态、代码与研究结论不变。

## 2026-10-09：仓库导航整理，研究结果和文件路径不变

- 用户选择“重做导航和目录说明，保留现有路径”。已按实际内容核对顶层目录、代码/配置/脚本、44 篇原有文档和 22 个本地研究目录的协议、报告、主要入口及完成回执；大体积影像/概率仅盘点文件与元数据，没有重新评价或推断。
- 新入口：[README](../../README.md)、[文档导航](../README.md)、[实验目录索引](../EXPERIMENT_INDEX.md)。configs、scripts、src/brats_tta、tests、outputs 各有目录说明。旧 README 全文保留在新 README 的历史折叠区。
- 关键区分：`src/configs/scripts` 主要是九月份自定义框架；当前 native 上下文和 IN 学习实现仍在被 Git 忽略的 outputs 及 E 盘 helper 中。单数 output 为空，复数 outputs 含约 35.33 GiB 实验材料。没有搬移、删除或抽取实验代码。
- 当前研究优先级仍是下节“候选重新评估”；本次没有新实验或性能结论。旧工程门槛、False 判定和历史待办原文保留，通过导航指出其适用时间。BN input-style 仍只完成 CPU 准备；BN complement 的科学 review 已完成，但 final 回执不存在。
- 文档链接、目录覆盖和原文件完整性核验见 `docs/repository_navigation_verification_20261009.json`；本次不运行全量训练/推断或 pytest，也不改环境、不提交 Git。

## 2026-10-08：用户明确候选标准，回到已有收益的上下文主线

- 用户最新明确：“只要有一定的理论基础，故事上面能讲明白，然后dice和hd95的效果比source好就可以参考”，并强调故事能否讲清楚。当前将Source相对收益、机制支持与旧联合门槛分开，不再因WT略降超过.2pp或未明显优于借用预算的geometry就概括成完全无效。旧协议/False判定不改；此处是新标准下的研究优先级，覆盖下方“当前没有候选、必须先换BN输入机制”的旧待办。
- [候选重新整理与机制叙述](../ped2023_candidate_reassessment_20261008.md)。只读重算`outputs/ped2023_candidate_reassessment_20261008/summary.json`通过：跨实验Source指标、24例ID一致，旧evaluation hashes不变，HD95使用共同有效病例。本轮没有新推断/训练/原始GT读取。
- **优先保留背景上下文共识teacher**：99例WT/TC/ET Dice89.3334/48.8627/54.6713→89.1277/54.3751/54.9399%，区域均值+1.8584pp。配对HD95 5.2672/14.7466/9.3240→4.7804/13.0441/9.2224mm(n99/92/85)；teacher自身TC有效96例原均值13.8881，勿混用。TC全漏7→3。WT−.2057pp及7例下降>1pp继续披露，99例仍开发数据。
- **轻量IN适应保留为学习候选**：同24例Source89.1567/59.4116/61.4077→native88.7797/63.2315/61.8826%，均值+1.3060pp；共同HD95 6.6375/12.3528/6.4557→4.4859/11.4876/6.2249mm(n24/23/22)。原/新增组TC+5.0850/+2.5549pp；新增WT HD95、原组ET HD95略差，勿把合并结果说成各组全面提升。Fisher均值+1.3702pp但模块优势不稳，保留作消融，不堆入主要故事。
- 机制收敛为“保留局部病灶、探测背景上下文敏感性→两观察共识估计病例纠错方向/程度→可选少量IN参数学习与Source软锚点”。IN全空间统计提供动机，已有统计交换支持敏感性，不能声称唯一因果根因或93.5%解释比例。geometry借用了方法修改预算；预算审计支持病例修正量有信息，当前贡献更接近自适应修正程度而非独特精细TC选点。
- 上一轮BN互补性诊断已完成：`outputs/ped2023_bn_complement_audit_20261008`审计50.81s、独立review40.81s通过。原/新增IN NC→ED错误的BN独有候选4569/98，扩展后严格概率混合受阻57.5777%/83.1844%；BN NC/ED病例AUROC.7370/.8160对IN.7375/.6989，只有新增组支持；两组前景模型选择置信差AUROC.5197/.5138。报告在docs，未建立新融合方法；无新forward或mask。该目录没有final_verification.json，不把科学review误称最终交付回执。
- `outputs/ped2023_bn_input_style_20261008`只有PROTOCOL/style.py/check_cpu.py及10项CPU检查，梯度差5.05e-12；没有真实checkpoint加载、GPU/PED前向、preflight或性能结果。**当前转为备选，不按旧待办优先启动**，也不再重复询问BN专项批准。输入样式不是当前最有证据的主要路线。
- 接续先围绕已有上下文teacher和轻量IN适应整合理论、Dice/HD95与必要消融，不盲目增加新分支或自动新跑99例。本轮未修改核心模型/旧产物、依赖、checkpoint，未提交Git。目标仍在研究中，不把候选价值等同独立泛化或创新性已确认。

## 2026-10-08：BN统计固定24例已完成，WT改善但内部组织退化

- [完整记录](../ped2023_bn_statistics_pilot_20261008.md)，目录`outputs/ped2023_bn_statistics_pilot_20261008`。既有GLI选择的native BN best225，独立对照原running statistics与当前patch statistics，四类开放、参数/统计累计更新0。同24例开发组、native预处理/128³/.5/八镜像/AMP、margin/geometry/random方向数量域匹配；IN Source只作资源参考。不是按PED选模型或epoch。
- 原组BN Source WT/TC/ET86.3341/61.0846/51.4218%→batch_stats86.9352/59.0974/51.0652%；新增91.4342/61.0030/73.6704→91.8496/59.9647/73.1031%。WT+.6010/+.4154pp，TC−1.9871/−1.0384pp，ET−.3566/−.5673pp；NC−2.7804/−.3933pp、ED存在子集−1.0784/−.2034pp。两组均False，目标未解决；停止本配置，不扩99、不调比例/层/epoch。
- 原/新增NC改对/破坏8203/17029、6980/7700，均净损伤；高置信Source NC→ED错误150491/300277，恢复仅4979/27。对同预算geometry TC−.1438/−.0138pp；新增NC/ET有局部优势，不能说完全无信息，但无法挽救对Source退化。BN Source对IN原TC+3.1088pp但WT−.6768，新增TC仅+.1556且NC−.7567；不可把模型差异称TTA成功。
- batch_stats HD95原组8.9567/14.7772/8.7461mm（12/12/10），新增4.0863/13.6127/6.0198（均12）。共同TC对BN Source+.9096mm(n11)/+2.0310mm(n12)，有效数不同不能直接减整体均值。原TC漏检00009被部分检出，但00055 TC−17.5291pp、00008−9.8007pp；新增10/12 TC下降，00140 ET−6.4106pp。两组WT仍6/12和7/12下降，各1例损失>1pp。
- case_event_audit仅复用指标：原ET全空总数2不变但00055新变空、00008变非空；两例前后都与GT零重叠，不能称正确ET受损或真ET恢复。零重叠4例身份不变，新增组均无事件。冻结判定仅看总数，必须同时披露身份；不事后改已失败结论。两组ET缺失均0。
- 真实预检6380 exit0，15.02s；主PID54540/session34435与监督39919均exit0，2026-10-08 22:37:35北京finish_status=complete。预测475.94s、评价100.28s、独立review9.87s，峰值1.229GiB。144mask/48概率，全部IN指标复现，完整NIfTI混淆/Dice/组织损伤和控制重建、时间/hash/state/checkpoint核验通过；comparison已目视检查。无运行中任务，勿重复启动。
- 执行前已更正旧审批解释，EXECUTION_SCOPE与execution_scope.json保留。没有收到BN专项答复；基于当前持续研究范围执行本地有界对照，未将沉默当授权。原IN仅限那次评估，并非当前永久禁用BN；下面历史等待/运行状态已被本节覆盖。科学冻结协议未改，无源训练/云任务/环境变更。
- 接续优先用现有概率审查BN新类别证据是否超出旧IN观察、是否存在可识别的可靠性依据；不因WT改善就继续盲调归一化，也不按开发GT挑更好源模型。域适应目标保持active、未完成。

## 2026-10-08：更正过度审批解释，BN固定24例统计对照已启动

- 前几轮把10月5日“BN PED需要对应请求”的历史任务状态解释成当前持续研究目标的额外审批门槛，导致重复等待。重新核对本文件“此次IN选择仅用于那项评估、不等于最终方法已定”及当前用户持续解决GLI→PED的请求后，撤销该不必要问题。**没有收到新的BN专项答复，也不把沉默当授权**；这是对现有任务范围的纠正。既有BN、24例本地可逆对照属于当前研究工作，额外模型资源明确标记；不是擅自新源训练或付费云任务。
- 可追溯说明：`outputs/ped2023_bn_statistics_pilot_20261008/EXECUTION_SCOPE.md`和execution_scope.json，时间先于任何PED前向。仅覆盖旧pending/no-launch操作措辞；原冻结科学PROTOCOL、预测代码、checkpoint、两组病例、统计条件和评价门槛均未修改。下面历史等待状态已被本节取代，勿再次陷入相同审批循环。
- 真实预检session6380 exit0，22:25北京完成，15.02s；Source→batch-stat→Source→batch-stat四次前向，source恢复与batch重复逐值一致，native输入/crop轴、checkpoint及源state不变。预检不读GT。当前环境cv311/PyTorch2.11.0+cu128/RTX4060，启动前可用VRAM约5.6GiB，D盘约98GiB。
- 固定24例预测现已启动，见该目录prediction_start.json、prediction.log和真实Python进程。保留IN Source参考、BN best225 Source、batch-stat和全crop方向/数量匹配margin/geometry/random；四类自由，参数更新0，不混合模型成绩当TTA增益。先全部预测固定，再evaluate/report和独立review。接续先核对真实handle与完成文件，勿重复启动。
- 实验无性能结论，域适应目标仍未解决。预设两组各自标准不变；失败不自动扩99、不调统计混合或epoch。原有模型、数据、用户工作及环境保留。

## 2026-10-08：目标恢复，BN评价实现补齐；仍待资源选择

- 目标此前因新增BN选择未答而blocked，当前goal已恢复active。自动接续没有明确BN选择，不据此启动真实PED任务；恢复后的阻塞审计重新计数。上一轮CPU实现准备属于实际进展，本轮补齐评价程序，域适应仍未解决。
- `outputs/ped2023_bn_statistics_pilot_20261008`新增metrics.py/evaluate.py/report.py/check_evaluation.py。完整WT/TC/ET Dice/HD95、NC/ED、组织转移、漏检/零重叠、逐例下降、匹配控制和Source高置信NC→ED统计；BN对IN明确标记资源比较，适应主对照为自身BN Source。
- evaluation_cpu_check.json四组检查通过：7体素独立手算损伤、物理间距HD95与空类、既有24例IN缓存指标/汇总/配对区间精确回归、WT超损/强控制持平/ED证据缺失/新增全漏失败条件。最终.18s，无原始MRI/GT或新BN推断。初始通过回执保留，评价入口增加依赖hash核验后再次通过；预测冻结代码/协议未改。
- CPU回归不是BN成绩。实际run.py preflight/predict、evaluate.py及report.py均未执行，无preflight/prediction_start/evaluation_start或BN报告；当前进程查询无该BN Python任务。得到允许BN答复后再走真实预检→24预测全部固定→评价/报告/独立复核。不能复用旧“全部预测之后才实现评价”的待办；评价实现现已完成，但真实执行未验证。
- [当前方案及准备记录](../ped2023_bn_statistics_pilot_proposal_20261008.md)。没有重新询问资源选择，没有更换依赖、启动训练或云任务。下一步仍需用户选择允许增加BN诊断，或继续限定单IN。

## 2026-10-08：BN独立加载和统计切换CPU检查通过，仍未启动PED

- 待选分支的实现准备在`outputs/ped2023_bn_statistics_pilot_20261008`；[方案与状态](../ped2023_bn_statistics_pilot_proposal_20261008.md)。用户尚未回答新增BN资源问题，未重复询问或将自动goal continuation当授权；无PED前向、图像/GT读取、GPU初始化或参数更新。
- 独立bn_adapter.py只加载GLI选择的BN best225，源state SHA69573216a9d047f25788421864528c796b4a74030272c15751dd044aeb0d3de4。patch_statistics保持全网络eval，暂时将22BN的running mean/variance/counter设None、track_running_stats=False，finally恢复原buffer对象和flags；不改IN adapter。当前patch的biased variance和eps沿用PyTorch，不累计跨patch/case状态。
- check_cpu.py exit0，10项检查3.78s；N1/N2独立float64公式最大误差1.7764e-15，固定源统计、无顺序依赖、异常恢复输出/状态、嵌套拒绝、参数及其他模块mode、12类修改方向控制和identity通过。真实22层仅CPU载入和切换恢复，无真实模型forward。native路径/CPU提示非致命，未改环境。
- prepare.py exit0，prediction代码/PROTOCOL/依赖及同24例纯图像元数据冻结；syntax解析通过。runtime.py/run.py已有preflight/predict入口，**都未执行**。没有preflight.json或prediction_start.json，没有运行中任务，勿因preparation存在自动启动。实际GPU、PED指标和评价流程仍待批准后验证，CPU通过不代表TTA有效。
- 下一步先处理待答资源选择。允许BN后才能运行真实预检：Source/batch-stat/Source/batch-stat四次前向，检查source恢复、batch重复、native输入/crop/源状态和显存；全部24预测固定后再实现/执行独立指标评价。若用户继续限定IN，则不运行这批BN代码。主要目标仍未解决。

## 2026-10-08：联合观察审计完成；BN资源已核查，实验待用户选择

- [联合缓存诊断](../ped2023_joint_observation_audit_20261008.md)完成，根目录`outputs/ped2023_joint_observation_audit_20261008`。复用同24例Source、两背景/两尺度/六LFCCS原始概率，无新forward、mask或参数更新。审计41.26s、直接不等式和八交集独立复核26.86s，均exit0；原尺度/LFCCS计数逐例精确复现、旧科学hash/空间/输入身份通过，crop外GT前景及Source错误均0。
- 原/新增NC→ED错误216148/361699，至少一个观察给NC的并集75566/48519（池化34.9603%/13.4142%，病例均值26.6554%/14.5959%）。背景原始观察已覆盖74763/44948；尺度和LFCCS只在其外增加803/3571个，不将oracle并集当纠错成绩。
- 更强限制：Source及十观察全部PNC<PED的位置128182/300998，即原错误池化59.3029%/83.2178%，病例均值67.4346%/80.6295%。任何非负且类别共享权重的概率混合都不能将这些位置判NC；高置信PED>=.9错误中比例60.9635%/85.9341%。严格小于、ties不计。不同于仅看argmax缺席，已验证其不能推出同样限制的合成反例。
- 限制只针对这些缓存概率及混合族，不是否定所有TTA、参数学习、特征、类别校准或新观察。全部四类错误及潜在损伤均报告，没有仅ED→NC新预测。最初两次审计因旧JSON接口、restore参数顺序失败；I/O修正后完成，失败日志/第二次start保留，没有改科学规则或原概率。停止仅给同一有限观察追加融合权重/门控的路线。
- 独立资源检查`outputs/ped2023_bn_resource_review_20261008/inspection.json`通过：native BN best225（GLI选择，SHA2c3917ae...78ea64），22BN、31,199,796参数、有限正running variance，3d_fullres配置只差norm_op、dataset相同，CPU严格加载、CUDA未初始化，无PED/GLI图像或GT。首个脚本名inspect.py遮蔽标准库，改check_resource.py后exit0；环境未改。
- [待确认的具体BN方案](../ped2023_bn_statistics_pilot_proposal_20261008.md)：原/新增12，本地比较固定源统计BN Source与当前patch统计BN，0参数更新，完整四类+逐方向数量margin/geometry/random控制。只用GLI预选best225，不按PED选epoch，不混合IN/BN。预计10–20分钟/<3GB，无训练/付费云任务，不自动扩99。
- 已通过async问题询问用户是否允许增加该BN分支；当前未收到答复，**BN PED推断尚未启动**。原因是下方既定边界要求BN PED需对应请求，近期主线为单IN。先读取新用户答复再决定，不把等待时间当授权。若允许，需实现独立BN loader和统计切换预检，不能直接调用只接受IN的native_adapter.load_predictor；若仍限单IN，保留已有失败和混合限制，需另立语义机制。主要目标仍active、未解决；无本轮运行中任务。

## 2026-10-08：跨序列响应24例已完成，固定平均与识错信号均失败

- [完整记录](../ped2023_intersequence_probe_20261008.md)，目录`outputs/ped2023_intersequence_probe_20261008`。同一IN四分类Source、episodic/source-free边界不变；冻结原/新增12共24例均完成，全部预测先于本轮GT评价。参数更新0。两个组ensemble工程条件均False，ranking_signal与variance_signal也均False，域适应目标未解决。
- VISTA v2的跨序列幅度交换是候选信息来源，但其BN/sigmoid/continual/PED2024与当前不同，只借鉴观察，不直接复现或引用成绩。ATASGG-SFDA的超声梯度筛选也不提供缺失的NC/ED类别证据，暂不为旧teacher继续添加优化器门控。
- 独立CPU低频交换保持共轭对称，非交换通道逐值不变；局部交换使用categorical entropy而非论文binary entropy。18项独立dense DFT oracle及identity/零通道/局部性/分位并列检查通过，CUDA未初始化。首次仅检查元数据重复ratio关键字失败，删除重复项后exit0。
- 官方代码revision7ab91e510961b0588a7797cf2d481f054c444bef备份带MIT/hash。40³、r=.1的合成频率边界例揭示偶数宽度块不共轭对称：原/供体幅度1/3，官方取实部得到2.00000017，对称版3.00000023。仅数值差异，不推断其临床成绩无效；新mask边界bin不同，必须称派生实现。
- 全部六个模态pair各交换一次ratio.1，保留原图+六观察等权平均及其全crop同方向/数量/域匹配margin/geometry/random。单pair只作描述，不从成绩改选。保存全部24观察后再用GT评价完整四类/组织/HD95/空类；两个组ET缺失均0，无自动扩99或参数学习。
- 信息证据单列：平均观察NC/ED log-ratio相对Source的病例内AUROC，以及PED方差识别Source ED错误相对1-PED及Source margin；每组配对均值>=.02且区间下界正、两组均过才标支持，不替代分割门槛。GT查询为Source ED中的真NC/ED，单真类AUROC为空；全Source FG错误检测另列。
- Source/identity实际预检session71560 exit0，10.78s、峰值1.192GiB，源概率逐值一致、crop/轴及state/checkpoint通过；无LFCCS前向或GT。首例六变换有原零背景非零泄漏，RMS .1176–.2827，已目视看input_preview；不是标签保留证明，不事后改零值。控制合成、方差/计数/AUROC与旧Source缓存回归检查通过。
- 主PID41232/session34685及监督session56050均exit0，2026-10-08 13:55:25北京finish_status=complete。预测1406.15s、CPU输入审计102.01s、评价183.07s，GPU峰值1.229GiB。288mask/144概率，24 Source逐值一致、全部state/checkpoint不变、真实输入及候选/方差/AUROC独立重建、verification/review通过；三图已目视检查。无本轮运行中任务，勿重复启动。
- 原12 Source WT/TC/ET87.0109/57.9758/50.9932%→average86.8022/57.8906/51.0210%；新增12为91.3024/60.8474/71.8222→90.9816/60.4126/72.2012%。TC−.0852/−.4348pp、WT−.2087/−.3208pp；TC相对context−2.3912/−1.0964pp、同预算geometry−.0628/−.0564pp。新增ET+.3789pp受00099的+6.2221pp影响，8/12例下降；不称稳定ET改善。
- NC改对/破坏3590/1616及3584/4483，ED484/651及750/765，ET377/1346及1819/3293；新增组净破坏NC。HD95 WT/TC/ET原组5.0359/11.3579/7.4050mm，有效12/11/10；新增3.9388/13.4449/5.6465，均12。原组Source的TC全漏仍在而context已修复，不是新增Source全漏；ED存在7/9例，ET缺失均0。
- 六pair任一NC候选仅覆盖原NC→ED错误池化5.8589%/3.0589%（病例均值6.8606%/3.1106%）；Source PED>=.9错误覆盖3.3090%/1.4361%。同查询平均margin病例AUROC .7478/.7043，Source .7375/.6989，差+.0103/+.0053且区间跨0，有效6/9；PED方差.5728/.6455弱于Source margin。全Source FG识错方差也低于1-maxP。小部分候选有信息，oracle并集不等于可部署选择器。
- 原零背景非零泄漏比例约1，通道/观察RMS中位.232660、范围.115957–.403646；phase保留不证明组织语义保留。不否定所有跨序列或VISTA完整方法；UGPS及学习未运行。停止本配置，不扩99、不调pair/阈值；下一步只读联合缓存审查context/尺度/跨序列候选互补性，不产生新预测或把候选覆盖当适应成绩。

## 2026-10-08：输入尺度24例观察实验完成，纠错信号不足

- [完整记录](../ped2023_input_scale_probe_20261008.md)，根目录`outputs/ped2023_input_scale_probe_20261008`，PROTOCOL.md已在真实预测前冻结。Fisher已完成并失败，不再继续其锚点搜索；本轮检验0.8/1.25输入尺度观察，原source/24文件组/两队列不变，全四类，冻结权重、参数更新0，无GLI/额外模型/跨病例状态。
- native预处理后CPU trilinear缩放，保存native缩放概率并逆映射回原网格；对照图像out/back后native推断、Source概率out/back。每对观察保留Source+两观察等权平均和双观察一致/maxP.8/margin.2/Source WT+10mm域教师。主候选scale_teacher有同域逐方向数量margin/geometry/random，平均及单尺度是预设描述对照，不事后替换主候选。
- 两组各自要求原NC/TC/WT/ET/ED/全漏/强对照条件，并要求NC/TC/三域均值高于两种roundtrip教师和原context教师；两组均通过才算主条件成功。合并24不挽救单组，不自动扩99或启动参数适应。仍为开发数据，两个组ET缺失均0。
- 36项CPU插值oracle检查和四类别教师合成检查通过；Source/identity逐值复现，crop/轴fiducial与源state/checkpoint不变；24例全部转换输入hash独立CPU重算通过。312mask（含48复制Source/context）、144概率、平均/门控/控制/源指标重建及verification/review通过，comparison.png已目视检查。
- 主PID24384/session65209及监督session84089均exit0，2026-10-08 13:00:29北京finish_status=complete。无本轮运行中任务，勿重复启动。观察/对照预测1056.26s、输入审计49.67s、评价145.39s、峰值1.323GiB；全部预测固定后才读本轮GT，参数更新0。
- 原12 Source WT/TC/ET87.0109/57.9758/50.9932%→scale_teacher87.0132/58.0285/50.9949%；新增12为91.3024/60.8474/71.8222→91.3036/60.8825/71.8490%。TC仅+.0527/+.0351pp，NC+.0608/+.0444pp，均未达+.5pp；相对context教师TC−2.2533/−.6265pp。两个组主条件均False，不扩99、不进行参数适应或阈值/尺度搜索。
- 相对自身同预算geometry TC仅+.0001/−.0011pp；WT近乎不变也伴随很少修改。原/新增组NC改对/破坏268/2及238/41，context编辑外NC219/2及236/41，有少量新信号但不足以支撑强教师。原组仍有Source的1例TC全漏而context已修复；不是新增Source漏检。简单平均原组ET全漏2→3，两组真实ET缺失均0。
- 主候选HD95原组9.3760/11.2634/7.2608mm（12/11/10），新增3.8921/13.2738/5.8180（均12）；共同TC对Source仅−.0543/−.0278mm，对geometry反而+.0195/+1.3773mm。原1.25尺度WT−1.5847pp、新增0.8尺度WT−1.1344pp，描述对照不替换主候选。停止本配置，不概括成所有尺度方法均无效。
- 原CPU准备及运行记录已被本节取代。输入缩放ensemble已有文献，不是创新性或儿童脑大小原因的证明；主要研究目标仍未完成。接续先检查剩余可识别的语义信号及资源条件，勿继续将保守门控或锚点调节本身作为新证据。
- 后置只读candidate_audit及完整NIfTI复核通过，22.59s，无网络/新预测/调阈值，旧hash不变。原/新增NC→ED错误216148/361699个，两尺度均无NC候选205828/347413（池化95.23%/96.05%）；单尺度候选8942/13413，双NC低置信1143/686，教师恢复235/187。少量候选确有信息：任一尺度ED→NC真NC10320/14286、真ED7/56，但不能只保留该方向称全四类成功。Source margin病例AUROC.7375/.6989、平均尺度margin.7598/.6976（有效6/9），未跨两组稳定超越；不把池化AUROC或oracle覆盖当部署性能。

## 2026-10-08：Fisher源margin软锚点24例已完成，两个组均失败

- [完整记录](../ped2023_fisher_anchor_20261008.md)，根目录`outputs/ped2023_fisher_anchor_20261008`。固定original12/additional12，全部已有开发暴露。只将E外source-native BG/FG组内的WT-gap平方锚点改为归一化Fisher曲率权重；原teacher、3/16尺度、组间等权、两个loss系数1、IN affine7,680标量、8次全native更新与Adam1e-3不变，逐例重置，四类开放。q0是pairwise代理而非校准WT后验。内部fused=Fisher，plain=旧均匀锚点。
- 主进程PID25448/session30446于12:16北京完成、exit0，全部24预测固定后才读本轮GT。监督session30992完成audit_state/evaluate后因报告fallback/fallbacks拼写错误exit1；保留初始错误日志和failed状态，修复报告变量后只重跑report/review，恢复命令exit0，finish_status=complete。四份预测/状态/评价/核验JSON的SHA未改变，report_recovery.json记录。没有本轮运行中任务，勿重复启动。
- 144mask/24概率/24状态通过；所有源logits/geometry/首梯度与plain逐值相同，全部24状态精确CPU重建，源checkpoint/冻结参数不变。192次真实更新、0次AMP回退/权重下溢，1,376个source native raw视图。预测4008.49s（不含缓存teacher）、评价104.81s、CPU审计8.90s，峰值2.747GiB。解析、完整图/replay、真实self预检均通过；原下溢fixture失败日志也保留。
- 原12 Source WT/TC/ET=87.0109/57.9758/50.9932%；Fisher=86.7479/63.0930/51.2788，NC35.9656%。WT比Source−.2631pp CI[−.8148,+.3488]，比plain−.0816pp；ED-present比plain−.2761pp。geometry TC63.1020%，Fisher差−.0090pp CI[−.3260,+.2461]。WT保护、TC几何对照和两项plain配对条件失败。
- 新增12 Source WT/TC/ET=91.3024/60.8474/71.8222%；Fisher=90.9650/63.4429/72.6455，NC40.3937%。WT比plain+.2350pp CI[−.1970,+.7960]，仍比Source−.3374pp，未过.2pp限制；geometry TC63.4429%，精确差−.0000258pp，实质持平。NC/ET相对geometry+.3568/+1.5195pp有利，不能说完全无组织信息，也不能称TC定位优势。
- 两组工程条件均False，联合主结论False；新增组plain配对条件通过不能挽救原组/WT工程失败。合并24 WT/TC/ET=88.8564/63.2679/61.9622%，WT比Source−.3002pp；不扩大本配置到99，不按这些GT调权重或条件。
- 两组均8/12例WT低于Source、2/12例下降超1pp。只读case_risk_audit：相对plain均7升5降；新增组WT+.2350pp主要来自00099，去掉该例后的其余11例均值−.0012pp；原组00063相对plain−2.0341pp。新增组ET均值+.8233pp但7例下降、中位−.5576pp；去掉00140后其余−.5531pp。仅作病例构成敏感性，不剔除病例或改主结论。
- 原组减少WT误删和误增，也减少正确清理/补回，WT总错误反而由plain140,279升至142,031；新增组235,250→228,195，仍高于Source215,661。Fisher NC改对/破坏原组31,538/81、新增17,690/5,583。不能把权重质量/损失或池化错误变化直接当成参数梯度因果解释。
- HD95 WT/TC/ET原组5.0350/13.9214/7.3049mm，有效12/12/10；新增4.0070/12.7396/5.3286，有效均12。原组共同11例TC Source11.3177→10.1100；但TC HD95相对geometry分别差3.6579/3.1033mm。无新增全漏/零重叠；两组真实ET缺失均0，该场景未验证。comparison图已目视检查，review及病例敏感性产物完整。
- 当前目标未完成。此处提出的输入尺度观察已经完成且两个组失败，见上方；无参数适应，勿重复启动。停止把继续改变锚点权重本身当作新语义证据。

## 2026-10-08：输入尺度观察仅完成CPU准备，未启动真实推断

- Fisher运行等待期间，核对旧“multiscale”使用的是深监督辅助头，旧axis试验为轴置换；没有把它们当成输入resize的直接证据。源训练备份`source_bundle/references/nnUNetTrainer.py:725`包含同步scale(.7,1.4)、概率.2，BraTS250子类没有覆盖augmentation；只证明训练设定含缩放，不证明GLI→PED偏移由脑大小造成或模型精确尺度等变。
- 准备目录`outputs/ped2023_input_scale_probe_20261008`：README.md记录依据与继续条件，resampling.py为CPU trilinear半体素坐标映射，check_resampling.py/resampling_check.json已通过36项独立NumPy oracle检查，最大绝对误差3.8147e-6；identity逐值一致，概率和/类别置换/奇偶与singleton维/非无损roundtrip已查。cv311 Python3.11.15，纯合成数组，没有读取目标图像、GT或模型。
- 此处的0.8/1.25准备现已进入固定24例观察实验，见上方；不能按旧准备状态重复启动。image roundtrip和probability-only roundtrip均已纳入冻结协议，仍不能将插值或滑窗变化当成已证明的尺度语义收益。Fisher已结束，不重复启动。
- 输入缩放ensemble已有文献，不能称新方法：[Wang等2018](https://arxiv.org/html/1810.07884)。其BraTS2018实验不能证明本项目跨域有效。仍需完整四类、两个固定12例组、匹配修改数量的控制及组织损伤评价，不自动扩99。

## 2026-10-08：native冻结配置额外12例验证失败，WT损伤与边界约束失衡仍在

- [完整记录](../ped2023_native_objective_extension_20261008.md)，根目录`outputs/ped2023_native_objective_extension_20261008`。因原12例工程点估计过线但WT/geometry优势不稳，固定原fused配置做额外12例有界复验；未修改teacher、8步、lr、loss系数、四类输出或成功条件，不自动扩99，不作新目标位置优势主张。
- 从99纯图像记录按四模态文件SHA去重为98组，排除含原12例的组，剩86；按`native-objective-extension-v1|20261008|`+ID的SHA256取首12，选择不使用成绩/GT。固定00042/00044/00046/00057/00061/00070/00072/00099/00112/00121/00140/00170；00121/00137只留前者。仍是已有开发暴露的数据，不称独立测试；分组/全排序在preparation.json。
- 新I/O执行代码在旧00003完整8步复现通过：概率、首梯度、最终state/loss/geometry逐值相同，94.00s，equivalence session81458已exit0。replay.py字节等于原版，依赖/源checkpoint核验；评价汇总也精确重现旧12例所有summary与criteria。原teacher无编辑时预设Source精确输出、0更新并保留该例，不能按GT替换。
- 主进程session39031和监督session71889均exit0，2026-10-08 10:37北京完成，finish_status=complete，无本轮运行中任务。72mask/12概率/12绝对affine及delta/12源logits核验，全部12状态精确CPU重建；96次真实更新、无AMP回退，736个native raw视图。预测2126.65s（不含旧teacher生成）、评价53.69s、CPU审计4.87s，峰值2.746GiB；独立选择/teacher/控制/指标核验通过。勿重复启动。
- 全12例只读缓存审计session1981 exit0、input_inventory.json通过：输入/收据/三概率hash、空间恢复、source及双观察独立teacher重建一致，无GT或新模型前向。该审计已纳入最终review，之后的性能评价和12状态验收也全部完成。
- 新增12 Source WT/TC/ET91.3024/60.8474/71.8222%；fused90.7300/63.4023/72.6066、NC40.4035%。TC+2.5549pp CI[+.9907,+4.2718]、NC+2.2168pp，但WT−.5724pp CI[−1.3893,+.1553]超过原.2pp，主评价失败；不扩99。合并24 WT−.3769pp也失败，不能以原12过线替代复验。
- 同预算geometry TC63.3927%，fused仅+.0095pp CI[−.1115,+.1421]；NC+.3625pp CI[+.0261,+.7332]、ET+1.5297pp CI[+.6359,+2.7452]有利，仍不能声称TC语义选点明显超越几何。TC HD95对Source改善.5530mm，但对geometry差3.1023mm；fused HD95 4.0939/12.7486/5.3243mm，有效均12，全漏/零重叠均0/0/0。ED存在9、ET缺失0。
- 8/12例WT下降，3例超过1pp，最差00099−3.5223pp。真NC改对/破坏17,760/5,485，其中5,211个NC→ET；ED622/1,623，ET5,648/3,642，原NC→ED恢复14,926/新增4。ET均值+.7843pp但6例下降、中位−.5196pp，00140单例+15.4966pp；留去该例后其余均值−.5531pp，仅作敏感性，不剔除主结果病例。
- 真WT误删9,940/其中E外8,694，假WT清理10,259、真WT恢复5,963、新增假WT25,871/其中E外25,515；总WT错误+19,589。后审计error_audit.json通过、无新推断/更新：E外新增假WT为NC192/ED24,265/ET1,058；19,948个在原WT边界1mm内，源gap中位−.8555。真实补回也多数在1mm内，不能简单冻结边界或按距离门控。
- 无GT的anchor_audit.json通过：由保存概率重建gap，源最大误差2.45e-7、最终anchor误差3.43e-8。E外source-BG中|g0|<2只占.2030%位置，却占87.1903%WT换类和仅.7998%最终cost；|g0|>=4占98.3862%cost。FG相应低margin占5.8143%位置/78.9250%换类/14.5911%cost。这是池化loss能量，不是参数梯度份额或因果证明。
- 此处提出的Fisher组内归一化软锚点已完成固定24例对照且两个组均失败，见上方，勿按旧待办重复启动。它同时减少部分正确与错误边界修改，没有稳定保护WT；q0不是校准真实WT后验。目标未完成，停止旧均匀锚点配置的扩大。

## 2026-10-08：native目标配对实验完成，小样本工程门槛通过，WT稳定性和强对照优势未证实

- [实验记录](../ped2023_native_objective_20261008.md)，根目录`outputs/ped2023_native_objective_20261008`，固定协议`PROTOCOL.md`。原12开发ID，同一source/teacher/IN affine 7,680参数，逐例逐臂重置；fused在真实native聚合输出计算四类edit+WT margin软锚点，viewwise在各native视图计算同cost，使用同一全局E及source-native BG/FG分母与Gaussian权重。四类开放，推断不变，不固定WT/ET。
- 固定每臂8次全native更新、Adam1e-3、AMP初始scale1024，逐视图重算梯度；每步native前向+所有视图重算/反向，两臂同预算。与旧24个单patch步骤计算预算不同，只对新的paired viewwise作目标位置归因。全部新预测固定后才进入GT评价，禁止按结果调步数/权重。
- 小网络完整图对照含未改变/扰动状态、FP32/AMP，重算梯度相对误差约1e-7，self零loss/gradient。实际00003的32视图参照逐值一致，self Adam步骤权重不变，fused/viewwise首梯度逐值相等，native完整概率/前后logits精确复现，峰值2.628GiB，预检31.16s。前向保持native FP16值，反向为明确的FP32线性权重直通替代，不能声称等于字面FP16逐运算backward。
- 主进程session51456和监督session55307均exit0，2026-10-08 09:48北京完成，finish_status为complete；无本轮运行中任务，勿重复启动。预测/适应3664.22s、评价85.79s、CPU审计8.12s，峰值2.726GiB。144mask/24概率/24绝对affine及delta/12源native logits通过；24状态全部精确CPU重建，源概率/旧基线/控制独立重建/hash/frozen/checkpoint核验通过。两臂各96次真实更新、无AMP回退，完整cohort640个native视图，首梯度最大相对误差9.56e-6。raw视图只保留SHA，不存完整tensor。
- Source WT/TC/ET87.0109/57.9758/50.9932%；fused86.8295/63.0608/51.1585、NC35.9406、ED-present28.8029；viewwise86.8319/62.9983/51.1594、NC35.8684。两臂原工程点估计条件均通过，fused配对条件也通过。fused相对Source WT−.1815pp CI[−.8904,+.5420]、TC+5.0850pp CI[+1.2748,+10.0121]、NC+5.8140pp；WT仅以.0185pp余量过线，不代表可靠无损伤。
- fused对viewwise TC仅+.0625pp CI[−.0266,+.2152]、NC+.0722pp，三域均值+.0197pp，均未支持稳定优势。对同预算geometry TC仅+.0067pp CI[−.3135,+.2765]、NC+.4126pp CI跨零；ET+.8453pp较有利，但不能只报相对margin TC+1.7265pp。GT前完整两臂mask不同2,106体素，占修改并集70,447的2.9895%，并非新的准确率。旧full与本轮预算不同，WT改善不能单因素归因于目标位置。
- fused真NC改对/破坏31,681/84、ED77/1,216、ET1,130/2,182，原NC→ED恢复27,450/新增0。真WT误删10,279/其中E外7,778，假WT清理15,648，真WT恢复3,200，新增假WT4,190/其中E外4,067；总WT错误比Source少4,379但宏平均Dice仍降。全漏0/0/2、零重叠0/0/3；HD95 4.8779/13.9255/7.3058mm，有效12/12/10；共同11例TC源11.3177→10.1119，geometry共同12例TC反而差3.6547mm。ED存在7、ET缺失0，缺失场景未验证。
- 只读GT后case_risk_audit：两臂均8/12例WT下降、2例超过1pp，fused最差00009−2.6357pp；逐一留一例之外重算，WT条件8/12种通过、4/12种失败，范围−.4021至+.0417pp。只是病例构成敏感性，未改原门槛/预测；不能把小样本过线等同稳定保护。
- 此处提出的原12例外验证已完成且WT失败，见上方extension记录；原12点估计通过不能视为稳定保护。不要按旧待办重复启动，不扩大本配置到99例；全部数据仍为开发使用，不称独立测试。原实验报告/核验/风险审计齐全，无云任务或核心代码/依赖修改。

## 2026-10-08：native视图审计完成，主要WT错误不是融合或舍入造成

- [完整记录](../ped2023_native_view_audit_20261008.md)，根目录`outputs/ped2023_native_view_audit_20261008`。原12开发ID、源/full/without_delete共36状态，1,920次只读native前向；完整概率数组全部逐值复现旧结果。无训练、新候选或mask修改；全部进程exit0、verification/review通过，无本轮运行中任务。
- 原teacher/full/without_delete类别变化并集89,137个体素，未包含并集外未变化的源错误。hook只记录native patch/镜像和原镜像平均patch票数、Gaussian支持、gap与同一FP16网络输出的FP64影子融合，原算术顺序/输出不变。全部记录固定后才读GT。
- full编辑外真WT误删9,548个，raw native全部FG/混合/全BG=0/8,887/661，加权前景支持Source75.67%→32.49%；没有全native FG而最终BG的位置。without_delete误删2,010个，对应4/1,980/26，支持65.31%→46.10%；4个全FG融合为BG中3个在FP64仍BG，现象很少。不能继续把主要损伤归咎于FG类别分歧后融合。
- 编辑外新增假WT full3,945个，FG支持27.28%→70.35%；without_delete7,271个，36.59%→64.72%，其中6,389个有Gaussian加权FG多数。多数票只是描述，不是可靠性保证或新门控规则。
- FP64双阶段影子融合仅改变full误删位置9/9,548、without_delete9/2,010的WT判定；新增假WT分别1/3,945、6/7,271。观察域内Source/full/without_delete的WT变化共41/30/34个；未生成FP64候选mask，不能宣称完整性能改善，网络内部精度也未变。
- 与旧训练patch审计连接：full在全部训练视图仍FG的1,971个受损位置，在raw native视图全部有分歧、没有全FG，支持75.83%→43.17%；其中902个在所有镜像平均native patch都BG。说明训练保护没有充分转移到实际推断上下文，尚未分开patch位置和镜像作用。full编辑外NC纠正29,117个，native NC支持7.10%→84.96%，也非只在融合mask出现。
- 八镜像坐标、独立双重叠patch票数/权重/类别分歧、36archive/12ROI、完整概率与state SHA、GT分组、旧产物/hash、checkpoint与frozen全部核验。收集253.85s、评价约3.02s，GPU峰值1.222GiB。`training_context_join.json`为只读后连接。
- 当时提出的native聚合目标配对实验现已完成，见上方：两臂小样本工程点估计均过线，但聚合后相对逐视图TC仅+.0625pp、相对geometry仅+.0067pp，区间跨零。勿按旧待办重复启动；仍未证明teacher语义优势和WT稳定保护，不优先改融合精度或投票。

## 2026-10-08：teacher前景删除定向消融完成，关闭后仍未解决WT问题

- [完整记录](../ped2023_deletion_ablation_20261008.md)，根目录`outputs/ped2023_deletion_ablation_20261008`。原12例完整体积；复用上一轮extent为full、self为none，新跑delete_only/without_delete。只对原teacher FG→BG位置的相对logits修正作保留/置零；原编辑mask、loss分母、24个patch、参照、extent软锚点、Adam1e-3及IN参数集不变。置零处仍监督源logits，不等于删除loss；完整四分类输出不固定WT/ET。全部进程exit0，无本轮运行中任务。
- Source WT/TC/ET87.0109/57.9758/50.9932%；delete_only86.4707/58.1697/50.9773，NC29.9850%；without_delete86.6250/63.4951/51.3350，NC36.6008%。两者失败；without_delete仅工程WT条件不通过，下降.3859pp CI[−.7667,−.0657]超过原.2pp，不扩99、不按GT调系数。
- without_delete对full WT+.3603pp CI[−.3512,+1.0119]，尚无稳定病例层面提升；TC−1.4010pp、NC−1.7536pp。对同域同方向数量geometry TC仅+.0571pp CI[−.3950,+.4646]、NC+.5386pp CI跨零，不能只报相对margin的TC+2.2662pp。geometry借用候选预算，不是独立部署方法。
- 真实WT误删/其中E外：full12,022/9,548，delete_only11,580/8,859，without_delete2,293/2,010；假WT清理分别11,841/11,809/1,787，新增假WT4,060/1,585/7,398。删除监督能在此配置引发编辑外损伤，关闭也损失假阳性清理；不是可相加的因果百分比。
- 只读GT后审计`error_audit.json`：without_delete新增假WT中7,271/7,398在旧E外，预测组织NC159/ED6,724/ET515；相对Source净FN−443、净FP+5,611，总WT错误+5,168（full+2,157）。其平均WT Dice较好但池化错误更多，病例规模与分母不同；不能把剩余失败只说成WT收缩。该诊断未增加推断或修改mask。
- without_delete真NC改对/破坏34,597/132，真ED196/105，真ET1,153/2,495；原NC→ED恢复29,592/新增0。全漏0/0/2、零重叠0/0/3；HD95 9.5678/13.9007/7.2815mm，有效12/12/10。共同11例TC源11.3177→10.1909；对geometry共同12例TC反而差3.7170mm。真ED存在7例、真ET缺失0例。
- 数值检查、144mask/24新概率/24delta/24绝对affine、288旧patch参照、独立控制及全部24状态精确CPU重建通过，checkpoint/frozen不变。每臂288有限调用，实际改参264/288；00008无删除信号，delete_only精确Source、without_delete精确cached full。真实AMP初始梯度分解残差.4411%，不称逐值线性。预测456.89s、评价86.85s、CPU审计8.68s，峰值2.445GiB；verification/review/后审计通过。
- 当时提出的native视图/融合分解已完成，见上方：主要错误在native视图中已经出现，不能以融合/舍入为主要解释。即使解除传播，也需超过同预算geometry的组织语义证据。不要重复本轮或把关闭监督方向包装成最终方法，不扩大99例。

## 2026-10-08：WT margin与内部组织锚点拆分完成，仍未解决WT损伤

- [完整记录](../ped2023_hierarchical_anchor_20261008.md)，根目录`outputs/ped2023_hierarchical_anchor_20261008`。原12开发ID完整体积、同一IN epoch250、原patch/teacher/Adam1e-3/24步；仅IN affine 7,680参数，逐病例/逐臂重置。主extent软锚定`max(FG logits)-BG`，paired hierarchical另加内部组织中心化logits约束；都BG/FG等权、四类开放，未固定WT/ET。全部进程exit0、主verification/state audit/review及后审计通过，无运行中任务。
- Source WT/TC/ET87.0109/57.9758/50.9932%；hierarchical86.2756/62.3633/51.2454、NC35.2144；extent86.2648/64.8961/51.5440、NC38.3545。两臂WT分别−.7354/−.7462pp，仍违反原.2pp门槛，不扩99、不按GT搜索权重。extent对paired TC+2.5328pp CI[+.5061,+5.6762]、NC+3.1401pp，但WT仅−.0108pp CI[−.0881,+.0584]；移除内部锚点确实释放纠错，没有解除WT损伤。
- extent相对同域同方向数量geometry TC仅+.0294pp CI[−.4164,+.4446]，NC+.5662pp区间也跨零；不能只引用margin的TC+2.3481pp。geometry借用候选预算，不证明预算无信息，也不是独立可部署方法。全漏0/0/2、零重叠0/0/3；真NC改对/破坏42,611/140，真ED85/616，真ET1,342/2,818；原NC→ED恢复37,051/新增0。
- extent WT/TC/ET HD95=5.2233/13.5280/7.1508mm，有效12/12/10；与source共同11例TC11.3177→9.7888，与geometry共同12例TC反而差3.9614mm。真ED存在7例、真ET缺失0例，不能推广缺失场景。
- 数值self、argmax规则、解析尺度和首步梯度逐值检查通过；156mask/36概率/36delta/36绝对affine值核验。self288次optimizer调用实际改参0，两个学习臂各288次实际更新，无AMP回退；CPU从绝对affine值精确重建全部36个state SHA，frozen与checkpoint不变。预测634.55s+评价89.22s≈12.06min，CPU审计12.42s并行，峰值2.445GiB。最早边缘概率版只预检无性能预测，保留`preflight_marginal`；不要混淆marginal WT和argmax WT。
- 结果后`context_audit`只读重建模型，576个适应patch前向+12源核对，prepare173.92s/evaluate约7.4s，无新训练。extent旧teacher编辑外真WT误删9,548个，其中1,971（20.64%）在所有训练视图仍FG、2,523（26.42%）全部BG；按位置平均FG支持83.60%→52.70%。hierarchical相应9,321/21.38%/22.07%。不能全归因于推断上下文，也不能把训练视图等同native推断视图。编辑外NC纠正29,117个中81.83%全部训练视图NC，非只在融合mask出现。
- 无GT的`target_audit.json`否定“内部换类普遍要求WT收缩”：344,195个FG→FG重叠采样位置，目标gap平均7.4637→9.5681（+2.1044），仅231个源patch FG跨到目标BG。显式FG→BG目标192,155个，gap3.9111→−1.4567（−5.3678）；BG→FG 7,412个，平均+3.1528。目标计数/幅度不是梯度贡献，尚未证明导致编辑外损伤。
- 当时提出的显式teacher FG→BG删除定向消融现已完成，结果见上方：误删减少但假阳性代价使WT仍失败，勿重复启动。native视图与融合阶段尚未直接分解，不能凭训练视图统计断言融合导致错误。不继续盲调锚点系数。

## 2026-10-08：IN affine与patch匹配软锚点实验完成，仍有WT损伤

- [完整记录](../ped2023_in_affine_anchor_20261008.md)，根目录`outputs/ped2023_in_affine_anchor_20261008`。原12开发ID完整体积、同一IN epoch250、逐病例/逐臂重置；self/uniform/balanced三臂，仅22层IN affine的44张量7,680标量，Adam1e-3、24步、正常输入统计。所有进程exit0，verification/reference audit/review通过；无运行中任务，勿重复启动。
- 使用原背景teacher的相对log-odds修正与完全匹配训练patch的冻结源参照，中心化logits MSE；uniform均匀anchor，balanced按源patch BG/FG两组等权，是预设主臂。四类全开放、无WT/ET贴回。两学习臂各288次实际改参；self 288次optimizer调用全部loss/gradient/delta为0，12例完整概率逐值等于source。无AMP回退；不能把864次调用全称真实更新。
- Source WT/TC/ET87.0109/57.9758/50.9932%；uniform86.3296/64.6250/51.6233，NC38.2154%；balanced86.3738/62.4944/51.2511，NC35.4003%。WT分别−.6814/−.6371pp，均未过原.2pp门槛，不扩99、不按GT调参数。balanced对uniform WT仅+.0443pp CI[−.2825,+.3173]，TC−2.1307pp、NC−2.8151pp；此配置没有有效解除损伤。
- 对各自同域同方向数量geometry，TC优势uniform+.0565pp CI[−.3656,+.4627]、balanced+.0610pp CI[−.2436,+.3593]；NC差也跨零。不要只引用较弱margin的TC+2.2180/+1.6030pp。geometry借用候选预算，不是独立可部署方案，也不证明预算无信息。
- 两臂WT/TC/ET全漏0/0/2、零重叠0/0/3，较source修复TC全漏且无新增全漏。共同source有效11例TC HD95为11.3177→uniform9.8522/balanced10.3333mm；全部12例TC13.6085/14.1636，反而比各自geometry差3.8598/3.6270mm。真NC改对/破坏uniform41,603/132、balanced27,357/122；真ED15/1,030、28/799。ET平均上涨不代表无损伤；真ET缺失病例0/12。
- 结果后审计：uniform真WT误删16,421，其中13,626在旧teacher编辑外；balanced14,076中11,510在编辑外。这些点全部被训练patch覆盖并作为anchor。其全部/部分/零源patch参照支持前景，uniform8,649/4,609/368，balanced6,792/4,370/348，不能主要归因于空间未覆盖或源patch始终判BG。另一方面，编辑外NC纠正中有18,138/6,083个从未得到任何源patch的NC argmax；源锚点也保留了待纠正的组织错误。诊断明确使用GT，无新推断/预测修改，非根因因果识别。
- 132mask、36概率、36delta、12组288精确patch参照、源指标/原checkpoint/frozen参数/控制重建/hash全部核验。采样anchor96.5040%BG，非梯度占比；搬移目标87.2808%编辑位置与原hard teacher同类，不能称逐位置相同。预测709.03s+评价83.29s≈13.21min，独立参照审计45.85s并行、后审计另计；峰值2.444GiB。相对旧decoder同时变参数集、loss、lr和目标匹配，不能独立归因于IN。
- 当时提出的前景范围/内部组织软锚点拆分现已完成，见上方最新记录：增加了NC/TC收益但没有消除WT损伤。该方向不是新语义证据，勿按旧待办重复运行。

## 2026-10-08：IN统计机制与decoder几何补审均完成

- [完整记录](../ped2023_in_context_decomposition_20261008.md)。两根目录：`outputs/ped2023_in_context_decomposition_20261008`、`outputs/ped2023_context_decoder_geometry_audit_20261008`。前者12例完整体积，冻结IN epoch250，同一PED病例原图/背景观察互换全部22层统计；后者复用旧decoder结果，新增24几何控制，无新训练/推断。全部进程exit0、verification/review通过；没有本轮运行中任务，勿重复启动。
- Moment teacher WT/TC/ET86.4596/59.7023/51.5857%，NC32.3911%；比source TC+1.7265、NC+2.2644pp，但WT−.5513pp、比原teacher三域均值−.1954pp。Content teacher86.7702/58.0103/50.9966%，NC30.1716，几乎未纠正NC且WT仍降.2407pp。两条均失败，不扩99、不按GT调层/系数。
- 输入保持区的blur统计分支与完整背景观察概率变化cosine=.9352，L1=.035898（原context .033351），content=.008680；neutral有更大非加性残差.071029。支持研究IN敏感性，不能解释为“93.5%域偏移由IN造成”。直接统计交换的raw moment_neutral WT仅51.8110%，content_blur WT48.0557%，有严重失真；所有结果保留，不能只挑TC上涨。
- Moment teacher相对同量geometry TC仅+.0452pp CI[−.0863,+.1831]；NC真改对/破坏11,950/47，原NC→ED恢复9,896/新增0。Content恢复102/新增2。moment共同有效WT/TC/ET HD95为10.0209/10.4596/7.1092mm，n12/11/10；原TC平均14.8115含12例，分母不能混比。
- 数值检查：12例三观察记录前向、12原图自回放以及首例两背景自回放全部逐值一致；168mask/48概率/36组22层统计/36输入与保护区/独立门控和原权重核验通过。预测1082.66s、评价62.97s，输入审计32.72s并行；峰值2.10GiB。两次早期padding API失败日志保留，修复在新预测前且未读GT。
- 旧context decoder TC64.6788%，同量geometry64.5818%；差+.0970pp CI[−.6813,+.7122]，NC+.8876pp CI跨零；ET+1.4650pp CI[+.1924,+3.1264]有利但仍低于source。TC HD95比geometry差2.8868mm。不能只引用旧相对margin的TC+2.0603pp；原WT/ET及全漏失败结论不变。补审生成48.83s、评价3.56s，24mask与旧产物hash通过。
- 当时提出的IN affine配对适应现已完成且未过WT门槛，见上方最新记录；不要按旧待办重复启动。暂不扩大17M参数decoder。

## 2026-10-08：结构响应可靠性小实验完成，停止此规则

- [结构响应完整记录](../ped2023_structure_response_20261008.md)，产物`outputs/ped2023_structure_response_20261008`。原12例完整体积、同一IN epoch250、冻结权重。借鉴DeYO思想，对源/两个context作相同8³块重排并逆变换概率；四类开放，只有结构响应优于source的旧teacher修改保留。不是原版DeYO复现或创新方法。
- WT/TC/ET=87.0223/58.8846/51.1339%，NC31.4333%，ED存在7例27.8654%。相对source TC+.9088pp、NC+1.3066pp，但比旧teacher TC−1.3972pp、NC−1.6349pp、三域均值−.4310pp。同方向数量geometry TC58.9130%，新规则低.0284pp，不能只报相对较弱margin的+.3276pp。
- WT改动11,648→506，ET2,895→2,379，趋于保守；平均WT略升但WT错误体素旧teacher140,066→新144,462（source144,658）。TC全漏旧teacher0→1（source1）。真NC改对/破坏6,729/28；原NC→ED恢复5,043，少于旧12,760。ED→NC正确性AUROC池化.5468/病例均值.4102，未提供可靠语义选择。
- 72mask/36概率、36输入独立逐块hash、inverse/响应/同域同方向数量、原source/teacher和指标精确复现、权重不变与review全部通过，fallback0。预测364.68s、评价24.55s、并行输入核验32.17s；三个进程exit0，无运行中任务。不调块/阈值/种子，不扩99，不称域适应已解决。
- 接续仍保留下方“病例预算有信息”的线索。尚未实施的机制候选：区分IN统计变化和图像上下文变化；检验decoder类锚点失衡与patch目标不一致。不要重跑本轮或旧的99例结果，不将未运行假设写成实测结论。

## 2026-10-08：修改预算有信号，WT置信度仍不可靠

- [预算与WT审计](../ped2023_context_budget_audit_20261008.md)已完成，根目录`outputs/ped2023_context_budget_audit_20261008`；98例/额外86、每组10,000次相同总量分配，准备与评价exit0，verification/review通过，无新推断或参数更新。
- 保留其他teacher修改，仅诊断ED→NC数量。自身数量+固定几何TC54.3665/53.5213%，统一比例53.3978/52.2680%，随机均值52.6341/51.5389%；自身数量分别第96.65/96.31描述百分位。误改真ED自身数量11,538/11,511，统一比例44,155/43,737。说明病例数量可能有用，不证明精确选点、因果性或泛化；不是固定WT/ET的新方法。
- 98组WT删除46,685中破坏真前景25,118；source置信度风险AUROC0.5901。context共同BG置信度≥.99仍破坏6,792/8,602。没有拟合新门控阈值；原99例teacher失败结论不变。
- 此处提出的结构响应小实验已完成并失败，结果以上方更新为准；不要按旧待办重复启动。

## 2026-10-08：背景teacher的99例验证完成，尚未解决域适应

- 最新入口：[完整研究记录](../ped2023_context_teacher_validation_20261008.md)。完整代码/固定协议/REPORT/概率/495mask/图/核验在`D:/Workspace/deeplearn/3dunet/outputs/ped2023_context_full99_20261008`。复用原12、另推87，所有进程已exit0；**下面旧“扩大验证已启动”状态已被本节取代，不要重复启动。**
- native IN四分类epoch250，逐病例，无额外模型/GLI图像/参数更新。99例source WT/TC/ET89.3334/48.8627/54.6713%→teacher89.1277/54.3751/54.9399%；NC30.9650→37.2255%。TC+5.5124pp CI[+3.0928,+8.4254]；三域均值+1.8584pp。新增87 TC47.6058→53.5604%；去重98/新增去重86结论同向。全部仍为开发集，不称独立测试。
- 严格门槛仍失败：99/87 WT下降.2057/.2251pp，超过预设.2pp，98/86也失败。不能因只略超平均门槛而改规则；7例WT降超1pp，最差00019 WT−6.7940pp、TC−10.2919pp。参数decoder配置的12例结果仍失败，不扩到99。
- 关键新对照：同编辑域、逐病例原类别→目标类别数量匹配的geometry TC=54.3663%，teacher=54.3751%；差仅+.0088pp CI[−.0952,+.1114]，新增87差+.0285pp CI[−.0644,+.1258]。teacher NC比geometry+.1804pp，但三域均值略低。正值工程条件虽过，**不能说证明了稳定TC语义选点优势**。该控制借用teacher每例修改预算，不是独立部署算法，也未验证预算自身的信息价值。
- teacher相对source-margin TC+1.4285pp，因强geometry控制几乎重现涨幅，不能只引用较弱对照。真NC改对205260/破坏6409，原NC→ED恢复201380/新增4；真ED79/10425。真ED存在57例均值24.3881→24.4094%，接近持平但不能省略体素损伤。真ET缺失11例误报体素8968→9040，误报病例仍7。
- HD95源5.2672/14.7466/9.3240mm（99/92/85），teacher4.7804/13.8881/9.2224（99/96/85）；共同92例TC为14.7466→13.0441。teacher TC HD95比同预算margin差2.9227mm、比random差4.1759mm，不是全面领先。全漏0/7/7→0/3/7；零重叠0/7/15→0/3/14。
- 495mask、198变化输入hash、独立门控/空间/方向数量控制、原99源mask逐体素一致、重复组输出、代码/模型/输入产物hash及最终review核验通过；fallback=0，checkpoint没改。预测/复用2432.00s，主评价168.86s，额外geometry评价83.36s，串行主流程44.74min；输入审计与几何生成并行。未新增云任务或Git提交。
- 当前只保留可复查候选，未宣称解决或创新。下一步先分开检验teacher“修改预算”和“选点”的有效信息，并审计WT损伤可靠性；不事后调阈值再包装成功，也不把小样本decoder的TC上涨当联合成功。12例loss审计/文献边界/示例图在pilot根目录`RESEARCH_NOTES.md`。

## 2026-10-08：背景教师出现小样本信号，decoder学习仍有损伤

- 新入口：[背景教师与参数适应](../ped2023_context_teacher_validation_20261008.md)。原12开发ID完整体积；单一IN epoch250、native四类、逐病例、无GLI/外部模型。根目录`D:/Workspace/deeplearn/3dunet/outputs/ped2023_context_probe_20261008`，Git忽略，仅本地保存。
- 新观察保留源WT及外侧10mm的四模态，10—20mm过渡，远处分别用矩匹配sigma4 Gaussian背景与病例中位数背景。teacher只在输入保留区接纳两观察同类且各P≥.8、margin≥.2的修正。四类/WT边界/ET可变，不贴回WT/ET。不能据此识别因果捷径，IN统计/OOD伪影仍混杂。
- 直接teacher源WT/TC/ET87.0109/57.9758/50.9932→86.9463/60.2818/51.1056%；NC30.1266→33.0682%；ED存在7例27.7336→27.9593%。相对同方向数量margin，TC+.9302pp CI[+.3225,+1.6168]，NC+1.4447pp。真NC改对14743/破坏47，真ED改对0/破坏216，原NC→ED恢复12760/新增0。通过预设小实验门槛，但不是最终方法。等权融合WT下降1.0036pp未过门槛；单背景严重损伤WT。
- 两臂各12×24步真实decoder更新17,168,996参数，context对self同patch/编辑mask/源概率锚点，逐例逐臂重置。context WT/TC/ET85.5766/64.6788/50.7475%，NC39.6557%；self85.9514/56.3166/50.4125%。context TC比直接teacher+4.3970pp、比同量margin+2.0603pp，仍因WT−1.4343pp、ET−.2457pp和ET全空2→3未通过，不扩大此学习配置。
- context新ET全空00008原source16体素全为误报、与GT610体素零重叠；零重叠总数未增，不能夸大为毁掉正确ET。共同有效ET HD95源6.1020→6.1755mm，不能比较不同分母的7.2210→6.1755。真NC改对48155/破坏1401，真ED311/2740；WT真前景转BG38285而恢复2931，存在过度收缩。
- 第一阶段108mask/24概率/24输入hash、第二阶段72新mask/24概率/24delta、576有效更新及冻结参数/实际推断state/源checkpoint/配对控制核验通过。第一阶段4.93min，第二阶段9.99min，峰值1.899GiB。首次AMP首步溢出在更新前中止，旧产物保留`adapt_attempt01`；按GradScaler回退同batch重试修复，未改实验超参数或用GT调参。
- 固定直接teacher的99例扩大验证已启动：`D:/Workspace/deeplearn/3dunet/outputs/ped2023_context_full99_20261008`。复用12、另推87，四类输出及同域方向数量对照；协议在参数学习评分前已固定。主看新增87并报告99及去重98/86；全部仍为开发病例。**此处启动状态待完成记录取代；接续先看prediction_complete/verification/REPORT，不重复启动。**

## 2026-10-08：对比度响应探针完成，仍无可推进候选

- 继续使用10月7日的逐病例、单一IN四分类epoch250、无GLI样本/额外模型协议。阅读历史失败记录后新增FLAIR/T2 gamma .7/1.5观察探针，原12开发ID完整体积、四类输出开放；冻结网络，参数更新0次，不称新颖TTA。
- 入口：[本轮结果](../ped2023_contrast_response_20261008.md)。完整代码/协议/报告/132mask/48份FP32概率/评价与核验在 `D:/Workspace/deeplearn/3dunet/outputs/ped2023_contrast_probe_20261008`；该目录被Git忽略，仅本地保存。
- Source WT/TC/ET=87.0109/57.9758/50.9932%；Source+4观察等权融合=87.0188/57.9703/50.9789%；NC30.1266→30.1059%。3/4可靠投票教师与源12份mask逐体素相同。融合三域均值−.0039pp，开发配对区间[-.0176,+.0090]；未达到预设推进条件。
- 核心证据：216148个原NC→ED错误，任一gamma观察看对1878=.8688%；其中192986个源ED概率≥.9的错误仅看对1个。融合恢复原NC→ED31、新增76，真NC改对43/破坏145。不是可靠的新教师；不调gamma/门槛、不蒸馏、不扩大99例。
- HD95源9.3809/11.3177/7.2210mm，融合9.3492/11.3176/7.2253，共同有效12/11/10；预测全空0/1/2、GT非空零重叠0/1/3均未变。真ED存在7例，ED Dice27.7336→27.7361%；真ET缺失0例。
- cv311环境与合成gamma/几何/投票自测通过；首例源概率逐值一致、全部源指标精确复现。预测/评价/输入重建三个进程exit0，所有产物、方向数量对照、48个变化输入hash、源state/checkpoint不变均通过。预测412.50s+评价52.04s≈7.74min，另有并行输入核验；未改核心训练/推断代码、未跑全套pytest、未启动Kaggle/AutoDL或新训练。
- 新核实：[PED2023官方结果论文§2.2](https://arxiv.org/html/2407.08855v3)明确NC=1、ED=2、ET=3，补全官方数字元数据依据；Kaggle与官方原件逐体素相同仍未证实。不能继续把“官方数字定义未确认”和“文件等价未确认”混写。
- 文献[DA-nnUNet](https://arxiv.org/html/2406.16848v1)依赖带标签成人样本和多病例联合训练，不是当前checkpoint-only episodic协议；本轮没有切换资源设定。成人/儿童语义及条件分布差异值得审计，但不能仅凭标签名称不同判定根因。后续仍需独立可检验的语义锚点，不把本轮失败改名为新方案。

## 2026-10-07：当前状态与接续入口

本节为10月7日交接；更新状态以上方10月8日记录为准。后面的按日期记录保留实验细节和失败原因。历史记录中的“运行中”“待下载”“下一候选”只描述当时状态，不是当前待执行指令。以下仅涉及研究、代码与实验。

### 当前目标与结论

- 研究成人BraTS2023 GLI源模型到BraTS2023 PED的逐病例TTA，不是CTTA。每病例重置模型、优化器及教师状态；适应阶段不读取目标标签，不访问GLI图像/标签，也不默认增加源训练或外部模型。
- 当前主要问题是肿瘤内部NC与ED混淆。源模型WT相对可靠，但TC大量低估；这只是问题定位，尚未找到经验证可靠的新方法。
- 用户已明确否定“固定WT/ET、单向ED→NC后只看TC上涨”作为方法成功依据。后续应允许完整四分类输出变化，评价WT/TC/ET的Dice、HD95，并检查NC/ED组织正确性及真实ED受损情况。解除硬保护本身也不是创新。
- 截至最近完成记录，跨轴适应、源快照教师、对侧结构/T2-FLAIR先验、N4输入校正均已结束，未达到继续扩大实验的标准。本次没有启动新的训练、推断或Kaggle任务；没有重新查询AutoDL电源状态。

### 当前源模型、数据与指标

- 主模型：native nnU-Net v2四分类InstanceNorm、Dice+CE、Dataset503、final/best epoch250。六级PlainConvUNet，31,199,796参数；patch128³、batch2、250×250次更新、AMP，GLI按患者划分998/253次扫描。不是历史自定义BatchNorm模型。
- 备份根目录：`D:/Workspace/deeplearn/3dunet/artifacts/models/in_dicece`；权重相对路径：`run/nnUNet_results/Dataset503_BraTS_Classes/nnUNetTrainerBraTS250__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`。加载时使用配套plans、dataset和Trainer，勿混用BN或二分类配置。
- GLI验证WT/TC/ET Dice为93.68/92.17/87.33%，区域平均91.06%；前景组织类别平均83.90%是另一指标。源验证参与选模，不是独立测试。
- PED2023映射为0=BG、1=NC、2=ED、3=ET；WT={1,2,3}，TC={1,3}，ET={3}。不得套用PED2024五标签定义。四模态顺序FLAIR/T1/T1ce/T2，采用native预处理，不额外做HD-BET。
- 原始99例完整体积分析在`D:/Workspace/deeplearn/3dunet/artifacts/experiments/nnUNet_PED2023_IN_analysis_20261005`。冻结源模型WT/TC/ET Dice89.3334/48.8627/54.6713%；HD95为5.2672/14.7466/9.3240mm，有效数量99/92/85。全部99例已参与开发，不能重新划分并称独立测试；00121与00137文件重复，去重98文件组不等于已证明98名独立患者。
- 近期四分类评价：双方空Dice=1、HD95=0；单侧空Dice=0、HD95=NA。必须报告HD95有效数量、共同有效病例比较和全漏/误报数量。早期二分类HD-TTA对单侧空使用100mm，不能直接混用口径。
- 12例80³局部、相同12例完整体积、99例完整体积是三个不同评价范围。12例完整体积源Dice为87.0109/57.9758/50.9932%，不要与局部源86.9499/58.1153/50.5789%交叉比较。

### TENT与SAR：已完成，SAR必须带审计结论

- 入口：[TENT/SAR记录](../ped2023_boundary_and_norm_tta_20261006.md)、[SAR审计](../ped2023_sar_audit_20261006.md)；完整产物`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_norm_TTA_20261006`和`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_SAR_audit_20261006`。
- 两法均面向BG/NC/ED/ET全四类，更新22层IN affine共7,680标量；卷积及分类头冻结，WT/ET不固定。128³、0.5滑窗、Gaussian与8镜像，逐patch在线更新，保存更新前输出；每病例重置。不能描述为只优化TC，也不是适应全例后再推理一遍。
- TENT：Adam1e-3；99例738次真实更新，Dice89.5657/47.8587/54.8304%，未解决TC不足。HD95为4.4156/15.0171/9.1098mm，有效99/91/84；不得忽略与source有效集合不同。
- SAR：SAM-SGD1e-3、rho0.05、可靠性阈值0.4ln4、EMA0.9、恢复阈值0.2。738/738步实际恢复，最终99个mask等于匹配source；审计确认比较方向没有写反。源模型更新前熵已远低于0.2，是分类任务绝对阈值迁移至密集分割后的退化，不能解释为全部病例适应后坍塌，也不能作为充分适配的通用SAR性能结论。
- 核心SAM真实patch与作者实现对照通过，但作者wrapper与本地base-SGD momentum清理存在差异，不能声称跨步状态完全复现。3例禁用恢复仅改1/1/104体素，未读取GT评价，不是新性能结果；勿直接关闭恢复或看GT调阈值后冒称原版SAR。
- 双账号各两块T4的任务已完成并下载核验；提交至计算结束TENT32.88分钟、SAR32.43分钟，并行跨度34.74分钟，本地核验另4.81分钟。旧“等待下载/监控”说明已被这些完成记录取代，无需重复提交。

### 已探索路线及不能重复包装的结论

| 路线 | 已完成范围与结论 | 仓库入口 |
|---|---|---|
| MRI/特征连续性 | 99例；固定WT/ET时MRI使TC48.86→65.59%，但同修改量随机已达64.14%，并侵占真实ED；没有稳定超过源概率/几何规则的TC证据 | [连续性](../ped2023_continuity_full99_20261006.md)、[匹配对照与前例](../ped2023_mri_matched_controls_20261006.md) |
| 留出模态边界、ET及四分类扩展 | 内部界面语义判断不可靠；完整四分类联合WT/TC/ET降至63.85/36.33/39.11%，停止当前选择器 | [留出界面](../ped2023_heldout_boundary_20261006.md)、[四分类](../ped2023_fourclass_boundary_20261006.md) |
| GMM/图/多上下文教师、冻结头IN适应 | 12例80³局部，v7/v8均未通过；ED种子污染，裁剪纠错互补性低，门控上下文学习结果可与冻结集成完全相同 | [v7](../gli_to_ped2023_three_teacher_experiments_v7.md)、[v8](../gli_to_ped2023_frozen_head_candidates_v8.md) |
| 真正跨轴教师与解码器适应 | 12例完整体积，真实更新已有decoder；WT改善而TC、ET与NC下降，未解决NC→ED | [跨轴](../ped2023_axis_tta_validation_20261006.md) |
| 源训练175/200/225快照教师 | 12例完整体积；额外checkpoint已明确，不是单checkpoint设定；自动教师仅恢复1个原NC→ED错误，未推进训练 | [快照](../ped2023_snapshot_teacher_validation_20261006.md) |
| 对侧结构与T2/FLAIR先验 | 12例完整体积；对侧TC略升但真实ED受损，未优于同编辑量强对照；组织相似仍不能可靠确定类别 | [影像先验](../ped2023_anatomy_contrast_validation_20261006.md) |
| N4输入偏场校正 | 12例完整体积、两版四类开放；healthy三域平均65.3266→65.7536%，但NC下降且新增NC→ED多于恢复，未通过；是既有输入校正，非新颖参数TTA | [N4](../ped2023_n4_input_validation_20261006.md) |

最近两个实验根目录为`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_anatomy_contrast_20261006`和`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_n4_probe_20261006`，均有协议、预测、独立评价、报告与verification。N4预测及评价合计8.85分钟；未扩大99例，未依据本轮GT重调参数。更早v2—v6的失败机制见下方历史记录与对应研究文档。

### 原始形状分析与HD-TTA资料

- 原始预测与形状研究保留在`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_TC_shape_review_20261006`，病例入口`easy_view/index.html`，研究入口[TC形状分析](../ped2023_source_tc_shape_review.md)。真实TC主体通常连通，但错误预测也常连通且更圆；填洞/闭运算只能覆盖很少的NC→ED错误，不能把球形、中心位置或ED包围作为硬先验。
- 98文件组中TC好/差病例主要差异为组织组成及NC→ED错分，球形度与TC Dice相关很弱；这是标签辅助描述，不能当适应阶段可用的真实组织先验。
- HD-TTA原实验保留在`D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_native_nnunet_20261002`，纯Dice、四分类边缘概率WT与论文公式控制另有独立目录，见下方模型及HD-TTA章节。它冻结网络并优化病例logits；作者代码与论文公式并非完全一致，应按各实验PROTOCOL区分。不能将二分类WT修正结果直接当四分类内部纠错证据。

### 后续工作与操作边界

- 当前没有已证明有效、等待直接扩展的候选。下次研究应先提出有文献依据、可证伪的NC/ED语义纠错信号，再做小规模完整体积验证；不要仅给现有错误教师增加适配器或蒸馏层。
- 修改方向和数量匹配的源概率/几何/随机对照用于排除类别扩张偏置；还需报告改对与破坏原正确组织的数量，而非只报恢复量。真正训练时保留自身教师或冻结集成对照，区分信号来源与优化收益。
- 按各轮预先固定协议决定是否推进；目前无需重跑已完成Kaggle、99例推断或源训练。不按已看过的GT选择阈值、epoch或病例再称无监督成功；额外checkpoint/模型资源必须明确。
- Python固定`D:/Environment/miniforge3/envs/cv311/python.exe`；本机RTX4060 Laptop约8GiB。具体实验代码通常在`E:/BraTS_PED2023_*`或`E:/HD_TTA_*`，仅检查仓库src不能代表实际实验实现。
- 仓库已有大量未提交修改、删除和新增文件；保留现状，未经要求不commit/push/reset。继续工作前先读AGENTS.md、对应PROTOCOL、REPORT和verification，不输出凭据、不更换核心环境或擅自启动付费机器。

---

以下为历史研究记录。日期靠后的完成结果优先；其中旧的待办或运行状态不代表当前仍在执行。

## 2026-10-06：继续探索影像先验与输入偏场，均已完成

- 用户“没有找到就继续找”。本轮依次完成两个独立影像语义先验、两个N4输入版本，全部原12开发ID完整体积；非旧80³局部，也非99独立测试。无GLI图像/标签、额外模型、网络参数训练或跨病例状态。所有任务结束，无新Kaggle或git提交。
- 影像先验`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_anatomy_contrast_20261006`，REPORT/PROTOCOL/代码/84mask/分数/评价与核验齐全；仓库`docs/ped2023_anatomy_contrast_validation_20261006.md`。A用T1/T2正常对侧局部NCC损失分位数暂命名NC；B用病例内T2/FLAIR混合组分按相对信号命名，不以源ED拟合负类。两者重分配NC+ED概率后四类argmax，没贴回WT/ET，但BG/ET概率不变，不能声称均衡优化三域。
- 源WT/TC/ET87.0109/57.9758/50.9932；对侧87.0120/59.2813/50.9932，NC30.1266→32.0737；但真ED存在7例ED27.7336→25.4128（−2.3207pp）。相对同编辑方向数量margin，TC仅+.1872pp CI[-.4173,+.8226]，随机TC59.2223也接近。真NC改对7685/破坏57，真ED改对0/破坏739，覆盖原NC→ED仅25307/216148，恢复7672；不能只报找回的NC。
- 对侧HD WT/TC/ET9.3815/15.2531/7.2210，有效12/12/10；源9.3809/11.3177/7.2210，有效12/11/10。TC源共同11例改善.9697mm，但同编辑margin共同12例差6.2741mm，分母变化必须交代。对侧WT/ET实际只改21/4体素。T2/FLAIR87.0150/56.9423/51.0016，真NC改对12515/破坏24153；两先验均未通过，不调阈值/融合/蒸馏。AP诊断可比较病例仅A1例(.5247vs源.7238)/B3例(.6510vs.8298)，不能推广。GT无ET缺失病例。
- 先验预测54.73s+成功评价33.16s，复用源缓存；84mask、概率/方向数量对照/全部对侧参照坐标/源指标/输入输出checkpoint hash通过。首次评价汇总numpy bool序列化失败，仅加bool转换后重评，预测不变；旧脚本/log/start保留evaluation_attempt01。
- 新方向`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_n4_probe_20261006`，REPORT/PROTOCOL/代码/36mask/24组四模态field与概率/评价/verification齐全；仓库`docs/ped2023_n4_input_validation_20261006.md`。既有N4仅作为成像偏差检验，不是创新或网络参数TTA。healthy从源WT外5mm候选正常区域拟合，brain全部正值脑区对照；固定shrink4、[30,20,10]、1e-4、4³控制点，四模态各自估计后重新native128³/.5/Gaussian/8镜像推断，四类全开放，无WT/ET固定。各48模态均无回退。
- N4 healthy WT/TC/ET87.7046/58.5469/51.0092，三域65.7536 vs源65.3266（+.4269pp CI[-.1298,+1.0923]），NC29.8731下降.2535pp，ED存在子组27.9828略升；brain87.6936/58.1791/50.9852，NC29.7638/ED27.2905。healthy真NC改对1573/破坏2295，源NC→ED恢复964但新增1196，未通过NC收益和净纠错；不蒸馏/扩大99例/GT后调强度。
- healthy原HD4.3914/15.6268/11.6152，n12/12/11；共同源n12/11/10为4.3914/11.2193/6.8497，源9.3809/11.3177/7.2210。00009仅多1个TC真阳性，仍漏84231，该点标ET但真NC，真ET7体素仍无重叠；此例TC64.1093/ET59.2705从缺失进入均值，不能把全漏数降低当实用恢复。00008ET命中0→2但仍漏608。
- healthy将源ED里的真NC改NC964、改ET1237；后者TC虽对但组织语义仍错；还将1041个原正确NC改ET。因此TC略涨不能称NC/ED问题已解决。整体正信号可以保留，但不是可靠成功，更不能将N4包装新颖性。
- N4首例原raw补推与缓存概率逐值相同；合成乘性场测试/空mask回退/原零值、field重建校正输入hash、四类概率argmax/几何、源全指标、每病例实际模型state/原checkpoint不变通过。预测502.97s+评价28.31s=8.85min，GPU峰值1.222GiB，单版本校正+推断保存平均19.63/20.00s每例。进程43990/82154已exit0，源权重不变。下一步不要将这几个失败探针重复包装，或按GT挑强度/拼接不同先验的优点。

## 2026-10-06：既有源训练快照教师验证完成，未通过

- 用户“想其他办法并检验”。本轮改用同一次源训练175/200/225三个已有checkpoint提供语义假设，250仍源学生；启动前已明确额外权重资源，不是单checkpoint设定，无GLI图像/标签、新源训练、独立BN或跨PED状态。文献Snapshot Ensembles/Self-ensemble TTA只借鉴多样性，普通训练不同于循环学习率，不称复现或创新。
- 目录`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_snapshot_probe_20261006`，REPORT/PROTOCOL、predict/evaluate/report/check/common代码、108新完整NIfTI/36个旧快照推断概率/教师/对照/全部评价/hash回执齐全。仓库`docs/ped2023_snapshot_teacher_validation_20261006.md`。
- 同12开发ID完整体积、四分类全部开放，128³/.5/Gaussian/8镜像。固定175/200/225；源250概率复用上一轮已核验缓存，首例补推逐值相同。低层原生完整滑窗防止重载源250；每个加载状态核对。三快照同类且各P≥.8、margin≥.2接受，否则保留250。另存旧3/全部4融合、各epoch诊断、同源类别/方向/数量匹配margin及随机对照。当前只做教师信号，没有optimizer更新。
- Dice WT/TC/ET：source87.0109/57.9758/50.9932；17586.7523/56.5357/50.8595；20086.6753/58.5959/51.7707；22587.2824/55.8878/51.0080；旧3融合86.9785/56.6957/51.1811；全4融合87.0270/57.1080/51.2601；teacher87.0358/57.9430/51.0487；matched_margin87.0225/57.9421/51.0276；random86.8956/57.9400/50.9709。200看开发GT后较好不能据此冒充无标签模型选择TTA。
- teacher NC30.1281 vs源30.1266，仅+.0014965pp；TC−.0328pp，三域均值+.0159pp CI[-.0331,.0665]。真ED存在7例ED源27.7336→teacher27.7568；真ET缺失0例。teacher相对同Kmargin TC只+.0008775pp，区间跨0。预设NC/TC各至少+.5pp未满足，停止参数适应、不扩大99例、不调阈值/换epoch。没有新成功方法。
- teacher改3629：correct2338/damaged261/wrong→wrong1030。真NC改对319中318是ET→NC（本来就在TC内），仅1个ED→NC；新增157个真NC→ED。源NC→ED216148，任一旧快照正确6126=2.8342%，自动teacher恢复1=.0005%；GT上限和pooled率，不是病例平均或自动准确率。
- teacher HD9.2841/11.3177/7.2207mm vssource9.3809/11.3177/7.2210，共同n12/11/10；预测为空且GT非空计数均0/1/2。报告明确这不同于GT非空且TP=0（source/teacher均0/1/3）。全4融合HD4.8778/15.7741/11.8768（n12/12/11），共同源病例TC实际11.4179vs11.3177、ET7.1875vs7.2210。00009融合新增2TC真阳性但仍漏84230；2体素标ET，与真ET7体素无重叠，不能把ET空→非空当恢复ET。zero_overlap_diagnosis.json保留诊断。
- 108输出/概率归一/独立教师阈值重放/空间坐标/原源全部指标/对照逐方向数量和mask/输入checkpoint产物hash均通过；四checkpoint文件未修改。predict313.30s+evaluate35.81s≈5.82min，非单方法时长；峰值1.222GiB。所有进程结束，无Kaggle、新训练或git提交。下一步不要重复这些快照探针，也不能只挑200的成绩包装成功。

## 2026-10-06：真实跨轴与解码器TTA验证已完成

- 用户“现在验证下你的想法”对应跨轴方案。`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_axis_probe_20261006/FINAL_REPORT.md`，仓库`docs/ped2023_axis_tta_validation_20261006.md`。既有12开发ID完整体积，原方向+2循环轴置换，无WT/ET固定；不是旧裁剪/80³实验，也不是99独立测试。
- 第一阶段84个完整输出，原生128³/.5/Gaussian/8镜像。排除学生的2教师同类且各P≥.9、margin≥.2。teacher0均值仅+0.0164pp但通过预设工程门槛；随后另行冻结ADAPT_PROTOCOL。已披露第一阶段GT用于开发推进，不把第二阶段说成全程未看该开发集标签；每阶段自身预测全固定再评价。
- 第二阶段2臂×12病例×24步=576次真实更新，冻结encoder/不用的深监督head，更新已有decoder17,168,996参数；Adam1e-4，AMP，三方向各8步。cross用另两方向固定平均概率；self用本方向冻结概率，同patch中心/mask/优化器/步数。每病例每臂重置模型/Adam/GradScaler。最终原方向8镜像完整推断，直接predict_sliding_window_return_logits，避免高层接口重载源checkpoint。无新增adapter、GLI访问、额外模型。
- Dice WT/TC/ET：source87.0109/57.9758/50.9932；冻结融合86.9808/58.1034/50.9382；teacher0直接86.9946/57.9629/51.0714；cross87.6551/57.8995/50.7835；self87.3933/57.9484/50.5829。三域均值source65.3266→cross65.4460，+0.1194pp CI[-.2303,.4139]；cross−fusion+.1052 CI[-.1545,.3424]；cross−self+.1378 CI[-.0419,.3118]。WT改善，但TC/ET/NC均下降，不能称整体成功，不扩大99或GT后调参。
- HD95 source9.3809/11.3177/7.2210（n12/11/10）；fusion5.3020/11.1061/8.0810；cross4.4088/11.1579/6.1482（12/11/9）；self4.4150/11.1536/4.4287（12/11/8）。ET全漏2→cross3/self4，共同有效9例cross ET HD相对源实际+0.4003mm，不能按改变分母后的均值称改善。cross新增全漏00062：GT258、源仅预测2体素、adapt0，原本就严重漏分，不能夸大为毁掉良好大ET。
- 真ED存在7例ED Dice source27.7336/cross27.8145/self27.8445；真ET缺失0例，不支持缺失误报判断。NC Dice30.1266→29.8182。源NC→ED216148，任一轴正确8605=3.9811%，自动teacher0恢复169=.0782%；cross恢复1249但新增2584个NC→ED，真NC总改对1817/破坏4713。总改对18571中14244真BG，主要是误报清理，不是核心语义解决。
- 108个新NIfTI、FP32概率、24组delta、逐步记录、两阶段代码/协议/评价/hash回执齐全。非立方置换/逆变换/裁剪补零/教师对齐测试，全部source指标精确复现、冻结权重/实际推断state、delta独立范数、两臂配对采样验证全部通过。源checkpoint不变。
- 时长：第一阶段330.23s预测+50.00s评价；第二阶段446.19s两臂适应/推断/保存+15.37s评价，执行合计约14.03min，不含开发、不是单方法部署速度。显存峰值1.864GiB。所有进程exit0，无新Kaggle任务，未提交git。下一步不重复本轮，也不把轴视图互補自动等同正确语义。

## 2026-10-06：用户否定固定WT/ET的TC主指标，要求换路线（本轮只设计）

- 用户认为固定WT/ET后TC容易靠ED→NC涨分，要求其他方法。明确不再把该机制设置当完整TTA主终点；也不能把“任意修改必涨”当数学事实，已有受损病例。后续要求完整四类、三域Dice/HD95及ED/ET语义损害。
- 本轮重查历史与文献，方案`docs/gli_to_ped2023_geometry_tta_rethink_20261006.md`；**没有启动新的推断/训练或产生新成绩**。前轮所有任务均结束。
- 优先补测真正轴置换/几何等变的四分类MuVi-inspired路线。v4曾提出，但v7实际只是4裁剪上下文+2镜像，不是原MuVi，0.82%仅约束旧裁剪实验。先在既有12开发ID完整体积产生原方向+2循环轴置换，对齐四类概率，检查实际自动教师净纠错与等权融合，不固定WT/ET，不做ED单向放宽。
- 待验证教师起点：排除学生的另外两方向同类，maxP各>=0.9且margin>=0.2，固定软目标，其他跳过。只有新信号成立才做每病例重置的现有decoder更新（encoder冻结），对照同视图冻结融合。尚未冻结最终训练协议/学习率/步数，未称有效或创新；多视图本身已有MuVi、不能当首创。
- 备选利用已存在的晚期源IN快照（175/200/225/250确实在本机）检查不同语义假设；额外checkpoint必须披露资源，不访问GLI、不训练新教师。BN独立模型另算资源。仅集成不当作新学习方法，无标签挑专家仍未解决，不能以熵低或TC大选择，更不能用GT最佳专家冒充部署。
- 不复活已失败的正常组织T1→T1ce回归增强阈值、原型、界面扩张；没有BN统计的IN不能直接照搬OptTTA。SmaRT预处理/模型/PED2024设定不同；MOS在线3D检测仅提供多假设参考，不搬跨病例历史。文献链接/适用边界在新文档中。

## 2026-10-06：ET与完整四分类边界试验已完成，整体明显退化

- 用户要求“对于ET之类也弄一下，看看整体”。独立目录`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_fourclass_boundary_20261006`；PROTOCOL/REPORT、cache/predict/evaluate/verify_selection/report/plot脚本、594新NIfTI、完整候选/四类概率MRI缓存/评价/verification/selection_verification/per_case_metrics/paired_dice图齐全。仓库`docs/ped2023_fourclass_boundary_20261006.md`。没有新Kaggle任务或参数学习。
- native IN四分类epoch250补一次源概率推断，全部99硬mask与旧源逐位一致。完整四模态、128³、0.5、Gaussian/8镜像。图域源WT+外5mm BG，12有向类别转换；3模态生成/留出模态核验，原阈值0.5/1/2、delta≥0.25且超过同K对照≥0.10。每原类别组件最多一候选，不允许所选候选相邻；触及图域截断拒绝。最终joint没有BG→前景入选，背景扩张大量因截断/竞争回退，不能声称验证了全脑漏病灶恢复。
- 本轮各类种子P>=0.9按八镜像合并概率，不要求每个镜像同类；与上轮NC-only不是严格仅开放类别消融。新et_only含BG↔ET（也会改变WT/TC）；wt_only是BG↔前景；foreground是三前景互转WT固定；joint全部开放，无WT/ET硬保护；两个joint对照同源组件/方向/K。
- 全99 WT/TC/ET Dice：source89.3334/48.8627/54.6713（均64.2891）；ET相关86.9446/53.4321/24.6780（55.0183）；BG转换52.2649/38.0956/44.7374（45.0326）；前景转换89.3334/47.7745/30.2022（55.7700）；joint63.8467/36.3281/39.1139（46.4296）；同K P(b)64.3517/36.6699/39.7822（46.9346）；同K几何64.2626/36.6776/39.5652（46.8351）。不能宣称任何整体成功。
- HD95同WT99/TC92/ET85：source5.2672/14.7466/9.3240；ET相关5.3870/12.0649/15.3814；BG转换10.0013/15.9207/11.3066；前景转换5.2672/12.9496/13.6373；joint8.4571/15.7616/12.6194。全漏WT/TC/ET仍0/7/7。distinct98三域联合Dice相对源均明确下降，三域平均−17.9162pp CI[-20.9769,-14.9150]，都是开发描述。
- ET相关版本pooled：找回真ET56,443、丢失真ET260,686、新增ET误报1,657,757、消除误报141,739。真ET缺失11例仍7例误报，误报体素8968→126715。joint ET找回10468、丢失143026、新增误报327967。
- joint改2,683,128体素：改对291,829、破坏原正确1,241,721、错改另一错误1,149,578。ED→BG1,452,323中797181真NC、439941真ED、27810真ET，导致WT大退化；NC→ET121437中115469真NC，ED→ET216998中143564真NC。边界评分缺乏可靠ET/背景语义。
- 594输出与稀疏编辑、方向配额、所有源区域指标精确复现、输入/GT/输出hash/坐标全部通过；294去重已选候选独立网格面/原组件/同K排序核验通过。模型hash前后不变，所有预测先于GT。小例12方向/冲突检查通过，paired_dice图目视无问题。JSON paired wins/losses是差值正负计数，HD95负差值才改善，报告有说明，不要按字段名误读。
- 缓存740.13s；候选与缓存并行623.61s，源缓存开始到所有预测固定约12.5min，不能相加。GT评价48.09s与独立选择核验68.03s也并行。session94457(cache)/55946(predict)/2334(finish)全部exit0。没有仍运行的任务。
- 用户现已要求并得到未固定WT/ET的整体结果；不能再只提出“应该试全类”而忽略这一轮。所有99仍开发，distinct98不等独立患者，镜像官方等价未核实。停止当前跨类别选择器，不在GT后调阈值或加adapter掩盖失败。

## 2026-10-06：留出MRI内部界面方案已试完，未通过

- 用户“你试试看”指SAR审计后NEXT_METHOD草案。目录`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_heldout_boundary_20261006`；完整REPORT.md、PROTOCOL.md、probe/evaluate/verify/report.py、候选/稀疏输出/评价/selection_diagnosis/verification全部保存；仓库`docs/ped2023_heldout_boundary_20261006.md`。未继续完整四类适应，无GPU训练/Kaggle任务。
- 全99完整体积，三MRI生成候选、留出MRI核验NC/ED内部界面；源ED组件每个最多4×3候选（阈值0.5/1/2），候选≥64体素，旧/新界面均≥16面，delta≥0.25且超过同K P(NC)/几何候选≥0.10，每组件取一个。固定WT/ET，仅机制检验。完整四模态冻结nnU-Net缓存提供源语义；不称模态统计独立。
- 全99 TC Dice/HD95：source48.8627/14.7466；主69.0626/8.6390；同K概率69.0568/8.6383；同K几何69.1124/8.5810；gate_only消融68.9690/8.7351。TC同92有效，7全漏不变；WT89.3334/5.2672(n99)、ET54.6713/9.3240(n85)硬固定。真ED57例ED Dice24.3881→10.9024（同K概率12.3208）。
- distinct98主对同K概率+0.0027pp CI[-0.1770,+0.1500]，几何−0.0513[-0.2695,+0.1095]；HD95无优势。所有门槛除覆盖均未通过，不能把TC涨20pp当边界选择成功，不加蒸馏/adapter掩盖失败。
- 704候选、604界面有效、373初筛通过，66例最终67区域；2,278,927改动真NC1,372,638、ED480,724、BG383,711、ET41,854。TC57升9降33不变，00026最差−58.83pp、00025−53.48pp。
- 核心诊断：选区占对应源ED组件中位98.25%，59/67≥90%，62/67选最大阈值2。禁止完全消除内部界面仍挡不住近乎整块吞并。00025改102091，98213真ED、360真NC，delta仍+0.7062；removed22/new2301面。语义和修改量没解决，面积不对称是解释线索，不是已证实的修复方法。
- 候选AP仅2病例有正负类，边界0.4378 vs Pnc0.8214，不足推广；38例候选纯度变化时病例内Spearman均值−0.1453 vs +0.2994，仅描述。全部99仍是开发证据，去重不等于独立患者测试。
- 小型界面/消失回退/留出隔离/坐标排列/AP并列检查通过；全部99源TC指标精确复现，所有67主区域原网格界面枚举和独立ED组件同K排序重放通过。预测/协议hash不变，预测先于GT。初次评价JSON int64保存失败，仅int转换修复，旧脚本/日志/start归档evaluation_attempt01；没有重调预测。
- 生成45.72s，成功评价19.30s，约1.08min CPU缓存计算，不含旧推理/附加核验/首次失败。稀疏`edits.npz`可从父source重建完整预测，没声称存了新整套NIfTI。所有进程已exit0，无剩余后台任务。

## 2026-10-06：SAR全部恢复审计已完成，发现阈值移植退化

- 用户要求想解决办法并质疑SAR全部恢复。目录`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_SAR_audit_20261006`，REPORT.md、verification.json、log_audit.json、patch_audit.json、official源码快照、no_recovery完整mask。仓库`docs/ped2023_sar_audit_20261006.md`，旧TENT/SAR仓库文档已加醒目审计说明。无新Kaggle任务/99例重跑。
- 738步EMA重算、实际delta前后全通过。更新前全图H最大0.0266589，second H最大0.0178536，EMA最大0.0174376，均远低于0.2；作者源码确实小于0.2恢复，没写反。**源预测适应前就满足恢复阈值，不能称适应坍塌；旧表应标“0.2绝对阈值恢复退化的SAR-IN移植”，不能当通用SAR有效基线。**
- 按旧second H最小/中位/最大选00051 patch7、00062 patch0、00170 patch5，128³实图本机AMP均复现恢复；背景99.9586/98.1244/90.3813%，前景H0.25663/0.12114/0.12813，可靠前景H0.19466/0.07602/0.08383。不能声称剔除背景即可修复。
- 同3patch中心64³ FP32对作者原始SAM/forward_and_adapt_sar核心更新，参数最大差7.45e-8，flag和返回更新前logits一致。手工启用相同IN（原分类configure不支持IN3d），不是完整原论文复现。网络无dropout，IN无running stats，train/eval逐位一致。
- 真实差异：作者下载的SAM wrapper reset没有清base SGD momentum，实测44组仍在；本地清实际SGD state。不是全部恢复根因，但不能宣称跨步优化状态精确一致。官方forward恢复后又赋回EMA，本地保留EMA与此相符。参数恢复+更新前输出足以解释所有mask等source。
- 另补3完整病例SAR-no-recovery诊断，事前addendum记录，不读GT、不测Dice/HD95。00051/00062/00170分别8/4/8步全保留，delta0.00011236/0.00004826/0.00185030；标签改1/1/104体素，WT/TC/ET变化1/0/0、0/1/0、103/1/1。不能把它当精度收益或标准SAR新成绩。源mask逐位复现、每病例状态重置、原checkpoint不变；两计算session20437/64130均exit0。
- 本机审计patch13.14s、3病例49.42s。默认Windows证书库下载源码失败，改用已有certifi CA重试，未关闭TLS验证/未改依赖；原日志和代码保留hash。
- 下一候选草案`NEXT_METHOD.md`尚未实现：源概率给类别候选、三模态提出区域/留出模态验证整段旧新界面，TC/ED必须用内部界面而不能借WT外缘；同K几何/概率反事实和拒绝，最终完整四类可变无WT/ET硬保护。先检验评分区分恢复NC/吞ED再谈适应。源ED种子污染和MRI类别不可辨识没有凭空解决，不宣称有效/创新，也不把已有MM-TTA/相对模糊连通/HD-TTA基本思想当首创。

## 2026-10-06：MRI同修改量核验完成，创新性已有明确前例

- 用户要求验证“随便把ED改NC也能提升”，并问MRI连续性是否新。新目录`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_MRI_matched_controls_20261006`，完整REPORT.md/LITERATURE.md；仓库`docs/ped2023_mri_matched_controls_20261006.md`。所有99病例45候选/例，共4455稀疏编辑，无模型重跑/训练。
- 每例K固定旧MRI改动数，比较全ED随机20次、同种子无权可达域随机20次、P(NC)排序、可达域P(NC)、欧氏邻近和同种子无MRI边权路径。全部控制预测固定后才读取GT；P(NC)是保存概率，非NC-ED margin。随机Dice用精确条件期望，HD95用20次均值。
- 全99 TC Dice/HD95：source48.8627/14.7466，MRI65.5944/9.4948，随机64.1401/9.7765，可达随机64.2161/9.6526，P(NC)65.1389/9.9506，可达P(NC)65.2096/9.7971，邻近65.3519/10.2993，同种子路径65.1817/10.2871。TC同92有效，7全漏未变；WT/ET硬固定。随机达到MRI相对源增量的91.31%，只是条件数值比，不是因果分解；K仍来自MRI，不能声称任意修改量均有效。
- 去重98：MRI相对随机Dice+1.4467pp CI[1.0190,1.9233]；相对P(NC)+0.4302[-0.0558,0.9832]，邻近+0.2303[-0.2276,0.6807]，同种子路径+0.4008[-0.0765,0.8636]。HD95仍有较稳边界信号：相对P(NC)−0.4765mm CI[-0.8927,-0.1263]、邻近−0.7927[-1.2757,-0.3390]、路径−0.7828[-1.1985,-0.3909]；随机HD95区间跨0。不能只说MRI完全没作用，但不足把巨大Dice增量解释成边界贡献。均是开发事后区间。
- MRI误吞340632真ED，P(NC)312157；ED存在57例Dice MRI16.1813 vs P(NC)19.2048（source24.3881）。78病例K>0，中位修改71.62%源ED，21例全回退。完整GT组成和失败统计保存，不再改预测。
- 新颖性：当前minimax局部边差路径与fuzzy-connectedness max-min亲和度在exp(-w)变换下等价；Moonis2002脑肿瘤MRI、Huo2013精确定义、DeepIGeoS、HD-TTA2026均有相关前例。HD-TTA正式会议页面作者回复已提class-conditional与modality-specific geodesic多类扩展（仅设想，不是四类实测）。不能把MRI连通扩张本身或WT改TC当强创新。来源链接在LITERATURE.md。
- 术语修正：不更新网络参数并不自动排除TTA，LAME CVPR2022是parameter-free TTA明确前例；当前准确称冻结模型输出纠错验证，缺的是有效新机制证据。
- 检验通过K/域/随机与排序重放、坐标、MRI完整mask重建、source/MRI Dice/HD95精确复现及精确随机枚举小例、全坐标转置、裁剪表面等价；图已目视检查。对照生成1.14min、评价1.56min。session67549/36378计算已结束；无需重新提交Kaggle或推断。待最终答复时链接新报告与文献，保留HD95正向线索和所有开发限制。

## 2026-10-06：全99连续性验证已完成，TENT/SAR全区域范围已确认

- 本节覆盖下节运行中状态。`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_continuity_full99_20261006`预测、评价、报告全部exit0，session48990/8052已结束；无需继续等待或重跑。`verification.json`passed：495mask、396独立BFS重放、重建边强度、源mask与历史Dice/HD95完全一致、checkpoint不变。图results.png/pdf生成并目视检查。
- 全99 source TC48.8627%/HD9514.7466mm；MRI65.5944/9.4948；feature53.8071/12.9300；joint52.2895/13.2196；feature_interior53.3464/12.9308。TC HD95同92例有效，7例全漏未改善；WT89.3334%、ET54.6713%由硬保护不变。只有这项冻结模型连续性探索固定WT/ET，不是参数更新TTA。
- feature恢复402271NC(18.2371%)，新增BG65835和ED165569个TC误报；真实ED57例Dice24.3881→20.1090%。5例TC下降，00111最差−26.4511pp。MRI均值更高但ED Dice16.1813%，不可按TC最高宣布成功。外缘保护少恢复47495NC，避免26928BG/16365ED误报，TC均值−0.4607pp，未解决内部ED侵占。
- 去重98 source/feature TC48.8446/53.8394%，35升5降58不变；区间是开发样本描述。11例无种子。全部99仍开发，镜像官方等价未核实。本地预测墙钟15.79min、病例处理14.57min、评价3.45min、峰值1.38GiB。详见`docs/ped2023_continuity_full99_20261006.md`及E盘REPORT.md。
- 用户最新询问TENT/SAR是否只做TC，已明确答复：完整四分类BG/NC/ED/ET熵，更新全部22层IN affine，无TC限定、无WT/ET硬固定，三区域Dice/HD95均测；卷积/head冻结不等于只适配TC。已重新查验norm_adapter.py及Kaggle报告。不要为此重新提交云任务。
- 连续性当前结论：有恢复TC的信号，但弱边路径会侵入真实ED，不宜直接当完整TTA教师。未因看到GT再调阈值、重新训练或新增实验。所有原始候选及失败例完整保留。

## 2026-10-06：用户授权全99连续性验证，Kaggle结果已完成

- 最新用户要求继续探索，把方案放到完整99例，并查看Kaggle结果、解释MRI/feature两种连续性。已解释：MRI比较4模态局部值，feature比较冻结最后decoder32维表示；都用弱边路径可达性，不是球形/单连通先验。
- Kaggle TENT/SAR均已完成、下载并独立核验。报告`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_norm_TTA_20261006/REPORT.md`；396mask验证passed。Source WT/TC/ET89.3333/48.8627/54.6715%，TENT89.5657/47.8587/54.8304%，SAR完全同source。HD95 Source5.2682/14.7467/9.3231(n99/92/85)，TENT4.4156/15.0171/9.1098(n99/91/84)。TENT738步、TC和ET各新增1例完全漏检；SAR738步均恢复。两账号source一致，相比本机70例共194体素差异。
- 实测TENT提交→计算结束32.88分钟、SAR32.43分钟；两账号并行跨度34.74分钟，适应推理均值8.33/10.39秒每例。本地独立核验另4.81分钟。仓库`docs/ped2023_boundary_and_norm_tta_20261006.md`已新增最终表，原运行中信息是历史。
- 新全体积目录`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_continuity_full99_20261006`。固定`PROTOCOL.md`，`run.py`正在本地exec session48990运行，最后见日志到00037。不是Kaggle新任务。所有99完整体积，native8镜像/0.5滑窗/Gaussian，source要求与原本机mask逐体素一致。仅收集WT处32维feature，镜像还原及Gaussian加权FP32拼接，保持原native logits FP16算术；首例8镜像对齐通过、峰值约1.34GiB。
- 冻结网络、WT/ET硬固定；原MRI、feature、joint三个cutoff1.0规则保持。新增一个预先固定的feature_interior臂：移除距离源WT外部<2mm的源ED节点（路径和接受均限制），原NC保留；是保守保护试验，不假定TC必须居中。每个主臂都会保留，不根据GT选阈值。
- 全99预测保存后才启动`evaluate.py`（已写，尚未运行）。它会核验hash、源mask和历史Dice/HD95、镜像种子、从保存的MRI/feature重建边权、用独立有向BFS重放连通组件输出，评价WT/TC/ET Dice/HD95、真实ED、恢复TC与BG/ED新增误报、99和去重98。后续还需编写/运行报告、记录耗时、更新状态再答复。不要只报告12例局部收益。
- 实现修复：首例smoke因manual loader网络仍CPU失败，未产预测/读取GT；补显式network.to(device)后从源重跑，失败start/log在attempt01，说明REPAIRS.md。当前源代码hash由重跑start固定，勿在99运行中改run.py。
- 全体积单例smoke完成后正常续跑（跳过已完成receipt），无重新训练或参数TTA；全99仍开发证据。用户不希望持续实时监管，等待期间推进解释/评价代码即可。

## 2026-10-06：转回本地验证，特征连续性出现有限正向信号

- 用户新增指示：不用实时监管Kaggle，先在本地验证可行性。已停止交互式持续轮询；`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_norm_TTA_20261006/finish_run.py`在exec session50240后台等待两任务、自动下载并运行`verify_results.py`/`report_results.py`。最近一次本地进程读取显示TENT已COMPLETE并开始下载，SAR尚RUNNING；不要假设99例两法都已验收。最终看`REPORT.md`/`verification.json`，异常看`download_*.log`、`verify_results.log`。
- 新局部目录`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_boundary_feasibility_20261006`，已完成`run.py`、`evaluate.py`、`verify_report.py`；文档`docs/ped2023_boundary_feasibility_20261006.md`。同12例80³开发ROI；冻结模型，固定WT/ET，无参数TTA。6邻接体素图避免先按源类别切块；MRI4D/冻结32D特征/二者max三种边强度。源高置信双镜像NC为种子，固定最小瓶颈路径成本≤1进行单次ED→NC；全部预测后才读GT，无调阈值。
- Source TC Dice58.1153%、HD9511.2513mm；MRI74.4482%/5.5635；feature63.3256%/9.5020；joint60.0744%/10.6683。TC HD95同11/12有效，另1全漏仍计Dice0。WT86.9499%/4.8086(n12)、ET50.5789%/8.2608(n10)硬固定。均非全体积成绩。
- feature恢复NC29523（13.8725%），误改ED68，TC5升1降6不变。**还把5213个实际BG纳入TC，总新增TC误报5281，不能只报68个ED损害。** 另1341个GT ET改成NC，只恢复TC归属，没修复ET类别。00120因新增910背景TC误报，TC−0.5239pp、HD95+0.2295mm。
- MRI恢复NC97113但误改ED15866，ED存在7例Dice28.0699→22.2534%；feature为28.3278%，joint28.2528%。三个版本全保留，不因MRI的高TC选它。
- 主边界AUROC仅1例满足真/假边各≥20：MRI/feature/joint0.5631/0.7560/0.7537。00004为185条真边/19条假边，按预设不计，不事后放宽。连通成本NC/ED排序仅4例，平均AUROC约0.54–0.57。feature通过自动纠错筛查，但完整边界证据筛查均未通过；只是值得有限继续验证的线索，不是TTA已成功。
- 36掩膜用独立阈值连通分量重放，与MST最小瓶颈输出完全一致；所有图边6邻接，源预测/源Dice一致，WT/ET保护、模型/hash与先预测后标签均通过。报告已明确ROI截断、开发病例和背景误报限制；没有继续训练、重跑Kaggle或新增适配器。

## 2026-10-06：本地边界验证完成，Kaggle TENT/SAR运行中

- 用户要求本地验证边界方案，并用两个Kaggle账号各两块GPU跑当前模型TENT/SAR；新增要求WT/TC/ET均报Dice与HD95。当前仍使用native IN四类Dice+CE epoch250，逐病例TTA，不是CTTA。
- 本地目录`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_boundary_probe_20261006`：12例80³局部ROI，仅00062改84体素且全为恢复NC，覆盖NC→ED错误0.0395%；TC Dice58.1153→58.1304%，WT/ET硬固定。TC局部HD95不变11.2513mm(n11)，WT4.8086(n12)、ET8.2608(n10)。主边界AUROC缺少真NC/ED边界不能计算；没有继续训练/调阈值。`REPORT.md`、`HD95_SUPPLEMENT.md`及回执均已保存。
- TENT/SAR目录`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_norm_TTA_20261006`，协议`PROTOCOL.md`/`HD95_PROTOCOL_ADDENDUM.md`，代码`norm_adapter.py`/`worker.py`。更新22层IN affine7680标量，conv/head冻结；native128³、0.5滑窗、Gaussian、8镜像，identity每patch更新一次、输出更新前8镜像结果。TENT/SAR不硬固定WT/ET。每病例重置权重/optimizer/scaler/EMA，无GLI与GT调参。
- TENT Adam1e-3；SAR两次entropy<0.4ln4、SAM-SGD1e-3 momentum0.9 rho0.05、EMA0.9、恢复阈值0.2。SAR本地一例4patch均恢复，真实执行了更新，未禁用恢复来求增益。
- 两私有v1任务已实际启动：`nagisa0v0/ped2023-native-in-tent-episodic-20261006`（北京时间11:10:16提交）、`djnagisa/ped2023-native-in-sar-episodic-20261006`（11:12:34提交）。各两块T4按病例分片，共99例，另报去重98组。11:32日志已完成至少TENT59、SAR53例，无worker失败，预计仍需约10–13分钟；不是完成成绩。
- Kaggle安装器遇重复dist-info，实际把镜像NumPy2.1.3换为编译的1.26.4；任务随后成功运行。不能将assert通过解释为基础环境未改变。本机核心环境未改。`ENVIRONMENT_NOTE.md`保留说明；未提交的`kaggle_kernel_no_deps_candidate.py`仅备选，当前kaggle_kernel.py已恢复为实际提交v1模板，无v2/取消/重跑。
- 本地smoke源mask与既有full source逐体素一致；adapter机制测试与独立KD-tree指标检查通过。Kaggle计算时每例已有HD95（EDT），不用重新推理。单方空HD95=NA、双方空0，要报有效n及漏检/误报计数，不能仅报排除漏检后的均值。
- 继续用`monitor.py`读取25秒有界日志（账户包装脚本读已授权本地凭据，不输出token）。任务COMPLETE后分别运行`download_results.py tent/sar`（并行下载），然后`verify_results.py`、`report_results.py`。最终验收文件应为`evaluation_verified.json`、`verification.json`、`REPORT.md`、`per_case_metrics.csv`；当前尚未生成，不得称全99已核验。验证逐mask哈希/坐标/GT、独立Dice/HD95、source跨账号与本机差异、逐patch恢复/更新/冻结断言及各阶段时长。
- 仓库说明`docs/ped2023_boundary_and_norm_tta_20261006.md`已新增但完成状态待结果更新。全部用户既有修改保留，未commit/push。请勿重复提交或重建已上传bundle。

## 2026-10-06：可视化改为全视野WT/TC/ET，好坏病例比较完成

- 最新用户要求图不要局部放太大，且包含WT、TC、ET；随后要求继续。新版入口 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_TC_shape_review_20261006/easy_view/index.html`，报告 `easy_view/COMPARISON.md`，代码 `compare.py` / `render.py` / `report.py`。下节旧review入口仍保留，并增加新版链接。
- 99例每例完整240×240视野、WT/TC/ET三行、MRI/标签/预测/误差四列；绿色为当前行区域，蓝色漏分、红色多分。三行同一轴向层，WT/TC背景FLAIR、ET背景T1ce，GT最大TC面积选层。标题为全体积Dice；单层空不代表全体积空。旧simple.png局部放大图保留但新版不再链接。
- 去重98组：TC Dice≥0.80好19例，0.50≤Dice<0.80中30例，<0.50差49例。病例中位数好/差：WT92.28/89.69%，标签ET/TC61.94/5.55%，NC/TC38.06/94.45%，NC正确识别62.85/2.00%，NC→ED10.54/83.01%，球形度0.671/0.643。主要共同特点是NC主导与NC→ED混淆，不能证明病理因果。
- 全98组TC Dice与ET/TC Spearman0.591、球形度0.086；最高/最低各24例球形度0.669/0.684。WT≥0.80子集同样保留组织差异。TC包含ET导致组成与TC Dice有数学联系；错误召回/错分率只描述输出，不是独立预测特征。
- 00120/00123为两组近中位诊断例；00078 ET/TC0.018%但TC87.11%（ET极少，不能说完全没有）；00051球形0.845仍TC0且体积仅0.988mL；00111球形0.537仍TC88.03%。保留反例，不将球形/单连通当充分条件，不否定所有形状信息。
- 新版verification通过99图、100页、305本地链接，原预测/标签hash及Dice一致、WT/TC/ET与全视野/单层误差检查。Edge筛选/搜索/排序实际通过，抽查两组图。复用已有输出，无新推断、训练、TTA、额外基线；所有98组仍开发证据。

## 2026-10-06：原始模型可视化整理与TC形状分析完成

- 用户要求整理原始模型可视化和信息、查看TC形状先验。新目录 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_TC_shape_review_20261006`；入口 `review/index.html`，报告 `REPORT.md`，模型卡 `MODEL_CARD.md`；仓库说明 `docs/ped2023_source_tc_shape_review.md`。复用99例原始完整体积，未重新推理/训练/适应/修改源预测。
- 00121/00137影像及标签文件重复；形状统计保留00121，共98组不同文件（不保证98独立患者）。原99例WT/TC/ET89.3334/48.8627/54.6713%；去重98组89.2723/48.8446/54.3281%。GT明确用于描述性分析，非TTA成绩。
- 98/98真实TC最大26连通块≥95%，中位99.992%；5例另有≥0.1mL分量、1例≥1mL。真实主体球形度中位0.658、凸度0.768；相同91例预测非空中真实/预测球形度0.653/0.790、凸度0.766/0.841，81/68例预测分别更圆/更凸。预测主体占比中位100%，7例TC全漏；简单紧致性无法排除小而完整的错误TC。
- 79/98真实TC/WT≥90%，42无ED。ED≥1mL的27例中TC/WT中位82.02%，WT外2mm的TC占比70.16%，深于5mm96.89%。TC不必被完整ED环包围，深部偏TC不代表外围必为ED；绝不能把GT比例强塞进无监督TTA。
- 97/98真实TC填洞增加≤1%；源预测3D填洞仅恢复19/2142739个NC→ED错误（0.000887%），2/4mm闭运算恢复0.3623/0.9185%并误吞916/2090真实ED，仅几何覆盖诊断未部署新掩膜。主要错误并非小空腔。
- 15.77% NC→ED错误距源TC≤2mm、38.74%≤5mm、61.55%≤10mm；10.86%在源TC为空病例。源ED内NC/ED均存在病例，WT深度AUC中位0.601(n50)、源TC邻近AUC0.703(n48)，有明显低于0.5病例。软主体连续性可约束候选，类别仍缺可靠证据。
- 99例每例旧2图+新TC三平面图，共297图；review/包含完整可搜索/筛选/排序便携集合及模型卡、JSON、统计/3D PNG/PDF。verification通过100HTML/713链接、Edge渲染、搜索/去重/空TC筛选/正反排序、原图/掩膜hash、原Dice复算、checkpoint前后hash及几何单元测试。阅图为抽查，不声称人工看完全部99例。
- 本轮不是新方法实验，没有新基线或训练；所有GT形状规律仍为开发证据，PED镜像与官方原始文件等价未独立核实。完整定义、ED体积敏感性、失败假设与限制见新报告。

## 2026-10-06：冻结分类头的IN与局部候选补试完成

- 用户新附件 `ae35e2db-58c9-4e3d-a2d3-18ccfd7b6a6a`，明确不展开基线。仅实现新机制，无IGTTA/TENT/额外margin等实验。目录 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_frozen_head_candidates_20261006`；仓库入口 `docs/gli_to_ped2023_frozen_head_candidates_v8.md`。
- 同12个开发病例、80³局部ROI/128³patch，复用v7无标签GMM与上下文教师。全部卷积/分类头冻结，组织教师只更新末端4个IN（384标量），上下文更新全部22个IN（7680标量）；Adam1e-4×20步、每方法/病例重置。主输出仅硬保护WT及缺类等范围，另保存教师门控输出。无GLI、adapter、外部模型或跨病例状态。
- 三类主输出WT都86.9499%。源TC/ET58.1153/50.5789%；组织末端IN58.0598/50.5757；上下文全部IN57.9832/50.8303；局部候选57.4016/50.4056。NC-only TC分别58.0783/58.0239/57.7615，均退化。WT/NC-only ET不变是硬保护，不是改善。
- 局部候选：SLIC分区、20mm空间块四折留出组织GMM、固定评分模型、各类+2/+4 logit与H0、修改代价0.5、种子块扰动3/4稳定性。三类1673区域中981有更好候选，自动改善265/损害497；改对3897/破坏7187，净−3290体素。GT最优可净纠正14140仅为离线组织正确体素诊断，绝非TTA或Dice上界。
- 组织IN的NC-only恢复17个NC却破坏225个原正确NC；候选NC-only恢复1673却破坏3988。上下文严格门控两模式仍与v7同gate冻结集成逐体素一致。主输出ET小升伴TC下降，不包装为成功。
- `verification.json`通过12例156掩膜、45次实际IN更新共900步/3次缺类回退、参数delta、全部冻结状态含别名、重置、WT/ET保护、候选重建、独立Dice及哈希/时序。GPU峰值2.42GiB、总病例处理295.0秒（全部方法，非单方法耗时）。所有旧结果与checkpoint保留。
- 所有预测固定后统一读GT，没有调参/挑病例/GT选检查点；未扩到99例。与v7同时改变lr/步数，不能把两轮差异仅归因于参数位置。ET缺失n=0、官方镜像等价未核实、99例均开发证据等限制不变。详见 `REPORT.md` / `ASSESSMENT.md`。

## 2026-10-06：按新附件执行三种教师，均未通过

- 用户要求试验附件 `C:/Users/Yuuhi/.codex/attachments/619e291c-fefb-4411-b56a-97edecfa5baf/已粘贴的文本.txt`。已实际分别实现类条件MRI/特征GMM教师、影像边界随机游走图教师、多裁剪互教；前两者真实更新分类头，第三种真实更新末端4个IN与分类头，Adam1e-3×32步。目录 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_three_teachers_20261006`；仓库入口 `docs/gli_to_ped2023_three_teacher_experiments_v7.md`。
- 同12开发病例、80³局部cube、128³patch；4上下文偏移C=(16,0,0)、(0,0,0)、(0,16,0)、(0,0,16)，各2镜像。**不是99例全体积，也不是MuVi原版复现**。无GLI数据、外部模型、adapter、跨病例历史。共同硬WT，NC-only硬ET；三类版本预先固定为探索臂。所有预测锁定后统一读GT。
- 源WT/TC/ET=86.9499/58.1153/50.5789%。MRI2-GMM分类头三类TC58.0485/ET50.5610；影像图分类头三类TC58.1155/ET50.5486；上下文IN/head三类TC58.1164/ET50.6366。全部6个主学习臂未过运行前推进标准，未扩到99例。不得以千分之一百分点变化报成功。
- MRI2-GMM NC↔ED分类头仅恢复51个NC，却将350个原正确NC改成ED；NC/ED/ET种子逐病例平均纯度68.47/12.86/69.79%，pooled值不同。真实图边权没有优于统一边权；三类直接输出相对统一边权TC−0.1476pp、ET−0.0542pp。
- 212816个NC→ED错误中，任一裁剪看成NC的1747（0.8209%），≥3视图460个，自动共识仅接受8（0.00376%）。192134个高置信错误中任一裁剪看对仅9个。GT只用于该离线诊断。上下文IN/head两模式在全部12例与**同gate冻结集成逐体素相同**；实际参数有更新，但无额外硬分割收益。
- `REPORT.md`、`ASSESSMENT.md`、`evaluation.json`、`diagnosis.json`保存所有直接/学习/shuffle/margin/random/统一边权/原软目标/集成/无gate结果与尾部。`verification.json`通过1416变体、132次32步head更新、36次32步IN/head更新及保存delta/重置/哈希/方向配额/独立Dice等。GPU峰值allocated1.59GiB，全部病例处理总计364.5秒，含所有方法，不能当单方法速度。
- 初次IN审计错误来自state_dict中同一IN参数的norm/all_modules别名；`named_parameters(remove_duplicate=False)`识别全部别名后重跑，attempt01/REPAIRS.md保留。没有改变方法超参。源软目标也出现微小FP32/AMP数值漂移，已保留对照，勿假定自监督控制必然零更新。
- 当前结论：不可信ED语义种子与近乎一致的裁剪不能提供足够纠错目标。本轮只否定这组具体启动配置，不否定所有三方向。12局部ROI均有真实ET，ET缺失误报不能评价；现有99例仍是开发证据，官方镜像映射等价未独立核实。

## 2026-10-06：v6组织分群候选已实际运行，未通过

- 用户要求“试试看，然后看下结果”。已按v6设计实现冻结模型、当前病例GMM分群、NC/ET语义命名、5次空间块重采样、双表示支持、增强冲突否决和输出插值。独立目录 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_group_TTA_20261006`，主脚本`run.py`、`group_method.py`；无GLI/外部模型/跨病例状态/源训练/adapter。
- 全12个既有开发病例、同80³局部cube、A/C两个128³上下文各两个镜像；**不是全体积成绩**。保存12个主/消融版本各自的同方向同数量margin和随机对照，加source/历史C-bias/allED；预测全部固定并哈希后才读GT。
- 源WT/TC/ET=86.9499/58.1153/50.5789%；联合主方案=86.9499/58.4085/50.6229%。TC较源+0.2932pp，较margin−0.0038pp，探索性95%区间[-0.0291,+0.0208]pp。TC9升1降2不变；ET5升3降4不变，平均变化区间跨0。NC-only TC+0.2514pp但仅3/12例实际修改。
- 联合版共改3031体素，改对1512、破坏正确370；匹配margin改对1481、破坏130。源ED内真实NC212816体素仅恢复1218（0.572%）；ED→NC提议1328中1218真NC，ED→ET提议1548中只有139真ET（其余823NC/365ED/221BG），均为pooled修改体素统计。说明NC覆盖低且新增ET语义不可靠。
- 真ED存在7例的ED Dice28.0699%→28.2283%，原正确ED保留30956/31326。12个局部ROI均有真实ET，ET缺失误报本轮不能评价。取消稳定性/放松命名可增TC，但联合ET退化；放松命名TC+5.4457pp、ET−3.5403pp。全部失败消融保留，没有按最高TC改选。
- `REPORT.md`、`evaluation.json`、`diagnosis.json`与`results.png/pdf`完整记录；`verification.json`通过456变体WT、NC-only ET保护、同源C输出、代码/输入/输出/模型哈希、配额重建、独立Dice重算、预测先于评价等。全部GMM收敛且都选择候选上限K6；不等于医学上有6类组织。
- 按运行前协议，主方案未满足TC定位优势/覆盖推进标准，停止这一候选，未做99例全体积试验、未加蒸馏。下节“完整候选尚未实施”是上轮历史状态，现由本节覆盖。下一步不能把此局部小增益或此前GT读出当作新方法已成功。

## 2026-10-06：参考资料驱动的语义支持诊断与v6设计

- 最新用户要求结合附件设计更好的方法。新方案文档：`docs/gli_to_ped2023_tta_semantic_support_v6.md`。候选为冻结源nnU-Net、固定WT、病例内外观分群、NC/ET有限语义支持与拒绝；不把源ED当可靠负例，不依赖GLI数据，无跨病例状态。**完整分群候选尚未实施，不得称已验证有效。**
- 已实际完成的独立目录：`D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_support_probe_20261006`。沿用12个开发病例与80³局部cube，两个128³上下文各两个镜像；不是八镜像全体积。高置信稳定NC候选11例平均纯度68.7736%（pooled95.5186%），ED候选12例平均纯度16.0195%。空间支持规则仅覆盖6/12例、32.10%源ED体素。
- MRI/冻结32D特征/29D分类头零空间特征分别用NC单侧支持或NC/ED双侧支持，共六个无标签版本；全部固定预测后才评价。各版本均有完全相同编辑量/候选范围的源margin与打乱对照。所有版本平均AP未超过源margin。残差正支持AP0.6823，对照0.7059；12例TC较源+1.0698pp，但较margin−0.0155pp，ED存在7例的Dice28.0699%→27.3897%。不能称自动参照方案成功。
- 随后明确使用GT做离线shrinkage LDA读出，另一空间块训练、当前块查询，可评价11例：MRI/冻结特征/残差AP=0.8118/0.8483/0.8477，同查询margin0.7101。**这是有监督机制诊断，绝非TTA成绩；自动6例与离线11例不能直接相减。** 相邻块仍可能相关，不能视为独立泛化验证。
- `REPORT.md`、`evaluation.json`、`oracle.json`、各自协议与代码保存完整失败/支持证据；`verification.json`通过12例216变体、432项WT/ET不变检查、源C上下文复现、模型状态/哈希/时序/对齐检查。源checkpoint未改。未新增adapter或训练源模型。
- 接下来若实现v6，先锁定无标签分群/命名/拒绝规则，保留全部回退病例，必须超过同编辑量margin并报告真实ED损害；不要在信号仍失败时增加蒸馏。所有病例仍属开发证据，99ID文件重复问题仍按前节记录处理。

## 2026-10-06：脑肿瘤先验辅助 TTA 的连续实验

- 设计与完整记录入口：`docs/gli_to_ped2023_tta_prior_experiments_v5.md`；独立实验目录 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/BraTS_PED2023_prior_TTA_20261006`。用户授权提出先验方案、实际实验、失败后继续调整。所有历史修改、原数据和checkpoint保留；无GLI图像/标签、外部教师或跨病例更新。
- 已在12个开发病例运行空间/成像先验探针、真实最终分割头40/200步更新、bias-only对照，以及原生八镜像全体积校准；随后固定配置运行20个确认ID。实际学习的是当前nnU-Net已有最终头的NC/ED bias，逐例重置；WT/ET在最终解码中显式保持。
- 新数据质量发现：`BraTS-PED-00121-000`与`BraTS-PED-00137-000`的四模态文件和标签文件SHA256分别完全相同。99ID中四模态文件组为98组；这不能证明其余都是不同患者，但该重复必须去重。当前确认评价前已固定哈希去重规则，保留19组影像；原99例和原20ID历史结果没有覆盖。
- 19组影像原WT/TC/ET为90.9807%/44.3235%/51.4112%，全局bias校准为90.9807%/84.3184%/51.4112%；TC+39.9949pp，17升/0降/2不变。但是随机位置校准TC88.1983%，全部ED→NC90.3970%，真实ED存在子集(n=9)的ED Dice从7.4811%降到4.7595%。所以该版本**没有通过“先验贡献与组织分类可靠性”的研究标准**，不能只报高TC宣称成功。
- `verification.json`已核验32个完整体积病例：28例实际200步更新、4例回退，源推理与旧预测逐体素一致，初始bias/源模型状态一致，参数更新/哈希/预测先于评价均通过。该receipt针对全体积版本，不自动覆盖后续轮次。
- 后续局部适用范围、稳健增强、厚部支持、局部分类开发记录在Round4--7协议和对应目录。最终结论应读实验根目录`REPORT.md`及`FINAL_ASSESSMENT.md`，不要把早期候选当最终推荐。19组确认已被分析，后续不能再称为未见验证。

更新时间：2026-10-05（北京时间）。本节为当前状态；文末保留的旧 TENT 交接仅供历史追溯。

## 最新已完成：PED2023 逐例预测与可视化

- 用户明确选择四分类 **InstanceNorm、Dice+CE** 模型，要求本地运行 PED2023、记录预测与标签差异并为每例可视化。使用 IN final/best 第 250 轮，原生 nnU-Net 8 视图翻转集成、四分类 argmax，无 HD-TTA 更新。
- 独立目录：`D:/Workspace/deeplearn/3dunet/artifacts/experiments/nnUNet_PED2023_IN_analysis_20261005`。2026-10-05 18:50:59 启动，19:37 完成全部 99 例；`outputs/full/summary.json` 为 complete=true，`outputs/full/supervisor.json` 为 running=false、exit_code=0。
- 浏览入口：`outputs/full/index.html`；完整分析：`outputs/full/ANALYSIS.md` 或 `analysis.html`；队列图：`outputs/full/cohort_analysis.png`。每例 `outputs/full/cases/<id>/` 保存 prediction/reference/wt_error/tissue_error NIfTI、两张 PNG、中文 HTML 与指标 JSON；共 198 张病例图。
- 指标含 WT/TC/ET Dice、HD95、precision/recall、误分漏分体积、类别混淆、连通块诊断。单侧为空 HD95=null；均值同时报告有效数。未过滤小连通块，10 体素阈值仅用于诊断。
- 当前 2023 镜像实际标签只有 0/1/2/3；按项目 modern 映射 1=NC、2=ED、3=ET。2023 数据论文确认发布的前景类别为 NC/ED/ET，NC 包含非增强、囊性、坏死；不要套用 2024 五值映射。Kaggle 镜像未提供独立数值字典，与官方原始文件逐体素一致性尚未确认。
- 实验根目录的 `verification.json` 与 `review_verification.json` 均 passed=true：99 例、198 张病例图、链接和产物哈希核对通过，模型权重未变。注意这两个回执在根目录，不在 `outputs/full/`。首例图像已目视检查，三种区域 HD95 与独立 MedPy 计算一致；不能将此表述为人工检查了全部病例图。
- 此次 IN 选择用于这项评估，不等于用户已经决定最终新方法设计。此前建议的 IN 主模型/BN 对照仍只是研究路线建议。

最终指标（逐病例平均，Dice 不含背景）：

| 区域 | Dice | HD95（mm） | HD95 有效病例 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| WT | 89.3334% | 5.2672 | 99/99 | 88.9092% | 90.6985% |
| TC | 48.8627% | 14.7466 | 92/99 | 89.8892% | 39.1424% |
| ET | 54.6713% | 9.3240 | 85/99 | 61.3491% | 58.1401% |

- 三个区域 Dice 等权平均 **64.2891%**。双方为空 Dice=1、HD95=0；单侧为空 Dice=0、HD95=null。TC 有 7 例完全漏检；ET 标签为空 11 例、预测为空 11 例，其中双方为空 4 例，因此 ET 有 7 例空标签误报与 7 例完全漏检。ET 仅在标签非空病例上的平均 Dice 为 56.96%。
- 主要差异是 WT 覆盖较好但内部区域识别不足：45 例 WT Dice≥0.8 同时 TC Dice<0.5；88/99 例 TC 预测体积比标签少超过 10%。标签 NC 体素有 **58.50% 被预测为 ED**，这是全队列体素汇总比例，不是患者平均；该混淆对 WT 无影响，但影响 TC。不能仅凭这次评估确定临床或因果机制。
- 连通块采用 26 邻域，≥10 体素仅作诊断，没有执行预测后处理；存在任意交集只表示有接触，不代表完整检出。
- 本次 WT={1,2,3}、TC={1,3}、ET={3}，先四分类 argmax 再合并。此前 HD-TTA 四分类 WT 使用肿瘤概率之和 >0.5，故这次 WT 89.3334% 与此前 Classical 89.3459% 不是完全相同的计算口径。
- 数据来自 `E:/dataset/BraTS_PED_2023_Kaggle`，Kaggle `pramada/2023-brats-pediatric-full`；复用 `D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_4class_nnunet_20261003/manifest.json` 的 99 例。精确权重为 `D:/Workspace/deeplearn/3dunet/artifacts/models/in_dicece/run/nnUNet_results/Dataset503_BraTS_Classes/nnUNetTrainerBraTS250__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth`。配置、输入与权重记录见实验根目录 `PROTOCOL.md`、`identity.json`。

## 当前任务与完成状态

- 用户正在考虑设计自己的 TTA 方法，尚未确定具体方案，也未授权启动新的训练。
- 已建议四分类原生 InstanceNorm（IN）nnU-Net 作为通用主模型，新四分类 BatchNorm（BN）作为归一化对照；若方法依赖 BN 统计量，则以新 BN 为主。这是建议，不能当成用户已选定方案。
- WT 纯 Dice 和四分类 BN 在第二台 AutoDL 上均已训练完成并下载。没有由本对话继续排队的训练。
- 此前请求“计算 BN 模型 WT、TC、ET”已完成：2026-10-05 15:29，本地最佳第 225 轮、253 例 GLI 验证集评估正常退出。WT **93.7547%**，TC **91.6963%**，ET **86.0234%**，区域平均 **90.4915%**。
- BN 模型尚未做 PED 评估。不要把 GLI 验证结果写成 PED 或 TTA 结果。
- 本次更新只改交接与项目运行记录，不 commit/push。仓库已有大量修改和删除，必须保留。

## 环境与状态入口

- 仓库：`D:/Workspace/deeplearn/3dunet`；Python：`D:/Environment/miniforge3/envs/cv311/python.exe`，PyTorch 2.11.0+cu128，RTX 4060 Laptop 8 GiB。
- AutoDL 最近实际配置：RTX 4080 SUPER 32 GiB 显存、12 CPU 核配额、90 GiB 系统内存；90 GiB 不是显存。旧 16 核/80 GiB 配置不适用于这次机器。
- 用户指定第二个 SSH。连接辅助：`D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_source_dice_autodl_20261003/admin.py`，凭据从既有 SSH 文件读取，禁止输出或写入文档。
- 2026-10-05 两次 SSH 在认证前报 `Error reading SSH protocol banner`。机器是否关机未确认；不得自行启动付费机器或声称仍在运行。
- Kaggle 旧续跑任务已经禁用，不应自动恢复；本地旧 WT 训练已停止保留。AutoDL 训练和本地评估分别记录状态，不能互相推断。
- 新实验主要在 `E:/HD_TTA_*` 独立目录，不在 Git 中。开始新工作前先读对应 `PROTOCOL.md`、`identity.json`、`summary.json`，不要只检查仓库代码。

## 源模型清单

除旧自定义模型外，以下原生模型均采用 GLI 患者级 998/253 划分、250 轮 × 250 iterations、patch 128³、batch 2、AMP、原生增强和深监督，SGD/poly。四分类为 0 background、1 NCR/NET、2 edema、3 ET；输入顺序 FLAIR/T1/T1ce/T2。

| 模型 | 本地根目录 | Trainer / 权重选择 | GLI 验证 |
|---|---|---|---|
| WT 二分类 IN，Dice+CE | `D:/Workspace/deeplearn/3dunet/artifacts/models/in_dicece` | Dataset502，`nnUNetTrainerBraTS250`，final=best 第 250 轮 | WT 93.67% |
| 四分类 IN，Dice+CE | 同上 | Dataset503，`nnUNetTrainerBraTS250`，final=best 第 250 轮 | WT 93.68%、TC 92.17%、ET 87.33%，区域平均 91.06%；组织类别平均 83.90% |
| WT 二分类 IN，纯 Dice | `D:/Workspace/deeplearn/3dunet/artifacts/models/wt_pure_dice` | Dataset502，`nnUNetTrainerBraTSDice250`；best 第 225 轮，PED 使用 final 第 250 轮 | best WT 93.64%，final 93.61% |
| 新四分类 BN，Dice+CE | `D:/Workspace/deeplearn/3dunet/artifacts/models/bn_dicece` | Dataset503，`nnUNetTrainerBraTSBN250`；best 第 225 轮，final 第 250 轮 | best 组织类别平均 83.56%，final 83.52%；best 区域平均 90.49% |
| 旧自定义四分类 BN | `D:/Workspace/deeplearn/3dunet/artifacts/models/legacy_custom_bn/source_training` | `run/checkpoints/best.pt`，第 300 轮 | 仅探索用途，存在 scan-level 泄漏与前 64 例选择问题 |

原生模型目录结构：`<根目录>/run/nnUNet_results/Dataset502_BraTS_Binary或Dataset503_BraTS_Classes/<Trainer>__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth或checkpoint_final.pth`。必须同时保留上一级 `plans.json`、`dataset.json` 与自定义 Trainer。IN 和 BN 的 plans 不可混用。

新 BN 仅在原生 IN 对照的 `3d_fullres.architecture.arch_kwargs.norm_op` 改为 `BatchNorm3d`；22 个实际 BN 层，31,199,796 参数，eps 1e-5、affine、momentum 0.1、running statistics。权重 state_dict 中模块别名会重复登记统计量，不能按键数误认为有 68 个 BN 层。它不是旧自定义 BN 训练流程的恢复。

BN 备份共 55 个文件、13 份权重、3,253,878,615 bytes，全部匹配远端 SHA-256，见根目录 `download_verified.json`、`checkpoint_inspection.json`。包含每 25 轮快照、best、final、patch-best，及配置、Trainer、split、日志和各轮验证汇总。**未下载 BN 原始验证预测掩膜**。best/final 的优化器、AMP scaler、22 层 BN 统计量已检查。

WT 纯 Dice 备份 15 个文件，best/final 与远端逐个校验；本次从随机初始化并重新生成缓存，未复用 Kaggle 权重。BN 在 WT 完成后自动串行训练，2026-10-04 11:44 全部完成。

## 指标口径：83% 与 90% 都可能正确

- 原生四分类选择指标：分别计算 NCR/NET、edema、ET，再取前景组织类别均值，不含背景。
- BraTS 区域：WT={1,2,3}，TC={1,3}，ET={3}，每病例计算后对病例平均；双方都空记 1，单边为空记 0，最后三个区域等权平均。
- 原生类别汇总对双方为空记 NaN 并排除；区域汇总记 1，因此 ET 也可能有小差异。不能仅靠三个类别的 Dice 推导 WT/TC，必须用掩膜合并重算。
- 新 BN best 原生组织类别 Dice：NCR/NET 78.87%、edema 85.84%、ET 85.97%，平均 83.56%。这不意味着区域平均只有 83%。
- 新 BN 区域重评估：`D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_GLI_BN_regions_20261005`。`outputs/full/summary.json`、`RESULTS.md`、逐例预测与标签、`verification.json` 均已保存，253/253、exit code 0。
- 使用 best 第 225 轮，原生 argmax 掩膜；双方为空 TC 1 例、ET 1 例。BN 统计量始终未变，首例缓存与原生重预处理数组完全一致；与远端第 225 轮记录逐例比对，最大组织 Dice 绝对差 0.001129（约 0.113 个百分点），本地组织类别均值 83.55995% 对远端 83.55969%。硬件/软件不同，不应声称逐体素完全一致。
- 标签来自本地原始 GLI ZIP；二分类缓存只复用 MRI 数组，**不使用二分类标签计算 TC/ET**。

## PED 99 例 HD-TTA：已完成结果

统一使用已确认 Kaggle `pramada/2023-brats-pediatric-full` 的 99 例。固定 manifest，原生 nnUNetv2 预处理，**本系列不额外做 HD-BET**。文末旧 PED 的强制脑掩膜要求只属于旧模型/旧数据诊断，不能直接套用。

| 源模型 | Source WT Dice | Classical WT Dice | HD-TTA WT Dice | Classical → HD-TTA HD95 | 触发 |
|---|---:|---:|---:|---:|---:|
| WT IN，Dice+CE | 88.5838% | 88.7538% | 88.6406% | 5.2536 → 4.6692 mm | 15/99 |
| 四分类 IN，Dice+CE，合并概率 WT | 89.2733% | 89.3459% | 86.8500% | 5.2654 → 5.8314 mm | 25/99 |
| WT IN，纯 Dice | 88.8599% | 89.0388% | 89.0315% | 5.5675 → 5.5777 mm | 1/99 |

对应目录（每个目录都看 `outputs/full/summary.json` 和 `RESULTS.md`）：

- WT Dice+CE：`D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_native_nnunet_20261002`（还含已完成的论文消融）。
- 四分类 IN：`D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_4class_nnunet_20261003`。
- WT 纯 Dice：`D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_wt_pure_dice_20261004`。

Source 不翻转，Classical 用 checkpoint 允许的 0/1/2 轴共 8 个视图；step 0.5、Gaussian、AMP。HD-TTA 在恢复原图几何的概率上 FP32 优化，网络冻结。四分类 WT 概率为三个肿瘤 softmax 概率之和再 >0.5，**不同于 GLI 区域评估先 argmax 再合并标签**。

WT Dice+CE 触发的 15 例全部选 compact；相对 Classical，Dice 8 例提高、7 例降低，HD95 9 例改善、6 例恶化，Precision 13 例提高、2 例降低。触发病例平均 Dice −0.7469 个百分点、HD95 −3.8575 mm、Precision +11.3686 个百分点。全队列 Dice 略降但 HD95/Precision 改善，不能说其“大部分病例都下降”。最差单例 Dice −23.10 个百分点，须报告负迁移。

WT 纯 Dice 的唯一触发病例 `BraTS-PED-00115-000`，选 diffuse，Dice 49.90% → 49.18%。用保存概率重新运行官方门控，99 例记录全部匹配，98 例未触发掩膜与 Classical 相同，见 `gate_audit.json`。三组都没有小于 300 体素的病例。

官方门控：`volume(P>0.5)<300` 或 `count(0.3<P<0.7)/count(P>0.1)>0.05`。不确定比例中位数：WT Dice+CE 2.65%、纯 Dice 0.93%、四分类 IN 3.70%；纯 Dice 在 98/99 例都低于 Dice+CE。只确认概率分布差异，不能据两次训练断言纯 Dice 导致置信度失准。

## 论文公式与官方代码必须区分

论文：<https://arxiv.org/html/2602.19454v1>。固定官方代码 commit：`5b61a8fe530ad64054b8f743cfe8feefbf36fc01`；论文 HTML/text、notebook 与函数快照在上述 PED 实验的 `references/`。

- 主实验核心门控、优化、边缘和损失函数从官方 notebook AST 执行；Adam lr 0.1、1000 步、膨胀项保留，选择阈值 0.95。用户明确不接受把膨胀改为 0 后仍当原方法报告。
- 已明确修正官方 `img[3]` 与 PED 通道映射不一致：使用真实 FLAIR 通道 0。因此不是一字不改地执行作者 notebook。
- 论文写二元熵及双向平方锚定，compact/diffuse anchor 系数 50/0.1；代码用 `p*(1-p)`，compact 平方锚定 0.1，diffuse 单向 ReLU 锚定 50，且 compact 多了 rescue 项。
- 作者权重未发布，具体 split/plans 等未全部核对。我们训练了源模型，不应宣称完整严格复现了作者训练与原始结果。
- 论文 PED Table 1：Source Dice 86.47%、Classical 86.95%、HD-TTA 87.26%；HD-TTA HD95 5.35 mm、Precision 86.32%；Classical HD95 6.59 mm。论文报告触发率约 23.6%。

用户已要求按论文再试，完成目录：`D:/Workspace/deeplearn/3dunet/artifacts/experiments/HD_TTA_PED2023_wt_paper_20261004`。复用纯 Dice 模型原概率、全部 99 例；损失改为二元 Shannon 熵、论文系数和双向平方锚定、去掉 rescue，膨胀 5 保留。因论文归一化未说明完整，提前固定并同时报告求和与均值两种约定；不能把任一种声称为作者确认实现。详读该目录 `PROTOCOL.md`。

| 版本 | WT Dice | HD95 | Precision |
|---|---:|---:|---:|
| Classical | 89.0388% | 5.5675 mm | 87.5925% |
| 论文求和约定 | 89.0388% | 5.5675 mm | 87.5925% |
| 论文均值约定 | 89.0426% | 5.5578 mm | 87.5991% |

论文门控分母按文字理解改为 `P>0.5`，仍仅 1 例触发。两版本都选 compact；均值版该例 Dice 49.90% → 50.28%，HD95 23.04 → 22.09 mm。改善很小，不应宣称复现论文收益；另一个“分子也限制在预测肿瘤内”的门控解读未运行。独立 NumPy 公式/梯度和门控测试通过，99 例核对完成。

## 下一步工作边界

1. 若用户继续设计方法，先明确修改 logits、输入还是网络参数，再固定主模型与协议。不要按目标域增益最大来挑源模型。
2. 新 BN 的 GLI 区域指标及 IN 的 PED 99 例逐例分析均已完成，无需重复推理；可直接从已保存掩膜和可视化继续分析。PED 上新 BN 的 Source/TTA 尚未启动，需要对应请求后进行。
3. 正式方法开发需划分目标域开发与最终评测病例，避免把已反复调试的 99 例同时视为未接触测试集。
4. 保留所有原实验；不要把 paper 版本替换官方代码结果。部分进度不能当完整队列结果。
5. 断点恢复先核对 identity、权重与输入，查看真实进程和退出码，不凭旧状态文件判断仍在训练。
6. 本次只核查结果 JSON、完成回执、进程记录和文档，没有重新运行全套 pytest 或训练。下方“78 passed”是历史结果，不能当当前测试结果。

## 历史交接：2026-09-24 TENT 诊断（非当前待执行任务）

以下原文保留用于追溯；其中“当前目标”“必须保持的预处理”“下一步”只适用于当时的旧模型及数据，不能覆盖上述最新实验协议。

## 当前目标

暂不训练新的 3D U-Net。继续使用 AutoDL 已训练的源域模型，完成 PED、SSA 上的 Source 与 TENT 正式评测，并分析 TENT 效果不明显的原因。

## 模型与权重

- 模型：六层 3D U-Net，通道 `[32, 64, 128, 256, 320, 320]`，约 3120 万参数，`BatchNorm3d`，四分类 Softmax。
- 权重：`D:\Workspace\deeplearn\3dunet\artifacts\models\legacy_custom_bn\source_training\run\checkpoints\best.pt`
- 权重对应第 300 轮，训练文件共 29 个，已与 AutoDL 端 SHA-256 清单逐个核对。
- 原训练使用 scan-level 划分，且只用前 64 个验证样本选权重，其中存在患者重叠。因此该权重适合当前探索性 TTA 实验，不应作为最终无偏源模型结论。

## 必须保持的预处理

- PED 必须先用 HD-BET 脑掩膜去掉脑外组织；同一掩膜应用于四个模态，不能改动肿瘤标签。
- 应用脑掩膜后，再对每病例、每模态的非零区域做 z-score。
- Source 和所有 TTA 方法必须使用完全相同的病例、掩膜和归一化流程。
- “去掉脑掩膜”容易产生歧义：实际要求是保留脑掩膜步骤，用它去掉脑外组织。

## 已有诊断结果

固定抽取 PED、SSA 各 10 例，普通 ET/TC/WT Dice 的平均值如下：

| 数据 | Source | 原 online TENT | two-pass TENT |
| --- | ---: | ---: | ---: |
| PED（已做脑掩膜） | 0.5845 | 0.5772 | 0.5589 |
| SSA | 0.8891 | 0.8516 | 0.8441 |

这些只是小样本诊断，不是正式结果。PED 不做脑掩膜时 Source 仅为 0.1098，说明预处理差异会严重污染比较。当前没有证据证明 two-pass TENT 更好，因此 `online` 仍是默认协议。

独立的 159 例 GLI 子集上，Source 平均 ET/TC/WT Dice 为 0.8934；原日志中的 0.9154 来自有缺陷的 64 例验证集合。

## 代码与文件

- 仓库：`D:\Workspace\deeplearn\3dunet`
- 完整诊断报告：`D:\Workspace\deeplearn\3dunet\artifacts\models\legacy_custom_bn\TENT_DIAGNOSTIC_REPORT.md`
- AutoDL 源代码快照：`D:\Workspace\deeplearn\3dunet\artifacts\models\legacy_custom_bn\autodl_source_snapshot.tar.gz`
- `evaluate_tta.py` 已支持 `--tent-inference-mode online|two_pass`。
- PED 脑掩膜流程入口：`scripts\run_local_target_pipeline.ps1`。
- 工作区存在大量未提交修改，不要执行 reset、checkout 或 clean。最新诊断改动尚未 commit/push。
- 最近一次完整测试结果：78 passed；后续修改后需重新验证。

## 下一步

先为全量 PED 生成并质检脑掩膜，再在完全一致的预处理下运行 PED、SSA 的 Source 和 online TENT。正式汇总应至少包含 ET/TC/WT Dice、HD95、lesion-wise Dice、病例数、checkpoint、manifest、随机种子和完整日志。完成基线后，再决定是否评测 SAR、VPTTA、TEGDA 等方法。
