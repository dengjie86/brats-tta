# 历史交接记录（整理前）

2026-10-15整理前的历史快照。文中的“当前”“运行中”“下一步”属于记录当时，不能作为执行指令。当前状态只读[根目录HANDOFF](../../../HANDOFF.md)。链接已调整为归档位置；逐字原文保存在[维护备份](../../../artifacts/maintenance/navigation_20261015/before.zip)。

---

# 当前交接

更新时间：2026-10-15 10:43后（新增跨Source队列已启动并通过精确预检，北京时间）。本文件是当前交接的唯一入口，后续直接在此更新。先读当前状态和方法说明；历史PID、运行中及待定判断仅对应其记录时点。

## 用户目标和偏好

- 研究 GLI→PED 测试时适应，希望方法有清楚的理论动机和可讲述的机制，Dice/HD95 相对 Source 有实际价值。小幅 WT 代价可以接受，不要擅自增加“所有区域所有指标必须显著提升”的门槛。
- 当前最明显的收益在 TC，希望进一步改善 WT/ET，同时检验 IN 依赖、BN 可用性和创新性。本轮开发主方法现已确定，独立泛化与论文创新性仍未证明。
- 用户明确反对把大量 token 花在实验中途监督：禁止逐病例模型轮询、反复复述无变化进度、重复读取完整交接或为了等待而增加无关审计。
- 希望全自动，偏好桌面端。新聊天已明确通过 `/goal` 要求继续改善GLI→PED TTA，允许更换切入点，并再次要求减少模型轮询。本聊天曾由工具确认Goal active；当前状态只按实际Goal工具记录，不由交接推定。

## 当前运行状态

- **已完成BN两组的即时Dice读数通过**：[阶段报告](../../../outputs/ped2023_cross_model_transfer_20261015/interim_readout/REPORT.md)，`result.json`保存720项混淆矩阵/直接区域计数交叉核验。各24例，BN225 main−Source WT/TC/ET=−.02271/+2.37210/−.08118pp，TC CI[+.20136,+5.49061]；BN250=−.10188/+1.08397/−.05765pp，TC CI[−.18711,+3.32268]。main−no_learning TC分别+2.34865/+1.06468pp；新teacher单独未显示稳定TC收益。此补充按用户即时结果请求提前读取已完成BN，未读IN200成绩或修改主队列；HD95和完整状态/输出复核仍待主队列。非Goal持续等待已停止，后台实验继续；不要据阶段Dice宣称通用性或全面改善。

- 用户新增要求：非Goal模式不用一直等待。工具等待cell252已终止，实验队列未终止；queue20040及IN200 child49564仍正常运行。BN225于11:55、BN250于13:17各24例预测完成。应用户询问已有结果，新增`cross_model_transfer/interim_readout/`独立只读汇总这两个已结束模型的Dice；不读IN200成绩，不改主队列参数。主队列仍自动完成IN200后做原完整评价。不要再次建立持续等待或启动第二份队列。

- 跨Source预检10:41:13北京通过，原IN25000003首梯度、affine/state、raw logits和主mask逐值一致，10:41:14预检exit0。队列已自动进入BN225 child76248，前2例完成且Source mask复现旧结果；没有新性能评分。queue20040/session69062继续，勿重复执行，后续等待整队列退出。

- **新增用户任务：跨Source检查已于10月15日10:38:25北京启动**：[固定协议](../../../outputs/ped2023_cross_model_transfer_20261015/PROTOCOL.md)。用户明确要求在其他几个模型上跑；选择native BN225/BN250/IN200，同原24组，各自对Source。它们是两个训练分支的checkpoint，不是独立架构。CPU四权重加载/身份/IN选择一致及BN冻结统计梯度检查通过；queue20040/session69062、preflight57296启动，10:38:54 GPU99%/4137MiB确认。先IN250/00003完整精确复现，再自动72个新模型-病例组合预测、360mask/1080独立区域和报告。当前无新跨模型成绩。BN只更新affine、Source running statistics保持，其他算法规则冻结，不调参/自动扩98。新任务是普通会话工作，原Goal仍为complete；未创建新Goal或调度。禁止重复prepare/execute，保留冻结代码和结果，等待进程退出而非逐病例轮询。

- Goal工具现已返回`complete`：本轮主方法选择、必要对照和可复现交付均已完成。此状态不代表独立泛化、BN迁移或论文创新性已得到证明。

- **主方法与本轮研究交付已完成**：[方法、结果与复现入口](../../GLI_PED_TTA_METHOD.md)。固定union2补充teacher + soft_extent八步IN学习 + no_come输出，删除COME分支和冗余末端stable，保留margin/presence。不按病例选分支，不再追加规则或参数搜索。
- **最终对照09:40:25完成**：[解释](../../../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/INTERPRETATION.md)。98主mask恒等/196控制mask/588独立区域、4阶段exit0/7终态hash通过，54代码/66记录冻结。相同输出流程下学习TC+3.00925pp，新teacher进入学习再增.29916pp；八家族既有基线分开IN/BN、98/24及预算限制汇总。queue34516/session19747/cell215已终态。
- **缓存交付09:50:25完成**：[回执](../../../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/delivery/replay_check.json)。98份主mask逐值一致，不读COME/GT、不调用模型/CUDA；session88450 exit0。所有实验与交付队列结束，无剩余科学阶段。脚本是冻结缓存重放，不是新MRI一键部署入口。
- 本轮完成了路线选择、组件删减、完整98组改进和必要学习对照；不是把单批实验结束当成整体完成。Goal状态以实际工具为准。全部98组开发暴露；WT/ET稳定收益、BN迁移、独立泛化和创新性未确立。未创建Scheduled或启用CLI原型，冻结产物不得覆盖或重跑。

## 本轮阶段记录（历史）

以下按原阶段保留，当时的运行中、下一步和未定方法已被上方完成状态取代。

- **主路线已选为union2+soft_extent+no_come；最终必要对照正在运行，Goal active**：[协议](../../../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/final_controls/PROTOCOL.md)。09:12:42北京queue34516/session19747启动，cv311；首例在不读取COME预测的解码调用中逐值复现主mask，两控制已输出。CPU对照同新teacher下的旧学习器与Source不学习，自动196mask/588独立区域后汇总已完成98/24基线，BN/IN分开、历史TENT/SAR配自身Source。代码编译及输入冻结通过，无新GPU/规则/阈值；只补主方法证据和最终文档，不再方法搜索。禁止重复prepare/execute或改冻结文件。
- **输出删减09:06:07完成并作出主路线选择**：[解释](../../../outputs/ped2023_teacher_complement_full98_20261015/decoder_pruning/INTERPRETATION.md)。196旧mask复现/196独立新mask/588区域/196实际末端保护恒等，3阶段exit0及5终态hash通过，52代码/44记录冻结；queue37676/session62447/cell203终态。选择no_come：删COME及冗余stable，保留margin+presence；主Dice89.14027/58.20171/54.46499，对Source−.13202/+9.35708/+.13691pp、HD95−.57316/−2.68657/−.16936mm(n98/91/84)。对旧soft_stable TC+.96119 CI[+.49263,+1.51049]，去最大收益仍+.84011；WT−.20461pp且HD95+.15206mm，必须保留代价。删presence会新增00021/00055两例ET全漏，保留其明确作用。主TC/ET全漏2/5，无GT ET FP4955。WT/ET均值及距离尾部仍有限制，不称全面稳定/独立泛化或新机制已证实。
- **union2完整98已于08:46:35完成**：[解释](../../../outputs/ped2023_teacher_complement_full98_20261015/INTERPRETATION.md)。复用48，新增50全部有建议并实际学习；450完整前向/27216调用，50状态/300mask/900区域/400输入复核通过，5阶段exit0及7终态hash通过，49代码/37记录冻结。queue45988/session53830/cell192终态。完整对旧soft_stable TC+.78169pp CI[+.34776,+1.30791]，去最大收益仍+.66012；同teacher仅输出上的学习贡献+.29444 CI[+.05530,+.63726]，去最大收益+.16784。完整对Source Dice+.05544/+9.17759/+.19803pp、HD95−.74485/−2.70641/−.25574mm(n98/91/84)。简单spatial对Source Dice−.15609/+9.80229/+.25337pp，ET HD95+.46645受00032尾部影响。不能把WT/ET均值当稳定全面优势；confidence学习仅48。正在用上方删除对照确定较短主输出，不再加规则。所有98开发暴露，无独立泛化/BN/创新性定论。
- **补充teacher的48组学习05:00:41完成**：[解释](../../../outputs/ped2023_teacher_complement_learning_20261015/INTERPRETATION.md)。96状态/384mask/1152区域、3阶段exit0及6终态hash通过，47代码/30记录冻结；queue50728/predict76144/session96161/cell180终态。union2完整对旧soft_stable Dice−.02290/+.56553/+.07353pp，HD95−.00807/−.09617/−.03247mm(n48/46/42)；TC CI[+.07381,+1.24321]，去00097后+.31002。old24/new24 TC−.03233/+1.16340，收益非两组一致。同teacher仅输出对照上学习贡献+.36270pp但CI跨零，去00097后+.10287；对confidence TC+.11258也跨零，不能宣称两票优势。WT/ET尾部与误报仍有限制。依据总TC收益及小WT成本决定上方固定主候选补齐50，不是自动扩展或定稿。
- **固定旧学习器接入00:05:47完成**：[解释](../../../outputs/ped2023_modality_context_20261014/proposal_union/decoder_transfer/INTERPRETATION.md)。96旧mask复现/192独立新mask/576区域、3阶段exit0及5终态hash通过，45代码/21记录冻结。union2对匹配旧spatial/stable TC+.18309/+.20284pp，WT−.05349/−.03877；去最大收益00002后TC仅+.05574/+.06720。新建议未完全被旧学习覆盖，但增益小且有WT/ET取舍，不能定最终方法。confidence两流程TC+.14530/+.16765。TC/ET全漏仍2/3。session61101/queue54192终态；作为新学习的同teacher output-only对照，解释及结果已冻结。工具内等待退出返回终态，无逐病例模型轮询，无Scheduled。
- **建议补充23:47:20完成**：[解释](../../../outputs/ped2023_modality_context_20261014/proposal_union/INTERPRETATION.md)。192mask/576区域、旧编辑保留、同转移方向数量控制及3阶段/5终态hash通过，43代码/13记录冻结。union2对旧teacher Dice−.15245/+.41769/+.08567pp，HD95+.15782/−.07668/+.12620mm(n48/46/41)；TC CI[+.13348,+.78902]，old24/new24+.30865/+.52673，去最大收益仍+.28615。confidence也TC+.40637，union2对其仅+.01132且CI跨零，无两票独有优势。ET均值尾部敏感，TC/ET全漏仍2/4。session29215/queue56768终态；工具内等待进程退出后直接返回，无逐病例模型轮询，但未建立桌面Scheduled。解释已由子研究冻结。
- **分模态背景观察22:26:14已完成并保留负结果**：[解释](../../../outputs/ped2023_modality_context_20261014/INTERPRETATION.md)。同48组528完整推断/30976网络调用、144mask/480输入hash重建/432独立区域通过，41代码/6记录冻结，3阶段exit0及终态hash通过；queue63876/predict70948已结束。均值teacher对旧teacher Dice+.17066/−2.49656/−.06470pp、HD95+1.03776/+.70873/+.52374mm(n48/45/41)，TC CI[−4.42732,−.86957]；全一致性TC−2.90928pp，精确平均输入−.75327pp。三臂不扩展。GT事后缓存诊断显示均值撤回74999个旧正确编辑，但单观察有额外正确候选；这仅是候选覆盖，不是可部署增益，促成上方独立补充对照。父解释/诊断已被子研究冻结，勿修改。
- loss checkpoint已在20:53:57北京完成；queue71188/predict37352历史，不再运行。未将阶段结束标作整个研究完成。
- **loss checkpoint保留为取舍消融，不扩98**：[解释](../../../outputs/ped2023_loss_checkpoint_20261014/INTERPRETATION.md)。57实际前缀精确重放及39状态复用，96状态/384mask/1152独立区域通过；3阶段exit0及日志、pipeline/6文件hash通过，44代码/15记录冻结。同48组best8/short4相对原spatial Dice+.01718/−.41317/+.04578及+.05822/−1.87931/+.33343pp；TC HD95+.12369/+.44282mm(n46)，固定共同45仍+.11243/+.37490mm。best8 TC CI跨零，short4明确损失；WT/ET小收益真实存在，但ET距离主要来自00032，无ET GT误报4369→5162/5823。不是训练目标选回优于固定8步的证据。
- 当前转向teacher信息；KL、IN分区、loss选回未改善主要TC收益，不继续扫loss/步数或根据病例追加gate。已读旧局部MRI/feature原型、GMM/图教师、分类头、小裁剪、跨序列频谱及联合观察负结果；本轮只改变保护区外模态背景组合，保留平均输入强度对照，不把模态缺失/相似度当作语义证据。主候选仍soft_stable与较简单spatial_context，最终方法未确定。
- **概率KL已完成并停止扩展**：[解释](../../../outputs/ped2023_probability_distillation_20261014/INTERPRETATION.md)。18:21:58北京完成48状态/192mask/576独立区域，3阶段exit0、pipeline及6文件hash通过；42代码/10记录冻结，queue63128历史。相同spatial输出对MSE Dice+.02154/−1.12042/−.19765pp，paired HD95−.00531/+.23746/−.00042mm(n48/46/42)，共同TC45仍+.23909mm。TC CI[−2.45246,−.27466]，old24/new24−.61417/−1.62666，去00135后仍−.58679pp；保留负对照，不扩98或扫KL参数。47例系数各步固定不代表参数梯度匹配；1例无编辑。
- loss checkpoint协议：[协议](../../../outputs/ped2023_loss_checkpoint_20261014/PROTOCOL.md)。同48组，回到原MSE，仅比较固定4步short4及0…8步total loss最小状态best8（相同值取最早）。旧曲线11/47例第8步高于此前最小值是动机，不是性能。冻结逐病例步数，无GT选择；39状态引用复用、57必要前缀学习，调用原冻结run_case且前缀loss/首梯度等须精确一致。现已完成，结果见顶部；不重复prepare/execute。best8部署仍需完整8步，不能宣称节省到选中步数；short4才减少更新。
- 用户16:35询问时KL完成14/48，按实际平均修正预计18:15–18:40，实际18:21:58完成；一次110分钟可中断等待后18:27读取终态，无中途轮询。不是完成事件自动唤醒，未建立Scheduled或CLI调度。
- 用户质疑方法是否确定后，已明确回答尚未定稿；10月14日用户要求“那就继续”。已由工具重新建立 **active** Goal。此前将阶段闭环视为整体Goal完成过早；今后不以单批实验结束代替研究目标完成。
- **组件删除队列已完成**：[结果解释](../../../outputs/ped2023_decoder_ablation_20261014/INTERPRETATION.md)，98组×10臂08:39:28北京完成，约51分钟。98旧mask逐值复现、980新mask独立重建、2940区域覆盖（2466独立EDT/474相同区域复用）通过，5终态哈希通过；15代码/6记录冻结。无模型forward/优化，PID34840历史，禁止重复prepare/execute。
- 简化端点spatial_context（无COME/margin/presence/stable）vs完整soft_stable Dice−.24711/+.63307/−.03135pp，paired HD95+.24368/−.26728/+.60251mm(n98/96/86)；固定20方法共同TC n90仍−.26407mm。vs Source Dice−.17452/+9.02897/+.08160pp，HD95−.48154/−2.75544/+.45229mm(n98/91/84)。保留较简单并行开发候选，小WT代价可接受但ET距离尾部仍待解决，未选定最终方法。
- 删除context学习TC−3.05772pp，删除输出teacher复制TC−1.37716pp；学习和teacher仍值得保留。移除margin TC+.50819pp，但ET尾部恶化。00032/00104为简化后的最大ET距离代价，不能按病例/类别追加规则。
- **原生视图可靠性完成并停止该门控方向**：[解释](../../../outputs/ped2023_context_view_reliability_20261014/INTERPRETATION.md)。10:42:48北京完成，98组392状态/23360网络调用逐值重现旧概率，1980两种AUC核对、3终态哈希通过。26代码/8记录冻结，PID64532历史，不重复执行。主50%匹配数量下context view相对confidence的病例建议正确率−.53822pp（CI跨零），全spatial变化−.30542pp（CI跨零）；student view也未建立超过student probability的独特信息，不新增镜像gate。不是新分割成绩。
- **IN自然分区完成**：[解释](../../../outputs/ped2023_in_subspace_20261014/INTERPRETATION.md)，正式终态在attempt02，15:35:07北京完成；96状态/384mask/1152独立区域，3阶段exit0及pipeline/6文件hash通过。41代码/8记录、修复2/4冻结，queue37824历史，不重复执行。48组同spatial输出，encoder/decoder TC−1.40386/−2.51475pp，CI均不跨零；WT+.02686/+.07782，ET+.10257/+.44758pp。encoder ET距离改善依赖00032；decoder无ET GT误报4369→8059。保留消融，不扩98或枚举层组合。
- 分区原队列63816在第一次forward前因prepare与旧run_case重复创建目录而exit1，原日志/空目录/冻结代码保留。已在[attempt02](../../../outputs/ped2023_in_subspace_20261014/attempt02/REPAIR.md)仅修复ROOT与控制器路径，96目录隔离检查通过、相同科学函数未改，独立队列已启动；以后读attempt02/launch.json及终态，不读根目录失败回执判定新队列。禁止重复原prepare/execute或覆盖旧失败。
- 分区attempt02启动检查通过：11:09:44北京完成00003全网soft8步逐值复现（97.43s）；首例00002 encoder/decoder各8次更新，4480/3200标量每步实际变化，首梯度均与旧全网对应子向量bitwise相等，max error0。耗时96.85/82.91s；这是工程证据，不是性能。queue37824当时存活，后续按命令身份重新核实，禁止逐病例轮询。
- 概率KL启动协议历史：[协议](../../../outputs/ped2023_probability_distillation_20261014/PROTOCOL.md)。同48组/全网7680 IN/teacher/soft目标/8Adam.001，仅编辑MSE换KL(Q||P)，逐病例lambda使Source初始融合logit监督梯度范数匹配旧MSE，之后固定；不保证参数梯度/Adam步长匹配。CPU闭式adjoint/24有限差分/自目标零梯度/范数/6FG置换通过；复用当日00003旧8步回执，不重复GPU审计。现已完成，结果见顶部；禁止重复prepare/execute或改冻结依赖。
- 本轮一次长等待无中途轮询，实际比保守预估提前结束约56分钟；用户询问时检查终态，已说明没有完成事件自动唤醒。后续据实测时长缩短一次等待，仍不频繁轮询。
- 旧[soft完整98](../../../outputs/ped2023_extent_factor_full98_20261013/INTERPRETATION.md)10月13日20:17:06北京完成；[固定解码学习器替换](../../../outputs/ped2023_extent_factor_full98_20261013/decoder_transfer/INTERPRETATION.md)21:14:01完成。queue42280/child3076/CPU70028均为历史身份，不再监控或重复prepare/execute。
- 父队列：旧24复用、新74状态/296mask/296独立重建/888区域EDT通过，Source reset，6终态哈希/4阶段exit0核对通过。32代码/18记录冻结。
- 子队列：旧完整stable流水线98mask逐值匹配，新98mask/294区域独立复核，5终态哈希通过。12代码/8记录/98组输入映射冻结。唯一更换学习raw概率，未新增输出规则、模型forward或优化。
- 优先继续验证的开发候选为 **soft_stable**：相对旧stable TC Dice+0.46236pp（描述性CI[+.24083,+.72910]），75例改善/18下降/5同，去最大收益仍+.39670；TC配对HD95−.32038mm(n96)，所有十方法共同n90仍−.08414mm。WT/ET Dice−.00989/−.00185pp，均值很小但不证明等效或改善。
- 原24/new74的TC相对旧stable分别+.57982/+.42426pp；全漏TC2/ET5病例集合不变，无ET GT的FP体素6524→6363。全部开发暴露，仍无BN、独立泛化或新颖性结论；流程依旧复杂。
- 两次一次性可中断长等待（16:30→20:45；20:55→21:35）均没有中途模型轮询；普通Python自动接续阶段。未创建Scheduled、未启用CLI调度器，未宣称桌面完成事件唤醒。未来长任务沿用低频/终态检查，不能为等待反复审计。

## 固定协议与环境

- 仓库：`D:\Workspace\deeplearn\3dunet`；Python：`D:\Environment\miniforge3\envs\cv311\python.exe`，已确认3.11.15。GPU是8 GiB RTX4060 Laptop；不改环境或核心依赖。
- 主 Source：native四分类nnU-Net，IN、Dice+CE、GLI epoch250；BN best225为独立归一化对照。权重位置及身份看[MODEL_INDEX](../../MODEL_INDEX.md)，资源看 `configs/paths.yaml`。
- PED2023采用native预处理，不追加HD-BET；通道 `t2f,t1n,t1c,t2w`，标签BG=0、NC=1、ED=2、ET=3。argmax后合并WT={1,2,3}、TC={1,3}、ET={3}。
- 98个唯一MRI/标签组，99条记录包含121/137重复组；固定24例是开发子集，全部GT含ET。24例与98组均值不能混用。
- both-empty：Dice=1、HD95=0；one-empty：Dice=0、HD95=null。HD95用双方共同有效病例配对比较并报告n，检查空预测和尾部病例。
- 所有当前队列均开发暴露。2023/2024存在大量MRI重叠，不能仅按年份宣称独立测试；详见[跨版本审计](../../../artifacts/maintenance/source_moment_replay_20261013/target_cross_version_identity/REPORT.md)。

## 当前主方法及98组结果

固定方法为union2补充teacher + soft_extent IN学习 + no_come输出。保留原teacher编辑，只补充旧teacher与Source一致位置上的无冲突类别建议；8步IN学习后使用范围池化、margin筛选及类别存在保护。COME及冗余末端stable已删除。完整算法及复现命令见[方法文档](../../GLI_PED_TTA_METHOD.md)。

| 98组开发结果 | WT | TC | ET |
|---|---:|---:|---:|
| Source Dice% | 89.27230 | 48.84463 | 54.32808 |
| 旧soft_stable Dice% | 89.34488 | 57.24052 | 54.44104 |
| 主方法 Dice% | 89.14027 | 58.20171 | 54.46499 |
| 主方法−Source Dice，pp | −.13202 | +9.35708 | +.13691 |
| 主方法−Source配对HD95，mm | −.57316 | −2.68657 | −.16936 |
| HD95共同有效n | 98 | 91 | 84 |

对旧soft_stable TC+.96119pp，描述性95%CI[+.49263,+1.51049]，去最大收益仍+.84011；WT−.20461pp、HD95+.15206mm，保留这一成本。TC/ET全漏2/5，无GT ET误报4955体素。WT/ET平均收益和距离受尾部影响；TC收益不能外推为全面稳定优势。

完整union2_stable和纯union2_spatial保留为WT/距离与TC取舍参照，不按病例切换。配对区间均为开发队列的描述，未作多重比较校正，无BN成功迁移、独立泛化或创新性定论。

## 前序24例：extent-factor学习及完整解读均完成

实验目录：[ped2023_extent_factor_learning_20261013](../../../outputs/ped2023_extent_factor_learning_20261013/README.md)。主preparation冻结23代码/配置、162记录，SHA256为 `3507934d5b1d3a2c75461a2fd6781424ea834a32ac34e2ca1495efb5b1830c13`。不要重跑prepare/execute或修改冻结文件。

`soft_extent`将Source与student的max-FG/BG竞争等权混合；`hard_extent`在最终输出锁定Source WT。三种前景组织之间的分数差保留，相同变换也用于teacher目标和学习。native IN/full Jacobian、原teacher编辑域、8步Adam lr=.001及7680个IN affine标量不变，没有TC专属门控。

`plain_soft/plain_hard`是恢复原plain状态后只在推断阶段使用相同变换的匹配对照；`*_raw`是新学习状态的未投影预测。应分别区分学习收益和输出保护收益。

从已生成的[REPORT](../../../outputs/ped2023_extent_factor_learning_20261013/REPORT.md)读取的combined24均值：

| 方法 | WT Dice% | TC Dice% | ET Dice% |
|---|---:|---:|---:|
| Source | 89.15667 | 59.41157 | 61.40770 |
| plain | 88.77975 | 63.23152 | 61.88256 |
| plain_soft | 89.08293 | 63.24909 | 61.87188 |
| plain_hard | 89.15667 | 63.22651 | 61.85371 |
| soft_extent | 89.10959 | 63.99995 | 61.79189 |
| hard_extent | 89.15667 | 64.17969 | 61.65171 |
| soft_extent_raw | 88.86005 | 64.00368 | 61.79171 |
| hard_extent_raw | 89.02343 | 64.24405 | 61.55364 |

| 方法−对照 | Dice差pp，WT/TC/ET | 配对HD95差mm，WT/TC/ET | HD95 n |
|---|---|---|---|
| soft_extent−Source | -0.04707/+4.58838/+0.38419 | -2.27985/-1.02494/-0.21381 | 24/23/22 |
| hard_extent−Source | 0/+4.76812/+0.24401 | 0/-1.11286/-0.14956 | 24/23/22 |
| soft_extent−plain_soft | +0.02666/+0.75086/-0.07999 | -0.00550/-0.15361/-0.01265 | 24/24/22 |
| hard_extent−plain_hard | 0/+0.95318/-0.20200 | 0/-0.22087/+0.02105 | 24/24/22 |

完整[解读](../../../outputs/ped2023_extent_factor_learning_20261013/INTERPRETATION.md)及[统计补充](../../../outputs/ped2023_extent_factor_learning_20261013/readout/TABLES.md)已完成：162个配对区域比较、1584项历史统计一致性检查，只有已保存指标读取，无模型/MRI/GT数组重算。

- soft−plain_soft TC +0.75086pp，描述性95%CI[+0.28423,+1.37449]；去最大收益仍+0.54672pp。TC配对HD95−0.15361mm（n24），固定所有参照共同23例−0.14233mm，两种区间均不跨零。raw−plain TC+0.77216pp也支持学习贡献。
- hard−plain_hard ET−0.20200pp，CI[−0.43876,−0.01596]，作为消融，不扩hard。
- soft−同24例stable_consensus Dice−0.35534/+1.16943/+0.05892pp，TC去最大收益仍+0.90304pp；WT代价明确，ET区间跨零。不同成本、有取舍，不能称全面胜出。
- soft−Source WT HD95−2.27985mm几乎来自00129，去该例后−0.00293mm；ET Dice去00140最大收益后变负。投影soft全漏ET00009/00055，raw仅00009，保留这一损失事实。
- 基于以上完整证据，当前明确决定扩soft固定方案，检验更简单方法在其余开发病例是否保留收益；不是从均值自动扩两臂。所有区间未经多重比较校正，仍无独立泛化结论。

必须保留的机制限制：

- [loss_geometry](../../../outputs/ped2023_extent_factor_learning_20261013/loss_geometry/README.md)说明未编辑原始范围loss/梯度按rho²缩放：soft为1/4、hard为0。不是加强原始网络的范围正则，编辑loss还含交叉项。
- [target_semantics](../../../outputs/ped2023_extent_factor_learning_20261013/target_semantics/README.md)说明原teacher高置信BG不保证投影后的前景组织建议可信。00099的1,828个FG→BG建议在hard下仍产生内部组织监督，不能仅凭teacher标签判断应停止学习。
- hard输出WT等于Source是构造保证，不是学会了更好的WT。池化有旧CPU先例，不据此宣称创新。

## 已有对照与其他完成工作

- 外部家族：TENT/SAR/COME/VISTA已完成98组；MEMO/CoTTA/InTEnt/VPTTA为固定24例。BN和IN相对各自Source比较，尚不是统一预算的最终排名；不能按方法是否提升决定是否保留。
- BN背景共识24例无整体收益，IN MEMO/CoTTA近Source；TEGDA/ReGA仅接口审计，不能算已完成native性能基线。
- 类别对蒸馏、稳定组织anchor、可信类别顺序约束等额外学习方案未改善plain，负结果保留。Source统计回放fixed_source相对plain TC再增1.1376pp，但WT/TC HD95恶化2.6887/2.1662mm；不作为已解决方案。总体看[RESEARCH_STATUS](../../RESEARCH_STATUS.md)。
- 用户给的四份QQ代码已完整对比：[报告](../../qq_source_code_comparison_20261013.md)。外部默认模型21,365,348参数，与native结构/训练均不同；validate有batch>1低报问题，WT/TC辅助函数和空HD95口径也不同。缺少实际YAML/loader/loss/checkpoint，不能推断真实训练性能。

## 自动运行与token问题：结论及真实状态

用户希望保留全自动且使用桌面端。此前先写CLI脚本、后核实桌面方案，顺序不理想；不要把CLI原型说成桌面Goal已修复。

1. **优先考虑桌面Scheduled定时续跑。**[官方说明](https://learn.chatgpt.com/docs/automations)支持回到当前聊天、分钟级间隔及本地项目。讨论建议每30分钟检查一次，仍在运行就结束本轮；完成/失败才分析，启动下一实验后结束。原Goal保持暂停，由Scheduled负责后续唤醒。电脑和桌面应用必须运行，执行位置选现有本地目录，不能用缺少未提交代码/忽略资源的新worktree。
2. **Scheduled尚未创建，也未实测。**新对话如果配置，应绑定新对话而非旧线程。这是减少检查次数的方案，每次定时检查仍有模型开销，完成后要等下一检查点，不是即时零token事件唤醒。本会话未提供本地Scheduled创建工具，不应虚构创建成功或直接改客户端数据库。
3. CLI原型是 `scripts/research_dispatch.py`、`tests/test_research_dispatch.py` 和[说明](../../RESEARCH_DISPATCH.md)。13项测试通过，包括模拟两轮交接；真实模型接续/认证/sandbox/桌面同步尚未验证，未启用。它不控制Goal开关，旧文档命令含旧线程ID；不要在新对话直接照抄执行。读取线程成功不能证明可靠接管或Goal状态不变。
4. 本机桌面版本26.1007.2314.0，npm CLI0.153.4。CLI支持queue，但桌面接入未验证；本机daemon管理命令报告只支持Unix。不要擅自安装服务、升级客户端、替换非官方fork或修改全局配置。
5. 公开反馈确有同类问题：[反复高推理轮询耗额度 #45974](https://github.com/openai/codex/issues/45974)、[Goal等待/唤醒需求 #28144](https://github.com/openai/codex/issues/28144)、[进程完成唤醒 #32188](https://github.com/openai/codex/issues/32188)。10月13日读取时均Open，不等于已在官方桌面版修复。

## 本轮交付与后续边界

1. 本轮所需预测、评价、独立复核、删减、学习对照、基线汇总及98份交付重放均完成。以[主方法文档](../../GLI_PED_TTA_METHOD.md)作为方法入口；不重跑旧prepare/execute或反复审计。
2. 公平预算比较、BN可用性、独立患者验证及论文创新性仍是未解决的研究问题，不能由本轮均值替代。PED2023/PED2024存在MRI重叠，不能按年份直接称独立测试。这里记录限制，不自动启动新研究。
3. 等待策略已使用普通脚本自动接续和工具内等待进程退出，无逐病例模型轮询。没有建立桌面Scheduled或启用旧CLI原型；上方桌面方案是历史讨论，不是当前运行机制。

项目原始数据保持现有位置；公共代码在`src/brats_tta/native/`，模型在`artifacts/models/`，实验在`outputs/`。保留全部未提交修改、冻结依赖、mask、适应状态与负结果；不commit/push，不删数据或权重。

---

## 历史研究记录（按日期阅读）

以下保留原有记录，供追溯实验过程。状态冲突以上方最新交接为准，不将历史待办、PID 或 Goal 状态当作当前执行指令。

更新：2026-10-13。整体进度先看[研究状态](../../RESEARCH_STATUS.md)：Source与协议已确定，八个外部TTA方法家族有native结果；98组较均衡候选的TC Dice对Source +7.9335pp，WT/ET收益很小且不稳定，最终方法尚未确定。最新Source统计学习24例及全部参照于09:37北京完成，两个MRI版本审计于07:53完成。最终方法、机制归因、BN迁移及独立泛化仍待验证。文件职责见[总说明](../../PROJECT_FILE_GUIDE.md)，本次状态更新前的8份文档保存在[备份](../../../artifacts/maintenance/source_moment_replay_20261013/completion_readout_20261013/before.zip)。下方更早的“最新”与启动身份均按日期理解，不是活跃任务队列。

用户提供的[四份训练代码对比](../../qq_source_code_comparison_20261013.md)已完成阅读和CPU核对：外部默认BN模型21,365,348参数，与native31,199,796的结构/训练均不同；原validate在4个完美预测上batch1报1、batch2报0.5，另确认类别均值误名WT/TC、空HD95及ET后处理口径。默认batch1不触发上述低报；实际YAML/loader/loss/checkpoint未提供，不能推断真实训练效果。源码副本与结果在`artifacts/maintenance/qq_code_comparison_20261013/`，外部原文件、native代码、当前GPU队列和历史成绩未修改。

## 已完成队列：学习中区分前景范围与内部组织（以下保留启动时记录）

12:46北京：原plain恢复已于12:40:59完成24/24例，旧概率、mask和state逐值一致，生成48份仅推断后修正的对照mask；24份receipt哈希及声明另行核对通过。controls child54972已结束，queue session8306/PID54792和prediction child16868核实存活，首例00003两臂各完成8步实际更新。hard的范围anchor为0，与设计一致；这些检查不代表分割改善。旧plain00003完整8步复现已于12:36通过。尚无本轮完整Dice/HD95。

[新固定24例队列](../../../outputs/ped2023_extent_factor_learning_20261013/README.md)于12:35北京启动，另冻23代码/配置与162记录。soft_extent使用Source与student前景/BG竞争的等权平均，hard_extent为Source硬WT控制；内部三类对比保留。相同变换用于理想teacher目标、学习与最终输出，保留原native IN及完整Jacobian、原teacher/E、8步Adam.001，不加TC专属规则。池化有旧CPU输出先例，本轮检验学习的额外价值。

[损失分解](../../../outputs/ped2023_extent_factor_learning_20261013/loss_geometry/README.md)新增36项代数、36项实际loss及36项anchor梯度CPU检查并通过，无患者数据或CUDA。soft使原始未编辑范围惩罚/梯度成为1/4，hard使其为0；编辑项仍有组织/范围误差交叉项。因此本轮不能解释为“加强网络范围正则”，须比较原始预测与plain、最终预测与同变换输出对照。该补充不改主23/162冻结身份或运行方案；12:53实际队列仍存活，2/24例完成，00008在运行。

[teacher目标含义核对](../../../outputs/ped2023_extent_factor_learning_20261013/target_semantics/README.md)补充12个CPU例子及已完成00099/hard的只读检查：该例1,828个原teacher修改均为FG→BG，硬约束改变全部目标标签后仍有非零内部组织监督，8步实际更新、WT精确保持Source。硬约束保留完整前景分数差，因此BG标签本身不能判断该例应否停止学习；BG高置信也不能证明其前景组织建议可信。未读取GT或新评分，原23/162不变，完整效果仍待主队列评价。见check.json，勿覆盖或重复运行。

[范围约束相关工作](../../ped2023_extent_related_work_20261013.md)补充BAPA、Shape Moments、ReGA及PIE，明确已有概念与本研究的差别。[ReGA接口核对](../../../artifacts/experiments/ReGA_interface_audit_20261013/AUDIT.md)保存固定公开commit的9份源文件（162,078字节），Git blob/SHA256与8份AST核对通过，未运行上游代码或下载权重。它使用IN，但还依赖dropout、级联输入、特征接口和跨病例状态；公开代码与论文配置也有差异，不能直接计为新增已跑基线。13:05观察当前extent队列4/24例完成、00042运行中；继续核实实际进程，尚无完整新指标。

CPU闭式梯度/36坐标有限差分/18次前景置换及独立指标检查通过；真实00003当前代码预检39.3秒、32视图通过，旧首梯度与Source原始输出逐值一致，两臂各更新7680标量，hard保持Source WT。首次通过记录因后续新增完成回执helper保留在runtime_preflight_01，当前完整代码已重查。预检不证明效果。

队列现运行24×2学习臂，之后自动统一评价、216mask/648区域独立复核、报告及共同病例/强参照影响审计。接续重新核实实际进程，不重复execute/prepare，不修改冻结代码或public native，不自动扩98。全部病例开发暴露且GT有ET，无BN或独立泛化结论。

[上轮传播诊断](../../../outputs/ped2023_source_moment_replay_20261013/transfer_diagnosis/INTERPRETATION.md)已完成，360绝对区域/432精确Dice分解/5105稀疏路径通过；fixed_source额外TC主要来自Source预测WT内部ED→NC，未accepted域贡献+0.9699pp。其外部误改与内部收益并存，不能一概锁回Source。诊断4代码/180记录冻结，不重复运行，不据GT逐例制定规则。状态更新前5份导航在[维护备份](../../../artifacts/maintenance/extent_factor_learning_20261013/before.zip)。

## 最新完成：IN统计学习改变了收益与边界风险的取舍

- [Source统计回放](../../../outputs/ped2023_source_moment_replay_20261013/INTERPRETATION.md)固定24例×2臂已完成预测、评价、48状态/192mask/576区域独立核验、报告、共同病例审计与七方法参照。主25代码/19记录、修复2/9、参照2/25冻结身份保留，Source不变。
- fixed_source对plain Dice +0.0482/+1.1376/+0.0752pp，配对HD95 +2.6887/+2.1662/+0.0364mm（n24/24/22；固定共同TC n23为+2.5391mm）。TC增益区间[+0.2447,+2.2982]，去掉最大收益仍+0.7263pp；00008造成显著WT/TC边界尾部风险，保留全部病例，不靠去除尾部宣布优势。
- detached_current对plain Dice +0.4512/−2.7463/+0.0641pp，HD95 −0.2000/+0.9970/−0.0266mm。对Source各均值改善，但WT/ET收益不稳定、TC低于已有学习和组合；同24例没有优于stable_consensus。两臂均不自动扩98或替换主候选。
- 固定Source统计仍来自本PED图像，每层/滑窗/镜像分别采集，并在学习及推断回放。它改变归一化前向与有效函数方向；detached_current保持native前向，使用停止统计梯度的替代反向。相同首次梯度与affine参数数目不能隔离统计漂移的因果贡献，见[算子检查](../../../outputs/ped2023_source_moment_replay_20261013/invariance_audit/REPORT.md)。
- 原队列于09:32因review缺少final赋值退出1；[review_fix01](../../../outputs/ped2023_source_moment_replay_20261013/review_fix01/README.md)按冻结条件恢复单一argmax赋值并完成接续。原失败脚本/日志保留，完成回执显式标识修复；学习/预测/评价没有重跑。[queue_complete](../../../outputs/ped2023_source_moment_replay_20261013/queue_complete.json)、[修复complete](../../../outputs/ped2023_source_moment_replay_20261013/review_fix01/complete.json)和[参照final](../../../outputs/ped2023_source_moment_replay_20261013/reference_comparison/final.json)齐全。session2454/8869及PID65356/68772/32664已结束，不重复execute/watch/review/final。
- 旧plain00003八步逐值等价与CPU/真实预检保留；成绩以完整报告为准，24例全开发暴露且GT有ET，无BN迁移、独立泛化或创新性结论。完整参照不与98组均值混用。

## 数据审计完成：PED2024不能按年份直接叫独立测试

[跨版本MRI报告](../../../artifacts/maintenance/source_moment_replay_20261013/target_cross_version_identity/REPORT.md)覆盖99条2023输入和261/257条2024输入。主/alternate分别82/80条目标在2023支持区域内四通道精确相同，其中79/77条有异ID对应；2023的00111对应2024的00091，重复121/137均对应00101。全图hash、mask或匿名ID不同不证明患者独立。

保守剩余完整记录129/127条，仍未排除患者重叠，其中129/126有旧Source评分。主255缺t1c，alternate255是唯一完整且没有历史Source评分的ID，尚不能据此建立独立测试。header单位与数值grid分别记录；未读GT数组或新评分，数据集未修改。2代码/5记录冻结，comparison/complete已完成，session20962/PID41256结束，不重复审计。

当前已据完整传播证据启动上方范围/组织学习队列，检验能否保留内部重分类并减轻范围风险；简化组合、公平比较与独立验证仍未解决。下方已完成负对照和所有原始结果继续保留。

## 最新学习实验已完成：可信类别顺序约束几乎未改变plain

- [新固定24例实验](../../../outputs/ped2023_decision_consensus_anchor_20261012/README.md)使用原accepted且Source=teacher的M。decision_stable以四类logit到可信类别决策集合的平方距离为附加损失：没有顺序冲突时为0。same_norm_score用完整中心化Source分数锚点，detach当前系数以匹配同状态辅助logit梯度范数，区分约束方向与惩罚强度；不控制IN参数梯度或Adam位移。
- 原edit+extent、完整teacher目标、Source IN250、8步Adam.001、22层IN affine和native推断不变。无新阈值或TC/ET专属规则；约束允许平局，不是argmax硬锁。M可能含共同错误，不能把与Source的分歧视为真实错误。
- CPU独立投影/KKT/96坐标有限差分/24类置换，48运行梯度与Source identity、72 FG置换及2项八视图回放检查通过。只读旧24例预测发现M有4,393,200体素，22例存在3,301严格冲突、4个平局翻类；无GT、无新网络或评分。Source不发生类别冲突时新正则首梯度为0。
- 全24例×2臂于2026-10-13 00:46北京完成预测、评价、48状态/192mask/576区域独立核验、报告、共同病例审计及自动参照；[study_complete.json](../../../outputs/ped2023_decision_consensus_anchor_20261012/study_complete.json)和完整回执哈希链通过。Source checkpoint未变。session82516已退出；原queue PID63084/predict PID59224只是历史身份，PID可能被其他进程复用。
- decision_stable对plain WT/TC/ET Dice为−0.00016/+0.00078/−0.00027pp，配对HD95为0/−0.00042/+0.00015mm（n24/24/22），没有实际增益；同辅助梯度范数的same_norm_score也近乎plain。对Source TC+3.82072pp来自既有plain收益，不能归功于新约束。保留负消融，详见[解释](../../../outputs/ped2023_decision_consensus_anchor_20261012/INTERPRETATION.md)。
- 对Source WT Dice−0.37708pp；WT HD95−2.15165mm在去掉00129后变为+0.20404mm。ET Dice+0.47459pp在去掉00140后变为−0.17854pp。七方法同病例参照已完成，相对extent_post没有综合优势；geometry输出控制TC与新臂近乎相同，不能归因于独特定位。
- 主preparation冻结27代码/配置与12记录，参照独立2代码/25记录；旧20/8、24/11及全部结果哈希保持不变。不要重复prepare/build/execute或已完成补充，不自动扩98。00003完整8步旧plain GPU复现于22:12通过，97.21秒，概率、首梯度、最终state与原loss字段逐值一致。全部病例开发暴露，GT有ET，无BN迁移、独立验证或创新性结论。


首个00003两臂均已实际完成8步，首梯度与旧plain逐值一致，44份delta/7,680标量确认更新，每步辅助范数匹配通过。decision臂冲突激活从0增至64；两臂最终state不同。它只验证机制确实参与学习，不证明分割改善；完整队列已结束，不重启。只读核查见[回执](../../../artifacts/maintenance/decision_consensus_anchor_20261012/first_case_verified.json)。


00008自然不激活病例已只读核验：两臂8步新增损失/梯度/系数均为0，但原teacher学习确实更新参数；首梯度、44份affine/delta、概率、最终state与mask逐值/哈希等于旧plain。完整分数漂移诊断达到0.65664，类别顺序仍有效。它验证不惩罚这类分数漂移的实现，不证明性能提升；[核查回执](../../../artifacts/maintenance/decision_consensus_anchor_20261012/inactive_case00008.json)保留，不重复执行。

## 最新学习结果：监督方向改善WT，同时有TC代价

- [固定24例实验](../../../outputs/ped2023_extent_preserving_targets_20261012/INTERPRETATION.md)只改变原teacher编辑位置的监督目标：Source与teacher均为FG时，保留三个FG相对分数，把理想目标max-FG/BG差设为Source值；BG改变的编辑仍用原full目标。同能量same_norm_full保留原方向。不是增加组织anchor、锁定WT输出或输出拼接。
- 24例×2臂于21:25完成预测，21:30完成评价、48状态/192mask/576区域独立复核、报告及共同病例审计；pipeline_complete与queue_complete均通过。Source不变。旧session80012、queue PID49644及predict PID48964为历史执行身份，不再当作运行队列或重启。
- source_extent对plain WT/TC/ET Dice为+0.1130/−0.1350/−0.0288pp，配对HD95−0.0578/+0.0508/−0.0065mm（n=24/24/22）；七方法固定共同TC n23仍+0.0433mm。对same_norm_full Dice+0.1144/−0.1006/−0.0423pp；WT收益与TC代价的本轮配对bootstrap区间未跨0，ET差异很小。对Source TC仍+3.6850pp，但WT−0.2639pp，ET小收益依赖尾部病例。保留折中研究，不据此替换现有候选。
- [七方法同病例参照](../../../outputs/ped2023_extent_preserving_targets_20261012/reference_comparison/README.md)也已完成。对extent_post Dice−0.2410/−0.1833/+0.0276pp，配对HD95+0.0694/+0.0691/−0.0274mm（n=24/24/22）；ET微差区间跨0且对00099敏感。对stable_consensus，WT−0.5722pp，TC/ET+0.2660/+0.1208pp但区间跨0。参照成本和机制不同，仅比较最终方案；复用已独立检查指标，不新增EDT评价。
- [全24例梯度审计](../../../outputs/ped2023_extent_preserving_targets_20261012/gradient_geometry/README.md)通过，原两例snapshot保留。新方向/同范数控制首梯度范数比均值0.95473，实际8步delta范数比均值0.99725、范围0.96669至1.03385；等目标能量不等于同梯度，不能解释成更新必然更小。只读保存文件，不读GT或做新网络前向。
- 无GT目标审计38,789个编辑点均保留理想teacher类别；内部FG编辑23,302点中原gap增大20,648、减小2,654，不能把WT损失归因于内部目标普遍收缩。[00099一致性检查](../../../artifacts/maintenance/extent_preserving_targets_20261012/null_direction_case99/README.md)验证其无内部FG编辑，两新臂与plain首梯度、state、delta、概率、mask及8步共有loss字段完全一致。00003旧plain完整8步GPU逐值复现也通过。
- preparation冻结主24代码/11记录，旧stable20/8及参照2/25保持不变。Source IN250、teacher、8步Adam.001、native extent及推断不变；新runner单独快照，使用公共native。不要再运行prepare/build/execute或两项已完成补充，不自动扩98。全部24例开发暴露且GT有ET，无BN迁移、独立泛化或创新性结论。

## 最新学习实验已完成：稳定组织锚点未改善plain

- [固定24例学习实验](../../../outputs/ped2023_stable_consensus_anchor_20261012/INTERPRETATION.md)于15:28完成全部阶段，恢复session95727正常退出0。48个IN学习状态、192份新mask及576项区域指标通过独立CPU恢复/EDT复核；[pipeline_complete.json](../../../outputs/ped2023_stable_consensus_anchor_20261012/pipeline_complete.json)与[queue_complete.json](../../../outputs/ped2023_stable_consensus_anchor_20261012/queue_complete.json)均通过。共同病例/逐例影响审计也已执行，不再启动execute、recover、prepare或补充审计。
- stable/all_anchors在原native fused edit+extent上增加相同conditional组织项，只改变选择域：原accepted且Source=teacher，或全部未编辑位置。8步Adam1e-3、IN affine-only；Source/teacher/plain复用旧结果。对Source，stable的WT/TC/ET Dice为−0.3833/+3.6358/+0.4946pp，配对HD95−2.1424/−0.8212/−0.2094mm（n=24/23/22）。TC保留收益，不能据此称新增项有效。
- 对plain，stable Dice为−0.0064/−0.1841/+0.0197pp，配对HD95+0.0092/+0.0511/+0.0213mm（n=24/24/22）。五方法固定共同n=24/23/22时TC HD95仍+0.0440mm。两组TC Dice相对plain均下降，WT/ET变化很小；保留为消融，不扩展当前固定方案。
- stable比all_anchors的TC/ET Dice均值高0.2476/0.1235pp，但TC差的bootstrap区间跨0，移除最大收益00055后TC只+0.0051pp。它提示全部未编辑组织约束可能更限制传播，未证明stable优于无conditional的plain，也未证明因果机制。
- WT相对Source的HD95负均值受00129影响，去掉该最大收益病例变+0.2137mm；ET Dice去掉最大收益00140后变−0.1138pp。所有病例仍保留主表，敏感性不用于择例。Source的00009空TC由各适应方法恢复；ET空预测00009/00055及零TP的00008未解决。24例全部GT有ET，不能评价无ET场景。
- stable相对同方向同数量geometry的TC Dice仅+0.0064pp，而TC HD95差+3.3984mm。当前结果未建立独特TC空间选点优势；这些输出控制也不控制学习约束的选择域。
- 原session88285及PID51728/33728在18例完成后消失，原因未知。独立恢复逐文件验证并复用18例/36臂，从Source与新optimizer重跑未完成00120/stable，保留中断文件及原日志，新增12臂。完成receipt记录恢复来源；seconds含中断间隔，不能当纯计算时间。工程预检、00003完整8步逐值plain复现和20项代码/8项记录继续保留。
- [A3-TTA官方代码静态审计](../../../artifacts/maintenance/stable_consensus_anchor_20261012/a3_tta_audit/findings.json)仅是代码核查，固定commit98675706、9份小型快照及SHA，尚无新基线成绩。实际2D BN/全部参数/EMA和bank行为不等同当前3D IN。文献边界见[记录](../../../artifacts/maintenance/stable_consensus_anchor_20261012/literature_scope.json)，可靠锚点/防遗忘已有先例。

## 最新进展：明确类别转换来源，共识保护只带来小修正

- [残余路径诊断](../../../outputs/ped2023_mechanism_merge_20261012/residual_transition_audit/INTERPRETATION.md)完成98组、2,058项绝对区域计数、8,706条稀疏路径及588项Dice分解，session14624结束。ED→NC贡献TC+7.7492pp；未accepted/context独有贡献TC+2.5522pp、ET−0.2518pp，不能把这些修改全部禁止。
- [teacher类别图](../../../outputs/ped2023_mechanism_merge_20261012/residual_transition_audit/GRAPH_REPORT.md)仅按每例S→T构图，73例四类全连通；跨连接context修改13,098体素中12,397改对，贡献TC+0.5862pp。64图/1,536次置换和294项分解通过，不采用类别图硬门控，不扫描阈值。
- [组合共识保护](../../../outputs/ped2023_mechanism_merge_20261012/local_support_filter/consensus_preservation/INTERPRETATION.md)12:03完成98×2、196份mask/588项独立区域核验，session49873结束。`stable_consensus`在原accepted且Source=teacher的位置保留Source，其余沿用margin6_guard；`all_consensus`恢复全部accepted teacher作为对照。
- stable对Source WT/TC/ET Dice **+0.0825/+7.9335/+0.1148pp**，配对HD95 **−0.7298/−2.2782/−0.1393mm**（n=98/91/84）；对前序Dice仅−0.0189/+0.0883/+0.0037pp。没有新增空区域，但ET仍依赖个别病例，保留为并列小修正，不称突破。新74组Dice+0.0092/+9.3977/+0.0465pp，仍开发暴露。
- all对Source Dice−0.0671/+8.0208/+0.1846pp，HD95−0.7005/−2.4708/−0.1631mm（n=98/90/83），重现2例TC、3例ET新增空预测。WT小幅代价符合用户允许折中，不能仅凭WT为负淘汰；固定共同病例与尾部检查未建立稳定优势，完整对照保留。
- 后续稳定共识学习锚点已完成，未改善plain，见本页顶部；旧hierarchical的全部未编辑锚点与当前native同预算控制须区别对待。此前输出组合工作均为CPU处理，新的学习实验只完成固定24例，原Source保持不变。

## 前序候选：限制修改数量并保留候选类别

- [分量支持诊断](../../../outputs/ped2023_mechanism_merge_20261012/component_support_audit/INTERPRETATION.md)完成98组、22,279个分量核验、1,176项Dice分解及882项绝对区域计数。没有Source同类支持的新NC多数正确，多方法支持的ET也可能误报，因此不采用Source连通性硬门控。
- [局部过滤](../../../outputs/ped2023_mechanism_merge_20261012/local_support_filter/INTERPRETATION.md)正式`attempt02/`完成98×3、294份mask/882项独立区域核验。opening6与同转换同修改数量margin6的Dice非常接近；margin的TC/ET HD95更好，未建立独特形态学选择优势。保留原F-order失败及修复记录，不重启旧任务。
- [类别存在性保护](../../../outputs/ped2023_mechanism_merge_20261012/local_support_filter/presence_guard/INTERPRETATION.md)11:44完成全部98组、196份mask/588项独立指标核验，session35169已正常退出。某类在候选R存在却被局部过滤全部清空时，恢复R该类，四类统一；仅3例触发。相对R不再新增空区域，不能保证真阳性。
- `margin6_guard`相对Source WT/TC/ET Dice **+0.1014/+7.8452/+0.1111pp**，配对HD95 **−0.7527/−2.2672/−0.1431mm**，n=98/91/84。绝对Dice89.3737/56.6898/54.4391%。保留为当前输出组合候选；WT/ET Dice与ET HD95置信区间跨0，ET Dice去掉最大收益后为−0.0778pp，不称稳定全面改善。新74组Dice+0.0324/+9.2840/+0.0433pp，HD95−0.1755/−2.7129/−0.1181mm，但仍是开发队列。
- `opening6_guard`相对Source Dice+0.1105/+7.8509/+0.0930pp，HD95−0.8230/−2.2154/+0.0585mm。其保护前ET HD95略优、保护后略差，须同时报告空预测与共同病例，不能挑选有利有效样本。所有分支固定共同n=98/90/83的核验另见[CONTROL_REPORT](../../../outputs/ped2023_mechanism_merge_20261012/local_support_filter/presence_guard/CONTROL_REPORT.md)。
- 00130只能恢复上游剩余3个TC真阳性，Source原有691个；00128/00083仍有ET下降，00051仍有WT下降。后续全队列转换来源诊断与共识组合保护已完成，见本页顶部。当前流程包含context适应、范围校正和COME组合，不能称为单一新损失或独立创新。无新网络推断，Source checkpoint SHA未变，全部98组开发暴露。

## 前序研究：分开约束范围与组织分类

- [错误来源诊断](../../../outputs/ped2023_mechanism_merge_20261012/provenance_audit/INTERPRETATION.md)已完成98组及补充核验。00019既有teacher直接删错也有额外传播；00083的ET损伤主要来自学习传播。112种合法标签组合与2,688次类别置换检查通过，不把这类事后分解写成新方法成绩。
- [四类等权概率池化](../../../outputs/ped2023_mechanism_merge_20261012/source_trust_pool/INTERPRETATION.md)正式结果在`attempt02/`，完成98×3预测及882项独立区域核验，session95600已结束。几何池化对Source Dice−0.1876/+2.9613/+0.2283pp，ET HD95+0.7760mm；比旧union少4.2441pp TC，不替换当前候选、不扫描权重。原attempt因历史概率key不同在第2例停止、未评分，冻结记录保留；新runner仅修正读取，输入/规则一致。
- [空间前景范围约束](../../../outputs/ped2023_mechanism_merge_20261012/spatial_extent_pool/INTERPRETATION.md)10:56完成全部98组、294份独立mask重建及882项区域核验，session47367已结束。只约束逐位置max前景/BG代理，保留适应后的NC/ED/ET相对分数；包含硬锁Source WT控制，以及与COME的对称union，没有新训练。
- 新`spatial_union`相对Source WT/TC/ET Dice为**+0.0911/+8.2473/+0.2607pp**，配对HD95 **−0.7868/−2.5945/+0.5416mm**（n=98/91/84）。相对旧union Dice+0.2223/+1.0419/−0.1516pp，配对HD95−0.1317/−0.6086/+0.4382mm（n=98/95/83）。WT/TC折中更好、ET边界更差，保留为并列候选，不能称全面胜出。新74的ET HD95仍+0.8477mm。
- soft单独TC57.3388%，硬WT对照57.2953%，两者接近，未建立soft池化独特优势。00085 TC恢复到43.6105%，不再因缺teacher种子而回退Source；00019/00083仍有残余错误。所有98组开发暴露，没有独立测试或BN迁移结论。
- [ET分量核验](../../../outputs/ped2023_mechanism_merge_20261012/spatial_extent_pool/tail_verification.json)确认：00032原有19体素远处ET误报扩到28，95分位跨过远处分量；00104由原teacher同时补回真ET并新增109体素远处假阳性。该组合空ET误报6,874体素（Source8,968、旧union9,926），仍7/11病例有误报。后续全98组分量级支持诊断、局部过滤与类别保护已完成，见本页顶部；不凭两例直接删小病灶。

## 前序候选：种子传播与机制组合

- [种子支持诊断](../../../outputs/ped2023_teacher_retention_20261011/seed_support_audit/INTERPRETATION.md)首次执行并完成98组及独立复核，session50351已退出。旧代码/协议写于10月11日，本日才有实际回执。被拒修改中95,034个具有稳定高置信邻接，但在既有C结果的Dice分解中WT/TC/ET为−0.0971/+0.2669/−0.1069pp；不是新方法成绩。00085没有任何高置信NC/ET种子，8,743个真实NC的ED→NC纠正全部无支持，解释了旧传播规则拒绝其43.5639pp TC收益；单纯增加稳定种子不能救回它，不直接扩展该规则。
- [机制合并](../../../outputs/ped2023_mechanism_merge_20261012/INTERPRETATION.md)固定全部98组，把seeded6与COME的非Source建议对称合并；目标唯一时采用，冲突回Source。另有两方法必须一致的intersection对照。没有区域专用选择，也没有新网络训练或大概率缓存。196份mask、588项独立Dice/HD95核验完成，session64458已退出。
- union对Source的WT/TC/ET Dice为−0.1312/+7.2053/+0.4123pp，配对HD95−0.6551/−1.9728/+0.0369mm（n=98/91/83）；对seeded6 Dice为+0.2679/−0.1441/+0.2330pp。WT平均代价缩小，TC保留大部分收益，但ET HD95仍稍差。intersection相对Source TC只+0.4193pp，保留为保守对照，不据三个Dice正均值替代主规则。
- union在GT空ET病例的假阳性体素增至9,926（seeded6为7,495、Source8,968）；00083 ET下降26.02pp、00019 WT/TC有大幅退化。00085仍未恢复。详见[逐例敏感性](../../../outputs/ped2023_mechanism_merge_20261012/behavior_audit.json)。组合是开发队列输出对照，不是独立测试或创新性证明；计算成本包含两条上游适应路径。
- 00019/00083来源诊断及后续两轮概率约束已完成，见本页顶部；不要重启上述已完成任务，不要重跑已否定的稳定邻接分支。

## 当前用户任务已完成：扩大方法种类，核查BN/IN处理

用户指出首轮对照种类不足，要求参考文献处理BN/IN。新增[扩展实验](../../../outputs/ped2023_expanded_baselines_20261011/README.md)及[归一化文献核查](../../../outputs/ped2023_expanded_baselines_20261011/NORMALIZATION_REVIEW.md)。MuVi分别训练BN/IN Source并分组比较，本轮沿用既有IN250/BN225，不替换权重或重新训练。

- 固定旧24组先导，BN组Source/current statistics/TENT/InTEnt/VPTTA于20:09北京完成预测、评价、360项区域指标独立复核和报告，session52716已退出。24份Source及24份统计控制逐体素匹配旧结果。TENT/InTEnt/VPTTA的WT Dice分别+0.9725/+0.5856/+0.2511pp，TC分别−2.4765/−0.1545/−2.6085pp；详见[BN报告](../../../outputs/ped2023_expanded_baselines_20261011/bn_controls/REPORT.md)及[解释](../../../outputs/ped2023_expanded_baselines_20261011/bn_controls/INTERPRETATION.md)。冻结`bn_controls/preparation.json`中的依赖，勿修改。
- IN组MEMO/CoTTA于20:30北京完成相同24组预测、评价、216项区域指标独立复核及报告，session1319已正常退出。相对IN Source的WT/TC/ET Dice分别为+0.0044/−0.0025/+0.0022pp和+0.0007/−0.0164/+0.0306pp，接近Source。两方法各执行172次更新，CoTTA的.9门控0/172触发；不能称增强分支已充分验证。见[IN解释](../../../outputs/ped2023_expanded_baselines_20261011/in_controls/INTERPRETATION.md)。保留冻结依赖，不重复启动`queue_in.ps1`。
- [同24组总表](../../../outputs/ped2023_expanded_baselines_20261011/REPORT.md)、[综合解释](../../../outputs/ped2023_expanded_baselines_20261011/INTERPRETATION.md)与[图](../../../outputs/ped2023_expanded_baselines_20261011/comparison.png)已生成；输入/GT、139份作者代码、两组连续状态哈希链及Source checkpoint核验通过。576项独立区域指标通过；24组全部GT有ET，不代表无ET病例的假阳性控制。新增方法尚未扩展到98组。
- 新增的是四种机制家族；TENT的BN/IN版本、不同步数及SAR消融不重复计作新方法。24组不与98组均值混用，BN与IN各自配对Source，负结果也报告。
- TEGDA当前公开路径的2D/3D接口不匹配已有隔离测试证据，native Source还缺少MC dropout，仍未建立合格移植；不编造其结果。详见扩展目录`tegda_code_audit.json`。
- 本轮前的入口在[维护备份](../../../artifacts/maintenance/full_cohort_status_20261011/expanded_baselines/before.zip)。首轮delivery记录的文档SHA是当时快照；不回写旧完成回执。

## 相关论文与首轮外部对照已完成

用户要求先核查本领域论文实际比较的方法，再跑适用实验；首轮已完成，下面的种子覆盖诊断仍不是正在执行的任务。见[本轮入口](../../../outputs/ped2023_literature_baselines_20261011/README.md)、[论文比较表](../../../outputs/ped2023_literature_baselines_20261011/LITERATURE_REVIEW.md)、[统一结果](../../../outputs/ped2023_literature_baselines_20261011/REPORT.md)及[结果解释](../../../outputs/ped2023_literature_baselines_20261011/INTERPRETATION.md)。TEGDA直接比较TENT/SAR/CoTTA/InTEnt/VPTTA；VISTA加入COME/EATA等，MuVi提供单病例3D对照参考。不得将异Source、二分类或BN依赖方法混入当前IN主表。

- `norm_controls/`：SAR-IN-no-recovery消融及COME-TENT-IN已于15:28北京完成98组预测、评价、独立复核和报告；98份新Source与原本地Source逐体素一致。COME的WT/TC/ET Dice为+0.2772/+0.1258/+0.3713pp，配对HD95为−0.8747/+0.0641/−0.1956mm（n=98/91/83）；SAR关闭恢复后几乎不变。见[解释与组织审计](../../../outputs/ped2023_literature_baselines_20261011/norm_controls/INTERPRETATION.md)，不隐去默认SAR恢复退化或HD95有效病例变化。
- `vista_controls/`：VISTA-region-IN逐例/连续和TENT十步后整例输出，98组于16:46北京完成评价、1,176项独立区域指标复核及报告；三条分支各980/980次有效更新。重置VISTA近乎Source；连续VISTA的Dice为−0.0317/−1.8650/+1.1667pp，配对HD95为−0.1249/+0.5697/+0.2515mm（n=98/89/78）。连续版ET总体增益含两组无ET假阳性清除，87组有ET的Dice反而下降0.9847pp。TENT十步版Dice为−0.2867/−3.5116/+2.7097pp，TC/ET配对HD95恶化。见[分支解释](../../../outputs/ped2023_literature_baselines_20261011/vista_controls/INTERPRETATION.md)。不能据此否定原BN/整例VISTA。
- 两个`pipeline_complete.json`、统一`synthesis_complete.json`和组织审计均已完成。session39204/61680/95838均已退出；`progress.json`保留最后预测事件，不代表仍在运行，不重复启动。旧Source、TENT/SAR及本方法结果保留；本轮Source与我们的既有Source指标完全一致，新结果全部是开发队列证据。
- 上述是首轮范围；后续CoTTA-IN、InTEnt/VPTTA-BN已完成扩展24组，移植限制见前节。TEGDA仍未完成合格native移植，MemBN尚无本轮结果。不根据方法有没有提升决定是否将其列入结果。

## 完整开发队列与最新传播约束结果

plain与前景校正均已完成预测、评价、独立核验和报告：plain于05:49北京完成，前景校正于05:51完成。原session72323/86813及PID60788/37128是历史执行身份，不能继续当作存活任务或据此重启。两轮均复用旧24组、新增74组；主要结果为98唯一MRI/标签组，99记录单列描述。所有病例已有开发暴露，不是独立测试。

- [plain报告](../../../outputs/ped2023_native_objective_full99_20261011/REPORT.md)、[完成回执](../../../outputs/ped2023_native_objective_full99_20261011/pipeline_complete.json)、[独立核验](../../../outputs/ped2023_native_objective_full99_20261011/verification.json)：98份适应state、392份mask、1,176项区域指标通过检查，Source checkpoint不变。
- [前景校正报告](../../../outputs/ped2023_pairwise_distillation_20261011/extent_projection/full_cohort/REPORT.md)、[完成回执](../../../outputs/ped2023_pairwise_distillation_20261011/extent_projection/full_cohort/pipeline_complete.json)、[结论与诊断](../../../outputs/ped2023_pairwise_distillation_20261011/extent_projection/full_cohort/INTERPRETATION.md)：新增74份mask独立重建和222项区域指标通过，旧24份结果按哈希复用、六组汇总核对一致。CPU校正没有新增网络前向。
- [仅保留teacher修改E](../../../outputs/ped2023_teacher_retention_20261011/INTERPRETATION.md)于12:30北京完成；[保留完整高置信共识C](../../../outputs/ped2023_teacher_retention_20261011/accepted_support/INTERPRETATION.md)于12:38完成。两者各生成98份新mask，各通过98份独立约束重建和294项区域指标复核；完成回执分别在[父实验](../../../outputs/ped2023_teacher_retention_20261011/pipeline_complete.json)及[子实验](../../../outputs/ped2023_teacher_retention_20261011/accepted_support/pipeline_complete.json)。没有新增网络前向，session80170/16154均为已结束的历史执行器。
- [种子约束传播](../../../outputs/ped2023_teacher_retention_20261011/seeded_propagation/INTERPRETATION.md)于13:02北京完成：6邻接主规则、预设26邻接敏感性和同数量距离对照，各生成98份mask；294份mask及882项区域指标独立实现复核通过，[完成回执](../../../outputs/ped2023_teacher_retention_20261011/seeded_propagation/pipeline_complete.json)可查。session29821已退出，没有新增网络前向。

主要98组Dice百分数如下。旧24例结果保留原报告，不能继续当完整队列结论。

| 方法 | WT | TC | ET | 三区域均值 |
|---|---:|---:|---:|---:|
| Source | 89.2723 | 48.8446 | 54.3281 | 64.1483 |
| 直接背景共识teacher | 89.0645 | 54.3912 | 54.6046 | 66.0201 |
| plain IN学习 | 87.1865 | 55.1585 | 53.6982 | 65.3477 |
| plain + 前景偏置校正 | 88.7357 | 55.7287 | 54.0372 | 66.1672 |
| 再保留teacher修改E | 88.5967 | 57.0662 | 54.1551 | 66.6060 |
| 再保留完整共识C | 88.6137 | 57.0565 | 54.1735 | 66.6146 |
| C后限制种子传播，6邻接 | 88.8732 | 56.1941 | 54.5074 | 66.5249 |
| 26邻接敏感性 | 88.8173 | 56.3515 | 54.4949 | 66.5546 |
| 同数量距离对照 | 88.8663 | 56.1775 | 54.5416 | 66.5285 |

校正相对plain的WT/TC/ET Dice为+1.5492/+0.5703/+0.3390pp，配对HD95为−0.4365/−0.1775/−0.1121mm（n=98/92/83）。相对Source则为Dice−0.5366/+6.8841/−0.2909pp、HD95+0.1671/−1.8259/+0.4915mm（n=98/90/83）。因此确认的是减轻plain损伤、保留TC收益，尚不是WT/ET也全面超过Source。

前景校正后三区域均值比teacher高0.1471pp，但新增74组反而低0.1045pp，不能称为稳定胜出。这是后续teacher保留实验的基线；全部结果、逐例分布和HD95共同有效口径见上述报告，均值改善不代表每例改善。

最新C保留在原teacher高置信集合内使用teacher标签，集合外沿用plain加前景偏置；E保留只约束Source与teacher标签不同的位置。阈值与四类处理一致，C包含E。C相对前景偏置的Dice为−0.1220/+1.3278/+0.1363pp，配对HD95为−0.0415/−0.3047/+0.1275mm（n=98/92/83）；相对Source为Dice−0.6586/+8.2119/−0.1546pp、HD95+0.1256/−2.0940/+0.6190mm（n=98/90/83）。C相对E仅为Dice+0.0170/−0.0097/+0.0184pp，不应把这个微小差异写成解决WT/ET问题。它们是TC收益更大的候选；直接teacher仍有更小的WT/ET代价，不能称新组合全面优于teacher。

最新6邻接规则只允许连接到同方向teacher修正的额外a→b变化，其他恢复Source。相对C，WT/TC/ET Dice为+0.2595/−0.8624/+0.3339pp；相对Source为−0.3991/+7.3495/+0.1793pp，配对HD95为−0.3287/−1.9751/−0.0236mm（n=98/91/84）。WT/ET平均代价减轻，仍有TC取舍。与同数量距离对照的Dice差仅+0.0069/+0.0166/−0.0343pp，未建立独特连通性优势。WT/ET的Source相对HD95均值对极端病例敏感，新74组两者仍略差；00085的TC相对C下降43.56pp。ET缺失病例假阳性体素由C的4,150回升至7,495。详见[敏感性与失败记录](../../../outputs/ped2023_teacher_retention_20261011/seeded_propagation/decision_audit.json)，不能只汇报三个HD95负均值。

## 哪些已经确定，哪些仍待验证

当前开发主方法已确定，见[方法说明](../../GLI_PED_TTA_METHOD.md)；身份、协议与结果核验不等于论文创新性或独立泛化已证明。

| 对象 | 当前状态 | 文件及维护方式 |
|---|---|---|
| Source和对照模型 | native四分类IN final250、BN best225等源训练完成 | [MODEL_INDEX](../../MODEL_INDEX.md)定位；checkpoint、plans、dataset、trainer配套保留，备份不就地覆盖 |
| 当前比较协议 | PED2023、native四通道、四标签、区域合并和HD95约定固定 | 见本页实验口径及各轮`PROTOCOL.md`、病例清单 |
| 已完成的主要结果 | Source/teacher/plain/前景校正、E/C保留及种子传播对照均有完整开发队列结果 | [实验索引](../../EXPERIMENT_INDEX.md)定位冻结代码、mask、参数状态、指标、报告和回执；新实验写新目录 |
| 已完成的对照与消融 | BN统计、BN背景共识、Fisher、类别对蒸馏、病例级输出偏置等24例 | 保留结果；完成不等于有优势，不回写原成绩 |
| 公共维护入口 | `native/`、`paths.py`、`paths.yaml`、`project.ps1` | 已有CPU回归和00003旧plain八步GPU逐值复现；加载器仅IN250，teacher CLI只处理缓存 |
| 实验实现 | 前景偏置在`outputs/.../extent_projection/projection.py`；E/C保留在`teacher_retention`父/子目录的`retention.py` | 均未并入公共包；前者校正plain概率，后两者组合标签，不新增全模型权重 |
| 尚无患者性能的分支 | BN input-style、全部竞争类别约束 | 仅CPU准备或数学检查，不能写成已验证方法 |
| 方法与论文结论 | 本轮主方法已确定：union2+soft_extent+no_come | 方法与98份重放已交付；WT/ET稳定性、BN迁移、因果机制、创新性和独立泛化未确认 |
| Git状态 | 公共新增代码和大量文档未提交 | 本机文件不等于远端已有；未经要求不commit/push |

## 已完成分支与机制证据

- [学习后四类矩校准24例](../../../outputs/ped2023_student_moment_calibration_20261011/INTERPRETATION.md)：新CPU实验已完成预测、评价、独立复核和报告，session2280正常退出；24mask与72项区域指标通过。以plain为起点匹配原teacher类别总量，相对单前景偏置Dice−0.0876/−2.4986/−0.2968pp、HD95+0.0069/+0.6853/+0.1382mm（n=24/24/22）。两组TC均下降，不扩大该固定方案。相对Source仍有小幅TC收益，不等于改善既有候选。

- [类别对蒸馏24例](../../../outputs/ped2023_pairwise_distillation_20261011/REPORT.md)：相对plain Dice−0.0093/−0.8912/−0.4027pp，配对HD95+0.0483/+0.2058/+0.1481mm（n=24/24/22）。未改善plain，保留消融，不扩至99例。最小输出修正范数不保证最小参数更新或最好性能。
- [BN背景共识24例](../../../outputs/ped2023_bn_context_20261011/INTERPRETATION.md)：相对自身BN Source Dice−0.0642/+0.0128/−0.2703pp，HD95+0.1511/+0.0078/+1.3991mm（n=24/23/22）。没有整体收益，不扩大此固定分支；[BN统计诊断](../../ped2023_bn_statistics_pilot_20261008.md)是另一项完成的实验，WT改善、TC/ET退化。
- [病例级输出偏置对照24例](../../../outputs/ped2023_pairwise_distillation_20261011/moment_bias_control/INTERPRETATION.md)：从Source出发，以teacher的soft类别总量拟合三个有效偏置，相对Source Dice+0.0584/+0.9263/+0.0282pp、配对HD95−0.1552/−0.1311/−0.0129mm（n=24/23/22）。保留低成本对照，TC低于plain；不是参数数量的严格消融。
- [完整队列传播诊断](../../../outputs/ped2023_pairwise_distillation_20261011/extent_projection/full_cohort/posthoc_receipt.json)已完成：校正相对Source的TC+6.8841pp中，teacher未编辑位置贡献+2.9539pp；WT−0.5366pp和ET−0.2909pp中，同一区域分别贡献−0.4499/−0.4624pp。TC收益与WT/ET损失仍并存于额外变化中，这是输出记账而非因果证明。
- [旧24例传播分解](../../../outputs/ped2023_pairwise_distillation_20261011/transfer_audit/INTERPRETATION.md)与[双观察方向诊断](../../../outputs/ped2023_pairwise_distillation_20261011/directional_support_audit/REPORT.md)保留原证据：两种观察的同向支持未统一胜过同数量Source margin，不据此加入通用门控。
- [归一化算子机制](../../../outputs/ped2023_bn_context_20261011/NORMALIZATION_MECHANISM.md)有CPU Jacobian检查；IN空间耦合可解释上下文影响，但分别训练的IN/BN模型不构成纯因果隔离。[相近文献核对](../../ped2023_context_related_work_20261011.md)表明背景干预、共识和单病例多视图学习已有先例，不宣称这些宽泛概念首次提出。

## 必须保持的实验口径

Source为native四分类IN/Dice+CE/epoch250；BN对照best225。模态顺序`t2f,t1n,t1c,t2w`；BG=0、NC=1、ED=2、ET=3；argmax后WT={1,2,3}、TC={1,3}、ET={3}。不额外HD-BET。

Both-empty Dice=1、HD95=0；one-empty Dice=0、HD95=null。HD95比较用共同有效病例并报告n，不能直接相减有效样本不同的均值。99记录中00137与00121重复，主要分析98组；99描述不赋bootstrap CI。九月份自定义BN、PED2024、脑提取与其他HD95约定不能混入当前协议。

独立验证资源只读核查（2026-10-12）：现有`BraTS-PEDs2024_Training`有261个病例目录，其中260个ID逐项对应九月份已完成的Source GT评价记录；唯一额外ID`00255`缺少`t1c`。`ped_Training`的257个目录中256个ID也出现在这些记录里，额外`00255`虽文件完整，跨版本影像配套尚未核验。不能把这两套已有资源整体写成此前未见的独立测试集；尚未建立新的独立验证队列。核查仅涉及目录、历史manifest/评价记录和既有五值schema，不读取新GT数组、不做推断，也未验证MRI体素或患者身份等价，详见[回执](../../../artifacts/maintenance/decision_consensus_anchor_20261012/target_exposure_preflight.json)。旧自定义BN成绩继续与当前native/PED2023分开。

几何对照借用对应方法的修改预算。plain完整队列并未建立独特精细TC选点优势；不要把plain预算对照误称为前景校正的匹配对照。

## 维护入口和下一步

目标仍是改善GLI→PED的TTA，并讲清动机与证据。用户允许合理WT折中，不新增“三类全部上涨”的硬门槛。学习后四类总量约束24例、四类Source等权池化98组、稳定组织学习锚点24例未改善各自主要对照，不扩大或扫描权重。E/C保留、种子传播、范围校正、各项来源/支持诊断、局部过滤、类别保护和共识保护均已完成，不重跑。margin6_guard与stable_consensus作为并列输出候选，后者只小幅改善TC，ET仍尾部敏感。旧锚点传播诊断、新范围保持学习24例及两项补充均已完成。下一步应收敛简洁方案，围绕TC纠正与WT/ET附带损伤隔离机制；不重复已完成队列，也不凭本轮小均值自动扩98。原native已在未编辑位置施加逐位置max-FG/BG范围锚点，本轮改变E内监督构造，不误写成首次加入空间anchor。全部病例开发暴露，不按GT择例或宣称独立泛化。

公共代码维护在[src/brats_tta/native](../../../src/brats_tta/native/README.md)，资源集中在[configs/paths.yaml](../../../configs/paths.yaml)。产生旧结果的runner、协议和哈希链保留。plain、前景校正、E保留、C保留、种子传播分别冻结14、13、6、7、9项代码/配置依赖，已完成也不代表可以覆盖原快照。传播规则在实验目录`propagation.py`，未并入公共包。公共代码未来修改应记录新版本与新实验依赖。

正式模型在`artifacts/models/`，项目共75个checkpoint路径，包含别名、周期快照和恢复状态；8个重复副本改成只读硬链接。训练输出写新目录。新实验在`outputs/<实验>/`，历史实验在`artifacts/experiments/`，维护在`artifacts/maintenance/`。数据集保留`E:/dataset`及原GLI ZIP，环境为`D:/Environment/miniforge3/envs/cv311/python.exe`。

本次[状态维护记录](../../../artifacts/maintenance/full_cohort_status_20261011)更新过时的运行状态及结果入口，未更改实验代码、模型、数据或旧指标。文件用途见[总说明](../../PROJECT_FILE_GUIDE.md)，历史清理见[存储沿革](../../STORAGE_LAYOUT.md)。
