# Native BN源统计路线：固定24例诊断

2026-10-08。当前状态：固定24例已完成，两个组均失败。[完成结果与核验](ped2023_bn_statistics_pilot_20261008.md)为最新入口。该方案增加已训练的BN模型作为独立研究分支；不替换当前IN主线，不合并两模型输出，不按PED分数选模型或epoch。下面为预先准备与历史范围说明，不能按旧状态重复启动。

此前要求额外确认是对历史交接状态的过度解读，已撤销：同一交接明确说明IN只针对原评估、未限定最终研究设计；当前持续研究目标包含本地有界对照。没有收到新的BN专项答复，也没有把等待当授权。详见[执行范围说明](../outputs/ped2023_bn_statistics_pilot_20261008/EXECUTION_SCOPE.md)，其时间先于任何PED前向。原准备回执和科学协议hash全部保留；下方旧等待措辞只用于追溯，已被此处取代。

## 为什么提出

现有IN教师、参数锚点、尺度及跨序列实验均未建立稳定的完整四类纠错。缓存联合审查进一步表明，原/新增12例的NC→ED错误至少59.30%/83.22%在Source及全部十观察中始终PNC<PED，因此这些概率的非负、类别共享加权也不能将其改为NC。这个结论只约束已保存观察及混合方式，不是不可适应的证明。

已有native BN控制保存了GLI训练期统计。它允许直接比较“固定源统计”和“当前病例patch统计”，这是当前无running statistics的IN模型无法执行的同模型对照。不预言BN更好，也不把更换源模型的收益当作TTA收益。

## 已准备并核实的资源

- 模型目录：`D:/Workspace/deeplearn/3dunet/artifacts/models/bn_dicece/run/nnUNet_results/Dataset503_BraTS_Classes/nnUNetTrainerBraTSBN250__nnUNetPlans__3d_fullres`。
- 唯一预选checkpoint：`fold_0/checkpoint_best.pth`，epoch225，由原GLI完整验证集选择，SHA256 `2c3917ae7a41a13f04c2825de407ecf398dde4cd6ae3a9b6dede4e50fc78ea64`。不在PED比较225/250再择优。
- [CPU核验](../outputs/ped2023_bn_resource_review_20261008/inspection.json)：22个BN，31,199,796参数，eps1e-5、momentum.1、affine/running statistics均有效；每层running variance有限且正。全部权重严格载入，网络eval，无GPU初始化或PED/GLI图像读取。
- 与当前IN的`3d_fullres`配置递归比较，仅`norm_op`不同，dataset元数据一致。两次独立训练后的权重不同，不能声称归一化统计是跨模型分数差异的唯一原因。
- `check_resource.py`可复现CPU检查。首次脚本命名`inspect.py`遮蔽标准库导致导入失败，改名后exit0；失败日志保留，未安装或升级依赖。

## 请求确认的具体实验范围

1. 仅本地RTX4060，复用已冻结original12/additional12和native预处理。无新源训练、GLI样本、付费机器或云任务。预计约10–20分钟，以实际耗时为准，输出保存在D盘，预计不超过3GB。
2. 两个真实前向条件：BN Source用原running statistics；BN batch-stat使用每个128³推断patch的当前batch/spatial statistics，沿用所有卷积和affine权重。后者不是全体积AdaBN，不累计跨patch/case的EMA统计，不使用GT、熵优化或参数更新。
3. 二者都用native .5滑窗、Gaussian融合、八镜像、AMP及完整四类argmax。先完成两组全部预测，再评价。每例恢复源状态，确认磁盘checkpoint及所有参数/源buffer逐值不变。统计切换实现、归一化identity和形状需预检；不直接改旧IN adapter的断言。
4. BN batch-stat以本例BN Source为基准，增加全crop逐修改方向/数量匹配的margin、geometry和random控制。IN既有Source作为资源比较参考，不能把BN相对IN的差值记为TTA收益。
5. 报告两组分别WT/TC/ET Dice和HD95、NC/ED组织、真ED受损、全漏/零重叠、每例胜负和源高置信NC→ED错误；ET缺失两组均0，不能宣称验证缺类鲁棒性。
6. 保持既有推进条件：NC/TC各至少+.5pp，三区均值提高，WT/ET及ED存在子集下降不超过.2pp，净NC纠正，无新增全漏/零重叠，优于匹配控制；两个组分别通过。失败就停止此统计切换配置，不调统计混合系数或选择有利epoch，不自动扩99。

这只是检验统计与新源模型信息的有界诊断，不是创新方法或域适应完成。若BN Source本身改善，而batch-stat无收益，只能报告模型差异；是否开展新的源统计约束适应需要据此另定机制和协议。

## 历史等待原因（已更正，当前执行状态以上方为准）

[HANDOFF](../HANDOFF.md)“下一步工作边界”明确写有“PED上新BN的Source/TTA尚未启动，需要对应请求后进行”，当前多轮主线保持单一IN。这里只做可逆、只读资源核查并形成具体方案；等待用户确认是否扩大到这个已训练BN分支。没有启动PED前向或更新参数。

## 已完成的实现准备

目录：[BN pilot准备](../outputs/ped2023_bn_statistics_pilot_20261008/README.md)。独立`bn_adapter.py`只加载best225，`patch_statistics`保持eval并暂时移除running buffers，退出或异常时恢复原tensor对象和flags；不改旧IN adapter。`runtime.py`/`run.py`准备了真实预检及24例预测入口，尚未调用。

[cpu_check.json](../outputs/ped2023_bn_statistics_pilot_20261008/cpu_check.json)记录10项检查通过，3.78s：独立float64均值/biased variance公式N=1/N=2最大误差1.7764e-15、固定源统计公式、顺序无状态、异常后Source输出和状态精确恢复、嵌套切换提前拒绝、卷积/affine/dropout模式不变、全12类修改方向控制及identity。实际22BN checkpoint仅CPU载入/切换/恢复，没有真实模型forward、图像或GT读取，CUDA未初始化。

[preparation.json](../outputs/ped2023_bn_statistics_pilot_20261008/preparation.json)已固定代码/协议/依赖和24例纯图像身份元数据，Python语法检查通过。准备状态明确标记用户选择待答；它不是启动授权。真实GPU预检、PED预测及其指标尚无结果，不能把CPU测试当适应有效性证明。

目标恢复后又补齐`metrics.py`、`evaluate.py`和`report.py`，用于完整三域/组织、方向损伤、共同有效HD95、逐例风险、强对照及高置信NC→ED统计。BN相对IN明确标记为资源比较；推进条件始终比较batch_stats与自身BN Source。评价入口必须先有全部24例预测完成回执并核对代码/依赖hash，才读原始GT。

[evaluation_cpu_check.json](../outputs/ped2023_bn_statistics_pilot_20261008/evaluation_cpu_check.json)四组检查通过：7个手算体素的改对/破坏、物理间距HD95与空类约定、既有24例IN实验完整汇总/配对区间精确回归，以及WT超损/与geometry持平/ED证据缺失/新增全漏必须失败的合成条件。最终检查.18s，只读取旧IN指标JSON，不读原始MRI/标签或新BN结果。初始通过回执另存，补充评价入口的依赖hash断言后重新检查通过；未改预测协议或阈值。真实预检、预测、评价均未执行，资源选择仍待答。
