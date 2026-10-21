# 公共流水线重构验证

2026-10-22。完整方法已提取到 [pipeline](../../src/brats_tta/pipeline/README.md)，运行入口为 `scripts/project.ps1 adapt`。历史实验仍保留自己的冻结源码和结果。

## 本地验证范围

| 检查 | 结果 |
|---|---|
| 默认 CPU suite | 7 个模块、45 项通过 |
| Ruff 关键错误 | 通过 |
| 冻结来源 | 19 份原文件保持 SHA-256 不变 |
| 提取的纯函数/类 | 24 个定义的 AST 一致，仅忽略 docstring 行末空格 |
| Native 缓存 decoder | 同一个病例的 8,928,000 体素与旧结果完全一致 |
| Native 单视角 8 步 GPU 学习 | 复用该例 teacher 缓存；逐步 loss、affine、bias、最终 mask 完全一致 |
| Senior 完整单病例 GPU 流程 | 从 MRI 重新生成 10 种观察并学习 8 步；Source、teacher、最终 mask 及 teacher 增补完全一致 |

CPU suite 还覆盖融合梯度与直接完整计算图、8 步优化、异常/成功状态恢复、几何和模态顺序、CLI 防覆盖及产物记录。Source 权重、数据和旧实验目录都不需要上传才能运行这些 CPU 测试。

## Senior 的数值限制

Senior 的适应参数没有通过逐位一致回归：5,836/5,888 个 affine 标量有差异，最大绝对差 0.0009176917，差值 L2 为 0.00202748；最大逐步 loss 差 0.00102448。

相同网络、输入和上游梯度的 128³ 单 patch 隔离探测中，旧版 CUDA 重放自身两次梯度最大差为 0.0000606615，旧与新最大差为 0.0000855661；32³ 探测逐位相同。严格 deterministic 模式报告 `max_pool3d_with_indices_backward_cuda` 缺少确定性实现。这说明原 GPU 路径也不保证逐位重复，但没有逐项解释完整 8 步中的所有差异。

没有重跑完整 98 例、恢复暂停的 Senior 队列或重做完整八视角学习的 GPU 轨迹验证。单病例一致不代表所有病例都一致，以上不是新的效果评测。既有环境是 Windows/cv311、PyTorch 2.11.0+cu128、RTX 4060 Laptop；未升级依赖。

## GitHub 发布范围

本分支基于远端 `main`，包含公共代码、测试、工程入口、文档及方法图。数据、权重、病例结果、缓存、原历史 runner 和本地维护回执没有随此分支上传。文档中的这些路径指原工作区资产；只克隆 GitHub 不能重建整个研究资产库。

默认 CPU 测试和 Ruff 已在隔离发布工作树运行通过。云端 CI 的实际状态查看 [GitHub Actions](https://github.com/dengjie86/brats-tta/actions)，不把本地通过当作远端通过。原工作区的已有改动保持原样。
