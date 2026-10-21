# 实验完成后自动接续研究

**2026-10-13交接更正：本文件描述尚未端到端验证的CLI原型，不控制桌面Goal开关，未启用。用户优先考虑桌面Scheduled，尚未创建。下面的旧线程ID不可直接用于新对话；CLI接续是否影响原Goal状态也未验证。当前推荐入口是[根目录交接](../HANDOFF.md)。**

此调度器把等待交给普通 Python 进程。等待期间每 30 秒检查本地文件和进程身份，不调用语言模型；实验完成或失败时才通过 `codex exec resume` 接续指定会话。模型分析后登记下一实验并结束本轮，调度器继续等待，因此正常情况下不需要逐轮手动恢复。

## 使用边界

- 适用于本仓库的 Windows 本地队列：成功写出 `queue_complete.json` 且 `passed=true`，各阶段写出 `*_stage.json` 和 `exit_code`。只有 prediction 完成不算整轮完成。
- 这是一个 CLI 调度流程，不是给桌面 Goal 新增设置。使用时让原 Goal 保持暂停，避免两个调度器同时驱动同一研究；也不要在它执行 Codex 分析时同时操作同一线程。要停下讨论，先停止派发并等待当前分析结束。
- 保持电脑和启动脚本的终端运行；当前没有安装 Windows 服务、开机自启或睡眠唤醒。
- 2026-10-13 已验证当前会话可以被本地 `app-server` 只读读取。真实模型接续、认证、Windows sandbox 可用性及桌面界面同步尚未端到端验证。`run` 会产生真实模型调用；一次接口读取成功不等于全链路已跑通。
- CLI 使用当前会话及现有账户配置，不另选模型，不修改全局配置。命令使用 `workspace-write` 和 `approval_policy=never`；遇到权限、认证或模型错误会停止并保留日志，不自动放宽权限或反复重试。

## 首次启动（在仓库根目录执行）

当前实验已经生成最终完成凭据，因此以下启动将直接触发一次结果分析。仅希望看脚本时不要执行 `run`。

```powershell
& 'D:\Environment\miniforge3\envs\cv311\python.exe' scripts/research_dispatch.py enqueue --thread 01a117dd-8008-7162-bd73-60e9b21043d0 --experiment outputs/ped2023_extent_factor_learning_20261013
& 'D:\Environment\miniforge3\envs\cv311\python.exe' scripts/research_dispatch.py run
```

如果首次接管的是尚未完成的实验，`enqueue` 还必须传入真实队列 PID：`--pid <PID>`。脚本记录进程创建时间，避免把复用的 PID 当作原实验。登记前确认实验使用上述完成凭据格式。

模型启动后续长实验时，必须用同一个 `enqueue` 命令登记其目录及队列 PID，然后立即结束本轮。不要创建/恢复 Goal，不要启动第二个调度器，不要调用模型逐病例轮询。保持原科学协议、冻结依赖和既有授权范围；不存在合理后续工作或需要用户输入时不登记新任务，调度器自然退出。

## 查看和停止

```powershell
& 'D:\Environment\miniforge3\envs\cv311\python.exe' scripts/research_dispatch.py status
& 'D:\Environment\miniforge3\envs\cv311\python.exe' scripts/research_dispatch.py stop
```

`stop` 阻止下一次派发，不杀死正在执行的模型或实验。需要立即终止当前模型时，使用其运行终端/客户端的中断控制；不要误杀训练。重启前显式移除 `outputs/research_dispatch/STOP`。

所有记录在 `outputs/research_dispatch/`：`pending.json` 是下一任务，`events/` 保存一次性派发凭据、提示、CLI 日志、最后答复和退出码。文件锁防止重复启动；每个事件在调用模型前登记一次性凭据。调用失败或在登记后崩溃时不会盲目重放，需要先核实实际执行情况。不要删除这些凭据来强行重复运行。

实现：[research_dispatch.py](../scripts/research_dispatch.py)。协议参考：[Codex App Server](https://learn.chatgpt.com/docs/app-server)。本工具尚未启动真实自动研究循环，不代表原研究 Goal 已恢复或实验效果已经审阅。

本地验证：`cv311/python.exe -m pytest -q tests/test_research_dispatch.py`，13 项通过，包括模拟两轮连续交接和真实 Windows 进程身份/文件锁检查。模拟没有调用模型；CLI 命令结构和当前线程只读访问已验证，真实接续仍需首次运行确认。
