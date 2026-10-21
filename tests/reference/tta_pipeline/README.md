# 冻结回归参考

此处是 refactor 前的纯算法参考副本，来源与 SHA-256 见 [ORIGIN.json](ORIGIN.json)。`decoder.py` 和 `learning.py` 仅把同目录导入改为相对导入；`senior.py` 只保留 `predict_volume`，去掉模型/数据路径及运行入口。

用于 teacher/decoder 的独立对照，以及必要的模型与概率重放诊断。生产包不得导入这些文件。不要随实现同步修改参考算法，否则会失去回归对照意义。
