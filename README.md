# 整数量化分析与部署工具（Python + NumPy）

一个完全用 NumPy 实现的训练、**整数真实模拟**量化、逐层敏感性分析与混合精度
搜索工具，配套一个零依赖（标准库 http.server + 原生 SVG）的离线可视化页面。

针对的真实问题：模型压成 INT8/INT4 后，**总体准确率只掉一点，但少数高风险
类别的识别率大幅下滑**，业务方无法接受。工具帮助定位是哪几层、哪几个通道
造成的，并在给定精度损失上限下自动选择每层精度。

## 它能回答的五个问题

1. **范围怎么定、校准样本不好会怎样** — 权重对称 per-channel/per-tensor、
   激活非对称 affine（含 zero-point）；MinMax 与分位截断两种定范围依据；
   分层 / 顺序 / 关键类偏置 / 不含关键类 / 小样本五种校准集直接对比。
2. **哪几层、哪几个通道掉点** — 逐层 SQNR、单层消融（只压该层）、逐通道
   权重/激活范围相对中位数的倍数，自动标记 outlier 通道。
3. **每层该用什么精度** — 每层在 FP32/INT8/INT4 间选择，双约束（总准确率
   上限 + 关键类召回上限）下全枚举求 Pareto 前沿，输出整数周期与字节，
   同一上限下的多种可行选法并排比较。
4. **关键类别是否守得住** — 所有方案都输出逐类别 precision/recall/f1 与
   混淆矩阵，关键类召回作为独立硬约束参与搜索。
5. **误差在哪一层被放大** — 选一个样本，逐层对比量化前后输出（SQNR、余弦、
   平均/最大偏差、分量级配对折线）。

## 整数真实性（不是假量化）

量化推理不是"量化完反量化再走浮点 matmul"：

* 权重量化到 int32 网格，激活量化到带整数 zero-point 的有符号网格；
* 所有 GEMM / 卷积（含注意力的 QK^T、A@V）用 **int64 整数累加**；
* 重定标用定点乘移 `out = round(acc * M0 / 2^shift) + zp`（TFLite 约定，
  `M0`、`shift` 均为整数），偏置也是整数；
* softmax / LayerNorm 走浮点微内核（与真实整数引擎一致的混合边界）；
* `tests/test_integer_path.py` 会断言到达重定标的累加器全部是整数 dtype。

## 四个模型（结构不同、量化敏感度不同）

| 模型 | 数据集（真实数据） | 结构 | 量化敏感度 |
|---|---|---|---|
| `mlp_robust` | OpenML credit-g 德国信用风控 | 全连接 61-64-32-2 | 低 |
| `mlp_sensitive` | credit-g（关键类=高风险 bad） | 全连接 + **FP 等价注入的 outlier 通道** | 高 |
| `cnn_sensitive` | sklearn 手写数字（关键数字降采样到 ~8%） | 两层卷积 im2col + 全连接 | 中 |
| `attn_sensitive` | 同上，8x8 视为长度 8 的序列 | 单头自注意力 + FFN | 高（INT4 明显） |

outlier 注入在浮点下**数学恒等**（正缩放穿过 ReLU 精确：`relu(sx)=s·relu(x)`），
由测试验证相对误差 < 1e-6，但在 per-tensor 整数网格下会放大整层量化误差。

## 快速开始

```bash
# 一键：训练 -> 分析 -> 起页面
bash experiments/run_all.sh
# 或分步
python3 -I tools/run_iso.py experiments/01_train_models.py
python3 -I tools/run_iso.py experiments/02_run_analysis.py
python3 -I tools/run_iso.py -m quant_tool.webapp.server 8765
# 打开 http://127.0.0.1:8765
```

`tools/run_iso.py` 以 `python3 -I` 隔离模式运行（cwd 不进 sys.path、不读
环境变量），并只显式加入可信的用户 site-packages，使数据缓存目录里的
不可信文件无法通过模块名劫持导入。

## 目录结构

```
quant_tool/
  data.py                 真实数据集（OpenML credit-g / sklearn digits）
  metrics.py              总体 + 逐类别指标、混淆矩阵
  models/                 纯 NumPy 层（Linear/Conv/Attention/LN）、Adam、
                          FP 等价 outlier 注入、checkpoint
  quant/
    kernel.py             量化原语 + 定点乘移重定标 + SQNR
    observer.py           MinMax / 分位截断观察器
    engine.py             混合精度整数模拟推理引擎
    cost_model.py         模型字节 + 整数 MAC/周期成本
  analysis/
    calibration.py        五种校准集构造
    sensitivity.py        逐层 SQNR / 单层消融 / 通道 outlier / 单样本追踪
    mixed.py              双约束混合精度搜索 + Pareto 前沿
  webapp/                 离线仪表盘（stdlib server + 原生 SVG/JS）
experiments/
  01_train_models.py      训练并保存四个 FP32 checkpoint
  02_run_analysis.py      生成 experiments/results/results.json
  run_all.sh              一键流水线
tests/                    数值梯度、学习冒烟、定点数学、整数路径、引擎
                          健全性、CDP 页面交互测试
```

## 典型结论（以 mlp_sensitive 为例，per-tensor）

| 方案 | 准确率变化 | 关键类召回变化 | 字节 | 估计整数周期 |
|---|---|---|---|---|
| 全 INT8 | −0.8pt | **−5.6pt** | 6800 | 6016 |
| 全 INT4 | −3.2pt | **−8.5pt** | 3792 | 3008 |
| 工具选择混合（L4 保留 FP32，其余 INT4） | −0.4pt | **−1.4pt** | 4016 | 3232 |

整体准确率几乎不变时关键类召回已经显著下滑；混合方案用很小的体积/周期
代价把关键类召回损失从 8.5pt 收回到 1.4pt，同时满足两条约束。
