# Evolution Search (ES) Module - 启发式交易规则搜索与进化

该模块是一个独立于 DRL（强化学习）分支的高性能搜索系统。它利用 **神经进化 (NeuroEvolution)** 和 **贝叶斯牵引 (Bayesian Steering)** 结合 **加权机器学习蒸馏** 技术，旨在从万亿级的买卖点规则组合中，筛选出具备最高累积期望收益的“狙击手”策略。

## 1. 核心设计哲学 (Core Philosophy)

*   **右侧交易确认 (Confirmation-based)**：模型不试图在混沌中“预言”涨跌，而是通过 19周+1日的形态序列识别具备统计优势的“启动迹象”。
*   **非线性风险定价**：引入奖励离散化与非线性映射，认为收益获取的难度随量级指数增长，从而强迫模型关注高频率、高确定性的波段，而非偶然的暴利。
*   **规则与权重的端到端协同**：系统不仅优化 LSTM 的神经连接权重，还同步进化买卖点原子规则的动态掩码 (Rule Mask)。

## 2. 特征工程设计 (Feature Engineering)

系统输入 34 维连续/离散混合向量，分为两大核心板块：

### A. 市场形态序列 (Market States, 20维)
*   **混合时序窗口**：由 19 步历史周线 (Status=3) 和 1 步当前日线 (Real-time Snapshot) 构成。
*   **指标矩阵**：每步包含 KDJ, MACD, DMA, AMA 的离散化状态 `{-2, -1, 0, 1, 2}`。
*   **逻辑深度**：LSTM 能够跨越 20 个时间步（约 5 个月）捕捉底背离、多头陷阱、水下金叉等复合形态。

### B. 账户动量特征 (Account States, 14维)
*   **盈亏感官**：包括当前 PnL、平均持仓收益、以及 PnL 的一阶差分 (d1) 和 二阶差分 (d2)。
*   **回撤感知**：实时监控持仓过程中的最大回撤，赋予模型“利润保护”的本能。

## 3. 训练与进化流程 (Training Pipeline)

本模块采用基于 **OpenAI-ES** 改进的并行进化算法，其流程如下：

1.  **种群 Ask (Generation)**：
    *   **GPU 直接变异**：在 Master 权重的中心，于 GPU 内部直接生成 32 个扰动子代，彻底消除 CPU-GPU 权重拷贝延迟。
    *   **贝叶斯规则抽样**：每个 Agent 的规则组合 (Rule Mask) 根据历史 Winner 的统计分布进行抽样，伴随 20% 的随机探索率。
2.  **高通量评估 (Massive Parallelism)**：
    *   **32,768 个并行环境**：PopSize 32 * 1024 Envs/Agent。单次推理 Batch Size 达 1024，填满 MPS/CUDA 计算核心。
    *   **共享内存模式**：所有环境共享同一份显存内的 DuckDB 市场数据，大幅降低显存占用。
3.  **奖励计算 (Reward Shaping)**：
    *   **离散化档位**：奖励映射至 `[-5, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 5]`。
    *   **卖出激励**：`Sell Exec = max(PnL - term, PnL)`。奖励成功的逃顶和超前止损行为。
4.  **贝叶斯 Tell (Update)**：
    *   根据 Fitness (Total Reward) 筛选前 20% 的 Winner。
    *   **加权规则蒸馏**：利用 **Random Forest Classifier** 分析规则组合，样本权重为 `|reward| + 0.01`。
    *   **先验修正**：将 RF 的特征重要性反馈给规则先验概率，实现进化速度的自我加速。

## 4. 运行说明

```bash
# 启动本地高负载进化搜索
python evolution_search/run_es_local.py
```

*   **硬件优化**：自动适配 Apple Silicon (MPS) 或 NVIDIA (CUDA)。
*   **监控量指标**：实时输出 `FitBest` (最高分)、`Pos` (持仓率)、`Trd` (交易频率) 及 `Speed` (每秒模拟步数)。
*   **产出**：定期保存 `evolution_search/best_checkpoint.pth`，包含最优神经网络权重与规则先验分布。
