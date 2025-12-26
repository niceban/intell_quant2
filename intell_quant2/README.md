# IntellQuant 2.0: High-Performance RL Trading System (v-Final)

**IntellQuant 2.0** 是一个基于深度强化学习的工业级量化交易框架。本项目采用了 **LSTM-DDDQN** 架构，支持双 GPU 并行参数搜索（Random Search），并建立了一套严苛的非线性评价体系。

---

## 🧠 核心策略逻辑 (Strategy Spec)

### 1. 交易规则 (Action Masking)
*   **买入**: `月线关注期` (全多头 or DEA+AMA) **AND** `周线 SKDJ 金叉 (State 2)`。
*   **卖出**: `持仓` **AND** `周线 SKDJ 死叉 (State -2)`。
*   **强制风控 (6 线全负)**: `持仓` **AND** (周线 SKDJ_D, SKDJ_K, MACD_DIF, MACD_DEA, MACD_BAR, AMA 全部 < 0)。

### 2. 交易摩擦 (Transaction Costs)
*   **买入成本**: `1.005` (0.5% 滑点+手续费)。
*   **当前/卖出净值**: `0.995` (0.5% 卖出费)。
*   *注：持仓必须涨幅超过 1% 才能实现正向净值回报。*

### 3. 非线性奖励函数 (Reward Formula)
$$R_{\text{pos}} = \sqrt{1 + \text{Ret}} - 1 + \text{Days}^{0.33} \times \text{AvgRet} - \text{MaxDD}$$
$$R_{\text{neg}} = -(1 - \text{Ret})^2 + 1 + \text{Days}^{0.33} \times \text{AvgRet} - \text{MaxDD}$$
*   **特性**: 根号压缩暴利，平方放大亏损，严惩慢牛，鼓励高效捕捉。

---

## 📊 评价体系 (Metrics Legend)

### 训练心跳指标 (TRAIN)
*   **Rw**: 平均单步奖励。
*   **QRR (Q-Realism Ratio)**: Q值真实性比例。$Q_{\text{avg}} / (\text{Rw} / (1-\gamma))$。若 $\gg 1$ 说明模型处于乐观幻觉中。
*   **Act(H/B/S)**: 动作频率分布（Hold / Buy / Sell）。

### 验证集指标 (VAL)
*   **ATR (Avg Trade Return)**: **核心指标**。单笔交易的平均算术收益率。
*   **MKR (Market Return)**: 验证期内标的的平均基准收益（看是否择时跑赢）。
*   **AHD (Avg Hold Days)**: 平均持仓天数。
*   **EFF (Efficiency)**: 持仓效率（ATR / AHD），即每持仓一天能赚多少。
*   **Shp (Sharpe)**: 交易级夏普率（收益/波动比）。
*   **PF (Profit Factor)**: 盈亏比。

---

## 🚀 模块使用

### 1. 启动 Random Search (双卡并行)
```bash
python random_search/run_search.py
```
*   自动利用 `cuda:0` 和 `cuda:1`。
*   结果保存在 `experiments/random_search_TIMESTAMP/`。

### 2. 监控看板
```bash
./monitor_search.sh
```
*   实时观察所有实验的 ATR、胜率和 QRR 状态。

---

## 📂 项目结构
*   `exploring/rl/env.py`: 核心环境，包含风控、成本与奖励逻辑。
*   `random_search/`: 独立搜索模块，包含调度器与 Worker。
*   `preprocessing/visualize_rules.py`: 单股规则验证工具。
*   `deploy_and_run.sh`: 一键同步代码并启动远程任务。
*   `stop_remote.sh`: 一键强力清理远程 GPU 残留进程。
