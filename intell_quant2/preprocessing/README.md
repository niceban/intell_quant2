# intell-quant2

状态隔离的多周期量化指标组件，覆盖日/周级别数据的批量与流式计算。核心目标：回测周级别指标时避免未来数据污染，同时保留批量快速计算能力。示例脚本基于 DuckDB 持久化中间结果，便于迭代调试。

## 配置与状态
- 默认标的：07226 07233 07234 FAS FAZ SOXL SOXS TECL TECS TQQQ SQQQ（见 `src/intell_quant2/config.py`）
- 主周期：日（1D），辅助周期：周（1W），周周期使用 `W-FRI` 分组
- 状态：1=周期内未确认，2=最新一条未确认，3=周期结束可安全落地

## 快速上手
1. 依赖：`pip install -e ".[data]"`（基础依赖 pandas/numpy，`[data]` 额外安装 yfinance/akshare/duckdb）。
2. 拉取日线：`python examples/01_generate_mock_daily.py` 会优先用 yfinance，港股缺数据时回退 akshare；结果存入 `outputs/duckdb/<symbol>.duckdb` 表 `daily`，最后一条标记 `status=2`。
3. 生成周度基表：`python examples/02_build_weekly_frames.py`，为每个 duckdb 生成 `weekly`（周聚合，末周=2）与 `weekly_snapshots`（每日对应周快照，周内=1，周末=3，最新=2）。
4. 日级批量指标：`python examples/03_batch_indicators.py`，写入表 `daily_ind`。
5. 周级流式指标：`python examples/04_stream_indicators.py`，仅在 `status>=2` 行推进状态，写入表 `weekly_ind_stream`。
6. 对比批量与流式：`python examples/05_compare_weekly_modes.py`，在 `status>=2` 的时间点对齐，输出指标最大差异。
7. 一键执行全部示例：`python run_examples.py`（遇错默认停止，可加 `--continue-on-error`）。

## 核心 API
```python
from intell_quant2.pipelines import build_base_frames, compute_indicators
from intell_quant2.config import STATUS_LATEST, STATUS_FINAL

# daily_df 至少包含 ts/open/high/low/close/volume/status，且已按 ts 升序
base = build_base_frames(daily_df)
daily = base["daily"]                   # 原始日线（status 建议全 3，末条可 2）
weekly = base["weekly"]                 # 周聚合（末周 status=2）
weekly_per_day = base["weekly_per_day"] # 每日对应周线快照（周内=1，周末=3，末条=2）

# 日级批量
daily_with_ind = compute_indicators(daily, mode="batch")

# 周级在日维度的流式：仅在 status>=2 时落地状态，其余行返回当前状态快照
mask_update = weekly_per_day["status"] >= STATUS_LATEST
weekly_with_ind = compute_indicators(weekly_per_day.assign(update_flag=mask_update), mode="stream")
```

## 指标实现
- MA：5/10/20/30/60/120/250 简单均线
- BBIBOLL：BBIBOLL=(MA3+MA6+MA12+MA24)/4，STD 窗口 10，倍数 3，流式版本仅在允许更新时推进 STD 与均线
- SKDJ：N=9；fastK 平滑参数 3，K 平滑 3，D 平滑 5，使用 SMA（而非 EMA），流式版按状态维护缓冲区
- MACD：EMA(12/26)，DEA=EMA(DIF,9)，BAR=(DIF-DEA)*2；流式版按状态缓存 EMA
- DMA：短/长均线差，默认 10/50，AMA 为 DMA 的 10 日均值；流式版使用滚动缓冲区

## 回测建议
1. 先计算 `weekly_per_day`，再用 `compute_indicators(..., mode="stream")` 保证仅在周线 `status=3` 更新状态，避免使用当前未完成周的数据。
2. 回测循环时，可同时使用日指标（batch 结果）与周指标（stream 结果）；若需要更严格控制，可用 `update_flag` 覆盖默认状态。
3. 若需要其他周期，复用 `aggregators.weekly_snapshots_for_backtest` 逻辑，调整 `W-FRI` 规则即可。

## 后续扩展
- 接入真实行情源（如券商/交易所 API）填充日线。
- 增加撮合/资金曲线回测模块。
- 加入缓存序列化，便于在实时环境中复用状态对象。
