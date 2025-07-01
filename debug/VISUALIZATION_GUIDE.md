# 📊 Comprehensive Backtest Visualization Guide

## Overview

The quantitative trading framework now includes powerful visualization features that help you verify your backtest results and understand strategy performance. The visualization shows:

- **Stock price chart** with moving averages and trade signals
- **Buy/sell signal markers** showing exactly when trades occurred
- **Portfolio performance** vs buy-and-hold benchmark
- **Drawdown analysis** showing risk periods
- **Trade win/loss breakdown** for performance verification
- **Performance summary statistics** in an easy-to-read format

## 🚀 Quick Start

```python
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine
from quant_trading.data.data_fetcher import create_sample_data

# 1. Create data and strategy
data = create_sample_data(days=100, trend=0.05, seed=42)
strategy = MovingAverageStrategy(short_window=10, long_window=20, quantity=100)

# 2. Run backtest
engine = BacktestEngine(
    initial_capital=100000,
    allow_short_selling=False,  # Beginner-friendly
    commission=0.001,
    slippage=0.001
)
engine.run_backtest(data, strategy)

# 3. Create comprehensive visualization
engine.plot_backtest_results(
    strategy_name="Moving Average Crossover",
    symbol="DEMO"
)

# 4. Get detailed trade analysis
trade_analysis = engine.analyze_trades()
print("Trade Summary:", trade_analysis["summary"])
```

## 📈 What the Visualization Shows

### 1. **Top Chart: Price with Signals**
- **Black line**: Stock price
- **Orange/Red lines**: Moving averages (strategy indicators)
- **Green triangles (▲)**: Buy signals
- **Red triangles (▼)**: Sell signals

**How to verify**: Check that buy signals occur when short MA crosses above long MA, and sell signals when short MA crosses below long MA.

### 2. **Portfolio Performance Comparison**
- **Blue line**: Your strategy performance
- **Gray dashed line**: Buy-and-hold benchmark
- **Text box**: Strategy return percentage

**How to verify**: Compare if your strategy outperforms simple buy-and-hold.

### 3. **Drawdown Analysis**
- **Red area**: Periods when portfolio was below its peak value
- **Text box**: Maximum drawdown percentage

**How to verify**: Look for extended red periods (high risk) and recovery patterns.

### 4. **Trade Win/Loss Breakdown**
- **Green bar**: Total winning trades value
- **Red bar**: Total losing trades value
- **Numbers**: Dollar amounts and trade counts

**How to verify**: Ensure wins > losses and reasonable win rate.

### 5. **Performance Summary**
- Total return and max drawdown
- Trade statistics (total, wins, losses, win rate)
- Average P&L per trade
- Commission and slippage costs

## 🔍 Strategy Verification Checklist

Use the visualization to verify your strategy is working correctly:

### ✅ Signal Timing
- [ ] Buy signals appear at logical market conditions
- [ ] Sell signals appear at logical market conditions  
- [ ] No signals during flat/sideways markets (if expected)

### ✅ Trade Execution
- [ ] Trades execute at reasonable prices (check slippage)
- [ ] Position sizes match strategy parameters
- [ ] No unexpected short positions when `allow_short_selling=False`

### ✅ Performance Logic
- [ ] Strategy performance makes sense given market conditions
- [ ] Drawdowns occur during expected market stress periods
- [ ] Win/loss ratio aligns with strategy expectations

### ✅ Risk Management
- [ ] Maximum drawdown is acceptable for your risk tolerance
- [ ] Position sizes respect the `max_position_size` parameter
- [ ] Commission and slippage costs are reasonable

## 📋 Advanced Usage

### Multiple Strategy Comparison
```python
strategies = {
    'Fast MA': MovingAverageStrategy(5, 15),
    'Slow MA': MovingAverageStrategy(20, 50)
}

for name, strategy in strategies.items():
    engine = BacktestEngine(initial_capital=100000)
    engine.run_backtest(data, strategy)
    
    # Create separate visualization for each
    engine.plot_backtest_results(strategy_name=name, symbol="COMPARISON")
```

### Detailed Trade Analysis
```python
# Get comprehensive trade breakdown
trade_analysis = engine.analyze_trades()

# Print trade summary
summary = trade_analysis["summary"]
print(f"Win Rate: {summary['win_rate']}")
print(f"Average P&L: {summary['avg_pnl_per_trade']}")
print(f"Largest Win: {summary['largest_win']}")
print(f"Largest Loss: {summary['largest_loss']}")

# Examine individual trades
for trade in trade_analysis["trades"][:5]:  # First 5 trades
    print(f"Trade {trade['trade_number']}: {trade['side']} "
          f"{trade['quantity']} @ {trade['entry_price']} → "
          f"{trade['exit_price']}, P&L: {trade['pnl']}")
```

### Custom Visualization Settings
```python
# Don't show plot immediately (for batch processing)
fig = engine.plot_backtest_results(show_plot=False)

# Save to file
if fig:
    fig.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
```

## 🛠️ Installation Requirements

For full visualization functionality, install matplotlib:

```bash
pip install matplotlib
```

The framework will work without matplotlib but won't display charts.

## 🔧 Troubleshooting

### No Signals Generated
- Check if your data has enough variation for strategy conditions
- Verify strategy parameters (window sizes, thresholds)
- Try longer time periods or different market conditions

### No Trades Executed
- When `allow_short_selling=False`, sell signals only close long positions
- Check if strategy generated buy signals before sell signals
- Verify sufficient capital for trade execution

### Visualization Not Showing
- Install matplotlib: `pip install matplotlib`
- Check that `show_plot=True` (default)
- Verify backtest was run before calling `plot_backtest_results()`

## 💡 Best Practices

1. **Always visualize** your backtest results to verify strategy logic
2. **Check signal timing** against market conditions and moving averages
3. **Compare performance** to buy-and-hold benchmark
4. **Analyze drawdown periods** to understand strategy risk
5. **Review individual trades** to spot patterns or issues
6. **Test multiple market conditions** (trending, sideways, volatile)

## 📚 Example Output Interpretation

### Good Strategy Signals:
- Buy signals during uptrend momentum
- Sell signals before major downturns
- Reasonable win rate (>40-50%)
- Controlled drawdowns (<20%)

### Warning Signs:
- Too many signals (overtrading)
- Poor signal timing (buying peaks, selling bottoms)
- High drawdowns (>30%)
- Very low win rate (<30%)

The visualization makes it easy to spot these patterns and verify your strategy is working as intended!