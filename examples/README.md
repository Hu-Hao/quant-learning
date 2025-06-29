# Quantitative Trading Framework - Examples

This directory contains example notebooks and scripts demonstrating how to use the quantitative trading framework.

## 📓 Google Colab Example

### [colab_example.ipynb](./colab_example.ipynb)

A comprehensive Jupyter notebook that demonstrates:

- **Installation**: How to install the framework directly from GitHub in Google Colab
- **Real Data**: Fetching Apple (AAPL) stock data from Yahoo Finance
- **Multiple Strategies**: Running backtests with Moving Average, Momentum, and Mean Reversion strategies
- **Performance Analysis**: Comprehensive metrics and visualizations
- **Trade Analysis**: Detailed breakdown of individual trades

### 🚀 Quick Start - Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hu-Hao/quant-learning/blob/main/examples/colab_example.ipynb)

### 💡 What You'll Learn

1. **Framework Installation**
   ```python
   !git clone https://github.com/Hu-Hao/quant-learning.git
   sys.path.append('/content/quant-learning')
   ```

2. **Strategy Setup**
   ```python
   from quant_trading.strategies.moving_average import MovingAverageStrategy
   strategy = MovingAverageStrategy(short_window=10, long_window=30)
   ```

3. **Backtesting**
   ```python
   from quant_trading.backtesting.engine import BacktestEngine
   engine = BacktestEngine(initial_capital=100000)
   engine.run_backtest(data, strategy)  # Only StrategyProtocol supported
   ```

4. **Performance Analysis**
   ```python
   performance = engine.get_performance_summary()
   print(f"Total Return: {performance['total_return']*100:.2f}%")
   ```

### 📊 Expected Results

The notebook will show you:
- Portfolio value charts over time
- Strategy performance comparison
- Risk-return analysis
- Drawdown visualization
- Individual trade details
- Benchmark comparison vs Buy & Hold

### 🧪 Testing Your Installation

Before running the examples, you can verify your installation with:

```bash
# From the project root directory
python examples/test_installation.py

# Or if you cloned to a different location
PYTHONPATH=/path/to/quant-learning python examples/test_installation.py
```

This will test:
- ✅ All module imports
- ✅ Strategy creation and protocol interface
- ✅ Data generation capabilities
- ✅ Backtesting engine functionality
- ✅ Signal generation
- ⚠️ Optional dependencies (matplotlib, yfinance, etc.)

### 🔧 Customization Options

You can easily modify the notebook to:
- **Test different stocks**: Change `symbol = "AAPL"` to any ticker
- **Adjust time periods**: Modify `start_date` and `end_date`
- **Tune strategy parameters**: Experiment with different windows and thresholds
- **Add new strategies**: Implement your own trading logic
- **Change capital**: Adjust `initial_capital` amount

### 📈 Strategy Examples Included

1. **Moving Average Crossover**
   - Fast (10/30 day) and Slow (20/50 day) versions
   - Golden Cross and Death Cross signals

2. **Momentum Strategy**
   - 20-day lookback period
   - 2% momentum threshold
   - RSI confirmation and volatility filtering

3. **Mean Reversion Strategy**
   - Bollinger Bands implementation
   - Z-score based entry/exit signals
   - Statistical mean reversion detection

### ⚠️ Important Notes

- **Educational Purpose**: This is for learning quantitative trading concepts
- **Paper Trading First**: Always test strategies with paper money before real capital
- **Risk Management**: Real trading requires proper risk controls
- **Market Conditions**: Past performance doesn't guarantee future results

### 🛠️ Troubleshooting

**Common Issues:**

1. **Module Import Errors**
   ```python
   # Make sure to run the installation cell first
   !git clone https://github.com/Hu-Hao/quant-learning.git
   sys.path.append('/content/quant-learning')
   ```

2. **Yahoo Finance Data Issues**
   ```python
   # Try a different date range or stock symbol
   symbol = "SPY"  # Use ETF instead of individual stock
   ```

3. **No Signals Generated**
   - Adjust strategy parameters
   - Use longer time periods
   - Check if market conditions suit the strategy

### 📚 Next Steps

After running the basic example:

1. **Explore Parameter Optimization**
   - Grid search for optimal parameters
   - Walk-forward analysis

2. **Add Risk Management**
   - Stop-loss mechanisms
   - Position sizing rules
   - Portfolio risk controls

3. **Multi-Asset Testing**
   - Test across different asset classes
   - Correlation analysis
   - Portfolio diversification

4. **Advanced Strategies**
   - Machine learning integration
   - Alternative data sources
   - Options and derivatives

### 🤝 Contributing

Have improvements or new examples? Please:
1. Fork the repository
2. Create your example
3. Submit a pull request

### 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Hu-Hao/quant-learning/issues)
- **Documentation**: Check the main repository README
- **Community**: Join discussions in GitHub Discussions

---

**Happy Trading! 📈**