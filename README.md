# Quant Trading Framework

A production-ready, modular quantitative trading framework built with clean code principles and comprehensive testing.

## 🌟 Features

- **Modular Architecture**: Clean separation of concerns with well-defined interfaces
- **Realistic Backtesting**: Variable slippage, commission costs, position limits
- **Multiple Strategies**: Moving averages, momentum, mean reversion with extensible base classes
- **Comprehensive Testing**: Full unit test coverage with realistic scenarios
- **Type Safety**: Type hints throughout for better code documentation
- **Configuration Management**: YAML/JSON configuration with validation
- **Error Handling**: Robust error handling and logging throughout
- **Performance Metrics**: Detailed risk and performance analysis

## 📁 Project Structure

```
quant_trading/
├── data/                    # Data fetching and processing
│   ├── __init__.py
│   └── data_fetcher.py     # DataFetcher with error handling
├── strategies/              # Trading strategies
│   ├── __init__.py
│   ├── base_strategy.py    # Abstract base class
│   ├── moving_average.py   # MA crossover strategy
│   ├── momentum.py         # Momentum trading
│   └── mean_reversion.py   # Mean reversion strategy
├── backtesting/            # Backtesting engine
│   ├── __init__.py
│   ├── engine.py          # BacktestEngine with slippage
│   └── metrics.py         # Performance metrics
├── utils/                  # Utilities and indicators
│   ├── __init__.py
│   └── indicators.py      # Technical indicators
├── config/                 # Configuration management
│   ├── __init__.py
│   └── settings.py        # Config classes with validation
└── tests/                  # Comprehensive test suite
    ├── __init__.py
    ├── test_backtesting.py
    └── test_strategies.py
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Hu-Hao/quant-learning.git
cd quant-learning

# Install dependencies (if using external data sources)
pip install pandas numpy pyyaml

# Run the example
python example_usage.py
```

### Basic Usage

```python
from quant_trading.data.data_fetcher import DataFetcher, DataSource
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

# 1. Get data
fetcher = DataFetcher(DataSource.SYNTHETIC)
data = fetcher.fetch_data("DEMO", "2023-01-01", "2023-12-31")

# 2. Create strategy
strategy = MovingAverageStrategy(short_window=10, long_window=30)

# 3. Run backtest
engine = BacktestEngine(
    initial_capital=100000,
    commission=0.001,
    slippage=0.0005
)
engine.run_backtest(data, strategy)

# 4. Analyze results
performance = engine.get_performance_summary()
print(f"Total Return: {performance['total_return']:.2%}")
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
```

## 🧠 Available Strategies

### 1. Moving Average Strategy
Classic trend-following using MA crossovers:
```python
strategy = MovingAverageStrategy(
    short_window=10,    # Fast MA period
    long_window=30,     # Slow MA period
    quantity=100        # Shares per trade
)
```

### 2. Momentum Strategy
Trades based on recent price momentum:
```python
strategy = MomentumStrategy(
    lookback_period=20,         # Momentum calculation period
    momentum_threshold=0.02,    # 2% threshold
    rsi_period=14,             # RSI confirmation
    quantity=100
)
```

### 3. Mean Reversion Strategy
Trades price deviations from statistical mean:
```python
strategy = MeanReversionStrategy(
    window=20,                  # Statistical window
    entry_threshold=2.0,        # Entry at 2 std devs
    exit_threshold=0.5,         # Exit at 0.5 std devs
    use_bollinger_bands=True    # Additional confirmation
)
```

## 🔧 Creating Custom Strategies

Extend the `BaseStrategy` class:

```python
from quant_trading.strategies.base_strategy import BaseStrategy, Signal, SignalType

class MyCustomStrategy(BaseStrategy):
    def __init__(self, param1=10, param2=0.5):
        super().__init__(
            name="MyCustomStrategy",
            param1=param1,
            param2=param2
        )
    
    def _setup_indicators(self, data):
        # Setup any indicators needed
        pass
    
    def generate_signals(self, data):
        # Your strategy logic here
        signals = []
        
        if self._should_buy(data):
            signal = Signal(
                symbol='default',
                action=SignalType.BUY,
                quantity=100,
                price=data['close'].iloc[-1],
                confidence=0.8
            )
            signals.append(signal)
        
        return signals
    
    def _should_buy(self, data):
        # Your buy logic
        return True  # Placeholder
```

## ⚙️ Configuration Management

### YAML Configuration
```yaml
# config.yaml
backtest:
  initial_capital: 100000.0
  commission: 0.001
  slippage: 0.0005
  max_position_size: 0.2

strategy:
  name: "MovingAverage"
  parameters:
    short_window: 10
    long_window: 30
    quantity: 100

data:
  source: "synthetic"
  start_date: "2023-01-01"
  end_date: "2023-12-31"
```

### Loading Configuration
```python
from quant_trading.config.settings import Config

# From file
config = Config.from_file("config.yaml")

# From environment variables
config = Config.from_env(prefix="QUANT_")

# Use default
config = get_default_config("MovingAverage")
```

## 📊 Performance Analysis

### Basic Metrics
```python
performance = engine.get_performance_summary()

print(f"Total Return: {performance['total_return']:.2%}")
print(f"Annualized Return: {performance['annualized_return']:.2%}")
print(f"Volatility: {performance['volatility']:.2%}")
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
```

### Advanced Analysis
```python
from quant_trading.backtesting.metrics import MetricsCalculator

calc = MetricsCalculator(risk_free_rate=0.02)

# Performance metrics
perf_metrics = calc.calculate_performance_metrics(engine.portfolio_values)
print(f"Sortino Ratio: {perf_metrics.sortino_ratio:.3f}")
print(f"Calmar Ratio: {perf_metrics.calmar_ratio:.3f}")

# Risk metrics
risk_metrics = calc.calculate_risk_metrics(engine.portfolio_values)
print(f"VaR (95%): {risk_metrics.var_95:.2%}")
print(f"Skewness: {risk_metrics.skewness:.3f}")

# Generate comprehensive report
report = calc.generate_report(engine.portfolio_values, strategy_name="MyStrategy")
print(report)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest quant_trading/tests/

# Run with coverage
python -m pytest quant_trading/tests/ --cov=quant_trading

# Run specific test
python -m pytest quant_trading/tests/test_backtesting.py::TestBacktestEngine::test_slippage_calculation
```

### Test Coverage
- **Backtesting Engine**: Order execution, slippage, position management
- **Strategies**: Signal generation, parameter validation, indicator calculation
- **Data Handling**: Fetching, validation, error handling
- **Configuration**: Loading, validation, type safety

## 🔬 Slippage Modeling

The framework includes realistic slippage modeling:

- **Variable Slippage**: Adjusts based on market volatility
- **Order Size Impact**: Larger orders have higher slippage
- **Volatility Scaling**: 0.5x to 3x multiplier based on recent volatility

```python
# Configure slippage
engine = BacktestEngine(
    slippage=0.001,  # Base slippage (0.1%)
    # Actual slippage varies from 0.05% to 0.3% based on conditions
)
```

## 📈 Risk Management

Built-in risk controls:

- **Position Size Limits**: Maximum position as % of capital
- **Capital Requirements**: Prevents over-leveraging
- **Commission and Slippage**: Realistic transaction costs

```python
engine = BacktestEngine(
    max_position_size=0.2,  # Max 20% in any position
    commission=0.001,       # 0.1% commission
    slippage=0.0005        # 0.05% base slippage
)
```

## 🛠️ Development Guidelines

### Code Quality
- **Type Hints**: All functions have proper type annotations
- **Documentation**: Comprehensive docstrings
- **Error Handling**: Graceful error handling with logging
- **Testing**: High test coverage with realistic scenarios

### Adding New Features
1. **Create Feature Branch**: `git checkout -b feature/new-strategy`
2. **Write Tests First**: Test-driven development
3. **Implement Feature**: Follow existing patterns
4. **Update Documentation**: Keep README current
5. **Submit PR**: Include tests and documentation

## 📚 Examples

See `example_usage.py` for comprehensive examples including:
- Strategy comparison
- Configuration management
- Performance analysis
- Error handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built with modern Python best practices
- Inspired by professional quantitative trading systems
- Designed for educational and research purposes

---

**Happy Trading! 📈**