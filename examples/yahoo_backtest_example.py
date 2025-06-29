#!/usr/bin/env python3
"""
Quantitative Trading Framework - Yahoo Finance Backtest Example

This script demonstrates how to use the quantitative trading framework
to run backtests with real Yahoo Finance data for Apple stock.

Usage:
    python yahoo_backtest_example.py

Requirements:
    pip install yfinance pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our quantitative trading framework
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.strategies.momentum import MomentumStrategy
from quant_trading.strategies.mean_reversion import MeanReversionStrategy
from quant_trading.backtesting.engine import BacktestEngine


def fetch_stock_data(symbol, days=730):
    """
    Fetch stock data from Yahoo Finance
    
    Args:
        symbol: Stock ticker symbol
        days: Number of days of historical data
        
    Returns:
        pandas.DataFrame: OHLCV data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"📈 Fetching {symbol} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    # Convert column names to lowercase for compatibility
    data.columns = [col.lower() for col in data.columns]
    
    print(f"✅ Successfully fetched {len(data)} trading days of data")
    return data


def setup_strategies():
    """Setup trading strategies with different parameters"""
    strategies = {
        'Moving Average (Fast)': MovingAverageStrategy(
            short_window=10,
            long_window=30,
            quantity=100
        ),
        'Moving Average (Slow)': MovingAverageStrategy(
            short_window=20,
            long_window=50,
            quantity=100
        ),
        'Momentum': MomentumStrategy(
            lookback_period=20,
            momentum_threshold=0.02,
            quantity=100,
            volatility_filter=True
        ),
        'Mean Reversion': MeanReversionStrategy(
            window=20,
            entry_threshold=2.0,
            exit_threshold=0.5,
            quantity=100,
            use_bollinger_bands=True
        )
    }
    
    print("🎯 Trading strategies configured:")
    for name, strategy in strategies.items():
        print(f"   • {name}")
    
    return strategies


def run_backtest(data, strategy, initial_capital=100000):
    """
    Run backtest for a single strategy
    
    Args:
        data: Market data
        strategy: Trading strategy implementing StrategyProtocol
        initial_capital: Starting capital
        
    Returns:
        dict: Backtest results
    """
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=0.001,  # 0.1% commission
        slippage=0.001,    # 0.1% slippage
        max_position_size=0.95
    )
    
    engine.run_backtest(data, strategy)
    
    return {
        'engine': engine,
        'performance': engine.get_performance_summary(),
        'portfolio_values': engine.portfolio_values.copy(),
        'trades': engine.trades.copy()
    }


def analyze_results(results, initial_capital, benchmark_return):
    """
    Analyze and display backtest results
    
    Args:
        results: Dictionary of strategy results
        initial_capital: Starting capital
        benchmark_return: Buy and hold return
    """
    print("\n" + "="*60)
    print("📊 BACKTEST RESULTS SUMMARY")
    print("="*60)
    
    # Create performance table
    performance_data = []
    for strategy_name, result in results.items():
        if result['portfolio_values']:
            final_value = result['portfolio_values'][-1]
            total_return = (final_value / initial_capital - 1) * 100
            perf = result['performance']
            
            performance_data.append({
                'Strategy': strategy_name,
                'Final Value': f"${final_value:,.2f}",
                'Total Return': f"{total_return:.2f}%",
                'Sharpe Ratio': f"{perf.get('sharpe_ratio', 0):.3f}",
                'Max Drawdown': f"{perf.get('max_drawdown', 0)*100:.2f}%",
                'Trades': len(result['trades'])
            })
    
    # Display results table
    df = pd.DataFrame(performance_data)
    print(df.to_string(index=False))
    
    # Compare to benchmark
    print(f"\n📈 BENCHMARK COMPARISON")
    print(f"Buy & Hold Return: {benchmark_return:.2f}%")
    
    # Find best strategy
    best_strategy = max(results.keys(), 
                       key=lambda x: results[x]['portfolio_values'][-1] if results[x]['portfolio_values'] else 0)
    best_return = (results[best_strategy]['portfolio_values'][-1] / initial_capital - 1) * 100
    
    print(f"\n🏆 BEST STRATEGY: {best_strategy}")
    print(f"Return: {best_return:.2f}% vs Buy & Hold: {benchmark_return:.2f}%")
    print(f"Outperformance: {best_return - benchmark_return:+.2f}%")


def create_visualization(data, results, symbol):
    """
    Create performance visualization
    
    Args:
        data: Market data
        results: Backtest results
        symbol: Stock symbol
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Portfolio value comparison
    for strategy_name, result in results.items():
        if result['portfolio_values']:
            dates = data.index[:len(result['portfolio_values'])]
            ax1.plot(dates, result['portfolio_values'], label=strategy_name, linewidth=2)
    
    # Add buy and hold benchmark
    initial_capital = 100000
    buy_hold_values = initial_capital * (data['close'] / data['close'].iloc[0])
    ax1.plot(data.index, buy_hold_values, label='Buy & Hold', linestyle='--', color='black', alpha=0.7)
    
    ax1.set_title(f'{symbol} - Portfolio Value Comparison')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Stock price with moving averages
    ax2.plot(data.index, data['close'], label='Close Price', color='blue', alpha=0.7)
    ax2.plot(data.index, data['close'].rolling(20).mean(), label='20-day MA', color='orange')
    ax2.plot(data.index, data['close'].rolling(50).mean(), label='50-day MA', color='red')
    ax2.set_title(f'{symbol} - Stock Price & Moving Averages')
    ax2.set_ylabel('Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Returns comparison
    strategy_returns = []
    strategy_names = []
    for strategy_name, result in results.items():
        if result['portfolio_values']:
            total_return = (result['portfolio_values'][-1] / initial_capital - 1) * 100
            strategy_returns.append(total_return)
            strategy_names.append(strategy_name)
    
    # Add benchmark
    benchmark_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
    strategy_returns.append(benchmark_return)
    strategy_names.append('Buy & Hold')
    
    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'lightgray']
    bars = ax3.bar(strategy_names, strategy_returns, color=colors[:len(strategy_names)])
    ax3.set_title('Total Returns Comparison')
    ax3.set_ylabel('Return (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, strategy_returns):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(strategy_returns) * 0.01,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # 4. Drawdown analysis
    for strategy_name, result in results.items():
        if result['portfolio_values']:
            values = pd.Series(result['portfolio_values'])
            running_max = values.expanding().max()
            drawdown = (values - running_max) / running_max * 100
            dates = data.index[:len(drawdown)]
            ax4.fill_between(dates, drawdown, 0, alpha=0.3, label=strategy_name)
    
    ax4.set_title('Drawdown Analysis')
    ax4.set_ylabel('Drawdown (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main execution function"""
    print("🚀 Quantitative Trading Framework - Yahoo Finance Example")
    print("="*60)
    
    # Configuration
    symbol = "AAPL"
    initial_capital = 100000
    
    try:
        # 1. Fetch data
        data = fetch_stock_data(symbol, days=730)  # 2 years
        
        # 2. Setup strategies
        strategies = setup_strategies()
        
        # 3. Run backtests
        print(f"\n🔄 Running backtests with ${initial_capital:,} initial capital...")
        results = {}
        
        for strategy_name, strategy in strategies.items():
            print(f"   Testing {strategy_name}...")
            results[strategy_name] = run_backtest(data, strategy, initial_capital)
            
            final_value = results[strategy_name]['portfolio_values'][-1] if results[strategy_name]['portfolio_values'] else initial_capital
            total_return = (final_value / initial_capital - 1) * 100
            print(f"   ✅ Completed! Return: {total_return:+.2f}%")
        
        # 4. Calculate benchmark
        benchmark_return = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
        
        # 5. Analyze results
        analyze_results(results, initial_capital, benchmark_return)
        
        # 6. Create visualization
        print(f"\n📊 Creating performance charts...")
        create_visualization(data, results, symbol)
        
        print(f"\n🎉 Analysis completed successfully!")
        print(f"\n💡 Next Steps:")
        print(f"   • Try different stocks by changing 'symbol' variable")
        print(f"   • Adjust strategy parameters for optimization")
        print(f"   • Add risk management features")
        print(f"   • Test on longer time periods")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"Make sure you have installed: pip install yfinance pandas numpy matplotlib")


if __name__ == "__main__":
    main()