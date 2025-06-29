#!/usr/bin/env python3
"""
Backtest Visualization Demo

This script demonstrates the comprehensive backtest visualization features
that help verify strategy correctness and analyze performance.

Features shown:
- Stock price chart with moving averages
- Buy/sell signal markers
- Portfolio performance vs benchmark
- Drawdown analysis
- Trade win/loss breakdown
- Performance summary statistics

Usage:
    python backtest_visualization_demo.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our quantitative trading framework
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine
from quant_trading.data.data_fetcher import create_sample_data


def run_demo():
    """Run the backtest visualization demo"""
    print("📊 Backtest Visualization Demo")
    print("=" * 50)
    
    # 1. Create sample data with interesting patterns
    print("\n📈 Creating sample market data...")
    data = create_sample_data(
        days=100, 
        trend=0.05,  # Slight upward trend
        volatility=0.02,
        seed=42
    )
    print(f"✅ Generated {len(data)} days of market data")
    
    # 2. Setup strategy
    print("\n🎯 Setting up trading strategy...")
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=20,
        quantity=100
    )
    print(f"✅ Strategy: {strategy.get_name()}")
    print(f"   Parameters: {strategy.get_parameters()}")
    
    # 3. Run backtest with beginner-friendly settings
    print("\n🚀 Running backtest...")
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.001,
        allow_short_selling=False,  # Beginner-friendly
        max_position_size=0.3
    )
    
    # Run the backtest
    engine.run_backtest(data, strategy)
    
    # 4. Display results summary
    print(f"\n📊 Backtest Results:")
    performance = engine.get_performance_summary()
    if performance:
        print(f"   Total Return: {performance['total_return']*100:.2f}%")
        print(f"   Max Drawdown: {performance['max_drawdown']*100:.2f}%")
        print(f"   Total Trades: {performance['total_trades']}")
        print(f"   Win Rate: {performance['win_rate']*100:.1f}%")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.3f}")
    
    # 5. Create comprehensive visualization
    print(f"\n📈 Creating comprehensive backtest visualization...")
    try:
        fig = engine.plot_backtest_results(
            strategy_name="Moving Average Crossover",
            symbol="DEMO",
            show_plot=True
        )
        
        if fig:
            print("✅ Visualization created successfully!")
            print("\n📋 What the visualization shows:")
            print("   🔹 Top chart: Stock price with moving averages and trade signals")
            print("   🔹 Portfolio performance vs buy-and-hold benchmark")
            print("   🔹 Drawdown analysis showing risk periods")
            print("   🔹 Trade win/loss breakdown")
            print("   🔹 Performance summary statistics")
            print("\n💡 Look for:")
            print("   • Green triangles (▲) = Buy signals")
            print("   • Red triangles (▼) = Sell signals") 
            print("   • Orange/red lines = Moving averages")
            print("   • Strategy performance vs benchmark")
        else:
            print("⚠️  Visualization not available (matplotlib required)")
            
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        print("💡 Install matplotlib: pip install matplotlib")
    
    # 6. Show detailed trade analysis
    print(f"\n📋 Detailed Trade Analysis:")
    trade_analysis = engine.analyze_trades()
    
    if "trades" in trade_analysis and trade_analysis["trades"]:
        print("\n📊 Trade Summary:")
        summary = trade_analysis["summary"]
        for key, value in summary.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n📑 Individual Trades:")
        for trade in trade_analysis["trades"][:5]:  # Show first 5 trades
            print(f"   Trade {trade['trade_number']}: {trade['side']} {trade['quantity']} @ {trade['entry_price']} → {trade['exit_price']}, P&L: {trade['pnl']}")
        
        if len(trade_analysis["trades"]) > 5:
            print(f"   ... and {len(trade_analysis['trades']) - 5} more trades")
    else:
        print("   No trades executed (strategy conditions not met)")
    
    # 7. Verification tips
    print(f"\n🔍 How to Verify Your Strategy:")
    print("   1. Check buy signals occur when short MA crosses above long MA")
    print("   2. Check sell signals occur when short MA crosses below long MA") 
    print("   3. Verify no short positions when allow_short_selling=False")
    print("   4. Compare strategy performance vs buy-and-hold")
    print("   5. Analyze trade timing and market conditions")
    
    print(f"\n🎉 Demo completed! The visualization helps you:")
    print("   • Verify strategy logic is working correctly")
    print("   • Identify good vs bad trade timing")
    print("   • Understand strategy performance characteristics")
    print("   • Compare against benchmarks")
    print("   • Analyze risk (drawdowns) vs returns")


def advanced_demo():
    """Run advanced demo with real-world-like data"""
    print("\n" + "=" * 50)
    print("🚀 Advanced Demo with Multiple Strategies")
    print("=" * 50)
    
    # Create more complex data
    data = create_sample_data(days=200, trend=0.02, volatility=0.025, seed=123)
    
    strategies = {
        'Fast MA (5/15)': MovingAverageStrategy(short_window=5, long_window=15, quantity=100),
        'Slow MA (20/50)': MovingAverageStrategy(short_window=20, long_window=50, quantity=100)
    }
    
    for name, strategy in strategies.items():
        print(f"\n📊 Testing {name}...")
        
        engine = BacktestEngine(
            initial_capital=100000,
            allow_short_selling=False,
            max_position_size=0.4
        )
        
        engine.run_backtest(data, strategy)
        performance = engine.get_performance_summary()
        
        print(f"   Return: {performance['total_return']*100:.1f}%, "
              f"Drawdown: {performance['max_drawdown']*100:.1f}%, "
              f"Trades: {performance['total_trades']}")
        
        # Create visualization for each strategy
        try:
            fig = engine.plot_backtest_results(
                strategy_name=name,
                symbol="ADVANCED_DEMO",
                show_plot=False  # Don't show immediately
            )
            if fig:
                plt.show()  # Show one at a time
        except:
            pass


if __name__ == "__main__":
    # Run basic demo
    run_demo()
    
    # Ask user if they want to see advanced demo
    try:
        user_input = input(f"\n❓ Run advanced demo with multiple strategies? (y/N): ").lower()
        if user_input in ['y', 'yes']:
            advanced_demo()
    except KeyboardInterrupt:
        print(f"\n👋 Demo finished!")
    
    print(f"\n📚 Next Steps:")
    print("   • Try different strategy parameters")
    print("   • Test with real market data using yfinance")
    print("   • Enable short selling for advanced strategies")
    print("   • Add risk management rules")
    print("   • Optimize parameters using grid search")