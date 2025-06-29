#!/usr/bin/env python3
"""
Simple Visualization Example

This example demonstrates how to use the new comprehensive backtest visualization
features to verify your strategy is working correctly.

Usage:
    python simple_visualization_example.py
"""

from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine
from quant_trading.data.data_fetcher import create_sample_data


def main():
    print("📊 Simple Backtest Visualization Example")
    print("=" * 50)
    
    # 1. Create trending data that will generate signals
    print("📈 Creating market data with strong trend...")
    data = create_sample_data(
        days=60,
        trend=0.15,  # Strong upward trend to generate signals
        volatility=0.03,
        seed=42
    )
    print(f"✅ Generated {len(data)} days of data")
    
    # 2. Setup responsive strategy
    print("\n🎯 Setting up responsive moving average strategy...")
    strategy = MovingAverageStrategy(
        short_window=5,   # Very responsive
        long_window=10,   # Quick crossovers
        quantity=100
    )
    print(f"✅ Strategy: {strategy.get_name()}")
    
    # 3. Run backtest
    print("\n🚀 Running backtest...")
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.001,
        allow_short_selling=False,  # Beginner mode
        max_position_size=0.5
    )
    
    engine.run_backtest(data, strategy)
    
    # 4. Show results
    performance = engine.get_performance_summary()
    print(f"\n📊 Results:")
    print(f"   Total Return: {performance['total_return']*100:.2f}%")
    print(f"   Total Trades: {performance['total_trades']}")
    print(f"   Win Rate: {performance.get('win_rate', 0)*100:.1f}%")
    print(f"   Max Drawdown: {performance['max_drawdown']*100:.1f}%")
    
    # 5. Create visualization
    print(f"\n📈 Creating comprehensive visualization...")
    print("   This will show:")
    print("   🔹 Stock price with moving averages")
    print("   🔹 Buy/sell signals marked with arrows")
    print("   🔹 Portfolio performance vs buy-and-hold")
    print("   🔹 Drawdown analysis")
    print("   🔹 Trade win/loss breakdown")
    print("   🔹 Performance summary")
    
    try:
        fig = engine.plot_backtest_results(
            strategy_name="Moving Average (5/10)",
            symbol="EXAMPLE",
            show_plot=True  # This will display the chart
        )
        
        if fig:
            print("✅ Visualization created! Check the chart window.")
        else:
            print("⚠️  Matplotlib not available. Install with: pip install matplotlib")
            
    except Exception as e:
        print(f"ℹ️  Visualization requires matplotlib: {e}")
        print("💡 Install with: pip install matplotlib")
    
    # 6. Show trade details
    print(f"\n📋 Trade Analysis:")
    trade_analysis = engine.analyze_trades()
    
    if trade_analysis.get("trades"):
        summary = trade_analysis["summary"]
        print(f"   📊 Summary:")
        print(f"      Total Trades: {summary['total_trades']}")
        print(f"      Win Rate: {summary['win_rate']}")
        print(f"      Total P&L: {summary['total_pnl']}")
        print(f"      Avg P&L/Trade: {summary['avg_pnl_per_trade']}")
        
        print(f"\n   📑 Recent Trades:")
        for trade in trade_analysis["trades"][:3]:  # Show first 3
            print(f"      #{trade['trade_number']}: {trade['side']} {trade['quantity']} shares")
            print(f"         Entry: {trade['entry_price']} → Exit: {trade['exit_price']}")
            print(f"         P&L: {trade['pnl']} ({trade['return_pct']})")
    else:
        print("   No trades executed (strategy conditions not met)")
    
    # 7. Verification tips
    print(f"\n🔍 How to Use This Visualization:")
    print("   1. Check that buy signals (green ▲) occur when short MA crosses above long MA")
    print("   2. Check that sell signals (red ▼) occur when short MA crosses below long MA")
    print("   3. Verify strategy performance vs benchmark (gray dashed line)")
    print("   4. Look for reasonable drawdown periods (red areas)")
    print("   5. Ensure trade timing makes sense with market movements")
    
    print(f"\n💡 This visualization helps you:")
    print("   • Verify your strategy logic is correct")
    print("   • Spot potential issues with signal timing")
    print("   • Compare performance vs simple buy-and-hold")
    print("   • Understand risk vs return characteristics")
    
    print(f"\n🎉 Example completed!")


if __name__ == "__main__":
    main()