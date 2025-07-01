#!/usr/bin/env python3
"""
Test Apple stock with fixed quantity strategy - investigate framework vs VectorBT differences
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

def fetch_apple_data(days=365):
    """Fetch real Apple stock data"""
    print("📊 Fetching Apple stock data...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        ticker = yf.Ticker("AAPL")
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError("No data fetched")
        
        # Handle timezone issues
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Convert column names to lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Remove any NaN values
        data = data.dropna()
        
        print(f"✅ Fetched {len(data)} days of Apple data")
        print(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"   Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
        
        return data
        
    except Exception as e:
        print(f"❌ Failed to fetch Apple data: {e}")
        return None

def test_our_framework(data, strategy, initial_capital=100000):
    """Test with our framework"""
    print(f"\n🔧 Testing Our Framework:")
    
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=0.001,
        slippage=0.0,
        max_position_size=1.0,  # Allow full allocation
        allow_short_selling=False
    )
    
    # Run backtest
    engine.run_backtest(data, strategy)
    
    # Get results
    final_value = engine.portfolio_values[-1]
    total_return = (final_value / initial_capital - 1) * 100
    trades = len(engine.trades)
    
    print(f"   Final Value: ${final_value:,.2f}")
    print(f"   Total Return: {total_return:+.2f}%")
    print(f"   Total Trades: {trades}")
    
    # Analyze signals
    all_signals = []
    for idx, _ in data.iterrows():
        partial_data = data.loc[:idx]
        signals = strategy.get_signals(partial_data, available_capital=initial_capital)
        all_signals.extend(signals)
    
    buy_signals = [s for s in all_signals if s.action.value == 'buy']
    sell_signals = [s for s in all_signals if s.action.value == 'sell']
    
    print(f"   Signals Generated: {len(buy_signals)} buys, {len(sell_signals)} sells")
    print(f"   Trades Executed: {trades} (execution rate: {trades/len(all_signals)*100:.1f}%)" if all_signals else "   No signals generated")
    
    # Show first few trades for debugging
    if engine.trades:
        print(f"   First few trades:")
        for i, trade in enumerate(engine.trades[:3]):
            print(f"     Trade {i+1}: {trade.quantity} shares at ${trade.entry_price:.2f}")
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'trades': trades,
        'signals_generated': len(all_signals),
        'engine': engine
    }

def test_vectorbt(data, strategy, initial_capital=100000):
    """Test with VectorBT"""
    print(f"\n🔧 Testing VectorBT:")
    
    try:
        import vectorbt as vbt
        
        # Generate signals using our strategy
        entries, exits = strategy.generate_vectorbt_signals(data, initial_capital)
        vbt_config = strategy.get_vectorbt_position_sizing(data, initial_capital)
        
        print(f"   VectorBT Config: {vbt_config}")
        print(f"   Signals: {entries.sum()} entries, {exits.sum()} exits")
        
        if entries.sum() > 0 or exits.sum() > 0:
            # Create VectorBT portfolio with fixed quantity
            vbt_portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries,
                exits=exits,
                size=strategy.quantity,  # Fixed 100 shares
                init_cash=initial_capital,
                fees=0.001,
                freq='D'
            )
            
            # Get VectorBT stats
            vbt_stats = vbt_portfolio.stats()
            vbt_return = vbt_stats['Total Return [%]']
            vbt_trades = vbt_stats['Total Trades']
            vbt_final_value = vbt_portfolio.value().iloc[-1]
            
            print(f"   Final Value: ${vbt_final_value:,.2f}")
            print(f"   Total Return: {vbt_return:+.2f}%")
            print(f"   Total Trades: {vbt_trades}")
            
            # Show VectorBT trade details
            trades = vbt_portfolio.trades
            if len(trades.records) > 0:
                print(f"   VectorBT Trade Details:")
                for i in range(min(3, len(trades.records))):
                    trade = trades.records[i]
                    print(f"     Trade {i+1}: {trade['size']} shares")
            
            return {
                'final_value': vbt_final_value,
                'total_return': vbt_return,
                'trades': vbt_trades,
                'portfolio': vbt_portfolio,
                'stats': vbt_stats
            }
        else:
            print("   No signals to process")
            return None
            
    except ImportError:
        print("   ❌ VectorBT not available")
        return None
    except Exception as e:
        print(f"   ❌ VectorBT error: {e}")
        return None

def analyze_differences(our_results, vbt_results):
    """Analyze differences between frameworks"""
    print(f"\n🔍 DIFFERENCE ANALYSIS:")
    
    if vbt_results is None:
        print("   Cannot compare - VectorBT results unavailable")
        return
    
    return_diff = abs(our_results['total_return'] - vbt_results['total_return'])
    trade_diff = abs(our_results['trades'] - vbt_results['trades'])
    value_diff = abs(our_results['final_value'] - vbt_results['final_value'])
    
    print(f"   Return Difference: {return_diff:.2f}% points")
    print(f"   Trade Count Difference: {trade_diff} trades")
    print(f"   Final Value Difference: ${value_diff:,.2f}")
    
    # Categorize the difference
    if return_diff > 5.0 or trade_diff > 2:
        print(f"   🚨 SIGNIFICANT DIFFERENCE DETECTED!")
        
        print(f"\n🔧 POTENTIAL CAUSES:")
        print(f"   1. Position sizing calculation differences")
        print(f"   2. Signal timing differences")
        print(f"   3. Order execution logic differences")
        print(f"   4. Capital allocation differences")
        print(f"   5. Commission/slippage handling differences")
        
        return True  # Significant difference
    elif return_diff > 2.0 or trade_diff > 1:
        print(f"   ⚠️ MODERATE DIFFERENCE")
        return False  # Acceptable difference
    else:
        print(f"   ✅ MINOR DIFFERENCE - Within acceptable range")
        return False

def debug_signal_timing(data, strategy, initial_capital=100000):
    """Debug signal timing differences"""
    print(f"\n🔍 DEBUGGING SIGNAL TIMING:")
    
    # Our framework signal generation
    our_signals = []
    our_signal_dates = []
    
    for idx, _ in data.iterrows():
        partial_data = data.loc[:idx]
        signals = strategy.get_signals(partial_data, available_capital=initial_capital)
        
        for signal in signals:
            our_signals.append(signal.action.value)
            our_signal_dates.append(idx)
    
    # VectorBT signal generation
    entries, exits = strategy.generate_vectorbt_signals(data, initial_capital)
    vbt_entry_dates = entries[entries].index.tolist()
    vbt_exit_dates = exits[exits].index.tolist()
    
    print(f"   Our Framework:")
    print(f"     Buy signals: {[d.strftime('%Y-%m-%d') for d in our_signal_dates if our_signals[our_signal_dates.index(d)] == 'buy'][:5]}")
    print(f"     Sell signals: {[d.strftime('%Y-%m-%d') for d in our_signal_dates if our_signals[our_signal_dates.index(d)] == 'sell'][:5]}")
    
    print(f"   VectorBT:")
    print(f"     Entry signals: {[d.strftime('%Y-%m-%d') for d in vbt_entry_dates[:5]]}")
    print(f"     Exit signals: {[d.strftime('%Y-%m-%d') for d in vbt_exit_dates[:5]]}")
    
    # Check for timing mismatches
    our_buy_dates = [d for d in our_signal_dates if our_signals[our_signal_dates.index(d)] == 'buy']
    our_sell_dates = [d for d in our_signal_dates if our_signals[our_signal_dates.index(d)] == 'sell']
    
    buy_timing_match = our_buy_dates == vbt_entry_dates
    sell_timing_match = our_sell_dates == vbt_exit_dates
    
    print(f"   Signal Timing Analysis:")
    print(f"     Buy signals match: {'✅' if buy_timing_match else '❌'}")
    print(f"     Sell signals match: {'✅' if sell_timing_match else '❌'}")
    
    if not buy_timing_match or not sell_timing_match:
        print(f"   🚨 SIGNAL TIMING MISMATCH DETECTED!")
        return True
    
    return False

def main():
    """Main test function"""
    print("🍎 APPLE STOCK FIXED QUANTITY STRATEGY TEST")
    print("=" * 60)
    
    # Fetch Apple data
    data = fetch_apple_data(365)
    if data is None:
        print("❌ Cannot proceed without data")
        return False
    
    # Create fixed quantity strategy
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        quantity=100  # Fixed 100 shares
    )
    
    print(f"\n📊 Strategy Configuration:")
    print(f"   Strategy: {strategy.get_name()}")
    print(f"   Parameters: {strategy.get_parameters()}")
    print(f"   Position Sizing: Fixed {strategy.quantity} shares")
    
    # Test both frameworks
    our_results = test_our_framework(data, strategy)
    vbt_results = test_vectorbt(data, strategy)
    
    # Analyze differences
    significant_diff = analyze_differences(our_results, vbt_results)
    
    # Debug signal timing if there are differences
    if significant_diff:
        timing_issues = debug_signal_timing(data, strategy)
        
        if timing_issues:
            print(f"\n🔧 NEXT STEPS:")
            print(f"   1. Investigate signal generation logic")
            print(f"   2. Check moving average calculations")
            print(f"   3. Verify get_signals() vs generate_vectorbt_signals() consistency")
    
    # Summary
    print(f"\n📋 TEST SUMMARY:")
    print(f"   Data: {len(data)} days of Apple stock")
    print(f"   Strategy: Moving Average (10/30) with 100 fixed shares")
    print(f"   Our Framework: {our_results['total_return']:+.2f}% ({our_results['trades']} trades)")
    if vbt_results:
        print(f"   VectorBT: {vbt_results['total_return']:+.2f}% ({vbt_results['trades']} trades)")
        print(f"   Difference: {abs(our_results['total_return'] - vbt_results['total_return']):.2f}% points")
    
    return significant_diff, our_results, vbt_results

if __name__ == "__main__":
    has_significant_diff, our_results, vbt_results = main()
    
    if has_significant_diff:
        print(f"\n🚨 SIGNIFICANT DIFFERENCES DETECTED!")
        print(f"   This test case should be added to the unit test suite")
    else:
        print(f"\n✅ Framework alignment is acceptable")