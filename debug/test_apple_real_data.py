#!/usr/bin/env python3
"""
Test Apple stock with real data - identify framework vs VectorBT differences
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

def fetch_apple_data():
    """Fetch real Apple stock data"""
    print("📊 Fetching real Apple stock data...")
    
    try:
        # Fetch 1 year of Apple data
        apple = yf.Ticker("AAPL")
        data = apple.history(period="1y")
        
        if data.empty:
            raise ValueError("No data returned")
        
        # Clean the data
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        data = data.dropna()
        
        print(f"✅ Successfully fetched {len(data)} days of Apple data")
        print(f"📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"💰 Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
        print(f"📈 Total return: {(data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100:.1f}%")
        
        return data
        
    except Exception as e:
        print(f"❌ Failed to fetch Apple data: {e}")
        return None

def test_our_framework(data, strategy, initial_capital=100000):
    """Test our framework with Apple data"""
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
    
    # Analyze signal generation vs execution
    all_signals = []
    for idx, _ in data.iterrows():
        partial_data = data.loc[:idx]
        signals = strategy.get_signals(partial_data, available_capital=initial_capital)
        all_signals.extend(signals)
    
    buy_signals = [s for s in all_signals if s.action.value == 'buy']
    sell_signals = [s for s in all_signals if s.action.value == 'sell']
    
    print(f"   Signals Generated: {len(buy_signals)} buys, {len(sell_signals)} sells")
    
    if all_signals:
        execution_rate = trades / len(all_signals) * 100
        print(f"   Execution Rate: {execution_rate:.1f}%")
        
        if execution_rate < 50:
            print(f"   ⚠️ Low execution rate - investigating...")
            print(f"   Possible causes: position limits, insufficient capital, order rejections")
    
    # Show sample trades
    if engine.trades:
        print(f"   Sample trades:")
        for i, trade in enumerate(engine.trades[:3]):
            print(f"     Trade {i+1}: {trade.quantity} shares at ${trade.entry_price:.2f}")
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'trades': trades,
        'signals_generated': len(all_signals),
        'buy_signals': len(buy_signals),
        'sell_signals': len(sell_signals),
        'engine': engine
    }

def test_vectorbt(data, strategy, initial_capital=100000):
    """Test VectorBT with Apple data"""
    print(f"\n🔧 Testing VectorBT:")
    
    try:
        import vectorbt as vbt
        
        # Generate signals using our strategy
        entries, exits = strategy.generate_vectorbt_signals(data, initial_capital)
        
        print(f"   VectorBT Signals: {entries.sum()} entries, {exits.sum()} exits")
        
        if entries.sum() > 0 or exits.sum() > 0:
            # Get position sizing configuration
            vbt_config = strategy.get_vectorbt_position_sizing(data, initial_capital)
            print(f"   VectorBT Config: {vbt_config}")
            
            # Create VectorBT portfolio with correct position sizing
            if hasattr(strategy, 'quantity') and strategy.quantity:
                # Fixed quantity mode
                vbt_portfolio = vbt.Portfolio.from_signals(
                    close=data['close'],
                    entries=entries,
                    exits=exits,
                    size=strategy.quantity,
                    init_cash=initial_capital,
                    fees=0.001,
                    freq='D'
                )
                print(f"   Position Sizing: Fixed {strategy.quantity} shares")
            else:
                # Auto sizing mode
                vbt_portfolio = vbt.Portfolio.from_signals(
                    close=data['close'],
                    entries=entries,
                    exits=exits,
                    init_cash=initial_capital,
                    fees=0.001,
                    freq='D'
                )
                print(f"   Position Sizing: Auto (VectorBT default)")
            
            # Get VectorBT results
            vbt_stats = vbt_portfolio.stats()
            vbt_final_value = vbt_portfolio.value().iloc[-1]
            vbt_return = vbt_stats['Total Return [%]']
            vbt_trades = vbt_stats['Total Trades']
            
            print(f"   Final Value: ${vbt_final_value:,.2f}")
            print(f"   Total Return: {vbt_return:+.2f}%")
            print(f"   Total Trades: {vbt_trades}")
            
            # Show additional VectorBT metrics
            print(f"   Win Rate: {vbt_stats['Win Rate [%]']:.1f}%")
            if vbt_stats['Total Trades'] > 0:
                print(f"   Avg Trade: {vbt_stats['Avg Trade [%]']:.2f}%")
            
            return {
                'final_value': vbt_final_value,
                'total_return': vbt_return,
                'trades': vbt_trades,
                'portfolio': vbt_portfolio,
                'stats': vbt_stats,
                'entries': entries,
                'exits': exits
            }
        else:
            print(f"   ℹ️ No signals generated")
            return None
            
    except ImportError:
        print(f"   ⚠️ VectorBT not available")
        return None
    except Exception as e:
        print(f"   ❌ VectorBT error: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_differences(our_results, vbt_results):
    """Analyze differences between frameworks"""
    print(f"\n🔍 FRAMEWORK DIFFERENCE ANALYSIS:")
    print("=" * 50)
    
    if vbt_results is None:
        print(f"   Cannot compare - VectorBT results unavailable")
        return False
    
    # Calculate differences
    return_diff = abs(our_results['total_return'] - vbt_results['total_return'])
    trade_diff = abs(our_results['trades'] - vbt_results['trades'])
    value_diff = abs(our_results['final_value'] - vbt_results['final_value'])
    
    print(f"📊 Quantitative Differences:")
    print(f"   Return difference: {return_diff:.2f} percentage points")
    print(f"   Trade count difference: {trade_diff}")
    print(f"   Final value difference: ${value_diff:,.2f}")
    
    # Signal generation comparison
    signal_diff = abs(our_results['signals_generated'] - (vbt_results['entries'].sum() + vbt_results['exits'].sum()))
    print(f"   Signal generation difference: {signal_diff}")
    
    # Execution rate comparison
    our_execution_rate = our_results['trades'] / our_results['signals_generated'] * 100 if our_results['signals_generated'] > 0 else 0
    vbt_execution_rate = vbt_results['trades'] / (vbt_results['entries'].sum() + vbt_results['exits'].sum()) * 100 if (vbt_results['entries'].sum() + vbt_results['exits'].sum()) > 0 else 0
    
    print(f"   Our execution rate: {our_execution_rate:.1f}%")
    print(f"   VectorBT execution rate: {vbt_execution_rate:.1f}%")
    
    # Categorize differences
    significant_issues = []
    
    if return_diff > 5.0:
        significant_issues.append(f"Large return difference ({return_diff:.1f}pp)")
    
    if trade_diff > 3:
        significant_issues.append(f"Large trade count difference ({trade_diff})")
    
    if signal_diff > 2:
        significant_issues.append(f"Signal generation mismatch ({signal_diff})")
    
    if abs(our_execution_rate - vbt_execution_rate) > 20:
        significant_issues.append(f"Execution rate difference ({abs(our_execution_rate - vbt_execution_rate):.1f}pp)")
    
    if significant_issues:
        print(f"\n🚨 SIGNIFICANT DIFFERENCES DETECTED:")
        for issue in significant_issues:
            print(f"   • {issue}")
        
        print(f"\n🔧 POTENTIAL CAUSES:")
        print(f"   1. Position sizing calculation differences")
        print(f"   2. Capital allocation logic differences")
        print(f"   3. Order execution constraints in our framework")
        print(f"   4. Signal timing differences")
        print(f"   5. Commission/slippage handling differences")
        
        return True
    else:
        print(f"\n✅ DIFFERENCES WITHIN ACCEPTABLE RANGE")
        print(f"   Both frameworks show consistent behavior")
        return False

def debug_signal_timing(data, strategy, our_results, vbt_results):
    """Debug signal timing differences"""
    print(f"\n🔍 SIGNAL TIMING DEBUG:")
    
    if vbt_results is None:
        return
    
    # Get our framework signals by date
    our_buy_dates = []
    our_sell_dates = []
    
    for idx, _ in data.iterrows():
        partial_data = data.loc[:idx]
        signals = strategy.get_signals(partial_data, available_capital=100000)
        
        for signal in signals:
            if signal.action.value == 'buy':
                our_buy_dates.append(idx)
            elif signal.action.value == 'sell':
                our_sell_dates.append(idx)
    
    # Get VectorBT signals
    vbt_buy_dates = vbt_results['entries'][vbt_results['entries']].index.tolist()
    vbt_sell_dates = vbt_results['exits'][vbt_results['exits']].index.tolist()
    
    print(f"   Our buy signals: {[d.strftime('%m-%d') for d in our_buy_dates[:5]]}")
    print(f"   VBT buy signals: {[d.strftime('%m-%d') for d in vbt_buy_dates[:5]]}")
    
    # Check if signal timing matches
    buy_timing_match = our_buy_dates == vbt_buy_dates
    sell_timing_match = our_sell_dates == vbt_sell_dates
    
    print(f"   Buy signal timing match: {'✅' if buy_timing_match else '❌'}")
    print(f"   Sell signal timing match: {'✅' if sell_timing_match else '❌'}")
    
    if not buy_timing_match or not sell_timing_match:
        print(f"   ⚠️ Signal timing mismatch detected!")
        return True
    
    return False

def main():
    """Main test function"""
    print("🍎 APPLE STOCK REAL DATA FRAMEWORK COMPARISON")
    print("=" * 60)
    
    # Fetch real Apple data
    data = fetch_apple_data()
    if data is None:
        print("❌ Cannot proceed without Apple data")
        return False
    
    # Create strategy - fixed quantity that you mentioned has issues
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        quantity=100  # Fixed 100 shares
    )
    
    print(f"\n📊 Strategy Configuration:")
    print(f"   Strategy: {strategy.get_name()}")
    print(f"   Parameters: {strategy.get_parameters()}")
    if hasattr(strategy, 'quantity') and strategy.quantity:
        print(f"   Position Sizing: Fixed {strategy.quantity} shares")
    
    # Test both frameworks
    our_results = test_our_framework(data, strategy)
    vbt_results = test_vectorbt(data, strategy)
    
    # Analyze differences
    has_significant_diff = analyze_differences(our_results, vbt_results)
    
    # Debug signal timing if needed
    if has_significant_diff:
        timing_issues = debug_signal_timing(data, strategy, our_results, vbt_results)
    
    # Summary
    print(f"\n📋 TEST SUMMARY:")
    print(f"   Data: {len(data)} days of real Apple stock")
    print(f"   Strategy: MA(10,30) with fixed 100 shares")
    print(f"   Our Framework: {our_results['total_return']:+.2f}% ({our_results['trades']} trades)")
    if vbt_results:
        print(f"   VectorBT: {vbt_results['total_return']:+.2f}% ({vbt_results['trades']} trades)")
        print(f"   Difference: {abs(our_results['total_return'] - vbt_results['total_return']):.2f}pp")
    
    print(f"   Significant differences: {'YES' if has_significant_diff else 'NO'}")
    
    if has_significant_diff:
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Create unit test to reproduce this difference")
        print(f"   2. Investigate position sizing calculations")
        print(f"   3. Check order execution logic")
        print(f"   4. Verify capital allocation methods")
    
    return has_significant_diff, our_results, vbt_results, data

if __name__ == "__main__":
    has_diff, our_results, vbt_results, data = main()
    
    if has_diff:
        print(f"\n🔧 CREATING UNIT TEST FOR THIS SCENARIO...")
        print(f"   This test case should be added to catch framework differences")
    else:
        print(f"\n✅ Frameworks are well aligned - no immediate action needed")