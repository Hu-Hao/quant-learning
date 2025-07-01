#!/usr/bin/env python3
"""
Debug the large framework differences between our engine and VectorBT
"""

import yfinance as yf
import numpy as np
import pandas as pd
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

def debug_framework_differences():
    """Debug what's causing the large differences"""
    print("🔍 DEBUGGING FRAMEWORK DIFFERENCES")
    print("=" * 60)
    
    # Get Apple data
    print("📊 Fetching Apple data...")
    apple = yf.Ticker("AAPL")
    data = apple.history(period="6mo")
    
    if data.empty:
        print("❌ Could not fetch Apple data")
        return
    
    # Clean data
    if hasattr(data.index, 'tz') and data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data.columns = [col.lower() for col in data.columns]
    data = data.dropna()
    
    print(f"✅ Got {len(data)} days of Apple data")
    print(f"📈 Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    print(f"📈 Average price: ${data['close'].mean():.2f}")
    
    # Create strategy
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        quantity=100
    )
    
    print(f"\n📊 Strategy: MA(10,30) with fixed 100 shares")
    
    # 1. EXAMINE SIGNALS FIRST
    print(f"\n🔍 STEP 1: Signal Analysis")
    print("-" * 40)
    
    # Generate signals manually to examine them
    all_signals = []
    for idx, row in data.iterrows():
        partial_data = data.loc[:idx]
        signals = strategy.get_signals(partial_data, available_capital=100000)
        if signals:
            for signal in signals:
                all_signals.append({
                    'date': idx,
                    'price': row['close'],
                    'action': signal.action,
                    'quantity': signal.quantity,
                    'signal_price': signal.price
                })
    
    print(f"   Total signals generated: {len(all_signals)}")
    for i, signal in enumerate(all_signals):
        print(f"   Signal {i+1}: {signal['date'].strftime('%Y-%m-%d')} - "
              f"{signal['action']} {signal['quantity']} @ ${signal['signal_price']:.2f} "
              f"(market: ${signal['price']:.2f})")
    
    # 2. GENERATE VECTORBT SIGNALS
    print(f"\n🔍 STEP 2: VectorBT Signal Comparison")
    print("-" * 40)
    
    try:
        import vectorbt as vbt
        
        entries, exits = strategy.generate_vectorbt_signals(data, 100000)
        
        print(f"   VectorBT entries: {entries.sum()} signals")
        print(f"   VectorBT exits: {exits.sum()} signals")
        
        # Show where VectorBT signals occur
        entry_dates = data.index[entries]
        exit_dates = data.index[exits]
        
        print(f"   Entry dates: {[d.strftime('%Y-%m-%d') for d in entry_dates]}")
        print(f"   Exit dates: {[d.strftime('%Y-%m-%d') for d in exit_dates]}")
        
        # Compare signal timing
        if len(all_signals) > 0:
            our_signal_dates = [s['date'] for s in all_signals]
            vbt_signal_dates = list(entry_dates) + list(exit_dates)
            
            print(f"   Our signal dates: {[d.strftime('%Y-%m-%d') for d in our_signal_dates]}")
            print(f"   VBT signal dates: {[d.strftime('%Y-%m-%d') for d in vbt_signal_dates]}")
            
            if set(our_signal_dates) != set(vbt_signal_dates):
                print(f"   ⚠️ SIGNAL TIMING MISMATCH!")
                print(f"   Our only: {set(our_signal_dates) - set(vbt_signal_dates)}")
                print(f"   VBT only: {set(vbt_signal_dates) - set(our_signal_dates)}")
        
    except ImportError:
        print("   ⚠️ VectorBT not available")
        return
    
    # 3. RUN OUR ENGINE WITH DETAILED LOGGING
    print(f"\n🔍 STEP 3: Our Engine Detailed Analysis")
    print("-" * 40)
    
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.001,
        max_position_size=1.0
    )
    
    print(f"   Engine settings:")
    print(f"   - Initial capital: ${engine.initial_capital:,}")
    print(f"   - Commission: {engine.commission:.3f}")
    print(f"   - Slippage: {engine.slippage:.3f}")
    print(f"   - Max position size: {engine.max_position_size:.1%}")
    
    # Run backtest
    engine.run_backtest(data, strategy)
    
    our_final_value = engine.portfolio_values[-1]
    our_return = (our_final_value / 100000 - 1) * 100
    our_trades = len(engine.trades)
    
    print(f"   Final value: ${our_final_value:,.2f}")
    print(f"   Total return: {our_return:+.2f}%")
    print(f"   Trades executed: {our_trades}")
    
    # Analyze each trade in detail
    if engine.trades:
        print(f"\n   📋 Trade Details:")
        total_pnl = 0
        for i, trade in enumerate(engine.trades):
            total_pnl += trade.pnl
            print(f"   Trade {i+1}: {trade.side} {trade.quantity} shares")
            print(f"      Entry: {trade.entry_time.strftime('%Y-%m-%d')} @ ${trade.entry_price:.2f}")
            print(f"      Exit:  {trade.exit_time.strftime('%Y-%m-%d')} @ ${trade.exit_price:.2f}")
            print(f"      Gross P&L: ${trade.gross_pnl:.2f}")
            print(f"      Commission: ${trade.commission_paid:.2f}")
            print(f"      Slippage: ${trade.slippage_cost:.2f}")
            print(f"      Net P&L: ${trade.pnl:.2f}")
            print(f"      Cumulative P&L: ${total_pnl:.2f}")
            print()
    
    # 4. RUN VECTORBT FOR COMPARISON
    print(f"\n🔍 STEP 4: VectorBT Detailed Analysis")
    print("-" * 40)
    
    if entries.sum() > 0 or exits.sum() > 0:
        vbt_portfolio = vbt.Portfolio.from_signals(
            close=data['close'],
            entries=entries,
            exits=exits,
            size=100,
            init_cash=100000,
            fees=0.001,
            freq='D'
        )
        
        vbt_final_value = vbt_portfolio.value().iloc[-1]
        vbt_return = (vbt_final_value / 100000 - 1) * 100
        vbt_trades = len(vbt_portfolio.trades.records)
        
        print(f"   Final value: ${vbt_final_value:,.2f}")
        print(f"   Total return: {vbt_return:+.2f}%")
        print(f"   Trades executed: {vbt_trades}")
        
        # Get VectorBT trade details
        if vbt_trades > 0:
            print(f"\n   📋 VectorBT Trade Details:")
            trades_df = vbt_portfolio.trades.records_readable
            
            for i, trade in trades_df.iterrows():
                print(f"   Trade {i+1}: {trade['Side']} {trade['Size']} shares")
                print(f"      Entry: {trade['Entry Timestamp'].strftime('%Y-%m-%d')} @ ${trade['Entry Price']:.2f}")
                print(f"      Exit:  {trade['Exit Timestamp'].strftime('%Y-%m-%d')} @ ${trade['Exit Price']:.2f}")
                print(f"      P&L: ${trade['PnL']:.2f}")
                print(f"      Return: {trade['Return']:.2%}")
                print()
        
        # 5. DETAILED COMPARISON
        print(f"\n🔍 STEP 5: Detailed Comparison")
        print("-" * 40)
        
        return_diff = abs(our_return - vbt_return)
        value_diff = abs(our_final_value - vbt_final_value)
        
        print(f"   Return difference: {return_diff:.2f}pp")
        print(f"   Value difference: ${value_diff:.2f}")
        print(f"   Trade count difference: {abs(our_trades - vbt_trades)}")
        
        # Check if it's a sign issue
        if our_return < 0 and vbt_return > 0:
            print(f"   ⚠️ SIGN ISSUE: Our framework shows loss, VBT shows gain")
        elif our_return > 0 and vbt_return < 0:
            print(f"   ⚠️ SIGN ISSUE: Our framework shows gain, VBT shows loss")
        
        # Check for magnitude issues
        if abs(our_return) > abs(vbt_return) * 2:
            print(f"   ⚠️ MAGNITUDE ISSUE: Our return is >2x VBT's return")
        elif abs(vbt_return) > abs(our_return) * 2:
            print(f"   ⚠️ MAGNITUDE ISSUE: VBT return is >2x our return")
        
        # Potential causes
        print(f"\n   🔍 Potential Causes:")
        
        if return_diff > 20:
            print(f"      • Signal timing mismatch")
            print(f"      • Different execution prices")
            print(f"      • Commission/slippage calculation differences")
            print(f"      • Position sizing calculation errors")
            print(f"      • Capital management differences")
        
        return {
            'our_return': our_return,
            'vbt_return': vbt_return,
            'return_diff': return_diff,
            'our_trades': our_trades,
            'vbt_trades': vbt_trades,
            'our_signals': len(all_signals),
            'vbt_signals': entries.sum() + exits.sum()
        }
    
    else:
        print("   ⚠️ No VectorBT signals generated")
        return None

if __name__ == "__main__":
    result = debug_framework_differences()
    
    if result:
        print(f"\n🎯 DEBUGGING SUMMARY:")
        print(f"   Our framework: {result['our_return']:+.2f}% ({result['our_trades']} trades)")
        print(f"   VectorBT: {result['vbt_return']:+.2f}% ({result['vbt_trades']} trades)")
        print(f"   Difference: {result['return_diff']:.2f}pp")
        print(f"   Our signals: {result['our_signals']}")
        print(f"   VBT signals: {result['vbt_signals']}")
        
        if result['return_diff'] > 20:
            print(f"\n   ❌ LARGE DIFFERENCE CONFIRMED - NEEDS INVESTIGATION")
        else:
            print(f"\n   ✅ Difference within reasonable range")