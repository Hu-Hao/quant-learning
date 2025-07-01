#!/usr/bin/env python3
"""
Simple debug to identify the key difference
"""

import yfinance as yf
import numpy as np
import pandas as pd
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

def debug_key_difference():
    """Debug the key difference causing large returns"""
    print("🔍 SIMPLE DEBUGGING - KEY DIFFERENCE")
    print("=" * 50)
    
    # Get Apple data
    apple = yf.Ticker("AAPL")
    data = apple.history(period="6mo")
    if hasattr(data.index, 'tz') and data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data.columns = [col.lower() for col in data.columns]
    data = data.dropna()
    
    # Create strategy
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        quantity=100
    )
    
    print(f"Data points: {len(data)}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # 1. Check our signals
    print(f"\n🔍 OUR SIGNALS:")
    all_signals = []
    for idx, row in data.iterrows():
        partial_data = data.loc[:idx]
        signals = strategy.get_signals(partial_data, available_capital=100000)
        if signals:
            for signal in signals:
                all_signals.append({
                    'date': idx,
                    'action': str(signal.action),
                    'price': signal.price,
                    'market_price': row['close']
                })
    
    for i, sig in enumerate(all_signals):
        print(f"   {i+1}: {sig['date'].strftime('%Y-%m-%d')} {sig['action']} @ ${sig['price']:.2f} (market: ${sig['market_price']:.2f})")
    
    # 2. Run our engine
    print(f"\n🔍 OUR ENGINE:")
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.001,
        max_position_size=1.0
    )
    
    engine.run_backtest(data, strategy)
    
    print(f"   Initial capital: ${engine.initial_capital:,}")
    print(f"   Final value: ${engine.portfolio_values[-1]:,.2f}")
    print(f"   Return: {(engine.portfolio_values[-1]/100000-1)*100:+.2f}%")
    print(f"   Trades: {len(engine.trades)}")
    
    # Print capital flow
    print(f"\n💰 CAPITAL FLOW:")
    print(f"   Started with: ${100000:,}")
    
    capital_tracker = 100000
    for i, trade in enumerate(engine.trades):
        capital_before = capital_tracker
        capital_tracker += trade.pnl
        
        print(f"   Trade {i+1}: {trade.side} {trade.quantity} @ ${trade.entry_price:.2f} → ${trade.exit_price:.2f}")
        print(f"      Gross P&L: ${trade.gross_pnl:.2f}")
        print(f"      Net P&L: ${trade.pnl:.2f} (after ${trade.commission_paid:.2f} commission + ${trade.slippage_cost:.2f} slippage)")
        print(f"      Capital: ${capital_before:,.2f} → ${capital_tracker:,.2f}")
    
    print(f"   Final capital: ${capital_tracker:,.2f}")
    print(f"   Engine final: ${engine.portfolio_values[-1]:,.2f}")
    
    # 3. Compare with VectorBT
    print(f"\n🔍 VECTORBT:")
    try:
        import vectorbt as vbt
        
        entries, exits = strategy.generate_vectorbt_signals(data, 100000)
        
        print(f"   Entries: {entries.sum()}")
        print(f"   Exits: {exits.sum()}")
        
        if entries.sum() > 0:
            vbt_portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries,
                exits=exits,
                size=100,
                init_cash=100000,
                fees=0.001,
                freq='D'
            )
            
            vbt_final = vbt_portfolio.value().iloc[-1]
            vbt_return = (vbt_final / 100000 - 1) * 100
            
            print(f"   Final value: ${vbt_final:,.2f}")
            print(f"   Return: {vbt_return:+.2f}%")
            print(f"   Trades: {len(vbt_portfolio.trades.records)}")
            
            # Key insight: Check if VectorBT is using different execution logic
            print(f"\n🎯 KEY DIFFERENCES:")
            our_return = (engine.portfolio_values[-1]/100000-1)*100
            print(f"   Our return: {our_return:+.2f}%")
            print(f"   VBT return: {vbt_return:+.2f}%")
            print(f"   Difference: {abs(our_return - vbt_return):.2f}pp")
            
            # Check if the issue is in execution prices vs signal prices
            print(f"\n🔍 EXECUTION PRICE ANALYSIS:")
            
            for i, trade in enumerate(engine.trades):
                print(f"   Our Trade {i+1}:")
                print(f"      Signal price: Looking for this...")
                print(f"      Execution price: ${trade.entry_price:.2f} → ${trade.exit_price:.2f}")
                print(f"      Market price at signal: Need to check")
                
                # Find the corresponding signal
                trade_date = trade.entry_time.date()
                matching_signals = [s for s in all_signals if s['date'].date() == trade_date]
                if matching_signals:
                    signal = matching_signals[0]
                    price_diff = abs(trade.entry_price - signal['price'])
                    print(f"      Signal vs execution: ${signal['price']:.2f} vs ${trade.entry_price:.2f} (diff: ${price_diff:.2f})")
                    
                    # This is likely the issue - slippage is making us buy higher and sell lower
                    if price_diff > 1.0:
                        print(f"      ⚠️ LARGE PRICE DIFFERENCE - SLIPPAGE ISSUE!")
            
            return {
                'our_return': our_return,
                'vbt_return': vbt_return,
                'difference': abs(our_return - vbt_return)
            }
    
    except ImportError:
        print("   VectorBT not available")
        return None

if __name__ == "__main__":
    result = debug_key_difference()
    
    if result:
        if result['difference'] > 20:
            print(f"\n❌ CONFIRMED: Large difference of {result['difference']:.2f}pp")
            print(f"   Most likely causes:")
            print(f"   1. Slippage calculation making execution prices worse")
            print(f"   2. Commission calculation differences") 
            print(f"   3. Different capital management logic")
        else:
            print(f"\n✅ Difference within reasonable range")