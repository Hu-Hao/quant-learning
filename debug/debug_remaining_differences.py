#!/usr/bin/env python3
"""
Debug the remaining differences between our framework and VectorBT
"""

import pandas as pd
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

def debug_remaining_differences():
    """Debug what's causing the remaining 31pp difference"""
    print("🔍 DEBUGGING REMAINING DIFFERENCES")
    print("=" * 50)
    
    # Use the same Apple data from our test
    apple_data = {
        '2024-07-01': {'close': 194.48}, '2024-07-02': {'close': 193.32}, '2024-07-03': {'close': 192.58},
        '2024-07-05': {'close': 191.29}, '2024-07-08': {'close': 188.89}, '2024-07-09': {'close': 191.15},
        '2024-07-10': {'close': 191.29}, '2024-07-11': {'close': 192.00}, '2024-07-12': {'close': 192.25},
        '2024-07-15': {'close': 191.52}, '2024-07-16': {'close': 191.29}, '2024-07-17': {'close': 194.68},
        '2024-07-18': {'close': 194.30}, '2024-07-19': {'close': 194.16}, '2024-07-22': {'close': 195.55},
        '2024-07-23': {'close': 196.28}, '2024-07-24': {'close': 194.86}, '2024-07-25': {'close': 192.78},
        '2024-07-26': {'close': 191.57}, '2024-07-29': {'close': 192.49}, '2024-07-30': {'close': 192.27},
        '2024-07-31': {'close': 192.01}, '2024-08-01': {'close': 188.40}, '2024-08-02': {'close': 185.04},
        '2024-08-05': {'close': 198.87}, '2024-08-06': {'close': 207.23}, '2024-08-07': {'close': 209.82},
        '2024-08-08': {'close': 213.31}, '2024-08-09': {'close': 216.24}, '2024-08-12': {'close': 217.53},
        '2024-08-13': {'close': 221.27}, '2024-08-14': {'close': 221.17}, '2024-08-15': {'close': 224.72},
        '2024-08-16': {'close': 224.24}, '2024-08-19': {'close': 225.77}
    }
    
    dates = pd.to_datetime(list(apple_data.keys()))
    data = pd.DataFrame({
        'open': [apple_data[d.strftime('%Y-%m-%d')]['close'] for d in dates],
        'high': [apple_data[d.strftime('%Y-%m-%d')]['close'] * 1.01 for d in dates],
        'low': [apple_data[d.strftime('%Y-%m-%d')]['close'] * 0.99 for d in dates],
        'close': [apple_data[d.strftime('%Y-%m-%d')]['close'] for d in dates],
        'volume': [1000000] * len(dates)
    }, index=dates)
    
    strategy = MovingAverageStrategy(
        short_window=5,
        long_window=15,
        quantity=100
    )
    
    print(f"Data points: {len(data)}")
    print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # 1. RUN OUR FRAMEWORK WITH DETAILED TRACKING
    print(f"\n🔍 OUR FRAMEWORK ANALYSIS:")
    
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.0,
        slippage=0.0,
        max_position_size=1.0
    )
    
    # Track step by step
    capital_history = []
    portfolio_history = []
    
    engine.reset()
    for i, (idx, row) in enumerate(data.iterrows()):
        engine.update_time(idx)
        current_price = row['close']
        engine.price_history.append(current_price)
        
        signals = strategy.get_signals(data.loc[:idx], engine.capital)
        
        capital_before = engine.capital
        positions_before = len(engine.positions)
        
        if signals:
            signal = signals[0]
            print(f"\n   Signal on {idx.strftime('%Y-%m-%d')}:")
            print(f"      Action: {signal.action}")
            print(f"      Price: ${signal.price:.2f} (market: ${current_price:.2f})")
            print(f"      Quantity: {signal.quantity}")
            print(f"      Capital before: ${capital_before:,.2f}")
            
            engine._process_signal(signal, current_price)
            
            print(f"      Capital after: ${engine.capital:,.2f}")
            print(f"      Positions: {len(engine.positions)}")
            
            if engine.positions:
                for symbol, pos in engine.positions.items():
                    market_value = pos.market_value(current_price)
                    print(f"      Position {symbol}: {pos.quantity} @ ${pos.avg_price:.2f}, value: ${market_value:,.2f}")
            
            if engine.trades:
                latest_trade = engine.trades[-1]
                print(f"      Latest trade P&L: ${latest_trade.pnl:.2f}")
        
        # Update portfolio value
        prices = {'default': current_price}
        portfolio_value = engine.get_portfolio_value(prices)
        engine.portfolio_values.append(portfolio_value)
        
        capital_history.append(engine.capital)
        portfolio_history.append(portfolio_value)
    
    our_final = engine.portfolio_values[-1]
    our_return = (our_final / 100000 - 1) * 100
    
    print(f"\n   Final Results:")
    print(f"      Final portfolio value: ${our_final:,.2f}")
    print(f"      Total return: {our_return:+.2f}%")
    print(f"      Trades: {len(engine.trades)}")
    print(f"      Final capital: ${engine.capital:,.2f}")
    
    # 2. RUN VECTORBT
    print(f"\n🔍 VECTORBT ANALYSIS:")
    
    try:
        import vectorbt as vbt
        
        entries, exits = strategy.generate_vectorbt_signals(data, 100000)
        
        print(f"   Entries: {entries.sum()}")
        print(f"   Exits: {exits.sum()}")
        
        if entries.sum() > 0:
            print(f"   Entry dates: {data.index[entries].strftime('%Y-%m-%d').tolist()}")
            print(f"   Exit dates: {data.index[exits].strftime('%Y-%m-%d').tolist()}")
            
            portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries,
                exits=exits,
                size=100,
                init_cash=100000,
                fees=0.0,
                freq='D'
            )
            
            vbt_final = portfolio.value().iloc[-1]
            vbt_return = (vbt_final / 100000 - 1) * 100
            
            print(f"   Final value: ${vbt_final:,.2f}")
            print(f"   Total return: {vbt_return:+.2f}%")
            print(f"   Trades: {len(portfolio.trades.records)}")
            
            # Get VectorBT trade details
            if len(portfolio.trades.records) > 0:
                print(f"   VectorBT trades:")
                for i, trade_record in enumerate(portfolio.trades.records):
                    print(f"      Trade {i+1}: size={trade_record[4]}, entry_price=${trade_record[6]:.2f}, exit_price=${trade_record[7]:.2f}")
                    pnl = trade_record[4] * (trade_record[7] - trade_record[6])  # size * (exit - entry)
                    print(f"                  P&L=${pnl:.2f}")
            
            # 3. DETAILED COMPARISON
            print(f"\n🔍 DETAILED COMPARISON:")
            diff = abs(our_return - vbt_return)
            print(f"   Our return: {our_return:+.2f}%")
            print(f"   VBT return: {vbt_return:+.2f}%")
            print(f"   Difference: {diff:.2f}pp")
            
            # Check if our trade P&L matches VectorBT
            if engine.trades and len(portfolio.trades.records) > 0:
                our_trade_pnl = sum(t.pnl for t in engine.trades)
                vbt_trade_pnl = sum(r[4] * (r[7] - r[6]) for r in portfolio.trades.records)
                
                print(f"   Our total trade P&L: ${our_trade_pnl:.2f}")
                print(f"   VBT total trade P&L: ${vbt_trade_pnl:.2f}")
                print(f"   Trade P&L difference: ${abs(our_trade_pnl - vbt_trade_pnl):.2f}")
                
                if abs(our_trade_pnl - vbt_trade_pnl) > 100:
                    print(f"   ❌ TRADE P&L MISMATCH - this is the root cause!")
                    
                    # Compare individual trades
                    print(f"   Individual trade comparison:")
                    for i, (our_trade, vbt_record) in enumerate(zip(engine.trades, portfolio.trades.records)):
                        our_pnl = our_trade.pnl
                        vbt_pnl = vbt_record[4] * (vbt_record[7] - vbt_record[6])
                        print(f"      Trade {i+1}: Our P&L=${our_pnl:.2f}, VBT P&L=${vbt_pnl:.2f}")
                        print(f"                 Our: {our_trade.quantity} @ ${our_trade.entry_price:.2f} → ${our_trade.exit_price:.2f}")
                        print(f"                 VBT: {vbt_record[4]} @ ${vbt_record[6]:.2f} → ${vbt_record[7]:.2f}")
                else:
                    print(f"   ✅ Trade P&L matches - issue is elsewhere")
            
            return {
                'our_return': our_return,
                'vbt_return': vbt_return,
                'difference': diff,
                'our_final': our_final,
                'vbt_final': vbt_final
            }
    
    except ImportError:
        print("   VectorBT not available")
        return None

if __name__ == "__main__":
    result = debug_remaining_differences()
    
    if result:
        print(f"\n🎯 ROOT CAUSE ANALYSIS:")
        if result['difference'] > 10:
            print(f"   ❌ MAJOR DIFFERENCE: {result['difference']:.2f}pp")
            print(f"   🔍 Need to investigate trade execution logic")
            print(f"   💰 Value difference: ${abs(result['our_final'] - result['vbt_final']):,.2f}")
        else:
            print(f"   ✅ Minor difference: {result['difference']:.2f}pp")
            print(f"   📈 Frameworks are reasonably aligned")