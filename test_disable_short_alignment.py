#!/usr/bin/env python3
"""
Test our framework with short selling disabled vs VectorBT
"""

import pandas as pd
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

def test_disable_short_vs_vectorbt():
    """Test our framework with short selling disabled vs VectorBT"""
    print("🔍 TESTING DISABLE SHORT vs VECTORBT")
    print("=" * 50)
    
    # Use the same Apple data
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
    
    print(f"Data: {len(data)} days, price range ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # 1. OUR FRAMEWORK WITH SHORT SELLING ENABLED (current behavior)
    print(f"\n🔍 OUR FRAMEWORK - SHORT SELLING ENABLED:")
    
    engine_short_enabled = BacktestEngine(
        initial_capital=100000,
        commission=0.0,
        slippage=0.0,
        max_position_size=1.0,
        allow_short_selling=True  # Default behavior
    )
    
    engine_short_enabled.run_backtest(data, strategy)
    
    short_enabled_return = (engine_short_enabled.portfolio_values[-1] / 100000 - 1) * 100
    print(f"   Final value: ${engine_short_enabled.portfolio_values[-1]:,.2f}")
    print(f"   Return: {short_enabled_return:+.2f}%")
    print(f"   Trades: {len(engine_short_enabled.trades)}")
    
    if engine_short_enabled.trades:
        for i, trade in enumerate(engine_short_enabled.trades):
            print(f"   Trade {i+1}: {trade.side} {trade.quantity} @ ${trade.entry_price:.2f} → ${trade.exit_price:.2f}, P&L: ${trade.pnl:.2f}")
    
    # 2. OUR FRAMEWORK WITH SHORT SELLING DISABLED
    print(f"\n🔍 OUR FRAMEWORK - SHORT SELLING DISABLED:")
    
    engine_short_disabled = BacktestEngine(
        initial_capital=100000,
        commission=0.0,
        slippage=0.0,
        max_position_size=1.0,
        allow_short_selling=False  # Disable short selling
    )
    
    # Track signals that get ignored
    ignored_signals = []
    
    engine_short_disabled.reset()
    for i, (idx, row) in enumerate(data.iterrows()):
        engine_short_disabled.update_time(idx)
        current_price = row['close']
        engine_short_disabled.price_history.append(current_price)
        
        signals = strategy.get_signals(data.loc[:idx], engine_short_disabled.capital)
        
        if signals:
            signal = signals[0]
            
            # Check what happens to this signal
            positions_before = len(engine_short_disabled.positions)
            capital_before = engine_short_disabled.capital
            
            if 'SELL' in str(signal.action):
                current_position = engine_short_disabled.positions.get('default')
                if current_position is None or current_position.quantity <= 0:
                    ignored_signals.append((idx, signal, "No long position to close"))
                    print(f"   IGNORED SELL signal on {idx.strftime('%Y-%m-%d')} @ ${signal.price:.2f} - No long position")
            
            engine_short_disabled._process_signal(signal, current_price)
            
            positions_after = len(engine_short_disabled.positions)
            capital_after = engine_short_disabled.capital
            
            if positions_before != positions_after or capital_before != capital_after:
                print(f"   EXECUTED signal on {idx.strftime('%Y-%m-%d')}: {signal.action} @ ${signal.price:.2f}")
                print(f"      Capital: ${capital_before:,.2f} → ${capital_after:,.2f}")
        
        # Update portfolio value
        prices = {'default': current_price}
        portfolio_value = engine_short_disabled.get_portfolio_value(prices)
        engine_short_disabled.portfolio_values.append(portfolio_value)
    
    short_disabled_return = (engine_short_disabled.portfolio_values[-1] / 100000 - 1) * 100
    print(f"   Final value: ${engine_short_disabled.portfolio_values[-1]:,.2f}")
    print(f"   Return: {short_disabled_return:+.2f}%")
    print(f"   Trades: {len(engine_short_disabled.trades)}")
    print(f"   Ignored signals: {len(ignored_signals)}")
    
    if engine_short_disabled.trades:
        for i, trade in enumerate(engine_short_disabled.trades):
            print(f"   Trade {i+1}: {trade.side} {trade.quantity} @ ${trade.entry_price:.2f} → ${trade.exit_price:.2f}, P&L: ${trade.pnl:.2f}")
    
    # 3. VECTORBT COMPARISON
    print(f"\n🔍 VECTORBT:")
    
    try:
        import vectorbt as vbt
        
        entries, exits = strategy.generate_vectorbt_signals(data, 100000)
        
        print(f"   Generated signals:")
        print(f"      Entries: {entries.sum()} on {data.index[entries].strftime('%Y-%m-%d').tolist()}")
        print(f"      Exits: {exits.sum()} on {data.index[exits].strftime('%Y-%m-%d').tolist()}")
        
        if entries.sum() > 0:
            portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries,
                exits=exits,
                size=100,
                init_cash=100000,
                fees=0.0,
                freq='D'
            )
            
            vbt_return = (portfolio.value().iloc[-1] / 100000 - 1) * 100
            print(f"   Final value: ${portfolio.value().iloc[-1]:,.2f}")
            print(f"   Return: {vbt_return:+.2f}%")
            print(f"   Trades: {len(portfolio.trades.records)}")
            
            # VectorBT trade details  
            if len(portfolio.trades.records) > 0:
                print(f"   VectorBT trade details:")
                try:
                    for i, record in enumerate(portfolio.trades.records):
                        # Try to access VectorBT record fields safely
                        if hasattr(record, 'size'):
                            size = record.size
                            entry_price = record.entry_price
                            exit_price = record.exit_price
                        else:
                            # Fallback: assume it's a structured array
                            size = record[2] if len(record) > 2 else 100
                            entry_price = record[5] if len(record) > 5 else 0
                            exit_price = record[6] if len(record) > 6 else 0
                        
                        pnl = size * (exit_price - entry_price)
                        print(f"      Trade {i+1}: {size} shares @ ${entry_price:.2f} → ${exit_price:.2f}, P&L: ${pnl:.2f}")
                except Exception as e:
                    print(f"      Could not parse VectorBT trade details: {e}")
                    print(f"      Trade count: {len(portfolio.trades.records)}")
            
            # 4. COMPARISON
            print(f"\n🎯 COMPARISON RESULTS:")
            print(f"   Our (short enabled):  {short_enabled_return:+.2f}%")
            print(f"   Our (short disabled): {short_disabled_return:+.2f}%")
            print(f"   VectorBT:             {vbt_return:+.2f}%")
            
            short_disabled_diff = abs(short_disabled_return - vbt_return)
            short_enabled_diff = abs(short_enabled_return - vbt_return)
            
            print(f"   Difference (short disabled): {short_disabled_diff:.2f}pp")
            print(f"   Difference (short enabled):  {short_enabled_diff:.2f}pp")
            
            if short_disabled_diff < short_enabled_diff:
                print(f"   ✅ SHORT DISABLED is closer to VectorBT!")
                print(f"   💡 VectorBT likely doesn't support short selling by default")
            else:
                print(f"   ⚠️ SHORT ENABLED is still closer to VectorBT")
            
            if short_disabled_diff < 1.0:
                print(f"   🎉 EXCELLENT ALIGNMENT with short selling disabled!")
            elif short_disabled_diff < 5.0:
                print(f"   ✅ GOOD ALIGNMENT with short selling disabled")
            else:
                print(f"   ❌ Still significant differences")
            
            return {
                'short_enabled_return': short_enabled_return,
                'short_disabled_return': short_disabled_return,
                'vbt_return': vbt_return,
                'short_disabled_diff': short_disabled_diff,
                'short_enabled_diff': short_enabled_diff
            }
        else:
            print(f"   No VectorBT signals generated")
            return None
    
    except ImportError:
        print("   VectorBT not available")
        return None

if __name__ == "__main__":
    result = test_disable_short_vs_vectorbt()
    
    if result:
        print(f"\n🎯 FINAL ANALYSIS:")
        
        if result['short_disabled_diff'] < result['short_enabled_diff']:
            improvement = result['short_enabled_diff'] - result['short_disabled_diff']
            print(f"   ✅ Disabling short selling improves alignment by {improvement:.2f}pp")
            print(f"   📝 Recommendation: Use allow_short_selling=False for VectorBT compatibility")
            
            if result['short_disabled_diff'] < 1.0:
                print(f"   🎉 FRAMEWORK ALIGNMENT ACHIEVED!")
            else:
                print(f"   📊 Remaining difference: {result['short_disabled_diff']:.2f}pp")
        else:
            print(f"   ⚠️ Short selling behavior doesn't explain the differences")
            print(f"   🔍 Other factors are causing the alignment issues")
    
    else:
        print(f"\n❓ Unable to compare with VectorBT")