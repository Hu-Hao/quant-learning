#!/usr/bin/env python3
"""
Debug the complete trade cycle to find where the capital is disappearing
"""

import yfinance as yf
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

def debug_trade_cycle():
    """Debug what happens during trade execution"""
    print("🔍 DEBUGGING COMPLETE TRADE CYCLE")
    print("=" * 50)
    
    # Get Apple data
    apple = yf.Ticker("AAPL")
    data = apple.history(period="6mo")
    if hasattr(data.index, 'tz') and data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data.columns = [col.lower() for col in data.columns]
    data = data.dropna()
    
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        quantity=100
    )
    
    # Create engine with minimal costs to isolate the issue
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.000,  # No commission
        slippage=0.000,    # No slippage
        max_position_size=1.0
    )
    
    print(f"Initial capital: ${engine.initial_capital:,}")
    
    # Track capital at each step
    capital_history = []
    portfolio_history = []
    
    engine.reset()
    
    signal_count = 0
    for i, (idx, row) in enumerate(data.iterrows()):
        engine.update_time(row.name if hasattr(row.name, 'date') else idx)
        current_price = row.get('close', row.iloc[-1])
        engine.price_history.append(current_price)
        
        # Get signals
        signals = strategy.get_signals(data.loc[:idx], engine.capital)
        
        if signals:
            signal_count += 1
            signal = signals[0]
            
            print(f"\n📊 SIGNAL {signal_count} on {idx.strftime('%Y-%m-%d')}:")
            print(f"   Market price: ${current_price:.2f}")
            print(f"   Signal: {signal.action} {signal.quantity} @ ${signal.price:.2f}")
            print(f"   Capital before: ${engine.capital:.2f}")
            print(f"   Positions before: {len(engine.positions)}")
            
            # Process the signal
            engine._process_signal(signal, current_price)
            
            print(f"   Capital after: ${engine.capital:.2f}")
            print(f"   Positions after: {len(engine.positions)}")
            
            if engine.positions:
                for symbol, pos in engine.positions.items():
                    print(f"   Position {symbol}: {pos.quantity} shares @ ${pos.avg_price:.2f}")
            
            # Check trades
            if engine.trades:
                latest_trade = engine.trades[-1]
                print(f"   Latest trade: {latest_trade.side} {latest_trade.quantity} @ ${latest_trade.entry_price:.2f} → ${latest_trade.exit_price:.2f}")
                print(f"   Trade P&L: ${latest_trade.pnl:.2f}")
        
        # Update portfolio value
        prices = {'default': current_price}
        portfolio_value = engine.get_portfolio_value(prices)
        
        capital_history.append(engine.capital)
        portfolio_history.append(portfolio_value)
        
        engine.portfolio_values.append(portfolio_value)
        
        # If we have all 4 expected signals, break early to analyze
        if signal_count >= 4:
            break
    
    print(f"\n🎯 FINAL ANALYSIS:")
    print(f"   Signals processed: {signal_count}")
    print(f"   Trades completed: {len(engine.trades)}")
    print(f"   Final capital: ${engine.capital:.2f}")
    print(f"   Final portfolio value: ${engine.portfolio_values[-1]:.2f}")
    
    # Analyze the capital flow
    print(f"\n💰 CAPITAL FLOW ANALYSIS:")
    trade_pnl_sum = sum(trade.pnl for trade in engine.trades)
    expected_capital = 100000 + trade_pnl_sum
    
    print(f"   Started with: $100,000")
    print(f"   Total trade P&L: ${trade_pnl_sum:.2f}")
    print(f"   Expected capital: ${expected_capital:.2f}")
    print(f"   Actual capital: ${engine.capital:.2f}")
    print(f"   Capital difference: ${engine.capital - expected_capital:.2f}")
    print(f"   Portfolio value: ${engine.portfolio_values[-1]:.2f}")
    
    # Check if there are open positions
    if engine.positions:
        print(f"\n🔍 OPEN POSITIONS:")
        total_position_value = 0
        for symbol, pos in engine.positions.items():
            market_value = pos.market_value(current_price)
            total_position_value += market_value
            print(f"   {symbol}: {pos.quantity} shares @ ${pos.avg_price:.2f}, market value: ${market_value:.2f}")
        
        theoretical_portfolio = engine.capital + total_position_value
        print(f"   Total position value: ${total_position_value:.2f}")
        print(f"   Theoretical portfolio: ${theoretical_portfolio:.2f}")
        print(f"   Actual portfolio: ${engine.portfolio_values[-1]:.2f}")
        print(f"   Portfolio calculation error: ${engine.portfolio_values[-1] - theoretical_portfolio:.2f}")
    else:
        print(f"   No open positions - all capital should be in cash")
        print(f"   Cash vs Portfolio difference: ${engine.portfolio_values[-1] - engine.capital:.2f}")
    
    # The key insight: Are we double-counting losses somewhere?
    print(f"\n🔍 DOUBLE-COUNTING CHECK:")
    
    for i, trade in enumerate(engine.trades):
        print(f"   Trade {i+1}: P&L ${trade.pnl:.2f}")
    
    # Compare with VectorBT to see what the correct result should be
    print(f"\n🔧 VECTORBT COMPARISON:")
    try:
        import vectorbt as vbt
        
        entries, exits = strategy.generate_vectorbt_signals(data, 100000)
        
        if entries.sum() > 0:
            vbt_portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries,
                exits=exits,
                size=100,
                init_cash=100000,
                fees=0.000,  # Same as our engine
                freq='D'
            )
            
            vbt_final = vbt_portfolio.value().iloc[-1]
            vbt_return = (vbt_final / 100000 - 1) * 100
            
            print(f"   VectorBT final value: ${vbt_final:.2f}")
            print(f"   VectorBT return: {vbt_return:+.2f}%")
            
            our_return = (engine.portfolio_values[-1] / 100000 - 1) * 100
            print(f"   Our return: {our_return:+.2f}%")
            print(f"   Difference: {abs(our_return - vbt_return):.2f}pp")
            
            # This will tell us if the issue is in our trade execution logic
            if abs(our_return - vbt_return) > 1:
                print(f"   ❌ SIGNIFICANT DIFFERENCE EVEN WITH NO COSTS!")
                return True
            else:
                print(f"   ✅ Returns match when costs are removed")
                return False
    
    except ImportError:
        print("   VectorBT not available")
        return None

if __name__ == "__main__":
    bug_confirmed = debug_trade_cycle()
    
    if bug_confirmed:
        print(f"\n🐛 BUG CONFIRMED:")
        print(f"   Even with zero commission and slippage, we have different returns")
        print(f"   This indicates a fundamental error in trade execution logic")
    elif bug_confirmed is False:
        print(f"\n💡 INSIGHT:")
        print(f"   Returns match when costs are removed")
        print(f"   The difference is likely due to commission/slippage calculation")
    else:
        print(f"\n❓ Unable to compare with VectorBT")