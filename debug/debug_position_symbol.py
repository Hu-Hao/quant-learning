#!/usr/bin/env python3
"""
Debug what symbol is used for positions
"""

import yfinance as yf
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

def debug_position_symbol():
    """Debug what symbol is actually used in positions"""
    print("🔍 DEBUGGING POSITION SYMBOL")
    print("=" * 40)
    
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
    
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        slippage=0.001,
        max_position_size=1.0
    )
    
    # Get the first signal to see what symbol it uses
    print("🔍 Getting first signal...")
    for idx, row in data.iterrows():
        partial_data = data.loc[:idx]
        signals = strategy.get_signals(partial_data, available_capital=100000)
        if signals:
            signal = signals[0]
            print(f"   First signal symbol: '{signal.symbol}'")
            print(f"   Signal action: {signal.action}")
            print(f"   Signal price: ${signal.price:.2f}")
            break
    
    # Run partial backtest to get positions
    print(f"\n🔍 Running partial backtest...")
    engine.run_backtest(data, strategy)
    
    print(f"   Positions created: {len(engine.positions)}")
    print(f"   Position symbols: {list(engine.positions.keys())}")
    
    if engine.positions:
        for symbol, pos in engine.positions.items():
            print(f"   Position '{symbol}': {pos.quantity} shares @ ${pos.avg_price:.2f}")
    
    # Check what price dict is used in the last update
    print(f"\n🔍 Checking price dict usage...")
    last_price = data['close'].iloc[-1]
    prices = {'default': last_price}
    print(f"   Price dict: {prices}")
    print(f"   Last market price: ${last_price:.2f}")
    
    # Calculate portfolio value manually
    manual_value = engine.capital
    print(f"   Capital: ${engine.capital:.2f}")
    
    for symbol, position in engine.positions.items():
        print(f"   Checking position '{symbol}' in price dict...")
        if symbol in prices:
            market_value = position.market_value(prices[symbol])
            manual_value += market_value
            print(f"      ✅ Found! Market value: ${market_value:.2f}")
        else:
            print(f"      ❌ NOT FOUND! Using last price instead...")
            market_value = position.market_value(last_price)
            print(f"      Market value should be: ${market_value:.2f}")
            manual_value += market_value
    
    engine_value = engine.get_portfolio_value(prices)
    
    print(f"\n🎯 COMPARISON:")
    print(f"   Engine portfolio value: ${engine_value:.2f}")
    print(f"   Manual calculation: ${manual_value:.2f}")
    print(f"   Final portfolio value: ${engine.portfolio_values[-1]:.2f}")
    print(f"   Difference: ${abs(engine_value - manual_value):.2f}")
    
    if abs(engine_value - manual_value) > 1:
        print(f"   ❌ BUG CONFIRMED: Portfolio value calculation is wrong!")
        return True
    else:
        print(f"   ✅ Portfolio value calculation looks correct")
        return False

if __name__ == "__main__":
    bug_found = debug_position_symbol()
    
    if bug_found:
        print(f"\n🐛 BUG IDENTIFIED:")
        print(f"   The position symbol doesn't match the price dict key")
        print(f"   This causes position values to be ignored in portfolio calculation")
        print(f"   Fix: Update the price dict to use the correct symbol")
    else:
        print(f"\n✅ No obvious bug in position symbol handling")