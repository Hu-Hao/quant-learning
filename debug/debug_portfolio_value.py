#!/usr/bin/env python3
"""
Debug the portfolio value calculation bug
"""

import yfinance as yf
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

def debug_portfolio_value_bug():
    """Debug the portfolio value calculation"""
    print("🔍 DEBUGGING PORTFOLIO VALUE CALCULATION BUG")
    print("=" * 60)
    
    # Get small dataset for easier debugging
    apple = yf.Ticker("AAPL")
    data = apple.history(period="1mo")
    if hasattr(data.index, 'tz') and data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data.columns = [col.lower() for col in data.columns]
    data = data.dropna()
    
    strategy = MovingAverageStrategy(
        short_window=5,
        long_window=10,
        quantity=10  # Smaller quantity for easier tracking
    )
    
    engine = BacktestEngine(
        initial_capital=10000,
        commission=0.001,
        slippage=0.001,
        max_position_size=1.0
    )
    
    print(f"Data points: {len(data)}")
    print(f"Initial capital: ${engine.capital:,.2f}")
    
    # Run backtest with debugging
    engine.reset()
    
    for i, (idx, row) in enumerate(data.iterrows()):
        engine.update_time(row.name if hasattr(row.name, 'date') else idx)
        current_price = row.get('close', row.iloc[-1])
        engine.price_history.append(current_price)
        
        # Get signals
        signals = strategy.get_signals(data.loc[:idx], engine.capital)
        
        print(f"\nDay {i+1} ({idx.strftime('%Y-%m-%d')}): Price=${current_price:.2f}")
        print(f"   Capital: ${engine.capital:.2f}")
        print(f"   Positions: {len(engine.positions)}")
        
        if engine.positions:
            for symbol, pos in engine.positions.items():
                print(f"   Position {symbol}: {pos.quantity} shares @ ${pos.avg_price:.2f}")
                market_value = pos.market_value(current_price)
                print(f"      Market value: ${market_value:.2f}")
        
        # Process signals
        if signals:
            for signal in signals:
                print(f"   Signal: {signal.action} {signal.quantity} @ ${signal.price:.2f}")
                engine._process_signal(signal, current_price)
                
                print(f"   After signal - Capital: ${engine.capital:.2f}")
                if engine.positions:
                    for symbol, pos in engine.positions.items():
                        print(f"   Position {symbol}: {pos.quantity} shares @ ${pos.avg_price:.2f}")
        
        # Update portfolio value - THIS IS WHERE THE BUG IS
        prices = {'default': current_price}
        print(f"   Price dict: {prices}")
        
        # Debug the portfolio value calculation
        total_value = engine.capital
        print(f"   Capital component: ${total_value:.2f}")
        
        if engine.positions:
            for symbol, position in engine.positions.items():
                print(f"   Looking for '{symbol}' in price dict: {symbol in prices}")
                if symbol in prices:
                    market_value = position.market_value(prices[symbol])
                    total_value += market_value
                    print(f"   Added ${market_value:.2f} for {symbol}")
                else:
                    print(f"   ⚠️ MISSING: {symbol} not found in prices dict!")
                    # This is the bug - we should use current_price
                    market_value = position.market_value(current_price)
                    print(f"   Should add ${market_value:.2f} for {symbol}")
        
        calculated_value = engine.get_portfolio_value(prices)
        print(f"   Calculated portfolio value: ${calculated_value:.2f}")
        
        engine.portfolio_values.append(calculated_value)
        
        # Stop after a few iterations or when we get a position
        if i > 10 or engine.positions:
            break
    
    print(f"\n🎯 FINAL ANALYSIS:")
    print(f"   Final capital: ${engine.capital:.2f}")
    print(f"   Final portfolio value: ${engine.portfolio_values[-1]:.2f}")
    print(f"   Positions: {len(engine.positions)}")
    
    if engine.positions:
        print(f"   Position details:")
        for symbol, pos in engine.positions.items():
            print(f"      {symbol}: {pos.quantity} shares @ ${pos.avg_price:.2f}")
            print(f"      Market value @ ${data['close'].iloc[-1]:.2f}: ${pos.market_value(data['close'].iloc[-1]):.2f}")
    
    # The bug is that positions use a symbol key (like 'default' or strategy-specific)
    # but we're passing {'default': price} but the position might have a different symbol
    return engine.positions

if __name__ == "__main__":
    positions = debug_portfolio_value_bug()
    
    if positions:
        print(f"\n🐛 BUG CONFIRMED:")
        print(f"   Position symbols: {list(positions.keys())}")
        print(f"   Price dict uses: 'default'")
        print(f"   If symbols don't match, position values aren't included in portfolio value!")
    else:
        print(f"\n📊 No positions created in test period")