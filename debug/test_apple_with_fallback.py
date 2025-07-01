#!/usr/bin/env python3
"""
Test Apple stock with robust data fetching and framework comparison
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def setup_environment():
    """Setup environment for data fetching"""
    # Set timezone for NYSE/NASDAQ
    os.environ['TZ'] = 'America/New_York'
    
    try:
        import time
        time.tzset()
        print("🕒 Timezone set to Eastern Time")
    except:
        pass

def try_download_apple():
    """Try to download real Apple data with multiple strategies"""
    print("📈 Attempting to download real Apple data...")
    
    setup_environment()
    
    try:
        import yfinance as yf
        
        strategies = [
            ("Standard 1-year", lambda: yf.Ticker("AAPL").history(period="1y")),
            ("6-month period", lambda: yf.Ticker("AAPL").history(period="6mo")),
            ("3-month period", lambda: yf.Ticker("AAPL").history(period="3mo")),
            ("Download method", lambda: yf.download("AAPL", period="6mo", progress=False)),
        ]
        
        for name, strategy in strategies:
            try:
                print(f"   🧪 Trying: {name}")
                data = strategy()
                
                if data is not None and not data.empty and len(data) > 50:
                    # Clean the data
                    if hasattr(data.index, 'tz') and data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    
                    # Standardize column names
                    data.columns = [col.lower() for col in data.columns]
                    data = data.dropna()
                    
                    print(f"   ✅ Success! Downloaded {len(data)} days")
                    print(f"   📅 Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                    return data, True
                    
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)[:60]}...")
                continue
        
        print("   ❌ All real data strategies failed")
        return None, False
        
    except ImportError:
        print("   ❌ yfinance not available")
        return None, False

def create_apple_like_data():
    """Create realistic Apple-like data"""
    print("🔄 Creating Apple-like sample data...")
    
    # 1 year of trading days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.dayofweek < 5]  # Only weekdays
    
    # Apple characteristics
    initial_price = 150.0
    n_days = len(dates)
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    # Generate realistic returns
    # Base trend: ~15% annual growth
    trend = 0.15 / 252  # Daily trend
    volatility = 0.25 / np.sqrt(252)  # Daily volatility (~25% annual)
    
    returns = np.random.normal(trend, volatility, n_days)
    
    # Add realistic patterns
    # 1. Earnings reactions (quarterly)
    quarters = [int(n_days * 0.25), int(n_days * 0.5), int(n_days * 0.75)]
    for q in quarters:
        if q < len(returns):
            returns[q] += np.random.choice([-0.03, 0.05], p=[0.4, 0.6])
    
    # 2. Volatility clustering
    for i in range(1, len(returns)):
        if abs(returns[i-1]) > 0.02:
            returns[i] *= 1.3
    
    # 3. Some trending periods
    trend_start = int(n_days * 0.3)
    for i in range(trend_start, min(trend_start + 20, len(returns))):
        returns[i] += 0.002  # Small positive trend
    
    # Calculate prices
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Generate intraday OHLC
        open_price = price * np.random.uniform(0.998, 1.002)
        close_price = price
        
        # High and low with realistic spreads
        high_price = max(open_price, close_price) * np.random.uniform(1.001, 1.01)
        low_price = min(open_price, close_price) * np.random.uniform(0.99, 0.999)
        
        # Apple typical volume
        volume = np.random.randint(50000000, 120000000)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates[:len(prices)])
    
    print(f"   ✅ Created {len(df)} days of Apple-like data")
    print(f"   📈 Total return: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:.1f}%")
    print(f"   💰 Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    return df, False

def test_framework_differences(data, is_real_data):
    """Test our framework vs VectorBT with the data"""
    print(f"\n🧪 TESTING FRAMEWORK DIFFERENCES")
    print(f"   Data source: {'Real Apple' if is_real_data else 'Apple-like simulation'}")
    print("=" * 50)
    
    # Import our framework
    from quant_trading.strategies.moving_average import MovingAverageStrategy
    from quant_trading.backtesting.engine import BacktestEngine
    
    # Create fixed quantity strategy (the problematic case)
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        quantity=100  # Fixed 100 shares - this is where differences occur
    )
    
    initial_capital = 100000
    
    print(f"📊 Strategy: {strategy.get_name()}")
    print(f"   Parameters: {strategy.get_parameters()}")
    print(f"   Position sizing: Fixed {strategy.quantity} shares")
    
    # Test our framework
    print(f"\n🔧 Testing Our Framework:")
    
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=0.001,
        slippage=0.0,
        max_position_size=1.0,  # Allow full allocation
        allow_short_selling=False
    )
    
    engine.run_backtest(data, strategy)
    
    our_final_value = engine.portfolio_values[-1]
    our_return = (our_final_value / initial_capital - 1) * 100
    our_trades = len(engine.trades)
    
    print(f"   Final Value: ${our_final_value:,.2f}")
    print(f"   Total Return: {our_return:+.2f}%")
    print(f"   Total Trades: {our_trades}")
    
    # Debug signal generation
    all_signals = []
    for idx, _ in data.iterrows():
        partial_data = data.loc[:idx]
        signals = strategy.get_signals(partial_data, available_capital=initial_capital)
        all_signals.extend(signals)
    
    buy_signals = [s for s in all_signals if s.action.value == 'buy']
    sell_signals = [s for s in all_signals if s.action.value == 'sell']
    
    print(f"   Signals: {len(buy_signals)} buys, {len(sell_signals)} sells")
    print(f"   Execution rate: {our_trades / len(all_signals) * 100:.1f}%" if all_signals else "   No signals")
    
    # Test VectorBT
    print(f"\n🔧 Testing VectorBT:")
    
    try:
        import vectorbt as vbt
        
        # Generate signals
        entries, exits = strategy.generate_vectorbt_signals(data, initial_capital)
        
        print(f"   VectorBT signals: {entries.sum()} entries, {exits.sum()} exits")
        
        if entries.sum() > 0 or exits.sum() > 0:
            # Create VectorBT portfolio
            vbt_portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries,
                exits=exits,
                size=strategy.quantity,  # Fixed 100 shares
                init_cash=initial_capital,
                fees=0.001,
                freq='D'
            )
            
            vbt_stats = vbt_portfolio.stats()
            vbt_final_value = vbt_portfolio.value().iloc[-1]
            vbt_return = vbt_stats['Total Return [%]']
            vbt_trades = vbt_stats['Total Trades']
            
            print(f"   Final Value: ${vbt_final_value:,.2f}")
            print(f"   Total Return: {vbt_return:+.2f}%")
            print(f"   Total Trades: {vbt_trades}")
            
            # Analysis
            return_diff = abs(our_return - vbt_return)
            trade_diff = abs(our_trades - vbt_trades)
            value_diff = abs(our_final_value - vbt_final_value)
            
            print(f"\n📊 DIFFERENCE ANALYSIS:")
            print(f"   Return difference: {return_diff:.2f} percentage points")
            print(f"   Trade count difference: {trade_diff}")
            print(f"   Value difference: ${value_diff:,.2f}")
            
            if return_diff > 5.0:
                print(f"   🚨 SIGNIFICANT RETURN DIFFERENCE!")
                print_debug_info(data, strategy, engine, entries, exits)
                return True  # Significant difference
            elif trade_diff > 2:
                print(f"   🚨 SIGNIFICANT TRADE COUNT DIFFERENCE!")
                print_debug_info(data, strategy, engine, entries, exits)
                return True
            else:
                print(f"   ✅ Differences within acceptable range")
                return False
        else:
            print(f"   ℹ️ No signals to compare")
            return False
            
    except ImportError:
        print(f"   ⚠️ VectorBT not available for comparison")
        return False
    except Exception as e:
        print(f"   ❌ VectorBT error: {e}")
        return False

def print_debug_info(data, strategy, engine, vbt_entries, vbt_exits):
    """Print debugging information for significant differences"""
    print(f"\n🔍 DEBUG INFORMATION:")
    
    # Check signal timing
    print(f"   Signal timing comparison:")
    
    # Our framework signals by date
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
    
    vbt_buy_dates = vbt_entries[vbt_entries].index.tolist()
    vbt_sell_dates = vbt_exits[vbt_exits].index.tolist()
    
    print(f"   Our buy signals: {[d.strftime('%m-%d') for d in our_buy_dates[:5]]}")
    print(f"   VBT buy signals: {[d.strftime('%m-%d') for d in vbt_buy_dates[:5]]}")
    print(f"   Our sell signals: {[d.strftime('%m-%d') for d in our_sell_dates[:5]]}")
    print(f"   VBT sell signals: {[d.strftime('%m-%d') for d in vbt_sell_dates[:5]]}")
    
    # Check actual trades executed
    print(f"   \n   Our framework trades:")
    for i, trade in enumerate(engine.trades[:3]):
        print(f"     Trade {i+1}: {trade.quantity} shares at ${trade.entry_price:.2f}")
    
    print(f"   \n   Potential causes:")
    print(f"     1. Capital allocation differences")
    print(f"     2. Position size calculation differences") 
    print(f"     3. Order rejection in our framework")
    print(f"     4. Commission handling differences")

def main():
    """Main test function"""
    print("🍎 APPLE STOCK FRAMEWORK COMPARISON TEST")
    print("=" * 60)
    
    # Try to get real data first
    data, is_real = try_download_apple()
    
    if data is None:
        # Fallback to simulated data
        data, is_real = create_apple_like_data()
    
    # Test framework differences
    has_significant_diff = test_framework_differences(data, is_real)
    
    print(f"\n📋 TEST SUMMARY:")
    print(f"   Data source: {'Real Apple stock' if is_real else 'Apple-like simulation'}")
    print(f"   Data points: {len(data)}")
    print(f"   Significant differences: {'YES' if has_significant_diff else 'NO'}")
    
    if has_significant_diff:
        print(f"\n🎯 RECOMMENDATION:")
        print(f"   This test case reveals framework differences and should be")
        print(f"   added to the unit test suite for regression testing.")
    
    return data, is_real, has_significant_diff

if __name__ == "__main__":
    data, is_real, has_diff = main()
    
    if has_diff:
        print(f"\n🔧 Next step: Create unit test to catch this difference")
    else:
        print(f"\n✅ Frameworks are aligned - no action needed")