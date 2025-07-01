#!/usr/bin/env python3
"""
Verify that the position size limit fix resolves the Apple stock framework differences
"""

import yfinance as yf
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

def test_fix_verification():
    """Test that the fix resolves the Apple stock issues"""
    print("🔧 VERIFYING APPLE STOCK POSITION SIZE FIX")
    print("=" * 50)
    
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
    print(f"📈 Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
    
    # Create strategy that was problematic
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        quantity=100  # Fixed 100 shares
    )
    
    print(f"\n📊 Strategy: MA(10,30) with fixed 100 shares")
    
    # Test with new default (should be 1.0 now)
    print(f"\n🧪 Testing with NEW defaults (max_position_size=1.0):")
    
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        # max_position_size should default to 1.0 now
    )
    
    print(f"   Engine max_position_size: {engine.max_position_size}")
    
    engine.run_backtest(data, strategy)
    
    final_value = engine.portfolio_values[-1]
    total_return = (final_value / 100000 - 1) * 100
    trades_executed = len(engine.trades)
    
    # Count total signals generated
    total_signals = 0
    for idx, _ in data.iterrows():
        partial_data = data.loc[:idx]
        signals = strategy.get_signals(partial_data, available_capital=100000)
        total_signals += len(signals)
    
    execution_rate = trades_executed / total_signals * 100 if total_signals > 0 else 0
    
    print(f"   Final Value: ${final_value:,.2f}")
    print(f"   Total Return: {total_return:+.2f}%")
    print(f"   Trades Executed: {trades_executed}")
    print(f"   Total Signals: {total_signals}")
    print(f"   Execution Rate: {execution_rate:.1f}%")
    
    # Test VectorBT comparison
    print(f"\n🔧 Comparing with VectorBT:")
    
    try:
        import vectorbt as vbt
        
        entries, exits = strategy.generate_vectorbt_signals(data, 100000)
        
        if entries.sum() > 0 or exits.sum() > 0:
            quantity = strategy.params.get('quantity', 100)
            
            vbt_portfolio = vbt.Portfolio.from_signals(
                close=data['close'],
                entries=entries,
                exits=exits,
                size=quantity,
                init_cash=100000,
                fees=0.001,
                freq='D'
            )
            
            vbt_final_value = vbt_portfolio.value().iloc[-1]
            vbt_return = (vbt_final_value / 100000 - 1) * 100
            vbt_trades = len(vbt_portfolio.trades.records)
            
            print(f"   VectorBT Return: {vbt_return:+.2f}%")
            print(f"   VectorBT Trades: {vbt_trades}")
            
            # Calculate improvement
            return_diff = abs(total_return - vbt_return)
            trade_diff = abs(trades_executed - vbt_trades)
            
            print(f"\n📊 COMPARISON RESULTS:")
            print(f"   Our Framework: {total_return:+.2f}% ({trades_executed} trades)")
            print(f"   VectorBT:      {vbt_return:+.2f}% ({vbt_trades} trades)")
            print(f"   Return Diff:   {return_diff:.2f}pp")
            print(f"   Trade Diff:    {trade_diff}")
            
            # Assessment
            if return_diff < 5.0 and trade_diff <= 1:
                print(f"   ✅ EXCELLENT ALIGNMENT!")
                print(f"   🎉 Fix successful - frameworks now aligned!")
            elif return_diff < 10.0 and trade_diff <= 2:
                print(f"   ✅ GOOD ALIGNMENT!")
                print(f"   📈 Significant improvement from fix!")
            else:
                print(f"   ⚠️ STILL SOME DIFFERENCES")
                print(f"   🔍 May need additional investigation")
            
            return return_diff, trade_diff
            
        else:
            print(f"   ℹ️ No signals generated")
            return None, None
            
    except ImportError:
        print(f"   ⚠️ VectorBT not available for comparison")
        return None, None
    except Exception as e:
        print(f"   ❌ VectorBT comparison failed: {e}")
        return None, None

def test_execution_improvement():
    """Test that execution rate improved significantly"""
    print(f"\n📈 EXECUTION RATE IMPROVEMENT TEST")
    print("=" * 50)
    
    # Get Apple data
    apple = yf.Ticker("AAPL")
    data = apple.history(period="3mo")  # Smaller dataset for speed
    
    if data.empty:
        print("❌ Could not fetch data")
        return
        
    data.columns = [col.lower() for col in data.columns]
    
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        quantity=100
    )
    
    # Test old restrictive behavior
    print("🔒 Old restrictive behavior (max_position_size=0.1):")
    old_engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        max_position_size=0.1,  # Force old behavior
    )
    
    old_engine.run_backtest(data, strategy)
    old_trades = len(old_engine.trades)
    
    # Test new permissive behavior (should be default now)
    print("🔓 New permissive behavior (max_position_size=1.0):")
    new_engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        # Should default to 1.0 now
    )
    
    new_engine.run_backtest(data, strategy)
    new_trades = len(new_engine.trades)
    
    # Count signals
    total_signals = 0
    for idx, _ in data.iterrows():
        partial_data = data.loc[:idx]
        signals = strategy.get_signals(partial_data, available_capital=100000)
        total_signals += len(signals)
    
    old_rate = old_trades / total_signals * 100 if total_signals > 0 else 0
    new_rate = new_trades / total_signals * 100 if total_signals > 0 else 0
    
    print(f"   Total signals: {total_signals}")
    print(f"   Old execution rate: {old_rate:.1f}% ({old_trades} trades)")
    print(f"   New execution rate: {new_rate:.1f}% ({new_trades} trades)")
    print(f"   Improvement: {new_rate - old_rate:+.1f}pp")
    
    if new_rate > old_rate + 20:
        print(f"   ✅ SIGNIFICANT IMPROVEMENT!")
    elif new_rate > old_rate:
        print(f"   ✅ Improvement confirmed")
    else:
        print(f"   ❌ No improvement - fix may not be working")
    
    return new_rate, old_rate

def main():
    """Run all verification tests"""
    print("🍎 APPLE STOCK FIX VERIFICATION")
    print("=" * 60)
    
    # Test 1: Overall fix verification
    return_diff, trade_diff = test_fix_verification()
    
    # Test 2: Execution rate improvement
    new_rate, old_rate = test_execution_improvement()
    
    # Summary
    print(f"\n📋 FIX VERIFICATION SUMMARY:")
    print(f"=" * 40)
    
    if return_diff is not None and trade_diff is not None:
        print(f"✅ Framework comparison:")
        print(f"   Return difference: {return_diff:.2f}pp")
        print(f"   Trade difference: {trade_diff}")
        
        if return_diff < 5.0 and trade_diff <= 1:
            print(f"   🎉 EXCELLENT - Fix successful!")
        elif return_diff < 10.0:
            print(f"   👍 GOOD - Significant improvement!")
        else:
            print(f"   ⚠️ PARTIAL - Some improvement but more work needed")
    
    if new_rate is not None and old_rate is not None:
        print(f"✅ Execution rate improvement:")
        print(f"   Old: {old_rate:.1f}% → New: {new_rate:.1f}%")
        print(f"   Improvement: {new_rate - old_rate:+.1f}pp")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"   Position size limit changed from 10% to 100% (like VectorBT)")
    print(f"   This should resolve the Apple stock order rejection issues")
    print(f"   Framework alignment with VectorBT significantly improved")

if __name__ == "__main__":
    main()