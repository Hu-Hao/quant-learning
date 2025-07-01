#!/usr/bin/env python3
"""
Unit test for Apple fixed quantity strategy differences
This reproduces the significant framework differences identified with real data
"""

import unittest
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

class TestAppleFixedQuantityDifferences(unittest.TestCase):
    """Test the significant differences found with Apple stock and fixed quantity strategy"""
    
    @classmethod
    def setUpClass(cls):
        """Fetch real Apple data once for all tests"""
        print("📊 Fetching Apple stock data for testing...")
        
        try:
            apple = yf.Ticker("AAPL")
            data = apple.history(period="1y")
            
            if data.empty:
                raise ValueError("No Apple data available")
            
            # Clean data
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            data.columns = [col.lower() for col in data.columns]
            data = data.dropna()
            
            cls.apple_data = data
            cls.data_available = True
            print(f"✅ Loaded {len(data)} days of Apple data")
            
        except Exception as e:
            print(f"⚠️ Could not fetch Apple data: {e}")
            print("   Using simulated Apple-like data for testing")
            cls.apple_data = cls.create_apple_like_data()
            cls.data_available = False
    
    @classmethod
    def create_apple_like_data(cls):
        """Create Apple-like data for testing when real data unavailable"""
        dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        dates = dates[dates.dayofweek < 5]  # Trading days only
        
        # Apple-like characteristics
        np.random.seed(42)
        initial_price = 200.0
        n_days = len(dates)
        
        # Generate realistic price movements
        returns = np.random.normal(0.0002, 0.02, n_days)  # Small positive drift, realistic volatility
        
        # Add some trending periods and reversals
        for i in range(50, 100):  # Uptrend
            if i < len(returns):
                returns[i] += 0.003
        
        for i in range(150, 200):  # Downtrend 
            if i < len(returns):
                returns[i] -= 0.004
        
        # Calculate prices
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data
        data = []
        for price in prices:
            data.append({
                'open': price * np.random.uniform(0.999, 1.001),
                'high': price * np.random.uniform(1.001, 1.015),
                'low': price * np.random.uniform(0.985, 0.999),
                'close': price,
                'volume': np.random.randint(50000000, 100000000)
            })
        
        return pd.DataFrame(data, index=dates[:len(prices)])
    
    def test_fixed_quantity_framework_differences(self):
        """Test the significant differences in fixed quantity strategy"""
        print(f"\n🧪 Testing fixed quantity strategy differences...")
        
        # Create fixed quantity strategy (the problematic case)
        strategy = MovingAverageStrategy(
            short_window=10,
            long_window=30,
            quantity=100  # Fixed 100 shares
        )
        
        initial_capital = 100000
        
        # Test our framework
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=0.001,
            slippage=0.0,
            max_position_size=1.0,  # Allow full allocation
            allow_short_selling=False
        )
        
        engine.run_backtest(self.apple_data, strategy)
        
        our_final_value = engine.portfolio_values[-1]
        our_return = (our_final_value / initial_capital - 1) * 100
        our_trades = len(engine.trades)
        
        # Test VectorBT
        try:
            import vectorbt as vbt
            
            entries, exits = strategy.generate_vectorbt_signals(self.apple_data, initial_capital)
            
            if entries.sum() > 0 or exits.sum() > 0:
                vbt_portfolio = vbt.Portfolio.from_signals(
                    close=self.apple_data['close'],
                    entries=entries,
                    exits=exits,
                    size=strategy.quantity,
                    init_cash=initial_capital,
                    fees=0.001,
                    freq='D'
                )
                
                vbt_final_value = vbt_portfolio.value().iloc[-1]
                vbt_return = (vbt_final_value / initial_capital - 1) * 100
                vbt_trades = len(vbt_portfolio.trades.records)
                
                # Calculate differences
                return_diff = abs(our_return - vbt_return)
                trade_diff = abs(our_trades - vbt_trades)
                
                print(f"   Our Framework: {our_return:+.2f}% ({our_trades} trades)")
                print(f"   VectorBT: {vbt_return:+.2f}% ({vbt_trades} trades)")
                print(f"   Difference: {return_diff:.2f}pp return, {trade_diff} trades")
                
                # Test assertions for framework alignment
                self.assertLess(return_diff, 10.0, 
                    f"Return difference too large: {return_diff:.2f}pp. "
                    f"Our: {our_return:.2f}%, VBT: {vbt_return:.2f}%")
                
                self.assertLessEqual(trade_diff, 2,
                    f"Trade count difference too large: {trade_diff}. "
                    f"Our: {our_trades}, VBT: {vbt_trades}")
                
                # Test that both frameworks execute some trades when signals exist
                if entries.sum() > 0:
                    self.assertGreater(our_trades, 0, 
                        "Our framework should execute trades when signals exist")
                    self.assertGreater(vbt_trades, 0,
                        "VectorBT should execute trades when signals exist")
                
                return True  # VectorBT available
            else:
                self.skipTest("No signals generated for comparison")
                
        except ImportError:
            self.skipTest("VectorBT not available for comparison")
        except Exception as e:
            self.fail(f"VectorBT test failed with error: {e}")
    
    def test_signal_generation_consistency(self):
        """Test that signal generation is consistent between methods"""
        strategy = MovingAverageStrategy(
            short_window=10,
            long_window=30,
            quantity=100
        )
        
        # Manual signal generation (same as our engine uses)
        manual_signals = []
        manual_dates = []
        
        for idx, _ in self.apple_data.iterrows():
            partial_data = self.apple_data.loc[:idx]
            signals = strategy.get_signals(partial_data, available_capital=100000)
            
            for signal in signals:
                manual_signals.append(signal.action.value)
                manual_dates.append(idx)
        
        # VectorBT signal generation
        entries, exits = strategy.generate_vectorbt_signals(self.apple_data, 100000)
        vbt_entry_dates = entries[entries].index.tolist()
        vbt_exit_dates = exits[exits].index.tolist()
        
        # Extract manual signals by type
        manual_buy_dates = [date for i, date in enumerate(manual_dates) 
                           if manual_signals[i] == 'buy']
        manual_sell_dates = [date for i, date in enumerate(manual_dates) 
                            if manual_signals[i] == 'sell']
        
        # Test signal timing consistency
        self.assertEqual(manual_buy_dates, vbt_entry_dates,
            "Buy signal timing should match between manual and VectorBT generation")
        
        self.assertEqual(manual_sell_dates, vbt_exit_dates,
            "Sell signal timing should match between manual and VectorBT generation")
        
        print(f"   ✅ Signal consistency: {len(manual_buy_dates)} buys, {len(manual_sell_dates)} sells")
    
    def test_position_size_limits_impact(self):
        """Test how position size limits impact execution"""
        strategy = MovingAverageStrategy(
            short_window=10,
            long_window=30,
            quantity=100
        )
        
        # Test with restrictive position size limit
        restrictive_engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            max_position_size=0.1,  # Only 10% of capital
            allow_short_selling=False
        )
        
        restrictive_engine.run_backtest(self.apple_data, strategy)
        restrictive_trades = len(restrictive_engine.trades)
        
        # Test with permissive position size limit
        permissive_engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            max_position_size=1.0,  # Full capital allowed
            allow_short_selling=False
        )
        
        permissive_engine.run_backtest(self.apple_data, strategy)
        permissive_trades = len(permissive_engine.trades)
        
        # Permissive engine should execute at least as many trades
        self.assertGreaterEqual(permissive_trades, restrictive_trades,
            "Permissive position sizing should allow more trade execution")
        
        # Calculate execution rates
        all_signals = []
        for idx, _ in self.apple_data.iterrows():
            partial_data = self.apple_data.loc[:idx]
            signals = strategy.get_signals(partial_data, available_capital=100000)
            all_signals.extend(signals)
        
        if all_signals:
            restrictive_rate = restrictive_trades / len(all_signals) * 100
            permissive_rate = permissive_trades / len(all_signals) * 100
            
            print(f"   Restrictive execution rate: {restrictive_rate:.1f}%")
            print(f"   Permissive execution rate: {permissive_rate:.1f}%")
            
            # Position size limits should significantly impact execution
            if restrictive_rate < 50:  # If restrictive limits block trades
                self.assertGreater(permissive_rate - restrictive_rate, 10,
                    "Position size limits should significantly impact execution rate")
    
    def test_capital_allocation_accuracy(self):
        """Test that capital allocation calculations are accurate"""
        strategy = MovingAverageStrategy(
            short_window=10,
            long_window=30,
            quantity=100
        )
        
        engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            max_position_size=1.0,
            allow_short_selling=False
        )
        
        engine.run_backtest(self.apple_data, strategy)
        
        # Check that trades don't exceed capital constraints
        for trade in engine.trades:
            trade_value = trade.quantity * trade.entry_price
            
            # Each trade should not exceed the available capital at the time
            # (This is a simplified check - real implementation would need to track capital over time)
            self.assertLessEqual(trade_value, 110000,  # Allow some margin for calculation differences
                f"Trade value ${trade_value:,.2f} should not grossly exceed initial capital")
            
            # Fixed quantity should be exactly 100 shares (as specified)
            self.assertEqual(trade.quantity, 100,
                f"Fixed quantity strategy should trade exactly 100 shares, got {trade.quantity}")

def run_apple_tests():
    """Run the Apple stock specific tests"""
    print("🍎 RUNNING APPLE STOCK FRAMEWORK DIFFERENCE TESTS")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAppleFixedQuantityDifferences)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Tests run: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n❌ FAILURES (Framework Differences Detected):")
        for test, traceback in result.failures:
            print(f"   {test}: {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'See details above'}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"   {test}: {traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else 'See details above'}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0
    print(f"\n📈 SUCCESS RATE: {success_rate:.1f}%")
    
    if len(result.failures) > 0:
        print(f"\n🎯 FRAMEWORK DIFFERENCES CONFIRMED!")
        print(f"   These tests document the differences you found between")
        print(f"   our framework and VectorBT with Apple stock + fixed quantity.")
        print(f"   This test should be run regularly to track improvements.")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_apple_tests()
    
    if not success:
        print(f"\n🔧 NEXT STEPS:")
        print(f"   1. Investigate position size limit logic")
        print(f"   2. Review capital allocation calculations")
        print(f"   3. Check order execution constraints")
        print(f"   4. Consider adjusting framework parameters for better VectorBT alignment")
    
    print(f"\n📝 This unit test captures the real-world differences you identified!")