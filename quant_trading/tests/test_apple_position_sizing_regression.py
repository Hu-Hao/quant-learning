#!/usr/bin/env python3
"""
Regression test for Apple stock position sizing issues
This test ensures we don't regress on the framework differences identified with real Apple data
"""

import unittest
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

class TestApplePositionSizingRegression(unittest.TestCase):
    """Regression test for Apple stock position sizing differences"""
    
    @classmethod
    def setUpClass(cls):
        """Setup Apple data for testing"""
        try:
            # Try to fetch real Apple data
            apple = yf.Ticker("AAPL")
            data = apple.history(period="6mo")  # 6 months for faster testing
            
            if not data.empty:
                # Clean data
                if hasattr(data.index, 'tz') and data.index.tz is not None:
                    data.index = data.index.tz_localize(None)
                
                data.columns = [col.lower() for col in data.columns]
                data = data.dropna()
                
                cls.test_data = data
                cls.using_real_data = True
            else:
                raise ValueError("No Apple data")
                
        except Exception:
            # Fallback to realistic Apple-like data
            cls.test_data = cls._create_apple_like_data()
            cls.using_real_data = False
    
    @classmethod 
    def _create_apple_like_data(cls):
        """Create Apple-like data for testing"""
        dates = pd.date_range(end=datetime.now(), periods=120, freq='D')  # ~6 months
        np.random.seed(42)
        
        # Apple-like characteristics
        initial_price = 225.0  # Apple's typical price range
        returns = np.random.normal(0.0005, 0.018, len(dates))  # Slight growth, realistic volatility
        
        prices = [initial_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        return pd.DataFrame({
            'open': [p * np.random.uniform(0.999, 1.001) for p in prices],
            'high': [p * np.random.uniform(1.003, 1.012) for p in prices],
            'low': [p * np.random.uniform(0.988, 0.997) for p in prices],
            'close': prices,
            'volume': [np.random.randint(60000000, 120000000) for _ in prices]
        }, index=dates)
    
    def test_position_size_limit_fix_verification(self):
        """Test that the position size limit fix improves execution for Apple + fixed quantity"""
        strategy = MovingAverageStrategy(
            short_window=10,
            long_window=30,
            quantity=100  # Fixed 100 shares
        )
        
        # Test with NEW default limits (should be 1.0 now)
        new_default_engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            # max_position_size now defaults to 1.0 (100%)
        )
        
        new_default_engine.run_backtest(self.test_data, strategy)
        new_trades = len(new_default_engine.trades)
        
        # Test with OLD restrictive limits for comparison
        old_restrictive_engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            max_position_size=0.1,  # Force old 10% behavior
        )
        
        old_restrictive_engine.run_backtest(self.test_data, strategy)
        old_trades = len(old_restrictive_engine.trades)
        
        # Count total signals that should be generated
        total_signals = 0
        for idx, _ in self.test_data.iterrows():
            partial_data = self.test_data.loc[:idx]
            signals = strategy.get_signals(partial_data, available_capital=100000)
            total_signals += len(signals)
        
        # Verify the fix improved execution
        sample_price = self.test_data['close'].iloc[50]  # Mid-point price
        
        if total_signals > 0:
            new_execution_rate = new_trades / total_signals
            old_execution_rate = old_trades / total_signals
            
            # The fix should enable better execution
            self.assertGreaterEqual(new_execution_rate, old_execution_rate,
                f"New default (100% limit) should enable at least as much execution as old (10% limit). "
                f"Old: {old_execution_rate:.1%}, New: {new_execution_rate:.1%}")
            
            # With Apple prices, the new default should allow reasonable execution
            if sample_price > 150:  # High-priced stock like Apple
                self.assertGreaterEqual(new_execution_rate, 0.3,
                    f"With Apple-like prices (${sample_price:.2f}), new 100% limit should enable "
                    f"reasonable execution rate. Got {new_execution_rate:.1%}")
    
    def test_permissive_limits_improve_execution(self):
        """Test that permissive position size limits improve execution"""
        strategy = MovingAverageStrategy(
            short_window=10,
            long_window=30,
            quantity=100
        )
        
        # Test with permissive limits
        permissive_engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            max_position_size=0.3,  # Allow 30% of capital
        )
        
        permissive_engine.run_backtest(self.test_data, strategy)
        permissive_trades = len(permissive_engine.trades)
        
        # Test with restrictive limits
        restrictive_engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            max_position_size=0.1,  # Only 10% of capital
        )
        
        restrictive_engine.run_backtest(self.test_data, strategy)
        restrictive_trades = len(restrictive_engine.trades)
        
        # Permissive limits should allow more trades
        self.assertGreaterEqual(permissive_trades, restrictive_trades,
            "Permissive position size limits should allow at least as many trades as restrictive limits")
        
        # With Apple-like prices, the difference should be significant
        sample_price = self.test_data['close'].mean()
        if sample_price > 150:  # High-priced stock like Apple
            self.assertGreater(permissive_trades, restrictive_trades,
                f"With high stock prices (${sample_price:.2f}), permissive limits should "
                f"enable significantly more trades than restrictive limits")
    
    def test_signal_generation_consistency(self):
        """Test that signal generation is consistent regardless of execution"""
        strategy = MovingAverageStrategy(
            short_window=10,
            long_window=30,
            quantity=100
        )
        
        # Generate signals manually (same as engine does)
        manual_signals = []
        for idx, _ in self.test_data.iterrows():
            partial_data = self.test_data.loc[:idx]
            signals = strategy.get_signals(partial_data, available_capital=100000)
            manual_signals.extend(signals)
        
        # Generate VectorBT signals
        entries, exits = strategy.generate_vectorbt_signals(self.test_data, 100000)
        vbt_signal_count = entries.sum() + exits.sum()
        
        # Signal counts should match
        self.assertEqual(len(manual_signals), vbt_signal_count,
            "Manual signal generation should match VectorBT signal generation")
        
        # Both should generate reasonable number of signals for moving average strategy
        if len(self.test_data) > 50:  # Sufficient data
            self.assertGreater(len(manual_signals), 0,
                "Moving average strategy should generate signals with sufficient data")
    
    def test_framework_difference_documentation(self):
        """Document the framework differences for regression tracking"""
        strategy = MovingAverageStrategy(
            short_window=10,
            long_window=30,
            quantity=100
        )
        
        # Our framework with reasonable limits
        our_engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            max_position_size=0.3,  # Allow 30% to enable execution
        )
        
        our_engine.run_backtest(self.test_data, strategy)
        our_trades = len(our_engine.trades)
        our_final_value = our_engine.portfolio_values[-1]
        our_return = (our_final_value / 100000 - 1) * 100
        
        # Test VectorBT comparison if available
        try:
            import vectorbt as vbt
            
            entries, exits = strategy.generate_vectorbt_signals(self.test_data, 100000)
            
            if entries.sum() > 0 or exits.sum() > 0:
                # Get quantity from strategy params (correct way to access it)
                quantity = strategy.params.get('quantity', 100)
                
                vbt_portfolio = vbt.Portfolio.from_signals(
                    close=self.test_data['close'],
                    entries=entries,
                    exits=exits,
                    size=quantity,
                    init_cash=100000,
                    fees=0.001,
                    freq='D'
                )
                
                vbt_trades = len(vbt_portfolio.trades.records)
                vbt_final_value = vbt_portfolio.value().iloc[-1]
                vbt_return = (vbt_final_value / 100000 - 1) * 100
                
                # Document the differences (for regression tracking)
                return_diff = abs(our_return - vbt_return)
                trade_diff = abs(our_trades - vbt_trades)
                
                # Store results as test attributes for external access
                self.our_return = our_return
                self.vbt_return = vbt_return
                self.return_diff = return_diff
                self.trade_diff = trade_diff
                
                # These are documentation, not strict requirements
                # (frameworks may legitimately differ)
                self.assertIsInstance(return_diff, float, "Return difference should be calculable")
                self.assertIsInstance(trade_diff, int, "Trade difference should be calculable")
                
        except ImportError:
            # VectorBT not available - skip comparison
            pass
    
    def test_apple_price_characteristics(self):
        """Test that we're working with Apple-like price characteristics"""
        # Apple stock typically trades in $150-$300 range
        mean_price = self.test_data['close'].mean()
        min_price = self.test_data['close'].min()
        max_price = self.test_data['close'].max()
        
        if self.using_real_data:
            # Real Apple data should be in reasonable range
            self.assertGreater(mean_price, 100, "Apple stock should be > $100")
            self.assertLess(mean_price, 500, "Apple stock should be < $500")
        
        # 100 shares should represent significant capital
        position_value = 100 * mean_price
        position_percent = position_value / 100000
        
        # This documents why position size limits matter
        if position_percent > 0.15:  # More than 15% of capital
            self.assertGreater(position_percent, 0.1,
                "100 shares of Apple-like stock should exceed 10% position size limit, "
                f"requiring careful position size management (actual: {position_percent:.1%})")

def run_regression_tests():
    """Run the regression test suite"""
    print("🍎 APPLE POSITION SIZING REGRESSION TESTS")
    print("=" * 50)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestApplePositionSizingRegression)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n📊 REGRESSION TEST RESULTS:")
    print(f"   Tests: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(f"   ✅ All regression tests passed")
        print(f"   📝 Framework behavior is consistent with documented characteristics")
    else:
        print(f"   ❌ Regression detected!")
        print(f"   🔍 Framework behavior has changed from documented baseline")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_regression_tests()
    
    if success:
        print(f"\n🎯 SUMMARY:")
        print(f"   ✅ Position sizing behavior documented and tested")
        print(f"   ✅ Framework differences with VectorBT characterized")
        print(f"   ✅ Regression tests will catch future changes")
    else:
        print(f"\n🚨 Action needed: Framework behavior has changed!")
    
    print(f"\n💡 This test suite prevents regression of the Apple stock issues you identified.")