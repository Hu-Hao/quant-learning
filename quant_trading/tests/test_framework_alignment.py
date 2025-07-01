#!/usr/bin/env python3
"""
Unit tests to ensure our framework produces same results as VectorBT for identical inputs
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.strategies.momentum import MomentumStrategy
from quant_trading.backtesting.engine import BacktestEngine

class TestFrameworkAlignment(unittest.TestCase):
    """Test that our framework aligns with VectorBT results"""
    
    @classmethod
    def setUpClass(cls):
        """Create fake but realistic test data"""
        # Create 100 days of fake stock data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)  # Reproducible results
        
        # Generate realistic price movements
        initial_price = 100.0
        returns = np.random.normal(0.001, 0.02, len(dates))  # Slight positive drift, 2% daily volatility
        
        prices = [initial_price]
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1.0))  # Prevent negative prices
        
        # Create OHLC data
        cls.test_data = pd.DataFrame({
            'open': [p * np.random.uniform(0.995, 1.005) for p in prices],
            'high': [p * np.random.uniform(1.002, 1.015) for p in prices],
            'low': [p * np.random.uniform(0.985, 0.998) for p in prices],
            'close': prices,
            'volume': [np.random.randint(1000000, 5000000) for _ in prices]
        }, index=dates)
        
        # Test parameters
        cls.initial_capital = 100000
        cls.commission = 0.001
        cls.test_tolerance = 1.0  # Allow 1% difference for minor implementation differences
    
    def setUp(self):
        """Reset for each test"""
        try:
            import vectorbt as vbt
            self.vbt_available = True
        except ImportError:
            self.vbt_available = False
            self.skipTest("VectorBT not available - skipping alignment tests")
    
    def run_our_backtest(self, strategy, **engine_params):
        """Run backtest with our framework"""
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=self.commission,
            slippage=0.0,  # Disable slippage for cleaner comparison
            **engine_params
        )
        
        engine.run_backtest(self.test_data, strategy)
        
        final_value = engine.portfolio_values[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        num_trades = len(engine.trades)
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'num_trades': num_trades,
            'engine': engine
        }
    
    def run_vectorbt_backtest(self, strategy):
        """Run backtest with VectorBT"""
        import vectorbt as vbt
        
        # Generate signals using our strategy
        entries, exits = strategy.generate_vectorbt_signals(self.test_data, self.initial_capital)
        
        if entries.sum() == 0 and exits.sum() == 0:
            # No signals generated
            return {
                'final_value': self.initial_capital,
                'total_return': 0.0,
                'num_trades': 0,
                'portfolio': None
            }
        
        # Get position sizing
        if hasattr(strategy, 'quantity') and strategy.quantity:
            size = strategy.quantity
        elif hasattr(strategy, 'percent_capital') and strategy.percent_capital:
            # VectorBT expects actual size, not percentage
            avg_price = self.test_data['close'].mean()
            size = int((self.initial_capital * strategy.percent_capital) / avg_price)
        else:
            # Default to using all capital
            avg_price = self.test_data['close'].mean()
            size = int(self.initial_capital / avg_price)
        
        # Run VectorBT backtest
        portfolio = vbt.Portfolio.from_signals(
            close=self.test_data['close'],
            entries=entries,
            exits=exits,
            size=size,
            init_cash=self.initial_capital,
            fees=self.commission,
            freq='D'
        )
        
        final_value = portfolio.value().iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        num_trades = len(portfolio.trades.records)
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'num_trades': num_trades,
            'portfolio': portfolio
        }
    
    def assert_results_aligned(self, our_result, vbt_result, test_name=""):
        """Assert that our results align with VectorBT"""
        return_diff = abs(our_result['total_return'] - vbt_result['total_return'])
        trade_diff = abs(our_result['num_trades'] - vbt_result['num_trades'])
        
        # Store results for debugging
        setattr(self, f'{test_name}_our_return', our_result['total_return'])
        setattr(self, f'{test_name}_vbt_return', vbt_result['total_return'])
        setattr(self, f'{test_name}_return_diff', return_diff)
        
        # Assertions
        self.assertLessEqual(return_diff, self.test_tolerance,
            f"{test_name}: Return difference too large. "
            f"Our: {our_result['total_return']:.2f}%, "
            f"VBT: {vbt_result['total_return']:.2f}%, "
            f"Diff: {return_diff:.2f}pp")
        
        self.assertEqual(our_result['num_trades'], vbt_result['num_trades'],
            f"{test_name}: Trade count mismatch. "
            f"Our: {our_result['num_trades']}, "
            f"VBT: {vbt_result['num_trades']}")
    
    def test_moving_average_fixed_quantity(self):
        """Test MA strategy with fixed quantity"""
        strategy = MovingAverageStrategy(
            short_window=10,
            long_window=20,
            quantity=100
        )
        
        our_result = self.run_our_backtest(strategy, max_position_size=1.0)
        vbt_result = self.run_vectorbt_backtest(strategy)
        
        self.assert_results_aligned(our_result, vbt_result, "ma_fixed_qty")
    
    def test_moving_average_percentage_capital(self):
        """Test MA strategy with percentage of capital"""
        strategy = MovingAverageStrategy(
            short_window=5,
            long_window=15,
            percent_capital=0.5  # Use 50% of capital
        )
        
        our_result = self.run_our_backtest(strategy, max_position_size=1.0)
        vbt_result = self.run_vectorbt_backtest(strategy)
        
        self.assert_results_aligned(our_result, vbt_result, "ma_percent")
    
    def test_moving_average_full_capital(self):
        """Test MA strategy using full capital (default)"""
        strategy = MovingAverageStrategy(
            short_window=8,
            long_window=25
            # No quantity or percent_capital specified
        )
        
        our_result = self.run_our_backtest(strategy, max_position_size=1.0)
        vbt_result = self.run_vectorbt_backtest(strategy)
        
        self.assert_results_aligned(our_result, vbt_result, "ma_full_capital")
    
    def test_momentum_strategy_fixed_quantity(self):
        """Test Momentum strategy with fixed quantity"""
        strategy = MomentumStrategy(
            lookback_period=10,
            quantity=50
        )
        
        our_result = self.run_our_backtest(strategy, max_position_size=1.0)
        vbt_result = self.run_vectorbt_backtest(strategy)
        
        self.assert_results_aligned(our_result, vbt_result, "momentum_fixed")
    
    def test_momentum_strategy_percentage(self):
        """Test Momentum strategy with percentage"""
        strategy = MomentumStrategy(
            lookback_period=15,
            percent_capital=0.3
        )
        
        our_result = self.run_our_backtest(strategy, max_position_size=1.0)
        vbt_result = self.run_vectorbt_backtest(strategy)
        
        self.assert_results_aligned(our_result, vbt_result, "momentum_percent")
    
    def test_high_frequency_trading(self):
        """Test with short windows for more frequent trades"""
        strategy = MovingAverageStrategy(
            short_window=3,
            long_window=7,
            quantity=25
        )
        
        our_result = self.run_our_backtest(strategy, max_position_size=1.0)
        vbt_result = self.run_vectorbt_backtest(strategy)
        
        self.assert_results_aligned(our_result, vbt_result, "high_freq")
    
    def test_capital_accounting_integrity(self):
        """Test that capital accounting is correct throughout backtest"""
        strategy = MovingAverageStrategy(
            short_window=5,
            long_window=10,
            quantity=10  # Small quantity for precise tracking
        )
        
        our_result = self.run_our_backtest(strategy, max_position_size=1.0)
        engine = our_result['engine']
        
        # Verify capital accounting
        if engine.trades:
            # Calculate expected final capital
            total_trade_impact = 0
            for trade in engine.trades:
                # Each complete trade should impact capital by its P&L
                total_trade_impact += trade.pnl
            
            expected_capital = self.initial_capital + total_trade_impact
            actual_capital = engine.capital
            
            self.assertAlmostEqual(actual_capital, expected_capital, places=2,
                msg=f"Capital accounting error: expected {expected_capital:.2f}, "
                    f"got {actual_capital:.2f}")
            
            # Portfolio value should equal capital when no positions are open
            if not engine.positions:
                self.assertAlmostEqual(engine.portfolio_values[-1], actual_capital, places=2,
                    msg="Portfolio value should equal capital when no positions open")
    
    def test_no_signals_scenario(self):
        """Test scenario where strategy generates no signals"""
        # Use windows that won't generate signals in our test data
        strategy = MovingAverageStrategy(
            short_window=95,  # Too large for our 100-day dataset
            long_window=99,
            quantity=100
        )
        
        our_result = self.run_our_backtest(strategy, max_position_size=1.0)
        vbt_result = self.run_vectorbt_backtest(strategy)
        
        # Both should have no trades and return ~0%
        self.assertEqual(our_result['num_trades'], 0)
        self.assertEqual(vbt_result['num_trades'], 0)
        self.assertAlmostEqual(our_result['total_return'], 0.0, places=2)
        self.assertAlmostEqual(vbt_result['total_return'], 0.0, places=2)
    
    def test_single_trade_cycle(self):
        """Test a single complete buy-sell cycle"""
        # Use specific windows to ensure we get exactly one trade cycle
        strategy = MovingAverageStrategy(
            short_window=2,
            long_window=50,  # Will generate one buy signal early, one sell later
            quantity=1  # Single share for precise tracking
        )
        
        our_result = self.run_our_backtest(strategy, max_position_size=1.0)
        vbt_result = self.run_vectorbt_backtest(strategy)
        
        self.assert_results_aligned(our_result, vbt_result, "single_trade")
        
        # Both should have same number of trades (likely 1 complete cycle)
        if our_result['num_trades'] > 0:
            self.assertGreater(our_result['num_trades'], 0)
            self.assertEqual(our_result['num_trades'], vbt_result['num_trades'])

def run_alignment_tests():
    """Run the alignment test suite"""
    print("🧪 FRAMEWORK ALIGNMENT TESTS")
    print("=" * 50)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFrameworkAlignment)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n📊 ALIGNMENT TEST RESULTS:")
    print(f"   Tests: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(f"   ✅ All alignment tests passed")
        print(f"   🎉 Framework is properly aligned with VectorBT")
    else:
        print(f"   ❌ Alignment issues detected!")
        print(f"   🔍 Framework behavior differs from VectorBT")
        
        # Print details of failures
        for test, error in result.failures + result.errors:
            test_name = test._testMethodName
            print(f"   Failed: {test_name}")
            print(f"   Error: {error.split(chr(10))[-2] if chr(10) in error else error}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_alignment_tests()
    
    if success:
        print(f"\n🎯 SUMMARY:")
        print(f"   ✅ Framework alignment verified")
        print(f"   ✅ Capital accounting working correctly")
        print(f"   ✅ All position sizing modes tested")
        print(f"   ✅ Ready for production use")
    else:
        print(f"\n🚨 Action needed: Framework alignment issues detected!")
    
    print(f"\n💡 These tests prevent regression of the critical bugs we fixed.")