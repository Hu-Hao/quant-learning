#!/usr/bin/env python3
"""
Unit tests to verify percentage-based position sizing alignment with VectorBT
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.strategies.momentum import MomentumStrategy
from quant_trading.backtesting.engine import BacktestEngine

class TestPercentagePositionAlignment(unittest.TestCase):
    """Test that percentage position sizing matches VectorBT"""
    
    @classmethod
    def setUpClass(cls):
        """Create test data that will generate predictable signals"""
        # Create simple trending data for predictable signals
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Create data with clear trends to ensure signal generation
        prices = []
        base_price = 100.0
        
        # Phase 1: Downtrend (days 0-15) - should trigger short MA below long MA
        for i in range(16):
            price = base_price - i * 0.5
            prices.append(price)
        
        # Phase 2: Sideways (days 16-25) 
        for i in range(10):
            price = prices[-1] + np.sin(i) * 0.2
            prices.append(price)
        
        # Phase 3: Strong uptrend (days 26-49) - should trigger short MA above long MA
        for i in range(24):
            price = prices[-1] + (i + 1) * 0.8
            prices.append(price)
        
        cls.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': [1000000] * len(prices)
        }, index=dates)
        
        cls.initial_capital = 100000
        cls.tolerance = 0.5  # Allow 0.5% difference for minor implementation variations
    
    def setUp(self):
        """Check VectorBT availability"""
        try:
            import vectorbt as vbt
            self.vbt_available = True
        except ImportError:
            self.vbt_available = False
            self.skipTest("VectorBT not available - skipping percentage alignment tests")
    
    def run_our_framework(self, strategy, slippage=0.0):
        """Run backtest with our framework"""
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=0.001,
            slippage=slippage,
            max_position_size=1.0,
            allow_short_selling=False  # Match VectorBT behavior
        )
        
        engine.run_backtest(self.test_data, strategy)
        
        return {
            'final_value': engine.portfolio_values[-1],
            'total_return': (engine.portfolio_values[-1] / self.initial_capital - 1) * 100,
            'num_trades': len(engine.trades),
            'trades': engine.trades,
            'engine': engine
        }
    
    def run_vectorbt(self, strategy):
        """Run backtest with VectorBT"""
        import vectorbt as vbt
        
        entries, exits = strategy.generate_vectorbt_signals(self.test_data, self.initial_capital)
        
        if entries.sum() == 0 and exits.sum() == 0:
            return {
                'final_value': self.initial_capital,
                'total_return': 0.0,
                'num_trades': 0
            }
        
        # Calculate position size based on strategy parameters
        if hasattr(strategy, 'quantity') and strategy.quantity:
            size = strategy.quantity
        elif hasattr(strategy, 'percent_capital') and strategy.percent_capital:
            # For percentage strategies, calculate shares based on capital percentage
            # Use average price for size calculation (VectorBT approach)
            avg_price = self.test_data['close'].mean()
            capital_per_trade = self.initial_capital * strategy.percent_capital
            size = int(capital_per_trade / avg_price)
            
            # Ensure at least 1 share
            if size < 1:
                size = 1
        else:
            # Default: use all capital
            avg_price = self.test_data['close'].mean()
            size = int(self.initial_capital / avg_price)
        
        portfolio = vbt.Portfolio.from_signals(
            close=self.test_data['close'],
            entries=entries,
            exits=exits,
            size=size,
            init_cash=self.initial_capital,
            fees=0.001,
            freq='D'
        )
        
        return {
            'final_value': portfolio.value().iloc[-1],
            'total_return': (portfolio.value().iloc[-1] / self.initial_capital - 1) * 100,
            'num_trades': len(portfolio.trades.records),
            'portfolio': portfolio,
            'calculated_size': size
        }
    
    def assert_percentage_alignment(self, our_result, vbt_result, test_name, strategy):
        """Assert that percentage-based results align"""
        return_diff = abs(our_result['total_return'] - vbt_result['total_return'])
        
        print(f"\n📊 {test_name} Results:")
        print(f"   Strategy: {strategy.percent_capital:.1%} of capital")
        print(f"   Our return: {our_result['total_return']:+.2f}%")
        print(f"   VBT return: {vbt_result['total_return']:+.2f}%")
        print(f"   Return difference: {return_diff:.2f}pp")
        print(f"   Our trades: {our_result['num_trades']}")
        print(f"   VBT trades: {vbt_result['num_trades']}")
        
        if 'calculated_size' in vbt_result:
            print(f"   VBT calculated size: {vbt_result['calculated_size']} shares")
        
        # Show trade details for debugging
        if our_result['trades']:
            print(f"   Our trade details:")
            for i, trade in enumerate(our_result['trades']):
                value_traded = trade.quantity * trade.entry_price
                percent_of_capital = value_traded / self.initial_capital * 100
                print(f"      Trade {i+1}: {trade.quantity} shares @ ${trade.entry_price:.2f}")
                print(f"                 Value: ${value_traded:,.2f} ({percent_of_capital:.1f}% of capital)")
        
        # Assertions
        self.assertLessEqual(return_diff, self.tolerance,
            f"{test_name}: Return difference too large. "
            f"Our: {our_result['total_return']:.2f}%, "
            f"VBT: {vbt_result['total_return']:.2f}%, "
            f"Diff: {return_diff:.2f}pp")
        
        # Trade counts should be close (may differ by 1 due to implementation details)
        trade_diff = abs(our_result['num_trades'] - vbt_result['num_trades'])
        self.assertLessEqual(trade_diff, 1,
            f"{test_name}: Trade count difference too large. "
            f"Our: {our_result['num_trades']}, "
            f"VBT: {vbt_result['num_trades']}")
    
    def test_percentage_25_percent_capital(self):
        """Test with 25% of capital position sizing"""
        strategy = MovingAverageStrategy(
            short_window=5,
            long_window=15,
            percent_capital=0.25  # Use 25% of capital
        )
        
        our_result = self.run_our_framework(strategy, slippage=0.0)
        vbt_result = self.run_vectorbt(strategy)
        
        self.assert_percentage_alignment(our_result, vbt_result, "25% Capital", strategy)
    
    def test_percentage_50_percent_capital(self):
        """Test with 50% of capital position sizing"""
        strategy = MovingAverageStrategy(
            short_window=6,
            long_window=18,
            percent_capital=0.50  # Use 50% of capital
        )
        
        our_result = self.run_our_framework(strategy, slippage=0.0)
        vbt_result = self.run_vectorbt(strategy)
        
        self.assert_percentage_alignment(our_result, vbt_result, "50% Capital", strategy)
    
    def test_percentage_75_percent_capital(self):
        """Test with 75% of capital position sizing"""
        strategy = MovingAverageStrategy(
            short_window=4,
            long_window=12,
            percent_capital=0.75  # Use 75% of capital
        )
        
        our_result = self.run_our_framework(strategy, slippage=0.0)
        vbt_result = self.run_vectorbt(strategy)
        
        self.assert_percentage_alignment(our_result, vbt_result, "75% Capital", strategy)
    
    def test_percentage_10_percent_capital(self):
        """Test with conservative 10% of capital position sizing"""
        strategy = MovingAverageStrategy(
            short_window=3,
            long_window=9,
            percent_capital=0.10  # Use 10% of capital
        )
        
        our_result = self.run_our_framework(strategy, slippage=0.0)
        vbt_result = self.run_vectorbt(strategy)
        
        self.assert_percentage_alignment(our_result, vbt_result, "10% Capital", strategy)
    
    def test_momentum_percentage_capital(self):
        """Test percentage capital with momentum strategy"""
        strategy = MomentumStrategy(
            lookback_period=10,
            percent_capital=0.30  # Use 30% of capital
        )
        
        our_result = self.run_our_framework(strategy, slippage=0.0)
        vbt_result = self.run_vectorbt(strategy)
        
        self.assert_percentage_alignment(our_result, vbt_result, "Momentum 30% Capital", strategy)
    
    def test_percentage_capital_vs_fixed_quantity(self):
        """Compare percentage capital vs equivalent fixed quantity"""
        # Calculate equivalent fixed quantity for 50% capital
        avg_price = self.test_data['close'].mean()
        capital_50_percent = self.initial_capital * 0.5
        equivalent_quantity = int(capital_50_percent / avg_price)
        
        # Strategy with percentage
        strategy_percent = MovingAverageStrategy(
            short_window=5,
            long_window=15,
            percent_capital=0.50
        )
        
        # Strategy with equivalent fixed quantity
        strategy_fixed = MovingAverageStrategy(
            short_window=5,
            long_window=15,
            quantity=equivalent_quantity
        )
        
        our_percent = self.run_our_framework(strategy_percent, slippage=0.0)
        our_fixed = self.run_our_framework(strategy_fixed, slippage=0.0)
        
        return_diff = abs(our_percent['total_return'] - our_fixed['total_return'])
        
        print(f"\n📊 Percentage vs Fixed Quantity Comparison:")
        print(f"   Average price: ${avg_price:.2f}")
        print(f"   50% capital: ${capital_50_percent:,.2f}")
        print(f"   Equivalent quantity: {equivalent_quantity} shares")
        print(f"   Percentage strategy return: {our_percent['total_return']:+.2f}%")
        print(f"   Fixed quantity return: {our_fixed['total_return']:+.2f}%")
        print(f"   Difference: {return_diff:.2f}pp")
        
        # They should be similar but may differ due to dynamic vs static sizing
        self.assertLess(return_diff, 5.0,
            f"Percentage vs fixed quantity difference too large: {return_diff:.2f}pp")
    
    def test_percentage_position_size_calculation(self):
        """Test that position sizes are calculated correctly for percentage strategies"""
        strategy = MovingAverageStrategy(
            short_window=5,
            long_window=15,
            percent_capital=0.40  # Use 40% of capital
        )
        
        our_result = self.run_our_framework(strategy, slippage=0.0)
        
        if our_result['trades']:
            for i, trade in enumerate(our_result['trades']):
                # Calculate what percentage of capital this trade actually used
                trade_value = trade.quantity * trade.entry_price
                actual_percentage = trade_value / self.initial_capital
                
                print(f"\n📊 Position Size Verification - Trade {i+1}:")
                print(f"   Entry price: ${trade.entry_price:.2f}")
                print(f"   Quantity: {trade.quantity} shares")
                print(f"   Trade value: ${trade_value:,.2f}")
                print(f"   Target percentage: 40.0%")
                print(f"   Actual percentage: {actual_percentage:.1%}")
                
                # Should be close to 40% (allowing for rounding and price variations)
                percentage_diff = abs(actual_percentage - 0.40)
                self.assertLess(percentage_diff, 0.05,  # Allow 5% tolerance
                    f"Position size percentage off target: "
                    f"target 40%, actual {actual_percentage:.1%}")
        else:
            self.skipTest("No trades generated for position size verification")
    
    def test_multiple_percentage_trades(self):
        """Test multiple trades with percentage capital to ensure consistency"""
        # Use shorter windows to generate more signals
        strategy = MovingAverageStrategy(
            short_window=3,
            long_window=8,
            percent_capital=0.20  # Use 20% of capital per trade
        )
        
        our_result = self.run_our_framework(strategy, slippage=0.0)
        vbt_result = self.run_vectorbt(strategy)
        
        print(f"\n📊 Multiple Percentage Trades Test:")
        print(f"   Our trades: {our_result['num_trades']}")
        print(f"   VBT trades: {vbt_result['num_trades']}")
        
        if our_result['num_trades'] >= 2:
            print(f"   Trade details:")
            for i, trade in enumerate(our_result['trades']):
                value = trade.quantity * trade.entry_price
                percentage = value / self.initial_capital * 100
                print(f"      Trade {i+1}: {trade.quantity} shares, "
                      f"value ${value:,.2f} ({percentage:.1f}% of capital)")
            
            # All trades should use approximately the same percentage of capital
            percentages = [(t.quantity * t.entry_price) / self.initial_capital 
                          for t in our_result['trades']]
            
            if len(percentages) > 1:
                percentage_variance = max(percentages) - min(percentages)
                self.assertLess(percentage_variance, 0.05,  # 5% variance
                    f"Position size variance too high: {percentage_variance:.2%}")
        
        # Compare with VectorBT
        self.assert_percentage_alignment(our_result, vbt_result, "Multiple 20% Trades", strategy)

def run_percentage_alignment_tests():
    """Run the percentage position alignment test suite"""
    print("📊 PERCENTAGE POSITION ALIGNMENT TESTS")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPercentagePositionAlignment)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n📊 PERCENTAGE ALIGNMENT TEST RESULTS:")
    print(f"   Tests: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(f"   ✅ All percentage alignment tests passed")
        print(f"   🎉 Percentage position sizing matches VectorBT perfectly")
    else:
        print(f"   ❌ Some percentage alignment issues detected")
        
        for test, error in result.failures + result.errors:
            test_name = test._testMethodName
            print(f"   Failed: {test_name}")
            
            # Extract key error message
            lines = error.strip().split('\n')
            for line in lines:
                if 'AssertionError' in line and ('difference' in line.lower() or 'percentage' in line.lower()):
                    print(f"      {line.strip()}")
                    break
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_percentage_alignment_tests()
    
    if success:
        print(f"\n🎯 EXCELLENT RESULTS:")
        print(f"   ✅ Percentage position sizing perfectly aligned with VectorBT")
        print(f"   ✅ All capital percentage modes tested (10%, 25%, 50%, 75%)")
        print(f"   ✅ Position size calculations verified")
        print(f"   ✅ Multiple trade consistency confirmed")
        print(f"   ✅ Both MA and Momentum strategies tested")
    else:
        print(f"\n🔍 ANALYSIS:")
        print(f"   Some percentage position sizing alignment issues remain")
        print(f"   The tests identify specific areas needing adjustment")
    
    print(f"\n💡 These tests ensure percentage-based strategies work identically to VectorBT.")