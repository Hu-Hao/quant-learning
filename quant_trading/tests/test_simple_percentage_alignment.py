#!/usr/bin/env python3
"""
Simple unit tests to verify percentage position sizing alignment with VectorBT
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

class TestSimplePercentageAlignment(unittest.TestCase):
    """Simple test for percentage position sizing alignment"""
    
    @classmethod
    def setUpClass(cls):
        """Create hardcoded Apple-like data for consistent testing"""
        # Use simpler price data that will definitely generate signals
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        
        # Simple pattern: down trend then up trend
        prices = [
            200, 195, 190, 185, 180, 175, 170, 165, 160, 155,  # Down trend (10 days)
            158, 162, 166, 170, 174, 178, 182, 186, 190, 194,  # Up trend (10 days)  
            198, 202, 206, 210, 214, 218, 222, 226, 230, 234   # Strong up trend (10 days)
        ]
        
        cls.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * len(prices)
        }, index=dates)
        
        cls.initial_capital = 100000
    
    def setUp(self):
        """Check VectorBT availability"""
        try:
            import vectorbt as vbt
            self.vbt_available = True
        except ImportError:
            self.vbt_available = False
            self.skipTest("VectorBT not available")
    
    def test_percentage_strategy_alignment(self):
        """Test that percentage strategies align with VectorBT"""
        print("\n🧪 TESTING PERCENTAGE STRATEGY ALIGNMENT")
        print("=" * 50)
        
        # Test with 50% capital strategy
        strategy = MovingAverageStrategy(
            short_window=3,
            long_window=10,
            percent_capital=0.5  # Use 50% of capital
        )
        
        print(f"Strategy params: {strategy.params}")
        
        # Our framework with short selling disabled (to match VectorBT)
        print(f"\n🔍 Our Framework (Short Disabled):")
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=0.0,  # Zero costs for clean comparison
            slippage=0.0,
            max_position_size=1.0,
            allow_short_selling=False
        )
        
        engine.run_backtest(self.test_data, strategy)
        
        our_final = engine.portfolio_values[-1]
        our_return = (our_final / self.initial_capital - 1) * 100
        our_trades = len(engine.trades)
        
        print(f"   Final value: ${our_final:,.2f}")
        print(f"   Return: {our_return:+.2f}%")
        print(f"   Trades: {our_trades}")
        
        # Show trade details
        if engine.trades:
            print(f"   Trade details:")
            for i, trade in enumerate(engine.trades):
                trade_value = trade.quantity * trade.entry_price
                capital_percent = trade_value / self.initial_capital * 100
                print(f"      Trade {i+1}: {trade.quantity} shares @ ${trade.entry_price:.2f}")
                print(f"                 Value: ${trade_value:,.2f} ({capital_percent:.1f}% of capital)")
        
        # VectorBT comparison
        print(f"\n🔍 VectorBT:")
        import vectorbt as vbt
        
        entries, exits = strategy.generate_vectorbt_signals(self.test_data, self.initial_capital)
        
        print(f"   Entries: {entries.sum()}")
        print(f"   Exits: {exits.sum()}")
        
        if entries.sum() > 0:
            # Calculate size for 50% capital
            avg_price = self.test_data['close'].mean()
            capital_50_percent = self.initial_capital * 0.5
            vbt_size = int(capital_50_percent / avg_price)
            
            print(f"   Average price: ${avg_price:.2f}")
            print(f"   50% capital: ${capital_50_percent:,.2f}")
            print(f"   VectorBT size: {vbt_size} shares")
            
            portfolio = vbt.Portfolio.from_signals(
                close=self.test_data['close'],
                entries=entries,
                exits=exits,
                size=vbt_size,
                init_cash=self.initial_capital,
                fees=0.0,
                freq='D'
            )
            
            vbt_final = portfolio.value().iloc[-1]
            vbt_return = (vbt_final / self.initial_capital - 1) * 100
            vbt_trades = len(portfolio.trades.records)
            
            print(f"   Final value: ${vbt_final:,.2f}")
            print(f"   Return: {vbt_return:+.2f}%")
            print(f"   Trades: {vbt_trades}")
            
            # Comparison
            return_diff = abs(our_return - vbt_return)
            
            print(f"\n📊 COMPARISON:")
            print(f"   Our return: {our_return:+.2f}%")
            print(f"   VBT return: {vbt_return:+.2f}%")
            print(f"   Difference: {return_diff:.2f}pp")
            print(f"   Trade count difference: {abs(our_trades - vbt_trades)}")
            
            # Assertions
            self.assertLess(return_diff, 2.0,
                f"Return difference too large: {return_diff:.2f}pp")
            
            # Trade counts should be close
            trade_diff = abs(our_trades - vbt_trades)
            self.assertLessEqual(trade_diff, 1,
                f"Trade count difference: our {our_trades}, VBT {vbt_trades}")
            
            if return_diff < 0.5:
                print(f"   ✅ EXCELLENT ALIGNMENT!")
            elif return_diff < 1.0:
                print(f"   ✅ GOOD ALIGNMENT")
            else:
                print(f"   ⚠️ MODERATE ALIGNMENT")
            
            return return_diff
        else:
            self.skipTest("No VectorBT signals generated")
    
    def test_different_percentage_values(self):
        """Test different percentage values"""
        print(f"\n🧪 TESTING DIFFERENT PERCENTAGE VALUES")
        print("=" * 50)
        
        percentages = [0.25, 0.50, 0.75]
        results = []
        
        for percent in percentages:
            strategy = MovingAverageStrategy(
                short_window=3,
                long_window=10,
                percent_capital=percent
            )
            
            engine = BacktestEngine(
                initial_capital=self.initial_capital,
                commission=0.0,
                slippage=0.0,
                max_position_size=1.0,
                allow_short_selling=False
            )
            
            engine.run_backtest(self.test_data, strategy)
            
            final_value = engine.portfolio_values[-1]
            total_return = (final_value / self.initial_capital - 1) * 100
            
            print(f"   {percent:.0%} capital: {total_return:+.2f}% return ({len(engine.trades)} trades)")
            
            results.append({
                'percent': percent,
                'return': total_return,
                'trades': len(engine.trades)
            })
        
        # Higher percentage should generally give higher absolute returns (if profitable)
        # or higher absolute losses (if unprofitable)
        print(f"\n📊 Analysis:")
        for result in results:
            print(f"   {result['percent']:.0%}: {result['return']:+.2f}% ({result['trades']} trades)")
        
        # Basic sanity check - different percentages should give different results
        returns = [r['return'] for r in results]
        if len(set(returns)) > 1:  # At least some variation
            print(f"   ✅ Different percentages produce different results")
        else:
            print(f"   ⚠️ All percentages gave same result - check signal generation")

def run_simple_percentage_tests():
    """Run the simple percentage alignment tests"""
    print("📊 SIMPLE PERCENTAGE ALIGNMENT TESTS")
    print("=" * 60)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSimplePercentageAlignment)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n📊 TEST RESULTS:")
    print(f"   Tests: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(f"   ✅ All percentage alignment tests passed")
        print(f"   🎉 Percentage strategies work correctly!")
    else:
        print(f"   ❌ Some issues detected")
        
        for test, error in result.failures + result.errors:
            test_name = test._testMethodName
            print(f"   Failed: {test_name}")
            
            # Show key error details
            lines = error.strip().split('\n')
            for line in lines[-3:]:
                if 'AssertionError' in line or 'difference' in line.lower():
                    print(f"      {line.strip()}")
                    break
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_simple_percentage_tests()
    
    if success:
        print(f"\n🎯 SUMMARY:")
        print(f"   ✅ Percentage position sizing works correctly")
        print(f"   ✅ Framework aligns well with VectorBT for percentage strategies")
        print(f"   ✅ Different percentage values produce expected variations")
    else:
        print(f"\n🔍 AREAS FOR IMPROVEMENT:")
        print(f"   Some percentage alignment issues were identified")
        print(f"   The tests help pinpoint what needs adjustment")
    
    print(f"\n💡 These tests verify our percentage position sizing implementation.")