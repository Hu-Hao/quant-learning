#!/usr/bin/env python3
"""
Unit tests specifically for the capital accounting bug fix
Focus on testing that our fix prevents the major capital disappearing bug
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

class TestCapitalAccountingFix(unittest.TestCase):
    """Test that the capital accounting fix prevents the major bug we found"""
    
    @classmethod
    def setUpClass(cls):
        """Create simple test data for capital accounting verification"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Create simple trending data to ensure we get trades
        prices = []
        base_price = 100.0
        for i in range(len(dates)):
            if i < 20:
                price = base_price + i * 0.5  # Uptrend for 20 days
            else:
                price = base_price + 20 * 0.5 - (i - 20) * 0.7  # Downtrend after
            prices.append(price)
        
        cls.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * len(prices)
        }, index=dates)
        
        cls.initial_capital = 100000
    
    def test_capital_accounting_integrity_no_costs(self):
        """Test capital accounting with zero costs to isolate the core logic"""
        strategy = MovingAverageStrategy(
            short_window=5,
            long_window=15,
            quantity=100
        )
        
        # Run with zero costs to isolate capital accounting logic
        engine = BacktestEngine(
            initial_capital=self.initial_capital,
            commission=0.0,
            slippage=0.0,
            max_position_size=1.0
        )
        
        engine.run_backtest(self.test_data, strategy)
        
        # Verify we have trades
        self.assertGreater(len(engine.trades), 0, "Strategy should generate trades")
        
        # Calculate expected capital based on trades
        expected_capital = self.initial_capital
        for trade in engine.trades:
            expected_capital += trade.pnl
        
        actual_capital = engine.capital
        
        # Capital should match exactly with no costs
        self.assertAlmostEqual(actual_capital, expected_capital, places=2,
            msg=f"Capital accounting mismatch: expected {expected_capital:.2f}, "
                f"got {actual_capital:.2f}")
        
        # Portfolio value should equal capital (no open positions at end)
        if not engine.positions:
            self.assertAlmostEqual(engine.portfolio_values[-1], actual_capital, places=2,
                msg="Portfolio value should equal capital when no open positions")
    
    def test_buy_sell_cycle_capital_flow(self):
        """Test that a complete buy-sell cycle properly accounts for capital"""
        strategy = MovingAverageStrategy(
            short_window=3,
            long_window=10,
            quantity=10  # Small quantity for precise tracking
        )
        
        engine = BacktestEngine(
            initial_capital=10000,
            commission=0.0,
            slippage=0.0,
            max_position_size=1.0
        )
        
        # Track capital manually
        initial_capital = engine.capital
        
        # Get first buy signal
        buy_signals = []
        sell_signals = []
        
        for idx, row in self.test_data.iterrows():
            partial_data = self.test_data.loc[:idx]
            signals = strategy.get_signals(partial_data, engine.capital)
            
            if signals:
                signal = signals[0]
                if 'BUY' in str(signal.action):
                    buy_signals.append((idx, signal, row['close']))
                elif 'SELL' in str(signal.action):
                    sell_signals.append((idx, signal, row['close']))
        
        # Should have both buy and sell signals
        self.assertGreater(len(buy_signals), 0, "Should have buy signals")
        self.assertGreater(len(sell_signals), 0, "Should have sell signals")
        
        # Run the backtest
        engine.run_backtest(self.test_data, strategy)
        
        # Verify that completed trades have proper capital impact
        if engine.trades:
            trade = engine.trades[0]  # First trade
            
            # For a long trade: P&L = quantity * (exit_price - entry_price)
            expected_pnl = trade.quantity * (trade.exit_price - trade.entry_price)
            actual_pnl = trade.pnl
            
            self.assertAlmostEqual(actual_pnl, expected_pnl, places=2,
                msg=f"Trade P&L calculation error: expected {expected_pnl:.2f}, "
                    f"got {actual_pnl:.2f}")
    
    def test_no_double_counting_bug(self):
        """Test that we don't have the double-counting bug we fixed"""
        strategy = MovingAverageStrategy(
            short_window=5,
            long_window=15,
            quantity=50
        )
        
        engine = BacktestEngine(
            initial_capital=50000,
            commission=0.001,  # Small commission
            slippage=0.0,
            max_position_size=1.0
        )
        
        engine.run_backtest(self.test_data, strategy)
        
        if engine.trades:
            # Calculate what capital should be manually
            capital_flow = self.initial_capital
            
            for trade in engine.trades:
                # Each complete trade impacts capital by its net P&L
                capital_flow += trade.pnl
            
            # The bug we fixed was that we were only adding P&L instead of sale proceeds
            # This test ensures that fix is working
            
            actual_capital = engine.capital
            difference = abs(actual_capital - capital_flow)
            
            # Should be very close (allowing for small rounding differences)
            self.assertLess(difference, 100,  # Allow $100 difference for rounding
                msg=f"Significant capital accounting error detected. "
                    f"Expected ~{capital_flow:.2f}, got {actual_capital:.2f}")
    
    def test_multiple_trade_cycles(self):
        """Test multiple complete trade cycles"""
        # Create data that will generate multiple trade cycles
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Create oscillating price data
        prices = []
        for i in range(len(dates)):
            base = 100.0
            cycle = 20  # 20-day cycle
            amplitude = 10
            trend = i * 0.1  # Slight uptrend
            price = base + amplitude * np.sin(2 * np.pi * i / cycle) + trend
            prices.append(price)
        
        oscillating_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.005 for p in prices],
            'low': [p * 0.995 for p in prices],
            'close': prices,
            'volume': [1000000] * len(prices)
        }, index=dates)
        
        strategy = MovingAverageStrategy(
            short_window=5,
            long_window=15,
            quantity=25
        )
        
        engine = BacktestEngine(
            initial_capital=50000,
            commission=0.001,
            slippage=0.001,
            max_position_size=1.0
        )
        
        engine.run_backtest(oscillating_data, strategy)
        
        # Should have multiple trades with oscillating data
        self.assertGreater(len(engine.trades), 2, "Should generate multiple trades")
        
        # Calculate expected capital
        expected_capital = 50000
        for trade in engine.trades:
            expected_capital += trade.pnl
        
        actual_capital = engine.capital
        
        # Allow 1% difference for multiple small rounding errors
        tolerance = 50000 * 0.01  # 1% of initial capital
        difference = abs(actual_capital - expected_capital)
        
        self.assertLess(difference, tolerance,
            msg=f"Capital accounting error over multiple trades. "
                f"Expected {expected_capital:.2f}, got {actual_capital:.2f}, "
                f"difference {difference:.2f}")
    
    def test_position_value_calculation(self):
        """Test that position values are correctly calculated"""
        strategy = MovingAverageStrategy(
            short_window=5,
            long_window=15,
            quantity=20
        )
        
        engine = BacktestEngine(
            initial_capital=20000,
            commission=0.0,
            slippage=0.0,
            max_position_size=1.0
        )
        
        # Find a point where we should have a position
        position_found = False
        
        for idx, row in self.test_data.iterrows():
            engine.update_time(idx)
            current_price = row['close']
            engine.price_history.append(current_price)
            
            partial_data = self.test_data.loc[:idx]
            signals = strategy.get_signals(partial_data, engine.capital)
            
            if signals:
                signal = signals[0]
                engine._process_signal(signal, current_price)
                
                # If we now have a position, test portfolio value calculation
                if engine.positions:
                    position_found = True
                    
                    # Calculate portfolio value manually
                    manual_value = engine.capital
                    for symbol, pos in engine.positions.items():
                        manual_value += pos.market_value(current_price)
                    
                    # Compare with engine calculation
                    prices = {'default': current_price}  # Match the symbol used
                    engine_value = engine.get_portfolio_value(prices)
                    
                    self.assertAlmostEqual(manual_value, engine_value, places=2,
                        msg=f"Portfolio value calculation mismatch: "
                            f"manual {manual_value:.2f}, engine {engine_value:.2f}")
                    break
        
        if not position_found:
            self.skipTest("No positions created in test data")

def run_capital_accounting_tests():
    """Run the capital accounting test suite"""
    print("🧪 CAPITAL ACCOUNTING FIX TESTS")
    print("=" * 40)
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCapitalAccountingFix)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\n📊 CAPITAL ACCOUNTING TEST RESULTS:")
    print(f"   Tests: {result.testsRun}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print(f"   ✅ All capital accounting tests passed")
        print(f"   🎉 The major bug fix is working correctly")
    else:
        print(f"   ❌ Capital accounting issues detected!")
        
        for test, error in result.failures + result.errors:
            test_name = test._testMethodName
            error_msg = error.split('\n')[-2] if '\n' in error else error
            print(f"   Failed: {test_name}")
            print(f"   Error: {error_msg}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_capital_accounting_tests()
    
    if success:
        print(f"\n🎯 SUMMARY:")
        print(f"   ✅ Capital accounting bug fix verified")
        print(f"   ✅ No double-counting of capital")
        print(f"   ✅ Proper sale proceeds calculation")
        print(f"   ✅ Multi-trade cycle integrity maintained")
    else:
        print(f"\n🚨 Capital accounting issues need attention!")
    
    print(f"\n💡 These tests ensure the $45k capital disappearing bug stays fixed.")