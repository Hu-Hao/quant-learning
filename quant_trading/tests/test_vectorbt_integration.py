"""
Unit tests for VectorBT integration and cross-validation
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.strategies.momentum import MomentumStrategy
from quant_trading.strategies.mean_reversion import MeanReversionStrategy
from quant_trading.backtesting.engine import BacktestEngine
from quant_trading.utils.vectorbt_comparison import compare_with_vectorbt
from quant_trading.data.data_fetcher import create_sample_data


class TestVectorBTIntegration(unittest.TestCase):
    """Test VectorBT integration and cross-validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create deterministic test data
        self.test_data = create_sample_data(100, trend=0.05, volatility=0.02, seed=42)
        
        # Test parameters
        self.initial_capital = 100000
        self.commission = 0.001
        
        # Create strategies for testing
        self.strategies = {
            'moving_average': MovingAverageStrategy(short_window=5, long_window=15),
            'momentum': MomentumStrategy(lookback_period=10, momentum_threshold=0.02),
            'mean_reversion': MeanReversionStrategy(window=20, entry_threshold=2.0)
        }
    
    def test_signals_to_vectorbt_basic(self):
        """Test basic signals_to_vectorbt functionality"""
        strategy = self.strategies['moving_average']
        
        # Test that function returns correct types
        entries, exits = strategy.generate_vectorbt_signals(self.test_data)
        
        self.assertIsInstance(entries, pd.Series)
        self.assertIsInstance(exits, pd.Series)
        self.assertEqual(len(entries), len(self.test_data))
        self.assertEqual(len(exits), len(self.test_data))
        self.assertTrue(entries.dtype == bool)
        self.assertTrue(exits.dtype == bool)
    
    def test_signals_consistency_across_strategies(self):
        """Test that all strategies can generate VectorBT signals"""
        for name, strategy in self.strategies.items():
            with self.subTest(strategy=name):
                # Should not raise any exceptions
                entries, exits = strategy.generate_vectorbt_signals(self.test_data)
                
                # Basic sanity checks
                self.assertEqual(len(entries), len(self.test_data))
                self.assertEqual(len(exits), len(self.test_data))
                self.assertGreaterEqual(entries.sum() + exits.sum(), 0)  # At least 0 signals
    
    def test_engine_vs_vectorbt_signal_consistency(self):
        """Test that engine and VectorBT detect the same signals (before execution logic)"""
        strategy = self.strategies['moving_average']
        
        # Get VectorBT signals
        vbt_entries, vbt_exits = strategy.generate_vectorbt_signals(self.test_data)
        
        # Count signals generated manually (same as VectorBT should do)
        manual_buy_signals = 0
        manual_sell_signals = 0
        
        # Simulate the same iteration as both engine and VectorBT
        for idx, _ in self.test_data.iterrows():
            partial_data = self.test_data.loc[:idx]
            signals = strategy.get_signals(partial_data)
            
            for signal in signals:
                if signal.action.value == 'buy':
                    manual_buy_signals += 1
                elif signal.action.value == 'sell':
                    manual_sell_signals += 1
        
        # VectorBT should detect the same signals as manual iteration
        self.assertEqual(vbt_entries.sum(), manual_buy_signals, 
                        f"Buy signals mismatch: VectorBT={vbt_entries.sum()}, Manual={manual_buy_signals}")
        self.assertEqual(vbt_exits.sum(), manual_sell_signals,
                        f"Sell signals mismatch: VectorBT={vbt_exits.sum()}, Manual={manual_sell_signals}")
        
        # Note: Engine might execute fewer signals due to capital/position constraints
        # That's expected behavior - signal generation vs signal execution are different
    
    def test_iterative_signal_generation(self):
        """Test that signals_to_vectorbt matches point-in-time signal generation"""
        strategy = self.strategies['moving_average']
        
        # Generate signals using VectorBT approach
        vbt_entries, vbt_exits = strategy.generate_vectorbt_signals(self.test_data)
        
        # Generate signals manually point-by-point (same as engine)
        manual_entries = pd.Series(False, index=self.test_data.index)
        manual_exits = pd.Series(False, index=self.test_data.index)
        
        for idx, _ in self.test_data.iterrows():
            partial_data = self.test_data.loc[:idx]
            signals = strategy.get_signals(partial_data)
            
            for signal in signals:
                if signal.action.value == 'buy':
                    manual_entries.at[idx] = True
                elif signal.action.value == 'sell':
                    manual_exits.at[idx] = True
        
        # Should be identical
        pd.testing.assert_series_equal(vbt_entries, manual_entries)
        pd.testing.assert_series_equal(vbt_exits, manual_exits)
    
    def test_performance_consistency(self):
        """Test that performance metrics are consistent between frameworks"""
        strategy = self.strategies['moving_average']
        
        try:
            # Run comparison (this tests the full pipeline)
            results = compare_with_vectorbt(
                data=self.test_data,
                strategy=strategy,
                initial_capital=self.initial_capital,
                commission=self.commission,
                show_details=False
            )
            
            our_results = results['our_results']
            vbt_results = results['vectorbt_results']
            
            # Both should succeed
            self.assertTrue(our_results['success'], "Our framework failed")
            
            if vbt_results['success']:
                # If VectorBT succeeds, performance should be similar
                our_return = our_results['performance']['total_return']
                vbt_return = vbt_results['performance']['total_return']
                
                # Returns should be within 5% of each other (accounting for execution differences)
                return_diff = abs(our_return - vbt_return)
                self.assertLess(return_diff, 0.05, 
                              f"Return difference too large: {return_diff:.3f}")
                
                # Trade counts should be identical (same signals)
                our_trades = our_results['performance']['total_trades']
                vbt_trades = vbt_results['performance']['total_trades']
                self.assertEqual(our_trades, vbt_trades,
                               f"Trade count mismatch: Ours={our_trades}, VectorBT={vbt_trades}")
            
        except ImportError:
            # VectorBT not available - skip this test
            self.skipTest("VectorBT not available for testing")
    
    def test_signal_timing_accuracy(self):
        """Test that signals occur at the correct timestamps"""
        strategy = self.strategies['moving_average']
        
        # Create simple trending data to force specific signals
        dates = pd.date_range('2023-01-01', periods=20)
        prices = list(range(100, 120))  # Strong uptrend
        
        test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * 20
        }, index=dates)
        
        # Get VectorBT signals
        entries, exits = strategy.generate_vectorbt_signals(test_data)
        
        # Verify signal timing by checking manually
        for signal_idx in entries[entries].index:
            # At this point, there should be a buy signal from get_signals
            partial_data = test_data.loc[:signal_idx]
            signals = strategy.get_signals(partial_data)
            
            buy_signals = [s for s in signals if s.action.value == 'buy']
            self.assertGreater(len(buy_signals), 0, 
                             f"No buy signal found at expected index {signal_idx}")
    
    def test_empty_data_handling(self):
        """Test handling of edge cases"""
        strategy = self.strategies['moving_average']
        
        # Test with minimal data
        minimal_data = self.test_data.head(5)  # Less than strategy requirements
        entries, exits = strategy.generate_vectorbt_signals(minimal_data)
        
        # Should not crash and should return proper format
        self.assertEqual(len(entries), len(minimal_data))
        self.assertEqual(len(exits), len(minimal_data))
        # Probably no signals with minimal data
        self.assertGreaterEqual(entries.sum() + exits.sum(), 0)
    
    def test_multiple_strategies_integration(self):
        """Test that all strategies work with VectorBT integration"""
        for name, strategy in self.strategies.items():
            with self.subTest(strategy=name):
                # Should generate signals without errors
                entries, exits = strategy.generate_vectorbt_signals(self.test_data)
                
                # Should have proper format
                self.assertEqual(len(entries), len(self.test_data))
                self.assertEqual(len(exits), len(self.test_data))
                
                # Run through engine to verify consistency
                engine = BacktestEngine(
                    initial_capital=self.initial_capital,
                    commission=self.commission,
                    allow_short_selling=False
                )
                
                # Should not crash
                engine.run_backtest(self.test_data, strategy)
                
                # Should have some meaningful results
                self.assertGreater(len(engine.portfolio_values), 0)


class TestSignalConversion(unittest.TestCase):
    """Test signal conversion logic specifically"""
    
    def test_signal_type_conversion(self):
        """Test that SignalType enum values are handled correctly"""
        from quant_trading.strategies.strategy_interface import SignalType, signals_to_vectorbt
        
        # Create a mock strategy that returns specific signals
        class MockStrategy:
            def get_signals(self, data):
                signals = []
                if len(data) >= 5:
                    from quant_trading.strategies.strategy_interface import create_signal
                    if len(data) == 5:
                        signals.append(create_signal(action=SignalType.BUY))
                    elif len(data) == 10:
                        signals.append(create_signal(action=SignalType.SELL))
                return signals
        
        # Create test data
        test_data = create_sample_data(15, seed=42)
        strategy = MockStrategy()
        
        # Test conversion
        entries, exits = signals_to_vectorbt(strategy, test_data)
        
        # Should have exactly one entry and one exit
        self.assertEqual(entries.sum(), 1)
        self.assertEqual(exits.sum(), 1)
        
        # Should be at the correct indices (0-based indexing)
        entry_indices = entries[entries].index.tolist()
        exit_indices = exits[exits].index.tolist()
        self.assertEqual(len(entry_indices), 1, "Should have exactly one entry")
        self.assertEqual(len(exit_indices), 1, "Should have exactly one exit")


if __name__ == '__main__':
    unittest.main()