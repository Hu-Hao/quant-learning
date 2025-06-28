"""
Tests for backtesting engine
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quant_trading.backtesting.engine import BacktestEngine, Trade, Position
from quant_trading.data.data_fetcher import create_sample_data
from quant_trading.strategies.moving_average import MovingAverageStrategy


class TestBacktestEngine(unittest.TestCase):
    """Test cases for BacktestEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,
            slippage=0.001
        )
        self.sample_data = create_sample_data(100, seed=42)
        
    def test_initialization(self):
        """Test engine initialization"""
        self.assertEqual(self.engine.initial_capital, 100000)
        self.assertEqual(self.engine.commission, 0.001)
        self.assertEqual(self.engine.slippage, 0.001)
        self.assertEqual(self.engine.capital, 100000)
        self.assertEqual(len(self.engine.positions), 0)
        self.assertEqual(len(self.engine.trades), 0)
        
    def test_reset(self):
        """Test engine reset functionality"""
        # Make some changes
        self.engine.capital = 50000
        self.engine.positions['TEST'] = Position('TEST', 100, 50.0, 'long', datetime.now())
        self.engine.portfolio_values = [100000, 105000]
        
        # Reset
        self.engine.reset()
        
        # Check reset state
        self.assertEqual(self.engine.capital, 100000)
        self.assertEqual(len(self.engine.positions), 0)
        self.assertEqual(len(self.engine.trades), 0)
        self.assertEqual(len(self.engine.portfolio_values), 0)
        
    def test_place_order_long(self):
        """Test placing long orders"""
        # Set up price history for slippage calculation
        self.engine.price_history = [100.0] * 10
        
        # Place long order
        success = self.engine.place_order('TEST', 100, 100.0, 'long')
        
        self.assertTrue(success)
        self.assertIn('TEST', self.engine.positions)
        
        position = self.engine.positions['TEST']
        self.assertEqual(position.quantity, 100)
        self.assertEqual(position.side, 'long')
        self.assertGreater(position.avg_price, 100.0)  # Should include slippage
        
        # Check capital reduction
        self.assertLess(self.engine.capital, 100000)
        
    def test_place_order_short(self):
        """Test placing short orders"""
        # Set up price history
        self.engine.price_history = [100.0] * 10
        
        # Place short order
        success = self.engine.place_order('TEST', 100, 100.0, 'short')
        
        self.assertTrue(success)
        self.assertIn('TEST', self.engine.positions)
        
        position = self.engine.positions['TEST']
        self.assertEqual(position.quantity, -100)
        self.assertEqual(position.side, 'short')
        self.assertLess(position.avg_price, 100.0)  # Should include slippage
        
        # Check capital increase
        self.assertGreater(self.engine.capital, 100000)
        
    def test_position_closure(self):
        """Test closing positions"""
        self.engine.price_history = [100.0] * 10
        
        # Open long position
        self.engine.place_order('TEST', 100, 100.0, 'long')
        self.assertIn('TEST', self.engine.positions)
        
        # Close position
        self.engine.place_order('TEST', 100, 105.0, 'short')
        self.assertNotIn('TEST', self.engine.positions)
        
        # Check trade was recorded
        self.assertEqual(len(self.engine.trades), 1)
        trade = self.engine.trades[0]
        self.assertEqual(trade.symbol, 'TEST')
        self.assertEqual(trade.quantity, 100)
        self.assertEqual(trade.side, 'long')
        self.assertGreater(trade.pnl, 0)  # Should be profitable
        
    def test_insufficient_capital(self):
        """Test order rejection due to insufficient capital"""
        # Try to place order larger than available capital
        success = self.engine.place_order('TEST', 1000000, 100.0, 'long')
        self.assertFalse(success)
        self.assertEqual(len(self.engine.positions), 0)
        
    def test_position_size_limits(self):
        """Test position size limit enforcement"""
        # Try to place order exceeding position size limit
        large_quantity = int(self.engine.initial_capital * 0.2 / 100)  # 20% position
        success = self.engine.place_order('TEST', large_quantity, 100.0, 'long')
        self.assertFalse(success)
        
    def test_slippage_calculation(self):
        """Test slippage calculation with different volatility"""
        # Low volatility
        self.engine.price_history = [100.0, 100.1, 99.9, 100.05, 99.95] * 2
        low_vol_price = self.engine._calculate_slippage(100.0, 'long', 100)
        
        # High volatility  
        self.engine.price_history = [100.0, 110.0, 90.0, 120.0, 80.0] * 2
        high_vol_price = self.engine._calculate_slippage(100.0, 'long', 100)
        
        # High volatility should result in more slippage
        self.assertGreater(high_vol_price - 100.0, low_vol_price - 100.0)
        
    def test_run_backtest_with_strategy(self):
        """Test running backtest with strategy"""
        strategy = MovingAverageStrategy(short_window=5, long_window=10)
        
        # Run backtest
        self.engine.run_backtest(self.sample_data, strategy)
        
        # Check that backtest ran
        self.assertGreater(len(self.engine.portfolio_values), 0)
        self.assertEqual(len(self.engine.portfolio_values), len(self.sample_data))
        
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        # Add some portfolio values
        self.engine.portfolio_values = [100000, 105000, 103000, 107000, 110000]
        
        metrics = self.engine.get_performance_summary()
        
        self.assertIn('total_return', metrics)
        self.assertIn('annualized_return', metrics)
        self.assertIn('volatility', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        
        # Check basic sanity
        self.assertGreater(metrics['total_return'], 0)  # Positive return
        self.assertGreater(metrics['max_drawdown'], 0)  # Some drawdown occurred


class TestTradeDataClass(unittest.TestCase):
    """Test Trade dataclass"""
    
    def test_trade_creation(self):
        """Test trade creation and properties"""
        trade = Trade(
            symbol='TEST',
            entry_time=datetime.now(),
            exit_time=datetime.now(),
            entry_price=100.0,
            exit_price=105.0,
            quantity=100,
            side='long',
            pnl=450.0,
            commission_paid=50.0,
            slippage_cost=25.0
        )
        
        self.assertEqual(trade.gross_pnl, 500.0)  # 100 * (105 - 100)
        self.assertEqual(trade.total_costs, 75.0)  # 50 + 25
        self.assertEqual(trade.return_pct, 0.045)  # 450 / 10000


class TestPositionDataClass(unittest.TestCase):
    """Test Position dataclass"""
    
    def test_position_creation(self):
        """Test position creation and properties"""
        position = Position(
            symbol='TEST',
            quantity=100,
            avg_price=50.0,
            side='long',
            entry_time=datetime.now()
        )
        
        self.assertEqual(position.value, 5000.0)  # 100 * 50
        self.assertEqual(position.market_value(55.0), 5500.0)  # 100 * 55
        self.assertEqual(position.unrealized_pnl(55.0), 500.0)  # 100 * (55 - 50)


if __name__ == '__main__':
    unittest.main()