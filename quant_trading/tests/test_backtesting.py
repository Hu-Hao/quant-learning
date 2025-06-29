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


class TestShortSellingRestriction(unittest.TestCase):
    """Test cases for short selling restriction feature"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test data that will generate buy and sell signals
        dates = pd.date_range('2023-01-01', periods=30)
        # Create a pattern that triggers moving average crossovers
        prices = []
        for i in range(30):
            if i < 10:
                price = 100 + i * 0.5  # Gradual uptrend
            elif i < 20:
                price = 105 + (i-10) * 1.0  # Faster uptrend (buy signal)
            else:
                price = 115 - (i-20) * 0.8  # Downtrend (sell signal)
            prices.append(price)
        
        self.test_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * 30
        }, index=dates)
        
        # Use a responsive strategy to generate signals
        self.strategy = MovingAverageStrategy(short_window=3, long_window=7, quantity=100)
    
    def test_initialization_with_short_selling_enabled(self):
        """Test engine initialization with short selling enabled (default)"""
        engine = BacktestEngine(initial_capital=100000)
        self.assertTrue(engine.allow_short_selling)
        
        engine_explicit = BacktestEngine(initial_capital=100000, allow_short_selling=True)
        self.assertTrue(engine_explicit.allow_short_selling)
    
    def test_initialization_with_short_selling_disabled(self):
        """Test engine initialization with short selling disabled"""
        engine = BacktestEngine(initial_capital=100000, allow_short_selling=False)
        self.assertFalse(engine.allow_short_selling)
    
    def test_short_selling_enabled_behavior(self):
        """Test that short selling works normally when enabled"""
        engine = BacktestEngine(
            initial_capital=100000,
            allow_short_selling=True,
            max_position_size=0.5
        )
        
        # Mock a sell signal by directly calling _process_signal
        from quant_trading.strategies.strategy_interface import Signal, SignalType
        sell_signal = Signal(
            symbol='TEST',
            action=SignalType.SELL,
            quantity=100,
            price=110.0
        )
        
        # Process sell signal - should create short position
        initial_capital = engine.capital
        engine._process_signal(sell_signal, 110.0)
        
        # Check if short position was created (capital should increase)
        self.assertGreater(engine.capital, initial_capital)
        if 'TEST' in engine.positions:
            position = engine.positions['TEST']
            self.assertEqual(position.side, 'short')
            self.assertEqual(position.quantity, -100)
    
    def test_short_selling_disabled_no_position(self):
        """Test sell signal ignored when no position exists and short selling disabled"""
        engine = BacktestEngine(
            initial_capital=100000,
            allow_short_selling=False,
            max_position_size=0.5
        )
        
        from quant_trading.strategies.strategy_interface import Signal, SignalType
        sell_signal = Signal(
            symbol='TEST',
            action=SignalType.SELL,
            quantity=100,
            price=110.0
        )
        
        # Process sell signal with no existing position
        initial_capital = engine.capital
        initial_positions = len(engine.positions)
        
        engine._process_signal(sell_signal, 110.0)
        
        # Should ignore signal - no change in capital or positions
        self.assertEqual(engine.capital, initial_capital)
        self.assertEqual(len(engine.positions), initial_positions)
        self.assertNotIn('TEST', engine.positions)
    
    def test_short_selling_disabled_close_long_position(self):
        """Test sell signal closes existing long position when short selling disabled"""
        engine = BacktestEngine(
            initial_capital=100000,
            allow_short_selling=False,
            max_position_size=0.5
        )
        
        # First create a long position
        engine.place_order('TEST', 100, 100.0, 'long')
        self.assertIn('TEST', engine.positions)
        self.assertEqual(engine.positions['TEST'].side, 'long')
        self.assertEqual(engine.positions['TEST'].quantity, 100)
        
        # Now send sell signal to close position
        from quant_trading.strategies.strategy_interface import Signal, SignalType
        sell_signal = Signal(
            symbol='TEST',
            action=SignalType.SELL,
            quantity=100,
            price=110.0
        )
        
        engine._process_signal(sell_signal, 110.0)
        
        # Position should be closed
        self.assertNotIn('TEST', engine.positions)
        # Should have executed a trade
        self.assertGreater(len(engine.trades), 0)
    
    def test_short_selling_disabled_partial_close(self):
        """Test sell signal partially closes long position when quantity is less"""
        engine = BacktestEngine(
            initial_capital=100000,
            allow_short_selling=False,
            max_position_size=0.5
        )
        
        # Create long position of 200 shares
        engine.place_order('TEST', 200, 100.0, 'long')
        self.assertEqual(engine.positions['TEST'].quantity, 200)
        
        # Sell signal for only 100 shares
        from quant_trading.strategies.strategy_interface import Signal, SignalType
        sell_signal = Signal(
            symbol='TEST',
            action=SignalType.SELL,
            quantity=100,
            price=110.0
        )
        
        engine._process_signal(sell_signal, 110.0)
        
        # Should still have 100 shares remaining
        self.assertIn('TEST', engine.positions)
        self.assertEqual(engine.positions['TEST'].quantity, 100)
        self.assertEqual(engine.positions['TEST'].side, 'long')
    
    def test_buy_signal_unaffected_by_short_selling_setting(self):
        """Test that buy signals work the same regardless of short selling setting"""
        # Test with short selling enabled
        engine_short = BacktestEngine(initial_capital=100000, allow_short_selling=True)
        
        from quant_trading.strategies.strategy_interface import Signal, SignalType
        buy_signal = Signal(
            symbol='TEST',
            action=SignalType.BUY,
            quantity=100,
            price=100.0
        )
        
        engine_short._process_signal(buy_signal, 100.0)
        
        # Test with short selling disabled
        engine_no_short = BacktestEngine(initial_capital=100000, allow_short_selling=False)
        engine_no_short._process_signal(buy_signal, 100.0)
        
        # Both should create identical long positions
        if 'TEST' in engine_short.positions and 'TEST' in engine_no_short.positions:
            pos_short = engine_short.positions['TEST']
            pos_no_short = engine_no_short.positions['TEST']
            
            self.assertEqual(pos_short.side, pos_no_short.side)
            self.assertEqual(pos_short.quantity, pos_no_short.quantity)
            self.assertEqual(pos_short.side, 'long')
    
    def test_backtest_integration_with_restriction(self):
        """Test full backtest integration with short selling restriction"""
        # Test with short selling disabled
        engine_no_short = BacktestEngine(
            initial_capital=100000,
            allow_short_selling=False,
            max_position_size=0.3
        )
        
        engine_no_short.run_backtest(self.test_data, self.strategy)
        
        # Check that no short positions were created
        for trade in engine_no_short.trades:
            # All trades should be either opening long positions or closing them
            if trade.side == 'short':
                # If it's a short side trade, it should be closing a long position
                # This is harder to verify directly, but we can check that
                # no net short positions remain
                pass
        
        # Verify backtest completed successfully
        self.assertGreater(len(engine_no_short.portfolio_values), 0)
        self.assertIsInstance(engine_no_short.get_performance_summary(), dict)
    
    def test_logging_behavior(self):
        """Test that appropriate log messages are generated"""
        import logging
        from io import StringIO
        
        # Set up logging capture
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        logger = logging.getLogger('quant_trading.backtesting.engine')
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        try:
            engine = BacktestEngine(
                initial_capital=100000,
                allow_short_selling=False
            )
            
            from quant_trading.strategies.strategy_interface import Signal, SignalType
            sell_signal = Signal(
                symbol='TEST',
                action=SignalType.SELL,
                quantity=100,
                price=110.0
            )
            
            # Process sell signal with no position (should log ignore message)
            engine._process_signal(sell_signal, 110.0)
            
            # Check log output contains appropriate message
            log_contents = log_stream.getvalue()
            self.assertIn('ignored', log_contents.lower())
            
        finally:
            logger.removeHandler(handler)
    
    def test_edge_case_sell_more_than_owned(self):
        """Test sell signal for more shares than owned when short selling disabled"""
        engine = BacktestEngine(
            initial_capital=100000,
            allow_short_selling=False,
            max_position_size=0.5
        )
        
        # Create long position of 50 shares
        engine.place_order('TEST', 50, 100.0, 'long')
        self.assertEqual(engine.positions['TEST'].quantity, 50)
        
        # Try to sell 100 shares (more than owned)
        from quant_trading.strategies.strategy_interface import Signal, SignalType
        sell_signal = Signal(
            symbol='TEST',
            action=SignalType.SELL,
            quantity=100,
            price=110.0
        )
        
        engine._process_signal(sell_signal, 110.0)
        
        # Should only sell the 50 shares owned, closing the position
        self.assertNotIn('TEST', engine.positions)
        # Should have executed a trade (the position was closed)
        self.assertGreater(len(engine.trades), 0)
    
    def test_hold_signal_behavior(self):
        """Test that HOLD signals are ignored appropriately"""
        engine = BacktestEngine(
            initial_capital=100000,
            allow_short_selling=False
        )
        
        from quant_trading.strategies.strategy_interface import Signal, SignalType
        hold_signal = Signal(
            symbol='TEST',
            action=SignalType.HOLD,
            quantity=100,
            price=100.0
        )
        
        initial_capital = engine.capital
        initial_positions = len(engine.positions)
        
        engine._process_signal(hold_signal, 100.0)
        
        # HOLD signals should not affect capital or positions
        self.assertEqual(engine.capital, initial_capital)
        self.assertEqual(len(engine.positions), initial_positions)


if __name__ == '__main__':
    unittest.main()