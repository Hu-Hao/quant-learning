"""
Tests for trading strategies
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.strategies.momentum import MomentumStrategy
from quant_trading.strategies.mean_reversion import MeanReversionStrategy
from quant_trading.strategies.base_strategy import Signal, SignalType
from quant_trading.data.data_fetcher import create_sample_data


class TestBaseStrategy(unittest.TestCase):
    """Test base strategy functionality"""
    
    def test_signal_creation(self):
        """Test Signal dataclass"""
        signal = Signal(
            symbol='TEST',
            action=SignalType.BUY,
            quantity=100,
            price=50.0,
            confidence=0.8
        )
        
        self.assertEqual(signal.symbol, 'TEST')
        self.assertEqual(signal.action, SignalType.BUY)
        self.assertEqual(signal.quantity, 100)
        self.assertEqual(signal.price, 50.0)
        self.assertEqual(signal.confidence, 0.8)
        
        # Test conversion to dict
        signal_dict = signal.to_dict()
        self.assertEqual(signal_dict['action'], 'buy')
        self.assertEqual(signal_dict['symbol'], 'TEST')


class TestMovingAverageStrategy(unittest.TestCase):
    """Test Moving Average Strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy = MovingAverageStrategy(short_window=5, long_window=10)
        self.data = create_sample_data(50, trend=0.1, volatility=0.02, seed=42)
        
    def test_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.params['short_window'], 5)
        self.assertEqual(self.strategy.params['long_window'], 10)
        self.assertEqual(self.strategy.name, 'MovingAverage')
        
    def test_parameter_validation(self):
        """Test parameter validation"""
        with self.assertRaises(ValueError):
            MovingAverageStrategy(short_window=10, long_window=5)  # Invalid: short >= long
            
        with self.assertRaises(ValueError):
            MovingAverageStrategy(short_window=0, long_window=10)  # Invalid: non-positive
            
    def test_data_validation(self):
        """Test data validation"""
        # Missing columns
        invalid_data = pd.DataFrame({'price': [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.strategy.validate_data(invalid_data)
            
        # Empty data
        empty_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.strategy.validate_data(empty_data)
            
    def test_signal_generation(self):
        """Test signal generation"""
        # Create trending data for clear signals
        dates = pd.date_range('2023-01-01', periods=30)
        
        # Create golden cross scenario (short MA crosses above long MA)
        prices = list(range(95, 110)) + list(range(110, 125))  # Uptrend
        data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': [1000000] * len(prices)
        }, index=dates)
        
        # Get signals for the full dataset
        signals = self.strategy.get_signal(data)
        
        # Should have some signals
        self.assertIsInstance(signals, list)
        
        # If there are signals, they should be properly formatted
        for signal in signals:
            self.assertIn('symbol', signal)
            self.assertIn('action', signal)
            self.assertIn('quantity', signal)
            self.assertIn('price', signal)
            
    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        short_data = create_sample_data(5)  # Less than long_window
        signals = self.strategy.get_signal(short_data)
        self.assertEqual(len(signals), 0)
        
    def test_indicator_values(self):
        """Test indicator calculation"""
        indicators = self.strategy.get_indicator_values(self.data)
        
        if not indicators.empty:
            self.assertIn('short_ma', indicators.columns)
            self.assertIn('long_ma', indicators.columns)
            self.assertIn('ma_diff', indicators.columns)
            self.assertIn('ma_ratio', indicators.columns)


class TestMomentumStrategy(unittest.TestCase):
    """Test Momentum Strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy = MomentumStrategy(
            lookback_period=10,
            momentum_threshold=0.02
        )
        self.data = create_sample_data(50, trend=0.15, volatility=0.03, seed=42)
        
    def test_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.params['lookback_period'], 10)
        self.assertEqual(self.strategy.params['momentum_threshold'], 0.02)
        self.assertEqual(self.strategy.name, 'Momentum')
        
    def test_parameter_validation(self):
        """Test parameter validation"""
        with self.assertRaises(ValueError):
            MomentumStrategy(lookback_period=1)  # Too short
            
        with self.assertRaises(ValueError):
            MomentumStrategy(momentum_threshold=1.5)  # Too high
            
    def test_momentum_calculation(self):
        """Test momentum calculation"""
        # Create simple test data
        test_data = pd.DataFrame({
            'close': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120],
            'open': [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120],
            'high': [101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121],
            'low': [99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119],
            'volume': [1000000] * 11
        })
        
        momentum = self.strategy._calculate_momentum(test_data)
        expected_momentum = (120 / 100) - 1  # 20% gain over 10 periods
        self.assertAlmostEqual(momentum, expected_momentum, places=3)
        
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        # Create test data with clear trend
        test_data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114],
            'volume': [1000000] * 16
        })
        
        rsi = self.strategy._calculate_rsi(test_data)
        
        # RSI should be between 0 and 100
        self.assertGreaterEqual(rsi, 0)
        self.assertLessEqual(rsi, 100)
        
        # With consistent uptrend, RSI should be high
        self.assertGreater(rsi, 50)


class TestMeanReversionStrategy(unittest.TestCase):
    """Test Mean Reversion Strategy"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.strategy = MeanReversionStrategy(
            window=20,
            entry_threshold=2.0,
            exit_threshold=0.5
        )
        self.data = create_sample_data(50, trend=0.0, volatility=0.05, seed=42)
        
    def test_initialization(self):
        """Test strategy initialization"""
        self.assertEqual(self.strategy.params['window'], 20)
        self.assertEqual(self.strategy.params['entry_threshold'], 2.0)
        self.assertEqual(self.strategy.params['exit_threshold'], 0.5)
        self.assertEqual(self.strategy.name, 'MeanReversion')
        
    def test_parameter_validation(self):
        """Test parameter validation"""
        with self.assertRaises(ValueError):
            MeanReversionStrategy(window=1)  # Too short
            
        with self.assertRaises(ValueError):
            MeanReversionStrategy(entry_threshold=0.5, exit_threshold=1.0)  # Invalid thresholds
            
    def test_z_score_calculation(self):
        """Test z-score calculation"""
        # Create test data with clear deviation
        base_price = 100
        prices = [base_price] * 19 + [base_price * 1.1]  # Last price is 10% above mean
        
        test_data = pd.DataFrame({
            'close': prices,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'volume': [1000000] * 20
        })
        
        z_score = self.strategy._calculate_z_score(test_data)
        
        # Z-score should be positive (price above mean)
        self.assertGreater(z_score, 0)
        
    def test_bollinger_signal(self):
        """Test Bollinger Bands signal generation"""
        # Create test data
        test_data = pd.DataFrame({
            'close': [100] * 19 + [120],  # Last price significantly above mean
            'open': [100] * 19 + [120],
            'high': [101] * 19 + [121],
            'low': [99] * 19 + [119],
            'volume': [1000000] * 20
        })
        
        signal = self.strategy._get_bollinger_signal(test_data)
        
        # Should generate sell signal for price above upper band
        self.assertEqual(signal, 'sell')
        
    def test_indicator_values(self):
        """Test indicator calculation"""
        indicators = self.strategy.get_indicator_values(self.data)
        
        if not indicators.empty:
            self.assertIn('sma', indicators.columns)
            self.assertIn('std', indicators.columns)
            self.assertIn('z_score', indicators.columns)
            self.assertIn('bb_upper', indicators.columns)
            self.assertIn('bb_lower', indicators.columns)


if __name__ == '__main__':
    unittest.main()