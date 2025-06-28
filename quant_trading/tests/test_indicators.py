"""
Tests for technical indicators
"""

import unittest
import pandas as pd
import numpy as np
from quant_trading.utils.indicators import TechnicalIndicators


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for TechnicalIndicators"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Generate realistic price series
        price_changes = np.random.normal(0.001, 0.02, 100)  # 0.1% daily return, 2% volatility
        prices = [100.0]
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
            
        self.price_data = pd.Series(prices, index=dates, name='close')
        
        # OHLC data for indicators that need it
        self.high_data = self.price_data * 1.02
        self.low_data = self.price_data * 0.98
        self.volume_data = pd.Series(np.random.randint(100000, 1000000, 100), index=dates)
        
    def test_sma_basic(self):
        """Test Simple Moving Average calculation"""
        sma = TechnicalIndicators.sma(self.price_data, period=20)
        
        # Check basic properties
        self.assertEqual(len(sma), len(self.price_data))
        self.assertTrue(pd.isna(sma.iloc[:19]).all())  # First 19 should be NaN
        self.assertFalse(pd.isna(sma.iloc[19:]).any())  # Rest should not be NaN
        
        # Check calculation accuracy for a simple case
        simple_data = pd.Series([1, 2, 3, 4, 5])
        simple_sma = TechnicalIndicators.sma(simple_data, period=3)
        expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
        pd.testing.assert_series_equal(simple_sma, expected)
        
    def test_sma_edge_cases(self):
        """Test SMA edge cases"""
        # Invalid period
        with self.assertRaises(ValueError):
            TechnicalIndicators.sma(self.price_data, period=0)
            
        with self.assertRaises(ValueError):
            TechnicalIndicators.sma(self.price_data, period=-1)
            
        # Period larger than data
        short_data = pd.Series([1, 2, 3])
        sma = TechnicalIndicators.sma(short_data, period=10)
        self.assertTrue(sma.isna().all())
        
    def test_ema_basic(self):
        """Test Exponential Moving Average calculation"""
        ema = TechnicalIndicators.ema(self.price_data, period=20)
        
        # Check basic properties
        self.assertEqual(len(ema), len(self.price_data))
        self.assertFalse(ema.isna().any())  # EMA should not have NaN values
        
        # EMA should be closer to recent prices than SMA
        sma = TechnicalIndicators.sma(self.price_data, period=20)
        current_price = self.price_data.iloc[-1]
        
        # EMA should be closer to current price than SMA
        ema_diff = abs(ema.iloc[-1] - current_price)
        sma_diff = abs(sma.iloc[-1] - current_price)
        self.assertLessEqual(ema_diff, sma_diff)
        
    def test_ema_custom_alpha(self):
        """Test EMA with custom alpha"""
        ema_default = TechnicalIndicators.ema(self.price_data, period=20)
        ema_custom = TechnicalIndicators.ema(self.price_data, period=20, alpha=0.5)
        
        # Different alpha should produce different results
        self.assertFalse(ema_default.equals(ema_custom))
        
    def test_rsi_basic(self):
        """Test RSI calculation"""
        rsi = TechnicalIndicators.rsi(self.price_data, period=14)
        
        # Check basic properties
        self.assertEqual(len(rsi), len(self.price_data))
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
        
        # Test with trending data
        trending_up = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115])
        rsi_up = TechnicalIndicators.rsi(trending_up, period=14)
        self.assertGreater(rsi_up.iloc[-1], 50)  # Uptrend should have RSI > 50
        
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        bb = TechnicalIndicators.bollinger_bands(self.price_data, period=20, std_dev=2.0)
        
        # Check structure
        self.assertIsInstance(bb, pd.DataFrame)
        self.assertIn('upper', bb.columns)
        self.assertIn('middle', bb.columns)
        self.assertIn('lower', bb.columns)
        
        # Check relationships
        valid_data = bb.dropna()
        self.assertTrue((valid_data['upper'] >= valid_data['middle']).all())
        self.assertTrue((valid_data['middle'] >= valid_data['lower']).all())
        
        # Middle band should equal SMA
        sma = TechnicalIndicators.sma(self.price_data, period=20)
        pd.testing.assert_series_equal(bb['middle'], sma, check_names=False)
        
    def test_macd(self):
        """Test MACD calculation"""
        macd = TechnicalIndicators.macd(self.price_data, fast_period=12, slow_period=26, signal_period=9)
        
        # Check structure
        self.assertIsInstance(macd, pd.DataFrame)
        self.assertIn('macd', macd.columns)
        self.assertIn('signal', macd.columns)
        self.assertIn('histogram', macd.columns)
        
        # Check relationships
        valid_data = macd.dropna()
        expected_histogram = valid_data['macd'] - valid_data['signal']
        pd.testing.assert_series_equal(valid_data['histogram'], expected_histogram, check_names=False)
        
    def test_macd_invalid_periods(self):
        """Test MACD with invalid periods"""
        with self.assertRaises(ValueError):
            TechnicalIndicators.macd(self.price_data, fast_period=26, slow_period=12)  # Fast >= Slow
            
    def test_stochastic(self):
        """Test Stochastic Oscillator calculation"""
        stoch = TechnicalIndicators.stochastic(
            self.high_data, self.low_data, self.price_data, k_period=14, d_period=3
        )
        
        # Check structure
        self.assertIsInstance(stoch, pd.DataFrame)
        self.assertIn('%K', stoch.columns)
        self.assertIn('%D', stoch.columns)
        
        # Check range (0-100)
        valid_data = stoch.dropna()
        self.assertTrue((valid_data['%K'] >= 0).all())
        self.assertTrue((valid_data['%K'] <= 100).all())
        self.assertTrue((valid_data['%D'] >= 0).all())
        self.assertTrue((valid_data['%D'] <= 100).all())
        
    def test_atr(self):
        """Test Average True Range calculation"""
        atr = TechnicalIndicators.atr(self.high_data, self.low_data, self.price_data, period=14)
        
        # Check basic properties
        self.assertEqual(len(atr), len(self.price_data))
        
        # ATR should be positive
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr > 0).all())
        
    def test_obv(self):
        """Test On-Balance Volume calculation"""
        obv = TechnicalIndicators.obv(self.price_data, self.volume_data)
        
        # Check basic properties
        self.assertEqual(len(obv), len(self.price_data))
        self.assertEqual(obv.name, 'OBV')
        
        # OBV should be cumulative
        self.assertTrue(obv.is_monotonic_increasing or obv.is_monotonic_decreasing or 
                      len(obv.diff().dropna().unique()) > 2)  # Not flat
        
    def test_williams_r(self):
        """Test Williams %R calculation"""
        williams = TechnicalIndicators.williams_r(
            self.high_data, self.low_data, self.price_data, period=14
        )
        
        # Check basic properties
        self.assertEqual(len(williams), len(self.price_data))
        
        # Williams %R should be between -100 and 0
        valid_williams = williams.dropna()
        self.assertTrue((valid_williams >= -100).all())
        self.assertTrue((valid_williams <= 0).all())
        
    def test_momentum(self):
        """Test Momentum calculation"""
        momentum = TechnicalIndicators.momentum(self.price_data, period=10)
        
        # Check basic properties
        self.assertEqual(len(momentum), len(self.price_data))
        
        # Test with simple data
        simple_data = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120])
        simple_momentum = TechnicalIndicators.momentum(simple_data, period=10)
        expected_last = 120 / 102  # Current / 10 periods ago
        self.assertAlmostEqual(simple_momentum.iloc[-1], expected_last, places=6)
        
    def test_rate_of_change(self):
        """Test Rate of Change calculation"""
        roc = TechnicalIndicators.rate_of_change(self.price_data, period=10)
        
        # Check basic properties
        self.assertEqual(len(roc), len(self.price_data))
        
        # Test with simple data
        simple_data = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120])
        simple_roc = TechnicalIndicators.rate_of_change(simple_data, period=10)
        expected_last = ((120 / 102) - 1) * 100  # Percentage change
        self.assertAlmostEqual(simple_roc.iloc[-1], expected_last, places=6)
        
    def test_edge_cases_empty_data(self):
        """Test indicators with empty data"""
        empty_series = pd.Series([], dtype=float)
        
        # Most indicators should handle empty data gracefully
        sma = TechnicalIndicators.sma(empty_series, period=10)
        self.assertTrue(sma.empty)
        
        ema = TechnicalIndicators.ema(empty_series, period=10)
        self.assertTrue(ema.empty)
        
    def test_indicators_with_insufficient_data(self):
        """Test indicators with insufficient data"""
        short_data = pd.Series([100, 101, 102])
        
        # RSI with insufficient data
        rsi = TechnicalIndicators.rsi(short_data, period=14)
        self.assertTrue(rsi.isna().all())
        
        # Bollinger Bands with insufficient data
        bb = TechnicalIndicators.bollinger_bands(short_data, period=20)
        self.assertTrue(bb.isna().all().all())


if __name__ == '__main__':
    unittest.main()