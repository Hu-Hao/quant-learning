"""
Tests for data fetcher module
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quant_trading.data.data_fetcher import (
    DataFetcher, 
    DataSource, 
    DataValidationError,
    create_sample_data
)


class TestDataFetcher(unittest.TestCase):
    """Test cases for DataFetcher"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fetcher = DataFetcher(DataSource.SYNTHETIC)
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 12, 31)
        
    def test_initialization(self):
        """Test DataFetcher initialization"""
        fetcher = DataFetcher(DataSource.SYNTHETIC)
        self.assertEqual(fetcher.source, DataSource.SYNTHETIC)
        
        fetcher_yahoo = DataFetcher(DataSource.YAHOO)
        self.assertEqual(fetcher_yahoo.source, DataSource.YAHOO)
        
    def test_fetch_synthetic_data(self):
        """Test synthetic data generation"""
        data = self.fetcher.fetch_data(
            symbol="TEST",
            start_date=self.start_date,
            end_date=self.end_date,
            seed=42
        )
        
        # Check data structure
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        
        # Check required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, data.columns)
            
        # Check data types
        for col in required_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(data[col]))
            
    def test_data_validation_valid_data(self):
        """Test validation with valid data"""
        data = self.fetcher.fetch_data("TEST", self.start_date, self.end_date, seed=42)
        
        # Should not raise exception
        self.fetcher._validate_data(data)
        
    def test_data_validation_empty_data(self):
        """Test validation with empty data"""
        empty_data = pd.DataFrame()
        
        with self.assertRaises(DataValidationError):
            self.fetcher._validate_data(empty_data)
            
    def test_data_validation_missing_columns(self):
        """Test validation with missing columns"""
        invalid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            # Missing 'low', 'close', 'volume'
        })
        
        with self.assertRaises(DataValidationError):
            self.fetcher._validate_data(invalid_data)
            
    def test_data_validation_negative_prices(self):
        """Test validation with negative prices"""
        invalid_data = pd.DataFrame({
            'open': [100, -101],  # Negative price
            'high': [102, 103],
            'low': [98, 99],
            'close': [101, 102],
            'volume': [1000, 2000]
        })
        
        with self.assertRaises(DataValidationError):
            self.fetcher._validate_data(invalid_data)
            
    def test_data_validation_invalid_ohlc(self):
        """Test validation with invalid OHLC relationships"""
        invalid_data = pd.DataFrame({
            'open': [100, 101],
            'high': [95, 96],  # High < Open (invalid)
            'low': [98, 99],
            'close': [101, 102],
            'volume': [1000, 2000]
        })
        
        with self.assertRaises(DataValidationError):
            self.fetcher._validate_data(invalid_data)
            
    def test_date_conversion(self):
        """Test date string conversion"""
        data = self.fetcher.fetch_data(
            symbol="TEST",
            start_date="2023-01-01",  # String date
            end_date="2023-01-10",    # String date
            seed=42
        )
        
        self.assertGreater(len(data), 0)
        self.assertIsInstance(data.index, pd.DatetimeIndex)
        
    def test_invalid_date_range(self):
        """Test invalid date range"""
        with self.assertRaises(ValueError):
            self.fetcher.fetch_data(
                symbol="TEST",
                start_date=self.end_date,    # Start after end
                end_date=self.start_date,    # End before start
                seed=42
            )
            
    def test_synthetic_data_parameters(self):
        """Test synthetic data generation with different parameters"""
        # High volatility
        high_vol_data = self.fetcher.fetch_data(
            "TEST", self.start_date, self.start_date + timedelta(days=30),
            volatility=0.1, seed=42
        )
        
        # Low volatility
        low_vol_data = self.fetcher.fetch_data(
            "TEST", self.start_date, self.start_date + timedelta(days=30),
            volatility=0.01, seed=42
        )
        
        # High volatility should have more price variation
        high_vol_std = high_vol_data['close'].std()
        low_vol_std = low_vol_data['close'].std()
        self.assertGreater(high_vol_std, low_vol_std)
        
    def test_reproducible_data(self):
        """Test data reproducibility with same seed"""
        data1 = self.fetcher.fetch_data(
            "TEST", self.start_date, self.start_date + timedelta(days=30),
            seed=42
        )
        
        data2 = self.fetcher.fetch_data(
            "TEST", self.start_date, self.start_date + timedelta(days=30),
            seed=42
        )
        
        # Same seed should produce identical data
        pd.testing.assert_frame_equal(data1, data2)
        
    def test_different_seeds(self):
        """Test data differences with different seeds"""
        data1 = self.fetcher.fetch_data(
            "TEST", self.start_date, self.start_date + timedelta(days=30),
            seed=42
        )
        
        data2 = self.fetcher.fetch_data(
            "TEST", self.start_date, self.start_date + timedelta(days=30),
            seed=123
        )
        
        # Different seeds should produce different data
        self.assertFalse(data1['close'].equals(data2['close']))
        
    def test_yahoo_data_fallback(self):
        """Test Yahoo data source fallback to synthetic"""
        yahoo_fetcher = DataFetcher(DataSource.YAHOO)
        
        # Should fallback to synthetic and log warning
        data = yahoo_fetcher.fetch_data(
            "TEST", self.start_date, self.start_date + timedelta(days=10),
            seed=42
        )
        
        self.assertGreater(len(data), 0)
        self.assertIn('close', data.columns)


class TestCreateSampleData(unittest.TestCase):
    """Test cases for create_sample_data utility function"""
    
    def test_create_sample_data_basic(self):
        """Test basic sample data creation"""
        data = create_sample_data(days=100)
        
        # Date range includes both start and end dates, so expect 100 days
        self.assertGreaterEqual(len(data), 100)
        self.assertIn('close', data.columns)
        self.assertIsInstance(data.index, pd.DatetimeIndex)
        
    def test_create_sample_data_parameters(self):
        """Test sample data with custom parameters"""
        data = create_sample_data(
            days=50,
            symbol="CUSTOM",
            initial_price=200.0,
            trend=0.2,
            volatility=0.05,
            seed=123
        )
        
        self.assertGreaterEqual(len(data), 50)
        self.assertAlmostEqual(data['close'].iloc[0], 200.0, places=0)
        
    def test_create_sample_data_reproducible(self):
        """Test sample data reproducibility"""
        data1 = create_sample_data(days=30, seed=42)
        data2 = create_sample_data(days=30, seed=42)
        
        # Check that price data is the same (ignore timestamp differences)
        pd.testing.assert_series_equal(data1['close'], data2['close'], check_names=False)
        pd.testing.assert_series_equal(data1['open'], data2['open'], check_names=False)
        pd.testing.assert_series_equal(data1['high'], data2['high'], check_names=False)
        pd.testing.assert_series_equal(data1['low'], data2['low'], check_names=False)


class TestDataSourceEnum(unittest.TestCase):
    """Test cases for DataSource enum"""
    
    def test_data_source_values(self):
        """Test DataSource enum values"""
        self.assertEqual(DataSource.SYNTHETIC.value, "synthetic")
        self.assertEqual(DataSource.YAHOO.value, "yahoo")
        self.assertEqual(DataSource.ALPHA_VANTAGE.value, "alpha_vantage")
        self.assertEqual(DataSource.QUANDL.value, "quandl")
        
    def test_data_source_comparison(self):
        """Test DataSource enum comparison"""
        self.assertEqual(DataSource.SYNTHETIC, DataSource.SYNTHETIC)
        self.assertNotEqual(DataSource.SYNTHETIC, DataSource.YAHOO)


class TestDataValidationError(unittest.TestCase):
    """Test cases for DataValidationError"""
    
    def test_data_validation_error(self):
        """Test DataValidationError exception"""
        with self.assertRaises(DataValidationError) as context:
            raise DataValidationError("Test error message")
        
        self.assertIn("Test error message", str(context.exception))


if __name__ == '__main__':
    unittest.main()