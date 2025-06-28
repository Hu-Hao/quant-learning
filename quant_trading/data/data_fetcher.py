"""
Data Fetcher Module
Handles data retrieval with proper error handling and validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from enum import Enum
import logging


class DataSource(Enum):
    """Available data sources"""
    SYNTHETIC = "synthetic"
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"


class DataValidationError(Exception):
    """Raised when data validation fails"""
    pass


class DataFetcher:
    """
    Data fetcher with robust error handling and validation
    """
    
    def __init__(self, source: DataSource = DataSource.SYNTHETIC):
        self.source = source
        self.logger = logging.getLogger(__name__)
        
    def fetch_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch market data with error handling
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional parameters for data source
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            DataValidationError: If data validation fails
            ValueError: If parameters are invalid
        """
        try:
            # Convert dates if strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
                
            # Validate date range
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
                
            # Fetch data based on source
            if self.source == DataSource.SYNTHETIC:
                data = self._generate_synthetic_data(symbol, start_date, end_date, **kwargs)
            elif self.source == DataSource.YAHOO:
                data = self._fetch_yahoo_data(symbol, start_date, end_date, **kwargs)
            else:
                raise NotImplementedError(f"Data source {self.source} not implemented")
                
            # Validate data
            self._validate_data(data)
            
            self.logger.info(f"Successfully fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            raise
    
    def _generate_synthetic_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_price: float = 100.0,
        trend: float = 0.1,
        volatility: float = 0.02,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        """Generate realistic synthetic market data"""
        
        if seed is not None:
            np.random.seed(seed)
            
        # Calculate number of trading days
        dates = pd.date_range(start_date, end_date, freq='D')
        days = len(dates)
        
        # Generate price movements
        daily_trend = trend / 252  # Convert annual to daily
        daily_returns = np.random.normal(daily_trend, volatility, days)
        
        # Calculate prices
        prices = [initial_price]
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
            
        # Generate OHLC data with realistic spreads
        opens = []
        highs = []
        lows = []
        closes = prices
        volumes = []
        
        for i, close_price in enumerate(closes):
            # Open price (gap from previous close)
            if i == 0:
                open_price = close_price
            else:
                gap = np.random.normal(0, volatility * 0.5)
                open_price = closes[i-1] * (1 + gap)
                
            # High and low with realistic spreads
            daily_range = abs(np.random.normal(0, volatility * 0.8))
            high_price = max(open_price, close_price) * (1 + daily_range)
            low_price = min(open_price, close_price) * (1 - daily_range)
            
            # Volume (correlated with volatility)
            base_volume = 1000000
            volume_mult = 1 + abs(daily_returns[i]) * 5  # Higher volume on big moves
            volume = int(base_volume * volume_mult * np.random.uniform(0.5, 2.0))
            
            opens.append(open_price)
            highs.append(high_price)
            lows.append(low_price)
            volumes.append(volume)
            
        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates)
    
    def _fetch_yahoo_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        **kwargs
    ) -> pd.DataFrame:
        """Fetch data from Yahoo Finance (placeholder for real implementation)"""
        # This would use yfinance or similar library
        # For now, return synthetic data as placeholder
        self.logger.warning("Yahoo Finance fetching not implemented, using synthetic data")
        return self._generate_synthetic_data(symbol, start_date, end_date, **kwargs)
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate fetched data quality
        
        Raises:
            DataValidationError: If validation fails
        """
        if data.empty:
            raise DataValidationError("Data is empty")
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
            
        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (data[col] <= 0).any():
                raise DataValidationError(f"Found non-positive values in {col}")
                
        # Check OHLC relationships
        invalid_ohlc = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_ohlc.any():
            raise DataValidationError("Invalid OHLC relationships found")
            
        # Check for excessive gaps (> 50% price change)
        price_changes = data['close'].pct_change().abs()
        if (price_changes > 0.5).any():
            self.logger.warning("Large price gaps detected (>50%)")
            
        self.logger.info("Data validation passed")


def create_sample_data(
    days: int = 252,
    symbol: str = "SAMPLE",
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to create sample data
    
    Args:
        days: Number of trading days
        symbol: Symbol name
        **kwargs: Additional parameters for data generation
        
    Returns:
        DataFrame with sample market data
    """
    fetcher = DataFetcher(DataSource.SYNTHETIC)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return fetcher.fetch_data(symbol, start_date, end_date, **kwargs)