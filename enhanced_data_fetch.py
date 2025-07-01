#!/usr/bin/env python3
"""
Enhanced data fetching with multiple fallback options
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def fetch_stock_data_robust(symbol="AAPL", days=365, fallback_to_sample=True):
    """
    Robust stock data fetching with multiple fallback options
    
    Args:
        symbol: Stock symbol to fetch
        days: Number of days of historical data
        fallback_to_sample: Whether to use sample data if real data fails
        
    Returns:
        pandas.DataFrame: Stock data with OHLCV columns
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"📊 Fetching {symbol} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    
    # Strategy 1: Try yfinance with different parameters
    strategies = [
        # Standard approach
        lambda: yf.Ticker(symbol).history(start=start_date, end=end_date),
        
        # Try with period parameter instead of dates
        lambda: yf.Ticker(symbol).history(period="1y"),
        
        # Try shorter period
        lambda: yf.Ticker(symbol).history(period="6mo"),
        
        # Try different symbol formats
        lambda: yf.Ticker(f"{symbol}.US").history(start=start_date, end=end_date) if symbol == "AAPL" else None,
    ]
    
    for i, strategy in enumerate(strategies):
        try:
            print(f"  Trying strategy {i+1}...")
            data = strategy()
            
            if data is None or data.empty:
                continue
            
            # Handle timezone issues
            if hasattr(data.index, 'tz') and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Convert column names to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            # Basic validation
            if len(data) < 10:  # Need at least 10 days
                continue
                
            # Remove any NaN values
            data = data.dropna()
            
            if len(data) < 10:
                continue
            
            print(f"  ✅ Strategy {i+1} successful!")
            print(f"     Fetched {len(data)} days of data")
            print(f"     Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            print(f"     Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
            print(f"     Total return: {(data['close'].iloc[-1]/data['close'].iloc[0] - 1)*100:.1f}%")
            
            return data, True  # Return data and success flag
            
        except Exception as e:
            print(f"  ❌ Strategy {i+1} failed: {e}")
            continue
    
    # If all strategies failed and fallback is enabled
    if fallback_to_sample:
        print("📊 All real data strategies failed. Using sample data...")
        
        try:
            # Import our sample data generator
            import sys
            import os
            sys.path.append('.')
            from quant_trading.data.data_fetcher import create_sample_data
            
            # Create realistic sample data based on the requested symbol
            if symbol == "AAPL":
                sample_data = create_sample_data(
                    days=min(days, 252),  # Max 1 year
                    initial_price=150,
                    trend=0.15,  # 15% annual growth
                    volatility=0.25,  # 25% annual volatility
                    seed=42
                )
            else:
                # Generic sample data
                sample_data = create_sample_data(
                    days=min(days, 252),
                    initial_price=100,
                    trend=0.10,
                    volatility=0.20,
                    seed=42
                )
            
            print(f"  ✅ Generated {len(sample_data)} days of sample data")
            print(f"     Date range: {sample_data.index[0].strftime('%Y-%m-%d')} to {sample_data.index[-1].strftime('%Y-%m-%d')}")
            print(f"     Price range: ${sample_data['low'].min():.2f} - ${sample_data['high'].max():.2f}")
            print(f"     Total return: {(sample_data['close'].iloc[-1]/sample_data['close'].iloc[0] - 1)*100:.1f}%")
            print(f"     📝 Note: Using realistic sample data due to API issues")
            
            return sample_data, False  # Return data and failed flag
            
        except Exception as e:
            print(f"❌ Sample data generation also failed: {e}")
            raise RuntimeError("All data fetching strategies failed")
    else:
        raise RuntimeError(f"Could not fetch data for {symbol}")

def test_data_fetching():
    """Test the robust data fetching"""
    symbols_to_test = ["AAPL", "MSFT", "GOOGL", "SPY"]
    
    for symbol in symbols_to_test:
        print(f"\n{'='*50}")
        print(f"Testing {symbol}")
        print('='*50)
        
        try:
            data, is_real = fetch_stock_data_robust(symbol, days=365)
            source = "Real market data" if is_real else "Sample data"
            print(f"✅ {symbol}: Successfully fetched {len(data)} days of {source}")
            
            # Quick validation
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                print(f"⚠️ Missing columns: {missing_cols}")
            else:
                print("✅ All required columns present")
                
            break  # If we get one successful fetch, we're good
            
        except Exception as e:
            print(f"❌ {symbol}: Failed - {e}")
            continue
    
    print(f"\n{'='*50}")
    print("Data fetching test completed!")

if __name__ == "__main__":
    test_data_fetching()