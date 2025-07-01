#!/usr/bin/env python3
"""
Fix yfinance timezone issues for Apple stock data download
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def setup_timezone_environment():
    """Setup proper timezone environment for yfinance"""
    print("🕒 Setting up timezone environment...")
    
    # Set timezone environment variables
    os.environ['TZ'] = 'America/New_York'  # NYSE/NASDAQ timezone
    
    # Alternative: Pacific Time where Apple is headquartered
    # os.environ['TZ'] = 'America/Los_Angeles'
    
    try:
        import time
        time.tzset()  # Apply timezone setting (Unix/Linux/Mac only)
        print("   ✅ Timezone set to Eastern Time (NYSE/NASDAQ)")
    except AttributeError:
        # Windows doesn't have tzset
        print("   ⚠️ Windows detected - timezone setting limited")
    
    # Also set pandas timezone display
    pd.set_option('display.timezone', 'America/New_York')

def fix_yfinance_session():
    """Create a properly configured yfinance session"""
    print("🔧 Configuring yfinance session...")
    
    # Update yfinance user agent to avoid blocking
    import requests
    
    # Create a session with proper headers
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    # Set timeout
    session.timeout = 30
    
    return session

def download_apple_data_robust(period="1y", retry_count=3):
    """Robust Apple data download with multiple fallback strategies"""
    print(f"📈 Downloading Apple stock data (period: {period})...")
    
    # Setup environment
    setup_timezone_environment()
    
    strategies = [
        # Strategy 1: Standard yfinance with period
        {
            'name': 'Standard yfinance',
            'func': lambda: yf.Ticker("AAPL").history(period=period)
        },
        
        # Strategy 2: With explicit dates
        {
            'name': 'Explicit date range',
            'func': lambda: yf.Ticker("AAPL").history(
                start=datetime.now() - timedelta(days=365),
                end=datetime.now()
            )
        },
        
        # Strategy 3: Using download function
        {
            'name': 'yf.download method',
            'func': lambda: yf.download("AAPL", period=period, progress=False)
        },
        
        # Strategy 4: With timezone specification
        {
            'name': 'With timezone handling',
            'func': lambda: download_with_timezone()
        },
        
        # Strategy 5: Alternative ticker format
        {
            'name': 'Alternative format',
            'func': lambda: yf.Ticker("AAPL.O").history(period=period)  # Nasdaq format
        }
    ]
    
    for attempt in range(retry_count):
        print(f"\n🔄 Attempt {attempt + 1}/{retry_count}")
        
        for strategy in strategies:
            print(f"   🧪 Trying: {strategy['name']}")
            
            try:
                data = strategy['func']()
                
                if data is not None and not data.empty and len(data) > 10:
                    # Success! Clean up the data
                    data = clean_yfinance_data(data)
                    
                    print(f"   ✅ Success with {strategy['name']}!")
                    print(f"   📊 Downloaded {len(data)} rows of data")
                    print(f"   📅 Date range: {data.index[0]} to {data.index[-1]}")
                    print(f"   💰 Price range: ${data['Low'].min():.2f} - ${data['High'].max():.2f}")
                    
                    return data
                else:
                    print(f"   ❌ {strategy['name']}: No data returned")
                    
            except Exception as e:
                print(f"   ❌ {strategy['name']}: {str(e)[:80]}...")
                continue
        
        if attempt < retry_count - 1:
            print(f"   ⏳ Waiting 5 seconds before next attempt...")
            import time
            time.sleep(5)
    
    print(f"\n❌ All download strategies failed after {retry_count} attempts")
    return None

def download_with_timezone():
    """Download with explicit timezone handling"""
    # Create ticker with explicit timezone
    ticker = yf.Ticker("AAPL")
    
    # Get data and handle timezone
    data = ticker.history(period="1y", auto_adjust=True, prepost=True)
    
    if not data.empty:
        # Ensure timezone is properly set
        if data.index.tz is None:
            data.index = data.index.tz_localize('America/New_York')
        else:
            data.index = data.index.tz_convert('America/New_York')
    
    return data

def clean_yfinance_data(data):
    """Clean and standardize yfinance data"""
    print("🧹 Cleaning downloaded data...")
    
    # Handle timezone issues
    if hasattr(data.index, 'tz') and data.index.tz is not None:
        # Convert to naive datetime (remove timezone for consistency)
        data.index = data.index.tz_localize(None)
    
    # Standardize column names
    column_mapping = {
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume',
        'Adj Close': 'adj_close'
    }
    
    # Rename columns to lowercase
    for old_name, new_name in column_mapping.items():
        if old_name in data.columns:
            data = data.rename(columns={old_name: new_name})
    
    # Remove any NaN values
    data = data.dropna()
    
    # Ensure we have the basic OHLCV columns
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"   ⚠️ Missing columns: {missing_columns}")
        # Try alternative column names
        alt_mapping = {
            'Open': 'open', 'HIGH': 'high', 'LOW': 'low', 
            'CLOSE': 'close', 'VOLUME': 'volume'
        }
        for old, new in alt_mapping.items():
            if old in data.columns and new not in data.columns:
                data[new] = data[old]
    
    # Validate data quality
    if len(data) < 10:
        raise ValueError("Insufficient data rows")
    
    # Check for reasonable price ranges (Apple typically $100-$250)
    if data['close'].min() < 50 or data['close'].max() > 500:
        print(f"   ⚠️ Unusual price range detected: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    print(f"   ✅ Data cleaned successfully")
    return data

def test_download():
    """Test the robust download function"""
    print("🍎 TESTING ROBUST APPLE DATA DOWNLOAD")
    print("=" * 50)
    
    # Try different periods
    periods = ["1y", "6mo", "3mo"]
    
    for period in periods:
        print(f"\n📊 Testing period: {period}")
        data = download_apple_data_robust(period=period, retry_count=2)
        
        if data is not None:
            print(f"✅ Successfully downloaded {period} data!")
            print(f"   Rows: {len(data)}")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Sample prices: {data['close'].tail(3).values}")
            return data
        else:
            print(f"❌ Failed to download {period} data")
    
    print(f"\n❌ All download attempts failed")
    return None

def create_fallback_apple_data():
    """Create realistic Apple data as fallback"""
    print(f"\n🔄 Creating fallback Apple-like data...")
    
    # Use real Apple characteristics
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')  # 1 trading year
    
    # Apple stock characteristics (approximate)
    initial_price = 155.0
    annual_return = 0.15  # 15% typical annual return
    annual_volatility = 0.25  # 25% annual volatility
    
    daily_return = annual_return / 252
    daily_vol = annual_volatility / np.sqrt(252)
    
    # Generate price path
    np.random.seed(42)  # Reproducible
    returns = np.random.normal(daily_return, daily_vol, len(dates))
    
    # Add some realistic patterns
    # Earnings reactions (quarterly)
    earnings_days = [60, 120, 180, 240]  # Approximate quarterly earnings
    for day in earnings_days:
        if day < len(returns):
            returns[day] += np.random.choice([-0.04, 0.06], p=[0.3, 0.7])  # Earnings surprise
    
    # Calculate cumulative prices
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': [p * np.random.uniform(0.998, 1.002) for p in prices],
        'high': [p * np.random.uniform(1.005, 1.015) for p in prices],
        'low': [p * np.random.uniform(0.985, 0.995) for p in prices],
        'close': prices,
        'volume': [np.random.randint(40000000, 120000000) for _ in prices]  # Apple's volume range
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        o, h, l, c = data.iloc[i][['open', 'high', 'low', 'close']]
        data.iloc[i, data.columns.get_loc('low')] = min(o, h, l, c)
        data.iloc[i, data.columns.get_loc('high')] = max(o, h, l, c)
    
    print(f"   ✅ Created {len(data)} days of Apple-like data")
    print(f"   📈 Total return: {(data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100:.1f}%")
    
    return data

def main():
    """Main function to get Apple data"""
    print("🚀 ROBUST APPLE DATA ACQUISITION")
    print("=" * 50)
    
    # Try to download real data
    data = test_download()
    
    if data is not None:
        print(f"\n✅ SUCCESS: Real Apple data downloaded!")
        return data, True  # Real data
    else:
        print(f"\n🔄 Falling back to simulated Apple data...")
        data = create_fallback_apple_data()
        print(f"✅ Fallback data created successfully!")
        return data, False  # Simulated data

if __name__ == "__main__":
    apple_data, is_real = main()
    
    print(f"\n📊 FINAL RESULT:")
    print(f"   Data source: {'Real Apple stock' if is_real else 'Simulated Apple-like'}")
    print(f"   Rows: {len(apple_data)}")
    print(f"   Date range: {apple_data.index[0].strftime('%Y-%m-%d')} to {apple_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"   Price range: ${apple_data['low'].min():.2f} - ${apple_data['high'].max():.2f}")
    print(f"   Recent prices: {apple_data['close'].tail(3).values}")
    print(f"\n🎯 Ready for framework testing!")