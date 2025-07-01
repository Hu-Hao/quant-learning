#!/usr/bin/env python3
"""
Debug why Apple stock data cannot be downloaded
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_yfinance_issues():
    """Test various yfinance download strategies"""
    print("🔍 DEBUGGING YFINANCE DATA FETCH ISSUES")
    print("=" * 50)
    
    # Common reasons for yfinance failures:
    print("📋 Common yfinance Issues:")
    print("   1. Rate limiting from Yahoo Finance")
    print("   2. Network connectivity issues")
    print("   3. Yahoo Finance API changes")
    print("   4. Temporary server outages")
    print("   5. User-agent blocking")
    print("   6. Geographic restrictions")
    
    strategies = [
        ("Standard approach", lambda: yf.Ticker("AAPL").history(period="1y")),
        ("With dates", lambda: yf.Ticker("AAPL").history(start="2023-01-01", end="2024-01-01")),
        ("Shorter period", lambda: yf.Ticker("AAPL").history(period="6mo")),
        ("Different method", lambda: yf.download("AAPL", period="1y")),
        ("With headers", lambda: test_with_headers()),
    ]
    
    for name, strategy in strategies:
        print(f"\n🧪 Testing: {name}")
        try:
            data = strategy()
            if data is None or data.empty:
                print(f"   ❌ No data returned")
            else:
                print(f"   ✅ Success! Got {len(data)} rows")
                print(f"   📅 Date range: {data.index[0]} to {data.index[-1]}")
                return data
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:100]}...")
    
    return None

def test_with_headers():
    """Test with custom headers"""
    import yfinance as yf
    
    # Sometimes adding headers helps
    session = yf.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    ticker = yf.Ticker("AAPL", session=session)
    return ticker.history(period="1y")

def create_realistic_apple_data(days=365):
    """Create realistic Apple-like data when yfinance fails"""
    print(f"\n📊 Creating realistic Apple-like sample data...")
    
    # Based on Apple's historical characteristics
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Apple typically:
    # - Starts around $150-180
    # - Has 10-20% annual growth trend
    # - Has 20-30% annual volatility
    # - Shows occasional strong moves
    
    np.random.seed(42)  # For reproducibility
    
    # Generate price path
    initial_price = 155.0
    daily_drift = 0.15 / 252  # 15% annual growth
    daily_vol = 0.25 / np.sqrt(252)  # 25% annual volatility
    
    # Generate random returns
    returns = np.random.normal(daily_drift, daily_vol, days)
    
    # Add some Apple-like patterns
    # Occasional earnings jumps
    earnings_days = np.random.choice(days, size=4, replace=False)
    for day in earnings_days:
        returns[day] += np.random.choice([-0.05, 0.05], p=[0.3, 0.7])  # Earnings surprises
    
    # Tech stock volatility clusters
    for i in range(1, days):
        if abs(returns[i-1]) > 0.03:  # If previous day was volatile
            returns[i] *= 1.5  # Increase today's volatility
    
    # Calculate prices
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': [p * np.random.uniform(0.995, 1.005) for p in prices],
        'high': [p * np.random.uniform(1.001, 1.02) for p in prices],
        'low': [p * np.random.uniform(0.98, 0.999) for p in prices],
        'close': prices,
        'volume': [np.random.randint(50000000, 150000000) for _ in prices]  # Apple's typical volume
    }, index=dates)
    
    # Ensure OHLC consistency
    for i in range(len(data)):
        low_val = min(data.iloc[i]['open'], data.iloc[i]['close'], data.iloc[i]['low'])
        high_val = max(data.iloc[i]['open'], data.iloc[i]['close'], data.iloc[i]['high'])
        data.iloc[i, data.columns.get_loc('low')] = low_val
        data.iloc[i, data.columns.get_loc('high')] = high_val
    
    print(f"   ✅ Created {len(data)} days of Apple-like data")
    print(f"   📅 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"   💰 Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
    print(f"   📈 Total return: {(data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100:.1f}%")
    
    return data

def main():
    """Main debugging function"""
    import numpy as np
    
    print("🔬 YFINANCE DEBUGGING AND FALLBACK CREATION")
    print("=" * 60)
    
    # Try to fetch real data
    real_data = test_yfinance_issues()
    
    if real_data is not None:
        print(f"\n✅ Successfully fetched real Apple data!")
        return real_data
    else:
        print(f"\n⚠️ All yfinance strategies failed. Common solutions:")
        print(f"   1. Wait and try again later (rate limiting)")
        print(f"   2. Use a VPN if geographically blocked")
        print(f"   3. Update yfinance: pip install --upgrade yfinance")
        print(f"   4. Use alternative data sources (Alpha Vantage, Quandl, etc.)")
        print(f"   5. Use cached data or sample data for testing")
        
        print(f"\n🔄 Falling back to realistic sample data...")
        sample_data = create_realistic_apple_data(365)
        
        print(f"\n💡 This sample data has Apple-like characteristics:")
        print(f"   • Similar price levels and volatility")
        print(f"   • Realistic trading patterns")
        print(f"   • Earnings-like jumps")
        print(f"   • Tech stock volatility clustering")
        print(f"   • Perfect for testing framework differences")
        
        return sample_data

if __name__ == "__main__":
    data = main()
    
    print(f"\n📊 DATA SUMMARY:")
    print(f"   Rows: {len(data)}")
    print(f"   Columns: {list(data.columns)}")
    print(f"   Sample prices: {data['close'].head(3).values}")
    print(f"   Ready for framework testing!")