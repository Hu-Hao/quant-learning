#!/usr/bin/env python3
"""
Installation Test Script for Quantitative Trading Framework

This script verifies that the framework is properly installed and all
components are working correctly.

Usage:
    python test_installation.py
"""

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from quant_trading.strategies.moving_average import MovingAverageStrategy
        print("✅ MovingAverageStrategy imported")
        
        from quant_trading.strategies.momentum import MomentumStrategy
        print("✅ MomentumStrategy imported")
        
        from quant_trading.strategies.mean_reversion import MeanReversionStrategy
        print("✅ MeanReversionStrategy imported")
        
        from quant_trading.backtesting.engine import BacktestEngine
        print("✅ BacktestEngine imported")
        
        from quant_trading.data.data_fetcher import create_sample_data
        print("✅ Data fetcher imported")
        
        from quant_trading.utils.visualization import PerformanceVisualizer
        print("✅ PerformanceVisualizer imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_strategy_creation():
    """Test strategy creation and protocol interface"""
    print("\n🏗️ Testing strategy creation...")
    
    try:
        from quant_trading.strategies.moving_average import MovingAverageStrategy
        from quant_trading.strategies.momentum import MomentumStrategy
        from quant_trading.strategies.mean_reversion import MeanReversionStrategy
        
        # Test Moving Average Strategy
        ma_strategy = MovingAverageStrategy(short_window=10, long_window=20)
        assert ma_strategy.get_name() == "MovingAverage"
        assert isinstance(ma_strategy.get_parameters(), dict)
        print("✅ MovingAverageStrategy creation and protocol")
        
        # Test Momentum Strategy
        momentum_strategy = MomentumStrategy(lookback_period=15)
        assert momentum_strategy.get_name() == "Momentum"
        assert isinstance(momentum_strategy.get_parameters(), dict)
        print("✅ MomentumStrategy creation and protocol")
        
        # Test Mean Reversion Strategy
        mr_strategy = MeanReversionStrategy(window=20)
        assert mr_strategy.get_name() == "MeanReversion"
        assert isinstance(mr_strategy.get_parameters(), dict)
        print("✅ MeanReversionStrategy creation and protocol")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy creation failed: {e}")
        return False


def test_data_generation():
    """Test synthetic data generation"""
    print("\n📊 Testing data generation...")
    
    try:
        from quant_trading.data.data_fetcher import create_sample_data
        
        # Test basic data creation
        data = create_sample_data(days=50, seed=42)
        assert len(data) >= 50
        assert all(col in data.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        print("✅ Sample data generation")
        
        # Test reproducibility
        data1 = create_sample_data(days=30, seed=123)
        data2 = create_sample_data(days=30, seed=123)
        assert data1['close'].equals(data2['close'])
        print("✅ Data reproducibility")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return False


def test_backtesting():
    """Test backtesting engine"""
    print("\n🚀 Testing backtesting engine...")
    
    try:
        from quant_trading.strategies.moving_average import MovingAverageStrategy
        from quant_trading.backtesting.engine import BacktestEngine
        from quant_trading.data.data_fetcher import create_sample_data
        
        # Create test data and strategy
        data = create_sample_data(days=50, trend=0.1, seed=42)
        strategy = MovingAverageStrategy(short_window=5, long_window=15)
        
        # Test backtesting with StrategyProtocol
        engine = BacktestEngine(initial_capital=100000)
        engine.run_backtest(data, strategy)
        
        assert len(engine.portfolio_values) > 0
        assert engine.initial_capital == 100000
        print("✅ Backtesting execution with StrategyProtocol")
        
        # Test performance calculation
        performance = engine.get_performance_summary()
        assert isinstance(performance, dict)
        assert 'total_return' in performance
        print("✅ Performance metrics calculation")
        
        return True
        
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")
        return False


def test_signal_generation():
    """Test signal generation"""
    print("\n📡 Testing signal generation...")
    
    try:
        from quant_trading.strategies.moving_average import MovingAverageStrategy
        from quant_trading.data.data_fetcher import create_sample_data
        
        # Create trending data to generate signals
        data = create_sample_data(days=50, trend=0.2, seed=42)
        strategy = MovingAverageStrategy(short_window=5, long_window=10)
        
        # Test signal generation
        signals = strategy.get_signals(data)
        assert isinstance(signals, list)
        print("✅ Signal generation")
        
        # Test signal format if any signals generated
        if signals:
            signal = signals[0]
            assert hasattr(signal, 'symbol')
            assert hasattr(signal, 'action')
            assert hasattr(signal, 'quantity')
            assert hasattr(signal, 'price')
            print("✅ Signal format validation")
        else:
            print("✅ No signals generated (this is normal)")
            
        return True
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False


def test_optional_dependencies():
    """Test optional dependencies"""
    print("\n📦 Testing optional dependencies...")
    
    optional_packages = {
        'matplotlib': 'Plotting and visualization',
        'yfinance': 'Yahoo Finance data fetching',
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing'
    }
    
    for package, description in optional_packages.items():
        try:
            __import__(package)
            print(f"✅ {package} available ({description})")
        except ImportError:
            print(f"⚠️  {package} not available ({description})")
    
    return True


def main():
    """Run all tests"""
    print("🧪 QUANTITATIVE TRADING FRAMEWORK - INSTALLATION TEST")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Strategy Creation Test", test_strategy_creation),
        ("Data Generation Test", test_data_generation),
        ("Backtesting Test", test_backtesting),
        ("Signal Generation Test", test_signal_generation),
        ("Optional Dependencies", test_optional_dependencies)
    ]
    
    passed = 0
    total = len(tests) - 1  # Don't count optional dependencies as pass/fail
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            result = test_func()
            if test_name != "Optional Dependencies" and result:
                passed += 1
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Framework is ready to use.")
        print("\n🚀 Next steps:")
        print("   • Try the examples/colab_example.ipynb notebook")
        print("   • Run examples/yahoo_backtest_example.py")
        print("   • Explore the strategy implementations")
        print("   • Check out the documentation in the README")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        print("\n🔧 Troubleshooting:")
        print("   • Make sure you're in the correct directory")
        print("   • Verify the framework is properly installed")
        print("   • Check that all required dependencies are installed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)