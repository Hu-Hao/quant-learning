"""
Example Usage of Quant Trading Framework
Demonstrates the complete workflow with clean, modular code
"""

import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our framework
from quant_trading.data.data_fetcher import DataFetcher, DataSource
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.strategies.momentum import MomentumStrategy
from quant_trading.strategies.mean_reversion import MeanReversionStrategy
from quant_trading.backtesting.engine import BacktestEngine
from quant_trading.backtesting.metrics import MetricsCalculator
from quant_trading.config.settings import Config, get_default_config


def main():
    """Main execution function"""
    logger.info("🚀 Starting Quant Trading Framework Demo")
    
    # === 1. DATA PREPARATION ===
    logger.info("📊 Fetching market data...")
    
    data_fetcher = DataFetcher(DataSource.SYNTHETIC)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    
    try:
        data = data_fetcher.fetch_data(
            symbol="DEMO_STOCK",
            start_date=start_date,
            end_date=end_date,
            trend=0.08,      # 8% annual trend
            volatility=0.25, # 25% volatility
            seed=42          # Reproducible results
        )
        logger.info(f"✅ Fetched {len(data)} days of data")
        
    except Exception as e:
        logger.error(f"❌ Data fetching failed: {e}")
        return
    
    # === 2. STRATEGY SETUP ===
    logger.info("🧠 Setting up trading strategies...")
    
    strategies = {
        "Moving Average": MovingAverageStrategy(
            short_window=10,
            long_window=30,
            quantity=100
        ),
        "Momentum": MomentumStrategy(
            lookback_period=20,
            momentum_threshold=0.03,
            quantity=100
        ),
        "Mean Reversion": MeanReversionStrategy(
            window=20,
            entry_threshold=1.5,
            exit_threshold=0.5,
            quantity=100
        )
    }
    
    # === 3. BACKTESTING ===
    logger.info("⚡ Running backtests...")
    
    results = {}
    metrics_calc = MetricsCalculator(risk_free_rate=0.02)
    
    for name, strategy in strategies.items():
        logger.info(f"  Testing {name} strategy...")
        
        # Create backtest engine with realistic parameters
        engine = BacktestEngine(
            initial_capital=100000,
            commission=0.001,      # 0.1% commission
            slippage=0.0005,       # 0.05% slippage
            max_position_size=0.2,  # Max 20% position size
            risk_free_rate=0.02
        )
        
        try:
            # Run backtest
            engine.run_backtest(data, strategy)
            
            # Calculate performance metrics
            performance = engine.get_performance_summary()
            detailed_metrics = metrics_calc.calculate_performance_metrics(
                engine.portfolio_values
            )
            
            results[name] = {
                'engine': engine,
                'performance': performance,
                'metrics': detailed_metrics
            }
            
            logger.info(f"    ✅ {name}: {performance['total_return']:.2%} return, "
                       f"{performance['sharpe_ratio']:.3f} Sharpe")
            
        except Exception as e:
            logger.error(f"    ❌ {name} failed: {e}")
            continue
    
    # === 4. RESULTS ANALYSIS ===
    logger.info("📈 Analyzing results...")
    
    print("\n" + "="*80)
    print("STRATEGY COMPARISON RESULTS")
    print("="*80)
    
    # Print comparison table
    print(f"{'Strategy':<15} {'Return':<10} {'Sharpe':<8} {'MaxDD':<8} {'Trades':<8} {'Win Rate':<10}")
    print("-" * 80)
    
    for name, result in results.items():
        perf = result['performance']
        print(f"{name:<15} "
              f"{perf['total_return']:>8.2%} "
              f"{perf['sharpe_ratio']:>7.3f} "
              f"{perf['max_drawdown']:>7.2%} "
              f"{perf['total_trades']:>7d} "
              f"{perf['win_rate']:>9.1%}")
    
    # === 5. DETAILED ANALYSIS FOR BEST STRATEGY ===
    if results:
        # Find best strategy by Sharpe ratio
        best_strategy = max(results.items(), 
                           key=lambda x: x[1]['performance']['sharpe_ratio'])
        best_name, best_result = best_strategy
        
        print(f"\n📊 DETAILED ANALYSIS: {best_name}")
        print("="*50)
        
        # Generate detailed report
        report = metrics_calc.generate_report(
            best_result['engine'].portfolio_values,
            strategy_name=best_name
        )
        print(report)
        
        # Trade analysis
        trades = best_result['engine'].trades
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            print("TRADE ANALYSIS")
            print("-" * 30)
            print(f"Total Trades:        {len(trades)}")
            print(f"Winning Trades:      {len(winning_trades)} ({len(winning_trades)/len(trades):.1%})")
            print(f"Losing Trades:       {len(losing_trades)} ({len(losing_trades)/len(trades):.1%})")
            
            if winning_trades:
                avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades)
                print(f"Average Win:         ${avg_win:.2f}")
                
            if losing_trades:
                avg_loss = sum(abs(t.pnl) for t in losing_trades) / len(losing_trades)
                print(f"Average Loss:        ${avg_loss:.2f}")
                
                if avg_loss > 0:
                    profit_factor = avg_win / avg_loss if winning_trades else 0
                    print(f"Profit Factor:       {profit_factor:.2f}")
    
    # === 6. CONFIGURATION EXAMPLE ===
    logger.info("⚙️  Configuration management example...")
    
    try:
        # Load default config
        config = get_default_config("MovingAverage")
        
        # Modify parameters
        config.strategy.set_param("short_window", 8)
        config.strategy.set_param("long_window", 25)
        config.backtest.commission = 0.0015
        
        # Save configuration
        config.save("configs/custom_ma_config.yaml")
        logger.info("✅ Configuration saved")
        
        # Validate configuration
        config.validate()
        logger.info("✅ Configuration validated")
        
    except Exception as e:
        logger.error(f"❌ Configuration example failed: {e}")
    
    logger.info("🎯 Demo completed successfully!")


def run_unit_tests():
    """Run the test suite"""
    import unittest
    
    logger.info("🧪 Running unit tests...")
    
    # Discover and run tests
    loader = unittest.TestLoader()
    test_dir = 'quant_trading/tests'
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        logger.info("✅ All tests passed!")
    else:
        logger.error(f"❌ {len(result.failures + result.errors)} tests failed")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run unit tests first
    if run_unit_tests():
        # Run main demo if tests pass
        main()
    else:
        logger.error("Tests failed, skipping demo")