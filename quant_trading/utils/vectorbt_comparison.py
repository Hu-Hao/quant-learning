"""
VectorBT Cross-Validation Utilities

Helper functions to compare strategies between our framework and VectorBT
for validation and performance benchmarking.

Uses generic signals_to_vectorbt() function which works with ANY strategy
that implements get_signals() correctly - no strategy-specific logic needed.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings

from ..strategies.strategy_interface import StrategyProtocol
from ..backtesting.engine import BacktestEngine


def compare_with_vectorbt(
    data: pd.DataFrame,
    strategy: StrategyProtocol,
    initial_capital: float = 100000,
    commission: float = 0.001,
    show_details: bool = True
) -> Dict[str, Any]:
    """
    Compare strategy performance between our framework and VectorBT
    
    Args:
        data: Market data (OHLCV with datetime index)
        strategy: Strategy implementing StrategyProtocol
        initial_capital: Starting capital
        commission: Trading commission
        show_details: Whether to print comparison details
        
    Returns:
        Dictionary with comparison results
    """
    if show_details:
        print("🔄 Framework Cross-Validation: Our Framework vs VectorBT")
        print("=" * 60)
        print(f"Strategy: {strategy.get_name()}")
        print(f"Data period: {len(data)} days")
        print(f"Initial capital: ${initial_capital:,}")
        print(f"Commission: {commission:.3f}")
    
    # Run on our framework
    if show_details:
        print("\n📊 Running on Our Framework...")
    
    our_results = _run_our_framework(data, strategy, initial_capital, commission)
    
    if show_details and our_results['success']:
        our_perf = our_results['performance']
        print(f"   ✅ Complete! Return: {our_perf.get('total_return', 0)*100:.2f}%, "
              f"Trades: {our_perf.get('total_trades', 0)}")
    
    # Run on VectorBT
    if show_details:
        print("\n⚡ Running on VectorBT...")
    
    vbt_results = _run_vectorbt_framework(data, strategy, initial_capital, commission)
    
    if show_details:
        if vbt_results['success']:
            vbt_perf = vbt_results['performance']
            print(f"   ✅ Complete! Return: {vbt_perf.get('total_return', 0)*100:.2f}%, "
                  f"Trades: {vbt_perf.get('total_trades', 0)}")
        else:
            print(f"   ⚠️  {vbt_results.get('error', 'Failed')}")
    
    # Compare results
    comparison = _analyze_comparison(our_results, vbt_results)
    
    if show_details:
        _print_comparison_summary(comparison, our_results, vbt_results)
    
    return {
        'our_results': our_results,
        'vectorbt_results': vbt_results,
        'comparison': comparison
    }


def quick_vectorbt_validation(
    data: pd.DataFrame,
    strategy: StrategyProtocol,
    **kwargs
) -> bool:
    """
    Quick validation check - returns True if frameworks agree well
    
    Args:
        data: Market data
        strategy: Strategy to compare
        **kwargs: Additional parameters
        
    Returns:
        True if frameworks agree, False if significant differences
    """
    try:
        results = compare_with_vectorbt(data, strategy, show_details=False, **kwargs)
        comparison = results['comparison']
        
        # Check if results are similar
        return_diff = abs(comparison.get('return_difference', 1.0))
        correlation = comparison.get('correlation', 0.0)
        
        # Good agreement: <2% return difference and >0.8 correlation
        return return_diff < 0.02 and correlation > 0.8
        
    except Exception:
        return False


def create_vectorbt_comparison_plots(
    data: pd.DataFrame,
    strategy: StrategyProtocol,
    initial_capital: float = 100000,
    show_technical_indicators: bool = True
):
    """
    Create side-by-side comparison plots between our framework and VectorBT
    
    Args:
        data: Market data
        strategy: Strategy to compare
        initial_capital: Starting capital
        show_technical_indicators: Whether to show technical indicators
        
    Returns:
        Matplotlib figure with comparison plots
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available for plotting")
        return None
    
    # Run comparison
    results = compare_with_vectorbt(data, strategy, initial_capital, show_details=False)
    our_results = results['our_results']
    vbt_results = results['vectorbt_results']
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Framework Comparison: {strategy.get_name()}', fontsize=14, fontweight='bold')
    
    # 1. Price chart with technical indicators
    ax1 = axes[0, 0]
    ax1.plot(data.index, data['close'], label='Price', color='black', linewidth=1)
    
    if show_technical_indicators:
        indicators = strategy.get_technical_indicators(data)
        for name, series in indicators.items():
            ax1.plot(data.index, series, label=name, alpha=0.7, linewidth=1)
    
    ax1.set_title('Price Chart with Indicators')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Portfolio value comparison
    ax2 = axes[0, 1]
    
    if our_results['success'] and 'portfolio_values' in our_results:
        our_values = our_results['portfolio_values']
        ax2.plot(data.index[:len(our_values)], our_values, 
                label='Our Framework', color='blue', linewidth=2)
    
    if vbt_results['success'] and 'portfolio_values' in vbt_results:
        vbt_values = vbt_results['portfolio_values']
        ax2.plot(data.index[:len(vbt_values)], vbt_values, 
                label='VectorBT', color='red', linewidth=2, linestyle='--')
    
    # Buy and hold benchmark
    buy_hold = initial_capital * (data['close'] / data['close'].iloc[0])
    ax2.plot(data.index, buy_hold, label='Buy & Hold', 
            color='gray', linestyle=':', alpha=0.7)
    
    ax2.set_title('Portfolio Performance Comparison')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance metrics comparison
    ax3 = axes[1, 0]
    
    metrics = ['Total Return (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 'Total Trades']
    our_values_list = []
    vbt_values_list = []
    
    if our_results['success']:
        our_perf = our_results['performance']
        our_values_list = [
            our_perf.get('total_return', 0) * 100,
            our_perf.get('max_drawdown', 0) * 100,
            our_perf.get('sharpe_ratio', 0),
            our_perf.get('total_trades', 0)
        ]
    else:
        our_values_list = [0, 0, 0, 0]
    
    if vbt_results['success']:
        vbt_perf = vbt_results['performance']
        vbt_values_list = [
            vbt_perf.get('total_return', 0) * 100,
            vbt_perf.get('max_drawdown', 0) * 100,
            vbt_perf.get('sharpe_ratio', 0),
            vbt_perf.get('total_trades', 0)
        ]
    else:
        vbt_values_list = [0, 0, 0, 0]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, our_values_list, width, label='Our Framework', alpha=0.7, color='blue')
    ax3.bar(x + width/2, vbt_values_list, width, label='VectorBT', alpha=0.7, color='red')
    
    ax3.set_title('Performance Metrics Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Framework status and speed comparison
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create status text
    our_time = our_results.get('execution_time', 0)
    vbt_time = vbt_results.get('execution_time', 0)
    speed_ratio = our_time / vbt_time if vbt_time > 0 else 0
    
    status_text = f"""Framework Comparison Results:

Our Framework:
✅ Success: {our_results['success']}
📊 Return: {our_results['performance'].get('total_return', 0)*100:.2f}%
🔢 Trades: {our_results['performance'].get('total_trades', 0)}
⏱️ Time: {our_time:.3f}s

VectorBT:
{'✅' if vbt_results['success'] else '❌'} Success: {vbt_results['success']}
📊 Return: {vbt_results['performance'].get('total_return', 0)*100:.2f}%
🔢 Trades: {vbt_results['performance'].get('total_trades', 0)}
⏱️ Time: {vbt_time:.3f}s

⚡ Speed Ratio: {speed_ratio:.1f}x
"""
    
    if not vbt_results['success']:
        status_text += f"\nVectorBT Error: {vbt_results.get('error', 'Unknown')}"
    
    ax4.text(0.1, 0.9, status_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig


def _run_our_framework(
    data: pd.DataFrame,
    strategy: StrategyProtocol,
    initial_capital: float,
    commission: float
) -> Dict[str, Any]:
    """Run backtest using our framework"""
    import time
    
    try:
        start_time = time.time()
        
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.001,
            allow_short_selling=False,  # Match VectorBT's longonly
            max_position_size=0.95
        )
        
        engine.run_backtest(data, strategy)
        performance = engine.get_performance_summary()
        
        execution_time = time.time() - start_time
        
        return {
            'success': True,
            'performance': performance,
            'portfolio_values': engine.portfolio_values.copy(),
            'trades': engine.trades.copy(),
            'execution_time': execution_time,
            'engine': engine
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'performance': {},
            'portfolio_values': [],
            'trades': [],
            'execution_time': 0
        }


def _run_vectorbt_framework(
    data: pd.DataFrame,
    strategy: StrategyProtocol,
    initial_capital: float,
    commission: float
) -> Dict[str, Any]:
    """Run backtest using VectorBT framework"""
    import time
    
    try:
        import vectorbt as vbt
    except ImportError:
        return {
            'success': False,
            'error': 'VectorBT not installed. Install with: pip install vectorbt',
            'performance': {},
            'portfolio_values': [],
            'execution_time': 0
        }
    
    try:
        start_time = time.time()
        
        # Generate signals using our strategy
        entries, exits = strategy.generate_vectorbt_signals(data)
        
        # Run VectorBT backtest with longonly direction
        pf = vbt.Portfolio.from_signals(
            data['close'],
            entries,
            exits,
            direction='longonly',  # Disable short selling
            init_cash=initial_capital,
            fees=commission,
            freq='1D'
        )
        
        execution_time = time.time() - start_time
        
        # Extract performance metrics
        performance = {
            'total_return': pf.total_return if not np.isnan(pf.total_return) else 0,
            'max_drawdown': pf.max_drawdown if not np.isnan(pf.max_drawdown) else 0,
            'sharpe_ratio': pf.sharpe_ratio if not np.isnan(pf.sharpe_ratio) else 0,
            'total_trades': pf.trades.count if hasattr(pf.trades, 'count') else 0,
            'win_rate': pf.trades.win_rate if hasattr(pf.trades, 'win_rate') and not np.isnan(pf.trades.win_rate) else 0,
            'annualized_return': pf.annualized_return if not np.isnan(pf.annualized_return) else 0,
            'volatility': pf.annualized_volatility if not np.isnan(pf.annualized_volatility) else 0
        }
        
        return {
            'success': True,
            'performance': performance,
            'portfolio_values': pf.value.values.tolist(),
            'execution_time': execution_time,
            'pf': pf
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'VectorBT error: {str(e)}',
            'performance': {},
            'portfolio_values': [],
            'execution_time': 0
        }


def _analyze_comparison(our_results: Dict, vbt_results: Dict) -> Dict[str, Any]:
    """Analyze comparison between frameworks"""
    if not our_results['success'] or not vbt_results['success']:
        return {
            'status': 'INCOMPLETE',
            'return_difference': 0.0,
            'correlation': 0.0,
            'speed_ratio': 0.0,
            'insights': ['One or both frameworks failed to run']
        }
    
    our_perf = our_results['performance']
    vbt_perf = vbt_results['performance']
    
    # Calculate differences
    return_diff = our_perf.get('total_return', 0) - vbt_perf.get('total_return', 0)
    sharpe_diff = our_perf.get('sharpe_ratio', 0) - vbt_perf.get('sharpe_ratio', 0)
    trade_diff = our_perf.get('total_trades', 0) - vbt_perf.get('total_trades', 0)
    
    # Calculate speed ratio
    our_time = our_results.get('execution_time', 1)
    vbt_time = vbt_results.get('execution_time', 1)
    speed_ratio = our_time / vbt_time if vbt_time > 0 else 0
    
    # Calculate portfolio value correlation
    correlation = 0.0
    our_portfolio = our_results.get('portfolio_values', [])
    vbt_portfolio = vbt_results.get('portfolio_values', [])
    
    if our_portfolio and vbt_portfolio:
        min_len = min(len(our_portfolio), len(vbt_portfolio))
        if min_len > 1:
            our_values = np.array(our_portfolio[:min_len])
            vbt_values = np.array(vbt_portfolio[:min_len])
            correlation = np.corrcoef(our_values, vbt_values)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
    
    # Generate insights
    insights = []
    
    if abs(return_diff) < 0.01:  # Less than 1% difference
        insights.append("Returns are very similar between frameworks")
    elif abs(return_diff) > 0.05:  # More than 5% difference
        insights.append("Significant return difference - investigate implementation")
    
    if abs(trade_diff) > 2:
        insights.append("Different number of trades executed - check signal timing")
    
    if speed_ratio > 2:
        insights.append(f"Our framework is {speed_ratio:.1f}x slower than VectorBT")
    elif speed_ratio < 0.5:
        insights.append(f"Our framework is {1/speed_ratio:.1f}x faster than VectorBT")
    
    if correlation > 0.9:
        insights.append("Excellent portfolio value correlation")
    elif correlation < 0.7:
        insights.append("Low portfolio correlation - check signal differences")
    
    # Determine status
    if abs(return_diff) < 0.02 and abs(trade_diff) <= 2 and correlation > 0.8:
        status = 'PASS'
    elif abs(return_diff) < 0.05:
        status = 'REVIEW'
    else:
        status = 'INVESTIGATE'
    
    return {
        'status': status,
        'return_difference': return_diff,
        'sharpe_difference': sharpe_diff,
        'trade_difference': trade_diff,
        'correlation': correlation,
        'speed_ratio': speed_ratio,
        'insights': insights
    }


def _print_comparison_summary(comparison: Dict, our_results: Dict, vbt_results: Dict):
    """Print detailed comparison summary"""
    print(f"\n📊 CROSS-VALIDATION RESULTS")
    print("=" * 50)
    
    if not our_results['success']:
        print(f"❌ Our framework failed: {our_results.get('error', 'Unknown error')}")
        return
    
    if not vbt_results['success']:
        print(f"❌ VectorBT failed: {vbt_results.get('error', 'Unknown error')}")
        return
    
    # Performance comparison
    our_perf = our_results['performance']
    vbt_perf = vbt_results['performance']
    
    print(f"📈 Performance Comparison:")
    print(f"{'Metric':<20} {'Our Framework':<15} {'VectorBT':<15} {'Difference':<15}")
    print("-" * 65)
    
    our_ret = our_perf.get('total_return', 0) * 100
    vbt_ret = vbt_perf.get('total_return', 0) * 100
    ret_diff = our_ret - vbt_ret
    print(f"{'Total Return (%)':<20} {our_ret:<15.2f} {vbt_ret:<15.2f} {ret_diff:+.2f}")
    
    our_trades = our_perf.get('total_trades', 0)
    vbt_trades = vbt_perf.get('total_trades', 0)
    trade_diff = our_trades - vbt_trades
    print(f"{'Total Trades':<20} {our_trades:<15} {vbt_trades:<15} {trade_diff:+}")
    
    our_sharpe = our_perf.get('sharpe_ratio', 0)
    vbt_sharpe = vbt_perf.get('sharpe_ratio', 0)
    sharpe_diff = our_sharpe - vbt_sharpe
    print(f"{'Sharpe Ratio':<20} {our_sharpe:<15.3f} {vbt_sharpe:<15.3f} {sharpe_diff:+.3f}")
    
    # Speed comparison
    our_time = our_results.get('execution_time', 0)
    vbt_time = vbt_results.get('execution_time', 0)
    print(f"{'Execution Time (s)':<20} {our_time:<15.3f} {vbt_time:<15.3f} {our_time-vbt_time:+.3f}")
    
    # Status and insights
    print(f"\n✅ Status: {comparison['status']}")
    print(f"📊 Portfolio Correlation: {comparison['correlation']:.3f}")
    print(f"⚡ Speed Ratio: {comparison['speed_ratio']:.1f}x")
    
    if comparison['insights']:
        print(f"\n💡 Insights:")
        for insight in comparison['insights']:
            print(f"   • {insight}")
    
    # Recommendations
    print(f"\n🎯 Recommendations:")
    if comparison['status'] == 'PASS':
        print("   ✅ Frameworks show excellent agreement")
        print("   • Strategy implementation validated")
        print("   • Results can be trusted")
        print("   • VectorBT provides speed advantage for optimization")
    elif comparison['status'] == 'REVIEW':
        print("   ⚠️  Minor differences detected")
        print("   • Review signal generation timing")
        print("   • Check order execution details")
        print("   • Consider using VectorBT for parameter optimization")
    else:
        print("   🔍 Significant differences found")
        print("   • Investigate strategy implementation")
        print("   • Check framework configuration")
        print("   • Verify data consistency")
        print("   • Review crossover detection logic")