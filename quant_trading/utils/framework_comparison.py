"""
Framework Comparison Utilities

Helper functions to easily compare strategies between our framework and Backtrader
for validation and learning from industry standards.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings

from ..strategies.strategy_interface import StrategyProtocol, run_backtrader_comparison
from ..backtesting.engine import BacktestEngine


def compare_frameworks(
    data: pd.DataFrame,
    strategy: StrategyProtocol,
    initial_capital: float = 100000,
    commission: float = 0.001,
    show_details: bool = True
) -> Dict[str, Any]:
    """
    Compare strategy performance between our framework and Backtrader
    
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
        print("🔄 Framework Comparison")
        print("=" * 40)
        print(f"Strategy: {strategy.get_name()}")
        print(f"Data period: {len(data)} days")
        print(f"Initial capital: ${initial_capital:,}")
    
    # Run on our framework
    if show_details:
        print("\n📊 Running on Our Framework...")
    
    our_results = _run_our_framework(data, strategy, initial_capital, commission)
    
    if show_details and our_results['success']:
        our_perf = our_results['performance']
        print(f"   ✅ Complete! Return: {our_perf.get('total_return', 0)*100:.2f}%, "
              f"Trades: {our_perf.get('total_trades', 0)}")
    
    # Run on Backtrader
    if show_details:
        print("\n📈 Running on Backtrader...")
    
    bt_results = run_backtrader_comparison(data, strategy, initial_capital, commission)
    
    if show_details:
        if bt_results['success']:
            bt_perf = bt_results['performance']
            print(f"   ✅ Complete! Return: {bt_perf.get('total_return', 0)*100:.2f}%, "
                  f"Trades: {bt_perf.get('total_trades', 0)}")
        else:
            print(f"   ⚠️  {bt_results.get('error', 'Failed')}")
    
    # Compare results
    comparison = _analyze_comparison(our_results, bt_results)
    
    if show_details:
        _print_comparison_summary(comparison, our_results, bt_results)
    
    return {
        'our_results': our_results,
        'backtrader_results': bt_results,
        'comparison': comparison
    }


def quick_comparison(
    data: pd.DataFrame,
    strategy: StrategyProtocol,
    **kwargs
) -> bool:
    """
    Quick comparison check - returns True if frameworks agree well
    
    Args:
        data: Market data
        strategy: Strategy to compare
        **kwargs: Additional parameters
        
    Returns:
        True if frameworks agree, False if significant differences
    """
    try:
        results = compare_frameworks(data, strategy, show_details=False, **kwargs)
        comparison = results['comparison']
        
        # Check if results are similar
        return_diff = abs(comparison.get('return_difference', 1.0))
        correlation = comparison.get('correlation', 0.0)
        
        # Good agreement: <2% return difference and >0.7 correlation
        return return_diff < 0.02 and correlation > 0.7
        
    except Exception:
        return False


def create_comparison_plots(
    data: pd.DataFrame,
    strategy: StrategyProtocol,
    initial_capital: float = 100000,
    show_technical_indicators: bool = True
):
    """
    Create side-by-side comparison plots
    
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
    results = compare_frameworks(data, strategy, initial_capital, show_details=False)
    our_results = results['our_results']
    bt_results = results['backtrader_results']
    
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
    
    # Note: Backtrader portfolio values would need to be extracted differently
    # For now, we'll show the final values as comparison
    
    # Buy and hold benchmark
    buy_hold = initial_capital * (data['close'] / data['close'].iloc[0])
    ax2.plot(data.index, buy_hold, label='Buy & Hold', 
            color='gray', linestyle='--', alpha=0.7)
    
    ax2.set_title('Portfolio Performance')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance metrics comparison
    ax3 = axes[1, 0]
    
    metrics = ['Total Return (%)', 'Max Drawdown (%)', 'Sharpe Ratio', 'Total Trades']
    our_values_list = []
    bt_values_list = []
    
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
    
    if bt_results['success']:
        bt_perf = bt_results['performance']
        bt_values_list = [
            bt_perf.get('total_return', 0) * 100,
            bt_perf.get('max_drawdown', 0) * 100,
            bt_perf.get('sharpe_ratio', 0),
            bt_perf.get('total_trades', 0)
        ]
    else:
        bt_values_list = [0, 0, 0, 0]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, our_values_list, width, label='Our Framework', alpha=0.7)
    ax3.bar(x + width/2, bt_values_list, width, label='Backtrader', alpha=0.7)
    
    ax3.set_title('Performance Metrics Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Framework status
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create status text
    status_text = f"""Framework Comparison Status:

Our Framework:
✅ Success: {our_results['success']}
📊 Return: {our_results['performance'].get('total_return', 0)*100:.2f}%
🔢 Trades: {our_results['performance'].get('total_trades', 0)}

Backtrader:
{'✅' if bt_results['success'] else '❌'} Success: {bt_results['success']}
📊 Return: {bt_results['performance'].get('total_return', 0)*100:.2f}%
🔢 Trades: {bt_results['performance'].get('total_trades', 0)}
"""
    
    if not bt_results['success']:
        status_text += f"\nError: {bt_results.get('error', 'Unknown')}"
    
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
    try:
        engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            slippage=0.001,
            allow_short_selling=False,  # Match beginner mode
            max_position_size=0.95
        )
        
        engine.run_backtest(data, strategy)
        performance = engine.get_performance_summary()
        
        return {
            'success': True,
            'performance': performance,
            'portfolio_values': engine.portfolio_values.copy(),
            'trades': engine.trades.copy(),
            'engine': engine
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'performance': {},
            'portfolio_values': [],
            'trades': []
        }


def _analyze_comparison(our_results: Dict, bt_results: Dict) -> Dict[str, Any]:
    """Analyze comparison between frameworks"""
    if not our_results['success'] or not bt_results['success']:
        return {
            'status': 'INCOMPLETE',
            'return_difference': 0.0,
            'correlation': 0.0,
            'insights': ['One or both frameworks failed to run']
        }
    
    our_perf = our_results['performance']
    bt_perf = bt_results['performance']
    
    # Calculate differences
    return_diff = our_perf.get('total_return', 0) - bt_perf.get('total_return', 0)
    sharpe_diff = our_perf.get('sharpe_ratio', 0) - bt_perf.get('sharpe_ratio', 0)
    trade_diff = our_perf.get('total_trades', 0) - bt_perf.get('total_trades', 0)
    
    # Calculate correlation if possible
    correlation = 0.0
    our_portfolio = our_results.get('portfolio_values', [])
    # Note: Would need Backtrader portfolio values for real correlation
    
    # Generate insights
    insights = []
    
    if abs(return_diff) < 0.01:  # Less than 1% difference
        insights.append("Returns are very similar between frameworks")
    elif abs(return_diff) > 0.05:  # More than 5% difference
        insights.append("Significant return difference - investigate implementation")
    
    if abs(trade_diff) > 2:
        insights.append("Different number of trades executed - check signal timing")
    
    # Determine status
    if abs(return_diff) < 0.02 and abs(trade_diff) <= 2:
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
        'insights': insights
    }


def _print_comparison_summary(comparison: Dict, our_results: Dict, bt_results: Dict):
    """Print detailed comparison summary"""
    print(f"\n📊 COMPARISON RESULTS")
    print("=" * 40)
    
    if not our_results['success']:
        print(f"❌ Our framework failed: {our_results.get('error', 'Unknown error')}")
        return
    
    if not bt_results['success']:
        print(f"❌ Backtrader failed: {bt_results.get('error', 'Unknown error')}")
        return
    
    # Performance comparison
    our_perf = our_results['performance']
    bt_perf = bt_results['performance']
    
    print(f"📈 Performance Comparison:")
    print(f"{'Metric':<20} {'Our Framework':<15} {'Backtrader':<15} {'Difference':<15}")
    print("-" * 65)
    
    our_ret = our_perf.get('total_return', 0) * 100
    bt_ret = bt_perf.get('total_return', 0) * 100
    ret_diff = our_ret - bt_ret
    print(f"{'Total Return (%)':<20} {our_ret:<15.2f} {bt_ret:<15.2f} {ret_diff:+.2f}")
    
    our_trades = our_perf.get('total_trades', 0)
    bt_trades = bt_perf.get('total_trades', 0)
    trade_diff = our_trades - bt_trades
    print(f"{'Total Trades':<20} {our_trades:<15} {bt_trades:<15} {trade_diff:+}")
    
    our_sharpe = our_perf.get('sharpe_ratio', 0)
    bt_sharpe = bt_perf.get('sharpe_ratio', 0)
    sharpe_diff = our_sharpe - bt_sharpe
    print(f"{'Sharpe Ratio':<20} {our_sharpe:<15.3f} {bt_sharpe:<15.3f} {sharpe_diff:+.3f}")
    
    # Status and insights
    print(f"\n✅ Status: {comparison['status']}")
    
    if comparison['insights']:
        print(f"\n💡 Insights:")
        for insight in comparison['insights']:
            print(f"   • {insight}")
    
    # Recommendations
    print(f"\n🎯 Recommendations:")
    if comparison['status'] == 'PASS':
        print("   ✅ Frameworks show good agreement")
        print("   • Strategy implementation validated")
        print("   • Results can be trusted")
    elif comparison['status'] == 'REVIEW':
        print("   ⚠️  Minor differences detected")
        print("   • Review signal generation timing")
        print("   • Check order execution details")
    else:
        print("   🔍 Significant differences found")
        print("   • Investigate strategy implementation")
        print("   • Check framework configuration")
        print("   • Verify data consistency")