"""
Performance Visualization
Charts and plots for backtesting results analysis
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings

# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    # Create dummy classes for type hints when matplotlib isn't available
    class Figure:
        pass


class PerformanceVisualizer:
    """
    Visualization tools for backtesting results
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size (width, height)
        """
        if not HAS_MATPLOTLIB:
            warnings.warn("Matplotlib not available. Install with: pip install matplotlib")
        
        self.figsize = figsize
    
    def plot_portfolio_performance(
        self,
        portfolio_values: List[float],
        timestamps: Optional[List] = None,
        benchmark_values: Optional[List[float]] = None,
        title: str = "Portfolio Performance"
    ) -> Optional[Figure]:
        """
        Plot portfolio value over time
        
        Args:
            portfolio_values: Portfolio value series
            timestamps: Time index
            benchmark_values: Optional benchmark series
            title: Plot title
            
        Returns:
            Matplotlib figure or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return None
        
        if not portfolio_values:
            print("No portfolio values to plot")
            return None
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create time index if not provided
        if timestamps is None:
            x_axis = range(len(portfolio_values))
            ax.set_xlabel("Days")
        else:
            x_axis = timestamps
            ax.set_xlabel("Date")
            
            # Format dates if datetime
            if hasattr(timestamps[0], 'strftime'):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot portfolio
        ax.plot(x_axis, portfolio_values, label='Portfolio', linewidth=2, color='blue')
        
        # Plot benchmark if provided
        if benchmark_values and len(benchmark_values) == len(portfolio_values):
            ax.plot(x_axis, benchmark_values, label='Benchmark', 
                   linewidth=1, color='gray', alpha=0.7)
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add performance stats as text
        if len(portfolio_values) > 1:
            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
            ax.text(0.02, 0.98, f'Total Return: {total_return:.2%}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_drawdown(
        self,
        portfolio_values: List[float],
        timestamps: Optional[List] = None,
        title: str = "Drawdown Analysis"
    ) -> Optional[Figure]:
        """
        Plot drawdown over time
        
        Args:
            portfolio_values: Portfolio value series
            timestamps: Time index
            title: Plot title
            
        Returns:
            Matplotlib figure or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return None
        
        if len(portfolio_values) < 2:
            print("Insufficient data for drawdown plot")
            return None
        
        # Calculate drawdown
        values = pd.Series(portfolio_values)
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create time index
        if timestamps is None:
            x_axis = range(len(portfolio_values))
            ax.set_xlabel("Days")
        else:
            x_axis = timestamps
            ax.set_xlabel("Date")
        
        # Plot drawdown
        ax.fill_between(x_axis, drawdown * 100, 0, alpha=0.3, color='red', label='Drawdown')
        ax.plot(x_axis, drawdown * 100, color='red', linewidth=1)
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add max drawdown stat
        max_dd = abs(drawdown.min())
        ax.text(0.02, 0.02, f'Max Drawdown: {max_dd:.2%}', 
               transform=ax.transAxes, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_returns_distribution(
        self,
        portfolio_values: List[float],
        title: str = "Returns Distribution"
    ) -> Optional[Figure]:
        """
        Plot histogram of returns
        
        Args:
            portfolio_values: Portfolio value series
            title: Plot title
            
        Returns:
            Matplotlib figure or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return None
        
        if len(portfolio_values) < 2:
            print("Insufficient data for returns distribution")
            return None
        
        # Calculate returns
        returns = pd.Series(portfolio_values).pct_change().dropna()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot histogram
        ax.hist(returns * 100, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add vertical lines for mean and percentiles
        mean_return = returns.mean() * 100
        p5 = np.percentile(returns * 100, 5)
        p95 = np.percentile(returns * 100, 95)
        
        ax.axvline(mean_return, color='green', linestyle='--', label=f'Mean: {mean_return:.2f}%')
        ax.axvline(p5, color='red', linestyle='--', label=f'5th percentile: {p5:.2f}%')
        ax.axvline(p95, color='red', linestyle='--', label=f'95th percentile: {p95:.2f}%')
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Daily Returns (%)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_strategy_comparison(
        self,
        results: Dict[str, Dict],
        metric: str = 'total_return',
        title: Optional[str] = None
    ) -> Optional[Figure]:
        """
        Plot comparison of multiple strategies
        
        Args:
            results: Dictionary of {strategy_name: {performance_dict}}
            metric: Metric to compare
            title: Plot title
            
        Returns:
            Matplotlib figure or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return None
        
        if not results:
            print("No results to plot")
            return None
        
        # Extract data
        strategy_names = list(results.keys())
        values = []
        
        for name in strategy_names:
            if 'performance' in results[name] and metric in results[name]['performance']:
                values.append(results[name]['performance'][metric])
            else:
                values.append(0)
        
        # Convert to percentage if it's a ratio
        if metric in ['total_return', 'annualized_return', 'max_drawdown', 'win_rate']:
            values = [v * 100 for v in values]
            ylabel = f"{metric.replace('_', ' ').title()} (%)"
        else:
            ylabel = metric.replace('_', ' ').title()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create bar plot
        bars = ax.bar(strategy_names, values, alpha=0.7, 
                     color=['skyblue', 'lightgreen', 'salmon', 'gold', 'lightcoral'][:len(strategy_names)])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   f'{value:.1f}', ha='center', va='bottom')
        
        # Formatting
        if title is None:
            title = f"Strategy Comparison - {metric.replace('_', ' ').title()}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        if len(max(strategy_names, key=len)) > 10:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(
        self,
        returns_data: Dict[str, List[float]],
        title: str = "Strategy Correlation Matrix"
    ) -> Optional[Figure]:
        """
        Plot correlation heatmap of strategy returns
        
        Args:
            returns_data: Dictionary of {strategy_name: returns_list}
            title: Plot title
            
        Returns:
            Matplotlib figure or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return None
        
        if len(returns_data) < 2:
            print("Need at least 2 strategies for correlation analysis")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(returns_data)
        correlation_matrix = df.corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        im = ax.imshow(correlation_matrix, cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation')
        
        # Set ticks and labels
        ax.set_xticks(range(len(correlation_matrix.columns)))
        ax.set_yticks(range(len(correlation_matrix.index)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(correlation_matrix.index)
        
        # Add correlation values as text
        for i in range(len(correlation_matrix.index)):
            for j in range(len(correlation_matrix.columns)):
                text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black")
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_dashboard(
        self,
        portfolio_values: List[float],
        timestamps: Optional[List] = None,
        benchmark_values: Optional[List[float]] = None,
        trades: Optional[List] = None,
        title: str = "Trading Dashboard"
    ) -> Optional[Figure]:
        """
        Create comprehensive dashboard
        
        Args:
            portfolio_values: Portfolio value series
            timestamps: Time index
            benchmark_values: Optional benchmark series
            trades: List of trade objects
            title: Dashboard title
            
        Returns:
            Matplotlib figure or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return None
        
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Portfolio Performance
        ax1 = plt.subplot(2, 2, 1)
        self._plot_performance_subplot(ax1, portfolio_values, timestamps, benchmark_values)
        
        # 2. Drawdown
        ax2 = plt.subplot(2, 2, 2)
        self._plot_drawdown_subplot(ax2, portfolio_values, timestamps)
        
        # 3. Returns Distribution
        ax3 = plt.subplot(2, 2, 3)
        self._plot_returns_subplot(ax3, portfolio_values)
        
        # 4. Trade Analysis
        ax4 = plt.subplot(2, 2, 4)
        self._plot_trades_subplot(ax4, trades)
        
        plt.tight_layout()
        return fig
    
    def _plot_performance_subplot(self, ax, portfolio_values, timestamps, benchmark_values):
        """Helper method for performance subplot"""
        x_axis = timestamps if timestamps else range(len(portfolio_values))
        ax.plot(x_axis, portfolio_values, label='Portfolio', linewidth=2)
        
        if benchmark_values:
            ax.plot(x_axis, benchmark_values, label='Benchmark', alpha=0.7)
        
        ax.set_title('Portfolio Performance')
        ax.set_ylabel('Value ($)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_drawdown_subplot(self, ax, portfolio_values, timestamps):
        """Helper method for drawdown subplot"""
        values = pd.Series(portfolio_values)
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max
        
        x_axis = timestamps if timestamps else range(len(portfolio_values))
        ax.fill_between(x_axis, drawdown * 100, 0, alpha=0.3, color='red')
        ax.plot(x_axis, drawdown * 100, color='red', linewidth=1)
        
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
    
    def _plot_returns_subplot(self, ax, portfolio_values):
        """Helper method for returns subplot"""
        if len(portfolio_values) > 1:
            returns = pd.Series(portfolio_values).pct_change().dropna()
            ax.hist(returns * 100, bins=30, alpha=0.7, color='skyblue')
            ax.axvline(returns.mean() * 100, color='red', linestyle='--')
        
        ax.set_title('Returns Distribution')
        ax.set_xlabel('Daily Returns (%)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
    
    def _plot_trades_subplot(self, ax, trades):
        """Helper method for trades subplot"""
        if trades and len(trades) > 0:
            pnls = [t.pnl for t in trades if hasattr(t, 'pnl') and t.pnl is not None]
            
            if pnls:
                colors = ['green' if p > 0 else 'red' for p in pnls]
                ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                ax.set_title('Trade P&L')
                ax.set_xlabel('Trade Number')
                ax.set_ylabel('P&L ($)')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No trade data available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Trade P&L')
        else:
            ax.text(0.5, 0.5, 'No trades executed', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Trade P&L')
    
    def plot_backtest_analysis(
        self,
        data: pd.DataFrame,
        portfolio_values: List[float],
        trades: Optional[List] = None,
        signals: Optional[List] = None,
        strategy_name: str = "Trading Strategy",
        symbol: str = "Asset"
    ) -> Optional[Figure]:
        """
        Create comprehensive backtest analysis visualization
        
        Shows:
        - Stock price with moving averages
        - Buy/sell signals and trade markers
        - Portfolio value over time
        - Trade analysis and performance metrics
        
        Args:
            data: OHLCV market data
            portfolio_values: Portfolio value series
            trades: List of trade objects
            signals: List of signal objects
            strategy_name: Name of the trading strategy
            symbol: Asset symbol
            
        Returns:
            Matplotlib figure or None if matplotlib unavailable
        """
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available for plotting")
            return None
        
        if data.empty or not portfolio_values:
            print("No data to plot")
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'{strategy_name} - Backtest Analysis for {symbol}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Price chart with signals and indicators
        ax1 = plt.subplot(3, 2, (1, 2))  # Top row, span 2 columns
        self._plot_price_with_signals(ax1, data, trades, signals, symbol)
        
        # 2. Portfolio value over time
        ax2 = plt.subplot(3, 2, 3)
        self._plot_portfolio_performance_detailed(ax2, portfolio_values, data.index, data['close'])
        
        # 3. Drawdown analysis
        ax3 = plt.subplot(3, 2, 4)
        self._plot_drawdown_analysis(ax3, portfolio_values, data.index)
        
        # 4. Trade analysis
        ax4 = plt.subplot(3, 2, 5)
        self._plot_trade_analysis(ax4, trades)
        
        # 5. Performance summary
        ax5 = plt.subplot(3, 2, 6)
        self._plot_performance_summary(ax5, portfolio_values, trades)
        
        plt.tight_layout()
        return fig
    
    def _plot_price_with_signals(self, ax, data, trades, signals, symbol):
        """Plot price chart with signals and trade markers"""
        # Plot price
        ax.plot(data.index, data['close'], label='Close Price', color='black', linewidth=1.5)
        
        # Add moving averages if data has enough points
        if len(data) >= 50:
            ma20 = data['close'].rolling(20).mean()
            ma50 = data['close'].rolling(50).mean()
            ax.plot(data.index, ma20, label='MA 20', color='orange', alpha=0.7, linewidth=1)
            ax.plot(data.index, ma50, label='MA 50', color='red', alpha=0.7, linewidth=1)
        elif len(data) >= 20:
            ma10 = data['close'].rolling(10).mean()
            ma20 = data['close'].rolling(20).mean()
            ax.plot(data.index, ma10, label='MA 10', color='orange', alpha=0.7, linewidth=1)
            ax.plot(data.index, ma20, label='MA 20', color='red', alpha=0.7, linewidth=1)
        
        # Plot trade markers
        if trades:
            buy_dates = []
            buy_prices = []
            sell_dates = []
            sell_prices = []
            
            for trade in trades:
                if hasattr(trade, 'entry_time') and hasattr(trade, 'entry_price'):
                    if hasattr(trade, 'side') and trade.side == 'long':
                        buy_dates.append(trade.entry_time)
                        buy_prices.append(trade.entry_price)
                    
                if hasattr(trade, 'exit_time') and hasattr(trade, 'exit_price'):
                    sell_dates.append(trade.exit_time)
                    sell_prices.append(trade.exit_price)
            
            # Plot buy signals
            if buy_dates and buy_prices:
                ax.scatter(buy_dates, buy_prices, color='green', marker='^', 
                          s=100, label='Buy Signal', alpha=0.8, zorder=5)
            
            # Plot sell signals
            if sell_dates and sell_prices:
                ax.scatter(sell_dates, sell_prices, color='red', marker='v', 
                          s=100, label='Sell Signal', alpha=0.8, zorder=5)
        
        # Plot additional signals if provided
        if signals:
            for signal in signals:
                if hasattr(signal, 'action') and hasattr(signal, 'price'):
                    try:
                        # Try to get the signal date from data index
                        signal_date = None
                        if hasattr(signal, 'timestamp'):
                            signal_date = signal.timestamp
                        elif len(data) > 0:
                            # Use a reasonable approximation
                            signal_date = data.index[min(len(data)-1, len(signals)//2)]
                        
                        if signal_date and signal_date in data.index:
                            action_str = str(signal.action).lower()
                            if 'buy' in action_str:
                                ax.scatter(signal_date, signal.price, color='lightgreen', 
                                         marker='^', s=60, alpha=0.6, zorder=4)
                            elif 'sell' in action_str:
                                ax.scatter(signal_date, signal.price, color='lightcoral', 
                                         marker='v', s=60, alpha=0.6, zorder=4)
                    except:
                        continue
        
        ax.set_title(f'{symbol} Price Chart with Signals')
        ax.set_ylabel('Price ($)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if len(data) > 30:
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_portfolio_performance_detailed(self, ax, portfolio_values, timestamps, prices):
        """Plot portfolio performance vs buy-and-hold"""
        # Portfolio performance
        ax.plot(timestamps[:len(portfolio_values)], portfolio_values, 
               label='Strategy', color='blue', linewidth=2)
        
        # Buy and hold benchmark
        if len(prices) >= len(portfolio_values) and portfolio_values:
            initial_capital = portfolio_values[0]
            buy_hold = initial_capital * (prices[:len(portfolio_values)] / prices.iloc[0])
            ax.plot(timestamps[:len(portfolio_values)], buy_hold, 
                   label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)
        
        ax.set_title('Portfolio Performance Comparison')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add performance stats
        if len(portfolio_values) > 1:
            total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
            ax.text(0.02, 0.98, f'Strategy Return: {total_return:.1f}%', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def _plot_drawdown_analysis(self, ax, portfolio_values, timestamps):
        """Plot drawdown with underwater curve"""
        if len(portfolio_values) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Drawdown Analysis')
            return
        
        values = pd.Series(portfolio_values)
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max * 100
        
        ax.fill_between(timestamps[:len(portfolio_values)], drawdown, 0, 
                       alpha=0.3, color='red', label='Drawdown')
        ax.plot(timestamps[:len(portfolio_values)], drawdown, color='red', linewidth=1)
        
        ax.set_title('Drawdown Analysis')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # Add max drawdown
        max_dd = abs(drawdown.min())
        ax.text(0.02, 0.02, f'Max Drawdown: {max_dd:.1f}%', 
               transform=ax.transAxes, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    def _plot_trade_analysis(self, ax, trades):
        """Plot trade win/loss analysis"""
        if not trades or len(trades) == 0:
            ax.text(0.5, 0.5, 'No trades executed', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title('Trade Analysis')
            return
        
        # Analyze trades
        wins = []
        losses = []
        
        for trade in trades:
            if hasattr(trade, 'pnl') and trade.pnl is not None:
                if trade.pnl > 0:
                    wins.append(trade.pnl)
                else:
                    losses.append(abs(trade.pnl))
        
        # Create win/loss chart
        categories = []
        values = []
        colors = []
        
        if wins:
            categories.append(f'Wins ({len(wins)})')
            values.append(sum(wins))
            colors.append('green')
        
        if losses:
            categories.append(f'Losses ({len(losses)})')
            values.append(-sum(losses))
            colors.append('red')
        
        if categories:
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., 
                       height + (max(values) * 0.05 if height >= 0 else min(values) * 0.05),
                       f'${abs(value):.0f}', ha='center', 
                       va='bottom' if height >= 0 else 'top')
        
        ax.set_title('Trade Win/Loss Analysis')
        ax.set_ylabel('P&L ($)')
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_summary(self, ax, portfolio_values, trades):
        """Plot key performance metrics as text summary"""
        ax.axis('off')  # Turn off axis for text display
        
        if not portfolio_values or len(portfolio_values) < 2:
            ax.text(0.5, 0.5, 'Insufficient data for analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
        
        # Calculate metrics
        total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
        
        # Drawdown
        values = pd.Series(portfolio_values)
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100
        
        # Trade statistics
        total_trades = len(trades) if trades else 0
        winning_trades = 0
        losing_trades = 0
        total_pnl = 0
        
        if trades:
            for trade in trades:
                if hasattr(trade, 'pnl') and trade.pnl is not None:
                    total_pnl += trade.pnl
                    if trade.pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Create summary text
        summary_text = f"""Performance Summary:
        
Total Return: {total_return:.1f}%
Max Drawdown: {max_drawdown:.1f}%

Total Trades: {total_trades}
Winning Trades: {winning_trades}
Losing Trades: {losing_trades}
Win Rate: {win_rate:.1f}%

Total P&L: ${total_pnl:.0f}
Avg P&L per Trade: ${total_pnl/total_trades:.0f}""" if total_trades > 0 else f"""Performance Summary:

Total Return: {total_return:.1f}%
Max Drawdown: {max_drawdown:.1f}%

No trades executed"""
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))