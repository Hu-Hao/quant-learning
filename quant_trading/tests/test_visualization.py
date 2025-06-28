"""
Tests for visualization module
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from quant_trading.utils.visualization import PerformanceVisualizer


class TestPerformanceVisualizer(unittest.TestCase):
    """Test cases for PerformanceVisualizer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.visualizer = PerformanceVisualizer(figsize=(10, 6))
        
        # Create sample data
        np.random.seed(42)
        self.portfolio_values = [100000]
        returns = np.random.normal(0.001, 0.02, 100)
        for ret in returns:
            self.portfolio_values.append(self.portfolio_values[-1] * (1 + ret))
            
        # Create timestamps
        self.timestamps = pd.date_range('2023-01-01', periods=len(self.portfolio_values), freq='D')
        
        # Create benchmark values
        self.benchmark_values = [100000]
        benchmark_returns = np.random.normal(0.0008, 0.015, 100)
        for ret in benchmark_returns:
            self.benchmark_values.append(self.benchmark_values[-1] * (1 + ret))
            
        # Create mock trades
        self.mock_trades = []
        for i in range(10):
            trade = MagicMock()
            trade.pnl = np.random.normal(100, 500)
            self.mock_trades.append(trade)
            
    def test_initialization(self):
        """Test PerformanceVisualizer initialization"""
        viz = PerformanceVisualizer(figsize=(12, 8))
        self.assertEqual(viz.figsize, (12, 8))
        
        # Default figsize
        viz_default = PerformanceVisualizer()
        self.assertEqual(viz_default.figsize, (12, 8))
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', False)
    def test_no_matplotlib_warning(self):
        """Test behavior when matplotlib is not available"""
        viz = PerformanceVisualizer()
        
        # Should return None and print warning
        result = viz.plot_portfolio_performance(self.portfolio_values)
        self.assertIsNone(result)
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    @patch('quant_trading.utils.visualization.plt')
    def test_plot_portfolio_performance_basic(self, mock_plt):
        """Test basic portfolio performance plotting"""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.visualizer.plot_portfolio_performance(self.portfolio_values)
        
        # Should create plot
        mock_plt.subplots.assert_called_once()
        mock_ax.plot.assert_called()
        mock_ax.set_title.assert_called()
        mock_ax.set_ylabel.assert_called()
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    @patch('quant_trading.utils.visualization.plt')
    def test_plot_portfolio_performance_with_benchmark(self, mock_plt):
        """Test portfolio performance plotting with benchmark"""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.visualizer.plot_portfolio_performance(
            self.portfolio_values,
            benchmark_values=self.benchmark_values
        )
        
        # Should plot both portfolio and benchmark
        self.assertEqual(mock_ax.plot.call_count, 2)
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    @patch('quant_trading.utils.visualization.plt')
    def test_plot_portfolio_performance_with_timestamps(self, mock_plt):
        """Test portfolio performance plotting with timestamps"""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.visualizer.plot_portfolio_performance(
            self.portfolio_values,
            timestamps=self.timestamps
        )
        
        mock_ax.plot.assert_called()
        mock_ax.set_xlabel.assert_called_with("Date")
        
    def test_plot_portfolio_performance_empty_data(self):
        """Test plotting with empty data"""
        result = self.visualizer.plot_portfolio_performance([])
        self.assertIsNone(result)
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    @patch('quant_trading.utils.visualization.plt')
    def test_plot_drawdown(self, mock_plt):
        """Test drawdown plotting"""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.visualizer.plot_drawdown(self.portfolio_values)
        
        mock_plt.subplots.assert_called_once()
        mock_ax.fill_between.assert_called()
        mock_ax.plot.assert_called()
        mock_ax.set_title.assert_called()
        
    def test_plot_drawdown_insufficient_data(self):
        """Test drawdown plotting with insufficient data"""
        result = self.visualizer.plot_drawdown([100000])  # Single value
        self.assertIsNone(result)
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    @patch('quant_trading.utils.visualization.plt')
    def test_plot_returns_distribution(self, mock_plt):
        """Test returns distribution plotting"""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        result = self.visualizer.plot_returns_distribution(self.portfolio_values)
        
        mock_plt.subplots.assert_called_once()
        mock_ax.hist.assert_called()
        mock_ax.axvline.assert_called()  # For mean and percentiles
        mock_ax.set_title.assert_called()
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    @patch('quant_trading.utils.visualization.plt')
    def test_plot_strategy_comparison(self, mock_plt):
        """Test strategy comparison plotting"""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        
        results = {
            'Strategy1': {'performance': {'total_return': 0.15}},
            'Strategy2': {'performance': {'total_return': 0.10}},
            'Strategy3': {'performance': {'total_return': 0.20}}
        }
        
        result = self.visualizer.plot_strategy_comparison(results, metric='total_return')
        
        mock_plt.subplots.assert_called_once()
        mock_ax.bar.assert_called()
        mock_ax.set_title.assert_called()
        
    def test_plot_strategy_comparison_empty_results(self):
        """Test strategy comparison with empty results"""
        result = self.visualizer.plot_strategy_comparison({})
        self.assertIsNone(result)
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    @patch('quant_trading.utils.visualization.plt')
    def test_plot_correlation_heatmap(self, mock_plt):
        """Test correlation heatmap plotting"""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.colorbar.return_value = MagicMock()
        
        returns_data = {
            'Strategy1': np.random.normal(0.001, 0.02, 100),
            'Strategy2': np.random.normal(0.0008, 0.018, 100),
            'Strategy3': np.random.normal(0.0012, 0.025, 100)
        }
        
        result = self.visualizer.plot_correlation_heatmap(returns_data)
        
        mock_plt.subplots.assert_called_once()
        mock_ax.imshow.assert_called()
        mock_plt.colorbar.assert_called()
        
    def test_plot_correlation_heatmap_insufficient_strategies(self):
        """Test correlation heatmap with insufficient strategies"""
        returns_data = {'Strategy1': [0.1, 0.2, 0.3]}
        result = self.visualizer.plot_correlation_heatmap(returns_data)
        self.assertIsNone(result)
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    @patch('quant_trading.utils.visualization.plt')
    def test_create_dashboard(self, mock_plt):
        """Test dashboard creation"""
        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplot.return_value = MagicMock()
        
        result = self.visualizer.create_dashboard(
            self.portfolio_values,
            timestamps=self.timestamps,
            benchmark_values=self.benchmark_values,
            trades=self.mock_trades
        )
        
        mock_plt.figure.assert_called_once()
        # Should create 4 subplots
        self.assertEqual(mock_plt.subplot.call_count, 4)
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    @patch('quant_trading.utils.visualization.plt')
    def test_create_dashboard_minimal(self, mock_plt):
        """Test dashboard creation with minimal data"""
        mock_fig = MagicMock()
        mock_plt.figure.return_value = mock_fig
        mock_plt.subplot.return_value = MagicMock()
        
        result = self.visualizer.create_dashboard(self.portfolio_values)
        
        mock_plt.figure.assert_called_once()
        self.assertEqual(mock_plt.subplot.call_count, 4)
        
    def test_helper_methods_exist(self):
        """Test that helper methods exist"""
        # Check that helper methods are defined
        self.assertTrue(hasattr(self.visualizer, '_plot_performance_subplot'))
        self.assertTrue(hasattr(self.visualizer, '_plot_drawdown_subplot'))
        self.assertTrue(hasattr(self.visualizer, '_plot_returns_subplot'))
        self.assertTrue(hasattr(self.visualizer, '_plot_trades_subplot'))
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    def test_plot_performance_subplot(self):
        """Test performance subplot helper"""
        mock_ax = MagicMock()
        
        self.visualizer._plot_performance_subplot(
            mock_ax, 
            self.portfolio_values, 
            self.timestamps, 
            self.benchmark_values
        )
        
        mock_ax.plot.assert_called()
        mock_ax.set_title.assert_called_with('Portfolio Performance')
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    def test_plot_drawdown_subplot(self):
        """Test drawdown subplot helper"""
        mock_ax = MagicMock()
        
        self.visualizer._plot_drawdown_subplot(
            mock_ax, 
            self.portfolio_values, 
            self.timestamps
        )
        
        mock_ax.fill_between.assert_called()
        mock_ax.set_title.assert_called_with('Drawdown')
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    def test_plot_returns_subplot(self):
        """Test returns subplot helper"""
        mock_ax = MagicMock()
        
        self.visualizer._plot_returns_subplot(mock_ax, self.portfolio_values)
        
        mock_ax.hist.assert_called()
        mock_ax.set_title.assert_called_with('Returns Distribution')
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    def test_plot_trades_subplot_with_trades(self):
        """Test trades subplot helper with trades"""
        mock_ax = MagicMock()
        
        self.visualizer._plot_trades_subplot(mock_ax, self.mock_trades)
        
        mock_ax.bar.assert_called()
        mock_ax.set_title.assert_called_with('Trade P&L')
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    def test_plot_trades_subplot_no_trades(self):
        """Test trades subplot helper without trades"""
        mock_ax = MagicMock()
        
        self.visualizer._plot_trades_subplot(mock_ax, [])
        
        mock_ax.text.assert_called()
        mock_ax.set_title.assert_called_with('Trade P&L')
        
    @patch('quant_trading.utils.visualization.HAS_MATPLOTLIB', True)
    def test_plot_trades_subplot_none_trades(self):
        """Test trades subplot helper with None trades"""
        mock_ax = MagicMock()
        
        self.visualizer._plot_trades_subplot(mock_ax, None)
        
        mock_ax.text.assert_called()
        mock_ax.set_title.assert_called_with('Trade P&L')


if __name__ == '__main__':
    unittest.main()