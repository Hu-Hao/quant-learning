"""
Tests for performance and risk metrics
"""

import unittest
import numpy as np
from quant_trading.backtesting.metrics import (
    MetricsCalculator,
    PerformanceMetrics,
    RiskMetrics
)


class TestMetricsCalculator(unittest.TestCase):
    """Test cases for MetricsCalculator"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calc = MetricsCalculator(risk_free_rate=0.02)
        
        # Create sample portfolio values (growth trend with some volatility)
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # Daily returns
        portfolio_values = [100000]
        for ret in returns:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        self.portfolio_values = portfolio_values
        
        # Create benchmark values
        benchmark_returns = np.random.normal(0.0008, 0.015, 252)
        benchmark_values = [100000]
        for ret in benchmark_returns:
            benchmark_values.append(benchmark_values[-1] * (1 + ret))
        self.benchmark_values = benchmark_values
        
    def test_initialization(self):
        """Test MetricsCalculator initialization"""
        calc = MetricsCalculator(risk_free_rate=0.03)
        self.assertEqual(calc.risk_free_rate, 0.03)
        
        # Default risk-free rate
        calc_default = MetricsCalculator()
        self.assertEqual(calc_default.risk_free_rate, 0.02)
        
    def test_calculate_performance_metrics_basic(self):
        """Test basic performance metrics calculation"""
        metrics = self.calc.calculate_performance_metrics(self.portfolio_values)
        
        self.assertIsInstance(metrics, PerformanceMetrics)
        
        # Check that metrics are calculated
        self.assertIsInstance(metrics.total_return, float)
        self.assertIsInstance(metrics.annualized_return, float)
        self.assertIsInstance(metrics.volatility, float)
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertIsInstance(metrics.sortino_ratio, float)
        self.assertIsInstance(metrics.calmar_ratio, float)
        self.assertIsInstance(metrics.max_drawdown, float)
        self.assertIsInstance(metrics.avg_drawdown, float)
        
        # Check reasonable ranges
        self.assertGreaterEqual(metrics.volatility, 0)
        self.assertGreaterEqual(metrics.max_drawdown, 0)
        self.assertGreaterEqual(metrics.avg_drawdown, 0)
        
    def test_calculate_performance_metrics_empty_data(self):
        """Test performance metrics with empty data"""
        metrics = self.calc.calculate_performance_metrics([])
        
        # Should return empty metrics
        self.assertEqual(metrics.total_return, 0.0)
        self.assertEqual(metrics.annualized_return, 0.0)
        self.assertEqual(metrics.volatility, 0.0)
        
    def test_calculate_performance_metrics_single_value(self):
        """Test performance metrics with single value"""
        metrics = self.calc.calculate_performance_metrics([100000])
        
        # Should return empty metrics
        self.assertEqual(metrics.total_return, 0.0)
        self.assertEqual(metrics.annualized_return, 0.0)
        
    def test_calculate_risk_metrics_basic(self):
        """Test basic risk metrics calculation"""
        metrics = self.calc.calculate_risk_metrics(self.portfolio_values)
        
        self.assertIsInstance(metrics, RiskMetrics)
        
        # Check that metrics are calculated
        self.assertIsInstance(metrics.var_95, float)
        self.assertIsInstance(metrics.var_99, float)
        self.assertIsInstance(metrics.cvar_95, float)
        self.assertIsInstance(metrics.skewness, float)
        self.assertIsInstance(metrics.kurtosis, float)
        self.assertIsInstance(metrics.tail_ratio, float)
        
        # Check logical relationships
        self.assertLessEqual(metrics.var_99, metrics.var_95)  # 99% VaR should be more extreme
        self.assertLessEqual(metrics.cvar_95, metrics.var_95)  # CVaR should be more extreme than VaR
        
    def test_calculate_risk_metrics_with_benchmark(self):
        """Test risk metrics with benchmark"""
        metrics = self.calc.calculate_risk_metrics(
            self.portfolio_values, 
            self.benchmark_values
        )
        
        # Should include capture ratios
        self.assertIsInstance(metrics.upside_capture, float)
        self.assertIsInstance(metrics.downside_capture, float)
        self.assertGreaterEqual(metrics.upside_capture, 0)
        self.assertGreaterEqual(metrics.downside_capture, 0)
        
    def test_drawdown_calculation(self):
        """Test drawdown calculation specifically"""
        # Create portfolio with known drawdown
        values = [100000, 105000, 110000, 100000, 95000, 105000, 115000]
        metrics = self.calc.calculate_performance_metrics(values)
        
        # Max drawdown should be around 13.6% (110000 to 95000)
        expected_max_dd = (110000 - 95000) / 110000
        self.assertAlmostEqual(metrics.max_drawdown, expected_max_dd, places=2)
        
    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        # Create portfolio with positive returns
        values = [100000]
        for i in range(252):
            values.append(values[-1] * 1.002)  # 0.2% daily return
            
        metrics = self.calc.calculate_performance_metrics(values)
        
        # Sharpe should be positive for consistently positive returns
        self.assertGreater(metrics.sharpe_ratio, 0)
        
    def test_sortino_ratio_calculation(self):
        """Test Sortino ratio calculation"""
        # Create portfolio with mixed returns (more positive than negative)
        np.random.seed(42)
        values = [100000]
        for i in range(252):
            ret = np.random.normal(0.002, 0.01)  # Positive bias
            values.append(values[-1] * (1 + ret))
            
        metrics = self.calc.calculate_performance_metrics(values)
        
        # Sortino should typically be higher than Sharpe for positively skewed returns
        self.assertGreaterEqual(metrics.sortino_ratio, 0)
        
    def test_var_calculation(self):
        """Test VaR calculation"""
        # Create portfolio with known distribution
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 1000)  # Normal distribution
        values = [100000]
        for ret in returns:
            values.append(values[-1] * (1 + ret))
            
        metrics = self.calc.calculate_risk_metrics(values)
        
        # VaR should be negative (representing losses)
        self.assertLess(metrics.var_95, 0)
        self.assertLess(metrics.var_99, 0)
        self.assertLess(metrics.var_99, metrics.var_95)  # 99% should be more extreme
        
    def test_capture_ratios(self):
        """Test capture ratio calculation"""
        # Create portfolio that outperforms in up markets, underperforms in down markets
        portfolio_returns = []
        benchmark_returns = []
        
        for i in range(100):
            bench_ret = np.random.normal(0, 0.02)
            if bench_ret > 0:
                port_ret = bench_ret * 1.2  # 120% upside capture
            else:
                port_ret = bench_ret * 0.8  # 80% downside capture
            
            portfolio_returns.append(port_ret)
            benchmark_returns.append(bench_ret)
            
        # Convert to portfolio values
        port_values = [100000]
        bench_values = [100000]
        
        for i in range(100):
            port_values.append(port_values[-1] * (1 + portfolio_returns[i]))
            bench_values.append(bench_values[-1] * (1 + benchmark_returns[i]))
            
        metrics = self.calc.calculate_risk_metrics(port_values, bench_values)
        
        # Should reflect the constructed capture ratios
        self.assertGreater(metrics.upside_capture, 1.0)  # > 100% upside capture
        self.assertLess(metrics.downside_capture, 1.0)   # < 100% downside capture
        
    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary"""
        perf_metrics = self.calc.calculate_performance_metrics(self.portfolio_values)
        risk_metrics = self.calc.calculate_risk_metrics(self.portfolio_values)
        
        perf_dict = perf_metrics.to_dict()
        risk_dict = risk_metrics.to_dict()
        
        # Check structure
        self.assertIsInstance(perf_dict, dict)
        self.assertIsInstance(risk_dict, dict)
        
        # Check key metrics are present
        self.assertIn('total_return', perf_dict)
        self.assertIn('sharpe_ratio', perf_dict)
        self.assertIn('max_drawdown', perf_dict)
        
        self.assertIn('var_95', risk_dict)
        self.assertIn('skewness', risk_dict)
        self.assertIn('tail_ratio', risk_dict)
        
    def test_generate_report(self):
        """Test report generation"""
        report = self.calc.generate_report(
            self.portfolio_values,
            strategy_name="Test Strategy"
        )
        
        self.assertIsInstance(report, str)
        self.assertIn("Test Strategy", report)
        self.assertIn("PERFORMANCE REPORT", report)
        self.assertIn("Total Return", report)
        self.assertIn("Sharpe Ratio", report)
        self.assertIn("Max Drawdown", report)
        
    def test_generate_report_with_benchmark(self):
        """Test report generation with benchmark"""
        report = self.calc.generate_report(
            self.portfolio_values,
            benchmark_values=self.benchmark_values,
            strategy_name="Test Strategy"
        )
        
        self.assertIsInstance(report, str)
        self.assertIn("CAPTURE RATIOS", report)
        self.assertIn("Upside Capture", report)
        self.assertIn("Downside Capture", report)
        
    def test_recovery_time_calculation(self):
        """Test recovery time calculation"""
        # Create portfolio with clear drawdown and recovery
        values = [100000, 110000, 120000, 100000, 90000, 95000, 105000, 115000, 125000]
        metrics = self.calc.calculate_performance_metrics(values)
        
        # Should have some recovery time
        self.assertIsInstance(metrics.recovery_time, (int, type(None)))
        
    def test_edge_case_zero_volatility(self):
        """Test with zero volatility (constant returns)"""
        values = [100000] * 100  # Constant portfolio value
        
        metrics = self.calc.calculate_performance_metrics(values)
        
        # Should handle gracefully
        self.assertEqual(metrics.volatility, 0.0)
        self.assertEqual(metrics.sharpe_ratio, 0.0)  # Should be 0 when volatility is 0
        
    def test_edge_case_all_negative_returns(self):
        """Test with all negative returns"""
        values = [100000]
        for i in range(50):
            values.append(values[-1] * 0.99)  # -1% daily return
            
        metrics = self.calc.calculate_performance_metrics(values)
        
        # Should have negative returns and Sharpe ratio
        self.assertLess(metrics.total_return, 0)
        self.assertLess(metrics.annualized_return, 0)
        self.assertLess(metrics.sharpe_ratio, 0)


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for PerformanceMetrics dataclass"""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation"""
        metrics = PerformanceMetrics(
            total_return=0.15,
            annualized_return=0.12,
            volatility=0.18,
            sharpe_ratio=0.67,
            sortino_ratio=0.85,
            calmar_ratio=0.55,
            max_drawdown=0.08,
            avg_drawdown=0.03,
            recovery_time=45
        )
        
        self.assertEqual(metrics.total_return, 0.15)
        self.assertEqual(metrics.sharpe_ratio, 0.67)
        self.assertEqual(metrics.recovery_time, 45)


class TestRiskMetrics(unittest.TestCase):
    """Test cases for RiskMetrics dataclass"""
    
    def test_risk_metrics_creation(self):
        """Test RiskMetrics creation"""
        metrics = RiskMetrics(
            var_95=-0.03,
            var_99=-0.05,
            cvar_95=-0.04,
            skewness=-0.2,
            kurtosis=3.5,
            tail_ratio=1.8,
            upside_capture=1.1,
            downside_capture=0.9
        )
        
        self.assertEqual(metrics.var_95, -0.03)
        self.assertEqual(metrics.upside_capture, 1.1)
        self.assertEqual(metrics.tail_ratio, 1.8)


if __name__ == '__main__':
    unittest.main()