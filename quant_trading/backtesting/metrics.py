"""
Performance and Risk Metrics
Comprehensive analysis tools for backtesting results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    avg_drawdown: float
    recovery_time: Optional[int]
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'max_drawdown': self.max_drawdown,
            'avg_drawdown': self.avg_drawdown,
            'recovery_time': self.recovery_time or 0
        }


@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    cvar_95: float  # Conditional VaR (95%)
    skewness: float
    kurtosis: float
    tail_ratio: float
    upside_capture: float
    downside_capture: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'var_95': self.var_95,
            'var_99': self.var_99,
            'cvar_95': self.cvar_95,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'tail_ratio': self.tail_ratio,
            'upside_capture': self.upside_capture,
            'downside_capture': self.downside_capture
        }


class MetricsCalculator:
    """Calculate comprehensive performance and risk metrics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator
        
        Args:
            risk_free_rate: Annual risk-free rate for ratio calculations
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_performance_metrics(
        self,
        portfolio_values: List[float],
        trading_days: int = 252
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics
        
        Args:
            portfolio_values: Time series of portfolio values
            trading_days: Number of trading days per year
            
        Returns:
            PerformanceMetrics object
        """
        if len(portfolio_values) < 2:
            return self._empty_performance_metrics()
        
        returns_series = pd.Series(portfolio_values)
        returns = returns_series.pct_change().dropna()
        
        # Basic returns
        total_return = (returns_series.iloc[-1] / returns_series.iloc[0]) - 1
        
        # Annualized return
        periods = len(returns_series)
        years = periods / trading_days
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(trading_days)
        
        # Sharpe Ratio
        excess_return = annualized_return - self.risk_free_rate
        sharpe_ratio = excess_return / volatility if volatility != 0 else 0
        
        # Sortino Ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(trading_days)
        sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0
        
        # Drawdown metrics
        drawdown_info = self._calculate_drawdown_metrics(returns_series)
        max_drawdown = drawdown_info['max_drawdown']
        avg_drawdown = drawdown_info['avg_drawdown']
        recovery_time = drawdown_info['recovery_time']
        
        # Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            avg_drawdown=avg_drawdown,
            recovery_time=recovery_time
        )
    
    def calculate_risk_metrics(
        self,
        portfolio_values: List[float],
        benchmark_values: Optional[List[float]] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics
        
        Args:
            portfolio_values: Time series of portfolio values
            benchmark_values: Optional benchmark for capture ratios
            
        Returns:
            RiskMetrics object
        """
        if len(portfolio_values) < 2:
            return self._empty_risk_metrics()
        
        returns_series = pd.Series(portfolio_values)
        returns = returns_series.pct_change().dropna()
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)  # 5th percentile (95% VaR)
        var_99 = np.percentile(returns, 1)  # 1st percentile (99% VaR)
        
        # Conditional VaR (Expected Shortfall)
        tail_returns = returns[returns <= var_95]
        cvar_95 = tail_returns.mean() if len(tail_returns) > 0 else var_95
        
        # Distribution metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Tail ratio (95th percentile / 5th percentile)
        upside_tail = np.percentile(returns, 95)
        downside_tail = np.percentile(returns, 5)
        tail_ratio = abs(upside_tail / downside_tail) if downside_tail != 0 else 0
        
        # Capture ratios (if benchmark provided)
        upside_capture = 0.0
        downside_capture = 0.0
        
        if benchmark_values and len(benchmark_values) == len(portfolio_values):
            benchmark_returns = pd.Series(benchmark_values).pct_change().dropna()
            if len(benchmark_returns) == len(returns):
                capture_ratios = self._calculate_capture_ratios(returns, benchmark_returns)
                upside_capture = capture_ratios['upside']
                downside_capture = capture_ratios['downside']
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio,
            upside_capture=upside_capture,
            downside_capture=downside_capture
        )
    
    def _calculate_drawdown_metrics(self, returns_series: pd.Series) -> Dict:
        """Calculate detailed drawdown metrics"""
        # Calculate drawdown series
        cumulative = (1 + returns_series.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = abs(drawdown.min())
        
        # Average drawdown
        drawdown_periods = drawdown[drawdown < 0]
        avg_drawdown = abs(drawdown_periods.mean()) if len(drawdown_periods) > 0 else 0
        
        # Recovery time (periods to recover from max drawdown)
        recovery_time = None
        max_dd_idx = drawdown.idxmin()
        
        if max_dd_idx is not None:
            # Find when portfolio recovers to previous high
            recovery_series = drawdown.loc[max_dd_idx:]
            recovery_points = recovery_series[recovery_series >= -0.001]  # Within 0.1%
            
            if len(recovery_points) > 0:
                recovery_idx = recovery_points.index[0]
                recovery_time = len(drawdown.loc[max_dd_idx:recovery_idx])
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'recovery_time': recovery_time
        }
    
    def _calculate_capture_ratios(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate upside and downside capture ratios"""
        
        # Separate upside and downside periods
        upside_mask = benchmark_returns > 0
        downside_mask = benchmark_returns < 0
        
        # Upside capture
        if upside_mask.sum() > 0:
            portfolio_upside = portfolio_returns[upside_mask].mean()
            benchmark_upside = benchmark_returns[upside_mask].mean()
            upside_capture = portfolio_upside / benchmark_upside if benchmark_upside != 0 else 0
        else:
            upside_capture = 0
        
        # Downside capture
        if downside_mask.sum() > 0:
            portfolio_downside = portfolio_returns[downside_mask].mean()
            benchmark_downside = benchmark_returns[downside_mask].mean()
            downside_capture = portfolio_downside / benchmark_downside if benchmark_downside != 0 else 0
        else:
            downside_capture = 0
        
        return {
            'upside': upside_capture,
            'downside': downside_capture
        }
    
    def _empty_performance_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics"""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            avg_drawdown=0.0,
            recovery_time=None
        )
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        """Return empty risk metrics"""
        return RiskMetrics(
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            skewness=0.0,
            kurtosis=0.0,
            tail_ratio=0.0,
            upside_capture=0.0,
            downside_capture=0.0
        )
    
    def generate_report(
        self,
        portfolio_values: List[float],
        benchmark_values: Optional[List[float]] = None,
        strategy_name: str = "Strategy"
    ) -> str:
        """
        Generate a comprehensive performance report
        
        Args:
            portfolio_values: Portfolio value series
            benchmark_values: Optional benchmark series
            strategy_name: Name of the strategy
            
        Returns:
            Formatted report string
        """
        perf_metrics = self.calculate_performance_metrics(portfolio_values)
        risk_metrics = self.calculate_risk_metrics(portfolio_values, benchmark_values)
        
        report = f"""
{'='*60}
{strategy_name.upper()} PERFORMANCE REPORT
{'='*60}

RETURN METRICS
--------------
Total Return:        {perf_metrics.total_return:>10.2%}
Annualized Return:   {perf_metrics.annualized_return:>10.2%}
Volatility:          {perf_metrics.volatility:>10.2%}

RISK-ADJUSTED METRICS
---------------------
Sharpe Ratio:        {perf_metrics.sharpe_ratio:>10.3f}
Sortino Ratio:       {perf_metrics.sortino_ratio:>10.3f}
Calmar Ratio:        {perf_metrics.calmar_ratio:>10.3f}

DRAWDOWN METRICS
----------------
Max Drawdown:        {perf_metrics.max_drawdown:>10.2%}
Avg Drawdown:        {perf_metrics.avg_drawdown:>10.2%}
Recovery Time:       {perf_metrics.recovery_time or 'N/A':>10}

RISK METRICS
------------
VaR (95%):          {risk_metrics.var_95:>10.2%}
VaR (99%):          {risk_metrics.var_99:>10.2%}
CVaR (95%):         {risk_metrics.cvar_95:>10.2%}
Skewness:           {risk_metrics.skewness:>10.3f}
Kurtosis:           {risk_metrics.kurtosis:>10.3f}
Tail Ratio:         {risk_metrics.tail_ratio:>10.3f}
"""
        
        if benchmark_values:
            report += f"""
CAPTURE RATIOS
--------------
Upside Capture:     {risk_metrics.upside_capture:>10.2%}
Downside Capture:   {risk_metrics.downside_capture:>10.2%}
"""
        
        report += f"\n{'='*60}\n"
        
        return report