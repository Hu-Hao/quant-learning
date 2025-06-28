"""
Quant Trading Framework
A modular, production-ready quantitative trading system
"""

__version__ = "1.0.0"
__author__ = "Quant Trading Team"

from .backtesting.engine import BacktestEngine
from .strategies.base_strategy import BaseStrategy
from .data.data_fetcher import DataFetcher

__all__ = [
    "BacktestEngine",
    "BaseStrategy", 
    "DataFetcher"
]