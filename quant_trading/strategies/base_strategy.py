"""
Base Strategy Class
Defines the interface and common functionality for all trading strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
import pandas as pd
import logging


class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    """Trading signal with metadata"""
    symbol: str
    action: SignalType
    quantity: int
    price: float
    timestamp: Optional[pd.Timestamp] = None
    confidence: float = 1.0  # 0-1 confidence score
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary for backtesting engine"""
        return {
            'symbol': self.symbol,
            'action': self.action.value,
            'quantity': self.quantity,
            'price': self.price,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'metadata': self.metadata or {}
        }


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies
    
    Provides common functionality and enforces interface contract
    """
    
    def __init__(self, name: str, **params):
        self.name = name
        self.params = params
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
        # Strategy state
        self._is_initialized = False
        self._trade_count = 0
        self._last_signal_time = None
        
    def initialize(self, data: pd.DataFrame) -> None:
        """
        Initialize strategy with historical data
        
        Args:
            data: Historical market data
        """
        self.validate_data(data)
        self._setup_indicators(data)
        self._is_initialized = True
        self.logger.info(f"Strategy {self.name} initialized with {len(data)} data points")
    
    def validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data format
        
        Args:
            data: Market data to validate
            
        Raises:
            ValueError: If data format is invalid
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if data.empty:
            raise ValueError("Data cannot be empty")
            
        if not isinstance(data.index, pd.DatetimeIndex):
            self.logger.warning("Data index is not DatetimeIndex, consider using datetime index")
    
    @abstractmethod
    def _setup_indicators(self, data: pd.DataFrame) -> None:
        """
        Setup technical indicators needed by the strategy
        
        Args:
            data: Historical market data
        """
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on current market data
        
        Args:
            data: Current market data (up to current time)
            
        Returns:
            List of trading signals
        """
        pass
    
    def get_signal(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Public interface for signal generation (compatible with backtesting engine)
        
        Args:
            data: Market data
            
        Returns:
            List of signal dictionaries for backtesting engine
        """
        if not self._is_initialized:
            self.initialize(data)
            
        try:
            signals = self.generate_signals(data)
            
            # Convert signals to dictionary format
            signal_dicts = []
            for signal in signals:
                signal_dict = signal.to_dict()
                signal_dicts.append(signal_dict)
                
            if signal_dicts:
                self._trade_count += len(signal_dicts)
                self._last_signal_time = data.index[-1] if hasattr(data.index, '__getitem__') else None
                
            return signal_dicts
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return []
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.params.copy()
    
    def update_parameters(self, **new_params) -> None:
        """
        Update strategy parameters
        
        Args:
            **new_params: New parameter values
        """
        self.params.update(new_params)
        self._is_initialized = False  # Force re-initialization
        self.logger.info(f"Updated parameters: {new_params}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        return {
            'name': self.name,
            'trade_count': self._trade_count,
            'last_signal_time': self._last_signal_time,
            'is_initialized': self._is_initialized,
            'parameters': self.params
        }
    
    def __str__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', params={self.params})"