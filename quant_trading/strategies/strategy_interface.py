"""
Strategy Interface using Protocols (no inheritance)
Clean interface definition without inheritance constraints
"""

from typing import Protocol, List, Dict, Any, Optional
import pandas as pd
from dataclasses import dataclass
from enum import Enum


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


class StrategyProtocol(Protocol):
    """
    Protocol defining the strategy interface
    No inheritance required - any class implementing these methods works
    
    This interface supports both our framework and VectorBT for cross-validation.
    
    Best Practice: Implement strategies with shared core logic:
    - Create private _calculate_* methods for core calculations
    - Use these in both get_signals() and generate_vectorbt_signals()
    - This ensures consistency and reduces maintenance burden
    """
    
    def get_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on market data
        
        Args:
            data: Market data up to current time
            
        Returns:
            List of trading signals
        """
        ...
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        ...
    
    def get_name(self) -> str:
        """Get strategy name"""
        ...
    
    # VectorBT compatibility methods
    def generate_vectorbt_signals(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals for VectorBT
        
        Args:
            data: Market data (full historical data)
            
        Returns:
            Tuple of (entries, exits) as boolean Series
        """
        ...
    
    def get_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get technical indicators for visualization (optional)
        
        Args:
            data: Market data
            
        Returns:
            Dictionary of indicator series
        """
        return {}


# Utility functions for strategy validation (no inheritance needed)
def validate_data(data: pd.DataFrame) -> None:
    """
    Validate input data format
    
    Args:
        data: Market data to validate
        
    Raises:
        ValueError: If data format is invalid
    """
    if data.empty:
        raise ValueError("Data cannot be empty")
        
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def create_signal(
    symbol: str = 'default',
    action: SignalType = SignalType.HOLD,
    quantity: int = 100,
    price: float = 0.0,
    confidence: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None
) -> Signal:
    """
    Utility function to create signals
    
    Args:
        symbol: Trading symbol
        action: Signal type
        quantity: Number of shares
        price: Signal price
        confidence: Signal confidence (0-1)
        metadata: Additional metadata
        
    Returns:
        Signal object
    """
    return Signal(
        symbol=symbol,
        action=action,
        quantity=quantity,
        price=price,
        confidence=confidence,
        metadata=metadata
    )


def signals_to_vectorbt(strategy: 'StrategyProtocol', data: pd.DataFrame, available_capital: float = 100000) -> tuple[pd.Series, pd.Series]:
    """
    Generic function to convert get_signals() output to VectorBT format
    
    Uses the same iteration pattern as BacktestEngine for consistency.
    
    Args:
        strategy: Strategy implementing StrategyProtocol
        data: Market data (full historical dataset)
        available_capital: Available capital for position sizing
        
    Returns:
        Tuple of (entries, exits) as boolean Series for VectorBT
    """
    # Initialize boolean series
    entries = pd.Series(False, index=data.index)
    exits = pd.Series(False, index=data.index)
    
    # Iterate through data exactly like BacktestEngine does
    for idx, _ in data.iterrows():
        # Get strategy signals (same as engine: data.loc[:idx])
        signals = strategy.get_signals(data.loc[:idx], available_capital)
        
        # Record any signals at this index
        for signal in signals:
            if signal.action == SignalType.BUY:
                entries.at[idx] = True
            elif signal.action == SignalType.SELL:
                exits.at[idx] = True
    
    return entries, exits