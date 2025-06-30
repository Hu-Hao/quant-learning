"""
Moving Average Strategy
Classic trend-following strategy using moving average crossovers
"""

import pandas as pd
from typing import List, Optional, Dict, Any
from .strategy_interface import StrategyProtocol, Signal, SignalType, validate_data, create_signal, signals_to_vectorbt


class MovingAverageStrategy:
    """
    Moving Average Crossover Strategy
    
    Generates buy signals when short MA crosses above long MA
    Generates sell signals when short MA crosses below long MA
    
    Generic Architecture:
    - get_signals(): Core strategy logic (point-in-time)
    - generate_vectorbt_signals(): Delegates to generic signals_to_vectorbt()
    
    This approach works for ANY strategy and eliminates duplicate logic.
    """
    
    def __init__(
        self,
        short_window: int = 10,
        long_window: int = 30,
        quantity: int = 100,
        min_periods: Optional[int] = None
    ):
        """
        Initialize Moving Average Strategy
        
        Args:
            short_window: Period for short moving average
            long_window: Period for long moving average  
            quantity: Number of shares to trade
            min_periods: Minimum periods required for signal generation
        """
        # Validate parameters
        if short_window >= long_window:
            raise ValueError("Short window must be less than long window")
        if short_window < 1 or long_window < 1:
            raise ValueError("Window periods must be positive")
            
        self.name = "MovingAverage"
        self.params = {
            'short_window': short_window,
            'long_window': long_window,
            'quantity': quantity,
            'min_periods': min_periods or long_window
        }
            
    
    def get_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate moving average crossover signals
        
        Args:
            data: Market data up to current time
            
        Returns:
            List of trading signals
        """
        validate_data(data)
        if len(data) < self.params['min_periods']:
            return []
            
        # Calculate moving averages
        short_ma = data['close'].rolling(
            window=self.params['short_window'],
            min_periods=self.params['short_window']
        ).mean()
        
        long_ma = data['close'].rolling(
            window=self.params['long_window'], 
            min_periods=self.params['long_window']
        ).mean()
        
        # Need at least 2 points to detect crossover
        if len(short_ma.dropna()) < 2 or len(long_ma.dropna()) < 2:
            return []
            
        signals = []
        
        # Get current and previous values
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]
        
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1] if hasattr(data.index, '__getitem__') else None
        
        # Golden Cross - Short MA crosses above Long MA (Buy signal)
        if (current_short > current_long and prev_short <= prev_long):
            signal = create_signal(
                symbol='default',
                action=SignalType.BUY,
                quantity=self.params['quantity'],
                price=current_price,
                confidence=self._calculate_confidence(current_short, current_long, data),
                metadata={
                    'short_ma': current_short,
                    'long_ma': current_long,
                    'crossover_type': 'golden_cross',
                    'price_momentum': (current_price / data['close'].iloc[-5] - 1) if len(data) >= 5 else 0
                }
            )
            signal.timestamp = current_time
            signals.append(signal)
            
        # Death Cross - Short MA crosses below Long MA (Sell signal)  
        elif (current_short < current_long and prev_short >= prev_long):
            signal = create_signal(
                symbol='default',
                action=SignalType.SELL,
                quantity=self.params['quantity'],
                price=current_price,
                confidence=self._calculate_confidence(current_short, current_long, data),
                metadata={
                    'short_ma': current_short,
                    'long_ma': current_long,
                    'crossover_type': 'death_cross',
                    'price_momentum': (current_price / data['close'].iloc[-5] - 1) if len(data) >= 5 else 0
                }
            )
            signal.timestamp = current_time
            signals.append(signal)
            
        return signals
    
    def _calculate_confidence(self, short_ma: float, long_ma: float, data: pd.DataFrame) -> float:
        """
        Calculate signal confidence based on various factors
        
        Args:
            short_ma: Current short moving average
            long_ma: Current long moving average
            data: Market data
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence
        confidence = 0.5
        
        # Increase confidence based on MA separation
        ma_separation = abs(short_ma - long_ma) / long_ma
        confidence += min(0.3, ma_separation * 10)  # Max +0.3 for 3% separation
        
        # Increase confidence based on volume
        if len(data) >= 5:
            recent_volume = data['volume'].iloc[-5:].mean()
            avg_volume = data['volume'].mean()
            volume_ratio = recent_volume / avg_volume
            confidence += min(0.2, (volume_ratio - 1) * 0.5)  # Max +0.2 for 2x volume
            
        # Decrease confidence if recent volatility is high
        if len(data) >= 10:
            recent_volatility = data['close'].iloc[-10:].pct_change().std()
            avg_volatility = data['close'].pct_change().std()
            if recent_volatility > avg_volatility * 1.5:
                confidence -= 0.1
                
        return max(0.1, min(1.0, confidence))
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters"""
        return self.params.copy()
    
    def get_name(self) -> str:
        """Get strategy name"""
        return self.name
    
    def get_indicator_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get current indicator values for analysis
        
        Args:
            data: Market data
            
        Returns:
            DataFrame with indicator values
        """
        if len(data) < self.params['long_window']:
            return pd.DataFrame()
            
        indicators = pd.DataFrame(index=data.index)
        
        indicators['short_ma'] = data['close'].rolling(
            window=self.params['short_window']
        ).mean()
        
        indicators['long_ma'] = data['close'].rolling(
            window=self.params['long_window']
        ).mean()
        
        indicators['ma_diff'] = indicators['short_ma'] - indicators['long_ma']
        indicators['ma_ratio'] = indicators['short_ma'] / indicators['long_ma']
        
        return indicators
    
    def get_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get technical indicators for visualization (StrategyProtocol interface)
        
        Args:
            data: Market data
            
        Returns:
            Dictionary of indicator series for plotting
        """
        if len(data) < self.params['long_window']:
            return {}
        
        return {
            'short_ma': data['close'].rolling(window=self.params['short_window']).mean(),
            'long_ma': data['close'].rolling(window=self.params['long_window']).mean()
        }
    
    def generate_vectorbt_signals(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals for VectorBT compatibility
        
        This uses the generic signals_to_vectorbt() function which works
        for ANY strategy by calling get_signals() point-by-point.
        
        Args:
            data: Market data (full historical dataset)
            
        Returns:
            Tuple of (entries, exits) as boolean Series for VectorBT
        """
        return signals_to_vectorbt(self, data)