"""
Mean Reversion Strategy
Trades based on price deviations from statistical mean
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from .strategy_interface import StrategyProtocol, Signal, SignalType, validate_data, create_signal, signals_to_vectorbt


class MeanReversionStrategy:
    """
    Mean Reversion Trading Strategy
    
    Buys when price is significantly below mean (oversold)
    Sells when price is significantly above mean (overbought)
    """
    
    def __init__(
        self,
        window: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        quantity: int = 100,
        use_bollinger_bands: bool = True,
        bollinger_std: float = 2.0
    ):
        """
        Initialize Mean Reversion Strategy
        
        Args:
            window: Period for calculating mean and standard deviation
            entry_threshold: Z-score threshold for entry signals (e.g., 2.0 = 2 std devs)
            exit_threshold: Z-score threshold for exit signals
            quantity: Number of shares to trade
            use_bollinger_bands: Whether to use Bollinger Bands for additional confirmation
            bollinger_std: Standard deviations for Bollinger Bands
        """
        self.name = "MeanReversion"
        self.params = {
            'window': window,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'quantity': quantity,
            'use_bollinger_bands': use_bollinger_bands,
            'bollinger_std': bollinger_std
        }
        
        # Validate parameters
        if window < 2:
            raise ValueError("Window must be at least 2")
        if entry_threshold <= exit_threshold:
            raise ValueError("Entry threshold must be greater than exit threshold")
    
    def get_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate mean reversion trading signals
        
        Args:
            data: Market data up to current time
            
        Returns:
            List of trading signals
        """
        validate_data(data)
        if len(data) < self.params['window'] + 1:
            return []
        
        signals = []
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1] if hasattr(data.index, '__getitem__') else None
        
        # Calculate z-score
        z_score = self._calculate_z_score(data)
        
        # Calculate Bollinger Bands if enabled
        bollinger_signal = None
        if self.params['use_bollinger_bands']:
            bollinger_signal = self._get_bollinger_signal(data)
        
        # Generate buy signal (price significantly below mean)
        if z_score < -self.params['entry_threshold']:
            # Confirm with Bollinger Bands if enabled
            if not self.params['use_bollinger_bands'] or bollinger_signal == 'buy':
                confidence = self._calculate_buy_confidence(z_score, data)
                
                signal = create_signal(
                    symbol='default',
                    action=SignalType.BUY,
                    quantity=self.params['quantity'],
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'z_score': z_score,
                        'signal_type': 'mean_reversion_buy',
                        'bollinger_confirmation': bollinger_signal == 'buy' if self.params['use_bollinger_bands'] else None
                    }
                )
                signal.timestamp = current_time
                signals.append(signal)
        
        # Generate sell signal (price significantly above mean)
        elif z_score > self.params['entry_threshold']:
            # Confirm with Bollinger Bands if enabled
            if not self.params['use_bollinger_bands'] or bollinger_signal == 'sell':
                confidence = self._calculate_sell_confidence(z_score, data)
                
                signal = create_signal(
                    symbol='default',
                    action=SignalType.SELL,
                    quantity=self.params['quantity'],
                    price=current_price,
                    confidence=confidence,
                    metadata={
                        'z_score': z_score,
                        'signal_type': 'mean_reversion_sell',
                        'bollinger_confirmation': bollinger_signal == 'sell' if self.params['use_bollinger_bands'] else None
                    }
                )
                signal.timestamp = current_time
                signals.append(signal)
        
        return signals
    
    def _calculate_z_score(self, data: pd.DataFrame) -> float:
        """
        Calculate z-score of current price relative to rolling mean
        
        Args:
            data: Market data
            
        Returns:
            Z-score of current price
        """
        window = self.params['window']
        recent_prices = data['close'].tail(window)
        
        mean_price = recent_prices.mean()
        std_price = recent_prices.std()
        current_price = data['close'].iloc[-1]
        
        if std_price == 0:
            return 0
        
        z_score = (current_price - mean_price) / std_price
        return z_score
    
    def _get_bollinger_signal(self, data: pd.DataFrame) -> Optional[str]:
        """
        Get Bollinger Bands signal
        
        Args:
            data: Market data
            
        Returns:
            'buy', 'sell', or None
        """
        window = self.params['window']
        std_mult = self.params['bollinger_std']
        
        recent_prices = data['close'].tail(window)
        
        sma = recent_prices.mean()
        std = recent_prices.std()
        
        upper_band = sma + (std * std_mult)
        lower_band = sma - (std * std_mult)
        
        current_price = data['close'].iloc[-1]
        
        if current_price <= lower_band:
            return 'buy'
        elif current_price >= upper_band:
            return 'sell'
        else:
            return None
    
    def _calculate_buy_confidence(self, z_score: float, data: pd.DataFrame) -> float:
        """
        Calculate confidence for buy signals
        
        Args:
            z_score: Current z-score
            data: Market data
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5
        
        # Higher negative z-score increases confidence
        z_factor = min(1.0, abs(z_score) / (self.params['entry_threshold'] * 1.5))
        confidence += z_factor * 0.3
        
        # Check if price is still falling (momentum confirmation)
        if len(data) >= 3:
            recent_momentum = (data['close'].iloc[-1] / data['close'].iloc[-3]) - 1
            if recent_momentum < 0:  # Price still falling
                confidence += 0.1
        
        # Check volume confirmation
        if len(data) >= 5:
            recent_volume = data['volume'].iloc[-3:].mean()
            avg_volume = data['volume'].tail(20).mean()
            if recent_volume > avg_volume:  # Higher volume on decline
                confidence += 0.1
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_sell_confidence(self, z_score: float, data: pd.DataFrame) -> float:
        """
        Calculate confidence for sell signals
        
        Args:
            z_score: Current z-score
            data: Market data
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5
        
        # Higher positive z-score increases confidence
        z_factor = min(1.0, z_score / (self.params['entry_threshold'] * 1.5))
        confidence += z_factor * 0.3
        
        # Check if price is still rising (momentum confirmation)
        if len(data) >= 3:
            recent_momentum = (data['close'].iloc[-1] / data['close'].iloc[-3]) - 1
            if recent_momentum > 0:  # Price still rising
                confidence += 0.1
        
        # Check volume confirmation
        if len(data) >= 5:
            recent_volume = data['volume'].iloc[-3:].mean()
            avg_volume = data['volume'].tail(20).mean()
            if recent_volume > avg_volume:  # Higher volume on rise
                confidence += 0.1
        
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
        if len(data) < self.params['window']:
            return pd.DataFrame()
        
        indicators = pd.DataFrame(index=data.index)
        window = self.params['window']
        
        # Calculate rolling statistics
        indicators['sma'] = data['close'].rolling(window=window).mean()
        indicators['std'] = data['close'].rolling(window=window).std()
        
        # Calculate z-score
        indicators['z_score'] = (data['close'] - indicators['sma']) / indicators['std']
        
        # Calculate Bollinger Bands
        std_mult = self.params['bollinger_std']
        indicators['bb_upper'] = indicators['sma'] + (indicators['std'] * std_mult)
        indicators['bb_lower'] = indicators['sma'] - (indicators['std'] * std_mult)
        indicators['bb_width'] = indicators['bb_upper'] - indicators['bb_lower']
        
        # Calculate position relative to bands
        indicators['bb_position'] = (data['close'] - indicators['bb_lower']) / indicators['bb_width']
        
        return indicators
    
    def get_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get technical indicators for visualization (StrategyProtocol interface)
        
        Args:
            data: Market data
            
        Returns:
            Dictionary of indicator series for plotting
        """
        if len(data) < self.params['window']:
            return {}
        
        indicators = self.get_indicator_values(data)
        
        return {
            'sma': indicators['sma'],
            'bb_upper': indicators['bb_upper'],
            'bb_lower': indicators['bb_lower']
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