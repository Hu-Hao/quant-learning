"""
Momentum Strategy
Trades based on recent price momentum and trend strength
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from .strategy_interface import StrategyProtocol, Signal, SignalType, validate_data, create_signal, signals_to_vectorbt


class MomentumStrategy:
    """
    Momentum Trading Strategy
    
    Buys when recent momentum is positive and strong
    Sells when recent momentum is negative and strong
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        momentum_threshold: float = 0.02,
        quantity: Optional[int] = None,
        percent_capital: Optional[float] = None,
        volatility_filter: bool = True,
        rsi_period: int = 14,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70
    ):
        """
        Initialize Momentum Strategy
        
        Args:
            lookback_period: Number of periods to calculate momentum
            momentum_threshold: Minimum momentum required for signal (e.g., 0.02 = 2%)
            quantity: Fixed number of shares to trade (if specified)
            percent_capital: Percentage of capital to use (0.0 to 1.0, if specified)
            volatility_filter: Whether to filter signals by volatility
            rsi_period: RSI calculation period
            rsi_oversold: RSI oversold threshold
            rsi_overbought: RSI overbought threshold
            
        Position Sizing Logic:
            1. If quantity is set: Use fixed quantity (e.g., quantity=100)
            2. If percent_capital is set: Use percentage of capital (e.g., percent_capital=0.1 for 10%)
            3. If neither is set: Use 100% of available capital (VectorBT default)
        """
        self.name = "Momentum"
        self.params = {
            'lookback_period': lookback_period,
            'momentum_threshold': momentum_threshold,
            'quantity': quantity,
            'percent_capital': percent_capital,
            'volatility_filter': volatility_filter,
            'rsi_period': rsi_period,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought
        }
        
        # Validate parameters
        if lookback_period < 2:
            raise ValueError("Lookback period must be at least 2")
        
        # Validate position sizing parameters
        if quantity is not None and percent_capital is not None:
            raise ValueError("Cannot specify both quantity and percent_capital")
        
        if percent_capital is not None and (percent_capital <= 0 or percent_capital > 1):
            raise ValueError("percent_capital must be between 0 and 1")
        if not 0 < momentum_threshold < 1:
            raise ValueError("Momentum threshold must be between 0 and 1")
    
    def get_position_size(self, current_price: float, available_capital: float) -> int:
        """
        Calculate position size based on strategy configuration
        
        Args:
            current_price: Current market price
            available_capital: Available capital for trading
            
        Returns:
            Number of shares to trade
        """
        # Case 1: Fixed quantity specified
        if self.params['quantity'] is not None:
            return self.params['quantity']
        
        # Case 2: Percentage of capital specified
        if self.params['percent_capital'] is not None:
            return max(1, int(available_capital * self.params['percent_capital'] / current_price))
        
        # Case 3: Default to 100% capital (VectorBT style)
        return max(1, int(available_capital / current_price))
    
    def get_signals(self, data: pd.DataFrame, available_capital: float = 100000) -> List[Signal]:
        """
        Generate momentum-based trading signals
        
        Args:
            data: Market data up to current time
            available_capital: Available capital for position sizing
            
        Returns:
            List of trading signals
        """
        validate_data(data)
        min_periods = max(self.params['lookback_period'], self.params['rsi_period']) + 1
        if len(data) < min_periods:
            return []
        
        signals = []
        current_price = data['close'].iloc[-1]
        current_time = data.index[-1] if hasattr(data.index, '__getitem__') else None
        
        # Calculate momentum
        momentum = self._calculate_momentum(data)
        
        # Calculate RSI for additional confirmation
        rsi = self._calculate_rsi(data)
        
        # Calculate volatility if filter is enabled
        volatility_ok = True
        if self.params['volatility_filter']:
            volatility_ok = self._check_volatility_filter(data)
        
        # Generate buy signal
        if (momentum > self.params['momentum_threshold'] and 
            rsi < self.params['rsi_overbought'] and
            volatility_ok):
            
            confidence = self._calculate_buy_confidence(momentum, rsi, data)
            
            position_size = self.get_position_size(current_price, available_capital)
            signal = create_signal(
                symbol='default',
                action=SignalType.BUY,
                quantity=position_size,
                price=current_price,
                confidence=confidence,
                metadata={
                    'momentum': momentum,
                    'rsi': rsi,
                    'signal_type': 'momentum_buy',
                    'volatility_filtered': self.params['volatility_filter']
                }
            )
            signal.timestamp = current_time
            signals.append(signal)
            
        # Generate sell signal
        elif (momentum < -self.params['momentum_threshold'] and 
              rsi > self.params['rsi_oversold'] and
              volatility_ok):
            
            confidence = self._calculate_sell_confidence(momentum, rsi, data)
            
            position_size = self.get_position_size(current_price, available_capital)
            signal = create_signal(
                symbol='default',
                action=SignalType.SELL,
                quantity=position_size,
                price=current_price,
                confidence=confidence,
                metadata={
                    'momentum': momentum,
                    'rsi': rsi,
                    'signal_type': 'momentum_sell',
                    'volatility_filtered': self.params['volatility_filter']
                }
            )
            signal.timestamp = current_time
            signals.append(signal)
            
        return signals
    
    def _calculate_momentum(self, data: pd.DataFrame) -> float:
        """
        Calculate price momentum over lookback period
        
        Args:
            data: Market data
            
        Returns:
            Momentum as percentage change
        """
        lookback = self.params['lookback_period']
        current_price = data['close'].iloc[-1]
        past_price = data['close'].iloc[-lookback-1]
        
        momentum = (current_price / past_price) - 1
        return momentum
    
    def _calculate_rsi(self, data: pd.DataFrame) -> float:
        """
        Calculate Relative Strength Index
        
        Args:
            data: Market data
            
        Returns:
            RSI value (0-100)
        """
        period = self.params['rsi_period']
        delta = data['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain.iloc[-1] / avg_loss.iloc[-1] if avg_loss.iloc[-1] != 0 else 100
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _check_volatility_filter(self, data: pd.DataFrame) -> bool:
        """
        Check if current volatility is within acceptable range
        
        Args:
            data: Market data
            
        Returns:
            True if volatility is acceptable
        """
        # Calculate recent volatility (20-day)
        returns = data['close'].pct_change()
        recent_vol = returns.tail(20).std() * np.sqrt(252)  # Annualized
        long_term_vol = returns.std() * np.sqrt(252)
        
        # Filter out periods of extreme volatility (> 1.5x normal)
        return recent_vol <= long_term_vol * 1.5
    
    def _calculate_buy_confidence(self, momentum: float, rsi: float, data: pd.DataFrame) -> float:
        """Calculate confidence for buy signals"""
        confidence = 0.5
        
        # Higher momentum increases confidence
        momentum_factor = min(1.0, momentum / (self.params['momentum_threshold'] * 2))
        confidence += momentum_factor * 0.3
        
        # RSI not being overbought increases confidence
        rsi_factor = (100 - rsi) / 100
        confidence += rsi_factor * 0.2
        
        return max(0.1, min(1.0, confidence))
    
    def _calculate_sell_confidence(self, momentum: float, rsi: float, data: pd.DataFrame) -> float:
        """Calculate confidence for sell signals"""
        confidence = 0.5
        
        # Stronger negative momentum increases confidence
        momentum_factor = min(1.0, abs(momentum) / (self.params['momentum_threshold'] * 2))
        confidence += momentum_factor * 0.3
        
        # RSI not being oversold increases confidence
        rsi_factor = rsi / 100
        confidence += rsi_factor * 0.2
        
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
        min_periods = max(self.params['lookback_period'], self.params['rsi_period']) + 1
        if len(data) < min_periods:
            return pd.DataFrame()
        
        indicators = pd.DataFrame(index=data.index)
        
        # Calculate rolling momentum
        lookback = self.params['lookback_period']
        indicators['momentum'] = data['close'].pct_change(periods=lookback)
        
        # Calculate RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.params['rsi_period']).mean()
        avg_loss = loss.rolling(window=self.params['rsi_period']).mean()
        
        rs = avg_gain / avg_loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate volatility
        returns = data['close'].pct_change()
        indicators['volatility'] = returns.rolling(window=20).std() * np.sqrt(252)
        
        return indicators
    
    def get_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get technical indicators for visualization (StrategyProtocol interface)
        
        Args:
            data: Market data
            
        Returns:
            Dictionary of indicator series for plotting
        """
        if len(data) < self.params['lookback_period']:
            return {}
        
        indicators = self.get_indicator_values(data)
        
        return {
            'momentum': indicators['momentum'],
            'rsi': indicators['rsi']
        }
    
    def get_vectorbt_position_sizing(self, data: pd.DataFrame, init_cash: float = 100000) -> Dict[str, Any]:
        """
        Get position sizing parameters for VectorBT Portfolio.from_signals()
        
        Args:
            data: Market data
            init_cash: Initial capital for percentage calculations
            
        Returns:
            Dictionary with VectorBT sizing parameters
        """
        # Case 1: Fixed quantity
        if self.params['quantity'] is not None:
            return {
                'size': self.params['quantity'],
                'size_type': 'shares'
            }
        
        # Case 2: Percentage of capital
        if self.params['percent_capital'] is not None:
            return {
                'size': self.params['percent_capital'],
                'size_type': 'percent'
            }
        
        # Case 3: Full capital (VectorBT default)
        return {
            'size': 1.0,  # 100% of available capital
            'size_type': 'percent'
        }
    
    def generate_vectorbt_signals(self, data: pd.DataFrame, available_capital: float = 100000) -> tuple[pd.Series, pd.Series]:
        """
        Generate entry and exit signals for VectorBT compatibility
        
        This uses the generic signals_to_vectorbt() function which works
        for ANY strategy by calling get_signals() point-by-point.
        
        Args:
            data: Market data (full historical dataset)
            available_capital: Available capital for position sizing
            
        Returns:
            Tuple of (entries, exits) as boolean Series for VectorBT
        """
        return signals_to_vectorbt(self, data, available_capital)