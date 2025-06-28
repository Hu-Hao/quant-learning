"""
Technical Indicators
Collection of common technical analysis indicators
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


class TechnicalIndicators:
    """
    Technical indicators with proper error handling and validation
    """
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average
        
        Args:
            data: Price series
            period: Moving average period
            
        Returns:
            SMA series
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        if len(data) < period:
            return pd.Series(index=data.index, dtype=float)
        
        return data.rolling(window=period, min_periods=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int, alpha: Optional[float] = None) -> pd.Series:
        """
        Exponential Moving Average
        
        Args:
            data: Price series
            period: EMA period
            alpha: Smoothing factor (default: 2/(period+1))
            
        Returns:
            EMA series
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        if len(data) == 0:
            return pd.Series(index=data.index, dtype=float)
        
        if alpha is None:
            alpha = 2.0 / (period + 1)
        
        return data.ewm(alpha=alpha, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index
        
        Args:
            data: Price series
            period: RSI calculation period
            
        Returns:
            RSI series (0-100)
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        if len(data) < period + 1:
            return pd.Series(index=data.index, dtype=float)
        
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(
        data: pd.Series, 
        period: int = 20, 
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Bollinger Bands
        
        Args:
            data: Price series
            period: Moving average period
            std_dev: Standard deviation multiplier
            
        Returns:
            DataFrame with 'upper', 'middle', 'lower' columns
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        if std_dev <= 0:
            raise ValueError("Standard deviation multiplier must be positive")
        
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period, min_periods=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return pd.DataFrame({
            'upper': upper,
            'middle': sma,
            'lower': lower
        }, index=data.index)
    
    @staticmethod
    def macd(
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence)
        
        Args:
            data: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            DataFrame with 'macd', 'signal', 'histogram' columns
        """
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        ema_fast = TechnicalIndicators.ema(data, fast_period)
        ema_slow = TechnicalIndicators.ema(data, slow_period)
        
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }, index=data.index)
    
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """
        Stochastic Oscillator
        
        Args:
            high: High price series
            low: Low price series  
            close: Close price series
            k_period: %K calculation period
            d_period: %D smoothing period
            
        Returns:
            DataFrame with '%K' and '%D' columns
        """
        if k_period <= 0 or d_period <= 0:
            raise ValueError("Periods must be positive")
        
        lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
        highest_high = high.rolling(window=k_period, min_periods=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period, min_periods=d_period).mean()
        
        return pd.DataFrame({
            '%K': k_percent,
            '%D': d_percent
        }, index=close.index)
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Average True Range
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: ATR calculation period
            
        Returns:
            ATR series
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR as EMA of True Range
        atr = TechnicalIndicators.ema(true_range, period)
        
        return atr
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume
        
        Args:
            close: Close price series
            volume: Volume series
            
        Returns:
            OBV series
        """
        price_change = close.diff()
        
        # Determine direction
        direction = np.where(price_change > 0, 1,
                           np.where(price_change < 0, -1, 0))
        
        # Calculate OBV
        obv = (direction * volume).cumsum()
        
        return pd.Series(obv, index=close.index, name='OBV')
    
    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Williams %R
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            period: Calculation period
            
        Returns:
            Williams %R series (-100 to 0)
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        
        highest_high = high.rolling(window=period, min_periods=period).max()
        lowest_low = low.rolling(window=period, min_periods=period).min()
        
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def momentum(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Price Momentum
        
        Args:
            data: Price series
            period: Lookback period
            
        Returns:
            Momentum series (current price / price n periods ago)
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        
        return data / data.shift(period)
    
    @staticmethod
    def rate_of_change(data: pd.Series, period: int = 10) -> pd.Series:
        """
        Rate of Change (ROC)
        
        Args:
            data: Price series
            period: Lookback period
            
        Returns:
            ROC series as percentage
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        
        return ((data / data.shift(period)) - 1) * 100