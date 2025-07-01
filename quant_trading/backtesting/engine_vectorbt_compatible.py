"""
VectorBT-compatible backtesting engine
Enhanced version that better matches VectorBT behavior
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    from ..strategies.strategy_interface import StrategyProtocol, Signal

from .engine import BacktestEngine, Trade, Position


class VectorBTCompatibleEngine(BacktestEngine):
    """
    Enhanced backtesting engine for better VectorBT compatibility
    
    Key improvements:
    - Better position size handling
    - More flexible trade execution
    - Automatic quantity adjustment
    - VectorBT-like capital allocation
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.001,
        max_position_size: float = 0.95,  # Default to 95%
        risk_free_rate: float = 0.02,
        benchmark_symbol: Optional[str] = None,
        allow_short_selling: bool = True,
        vectorbt_mode: bool = True,  # Enable VectorBT-like behavior
        auto_size_positions: bool = True,  # Automatically size positions
        min_trade_value: float = 100,  # Minimum trade value
    ):
        """
        Initialize VectorBT-compatible engine
        
        Args:
            vectorbt_mode: Enable VectorBT-like behavior
            auto_size_positions: Automatically adjust position sizes
            min_trade_value: Minimum trade value to execute
        """
        super().__init__(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            max_position_size=max_position_size,
            risk_free_rate=risk_free_rate,
            benchmark_symbol=benchmark_symbol,
            allow_short_selling=allow_short_selling
        )
        
        self.vectorbt_mode = vectorbt_mode
        self.auto_size_positions = auto_size_positions
        self.min_trade_value = min_trade_value
        
        # Track additional metrics for VectorBT compatibility
        self.signal_log = []  # Log all signals for debugging
        self.rejected_orders = []  # Track rejected orders
    
    def _calculate_optimal_quantity(
        self, 
        symbol: str, 
        price: float, 
        side: str, 
        requested_quantity: int
    ) -> int:
        """
        Calculate optimal quantity that fits within constraints
        
        Args:
            symbol: Trading symbol
            price: Order price
            side: 'long' or 'short'
            requested_quantity: Originally requested quantity
            
        Returns:
            Optimal quantity to trade
        """
        if not self.auto_size_positions:
            return requested_quantity
        
        # Get available capital
        if side == 'long':
            available_capital = self.capital
        else:
            # For short positions, consider existing positions
            current_position = self.positions.get(symbol)
            if current_position and current_position.side == 'long':
                # Closing long position - use position size
                return min(requested_quantity, current_position.quantity)
            else:
                # Opening short position
                available_capital = self.capital
        
        # Calculate maximum quantity based on position size limits
        current_portfolio_value = self.get_portfolio_value()
        max_position_value = current_portfolio_value * self.max_position_size
        
        # Account for existing position
        current_position = self.positions.get(symbol)
        if current_position and current_position.side == side:
            current_position_value = abs(current_position.quantity) * price
            remaining_position_value = max_position_value - current_position_value
        else:
            remaining_position_value = max_position_value
        
        # Calculate max quantity based on available capital and position limits
        max_qty_by_capital = int(available_capital / (price * (1 + self.commission)))
        max_qty_by_position = int(remaining_position_value / price)
        
        # Take the minimum of all constraints
        optimal_quantity = min(
            requested_quantity,
            max_qty_by_capital,
            max_qty_by_position
        )
        
        # Ensure minimum trade value
        min_quantity = max(1, int(self.min_trade_value / price))
        
        if optimal_quantity < min_quantity:
            return 0  # Don't trade if below minimum
        
        return max(1, optimal_quantity)
    
    def _process_signal(self, signal: 'Signal', current_price: float) -> None:
        """
        Enhanced signal processing with VectorBT compatibility
        
        Args:
            signal: Signal object with standardized interface
            current_price: Current market price for fallback
        """
        # Log signal for debugging
        self.signal_log.append({
            'timestamp': self.current_time,
            'action': signal.action.value,
            'quantity': signal.quantity,
            'price': signal.price,
            'symbol': signal.symbol
        })
        
        # Extract signal data
        symbol = signal.symbol
        action = signal.action.value if hasattr(signal.action, 'value') else signal.action
        quantity = signal.quantity
        price = signal.price
        
        # Convert action to string and normalize
        action_str = action.lower() if isinstance(action, str) else str(action).lower()
        
        # Calculate optimal quantity if auto-sizing is enabled
        if self.auto_size_positions:
            if action_str == 'buy':
                optimal_quantity = self._calculate_optimal_quantity(symbol, price, 'long', quantity)
            elif action_str == 'sell':
                optimal_quantity = self._calculate_optimal_quantity(symbol, price, 'short', quantity)
            else:
                optimal_quantity = quantity
            
            if optimal_quantity != quantity and optimal_quantity > 0:
                self.logger.info(f"Auto-sized quantity from {quantity} to {optimal_quantity}")
                quantity = optimal_quantity
            elif optimal_quantity == 0:
                self.rejected_orders.append({
                    'reason': 'Below minimum trade value',
                    'signal': signal,
                    'timestamp': self.current_time
                })
                return
        
        # Execute trades based on signal
        success = False
        
        if action_str == 'buy':
            success = self.place_order(symbol, quantity, price, 'long')
            
        elif action_str == 'sell':
            if self.allow_short_selling:
                # Check if we have a long position to close first
                current_position = self.positions.get(symbol)
                if current_position and current_position.side == 'long' and current_position.quantity > 0:
                    # Close long position first
                    close_quantity = min(quantity, current_position.quantity)
                    success = self.place_order(symbol, close_quantity, price, 'short')
                    
                    # If there's remaining quantity and we want to go short
                    remaining_quantity = quantity - close_quantity
                    if remaining_quantity > 0 and self.vectorbt_mode:
                        # Open short position with remaining quantity
                        remaining_optimal = self._calculate_optimal_quantity(
                            symbol, price, 'short', remaining_quantity
                        )
                        if remaining_optimal > 0:
                            self.place_order(symbol, remaining_optimal, price, 'short')
                else:
                    # No long position, open short position
                    success = self.place_order(symbol, quantity, price, 'short')
            else:
                # Only allow selling to close existing long positions
                current_position = self.positions.get(symbol)
                if current_position and current_position.side == 'long' and current_position.quantity > 0:
                    close_quantity = min(quantity, current_position.quantity)
                    success = self.place_order(symbol, close_quantity, price, 'short')
                else:
                    self.rejected_orders.append({
                        'reason': 'No long position to close (short selling disabled)',
                        'signal': signal,
                        'timestamp': self.current_time
                    })
        
        if not success:
            self.rejected_orders.append({
                'reason': 'Order execution failed',
                'signal': signal,
                'timestamp': self.current_time
            })
    
    def place_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        side: str,
        order_type: str = 'market'
    ) -> bool:
        """
        Enhanced order placement with better error handling
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares (positive)
            price: Order price
            side: 'long' or 'short'
            order_type: Order type ('market', 'limit')
            
        Returns:
            True if order executed successfully
        """
        if quantity <= 0:
            self.logger.warning(f"Invalid quantity: {quantity}")
            return False
        
        try:
            # Use parent class logic but with enhanced error handling
            return super().place_order(symbol, quantity, price, side, order_type)
            
        except Exception as e:
            self.logger.error(f"Enhanced order placement failed: {e}")
            return False
    
    def get_vectorbt_compatible_stats(self) -> Dict:
        """
        Get statistics in VectorBT-compatible format
        
        Returns:
            Dictionary with VectorBT-style statistics
        """
        performance = self.get_performance_summary()
        
        stats = {
            'Total Return [%]': performance.get('total_return', 0) * 100,
            'Total Trades': len(self.trades),
            'Win Rate [%]': 0,
            'Sharpe Ratio': performance.get('sharpe_ratio', 0),
            'Max Drawdown [%]': performance.get('max_drawdown', 0) * 100,
            'Final Value': self.portfolio_values[-1] if self.portfolio_values else self.initial_capital,
            'Start Value': self.initial_capital,
        }
        
        # Calculate win rate
        if self.trades:
            winning_trades = sum(1 for trade in self.trades if trade.pnl > 0)
            stats['Win Rate [%]'] = (winning_trades / len(self.trades)) * 100
        
        return stats
    
    def get_debug_info(self) -> Dict:
        """
        Get debugging information for troubleshooting
        
        Returns:
            Dictionary with debugging information
        """
        return {
            'signals_received': len(self.signal_log),
            'trades_executed': len(self.trades),
            'orders_rejected': len(self.rejected_orders),
            'rejection_reasons': [order['reason'] for order in self.rejected_orders],
            'vectorbt_mode': self.vectorbt_mode,
            'auto_size_positions': self.auto_size_positions,
            'max_position_size': self.max_position_size,
            'current_capital': self.capital,
            'portfolio_value': self.get_portfolio_value(),
        }


def create_vectorbt_compatible_engine(**kwargs) -> VectorBTCompatibleEngine:
    """
    Factory function to create VectorBT-compatible engine with good defaults
    
    Args:
        **kwargs: Additional arguments for engine configuration
        
    Returns:
        Configured VectorBTCompatibleEngine
    """
    default_kwargs = {
        'initial_capital': 100000,
        'commission': 0.001,
        'slippage': 0.0,  # No slippage for exact comparison
        'max_position_size': 0.95,
        'allow_short_selling': True,
        'vectorbt_mode': True,
        'auto_size_positions': True,
        'min_trade_value': 50,
    }
    
    # Update defaults with provided kwargs
    default_kwargs.update(kwargs)
    
    return VectorBTCompatibleEngine(**default_kwargs)