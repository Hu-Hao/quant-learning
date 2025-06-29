"""
Enhanced Backtesting Engine
Production-ready backtesting with proper execution modeling and risk controls
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..strategies.strategy_interface import StrategyProtocol, Signal
from dataclasses import dataclass
import logging


@dataclass
class Trade:
    """Record of a completed trade with full metadata"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # 'long' or 'short'
    pnl: float
    commission_paid: float
    slippage_cost: float
    
    @property
    def gross_pnl(self) -> float:
        """P&L before costs"""
        if self.side == 'long':
            return self.quantity * (self.exit_price - self.entry_price)
        else:
            return self.quantity * (self.entry_price - self.exit_price)
    
    @property
    def total_costs(self) -> float:
        """Total transaction costs"""
        return self.commission_paid + self.slippage_cost
    
    @property
    def return_pct(self) -> float:
        """Return as percentage of entry value"""
        entry_value = abs(self.quantity * self.entry_price)
        return self.pnl / entry_value if entry_value > 0 else 0


@dataclass
class Position:
    """Current position with risk tracking"""
    symbol: str
    quantity: int  # Positive for long, negative for short
    avg_price: float
    side: str
    entry_time: datetime
    
    @property
    def value(self) -> float:
        """Position value at average price"""
        return abs(self.quantity) * self.avg_price
    
    def market_value(self, current_price: float) -> float:
        """Current market value"""
        return self.quantity * current_price
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Unrealized P&L at current price"""
        if self.side == 'long':
            return self.quantity * (current_price - self.avg_price)
        else:
            return abs(self.quantity) * (self.avg_price - current_price)


class BacktestEngine:
    """
    Production-grade backtesting engine with realistic execution modeling
    
    Supports only StrategyProtocol-based strategies for clean, simple interface.
    Uses composition over inheritance for maximum flexibility.
    
    Beginner-friendly features:
    - Set allow_short_selling=False to disable short selling
    - Sell signals will only close existing long positions when short selling is disabled
    - Perfect for learning without the complexity of short selling
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.001,
        max_position_size: float = 0.1,  # Maximum position as fraction of capital
        risk_free_rate: float = 0.02,    # For Sharpe ratio calculation
        benchmark_symbol: Optional[str] = None,
        allow_short_selling: bool = True  # Allow short selling (sell signals)
    ):
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital
            commission: Commission as fraction of trade value
            slippage: Base slippage as fraction of price
            max_position_size: Maximum position size as fraction of capital
            risk_free_rate: Risk-free rate for Sharpe ratio
            benchmark_symbol: Symbol for benchmark comparison
            allow_short_selling: Allow short selling (False for beginners)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
        self.allow_short_selling = allow_short_selling
        self.benchmark_symbol = benchmark_symbol
        
        # State variables
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_values: List[float] = []
        self.timestamps: List[datetime] = []
        self.current_time: Optional[datetime] = None
        self.price_history: List[float] = []
        
        # Risk tracking
        self.max_drawdown_seen = 0.0
        self.peak_value = initial_capital
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def reset(self) -> None:
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.portfolio_values.clear()
        self.timestamps.clear()
        self.price_history.clear()
        self.current_time = None
        self.max_drawdown_seen = 0.0
        self.peak_value = self.initial_capital
        
    def update_time(self, timestamp: Union[datetime, pd.Timestamp, int]) -> None:
        """Update current timestamp"""
        if isinstance(timestamp, (int, np.integer)):
            self.current_time = datetime.now() + timedelta(days=timestamp)
        elif isinstance(timestamp, pd.Timestamp):
            self.current_time = timestamp.to_pydatetime()
        else:
            self.current_time = timestamp
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol"""
        return self.positions.get(symbol)
    
    def get_total_exposure(self) -> float:
        """Get total portfolio exposure"""
        total_exposure = 0.0
        for position in self.positions.values():
            total_exposure += position.value
        return total_exposure
    
    def _calculate_slippage(self, price: float, side: str, quantity: int) -> float:
        """
        Calculate variable slippage based on market conditions
        
        Args:
            price: Order price
            side: 'long' or 'short'
            quantity: Order quantity
            
        Returns:
            Execution price after slippage
        """
        base_slippage = self.slippage
        
        # Scale slippage based on recent volatility
        if len(self.price_history) >= 10:
            recent_prices = np.array(self.price_history[-10:])
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns)
            
            # Volatility multiplier (0.5x to 3x)
            vol_multiplier = min(3.0, max(0.5, volatility * 50))
            adjusted_slippage = base_slippage * vol_multiplier
        else:
            adjusted_slippage = base_slippage
        
        # Scale by order size (larger orders have more slippage)
        order_value = abs(quantity) * price
        size_multiplier = 1.0 + (order_value / self.capital) * 0.5
        final_slippage = adjusted_slippage * size_multiplier
        
        # Apply slippage direction
        if side == 'long':
            return price * (1 + final_slippage)  # Pay more when buying
        else:
            return price * (1 - final_slippage)  # Receive less when selling
    
    def place_order(
        self,
        symbol: str,
        quantity: int,
        price: float,
        side: str,
        order_type: str = 'market'
    ) -> bool:
        """
        Place trading order with realistic execution
        
        Args:
            symbol: Trading symbol
            quantity: Number of shares (positive)
            price: Order price
            side: 'long' or 'short'
            order_type: Order type ('market', 'limit')
            
        Returns:
            True if order executed successfully
        """
        try:
            if side not in ['long', 'short']:
                raise ValueError("Side must be 'long' or 'short'")
            
            if quantity <= 0:
                raise ValueError("Quantity must be positive")
            
            # Apply slippage to execution price
            execution_price = self._calculate_slippage(price, side, quantity)
            
            # Calculate costs
            trade_value = quantity * execution_price
            commission_cost = trade_value * self.commission
            slippage_cost = quantity * abs(execution_price - price)
            
            # Check position size limits
            if not self._check_position_limits(symbol, quantity, execution_price, side):
                self.logger.warning(f"Order rejected: position size limit exceeded")
                return False
            
            # Check capital requirements for long positions
            if side == 'long':
                total_cost = trade_value + commission_cost
                if self.capital < total_cost:
                    self.logger.warning(f"Order rejected: insufficient capital")
                    return False
            
            # Execute the order
            return self._execute_order(
                symbol, quantity, execution_price, side, 
                commission_cost, slippage_cost
            )
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return False
    
    def _check_position_limits(
        self, 
        symbol: str, 
        quantity: int, 
        price: float, 
        side: str
    ) -> bool:
        """Check if order respects position size limits"""
        order_value = quantity * price
        max_position_value = self.get_portfolio_value() * self.max_position_size
        
        current_position = self.positions.get(symbol)
        if current_position is None:
            # New position
            return order_value <= max_position_value
        else:
            # Adding to existing position
            if ((current_position.side == side) or 
                (current_position.side != side and quantity <= abs(current_position.quantity))):
                # Same side or reducing position
                new_position_value = abs(current_position.quantity + 
                                       (quantity if side == 'long' else -quantity)) * price
                return new_position_value <= max_position_value
            else:
                # Reversing position
                return order_value <= max_position_value
    
    def _execute_order(
        self,
        symbol: str,
        quantity: int,
        execution_price: float,
        side: str,
        commission_cost: float,
        slippage_cost: float
    ) -> bool:
        """Execute the validated order"""
        
        signed_quantity = quantity if side == 'long' else -quantity
        current_pos = self.positions.get(symbol)
        
        if current_pos is None:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=signed_quantity,
                avg_price=execution_price,
                side=side,
                entry_time=self.current_time or datetime.now()
            )
            
            # Update capital
            if side == 'long':
                self.capital -= (quantity * execution_price + commission_cost)
            else:
                self.capital += (quantity * execution_price - commission_cost)
                
        else:
            # Modify existing position
            new_quantity = current_pos.quantity + signed_quantity
            
            if new_quantity == 0:
                # Close position completely
                pnl = self._calculate_position_pnl(current_pos, execution_price)
                net_pnl = pnl - commission_cost
                
                # Record completed trade
                self.trades.append(Trade(
                    symbol=symbol,
                    entry_time=current_pos.entry_time,
                    exit_time=self.current_time or datetime.now(),
                    entry_price=current_pos.avg_price,
                    exit_price=execution_price,
                    quantity=abs(current_pos.quantity),
                    side=current_pos.side,
                    pnl=net_pnl,
                    commission_paid=commission_cost,
                    slippage_cost=slippage_cost
                ))
                
                self.capital += net_pnl
                del self.positions[symbol]
                
            elif ((current_pos.quantity > 0 and new_quantity > 0) or 
                  (current_pos.quantity < 0 and new_quantity < 0)):
                # Adding to position (same direction)
                total_cost = (current_pos.quantity * current_pos.avg_price + 
                             signed_quantity * execution_price)
                current_pos.avg_price = total_cost / new_quantity
                current_pos.quantity = new_quantity
                
                # Update capital
                if side == 'long':
                    self.capital -= (quantity * execution_price + commission_cost)
                else:
                    self.capital += (quantity * execution_price - commission_cost)
                    
            else:
                # Reducing position (opposite direction)
                close_quantity = min(abs(current_pos.quantity), quantity)
                partial_pnl = self._calculate_partial_pnl(
                    current_pos, execution_price, close_quantity
                )
                
                self.capital += partial_pnl - commission_cost
                current_pos.quantity = new_quantity
                
                # If position flipped sides, update metadata
                if new_quantity != 0 and np.sign(new_quantity) != np.sign(current_pos.quantity):
                    current_pos.side = 'long' if new_quantity > 0 else 'short'
                    current_pos.avg_price = execution_price
                    current_pos.entry_time = self.current_time or datetime.now()
        
        return True
    
    def _calculate_position_pnl(self, position: Position, exit_price: float) -> float:
        """Calculate P&L for position closure"""
        if position.side == 'long':
            return position.quantity * (exit_price - position.avg_price)
        else:
            return abs(position.quantity) * (position.avg_price - exit_price)
    
    def _calculate_partial_pnl(
        self, 
        position: Position, 
        exit_price: float, 
        quantity: int
    ) -> float:
        """Calculate P&L for partial position closure"""
        if position.side == 'long':
            return quantity * (exit_price - position.avg_price)
        else:
            return quantity * (position.avg_price - exit_price)
    
    def get_portfolio_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate current portfolio value
        
        Args:
            prices: Current market prices for positions
            
        Returns:
            Total portfolio value
        """
        total_value = self.capital
        
        if prices:
            for symbol, position in self.positions.items():
                if symbol in prices:
                    market_value = position.market_value(prices[symbol])
                    total_value += market_value
        
        return total_value
    
    def update_portfolio_value(self, prices: Dict[str, float]) -> None:
        """Update and record portfolio value"""
        portfolio_value = self.get_portfolio_value(prices)
        self.portfolio_values.append(portfolio_value)
        
        if self.current_time:
            self.timestamps.append(self.current_time)
        
        # Update drawdown tracking
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        
        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        self.max_drawdown_seen = max(self.max_drawdown_seen, current_drawdown)
    
    def run_backtest(
        self,
        data: pd.DataFrame,
        strategy: 'StrategyProtocol'
    ) -> None:
        """
        Run backtest with protocol-based strategy
        
        Args:
            data: Market data
            strategy: Strategy implementing StrategyProtocol interface
        """
        self.reset()
        
        for idx, row in data.iterrows():
            self.update_time(row.name if hasattr(row.name, 'date') else idx)
            
            # Track price history
            current_price = row.get('close', row.iloc[-1])
            self.price_history.append(current_price)
            
            # Get strategy signals
            signals = strategy.get_signals(data.loc[:idx])
            
            # Process signals
            if signals:
                for signal in signals:
                    self._process_signal(signal, current_price)
            
            # Update portfolio value
            prices = {'default': current_price}
            self.update_portfolio_value(prices)
    
    def _process_signal(self, signal: 'Signal', current_price: float) -> None:
        """
        Process a Signal object from StrategyProtocol
        
        Args:
            signal: Signal object with standardized interface
            current_price: Current market price for fallback
        """
        # Extract signal data
        symbol = signal.symbol
        action = signal.action.value if hasattr(signal.action, 'value') else signal.action
        quantity = signal.quantity
        price = signal.price
        
        # Convert action to string and normalize
        action_str = action.lower() if isinstance(action, str) else str(action).lower()
        
        # Execute trades based on signal
        if action_str == 'buy':
            self.place_order(symbol, quantity, price, 'long')
        elif action_str == 'sell':
            if self.allow_short_selling:
                # Allow short selling (creating new short position)
                self.place_order(symbol, quantity, price, 'short')
            else:
                # Only allow selling to close existing long positions
                current_position = self.positions.get(symbol)
                if current_position and current_position.side == 'long' and current_position.quantity > 0:
                    # Close long position (sell shares we own)
                    close_quantity = min(quantity, current_position.quantity)
                    self.place_order(symbol, close_quantity, price, 'short')
                    self.logger.info(f"Sell signal: Closing {close_quantity} shares of long position in {symbol}")
                else:
                    # Ignore sell signal - no long position to close
                    self.logger.info(f"Sell signal ignored: No long position in {symbol} (short selling disabled)")
    
    def get_performance_summary(self) -> Dict:
        """Generate comprehensive performance summary"""
        if not self.portfolio_values:
            return {}
        
        returns_series = pd.Series(self.portfolio_values)
        pct_returns = returns_series.pct_change().dropna()
        
        # Basic metrics
        total_return = (returns_series.iloc[-1] - returns_series.iloc[0]) / returns_series.iloc[0]
        
        # Annualized metrics
        trading_days = len(returns_series)
        years = trading_days / 252
        annualized_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        
        # Risk metrics
        volatility = pct_returns.std() * np.sqrt(252) if len(pct_returns) > 1 else 0
        sharpe_ratio = ((annualized_return - self.risk_free_rate) / volatility 
                       if volatility != 0 else 0)
        
        # Drawdown
        running_max = returns_series.expanding().max()
        drawdown = (returns_series - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Trade statistics
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0
            profit_factor = avg_win / avg_loss if avg_loss != 0 else float('inf')
            
            total_commission = sum(t.commission_paid for t in self.trades)
            total_slippage = sum(t.slippage_cost for t in self.trades)
        else:
            win_rate = 0
            profit_factor = 0
            total_commission = 0
            total_slippage = 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'final_capital': returns_series.iloc[-1],
            'initial_capital': self.initial_capital
        }