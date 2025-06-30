"""
Strategy Interface using Protocols (no inheritance)
Clean interface definition without inheritance constraints
"""

from typing import Protocol, List, Dict, Any, Optional, TYPE_CHECKING
import pandas as pd
from dataclasses import dataclass
from enum import Enum

# Optional Backtrader import for type hints
if TYPE_CHECKING:
    try:
        import backtrader as bt
    except ImportError:
        bt = None


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
    
    This interface now supports both our framework and Backtrader for 
    cross-validation and learning from industry standards.
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
    
    # Optional Backtrader compatibility methods
    def to_backtrader_strategy(self, **kwargs) -> 'BacktraderStrategyWrapper':
        """
        Convert this strategy to work with Backtrader framework
        
        Returns:
            Backtrader strategy class that wraps this strategy
        """
        return BacktraderStrategyWrapper(self, **kwargs)
    
    def get_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get technical indicators for visualization (optional)
        
        Args:
            data: Market data
            
        Returns:
            Dictionary of indicator series (e.g., {'ma_short': series, 'ma_long': series})
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


# Backtrader integration (optional - only works if backtrader is installed)
class BacktraderStrategyWrapper:
    """
    Wrapper to make our StrategyProtocol strategies work with Backtrader
    
    This enables side-by-side comparison between our framework and Backtrader
    without requiring complex adapters or inheritance.
    """
    
    def __init__(self, strategy_protocol: StrategyProtocol, **kwargs):
        self.strategy_protocol = strategy_protocol
        self.kwargs = kwargs
        
    def create_backtrader_strategy(self):
        """Create Backtrader strategy class that wraps our strategy"""
        
        try:
            import backtrader as bt
        except ImportError:
            raise ImportError("Backtrader not installed. Install with: pip install backtrader")
        
        strategy_protocol = self.strategy_protocol
        
        class WrappedStrategy(bt.Strategy):
            """Backtrader strategy that uses our StrategyProtocol"""
            
            def __init__(self):
                super().__init__()
                self.data_history = []
                self.last_signals = []
                
            def next(self):
                """Called for each bar - convert to our format and get signals"""
                # Build current data state
                current_data = {
                    'open': self.data.open[0],
                    'high': self.data.high[0],
                    'low': self.data.low[0], 
                    'close': self.data.close[0],
                    'volume': self.data.volume[0] if hasattr(self.data, 'volume') else 1000000,
                    'datetime': self.data.datetime.datetime(0)
                }
                self.data_history.append(current_data)
                
                # Convert to DataFrame
                df = pd.DataFrame(self.data_history)
                df.set_index('datetime', inplace=True)
                
                # Get signals from our strategy
                try:
                    signals = strategy_protocol.get_signals(df)
                    
                    # Process new signals
                    new_signals = signals[len(self.last_signals):]
                    
                    for signal in new_signals:
                        self._execute_signal(signal)
                    
                    self.last_signals = signals
                    
                except Exception:
                    # Strategy might need more data
                    pass
            
            def _execute_signal(self, signal: Signal):
                """Execute signal using Backtrader's order system"""
                if signal.action == SignalType.BUY:
                    if self.position.size <= 0:  # Not already long
                        # Calculate size based on available cash
                        size = min(signal.quantity,
                                 int(self.broker.getcash() / self.data.close[0] * 0.95))
                        if size > 0:
                            self.buy(size=size)
                            
                elif signal.action == SignalType.SELL:
                    if self.position.size > 0:  # Have long position
                        size = min(signal.quantity, abs(self.position.size))
                        self.sell(size=size)
                # Note: We skip short selling to match our beginner-friendly approach
        
        return WrappedStrategy


def run_backtrader_comparison(
    data: pd.DataFrame,
    strategy: StrategyProtocol,
    initial_capital: float = 100000,
    commission: float = 0.001
) -> Dict[str, Any]:
    """
    Run strategy on Backtrader for comparison with our framework
    
    Args:
        data: Market data (OHLCV with datetime index)
        strategy: Our StrategyProtocol strategy
        initial_capital: Starting capital
        commission: Trading commission
        
    Returns:
        Dictionary with backtrader results for comparison
    """
    try:
        import backtrader as bt
        from backtrader import feeds
    except ImportError:
        return {
            'success': False,
            'error': 'Backtrader not installed. Install with: pip install backtrader',
            'performance': {},
            'final_value': initial_capital
        }
    
    try:
        # Create Cerebro engine
        cerebro = bt.Cerebro()
        
        # Set up broker
        cerebro.broker.setcash(initial_capital)
        cerebro.broker.setcommission(commission=commission)
        
        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        
        # Convert data to Backtrader format
        bt_data = _pandas_to_backtrader_data(data)
        cerebro.adddata(bt_data)
        
        # Add our strategy
        wrapper = BacktraderStrategyWrapper(strategy)
        strategy_class = wrapper.create_backtrader_strategy()
        cerebro.addstrategy(strategy_class)
        
        # Run backtest
        results = cerebro.run()
        
        if results:
            strategy_result = results[0]
            analyzers = strategy_result.analyzers
            
            # Extract performance metrics
            performance = {}
            
            if hasattr(analyzers, 'returns'):
                returns_analysis = analyzers.returns.get_analysis()
                performance['total_return'] = returns_analysis.get('rtot', 0)
                performance['annualized_return'] = returns_analysis.get('rnorm', 0)
            
            if hasattr(analyzers, 'drawdown'):
                dd_analysis = analyzers.drawdown.get_analysis()
                performance['max_drawdown'] = dd_analysis.get('max', {}).get('drawdown', 0) / 100
            
            if hasattr(analyzers, 'trades'):
                trade_analysis = analyzers.trades.get_analysis()
                performance['total_trades'] = trade_analysis.get('total', {}).get('total', 0)
                won = trade_analysis.get('won', {}).get('total', 0)
                total = performance['total_trades']
                performance['win_rate'] = won / total if total > 0 else 0
            
            if hasattr(analyzers, 'sharpe'):
                sharpe_analysis = analyzers.sharpe.get_analysis()
                performance['sharpe_ratio'] = sharpe_analysis.get('sharperatio', 0) or 0
            
            final_value = cerebro.broker.getvalue()
            
            return {
                'success': True,
                'performance': performance,
                'final_value': final_value,
                'initial_capital': initial_capital,
                'cerebro': cerebro,
                'results': results
            }
        else:
            return {
                'success': False,
                'error': 'Backtrader run failed',
                'performance': {},
                'final_value': initial_capital
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f'Backtrader error: {str(e)}',
            'performance': {},
            'final_value': initial_capital
        }


def _pandas_to_backtrader_data(df: pd.DataFrame):
    """Convert pandas DataFrame to Backtrader data feed"""
    try:
        import backtrader as bt
        from backtrader import feeds
    except ImportError:
        raise ImportError("Backtrader not installed")
    
    # Ensure columns are properly formatted
    df_bt = df.copy()
    
    # Map column names if needed
    column_mapping = {
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    }
    
    for old_name, new_name in column_mapping.items():
        if old_name in df_bt.columns:
            df_bt.rename(columns={old_name: new_name}, inplace=True)
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df_bt.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Add volume if missing
    if 'volume' not in df_bt.columns:
        df_bt['volume'] = 1000000  # Default volume
    
    # Create Backtrader data feed
    return feeds.PandasData(dataname=df_bt)