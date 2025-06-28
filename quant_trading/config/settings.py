"""
Configuration Management
Type-safe configuration with validation
"""

import os
import json
import yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union
from pathlib import Path


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    commission: float = 0.001
    slippage: float = 0.001
    max_position_size: float = 0.1
    risk_free_rate: float = 0.02
    benchmark_symbol: Optional[str] = None
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if not 0 <= self.commission <= 1:
            raise ValueError("Commission must be between 0 and 1")
        if not 0 <= self.slippage <= 1:
            raise ValueError("Slippage must be between 0 and 1")
        if not 0 < self.max_position_size <= 1:
            raise ValueError("Max position size must be between 0 and 1")


@dataclass
class StrategyConfig:
    """Strategy configuration"""
    name: str
    parameters: Dict[str, Any]
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get parameter with default fallback"""
        return self.parameters.get(key, default)
    
    def set_param(self, key: str, value: Any) -> None:
        """Set parameter value"""
        self.parameters[key] = value


@dataclass
class DataConfig:
    """Data configuration"""
    source: str = "synthetic"
    symbols: list = None
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    frequency: str = "daily"
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["SPY"]


@dataclass
class Config:
    """Main configuration container"""
    backtest: BacktestConfig
    strategy: StrategyConfig
    data: DataConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary"""
        backtest_config = BacktestConfig(**config_dict.get('backtest', {}))
        strategy_config = StrategyConfig(**config_dict.get('strategy', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        
        return cls(
            backtest=backtest_config,
            strategy=strategy_config,
            data=data_config
        )
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from file
        
        Args:
            file_path: Path to configuration file (.json or .yaml)
            
        Returns:
            Config object
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() == '.json':
                config_dict = json.load(f)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {file_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls, prefix: str = "QUANT_") -> 'Config':
        """
        Load configuration from environment variables
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Config object
        """
        config_dict = {
            'backtest': {},
            'strategy': {'name': 'default', 'parameters': {}},
            'data': {}
        }
        
        # Map environment variables to config
        env_mapping = {
            f'{prefix}INITIAL_CAPITAL': ('backtest', 'initial_capital', float),
            f'{prefix}COMMISSION': ('backtest', 'commission', float),
            f'{prefix}SLIPPAGE': ('backtest', 'slippage', float),
            f'{prefix}MAX_POSITION_SIZE': ('backtest', 'max_position_size', float),
            f'{prefix}RISK_FREE_RATE': ('backtest', 'risk_free_rate', float),
            f'{prefix}STRATEGY_NAME': ('strategy', 'name', str),
            f'{prefix}DATA_SOURCE': ('data', 'source', str),
            f'{prefix}START_DATE': ('data', 'start_date', str),
            f'{prefix}END_DATE': ('data', 'end_date', str),
        }
        
        for env_var, (section, key, type_func) in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    config_dict[section][key] = type_func(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid value for {env_var}: {value}") from e
        
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'backtest': asdict(self.backtest),
            'strategy': asdict(self.strategy),
            'data': asdict(self.data)
        }
    
    def save(self, file_path: Union[str, Path]) -> None:
        """
        Save configuration to file
        
        Args:
            file_path: Output file path
        """
        file_path = Path(file_path)
        config_dict = self.to_dict()
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            if file_path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2)
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config file format: {file_path.suffix}")
    
    def validate(self) -> None:
        """Validate all configuration sections"""
        self.backtest.validate()
        
        if not self.strategy.name:
            raise ValueError("Strategy name cannot be empty")
        
        if self.data.source not in ['synthetic', 'yahoo', 'alpha_vantage']:
            raise ValueError(f"Unsupported data source: {self.data.source}")


# Default configurations
DEFAULT_MA_CONFIG = Config(
    backtest=BacktestConfig(),
    strategy=StrategyConfig(
        name="MovingAverage",
        parameters={
            "short_window": 10,
            "long_window": 30,
            "quantity": 100
        }
    ),
    data=DataConfig()
)

DEFAULT_MOMENTUM_CONFIG = Config(
    backtest=BacktestConfig(),
    strategy=StrategyConfig(
        name="Momentum",
        parameters={
            "lookback_period": 20,
            "momentum_threshold": 0.02,
            "quantity": 100
        }
    ),
    data=DataConfig()
)

DEFAULT_MEAN_REVERSION_CONFIG = Config(
    backtest=BacktestConfig(),
    strategy=StrategyConfig(
        name="MeanReversion",
        parameters={
            "window": 20,
            "entry_threshold": 2.0,
            "exit_threshold": 0.5,
            "quantity": 100
        }
    ),
    data=DataConfig()
)


def get_default_config(strategy_name: str) -> Config:
    """
    Get default configuration for a strategy
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        Default config for the strategy
    """
    configs = {
        'MovingAverage': DEFAULT_MA_CONFIG,
        'Momentum': DEFAULT_MOMENTUM_CONFIG,
        'MeanReversion': DEFAULT_MEAN_REVERSION_CONFIG
    }
    
    if strategy_name not in configs:
        raise ValueError(f"No default config for strategy: {strategy_name}")
    
    return configs[strategy_name]