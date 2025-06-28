"""
Tests for configuration management
"""

import unittest
import tempfile
import os
import json
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from quant_trading.config.settings import (
    Config, 
    BacktestConfig, 
    StrategyConfig, 
    DataConfig,
    get_default_config,
    DEFAULT_MA_CONFIG,
    DEFAULT_MOMENTUM_CONFIG,
    DEFAULT_MEAN_REVERSION_CONFIG
)


class TestBacktestConfig(unittest.TestCase):
    """Test cases for BacktestConfig"""
    
    def test_default_initialization(self):
        """Test default BacktestConfig initialization"""
        config = BacktestConfig()
        
        self.assertEqual(config.initial_capital, 100000.0)
        self.assertEqual(config.commission, 0.001)
        self.assertEqual(config.slippage, 0.001)
        self.assertEqual(config.max_position_size, 0.1)
        self.assertEqual(config.risk_free_rate, 0.02)
        self.assertIsNone(config.benchmark_symbol)
        
    def test_custom_initialization(self):
        """Test BacktestConfig with custom values"""
        config = BacktestConfig(
            initial_capital=50000.0,
            commission=0.002,
            slippage=0.0005,
            max_position_size=0.2,
            risk_free_rate=0.03,
            benchmark_symbol="SPY"
        )
        
        self.assertEqual(config.initial_capital, 50000.0)
        self.assertEqual(config.commission, 0.002)
        self.assertEqual(config.slippage, 0.0005)
        self.assertEqual(config.max_position_size, 0.2)
        self.assertEqual(config.risk_free_rate, 0.03)
        self.assertEqual(config.benchmark_symbol, "SPY")
        
    def test_validation_valid_config(self):
        """Test validation with valid configuration"""
        config = BacktestConfig()
        config.validate()  # Should not raise
        
    def test_validation_negative_capital(self):
        """Test validation with negative capital"""
        config = BacktestConfig(initial_capital=-1000)
        with self.assertRaises(ValueError):
            config.validate()
            
    def test_validation_invalid_commission(self):
        """Test validation with invalid commission"""
        config = BacktestConfig(commission=1.5)  # > 1
        with self.assertRaises(ValueError):
            config.validate()
            
        config = BacktestConfig(commission=-0.1)  # < 0
        with self.assertRaises(ValueError):
            config.validate()
            
    def test_validation_invalid_slippage(self):
        """Test validation with invalid slippage"""
        config = BacktestConfig(slippage=1.5)  # > 1
        with self.assertRaises(ValueError):
            config.validate()
            
        config = BacktestConfig(slippage=-0.1)  # < 0
        with self.assertRaises(ValueError):
            config.validate()
            
    def test_validation_invalid_position_size(self):
        """Test validation with invalid position size"""
        config = BacktestConfig(max_position_size=0)  # <= 0
        with self.assertRaises(ValueError):
            config.validate()
            
        config = BacktestConfig(max_position_size=1.5)  # > 1
        with self.assertRaises(ValueError):
            config.validate()


class TestStrategyConfig(unittest.TestCase):
    """Test cases for StrategyConfig"""
    
    def test_initialization(self):
        """Test StrategyConfig initialization"""
        params = {"param1": 10, "param2": 0.5}
        config = StrategyConfig(name="TestStrategy", parameters=params)
        
        self.assertEqual(config.name, "TestStrategy")
        self.assertEqual(config.parameters, params)
        
    def test_get_param(self):
        """Test parameter retrieval"""
        params = {"param1": 10, "param2": 0.5}
        config = StrategyConfig(name="TestStrategy", parameters=params)
        
        self.assertEqual(config.get_param("param1"), 10)
        self.assertEqual(config.get_param("param2"), 0.5)
        self.assertIsNone(config.get_param("nonexistent"))
        self.assertEqual(config.get_param("nonexistent", "default"), "default")
        
    def test_set_param(self):
        """Test parameter setting"""
        config = StrategyConfig(name="TestStrategy", parameters={})
        
        config.set_param("new_param", 42)
        self.assertEqual(config.get_param("new_param"), 42)
        
        # Update existing parameter
        config.set_param("new_param", 100)
        self.assertEqual(config.get_param("new_param"), 100)


class TestDataConfig(unittest.TestCase):
    """Test cases for DataConfig"""
    
    def test_default_initialization(self):
        """Test default DataConfig initialization"""
        config = DataConfig()
        
        self.assertEqual(config.source, "synthetic")
        self.assertEqual(config.symbols, ["SPY"])
        self.assertEqual(config.start_date, "2020-01-01")
        self.assertEqual(config.end_date, "2023-12-31")
        self.assertEqual(config.frequency, "daily")
        
    def test_custom_initialization(self):
        """Test DataConfig with custom values"""
        config = DataConfig(
            source="yahoo",
            symbols=["AAPL", "GOOGL"],
            start_date="2022-01-01",
            end_date="2022-12-31",
            frequency="hourly"
        )
        
        self.assertEqual(config.source, "yahoo")
        self.assertEqual(config.symbols, ["AAPL", "GOOGL"])
        self.assertEqual(config.start_date, "2022-01-01")
        self.assertEqual(config.end_date, "2022-12-31")
        self.assertEqual(config.frequency, "hourly")


class TestConfig(unittest.TestCase):
    """Test cases for main Config class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.backtest_config = BacktestConfig()
        self.strategy_config = StrategyConfig(name="TestStrategy", parameters={"param": 10})
        self.data_config = DataConfig()
        
    def test_initialization(self):
        """Test Config initialization"""
        config = Config(
            backtest=self.backtest_config,
            strategy=self.strategy_config,
            data=self.data_config
        )
        
        self.assertEqual(config.backtest, self.backtest_config)
        self.assertEqual(config.strategy, self.strategy_config)
        self.assertEqual(config.data, self.data_config)
        
    def test_from_dict(self):
        """Test Config creation from dictionary"""
        config_dict = {
            'backtest': {
                'initial_capital': 50000.0,
                'commission': 0.002
            },
            'strategy': {
                'name': 'TestStrategy',
                'parameters': {'param1': 20}
            },
            'data': {
                'source': 'yahoo',
                'symbols': ['AAPL']
            }
        }
        
        config = Config.from_dict(config_dict)
        
        self.assertEqual(config.backtest.initial_capital, 50000.0)
        self.assertEqual(config.backtest.commission, 0.002)
        self.assertEqual(config.strategy.name, 'TestStrategy')
        self.assertEqual(config.strategy.get_param('param1'), 20)
        self.assertEqual(config.data.source, 'yahoo')
        self.assertEqual(config.data.symbols, ['AAPL'])
        
    def test_from_dict_partial(self):
        """Test Config creation from partial dictionary"""
        config_dict = {
            'backtest': {
                'initial_capital': 75000.0
            }
            # Missing strategy and data sections
        }
        
        config = Config.from_dict(config_dict)
        
        # Should use defaults for missing sections
        self.assertEqual(config.backtest.initial_capital, 75000.0)
        self.assertEqual(config.backtest.commission, 0.001)  # Default
        self.assertIsNotNone(config.strategy)
        self.assertIsNotNone(config.data)
        
    def test_to_dict(self):
        """Test Config conversion to dictionary"""
        config = Config(
            backtest=self.backtest_config,
            strategy=self.strategy_config,
            data=self.data_config
        )
        
        config_dict = config.to_dict()
        
        self.assertIn('backtest', config_dict)
        self.assertIn('strategy', config_dict)
        self.assertIn('data', config_dict)
        
        self.assertEqual(config_dict['backtest']['initial_capital'], 100000.0)
        self.assertEqual(config_dict['strategy']['name'], 'TestStrategy')
        
    def test_save_and_load_json(self):
        """Test saving and loading JSON configuration"""
        config = Config(
            backtest=self.backtest_config,
            strategy=self.strategy_config,
            data=self.data_config
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
            
        try:
            # Save config
            config.save(config_path)
            self.assertTrue(os.path.exists(config_path))
            
            # Load config
            loaded_config = Config.from_file(config_path)
            
            # Verify loaded config
            self.assertEqual(loaded_config.backtest.initial_capital, 100000.0)
            self.assertEqual(loaded_config.strategy.name, 'TestStrategy')
            self.assertEqual(loaded_config.strategy.get_param('param'), 10)
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
                
    @unittest.skipUnless(HAS_YAML, "PyYAML not available")
    def test_save_and_load_yaml(self):
        """Test saving and loading YAML configuration"""
        config = Config(
            backtest=self.backtest_config,
            strategy=self.strategy_config,
            data=self.data_config
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
            
        try:
            # Save config
            config.save(config_path)
            self.assertTrue(os.path.exists(config_path))
            
            # Load config
            loaded_config = Config.from_file(config_path)
            
            # Verify loaded config
            self.assertEqual(loaded_config.backtest.initial_capital, 100000.0)
            self.assertEqual(loaded_config.strategy.name, 'TestStrategy')
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
                
    def test_from_file_nonexistent(self):
        """Test loading from nonexistent file"""
        with self.assertRaises(FileNotFoundError):
            Config.from_file("nonexistent_file.json")
            
    def test_from_file_unsupported_format(self):
        """Test loading from unsupported file format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            config_path = f.name
            f.write("some content")
            
        try:
            with self.assertRaises(ValueError):
                Config.from_file(config_path)
        finally:
            os.unlink(config_path)
            
    def test_save_unsupported_format(self):
        """Test saving to unsupported file format"""
        config = Config(
            backtest=self.backtest_config,
            strategy=self.strategy_config,
            data=self.data_config
        )
        
        with self.assertRaises(ValueError):
            config.save("config.txt")
            
    def test_from_env(self):
        """Test Config creation from environment variables"""
        # Set environment variables
        env_vars = {
            'QUANT_INITIAL_CAPITAL': '75000',
            'QUANT_COMMISSION': '0.002',
            'QUANT_STRATEGY_NAME': 'EnvStrategy',
            'QUANT_DATA_SOURCE': 'yahoo'
        }
        
        # Temporarily set environment variables
        old_env = {}
        for key, value in env_vars.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = value
            
        try:
            config = Config.from_env(prefix="QUANT_")
            
            self.assertEqual(config.backtest.initial_capital, 75000.0)
            self.assertEqual(config.backtest.commission, 0.002)
            self.assertEqual(config.strategy.name, 'EnvStrategy')
            self.assertEqual(config.data.source, 'yahoo')
            
        finally:
            # Restore environment
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value
                    
    def test_validate_valid_config(self):
        """Test validation with valid configuration"""
        config = Config(
            backtest=self.backtest_config,
            strategy=self.strategy_config,
            data=self.data_config
        )
        
        config.validate()  # Should not raise
        
    def test_validate_empty_strategy_name(self):
        """Test validation with empty strategy name"""
        strategy_config = StrategyConfig(name="", parameters={})
        config = Config(
            backtest=self.backtest_config,
            strategy=strategy_config,
            data=self.data_config
        )
        
        with self.assertRaises(ValueError):
            config.validate()
            
    def test_validate_invalid_data_source(self):
        """Test validation with invalid data source"""
        data_config = DataConfig(source="invalid_source")
        config = Config(
            backtest=self.backtest_config,
            strategy=self.strategy_config,
            data=data_config
        )
        
        with self.assertRaises(ValueError):
            config.validate()


class TestDefaultConfigs(unittest.TestCase):
    """Test cases for default configurations"""
    
    def test_default_ma_config(self):
        """Test default Moving Average configuration"""
        config = DEFAULT_MA_CONFIG
        
        self.assertEqual(config.strategy.name, "MovingAverage")
        self.assertEqual(config.strategy.get_param("short_window"), 10)
        self.assertEqual(config.strategy.get_param("long_window"), 30)
        self.assertEqual(config.strategy.get_param("quantity"), 100)
        
    def test_default_momentum_config(self):
        """Test default Momentum configuration"""
        config = DEFAULT_MOMENTUM_CONFIG
        
        self.assertEqual(config.strategy.name, "Momentum")
        self.assertEqual(config.strategy.get_param("lookback_period"), 20)
        self.assertEqual(config.strategy.get_param("momentum_threshold"), 0.02)
        
    def test_default_mean_reversion_config(self):
        """Test default Mean Reversion configuration"""
        config = DEFAULT_MEAN_REVERSION_CONFIG
        
        self.assertEqual(config.strategy.name, "MeanReversion")
        self.assertEqual(config.strategy.get_param("window"), 20)
        self.assertEqual(config.strategy.get_param("entry_threshold"), 2.0)
        
    def test_get_default_config(self):
        """Test get_default_config function"""
        ma_config = get_default_config("MovingAverage")
        self.assertEqual(ma_config.strategy.name, "MovingAverage")
        
        momentum_config = get_default_config("Momentum")
        self.assertEqual(momentum_config.strategy.name, "Momentum")
        
        mr_config = get_default_config("MeanReversion")
        self.assertEqual(mr_config.strategy.name, "MeanReversion")
        
    def test_get_default_config_invalid(self):
        """Test get_default_config with invalid strategy"""
        with self.assertRaises(ValueError):
            get_default_config("NonexistentStrategy")


if __name__ == '__main__':
    unittest.main()