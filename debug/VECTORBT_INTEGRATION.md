# VectorBT Integration - Generic Architecture

## Overview

This framework now supports **generic VectorBT integration** that works with ANY strategy without requiring strategy-specific logic.

## How It Works

### 🎯 Core Principle
Instead of implementing two versions of strategy logic, we use **one core implementation** (`get_signals()`) and convert it to VectorBT format using a generic function.

### 🏗️ Architecture

```
Strategy Implementation:
├── get_signals()                    # Core strategy logic (point-in-time)
└── generate_vectorbt_signals()      # Delegates to generic converter

Generic Converter:
└── signals_to_vectorbt()           # Works with ANY strategy
    ├── Uses same iteration as BacktestEngine
    ├── for idx, row in data.iterrows():
    ├──     signals = strategy.get_signals(data.loc[:idx])
    └── Converts to VectorBT boolean series
```

## Benefits

### ✅ **Single Implementation**
- Write strategy logic **once** in `get_signals()`
- Automatically get VectorBT support for free
- No duplicate code to maintain

### ✅ **Universal Compatibility**  
- Works with **ANY** strategy that implements `get_signals()`
- No strategy-specific VectorBT logic needed
- Future strategies automatically get VectorBT support

### ✅ **Consistency Guaranteed**
- Both frameworks use identical strategy logic
- Eliminates risk of implementations diverging
- Cross-validation is meaningful

### ✅ **Easy Maintenance**
- Changes only need to be made in one place
- Fewer bugs from duplicate logic
- Simpler codebase

### ✅ **Engine Consistency**
- Uses identical iteration pattern as BacktestEngine
- Same `data.loc[:idx]` slicing approach
- Guarantees identical behavior between frameworks

## Implementation Pattern

For any new strategy:

```python
class MyStrategy:
    def get_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Core strategy logic - implement this"""
        # Your strategy logic here
        return signals
    
    def generate_vectorbt_signals(self, data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """VectorBT support - just delegate to generic function"""
        return signals_to_vectorbt(self, data)
```

## Testing

All three current strategies now use this generic approach:
- `MovingAverageStrategy` ✅
- `MomentumStrategy` ✅  
- `MeanReversionStrategy` ✅

## Cross-Validation

The VectorBT comparison utilities automatically work with any strategy:

```python
from quant_trading.utils.vectorbt_comparison import compare_with_vectorbt

# Works with ANY strategy
results = compare_with_vectorbt(data, my_strategy)
```

## Key Files

- `strategy_interface.py`: Contains `signals_to_vectorbt()` generic function
- `vectorbt_comparison.py`: Cross-validation utilities
- All strategy files: Use generic `generate_vectorbt_signals()` implementation

This architecture ensures you **never need to implement strategy logic twice** while maintaining full compatibility with both frameworks.