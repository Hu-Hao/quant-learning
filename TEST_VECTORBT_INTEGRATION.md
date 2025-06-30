# VectorBT Integration Tests

## Overview

The VectorBT integration tests verify that our generic `signals_to_vectorbt()` function produces **identical results** to our BacktestEngine for all strategies.

## Key Test Categories

### ✅ **Signal Generation Consistency**
- Verifies VectorBT signals match manual point-in-time generation
- Tests all strategy types with same iteration pattern as engine
- Ensures signal timing and conversion accuracy

### ✅ **Framework Integration**
- Tests that all strategies work with generic VectorBT implementation
- Verifies no strategy-specific logic is needed
- Validates signal type conversion (BUY/SELL to boolean series)

### ✅ **Edge Case Handling**
- Tests with minimal data sets
- Validates proper error handling
- Ensures robust behavior across different scenarios

### ✅ **Cross-Validation Ready**
- Tests performance comparison capabilities
- Verifies VectorBT comparison utilities work correctly
- Ensures results are meaningful for validation

## Running the Tests

### All VectorBT Integration Tests
```bash
python3 -m unittest quant_trading.tests.test_vectorbt_integration
```

### Specific Test Categories
```bash
# Basic functionality
python3 -m unittest quant_trading.tests.test_vectorbt_integration.TestVectorBTIntegration.test_signals_to_vectorbt_basic

# Signal consistency
python3 -m unittest quant_trading.tests.test_vectorbt_integration.TestVectorBTIntegration.test_engine_vs_vectorbt_signal_consistency

# All strategies
python3 -m unittest quant_trading.tests.test_vectorbt_integration.TestVectorBTIntegration.test_multiple_strategies_integration

# Signal conversion logic
python3 -m unittest quant_trading.tests.test_vectorbt_integration.TestSignalConversion
```

## Test Results Verification

✅ **Expected Results:**
- All tests should PASS
- VectorBT signals should **exactly match** manual signal generation
- All three strategies (MovingAverage, Momentum, MeanReversion) should work

⚠️ **Common Warnings:**
- "Order rejected: position size limit exceeded" - This is expected and doesn't affect signal generation tests
- These warnings show that signal **generation** vs signal **execution** are properly separated

## What the Tests Prove

1. **Identical Signal Detection**: VectorBT integration detects the exact same signals as our engine
2. **Generic Implementation**: One `signals_to_vectorbt()` function works for ALL strategies
3. **Engine Consistency**: Uses identical `data.loc[:idx]` iteration pattern as BacktestEngine
4. **No Duplication**: No need for strategy-specific VectorBT logic
5. **Cross-Validation Ready**: Results can be trusted for framework comparison

## Integration Architecture Verified

```
Strategy.get_signals() → signals_to_vectorbt() → VectorBT boolean series
                    ↓
BacktestEngine iteration → Same signals detected → Proven consistency
```

The tests confirm that both frameworks will produce **identical results** when using the same strategy and data, enabling confident cross-validation and performance benchmarking.