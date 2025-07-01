#!/usr/bin/env python3
"""
Solutions to fix the Apple stock position sizing issue
"""

import yfinance as yf
from quant_trading.strategies.moving_average import MovingAverageStrategy
from quant_trading.backtesting.engine import BacktestEngine

def solution_1_adjust_position_limits():
    """Solution 1: Adjust position size limits for high-priced stocks"""
    print("🔧 SOLUTION 1: Adjust Position Size Limits")
    print("=" * 50)
    
    # Get Apple data
    apple = yf.Ticker("AAPL")
    data = apple.history(period="6mo")
    data.columns = [col.lower() for col in data.columns]
    
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        quantity=100  # Fixed 100 shares
    )
    
    # BEFORE: Default restrictive limits (causes -97% return)
    print("\n❌ BEFORE (Default Limits):")
    bad_engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        # max_position_size=0.1 (default 10%)
    )
    bad_engine.run_backtest(data, strategy)
    bad_return = (bad_engine.portfolio_values[-1] / 100000 - 1) * 100
    print(f"   Return: {bad_return:+.2f}% ({len(bad_engine.trades)} trades)")
    
    # AFTER: Appropriate limits for high-priced stocks
    print("\n✅ AFTER (Appropriate Limits):")
    good_engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        max_position_size=0.3,  # Allow 30% for high-priced stocks
    )
    good_engine.run_backtest(data, strategy)
    good_return = (good_engine.portfolio_values[-1] / 100000 - 1) * 100
    print(f"   Return: {good_return:+.2f}% ({len(good_engine.trades)} trades)")
    
    improvement = good_return - bad_return
    print(f"\n📈 Improvement: {improvement:+.2f} percentage points!")
    
    return good_return, bad_return

def solution_2_dynamic_position_limits():
    """Solution 2: Calculate appropriate limits dynamically"""
    print("\n🔧 SOLUTION 2: Dynamic Position Size Calculation")
    print("=" * 50)
    
    # Get Apple data
    apple = yf.Ticker("AAPL")
    data = apple.history(period="6mo")
    data.columns = [col.lower() for col in data.columns]
    
    strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        quantity=100
    )
    
    # Calculate appropriate position size limit
    avg_price = data['close'].mean()
    position_value = 100 * avg_price
    initial_capital = 100000
    required_limit = position_value / initial_capital
    safe_limit = required_limit + 0.1  # Add 10% buffer
    
    print(f"   Average Apple price: ${avg_price:.2f}")
    print(f"   100 shares value: ${position_value:,.2f}")
    print(f"   Required limit: {required_limit:.1%}")
    print(f"   Safe limit: {safe_limit:.1%}")
    
    # Use calculated limit
    smart_engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=0.001,
        max_position_size=safe_limit,
    )
    
    smart_engine.run_backtest(data, strategy)
    smart_return = (smart_engine.portfolio_values[-1] / initial_capital - 1) * 100
    print(f"   Result: {smart_return:+.2f}% ({len(smart_engine.trades)} trades)")
    
    return smart_return

def solution_3_percentage_based_strategy():
    """Solution 3: Use percentage-based position sizing instead of fixed quantity"""
    print("\n🔧 SOLUTION 3: Percentage-Based Position Sizing")
    print("=" * 50)
    
    # Get Apple data
    apple = yf.Ticker("AAPL")
    data = apple.history(period="6mo")
    data.columns = [col.lower() for col in data.columns]
    
    # Instead of fixed 100 shares, use percentage of capital
    percentage_strategy = MovingAverageStrategy(
        short_window=10,
        long_window=30,
        percent_capital=0.2  # Use 20% of capital per trade
    )
    
    engine = BacktestEngine(
        initial_capital=100000,
        commission=0.001,
        max_position_size=0.3,  # Allow up to 30%
    )
    
    engine.run_backtest(data, percentage_strategy)
    percentage_return = (engine.portfolio_values[-1] / 100000 - 1) * 100
    
    print(f"   20% capital strategy: {percentage_return:+.2f}% ({len(engine.trades)} trades)")
    print(f"   Automatically scales with capital and stock price")
    
    return percentage_return

def solution_4_validation_warnings():
    """Solution 4: Add validation to warn about position size issues"""
    print("\n🔧 SOLUTION 4: Add Position Size Validation")
    print("=" * 50)
    
    def validate_strategy_for_stock(strategy, stock_price, initial_capital, max_position_size):
        """Validate if strategy parameters work with stock characteristics"""
        warnings = []
        
        if hasattr(strategy, 'quantity') and strategy.quantity:
            position_value = strategy.quantity * stock_price
            position_percent = position_value / initial_capital
            max_allowed = max_position_size
            
            if position_percent > max_allowed:
                warnings.append(
                    f"⚠️ Fixed quantity ({strategy.quantity} shares) requires "
                    f"{position_percent:.1%} of capital but limit is {max_allowed:.1%}"
                )
                warnings.append(
                    f"   Suggestion: Increase max_position_size to {position_percent + 0.05:.1%} "
                    f"or use percent_capital={max_allowed:.1f} instead"
                )
        
        return warnings
    
    # Example validation
    strategy = MovingAverageStrategy(short_window=10, long_window=30, quantity=100)
    apple_price = 225.0  # Typical Apple price
    
    validation_warnings = validate_strategy_for_stock(
        strategy=strategy,
        stock_price=apple_price,
        initial_capital=100000,
        max_position_size=0.1  # Default limit
    )
    
    print("   Validation results:")
    for warning in validation_warnings:
        print(f"   {warning}")
    
    return validation_warnings

def solution_5_framework_method():
    """Solution 5: Add helper method to framework"""
    print("\n🔧 SOLUTION 5: Framework Helper Method")
    print("=" * 50)
    
    def calculate_appropriate_position_limit(strategy, stock_data, initial_capital, buffer=0.1):
        """Calculate appropriate position size limit for strategy and stock"""
        
        if hasattr(strategy, 'quantity') and strategy.quantity:
            # Fixed quantity strategy
            avg_price = stock_data['close'].mean()
            position_value = strategy.quantity * avg_price
            required_limit = position_value / initial_capital
            safe_limit = min(1.0, required_limit + buffer)  # Cap at 100%
            
            return safe_limit, f"Fixed {strategy.quantity} shares needs {safe_limit:.1%} limit"
            
        elif hasattr(strategy, 'percent_capital') and strategy.percent_capital:
            # Percentage strategy
            return strategy.percent_capital, f"Percentage strategy: {strategy.percent_capital:.1%}"
            
        else:
            # Full capital strategy
            return 1.0, "Full capital strategy: 100% limit"
    
    # Example usage
    apple = yf.Ticker("AAPL")
    data = apple.history(period="1mo")
    data.columns = [col.lower() for col in data.columns]
    
    strategy = MovingAverageStrategy(short_window=10, long_window=30, quantity=100)
    
    optimal_limit, explanation = calculate_appropriate_position_limit(
        strategy=strategy,
        stock_data=data,
        initial_capital=100000
    )
    
    print(f"   Calculated optimal limit: {optimal_limit:.1%}")
    print(f"   Explanation: {explanation}")
    
    return optimal_limit

def demonstrate_all_solutions():
    """Demonstrate all solutions"""
    print("🍎 APPLE STOCK POSITION SIZING - ALL SOLUTIONS")
    print("=" * 70)
    
    try:
        # Solution 1: Adjust limits
        good_return, bad_return = solution_1_adjust_position_limits()
        
        # Solution 2: Dynamic calculation
        smart_return = solution_2_dynamic_position_limits()
        
        # Solution 3: Percentage-based
        percentage_return = solution_3_percentage_based_strategy()
        
        # Solution 4: Validation
        warnings = solution_4_validation_warnings()
        
        # Solution 5: Helper method
        optimal_limit = solution_5_framework_method()
        
        print(f"\n📊 SOLUTION COMPARISON:")
        print(f"   Default limits (broken): {bad_return:+.2f}%")
        print(f"   Adjusted limits: {good_return:+.2f}%")
        print(f"   Dynamic calculation: {smart_return:+.2f}%")
        print(f"   Percentage-based: {percentage_return:+.2f}%")
        print(f"   Optimal limit calculated: {optimal_limit:.1%}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"   1. ⭐ Use adjusted limits: max_position_size=0.3 for high-priced stocks")
        print(f"   2. 🔄 Consider percentage-based strategies for scalability")
        print(f"   3. ⚠️ Add validation warnings to catch issues early")
        print(f"   4. 🤖 Implement dynamic limit calculation helpers")
        print(f"   5. 📚 Document position sizing best practices")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print(f"   Note: This requires real Apple data from yfinance")

if __name__ == "__main__":
    demonstrate_all_solutions()