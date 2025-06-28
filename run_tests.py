#!/usr/bin/env python3
"""
Comprehensive Test Runner for Quant Trading Framework
Runs all tests with coverage reporting and detailed output
"""

import unittest
import sys
import os
from io import StringIO
import time

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


class ColoredTextTestResult(unittest.TextTestResult):
    """Test result class with colored output"""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.success_count = 0
        self.verbosity = verbosity
        
    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1
        if self.verbosity > 1:
            self.stream.write("✅ ")
            self.stream.write(self.getDescription(test))
            self.stream.writeln()
            
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.write("💥 ERROR: ")
            self.stream.write(self.getDescription(test))
            self.stream.writeln()
            
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.write("❌ FAIL: ")
            self.stream.write(self.getDescription(test))
            self.stream.writeln()
            
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.write("⏭️  SKIP: ")
            self.stream.write(self.getDescription(test))
            self.stream.write(f" ({reason})")
            self.stream.writeln()


class ColoredTestRunner(unittest.TextTestRunner):
    """Test runner with colored output"""
    
    def __init__(self, **kwargs):
        kwargs['resultclass'] = ColoredTextTestResult
        super().__init__(**kwargs)


def run_test_suite(test_pattern='test_*.py', verbosity=2):
    """
    Run the complete test suite
    
    Args:
        test_pattern: Pattern to match test files
        verbosity: Test output verbosity level
        
    Returns:
        TestResult object
    """
    print("🧪 QUANT TRADING FRAMEWORK - TEST SUITE")
    print("=" * 60)
    print()
    
    # Discover tests
    loader = unittest.TestLoader()
    test_dir = os.path.join(project_root, 'quant_trading', 'tests')
    
    if not os.path.exists(test_dir):
        print(f"❌ Test directory not found: {test_dir}")
        return None
        
    suite = loader.discover(test_dir, pattern=test_pattern)
    
    # Count tests
    test_count = suite.countTestCases()
    print(f"📊 Discovered {test_count} tests")
    print()
    
    # Run tests
    start_time = time.time()
    runner = ColoredTestRunner(verbosity=verbosity, buffer=True)
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print()
    print("=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    duration = end_time - start_time
    print(f"⏱️  Duration: {duration:.2f} seconds")
    print(f"🏃 Tests run: {result.testsRun}")
    print(f"✅ Successes: {getattr(result, 'success_count', result.testsRun - len(result.failures) - len(result.errors))}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    print(f"⏭️  Skipped: {len(result.skipped)}")
    
    # Calculate success rate
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"📈 Success rate: {success_rate:.1f}%")
    
    print()
    
    # Print detailed failure/error information
    if result.failures:
        print("❌ FAILURES:")
        print("-" * 40)
        for test, traceback in result.failures:
            print(f"• {test}")
            if verbosity > 2:
                print(f"  {traceback}")
        print()
        
    if result.errors:
        print("💥 ERRORS:")
        print("-" * 40)
        for test, traceback in result.errors:
            print(f"• {test}")
            if verbosity > 2:
                print(f"  {traceback}")
        print()
    
    # Overall result
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
        
    print("=" * 60)
    
    return result


def run_specific_module(module_name, verbosity=2):
    """
    Run tests for a specific module
    
    Args:
        module_name: Name of the test module (e.g., 'test_backtesting')
        verbosity: Test output verbosity level
    """
    print(f"🧪 Running tests for module: {module_name}")
    print("=" * 40)
    
    # Load specific test module
    loader = unittest.TestLoader()
    test_dir = os.path.join(project_root, 'quant_trading', 'tests')
    
    try:
        suite = loader.loadTestsFromName(f'{module_name}', module=None)
        runner = ColoredTestRunner(verbosity=verbosity)
        result = runner.run(suite)
        return result
    except ImportError as e:
        print(f"❌ Failed to import test module: {e}")
        return None


def run_test_categories():
    """Run tests by category"""
    categories = {
        'Core Backtesting': ['test_backtesting.py'],
        'Trading Strategies': ['test_strategies.py'],
        'Data Management': ['test_data_fetcher.py'],
        'Technical Indicators': ['test_indicators.py'],
        'Configuration': ['test_config.py'],
        'Metrics & Analysis': ['test_metrics.py'],
        'Visualization': ['test_visualization.py']
    }
    
    print("🧪 RUNNING TESTS BY CATEGORY")
    print("=" * 60)
    
    overall_results = []
    
    for category, test_files in categories.items():
        print(f"\n📂 {category}")
        print("-" * 40)
        
        for test_file in test_files:
            result = run_test_suite(test_pattern=test_file, verbosity=1)
            if result:
                overall_results.append((category, test_file, result))
                status = "✅ PASS" if result.wasSuccessful() else "❌ FAIL"
                print(f"  {test_file}: {status} ({result.testsRun} tests)")
    
    # Summary by category
    print(f"\n{'='*60}")
    print("📊 CATEGORY SUMMARY")
    print(f"{'='*60}")
    
    for category, test_file, result in overall_results:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0
        print(f"{category:<20} {success_rate:>6.1f}% ({result.testsRun} tests)")


def check_dependencies():
    """Check if all required dependencies are available"""
    print("🔍 CHECKING DEPENDENCIES")
    print("=" * 30)
    
    required_modules = [
        'pandas',
        'numpy',
        'datetime',
        'typing',
        'dataclasses',
        'enum',
        'abc'
    ]
    
    optional_modules = [
        'matplotlib',
        'seaborn',
        'yaml'
    ]
    
    missing_required = []
    missing_optional = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing_required.append(module)
            print(f"❌ {module} (REQUIRED)")
    
    for module in optional_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing_optional.append(module)
            print(f"⚠️  {module} (OPTIONAL)")
    
    print()
    
    if missing_required:
        print("❌ Missing required dependencies:")
        for module in missing_required:
            print(f"   pip install {module}")
        return False
    
    if missing_optional:
        print("⚠️  Missing optional dependencies (some features may be limited):")
        for module in missing_optional:
            print(f"   pip install {module}")
    
    print("✅ All required dependencies available!")
    return True


def main():
    """Main test runner entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quant Trading Framework Test Runner')
    parser.add_argument('--module', '-m', help='Run tests for specific module')
    parser.add_argument('--pattern', '-p', default='test_*.py', help='Test file pattern')
    parser.add_argument('--verbosity', '-v', type=int, default=2, help='Test verbosity level')
    parser.add_argument('--categories', '-c', action='store_true', help='Run tests by category')
    parser.add_argument('--check-deps', action='store_true', help='Check dependencies only')
    parser.add_argument('--quick', '-q', action='store_true', help='Quick test run (verbosity 1)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.verbosity = 1
    
    # Check dependencies first
    if not check_dependencies():
        if not args.check_deps:
            print("\n❌ Cannot run tests due to missing dependencies")
            sys.exit(1)
        else:
            sys.exit(1)
    
    if args.check_deps:
        sys.exit(0)
    
    print()
    
    # Run specific test module
    if args.module:
        result = run_specific_module(args.module, args.verbosity)
        sys.exit(0 if result and result.wasSuccessful() else 1)
    
    # Run by categories
    if args.categories:
        run_test_categories()
        sys.exit(0)
    
    # Run full test suite
    result = run_test_suite(args.pattern, args.verbosity)
    
    if result is None:
        sys.exit(1)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()