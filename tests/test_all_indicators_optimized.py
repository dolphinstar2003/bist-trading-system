#!/usr/bin/env python3
"""
Test all indicators including optimized ones
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from indicators.indicator_calculator import IndicatorCalculator

def main():
    calculator = IndicatorCalculator()
    
    # Test all indicators
    symbol = 'THYAO'
    timeframe = '1h'
    
    print(f"\nCalculating ALL indicators for {symbol} {timeframe}")
    print("=" * 60)
    
    total_start = time.time()
    
    # Calculate each indicator separately to measure time
    results = {}
    timings = {}
    
    for ind_name in calculator.indicator_list:
        print(f"\nCalculating {ind_name}...")
        start_time = time.time()
        
        try:
            # Temporarily set to calculate only this indicator
            original_list = calculator.indicator_list
            calculator.indicator_list = [ind_name]
            
            result = calculator.calculate_all_indicators(symbol, timeframe)
            
            # Restore original list
            calculator.indicator_list = original_list
            
            elapsed = time.time() - start_time
            timings[ind_name] = elapsed
            
            if ind_name in result:
                results[ind_name] = result[ind_name]
                print(f"  ✓ Completed in {elapsed:.2f} seconds")
                if result[ind_name] is not None:
                    print(f"    Shape: {result[ind_name].shape}")
            else:
                print(f"  ✗ Failed")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            timings[ind_name] = 0
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nTimings by indicator:")
    for ind_name, timing in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ind_name:20s}: {timing:6.2f} seconds")
    
    print(f"\nTotal time: {total_elapsed:.2f} seconds")
    print(f"Indicators calculated: {len(results)}/{len(calculator.indicator_list)}")
    
    # Test batch calculation
    print("\n" + "=" * 60)
    print("BATCH CALCULATION TEST")
    print("=" * 60)
    
    # Fast indicators only
    fast_indicators = ['williams_vix_fix', 'wavetrend', 'squeeze_momentum', 
                      'adx_di', 'supertrend', 'macd']
    
    calculator.indicator_list = fast_indicators
    
    print(f"\nCalculating {len(fast_indicators)} fast indicators...")
    start_time = time.time()
    results = calculator.calculate_all_indicators(symbol, timeframe)
    elapsed = time.time() - start_time
    
    print(f"✓ Batch calculation completed in {elapsed:.2f} seconds")
    print(f"  Average per indicator: {elapsed/len(fast_indicators):.2f} seconds")
    
    # All indicators including slow ones
    calculator.indicator_list = list(calculator.indicator_classes.keys())
    
    print(f"\nCalculating ALL {len(calculator.indicator_list)} indicators...")
    start_time = time.time()
    results = calculator.calculate_all_indicators(symbol, timeframe)
    elapsed = time.time() - start_time
    
    print(f"✓ All indicators completed in {elapsed:.2f} seconds")
    print(f"  Average per indicator: {elapsed/len(calculator.indicator_list):.2f} seconds")


if __name__ == "__main__":
    main()