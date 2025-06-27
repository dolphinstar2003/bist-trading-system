#!/usr/bin/env python3
"""
Test fast indicator calculation
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from indicators.indicator_calculator import IndicatorCalculator

def main():
    calculator = IndicatorCalculator()
    
    # Yavaş indikatörleri devre dışı bırak
    calculator.indicator_list = [
        'williams_vix_fix',
        'wavetrend', 
        'squeeze_momentum',
        'adx_di',
        'supertrend',
        'macd'
    ]
    
    # Test için birkaç sembol
    symbols = ['THYAO', 'GARAN', 'AKBNK']
    timeframe = '1h'
    
    print(f"\nCalculating indicators for {len(symbols)} symbols")
    print("Excluded slow indicators: lorentzian, trend_vanguard")
    print("=" * 60)
    
    total_start = time.time()
    
    for symbol in symbols:
        print(f"\n{symbol}:")
        start_time = time.time()
        
        try:
            results = calculator.calculate_all_indicators(symbol, timeframe)
            elapsed = time.time() - start_time
            
            print(f"  ✓ Calculated {len(results)} indicators in {elapsed:.2f} seconds")
            
            for ind_name, df in results.items():
                if df is not None and not df.empty:
                    print(f"    - {ind_name}: {len(df)} bars")
                    
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed:.2f} seconds")
    print(f"Average per symbol: {total_elapsed/len(symbols):.2f} seconds")
    
    # Now test with optimized Lorentzian
    print("\n" + "=" * 60)
    print("Testing with optimized Lorentzian")
    print("=" * 60)
    
    calculator.indicator_list.append('lorentzian')
    
    symbol = 'THYAO'
    start_time = time.time()
    
    results = calculator.calculate_all_indicators(symbol, timeframe)
    elapsed = time.time() - start_time
    
    print(f"\n{symbol} with Lorentzian:")
    print(f"  ✓ Calculated {len(results)} indicators in {elapsed:.2f} seconds")
    
    if 'lorentzian' in results and results['lorentzian'] is not None:
        print(f"  ✓ Lorentzian calculated successfully: {len(results['lorentzian'])} bars")


if __name__ == "__main__":
    main()