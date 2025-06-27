#!/usr/bin/env python3
"""
Test optimized calculator without slow indicators
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from indicators.indicator_calculator_optimized import OptimizedIndicatorCalculator

def main():
    calculator = OptimizedIndicatorCalculator()
    
    # Yavaş indikatörleri devre dışı bırak
    calculator.slow_indicators = []  # Lorentzian ve trend_vanguard'ı çıkar
    
    # Test THYAO
    symbol = 'THYAO'
    timeframe = '1h'
    
    print(f"\nTesting optimized calculator for {symbol} {timeframe}")
    print("=" * 60)
    
    # Önce cache'i temizle
    calculator.cache.clear_old_cache(days=0)
    
    # İlk çalıştırma (cache yok)
    print("\nFirst run (no cache):")
    start_time = time.time()
    results = calculator.calculate_all_indicators_parallel(symbol, timeframe)
    elapsed = time.time() - start_time
    print(f"Calculated {len(results)} indicators in {elapsed:.2f} seconds")
    
    # İkinci çalıştırma (cache var)
    print("\nSecond run (with cache):")
    start_time = time.time()
    results = calculator.calculate_all_indicators_parallel(symbol, timeframe)
    elapsed = time.time() - start_time
    print(f"Calculated {len(results)} indicators in {elapsed:.2f} seconds")
    
    # Cache istatistikleri
    cache_stats = calculator.cache.get_stats()
    print(f"\nCache statistics: {cache_stats}")
    
    # Batch test
    print("\n\nBatch processing test:")
    print("=" * 60)
    symbols = ['THYAO', 'GARAN', 'AKBNK']
    start_time = time.time()
    calculator.process_all_symbols_optimized(symbols=symbols, timeframes=['1h'])
    elapsed = time.time() - start_time
    print(f"\nTotal batch processing time: {elapsed:.2f} seconds")
    
    # Final cache stats
    cache_stats = calculator.cache.get_stats()
    print(f"Final cache statistics: {cache_stats}")
    
    # Signal batch test
    print("\n\nSignal batch test:")
    print("=" * 60)
    start_time = time.time()
    all_signals = calculator.get_latest_signals_batch(symbols, '1h')
    elapsed = time.time() - start_time
    
    print(f"Got signals for {len(all_signals)} symbols in {elapsed:.2f} seconds")
    for symbol, signals in all_signals.items():
        signal_count = len(signals)
        print(f"  {symbol}: {signal_count} indicator signals")


if __name__ == "__main__":
    main()