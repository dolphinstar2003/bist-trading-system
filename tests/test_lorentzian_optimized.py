#!/usr/bin/env python3
"""
Test optimized Lorentzian Classification
"""

import sys
from pathlib import Path
import time
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from indicators.lorentzian_classification import LorentzianClassification
from indicators.lorentzian_classification_optimized import OptimizedLorentzianClassification
from utils.csv_data_manager import CSVDataManager

def main():
    csv_manager = CSVDataManager()
    
    # Load test data
    symbol = 'THYAO'
    timeframe = '1h'
    
    print(f"\nLoading data for {symbol} {timeframe}...")
    df = csv_manager.load_raw_data(symbol, timeframe)
    
    if df is None:
        print("Error: Could not load data")
        return
    
    # Use only last 500 bars for fair comparison
    df = df.iloc[-500:].copy()
    print(f"Using last {len(df)} bars for testing")
    
    # Test original version (if needed)
    print("\n" + "="*60)
    print("Testing Original Lorentzian Classification")
    print("="*60)
    
    original = LorentzianClassification(
        neighbors_count=8,
        max_bars_back=500,
        feature_count=5
    )
    
    start_time = time.time()
    result_original = original.calculate(df)
    original_time = time.time() - start_time
    
    print(f"Original calculation time: {original_time:.2f} seconds")
    print(f"Result shape: {result_original.shape}")
    
    # Count signals
    if not result_original.empty and 'signal' in result_original.columns:
        buy_signals = (result_original['signal'] == 1).sum()
        sell_signals = (result_original['signal'] == -1).sum()
        print(f"Buy signals: {buy_signals}, Sell signals: {sell_signals}")
    
    # Test optimized version
    print("\n" + "="*60)
    print("Testing Optimized Lorentzian Classification")
    print("="*60)
    
    optimized = OptimizedLorentzianClassification(
        neighbors_count=8,
        max_bars_back=500,
        feature_count=5,
        use_cache=True
    )
    
    # First run (no cache)
    start_time = time.time()
    result_optimized = optimized.calculate(df)
    optimized_time = time.time() - start_time
    
    print(f"Optimized calculation time: {optimized_time:.2f} seconds")
    print(f"Result shape: {result_optimized.shape}")
    
    # Count signals
    if not result_optimized.empty and 'signal' in result_optimized.columns:
        buy_signals = (result_optimized['signal'] == 1).sum()
        sell_signals = (result_optimized['signal'] == -1).sum()
        print(f"Buy signals: {buy_signals}, Sell signals: {sell_signals}")
    
    # Performance comparison
    print("\n" + "="*60)
    print("Performance Comparison")
    print("="*60)
    
    if original_time > 0:
        speedup = original_time / optimized_time
        print(f"Speedup: {speedup:.2f}x faster")
        print(f"Time saved: {original_time - optimized_time:.2f} seconds")
    
    # Test with different data sizes
    print("\n" + "="*60)
    print("Testing with Different Data Sizes")
    print("="*60)
    
    test_sizes = [100, 200, 500, 1000]
    
    for size in test_sizes:
        if size > len(df):
            continue
            
        test_df = df.iloc[-size:].copy()
        
        # Optimized version
        start_time = time.time()
        result = optimized.calculate(test_df)
        calc_time = time.time() - start_time
        
        signals = 0
        if not result.empty and 'signal' in result.columns:
            signals = (result['signal'] != 0).sum()
        
        print(f"Size: {size:4d} bars | Time: {calc_time:6.2f}s | Signals: {signals:3d}")
    
    # Test signal generation
    print("\n" + "="*60)
    print("Testing Signal Generation")
    print("="*60)
    
    if not result_optimized.empty:
        last_values = result_optimized.iloc[-1].to_dict()
        signal = optimized.get_signal(last_values)
        print(f"Latest signal: {signal}")
        print(f"Latest prediction: {last_values.get('prediction', 0):.3f}")
        print(f"Latest confidence: {last_values.get('confidence', 0):.3f}")


if __name__ == "__main__":
    main()