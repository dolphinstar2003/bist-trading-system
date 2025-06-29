#!/usr/bin/env python3
"""
Debug signal generation to understand why system is not working
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.trimode_orchestrator_aggressive import AggressiveTriModeOrchestrator
from utils.csv_data_manager import CSVDataManager
import pandas as pd

def debug_signals():
    print("=== SIGNAL GENERATION DEBUG ===\n")
    
    orchestrator = AggressiveTriModeOrchestrator(initial_capital=100000)
    csv_manager = CSVDataManager()
    
    # Test with a few symbols
    test_symbols = ['THYAO', 'GARAN', 'AEFES']
    
    for symbol in test_symbols:
        print(f"\n{symbol}:")
        print("-" * 40)
        
        # Load data
        data = csv_manager.load_raw_data(symbol, '1d')
        if data is None or len(data) < 100:
            print(f"  No data for {symbol}")
            continue
        
        # Get last 100 days
        data = data.tail(100)
        
        # Get EMA params
        if symbol in orchestrator.ema_params_cache:
            fast, slow = orchestrator.ema_params_cache[symbol]
        else:
            fast, slow = orchestrator.ema_params_cache.get('DEFAULT', (10, 25))
        
        print(f"  EMA Parameters: Fast={fast}, Slow={slow}")
        
        # Calculate EMAs
        ema_fast = data['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow, adjust=False).mean()
        
        # Check current state
        current_price = data['close'].iloc[-1]
        fast_value = ema_fast.iloc[-1]
        slow_value = ema_slow.iloc[-1]
        
        print(f"  Current Price: {current_price:.2f}")
        print(f"  Fast EMA ({fast}): {fast_value:.2f}")
        print(f"  Slow EMA ({slow}): {slow_value:.2f}")
        print(f"  Fast > Slow: {fast_value > slow_value}")
        
        # Check for crossovers in last 10 days
        crossovers = []
        for i in range(-10, 0):
            curr_fast_above = ema_fast.iloc[i] > ema_slow.iloc[i]
            prev_fast_above = ema_fast.iloc[i-1] > ema_slow.iloc[i-1]
            
            if curr_fast_above and not prev_fast_above:
                crossovers.append(('BUY', data.index[i], data['close'].iloc[i]))
            elif not curr_fast_above and prev_fast_above:
                crossovers.append(('SELL', data.index[i], data['close'].iloc[i]))
        
        print(f"  Crossovers (last 10 days): {len(crossovers)}")
        for signal, date, price in crossovers[-3:]:  # Show last 3
            print(f"    {signal} on {date.date()} @ {price:.2f}")
        
        # Check volume
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]
        current_volume = data['volume'].iloc[-1]
        rvol = current_volume / avg_volume if avg_volume > 0 else 0
        
        print(f"  Current Volume: {current_volume:,.0f}")
        print(f"  Avg Volume (20d): {avg_volume:,.0f}")
        print(f"  RVOL: {rvol:.2f}")
        
        # Generate signal in each mode
        print(f"\n  Signal Generation Test:")
        for mode in ['AGGRESSIVE', 'BALANCED', 'DEFENSIVE']:
            orchestrator.current_mode = eval(f"orchestrator.TradingMode.{mode}")
            
            # Prepare market data
            market_data = {'symbol_data': {symbol: data}, 'timeframe': '1d'}
            
            # Generate signal
            signals = orchestrator.generate_signals([symbol], '1d')
            
            if signals:
                signal = signals[0]
                print(f"    {mode}: BUY signal generated!")
                print(f"      Strength: {signal['strength']:.2f}")
                print(f"      Reasons: {signal['reasons']}")
            else:
                print(f"    {mode}: No signal")


if __name__ == "__main__":
    debug_signals()