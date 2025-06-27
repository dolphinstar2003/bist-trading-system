#!/usr/bin/env python3
"""
THYAO Backtest Test - Simplified
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Proje kökünü ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.csv_data_manager import CSVDataManager

def analyze_thyao_signals():
    """Analyze THYAO signals and potential trades"""
    
    print("\n" + "="*60)
    print("THYAO SIGNAL ANALYSIS")
    print("="*60)
    
    csv_manager = CSVDataManager()
    symbol = 'THYAO'
    timeframe = '1h'
    
    # Load price data
    df = csv_manager.load_raw_data(symbol, timeframe)
    if df is None:
        print("Error: No price data")
        return
        
    print(f"\nPrice data: {len(df)} bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Load indicators
    indicators_to_check = ['supertrend', 'squeeze_momentum', 'macd', 'wavetrend', 'adx_di']
    
    print("\n" + "-"*60)
    print("INDICATOR DATA CHECK:")
    print("-"*60)
    
    indicator_data = {}
    for ind in indicators_to_check:
        ind_df = csv_manager.load_indicator_data(symbol, timeframe, ind)
        if ind_df is not None:
            indicator_data[ind] = ind_df
            print(f"{ind:20s}: {len(ind_df)} bars loaded")
            
            # Check columns and latest signal
            print(f"  Columns: {', '.join(ind_df.columns[:5])}...")
            if 'signal' in ind_df.columns:
                last_signal = ind_df['signal'].iloc[-1]
                signal_text = "BUY" if last_signal > 0 else "SELL" if last_signal < 0 else "NEUTRAL"
                print(f"  Latest signal: {signal_text}")
        else:
            print(f"{ind:20s}: No data found")
    
    # Simple backtest
    print("\n" + "-"*60)
    print("SIMPLE BACKTEST:")
    print("-"*60)
    
    if 'supertrend' in indicator_data and 'squeeze_momentum' in indicator_data:
        # Get common index
        common_idx = df.index.intersection(indicator_data['supertrend'].index)
        common_idx = common_idx.intersection(indicator_data['squeeze_momentum'].index)
        
        df_aligned = df.loc[common_idx]
        st_aligned = indicator_data['supertrend'].loc[common_idx]
        sq_aligned = indicator_data['squeeze_momentum'].loc[common_idx]
        
        # Count potential trades
        buy_signals = 0
        sell_signals = 0
        
        for i in range(100, len(common_idx)):
            # Check for buy signal
            if (st_aligned['signal'].iloc[i] > 0 and 
                sq_aligned['signal'].iloc[i] > 0 and
                st_aligned['signal'].iloc[i-1] <= 0):  # New signal
                buy_signals += 1
                
            # Check for sell signal
            elif (st_aligned['signal'].iloc[i] < 0 and 
                  sq_aligned['signal'].iloc[i] < 0 and
                  st_aligned['signal'].iloc[i-1] >= 0):  # New signal
                sell_signals += 1
        
        print(f"Potential BUY signals: {buy_signals}")
        print(f"Potential SELL signals: {sell_signals}")
        print(f"Total signals: {buy_signals + sell_signals}")
        
        # Show recent signals
        print("\n" + "-"*60)
        print("LAST 10 SIGNAL CHANGES:")
        print("-"*60)
        
        signal_changes = []
        for i in range(100, len(common_idx)):
            curr_st = st_aligned['signal'].iloc[i]
            prev_st = st_aligned['signal'].iloc[i-1]
            
            curr_sq = sq_aligned['signal'].iloc[i]
            
            # Signal change detected
            if curr_st != prev_st and curr_st != 0 and curr_sq == curr_st:
                signal_type = "BUY" if curr_st > 0 else "SELL"
                price = df_aligned['close'].iloc[i]
                date = df_aligned.index[i]
                
                signal_changes.append({
                    'date': date,
                    'type': signal_type,
                    'price': price
                })
        
        # Show last 10
        for signal in signal_changes[-10:]:
            print(f"{signal['date']}: {signal['type']} at {signal['price']:.2f}")
    
    # Current status
    print("\n" + "-"*60)
    print("CURRENT STATUS:")
    print("-"*60)
    
    last_price = df['close'].iloc[-1]
    print(f"Last price: {last_price:.2f}")
    print(f"Last update: {df.index[-1]}")
    
    # Check all current signals
    print("\nCurrent indicator signals:")
    for ind_name, ind_df in indicator_data.items():
        if 'signal' in ind_df.columns:
            last_signal = ind_df['signal'].iloc[-1]
            signal_text = "BUY" if last_signal > 0 else "SELL" if last_signal < 0 else "NEUTRAL"
            print(f"  {ind_name:20s}: {signal_text}")


if __name__ == "__main__":
    analyze_thyao_signals()