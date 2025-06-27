#!/usr/bin/env python3
"""
Quick THYAO Backtest
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.csv_data_manager import CSVDataManager

def quick_backtest():
    """Quick and simple backtest for THYAO"""
    
    print("\n" + "="*60)
    print("QUICK THYAO BACKTEST")
    print("="*60)
    
    csv_manager = CSVDataManager()
    symbol = 'THYAO'
    timeframe = '1h'
    
    # Load data
    df = csv_manager.load_raw_data(symbol, timeframe)
    st_data = csv_manager.load_indicator_data(symbol, timeframe, 'supertrend')
    
    if df is None or st_data is None:
        print("Error loading data")
        return
        
    # Align data
    common_idx = df.index.intersection(st_data.index)
    df = df.loc[common_idx]
    st = st_data.loc[common_idx]
    
    # Simple strategy: Buy on supertrend buy signal, sell on sell signal
    initial_capital = 50000
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    
    print(f"\nInitial capital: {initial_capital:,} TL")
    print(f"Testing period: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df)}")
    
    # Check if buy/sell signals exist
    if 'buy_signal' not in st.columns:
        print("\nError: buy_signal column not found in supertrend data")
        print(f"Available columns: {list(st.columns)}")
        return
    
    # Run backtest
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        
        # Buy signal
        if st['buy_signal'].iloc[i] and position == 0:
            shares = int(capital / current_price)
            if shares > 0:
                position = shares
                entry_price = current_price
                capital -= shares * current_price
        
        # Sell signal
        elif st['sell_signal'].iloc[i] and position > 0:
            exit_price = current_price
            capital += position * exit_price
            pnl = (exit_price - entry_price) * position
            pnl_pct = (exit_price - entry_price) / entry_price
            
            trades.append({
                'entry_price': entry_price,
                'exit_price': exit_price,
                'shares': position,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'capital_after': capital
            })
            
            position = 0
    
    # Close open position
    if position > 0:
        exit_price = df['close'].iloc[-1]
        capital += position * exit_price
        pnl = (exit_price - entry_price) * position
        pnl_pct = (exit_price - entry_price) / entry_price
        
        trades.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': position,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'capital_after': capital
        })
    
    # Results
    final_capital = capital
    total_return = (final_capital - initial_capital) / initial_capital
    
    print("\n" + "-"*60)
    print("RESULTS:")
    print("-"*60)
    print(f"Final capital: {final_capital:,.2f} TL")
    print(f"Total return: {total_return*100:.2f}%")
    print(f"Total trades: {len(trades)}")
    
    if trades:
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        win_rate = winning_trades / len(trades)
        avg_win = np.mean([t['pnl_pct'] for t in trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl'] < 0]) if winning_trades < len(trades) else 0
        
        print(f"Win rate: {win_rate*100:.1f}%")
        print(f"Average win: {avg_win*100:.2f}%")
        print(f"Average loss: {avg_loss*100:.2f}%")
        
        # Last 5 trades
        print("\n" + "-"*60)
        print("LAST 5 TRADES:")
        print("-"*60)
        for i, trade in enumerate(trades[-5:], 1):
            print(f"{i}. Buy: {trade['entry_price']:.2f} -> Sell: {trade['exit_price']:.2f} "
                  f"= {trade['pnl_pct']*100:+.2f}% ({trade['pnl']:+,.2f} TL)")


if __name__ == "__main__":
    quick_backtest()