#!/usr/bin/env python3
"""
Single Symbol Backtest Test
"""

import sys
from pathlib import Path
import pandas as pd

# Proje kökünü ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtest.simple_indicator_backtest import SimpleIndicatorBacktest
from utils.csv_data_manager import CSVDataManager
import json

def test_single_symbol():
    """Test backtest with single symbol"""
    
    print("\n" + "="*60)
    print("SINGLE SYMBOL BACKTEST TEST")
    print("="*60)
    
    # Parameters
    symbol = 'THYAO'
    timeframe = '1h'
    initial_capital = 50000
    
    print(f"\nSymbol: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Initial Capital: {initial_capital:,} TL")
    print("\n" + "-"*60)
    
    # Create backtest instance
    bt = SimpleIndicatorBacktest(initial_capital=initial_capital, max_positions=5)
    
    # Run backtest for single symbol
    print(f"\nRunning backtest for {symbol}...")
    
    try:
        # Load raw data to check availability
        csv_manager = CSVDataManager()
        df = csv_manager.load_raw_data(symbol, timeframe)
        
        if df is not None:
            print(f"Data available: {len(df)} bars")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            
            # Run backtest for single symbol by modifying ASSETS temporarily
            from config import assets
            original_assets = assets.ASSETS.copy()
            assets.ASSETS = [symbol]  # Temporarily set only our symbol
            
            bt.run_all_symbols(timeframe)
            
            # Restore original assets
            assets.ASSETS = original_assets
            
            # Get results from the backtest
            result_file = Path(f'backtest_results/simple_indicator_{timeframe}_results.json')
            result = None
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    all_results = json.load(f)
                    # Find our symbol's result
                    for r in all_results.get('symbol_results', []):
                        if r['symbol'] == symbol:
                            result = r
                            break
            
            if result:
                print("\n" + "="*60)
                print("BACKTEST RESULTS")
                print("="*60)
                
                print(f"\nTotal Return: {result['total_return']*100:.2f}%")
                print(f"Final Capital: {result['final_capital']:,.2f} TL")
                print(f"Total Trades: {result['total_trades']}")
                print(f"Winning Trades: {result['winning_trades']}")
                print(f"Win Rate: {result['win_rate']*100:.1f}%")
                
                if result['total_trades'] > 0:
                    print(f"\nProfit Factor: {result.get('profit_factor', 0):.2f}")
                    print(f"Average Win: {result.get('avg_win', 0)*100:.2f}%")
                    print(f"Average Loss: {result.get('avg_loss', 0)*100:.2f}%")
                    print(f"Max Drawdown: {result.get('max_drawdown', 0)*100:.2f}%")
                    
                    # Confirmation analysis
                    if 'confirmation_stats' in result:
                        print("\n" + "-"*60)
                        print("CONFIRMATION STATISTICS")
                        print("-"*60)
                        conf_stats = result['confirmation_stats']
                        for conf, stats in conf_stats.items():
                            if stats['count'] > 0:
                                print(f"{conf:15s}: {stats['count']:3d} trades, "
                                      f"Win rate: {stats['win_rate']*100:5.1f}%")
                
                # Recent trades
                if 'trades' in result and len(result['trades']) > 0:
                    print("\n" + "-"*60)
                    print("LAST 5 TRADES")
                    print("-"*60)
                    
                    recent_trades = result['trades'][-5:]
                    for i, trade in enumerate(recent_trades, 1):
                        pnl_pct = trade['pnl_percentage'] * 100
                        print(f"{i}. {trade['entry_date']}: "
                              f"{'WIN' if trade['pnl'] > 0 else 'LOSS'} "
                              f"{pnl_pct:+6.2f}% "
                              f"({trade['holding_days']} days)")
                              
            else:
                print("No trades generated!")
                
        else:
            print(f"Error: No data available for {symbol}")
            
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_single_symbol()