#!/usr/bin/env python3
"""
Aggressive TriMode Backtest - Quick test version
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import json
from typing import Dict, List, Tuple
from tqdm import tqdm

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.trimode_orchestrator_aggressive import AggressiveTriModeOrchestrator, TradingMode
from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS


def run_quick_backtest(symbols: List[str], start_date: str = '2024-06-01', 
                      end_date: str = '2025-06-26', timeframe: str = '1d'):
    """Quick backtest without optimization"""
    
    print("="*80)
    print("AGGRESSIVE TRIMODE QUICK BACKTEST")
    print("="*80)
    print(f"Testing {len(symbols)} symbols from {start_date} to {end_date}")
    
    orchestrator = AggressiveTriModeOrchestrator(initial_capital=100000)
    csv_manager = CSVDataManager()
    
    # Get date range
    sample_data = csv_manager.load_raw_data(symbols[0], timeframe)
    if sample_data is None:
        logger.error("No data available")
        return
    
    sample_data = sample_data[start_date:end_date]
    dates = sample_data.index[100:]  # Skip first 100 for indicators
    
    # Tracking
    portfolio_history = []
    trade_history = []
    mode_history = []
    daily_returns = []
    
    print(f"\nRunning backtest for {len(dates)} days...")
    pbar = tqdm(total=len(dates), desc="Backtesting")
    
    for i, current_date in enumerate(dates):
        pbar.update(1)
        
        # Prepare market data
        market_data = {
            'symbol_data': {},
            'timeframe': timeframe
        }
        
        # Load data for each symbol
        for symbol in symbols:
            data = csv_manager.load_raw_data(symbol, timeframe)
            if data is not None:
                historical_data = data[data.index <= current_date]
                if len(historical_data) >= 100:
                    market_data['symbol_data'][symbol] = historical_data
        
        # 1. Determine mode
        mode = orchestrator.determine_mode(market_data)
        mode_history.append({
            'date': current_date,
            'mode': mode.value
        })
        
        # 2. Get current prices
        current_prices = {}
        for symbol in symbols:
            if symbol in market_data['symbol_data']:
                current_prices[symbol] = market_data['symbol_data'][symbol]['close'].iloc[-1]
        
        # 3. Manage positions
        closed_trades = orchestrator.manage_positions(market_data['symbol_data'])
        for trade in closed_trades:
            trade_history.append({**trade, 'date': current_date})
        
        # 4. Generate signals
        available_symbols = [s for s in symbols if s not in orchestrator.positions]
        signals = orchestrator.generate_signals(available_symbols, timeframe)
        
        # 5. Execute trades
        executed_trades = orchestrator.execute_trades(signals)
        for trade in executed_trades:
            trade_history.append({**trade, 'date': current_date})
        
        # 6. Calculate portfolio value
        portfolio_value = orchestrator.current_capital
        for symbol, position in orchestrator.positions.items():
            if symbol in current_prices:
                position['current_price'] = current_prices[symbol]
                portfolio_value += position['shares'] * current_prices[symbol]
        
        # 7. Record performance
        portfolio_history.append({
            'date': current_date,
            'total_value': portfolio_value,
            'cash': orchestrator.current_capital,
            'positions': len(orchestrator.positions),
            'mode': mode.value
        })
        
        # Calculate daily return
        if i > 0:
            prev_value = portfolio_history[-2]['total_value']
            daily_return = (portfolio_value - prev_value) / prev_value
            daily_returns.append(daily_return)
            orchestrator.daily_returns.append(daily_return)
    
    pbar.close()
    
    # Calculate results
    initial_value = 100000
    final_value = portfolio_history[-1]['total_value']
    total_return = (final_value - initial_value) / initial_value
    
    # Monthly return
    total_days = len(portfolio_history)
    monthly_return = total_return * 30 / total_days if total_days > 0 else 0
    
    # Trade analysis
    sell_trades = [t for t in trade_history if t['action'] == 'SELL']
    if sell_trades:
        win_rate = sum(1 for t in sell_trades if t['pnl'] > 0) / len(sell_trades)
        avg_win = np.mean([t['pnl'] for t in sell_trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in sell_trades) else 0
        avg_loss = np.mean([t['pnl'] for t in sell_trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in sell_trades) else 0
    else:
        win_rate = avg_win = avg_loss = 0
    
    # Mode distribution
    mode_df = pd.DataFrame(mode_history)
    mode_counts = mode_df['mode'].value_counts().to_dict()
    
    # Print results
    print("\n" + "="*80)
    print("AGGRESSIVE BACKTEST RESULTS")
    print("="*80)
    print(f"\nCAPITAL:")
    print(f"  Initial: ${initial_value:,.0f}")
    print(f"  Final: ${final_value:,.0f}")
    print(f"  Total Return: {total_return:.1%}")
    print(f"  Monthly Return: {monthly_return:.1%}")
    
    print(f"\nTRADES:")
    print(f"  Total Completed: {len(sell_trades)}")
    print(f"  Win Rate: {win_rate:.1%}")
    print(f"  Avg Win: ${avg_win:.2f}")
    print(f"  Avg Loss: ${avg_loss:.2f}")
    print(f"  Open Positions: {len(orchestrator.positions)}")
    
    print(f"\nMODE DISTRIBUTION:")
    for mode, count in mode_counts.items():
        pct = count / len(mode_history) * 100
        print(f"  {mode}: {count} days ({pct:.1f}%)")
    
    # Check target
    if monthly_return >= 0.10:
        print(f"\n✅ MONTHLY TARGET ACHIEVED: {monthly_return:.1%} >= 10%")
    else:
        print(f"\n❌ MONTHLY TARGET MISSED: {monthly_return:.1%} < 10%")
    
    print("="*80)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'initial_capital': initial_value,
        'final_capital': final_value,
        'total_return': total_return,
        'monthly_return': monthly_return,
        'total_trades': len(sell_trades),
        'win_rate': win_rate,
        'mode_distribution': mode_counts,
        'test_days': total_days
    }
    
    with open(f'backtest/aggressive_quick_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    # Quick test with top 30 symbols, full year
    symbols = ASSETS[:30]
    results = run_quick_backtest(symbols, start_date='2024-01-01', end_date='2025-06-26')