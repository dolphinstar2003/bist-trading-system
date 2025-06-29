#!/usr/bin/env python3
"""
Fixed EMA Cross Strategy Optimizer with Binary Search
Corrected logic and better optimization approach
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import json
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS


class EMACrossOptimizerFixed:
    """Fixed version of EMA cross strategy optimizer"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.csv_manager = CSVDataManager()
        self.initial_capital = initial_capital
        self.commission_rate = 0.002  # %0.2 BIST commission
        
        # EMA ranges
        self.fast_ema_range = (7, 50)
        self.slow_ema_range = (60, 210)
        
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def backtest_ema_cross(self, symbol: str, fast_period: int, slow_period: int, 
                          start_date: str = None, end_date: str = None, 
                          debug: bool = False) -> Dict:
        """Backtest EMA cross strategy for a single symbol"""
        
        try:
            # Load data
            data = self.csv_manager.load_raw_data(symbol, '1d')
            if data is None or len(data) < slow_period + 50:
                return {
                    'symbol': symbol,
                    'fast_ema': fast_period,
                    'slow_ema': slow_period,
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'num_trades': 0,
                    'win_rate': 0.0,
                    'final_value': self.initial_capital,
                    'error': 'Insufficient data'
                }
            
            # Filter date range
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # Need enough data after filtering
            if len(data) < slow_period + 20:
                return {
                    'symbol': symbol,
                    'fast_ema': fast_period,
                    'slow_ema': slow_period,
                    'total_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'num_trades': 0,
                    'win_rate': 0.0,
                    'final_value': self.initial_capital,
                    'error': 'Insufficient data after date filter'
                }
                
            # Calculate EMAs
            data['ema_fast'] = self.calculate_ema(data['close'], fast_period)
            data['ema_slow'] = self.calculate_ema(data['close'], slow_period)
            
            # Skip warm-up period
            data = data.iloc[slow_period:]
            
            # Generate signals
            data['signal'] = 0
            data.loc[data['ema_fast'] > data['ema_slow'], 'signal'] = 1
            data.loc[data['ema_fast'] <= data['ema_slow'], 'signal'] = -1
            
            # Get actual trading signals (changes only)
            data['position'] = data['signal'].diff()
            
            # Initialize portfolio tracking
            portfolio_value = []
            cash = self.initial_capital
            position = 0
            trades = []
            
            # Simulate trading
            for idx, row in data.iterrows():
                # Calculate current portfolio value
                current_value = cash
                if position > 0:
                    current_value += position * row['close']
                portfolio_value.append(current_value)
                
                # Buy signal
                if row['position'] > 0 and position == 0:
                    # Buy with all available cash
                    shares = int(cash / (row['close'] * (1 + self.commission_rate)))
                    if shares > 0:
                        cost = shares * row['close'] * (1 + self.commission_rate)
                        if cost <= cash:  # Double check
                            cash -= cost
                            position = shares
                            trades.append({
                                'date': idx,
                                'type': 'BUY',
                                'price': row['close'],
                                'shares': shares,
                                'cost': cost
                            })
                
                # Sell signal
                elif row['position'] < 0 and position > 0:
                    # Sell all position
                    proceeds = position * row['close'] * (1 - self.commission_rate)
                    cash += proceeds
                    trades.append({
                        'date': idx,
                        'type': 'SELL',
                        'price': row['close'],
                        'shares': position,
                        'proceeds': proceeds
                    })
                    position = 0
            
            # Close final position if any
            if position > 0:
                final_price = data.iloc[-1]['close']
                proceeds = position * final_price * (1 - self.commission_rate)
                cash += proceeds
                trades.append({
                    'date': data.index[-1],
                    'type': 'SELL',
                    'price': final_price,
                    'shares': position,
                    'proceeds': proceeds
                })
                portfolio_value[-1] = cash  # Update final value
            
            # Calculate metrics
            final_value = cash
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            # Calculate Sharpe ratio from portfolio values
            portfolio_series = pd.Series(portfolio_value, index=data.index)
            daily_returns = portfolio_series.pct_change().dropna()
            
            sharpe_ratio = 0.0
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            
            # Calculate win rate
            win_trades = 0
            loss_trades = 0
            
            for i in range(0, len(trades)-1, 2):  # Pair buy/sell trades
                if i+1 < len(trades) and trades[i]['type'] == 'BUY' and trades[i+1]['type'] == 'SELL':
                    buy_cost = trades[i]['cost']
                    sell_proceeds = trades[i+1]['proceeds']
                    if sell_proceeds > buy_cost:
                        win_trades += 1
                    else:
                        loss_trades += 1
            
            win_rate = win_trades / (win_trades + loss_trades) if (win_trades + loss_trades) > 0 else 0
            
            if debug:
                logger.debug(f"{symbol} EMA {fast_period}/{slow_period}: "
                           f"Return={total_return:.2%}, Sharpe={sharpe_ratio:.2f}, "
                           f"Trades={len(trades)}, WinRate={win_rate:.1%}")
            
            return {
                'symbol': symbol,
                'fast_ema': fast_period,
                'slow_ema': slow_period,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'num_trades': len(trades),
                'win_rate': win_rate,
                'final_value': final_value
            }
            
        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}")
            return {
                'symbol': symbol,
                'fast_ema': fast_period,
                'slow_ema': slow_period,
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'num_trades': 0,
                'win_rate': 0.0,
                'final_value': self.initial_capital,
                'error': str(e)
            }
    
    def grid_search_optimization(self, symbol: str, start_date: str = None, 
                               end_date: str = None) -> Dict:
        """Use grid search to find optimal EMA periods"""
        
        logger.info(f"\nOptimizing {symbol}...")
        
        best_result = {
            'symbol': symbol,
            'fast_ema': 0,
            'slow_ema': 0,
            'total_return': -999,
            'sharpe_ratio': -999,
            'num_trades': 0,
            'win_rate': 0,
            'final_value': 0
        }
        
        # Coarse grid search first
        fast_range = range(10, 50, 5)  # 10, 15, 20, 25, 30, 35, 40, 45
        slow_range = range(70, 210, 20)  # 70, 90, 110, 130, 150, 170, 190
        
        for fast in fast_range:
            for slow in slow_range:
                if slow <= fast + 20:  # Ensure enough gap
                    continue
                
                result = self.backtest_ema_cross(symbol, fast, slow, start_date, end_date)
                
                # Use total return as primary metric, Sharpe as secondary
                if (result['total_return'] > best_result['total_return'] or 
                    (result['total_return'] == best_result['total_return'] and 
                     result['sharpe_ratio'] > best_result['sharpe_ratio'])):
                    best_result = result
        
        # Fine-tune around best values
        if best_result['fast_ema'] > 0 and best_result['total_return'] > -999:
            logger.info(f"  Coarse best: EMA {best_result['fast_ema']}/{best_result['slow_ema']} "
                       f"Return={best_result['total_return']:.1%}")
            
            # Fine search around best values
            fast_center = best_result['fast_ema']
            slow_center = best_result['slow_ema']
            
            for fast_offset in range(-4, 5):  # -4 to +4
                for slow_offset in range(-10, 11, 2):  # -10 to +10, step 2
                    fast_test = fast_center + fast_offset
                    slow_test = slow_center + slow_offset
                    
                    # Check bounds
                    if (fast_test < self.fast_ema_range[0] or fast_test > self.fast_ema_range[1] or
                        slow_test < self.slow_ema_range[0] or slow_test > self.slow_ema_range[1] or
                        fast_test >= slow_test - 20):
                        continue
                    
                    result = self.backtest_ema_cross(symbol, fast_test, slow_test, start_date, end_date)
                    
                    if (result['total_return'] > best_result['total_return'] or 
                        (result['total_return'] == best_result['total_return'] and 
                         result['sharpe_ratio'] > best_result['sharpe_ratio'])):
                        logger.info(f"  Better: EMA {fast_test}/{slow_test} "
                                   f"Return={result['total_return']:.1%}")
                        best_result = result
        
        # If still no trades found, try some specific good combinations
        if best_result['num_trades'] == 0:
            logger.info(f"  No trades found, trying common parameters...")
            common_params = [(20, 100), (14, 100), (21, 89), (13, 55), (26, 100)]
            
            for fast, slow in common_params:
                # Adjust slow if needed
                if slow < self.slow_ema_range[0]:
                    slow = self.slow_ema_range[0]
                    
                result = self.backtest_ema_cross(symbol, fast, slow, start_date, end_date)
                if result['num_trades'] > 0 and result['total_return'] > best_result['total_return']:
                    best_result = result
        
        logger.info(f"Best for {symbol}: EMA {best_result['fast_ema']}/{best_result['slow_ema']}")
        logger.info(f"  Return: {best_result['total_return']:.1%}")
        logger.info(f"  Sharpe: {best_result['sharpe_ratio']:.2f}")
        logger.info(f"  Trades: {best_result['num_trades']}")
        logger.info(f"  Win Rate: {best_result['win_rate']:.1%}")
        
        return best_result
    
    def optimize_group(self, symbols: List[str], start_date: str = None, 
                      end_date: str = None) -> Dict:
        """Find optimal EMA parameters for a group of symbols"""
        
        logger.info(f"\nOptimizing GROUP of {len(symbols)} symbols...")
        
        best_result = {
            'fast_ema': 0,
            'slow_ema': 0,
            'avg_return': -999,
            'avg_sharpe': -999,
            'median_return': -999,
            'symbols_with_trades': 0
        }
        
        # Test combinations with coarse grid
        fast_range = [10, 14, 20, 25, 30, 35, 40]
        slow_range = [70, 90, 110, 130, 150, 170, 190]
        
        for fast in fast_range:
            for slow in slow_range:
                if slow <= fast + 20:
                    continue
                
                group_results = []
                
                # Test on all symbols
                for symbol in symbols:
                    result = self.backtest_ema_cross(symbol, fast, slow, start_date, end_date)
                    if result['num_trades'] > 0:  # Only count if trades were made
                        group_results.append(result)
                
                if len(group_results) == 0:
                    continue
                
                # Calculate group metrics
                returns = [r['total_return'] for r in group_results]
                sharpes = [r['sharpe_ratio'] for r in group_results]
                
                avg_return = np.mean(returns)
                avg_sharpe = np.mean(sharpes)
                median_return = np.median(returns)
                
                logger.info(f"  EMA {fast}/{slow}: AvgReturn={avg_return:.1%}, "
                           f"MedianReturn={median_return:.1%}, "
                           f"SymbolsWithTrades={len(group_results)}/{len(symbols)}")
                
                # Use median return as primary metric (more robust)
                if (median_return > best_result['median_return'] or
                    (median_return == best_result['median_return'] and 
                     len(group_results) > best_result['symbols_with_trades'])):
                    best_result = {
                        'fast_ema': fast,
                        'slow_ema': slow,
                        'avg_return': avg_return,
                        'avg_sharpe': avg_sharpe,
                        'median_return': median_return,
                        'symbols_with_trades': len(group_results),
                        'individual_results': group_results
                    }
        
        logger.info(f"\nBest GROUP parameters: EMA {best_result['fast_ema']}/{best_result['slow_ema']}")
        logger.info(f"  Average Return: {best_result['avg_return']:.1%}")
        logger.info(f"  Median Return: {best_result['median_return']:.1%}")
        logger.info(f"  Average Sharpe: {best_result['avg_sharpe']:.2f}")
        logger.info(f"  Symbols with trades: {best_result['symbols_with_trades']}/{len(symbols)}")
        
        return best_result
    
    def optimize_all_symbols(self, symbols: List[str], start_date: str = None, 
                           end_date: str = None, parallel: bool = True) -> pd.DataFrame:
        """Optimize EMA parameters for all symbols"""
        
        logger.info(f"Starting optimization for {len(symbols)} symbols...")
        logger.info(f"Fast EMA range: {self.fast_ema_range}")
        logger.info(f"Slow EMA range: {self.slow_ema_range}")
        
        results = []
        
        if parallel:
            # Parallel optimization
            with ProcessPoolExecutor(max_workers=4) as executor:
                futures = []
                for symbol in symbols:
                    future = executor.submit(
                        self.grid_search_optimization, 
                        symbol, start_date, end_date
                    )
                    futures.append((symbol, future))
                
                for symbol, future in futures:
                    try:
                        result = future.result(timeout=60)  # 60 second timeout
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error optimizing {symbol}: {e}")
                        # Add empty result
                        results.append({
                            'symbol': symbol,
                            'fast_ema': 0,
                            'slow_ema': 0,
                            'total_return': 0.0,
                            'sharpe_ratio': 0.0,
                            'num_trades': 0,
                            'win_rate': 0.0,
                            'final_value': self.initial_capital,
                            'error': str(e)
                        })
        else:
            # Sequential optimization
            for symbol in symbols:
                try:
                    result = self.grid_search_optimization(symbol, start_date, end_date)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error optimizing {symbol}: {e}")
                    results.append({
                        'symbol': symbol,
                        'fast_ema': 0,
                        'slow_ema': 0,
                        'total_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'num_trades': 0,
                        'win_rate': 0.0,
                        'final_value': self.initial_capital,
                        'error': str(e)
                    })
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by total return
        results_df = results_df.sort_values('total_return', ascending=False)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'backtest/ema_optimization_fixed_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        
        # Print summary
        print("\n" + "="*80)
        print("EMA CROSS OPTIMIZATION RESULTS (FIXED)")
        print("="*80)
        
        # Successful optimizations
        successful = results_df[results_df['num_trades'] > 0]
        print(f"\nSuccessful optimizations: {len(successful)}/{len(results_df)}")
        
        if len(successful) > 0:
            print(f"\nTop 10 Symbols by Return:")
            print(successful.head(10)[['symbol', 'fast_ema', 'slow_ema', 'total_return', 
                                     'sharpe_ratio', 'num_trades', 'win_rate']].to_string())
            
            print(f"\n\nAverage Results (successful only):")
            print(f"  Average Return: {successful['total_return'].mean():.1%}")
            print(f"  Average Sharpe: {successful['sharpe_ratio'].mean():.2f}")
            print(f"  Average Win Rate: {successful['win_rate'].mean():.1%}")
            print(f"  Average Trades: {successful['num_trades'].mean():.1f}")
        
        # Failed optimizations
        failed = results_df[results_df['num_trades'] == 0]
        if len(failed) > 0:
            print(f"\n\nFailed optimizations (no trades): {len(failed)}")
            print(failed[['symbol', 'fast_ema', 'slow_ema']].to_string())
        
        print(f"\nResults saved to: {results_file}")
        print("="*80)
        
        return results_df


def main():
    """Run fixed EMA cross optimization"""
    optimizer = EMACrossOptimizerFixed()
    
    # Test symbols
    test_symbols = ASSETS[:60]  # Start with 10 symbols
    
    # 1. Group optimization
    logger.info("="*80)
    logger.info("PHASE 1: GROUP OPTIMIZATION")
    logger.info("="*80)
    
    group_result = optimizer.optimize_group(
        symbols=test_symbols,
        start_date='2024-01-01',
        end_date='2025-06-26'
    )
    
    # 2. Individual optimization
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: INDIVIDUAL OPTIMIZATION")
    logger.info("="*80)
    
    individual_results = optimizer.optimize_all_symbols(
        symbols=test_symbols,
        start_date='2024-01-01',
        end_date='2025-06-26',
        parallel=True
    )
    
    # 3. Test group parameters on all symbols
    logger.info("\n" + "="*80)
    logger.info("TESTING GROUP PARAMETERS ON ALL SYMBOLS")
    logger.info("="*80)
    
    if group_result['fast_ema'] > 0:
        group_performance = []
        for symbol in test_symbols:
            result = optimizer.backtest_ema_cross(
                symbol, 
                group_result['fast_ema'], 
                group_result['slow_ema'],
                '2024-01-01',
                '2025-06-26'
            )
            if result['num_trades'] > 0:
                logger.info(f"{symbol}: Return={result['total_return']:.1%}, "
                           f"Trades={result['num_trades']}, "
                           f"WinRate={result['win_rate']:.1%}")
                group_performance.append(result)
        
        if group_performance:
            avg_group_return = np.mean([r['total_return'] for r in group_performance])
            logger.info(f"\nGroup strategy average return: {avg_group_return:.1%}")


if __name__ == "__main__":
    main()