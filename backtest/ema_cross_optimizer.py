#!/usr/bin/env python3
"""
EMA Cross Strategy Optimizer with Binary Search
Finds optimal EMA periods for each symbol using binary search
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


class EMACrossOptimizer:
    """Optimize EMA cross strategy parameters using binary search"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.csv_manager = CSVDataManager()
        self.initial_capital = initial_capital
        self.commission_rate = 0.002  # %0.2 BIST commission
        
        # EMA ranges
        self.fast_ema_range = (7, 50)
        self.slow_ema_range = (60, 210)
        
        # Optimization parameters
        self.min_improvement = 0.001  # %0.1 minimum improvement to continue search
        self.max_iterations = 10  # Maximum binary search iterations
        
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def backtest_ema_cross(self, symbol: str, fast_period: int, slow_period: int, 
                          start_date: str = None, end_date: str = None) -> Dict:
        """Backtest EMA cross strategy for a single symbol"""
        
        # Load data
        data = self.csv_manager.load_raw_data(symbol, '1d')
        if data is None or len(data) < slow_period + 50:
            return {'total_return': -1.0, 'sharpe': -999}
        
        # Filter date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
            
        # Calculate EMAs
        data['ema_fast'] = self.calculate_ema(data['close'], fast_period)
        data['ema_slow'] = self.calculate_ema(data['close'], slow_period)
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['ema_fast'] > data['ema_slow'], 'signal'] = 1
        data.loc[data['ema_fast'] <= data['ema_slow'], 'signal'] = -1
        
        # Get actual trading signals (changes only)
        data['position'] = data['signal'].diff()
        
        # Skip warm-up period
        data = data.iloc[slow_period:]
        
        # Track trades
        cash = self.initial_capital
        position = 0
        trades = []
        
        for idx, row in data.iterrows():
            # Buy signal
            if row['position'] > 0 and position == 0:
                # Buy with all available cash
                shares = int(cash / (row['close'] * (1 + self.commission_rate)))
                if shares > 0:
                    cost = shares * row['close'] * (1 + self.commission_rate)
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
        
        # Calculate metrics
        final_value = cash
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate daily returns for Sharpe ratio
        if len(trades) > 2:
            # Create equity curve
            equity_curve = pd.Series(index=data.index, dtype=float)
            equity_curve.iloc[0] = self.initial_capital
            
            current_cash = self.initial_capital
            current_position = 0
            trade_idx = 0
            
            for i, (idx, row) in enumerate(data.iterrows()):
                if trade_idx < len(trades) and trades[trade_idx]['date'] == idx:
                    trade = trades[trade_idx]
                    if trade['type'] == 'BUY':
                        current_cash = cash - trade['cost']
                        current_position = trade['shares']
                    else:  # SELL
                        current_cash = cash + trade['proceeds']
                        current_position = 0
                    trade_idx += 1
                
                # Calculate current equity
                equity_value = current_cash
                if current_position > 0:
                    equity_value += current_position * row['close']
                equity_curve.iloc[i] = equity_value
            
            # Calculate Sharpe ratio
            daily_returns = equity_curve.pct_change().dropna()
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Win rate
        win_trades = 0
        loss_trades = 0
        
        for i in range(0, len(trades)-1, 2):  # Pair buy/sell trades
            if i+1 < len(trades):
                buy_cost = trades[i]['cost']
                sell_proceeds = trades[i+1]['proceeds']
                if sell_proceeds > buy_cost:
                    win_trades += 1
                else:
                    loss_trades += 1
        
        win_rate = win_trades / (win_trades + loss_trades) if (win_trades + loss_trades) > 0 else 0
        
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
    
    def binary_search_optimization(self, symbol: str, start_date: str = None, 
                                 end_date: str = None) -> Dict:
        """Use binary search to find optimal EMA periods"""
        
        logger.info(f"\nOptimizing {symbol}...")
        
        best_result = {'total_return': -999, 'sharpe_ratio': -999}
        
        # Start with middle values
        fast_low, fast_high = self.fast_ema_range
        slow_low, slow_high = self.slow_ema_range
        
        # Binary search for fast EMA
        iteration = 0
        while iteration < self.max_iterations and (fast_high - fast_low) > 2:
            # Test three points
            fast_mid1 = fast_low + (fast_high - fast_low) // 3
            fast_mid2 = fast_high - (fast_high - fast_low) // 3
            
            # For slow EMA, test middle value initially
            slow_mid = (slow_low + slow_high) // 2
            
            # Test both mid points
            result1 = self.backtest_ema_cross(symbol, fast_mid1, slow_mid, start_date, end_date)
            result2 = self.backtest_ema_cross(symbol, fast_mid2, slow_mid, start_date, end_date)
            
            logger.info(f"  Fast EMA {fast_mid1}/{slow_mid}: Return={result1['total_return']:.1%}, Sharpe={result1['sharpe_ratio']:.2f}")
            logger.info(f"  Fast EMA {fast_mid2}/{slow_mid}: Return={result2['total_return']:.1%}, Sharpe={result2['sharpe_ratio']:.2f}")
            
            # Update best result
            if result1['sharpe_ratio'] > best_result['sharpe_ratio']:
                best_result = result1
            if result2['sharpe_ratio'] > best_result['sharpe_ratio']:
                best_result = result2
            
            # Narrow the search range
            if result1['sharpe_ratio'] > result2['sharpe_ratio']:
                fast_high = fast_mid2
            else:
                fast_low = fast_mid1
            
            iteration += 1
        
        # Now optimize slow EMA with best fast EMA
        best_fast = best_result.get('fast_ema', (fast_low + fast_high) // 2)
        
        iteration = 0
        while iteration < self.max_iterations and (slow_high - slow_low) > 5:
            # Test three points
            slow_mid1 = slow_low + (slow_high - slow_low) // 3
            slow_mid2 = slow_high - (slow_high - slow_low) // 3
            
            # Test both mid points
            result1 = self.backtest_ema_cross(symbol, best_fast, slow_mid1, start_date, end_date)
            result2 = self.backtest_ema_cross(symbol, best_fast, slow_mid2, start_date, end_date)
            
            logger.info(f"  EMA {best_fast}/{slow_mid1}: Return={result1['total_return']:.1%}, Sharpe={result1['sharpe_ratio']:.2f}")
            logger.info(f"  EMA {best_fast}/{slow_mid2}: Return={result2['total_return']:.1%}, Sharpe={result2['sharpe_ratio']:.2f}")
            
            # Update best result
            if result1['sharpe_ratio'] > best_result['sharpe_ratio']:
                best_result = result1
            if result2['sharpe_ratio'] > best_result['sharpe_ratio']:
                best_result = result2
            
            # Narrow the search range
            if result1['sharpe_ratio'] > result2['sharpe_ratio']:
                slow_high = slow_mid2
            else:
                slow_low = slow_mid1
            
            iteration += 1
        
        # Fine-tune around best values
        if best_result['fast_ema'] > 0:
            logger.info(f"\nFine-tuning around {best_result['fast_ema']}/{best_result['slow_ema']}...")
            
            for fast_offset in [-2, -1, 0, 1, 2]:
                for slow_offset in [-5, 0, 5]:
                    fast_test = best_result['fast_ema'] + fast_offset
                    slow_test = best_result['slow_ema'] + slow_offset
                    
                    # Check bounds
                    if (fast_test < self.fast_ema_range[0] or fast_test > self.fast_ema_range[1] or
                        slow_test < self.slow_ema_range[0] or slow_test > self.slow_ema_range[1] or
                        fast_test >= slow_test - 10):
                        continue
                    
                    result = self.backtest_ema_cross(symbol, fast_test, slow_test, start_date, end_date)
                    
                    if result['sharpe_ratio'] > best_result['sharpe_ratio']:
                        logger.info(f"  Better parameters found: {fast_test}/{slow_test} - Sharpe={result['sharpe_ratio']:.2f}")
                        best_result = result
        
        logger.info(f"\nBest for {symbol}: EMA {best_result.get('fast_ema', 0)}/{best_result.get('slow_ema', 0)}")
        logger.info(f"  Return: {best_result.get('total_return', 0):.1%}")
        logger.info(f"  Sharpe: {best_result.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Win Rate: {best_result.get('win_rate', 0):.1%}")
        
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
            'total_return': -999
        }
        
        # Test combinations
        fast_range = range(10, 45, 5)  # 10, 15, 20, 25, 30, 35, 40
        slow_range = range(70, 200, 20)  # 70, 90, 110, 130, 150, 170, 190
        
        for fast in fast_range:
            for slow in slow_range:
                if slow <= fast + 20:  # Ensure enough gap
                    continue
                
                group_returns = []
                group_sharpes = []
                group_final_values = []
                
                # Test on all symbols
                for symbol in symbols:
                    result = self.backtest_ema_cross(symbol, fast, slow, start_date, end_date)
                    if result['total_return'] > -0.9:  # Valid result
                        group_returns.append(result['total_return'])
                        group_sharpes.append(result['sharpe_ratio'])
                        group_final_values.append(result['final_value'])
                
                if len(group_returns) == 0:
                    continue
                
                # Calculate group metrics
                avg_return = np.mean(group_returns)
                avg_sharpe = np.mean(group_sharpes)
                
                # Simulate equal weight portfolio
                portfolio_value = np.mean(group_final_values)
                portfolio_return = (portfolio_value - self.initial_capital) / self.initial_capital
                
                logger.info(f"  EMA {fast}/{slow}: Avg Return={avg_return:.1%}, Avg Sharpe={avg_sharpe:.2f}, Portfolio={portfolio_return:.1%}")
                
                # Check if better
                if avg_sharpe > best_result['avg_sharpe']:
                    best_result = {
                        'fast_ema': fast,
                        'slow_ema': slow,
                        'avg_return': avg_return,
                        'avg_sharpe': avg_sharpe,
                        'total_return': portfolio_return,
                        'individual_results': list(zip(symbols, group_returns, group_sharpes))
                    }
        
        # Fine-tune best group parameters
        if best_result['fast_ema'] > 0:
            logger.info(f"\nFine-tuning group parameters around {best_result['fast_ema']}/{best_result['slow_ema']}...")
            
            for fast_offset in [-3, -1, 0, 1, 3]:
                for slow_offset in [-10, -5, 0, 5, 10]:
                    fast_test = best_result['fast_ema'] + fast_offset
                    slow_test = best_result['slow_ema'] + slow_offset
                    
                    if (fast_test < self.fast_ema_range[0] or fast_test > self.fast_ema_range[1] or
                        slow_test < self.slow_ema_range[0] or slow_test > self.slow_ema_range[1] or
                        fast_test >= slow_test - 20):
                        continue
                    
                    group_sharpes = []
                    for symbol in symbols:
                        result = self.backtest_ema_cross(symbol, fast_test, slow_test, start_date, end_date)
                        if result['total_return'] > -0.9:
                            group_sharpes.append(result['sharpe_ratio'])
                    
                    if len(group_sharpes) > 0:
                        avg_sharpe = np.mean(group_sharpes)
                        if avg_sharpe > best_result['avg_sharpe']:
                            logger.info(f"  Better group parameters: {fast_test}/{slow_test} - Avg Sharpe={avg_sharpe:.2f}")
                            best_result['fast_ema'] = fast_test
                            best_result['slow_ema'] = slow_test
                            best_result['avg_sharpe'] = avg_sharpe
        
        logger.info(f"\nBest GROUP parameters: EMA {best_result['fast_ema']}/{best_result['slow_ema']}")
        logger.info(f"  Average Return: {best_result['avg_return']:.1%}")
        logger.info(f"  Average Sharpe: {best_result['avg_sharpe']:.2f}")
        logger.info(f"  Portfolio Return: {best_result['total_return']:.1%}")
        
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
                        self.binary_search_optimization, 
                        symbol, start_date, end_date
                    )
                    futures.append(future)
                
                for future in futures:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error in optimization: {e}")
        else:
            # Sequential optimization
            for symbol in symbols:
                try:
                    result = self.binary_search_optimization(symbol, start_date, end_date)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error optimizing {symbol}: {e}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by Sharpe ratio
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f'backtest/ema_optimization_results_{timestamp}.csv'
        results_df.to_csv(results_file, index=False)
        
        # Print summary
        print("\n" + "="*80)
        print("EMA CROSS OPTIMIZATION RESULTS")
        print("="*80)
        print(f"\nTop 10 Symbols by Sharpe Ratio:")
        print(results_df.head(10)[['symbol', 'fast_ema', 'slow_ema', 'total_return', 'sharpe_ratio', 'win_rate']].to_string())
        
        print(f"\n\nAverage Results:")
        print(f"  Average Return: {results_df['total_return'].mean():.1%}")
        print(f"  Average Sharpe: {results_df['sharpe_ratio'].mean():.2f}")
        print(f"  Average Win Rate: {results_df['win_rate'].mean():.1%}")
        
        print(f"\nResults saved to: {results_file}")
        print("="*80)
        
        return results_df


def main():
    """Run EMA cross optimization"""
    optimizer = EMACrossOptimizer()
    
    # Test with a few symbols first
    test_symbols = ASSETS[:54]  # Start with 5 symbols
    
    # You can test with all symbols: test_symbols = ASSETS
    
    # 1. Find optimal parameters for the GROUP
    logger.info("="*80)
    logger.info("PHASE 1: GROUP OPTIMIZATION")
    logger.info("="*80)
    
    group_result = optimizer.optimize_group(
        symbols=test_symbols,
        start_date='2024-01-01',
        end_date='2025-06-26'
    )
    
    # 2. Find optimal parameters for EACH symbol individually
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: INDIVIDUAL OPTIMIZATION")
    logger.info("="*80)
    
    individual_results = optimizer.optimize_all_symbols(
        symbols=test_symbols,
        start_date='2024-01-01',
        end_date='2025-06-26',
        parallel=True
    )
    
    # 3. Compare group vs individual performance
    logger.info("\n" + "="*80)
    logger.info("COMPARISON: GROUP vs INDIVIDUAL")
    logger.info("="*80)
    
    # Test group parameters on all symbols
    group_performance = []
    for symbol in test_symbols:
        result = optimizer.backtest_ema_cross(
            symbol, 
            group_result['fast_ema'], 
            group_result['slow_ema'],
            '2024-01-01',
            '2025-06-26'
        )
        group_performance.append({
            'symbol': symbol,
            'group_return': result['total_return'],
            'group_sharpe': result['sharpe_ratio']
        })
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(group_performance)
    
    # Add individual results
    for idx, row in individual_results.iterrows():
        symbol = row['symbol']
        comparison_df.loc[comparison_df['symbol'] == symbol, 'individual_return'] = row['total_return']
        comparison_df.loc[comparison_df['symbol'] == symbol, 'individual_sharpe'] = row['sharpe_ratio']
        comparison_df.loc[comparison_df['symbol'] == symbol, 'individual_fast'] = row['fast_ema']
        comparison_df.loc[comparison_df['symbol'] == symbol, 'individual_slow'] = row['slow_ema']
    
    # Calculate differences
    comparison_df['return_diff'] = comparison_df['individual_return'] - comparison_df['group_return']
    comparison_df['sharpe_diff'] = comparison_df['individual_sharpe'] - comparison_df['group_sharpe']
    
    print("\nComparison Results:")
    print(comparison_df.to_string())
    
    print(f"\n\nGROUP Strategy (EMA {group_result['fast_ema']}/{group_result['slow_ema']}):")
    print(f"  Average Return: {comparison_df['group_return'].mean():.1%}")
    print(f"  Average Sharpe: {comparison_df['group_sharpe'].mean():.2f}")
    
    print(f"\nINDIVIDUAL Strategy:")
    print(f"  Average Return: {comparison_df['individual_return'].mean():.1%}")
    print(f"  Average Sharpe: {comparison_df['individual_sharpe'].mean():.2f}")
    
    print(f"\nBenefit of Individual Optimization:")
    print(f"  Return Improvement: {comparison_df['return_diff'].mean():.1%}")
    print(f"  Sharpe Improvement: {comparison_df['sharpe_diff'].mean():.2f}")
    
    # Save comparison
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comparison_df.to_csv(f'backtest/ema_comparison_{timestamp}.csv', index=False)
    
    return group_result, individual_results, comparison_df


if __name__ == "__main__":
    main()