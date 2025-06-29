#!/usr/bin/env python3
"""
Parallel ML Trading System Backtest
Optimized for speed with multiprocessing and caching
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime
import json
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import functools
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.ml_trading_system_fixed import MLTradingSystem
from ml_models.ml_signal_generator import MLSignalGenerator
from config.assets import ASSETS
from utils.csv_data_manager import CSVDataManager


class MLBacktesterParallel:
    """Parallel backtest ML trading strategies"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.ml_system = MLTradingSystem()
        self.signal_generator = MLSignalGenerator()
        self.csv_manager = CSVDataManager()
        
        # Capital and costs
        self.initial_capital = initial_capital
        self.commission_rate = 0.002  # %0.2 BIST commission
        self.position_size = 0.1  # Use 10% of capital per position
        self.max_positions = 5  # Maximum concurrent positions
        
        # Risk management
        self.max_holding_days = 30  # Maximum holding period
        self.trailing_stop_pct = 0.05  # %5 trailing stop
        self.target_monthly_return = 0.09  # %9 monthly target
        self.ml_target_return = 0.02  # ML model's %2 target (10 days)
        self.ml_stop_loss = 0.05  # ML model's %5 stop loss
        
        # Performance optimization
        self.num_workers = cpu_count() - 1  # Leave one CPU free
        self.data_cache = {}  # Cache for loaded data
        self.signal_cache = {}  # Cache for ML signals
        
    def preload_all_data(self, symbols: list, timeframe: str, start_date: str = None, end_date: str = None):
        """Preload all data into memory for faster access"""
        logger.info(f"Preloading data for {len(symbols)} symbols...")
        
        def load_symbol_data(symbol):
            data = self.csv_manager.load_raw_data(symbol, timeframe)
            if data is not None:
                if start_date:
                    data = data[data.index >= start_date]
                if end_date:
                    data = data[data.index <= end_date]
            return symbol, data
        
        # Parallel load all data
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(load_symbol_data, symbols))
        
        # Store in cache
        for symbol, data in results:
            if data is not None and len(data) > 200:
                self.data_cache[symbol] = data
                
        logger.info(f"Loaded data for {len(self.data_cache)} symbols")
        return self.data_cache
    
    def generate_signals_batch(self, symbols: list, date: pd.Timestamp, timeframe: str) -> List[Dict]:
        """Generate signals for multiple symbols in parallel"""
        
        def get_signal_for_symbol(symbol):
            # Check cache first
            cache_key = f"{symbol}_{date}_{timeframe}"
            if cache_key in self.signal_cache:
                return self.signal_cache[cache_key]
            
            # Get data from cache
            if symbol not in self.data_cache:
                return None
                
            data = self.data_cache[symbol]
            
            # Filter to current date
            data_to_date = data[data.index <= date]
            if len(data_to_date) < 200:
                return None
            
            # Get current price
            current_price = data_to_date.iloc[-1]['close']
            
            # Generate signal
            signal = self.signal_generator.generate_signal(symbol, timeframe)
            if signal and signal.signal != 0:
                result = {
                    'symbol': symbol,
                    'signal': signal.signal,
                    'confidence': signal.confidence,
                    'price': current_price
                }
                self.signal_cache[cache_key] = result
                return result
            
            return None
        
        # Generate signals in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(get_signal_for_symbol, symbols))
        
        # Filter out None values
        return [r for r in results if r is not None]
    
    def process_single_day(self, args: Tuple) -> Dict:
        """Process a single day's trading (for parallel execution)"""
        current_date, portfolio_state, symbols, timeframe = args
        
        # Deep copy portfolio state
        portfolio = {
            'cash': portfolio_state['cash'],
            'positions': portfolio_state['positions'].copy(),
            'trades': []
        }
        
        # Get signals for all symbols
        daily_signals = self.generate_signals_batch(symbols, current_date, timeframe)
        
        # Sort by confidence
        daily_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Process exits first
        positions_to_close = []
        for symbol, position in portfolio['positions'].items():
            if symbol not in self.data_cache:
                continue
                
            data = self.data_cache[symbol]
            if current_date not in data.index:
                continue
                
            current_price = data.loc[current_date, 'close']
            days_held = (current_date - position['entry_date']).days
            
            # Update highest price
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
                if current_price >= position['entry_price'] * (1 + self.ml_target_return * 0.8):
                    position['stop_loss'] = position['highest_price'] * (1 - self.trailing_stop_pct * 0.5)
                else:
                    position['stop_loss'] = max(
                        position['stop_loss'],
                        position['highest_price'] * (1 - self.trailing_stop_pct)
                    )
            
            # Check exit conditions
            exit_reason = None
            
            # Check all exit conditions
            exit_signal = next((s for s in daily_signals if s['symbol'] == symbol and s['signal'] == -1), None)
            if exit_signal:
                exit_reason = 'ML_SELL_SIGNAL'
            elif current_price >= position['target_price']:
                exit_reason = 'TARGET_REACHED'
            elif current_price <= position['stop_loss']:
                exit_reason = 'STOP_LOSS'
            elif days_held >= self.max_holding_days:
                exit_reason = 'MAX_HOLDING_PERIOD'
            elif days_held >= 20:
                position_return = (current_price - position['entry_price']) / position['entry_price']
                if position_return >= self.target_monthly_return * 0.7:
                    exit_reason = 'MONTHLY_TARGET'
            
            if exit_reason:
                positions_to_close.append((symbol, exit_reason, current_price))
        
        # Execute closes
        for symbol, exit_reason, exit_price in positions_to_close:
            position = portfolio['positions'].pop(symbol)
            
            shares = position['shares']
            entry_cost = position['cost']
            exit_value = shares * exit_price * (1 - self.commission_rate)
            pnl = exit_value - entry_cost
            pnl_pct = pnl / entry_cost
            
            portfolio['cash'] += exit_value
            
            trade = {
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'exit_date': current_date,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'shares': shares,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'confidence': position['confidence'],
                'exit_reason': exit_reason,
                'days_held': (current_date - position['entry_date']).days,
                'highest_price': position['highest_price']
            }
            portfolio['trades'].append(trade)
        
        # Open new positions
        buy_signals = [s for s in daily_signals if s['signal'] == 1]
        
        for signal in buy_signals:
            if len(portfolio['positions']) >= self.max_positions:
                break
                
            if signal['symbol'] in portfolio['positions']:
                continue
                
            position_value = portfolio['cash'] * self.position_size
            if position_value < 1000:
                continue
                
            shares = int(position_value / (signal['price'] * (1 + self.commission_rate)))
            if shares < 1:
                continue
                
            cost = shares * signal['price'] * (1 + self.commission_rate)
            
            if cost > portfolio['cash']:
                continue
                
            portfolio['positions'][signal['symbol']] = {
                'entry_date': current_date,
                'entry_price': signal['price'],
                'shares': shares,
                'cost': cost,
                'confidence': signal['confidence'],
                'highest_price': signal['price'],
                'lowest_price': signal['price'],
                'target_price': signal['price'] * (1 + self.ml_target_return),
                'stop_loss': signal['price'] * (1 - self.ml_stop_loss)
            }
            
            portfolio['cash'] -= cost
        
        # Calculate portfolio value
        total_value = portfolio['cash']
        for symbol, position in portfolio['positions'].items():
            if symbol in self.data_cache and current_date in self.data_cache[symbol].index:
                current_price = self.data_cache[symbol].loc[current_date, 'close']
            else:
                current_price = position['entry_price']
            total_value += position['shares'] * current_price
        
        return {
            'date': current_date,
            'portfolio': portfolio,
            'value': total_value,
            'positions_count': len(portfolio['positions'])
        }
    
    def run_backtest(self, symbols: list, timeframe: str, start_date: str = None, end_date: str = None):
        """Run parallel backtest on multiple symbols"""
        logger.info(f"Starting parallel ML backtest for {len(symbols)} symbols on {timeframe}")
        logger.info(f"Using {self.num_workers} CPU cores")
        
        # Load ML models
        success = self.ml_system.load_models(timeframe, timestamp=None)
        if not success:
            logger.error("Failed to load ML models")
            return None
            
        self.signal_generator.ml_system = self.ml_system
        self.signal_generator.min_confidence = 0.4  # 40% minimum confidence
        
        # Preload all data
        self.preload_all_data(symbols, timeframe, start_date, end_date)
        
        if not self.data_cache:
            logger.error("No data loaded")
            return None
        
        # Get date range
        sample_data = list(self.data_cache.values())[0]
        dates = sample_data.index[200:]  # Skip first 200 for indicators
        
        logger.info(f"Backtesting from {dates[0]} to {dates[-1]} ({len(dates)} days)")
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'history': [],
            'trades': []
        }
        
        # Process each day sequentially (portfolio state must be maintained)
        for i, current_date in enumerate(dates):
            if i % 20 == 0:  # Progress update every 20 days
                logger.info(f"Processing day {i+1}/{len(dates)} - {current_date}")
            
            # Process single day
            result = self.process_single_day((
                current_date,
                {
                    'cash': portfolio['cash'],
                    'positions': portfolio['positions'].copy()
                },
                symbols,
                timeframe
            ))
            
            # Update portfolio
            portfolio['cash'] = result['portfolio']['cash']
            portfolio['positions'] = result['portfolio']['positions']
            portfolio['trades'].extend(result['portfolio']['trades'])
            
            # Record history
            portfolio['history'].append({
                'date': current_date,
                'value': result['value'],
                'cash': portfolio['cash'],
                'positions': result['positions_count']
            })
        
        # Close remaining positions
        logger.info("\nClosing remaining positions...")
        for symbol, position in portfolio['positions'].items():
            if symbol not in self.data_cache:
                continue
                
            final_price = self.data_cache[symbol].iloc[-1]['close']
            
            shares = position['shares']
            entry_cost = position['cost']
            exit_value = shares * final_price * (1 - self.commission_rate)
            pnl = exit_value - entry_cost
            pnl_pct = pnl / entry_cost
            
            trade = {
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'exit_date': dates[-1],
                'entry_price': position['entry_price'],
                'exit_price': final_price,
                'shares': shares,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'confidence': position['confidence'],
                'exit_reason': 'END_OF_BACKTEST',
                'days_held': (dates[-1] - position['entry_date']).days,
                'highest_price': position.get('highest_price', position['entry_price'])
            }
            portfolio['trades'].append(trade)
            
            logger.info(f"CLOSE {symbol}: {shares} shares @ {final_price:.2f} | PnL: {pnl:.2f} ({pnl_pct:.1%})")
        
        # Calculate metrics (reuse from original)
        from ml_models.run_ml_backtest import MLBacktester
        original_backtester = MLBacktester()
        return original_backtester.calculate_metrics(portfolio, dates)


def main():
    """Run parallel ML backtest"""
    # Initialize backtester
    backtester = MLBacktesterParallel(initial_capital=100000)
    
    # Select symbols to test
    test_symbols = ASSETS  # Test with all 59 symbols
    
    # Run backtest
    logger.info("Starting Parallel ML Backtest...")
    start_time = datetime.now()
    
    results = backtester.run_backtest(
        symbols=test_symbols,
        timeframe='1d',
        start_date='2024-01-01',
        end_date='2025-06-26'
    )
    
    end_time = datetime.now()
    
    if results:
        logger.info(f"Backtest completed in {(end_time - start_time).total_seconds():.1f} seconds")
        logger.info("Backtest completed successfully!")
    else:
        logger.error("Backtest failed!")


if __name__ == "__main__":
    main()