#!/usr/bin/env python3
"""
Enhanced EMA Cross Strategy with Multiple Confirmations
Includes: Volume confirmation, ADX filter, Multi-timeframe analysis, ML overlay, ATR-based stops
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logger to be less verbose
logger.remove()
logger.add(sys.stderr, level="WARNING")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS
from ml_models.ml_trading_system_fixed import MLTradingSystem
from ml_models.ml_signal_generator import MLSignalGenerator


class EMAEnhancedStrategy:
    """Enhanced EMA Cross Strategy with multiple confirmations"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.csv_manager = CSVDataManager()
        self.initial_capital = initial_capital
        self.commission_rate = 0.002  # %0.2 BIST commission
        
        # ML System (optional)
        self.ml_system = None
        self.ml_signal_generator = None
        self.use_ml = False
        
        # Strategy parameters
        self.volume_multiplier = 1.2  # Volume must be 1.2x average (lowered from 1.5)
        self.adx_threshold = 20  # Minimum ADX for trend strength (lowered from 25)
        self.atr_multiplier = 2.0  # ATR multiplier for stop loss
        self.ml_confidence_threshold = 0.3  # ML confidence threshold (lowered from 0.4)
        self.min_confirmations = 1  # Minimum confirmations needed (lowered from 2)
        self.use_partial_position = False  # Use partial position sizing based on confirmations (DISABLED)
        self.partial_size_ratio = 0.5  # Start with 50% position if min confirmations
        
    def load_ml_system(self, timeframe: str = '1d'):
        """Load ML system for probability overlay"""
        try:
            self.ml_system = MLTradingSystem()
            self.ml_signal_generator = MLSignalGenerator()
            
            success = self.ml_system.load_models(timeframe, timestamp=None)
            if success:
                self.ml_signal_generator.ml_system = self.ml_system
                self.ml_signal_generator.min_confidence = self.ml_confidence_threshold
                self.use_ml = True
            else:
                logger.warning("Could not load ML models, continuing without ML overlay")
        except Exception as e:
            logger.warning(f"ML system loading failed: {e}, continuing without ML")
            self.use_ml = False
    
    def calculate_ema(self, data: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        # Calculate directional movement
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        # Calculate True Range
        tr = self.calculate_atr(high, low, close, 1)
        
        # Calculate directional indicators
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr.rolling(window=period).mean())
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def get_ml_probability(self, symbol: str, timeframe: str) -> float:
        """Get ML model probability for current position"""
        if not self.use_ml:
            return 1.0  # Neutral probability if ML not available
        
        try:
            signal = self.ml_signal_generator.generate_signal(symbol, timeframe)
            if signal and signal.signal != 0:
                # Convert confidence to probability modifier
                # High confidence in same direction = boost (up to 1.5x)
                # High confidence in opposite direction = reduce (down to 0.5x)
                return 1.0 + (signal.confidence - 0.5) * signal.signal
            return 1.0
        except:
            return 1.0
    
    def load_multi_timeframe_data(self, symbol: str, date_range: pd.DatetimeIndex) -> Dict:
        """Load data for multiple timeframes"""
        data = {}
        
        # Daily data (primary)
        daily = self.csv_manager.load_raw_data(symbol, '1d')
        if daily is not None:
            data['1d'] = daily
        
        # 4-hour data (if available)
        four_hour = self.csv_manager.load_raw_data(symbol, '4h')
        if four_hour is not None:
            data['4h'] = four_hour
        
        # Weekly data (create from daily)
        if '1d' in data:
            weekly = data['1d'].resample('W').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            data['1w'] = weekly
        
        return data
    
    def check_multi_timeframe_alignment(self, data: Dict, fast_period: int, slow_period: int, 
                                      current_date: pd.Timestamp) -> Dict:
        """Check if multiple timeframes are aligned"""
        alignment = {
            'daily_signal': 0,
            'four_hour_trend': 0,
            'weekly_trend': 0,
            'aligned': False
        }
        
        # Daily signal (primary)
        if '1d' in data and current_date in data['1d'].index:
            daily = data['1d'].loc[:current_date]
            if len(daily) >= slow_period:
                ema_fast = self.calculate_ema(daily['close'], fast_period)
                ema_slow = self.calculate_ema(daily['close'], slow_period)
                
                if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                    alignment['daily_signal'] = 1
                else:
                    alignment['daily_signal'] = -1
        
        # 4-hour trend
        if '4h' in data:
            four_hour = data['4h'].loc[:current_date]
            if len(four_hour) >= 50:
                ema_50 = self.calculate_ema(four_hour['close'], 50)
                if four_hour['close'].iloc[-1] > ema_50.iloc[-1]:
                    alignment['four_hour_trend'] = 1
                else:
                    alignment['four_hour_trend'] = -1
        
        # Weekly trend
        if '1w' in data:
            # Find the week containing current_date
            weekly = data['1w']
            weekly_dates = weekly.loc[:current_date]
            if len(weekly_dates) >= 20:
                ema_20 = self.calculate_ema(weekly_dates['close'], 20)
                if weekly_dates['close'].iloc[-1] > ema_20.iloc[-1]:
                    alignment['weekly_trend'] = 1
                else:
                    alignment['weekly_trend'] = -1
        
        # Check alignment
        if alignment['daily_signal'] == 1:
            # For buy signal, we want positive alignment
            if alignment['four_hour_trend'] >= 0 and alignment['weekly_trend'] >= 0:
                alignment['aligned'] = True
        elif alignment['daily_signal'] == -1:
            # For sell signal, alignment not required (exit position)
            alignment['aligned'] = True
        
        return alignment
    
    def backtest_enhanced(self, symbol: str, fast_period: int, slow_period: int,
                         start_date: str = None, end_date: str = None, debug: bool = False) -> Dict:
        """Backtest enhanced EMA strategy with all confirmations"""
        
        try:
            # Load primary data
            data = self.csv_manager.load_raw_data(symbol, '1d')
            if data is None or len(data) < slow_period + 50:
                return self._empty_result(symbol, fast_period, slow_period, "Insufficient data")
            
            # Filter date range
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # Calculate indicators
            data['ema_fast'] = self.calculate_ema(data['close'], fast_period)
            data['ema_slow'] = self.calculate_ema(data['close'], slow_period)
            data['volume_ma'] = data['volume'].rolling(window=20).mean()
            data['volume_pct'] = data['volume'].rolling(window=50).rank(pct=True)  # Volume percentile
            data['atr'] = self.calculate_atr(data['high'], data['low'], data['close'])
            data['adx'] = self.calculate_adx(data['high'], data['low'], data['close'])
            
            # Skip warm-up period
            data = data.iloc[slow_period:]
            
            # Load multi-timeframe data
            mtf_data = self.load_multi_timeframe_data(symbol, data.index)
            
            # Initialize portfolio
            portfolio_value = []
            cash = self.initial_capital
            position = 0
            entry_price = 0
            stop_loss = 0
            highest_price = 0  # For trailing stop
            bars_in_position = 0  # For time-based exit
            trades = []
            
            # Track reasons for trade decisions
            trade_reasons = []
            
            for idx, row in data.iterrows():
                # Calculate current portfolio value
                current_value = cash
                if position > 0:
                    current_value += position * row['close']
                portfolio_value.append(current_value)
                
                # Position management
                if position > 0:
                    bars_in_position += 1
                    
                    # Update highest price for trailing stop
                    if row['high'] > highest_price:
                        highest_price = row['high']
                        # Update trailing stop
                        new_stop = highest_price - (row['atr'] * self.atr_multiplier)
                        stop_loss = max(stop_loss, new_stop)
                    
                    # Check various exit conditions
                    exit_triggered = False
                    exit_reason = ""
                    
                    # 1. Stop loss hit
                    if row['low'] <= stop_loss:
                        exit_price = min(stop_loss, row['close'])
                        exit_triggered = True
                        exit_reason = 'STOP_LOSS'
                    
                    # 2. Time-based exit (50 bars)
                    elif bars_in_position >= 50:
                        exit_price = row['close']
                        exit_triggered = True
                        exit_reason = 'TIME_EXIT'
                    
                    # 3. Profit target reached (4x ATR)
                    elif row['high'] >= entry_price + (row['atr'] * 4):
                        exit_price = row['close']
                        exit_triggered = True
                        exit_reason = 'PROFIT_TARGET'
                    
                    # Execute exit if triggered
                    if exit_triggered:
                        proceeds = position * exit_price * (1 - self.commission_rate)
                        cash += proceeds
                        
                        trades.append({
                            'date': idx,
                            'type': 'SELL',
                            'price': exit_price,
                            'shares': position,
                            'proceeds': proceeds,
                            'reason': exit_reason
                        })
                        
                        if debug:
                            pnl = proceeds - (position * entry_price * (1 + self.commission_rate))
                            logger.debug(f"EXIT {symbol}: price={exit_price:.2f}, reason={exit_reason}, "
                                       f"bars_held={bars_in_position}, PnL={pnl:.2f}")
                        
                        position = 0
                        bars_in_position = 0
                        continue
                
                # EMA crossover signals
                ema_signal = 0
                if row['ema_fast'] > row['ema_slow']:
                    ema_signal = 1
                elif row['ema_fast'] < row['ema_slow']:
                    ema_signal = -1
                
                # Previous bar signal for crossover detection
                if idx > data.index[0]:
                    prev_idx = data.index[data.index.get_loc(idx) - 1]
                    prev_row = data.loc[prev_idx]
                    prev_signal = 1 if prev_row['ema_fast'] > prev_row['ema_slow'] else -1
                else:
                    prev_signal = 0
                
                # Detect crossover
                crossover = ema_signal != prev_signal and prev_signal != 0
                
                # Buy conditions
                if crossover and ema_signal == 1 and position == 0:
                    buy_reasons = []
                    
                    # 1. Volume confirmation (either multiplier OR percentile)
                    volume_confirmed = (row['volume'] > row['volume_ma'] * self.volume_multiplier) or (row['volume_pct'] > 0.7)
                    if volume_confirmed:
                        if row['volume_pct'] > 0.7:
                            buy_reasons.append("high_volume_percentile")
                        else:
                            buy_reasons.append("volume_confirmed")
                    
                    # 2. ADX filter (trend strength)
                    # Also check if ADX is rising (trend getting stronger)
                    adx_confirmed = row['adx'] > self.adx_threshold
                    if idx > data.index[0]:
                        prev_idx = data.index[data.index.get_loc(idx) - 1]
                        adx_rising = row['adx'] > data.loc[prev_idx, 'adx']
                    else:
                        adx_rising = False
                    
                    # Accept if ADX above threshold OR if ADX is rising above 15
                    if adx_confirmed or (row['adx'] > 15 and adx_rising):
                        buy_reasons.append("strong_trend")
                        adx_confirmed = True
                    
                    # 3. Multi-timeframe alignment
                    mtf_alignment = self.check_multi_timeframe_alignment(
                        mtf_data, fast_period, slow_period, idx
                    )
                    if mtf_alignment['aligned']:
                        buy_reasons.append("mtf_aligned")
                    
                    # 4. ML probability overlay
                    ml_prob = self.get_ml_probability(symbol, '1d')
                    if ml_prob > 1.0:
                        buy_reasons.append(f"ml_positive_{ml_prob:.2f}")
                    
                    # Decide based on confirmations
                    confirmations = sum([
                        volume_confirmed,
                        adx_confirmed,
                        mtf_alignment['aligned'],
                        ml_prob > 1.0
                    ])
                    
                    # Adaptive confirmation requirement based on volatility
                    volatility_ratio = row['atr'] / row['close']
                    
                    # High volatility: need more confirmations
                    if volatility_ratio > 0.03:  # 3% volatility
                        required_confirmations = 2
                    elif volatility_ratio > 0.02:  # 2% volatility
                        required_confirmations = self.min_confirmations
                    else:  # Low volatility
                        required_confirmations = 1
                    
                    # Check if we have enough confirmations
                    if confirmations >= required_confirmations:
                        # Calculate position size with partial sizing option
                        base_shares = int(cash / (row['close'] * (1 + self.commission_rate)))
                        
                        # Adjust size based on confirmation strength
                        if self.use_partial_position:
                            if confirmations == required_confirmations:
                                # Minimum confirmations = partial position
                                shares = int(base_shares * self.partial_size_ratio)
                            elif confirmations >= 3:
                                # Strong confirmations = full position
                                shares = base_shares
                            else:
                                # Medium confirmations = 75% position
                                shares = int(base_shares * 0.75)
                        else:
                            shares = base_shares
                        
                        if shares > 0:
                            cost = shares * row['close'] * (1 + self.commission_rate)
                            if cost <= cash:
                                cash -= cost
                                position = shares
                                entry_price = row['close']
                                
                                # ATR-based stop loss
                                stop_loss = entry_price - (row['atr'] * self.atr_multiplier)
                                highest_price = entry_price
                                bars_in_position = 0
                                
                                trades.append({
                                    'date': idx,
                                    'type': 'BUY',
                                    'price': row['close'],
                                    'shares': shares,
                                    'cost': cost,
                                    'reason': ','.join(buy_reasons),
                                    'stop_loss': stop_loss
                                })
                                
                                if debug:
                                    logger.debug(f"BUY {symbol}: price={row['close']:.2f}, "
                                               f"confirmations={confirmations}, reasons={buy_reasons}")
                
                # Sell conditions (EMA crossover down)
                elif crossover and ema_signal == -1 and position > 0:
                    proceeds = position * row['close'] * (1 - self.commission_rate)
                    cash += proceeds
                    
                    trades.append({
                        'date': idx,
                        'type': 'SELL',
                        'price': row['close'],
                        'shares': position,
                        'proceeds': proceeds,
                        'reason': 'EMA_CROSS_DOWN'
                    })
                    position = 0
            
            # Close final position
            if position > 0:
                final_price = data.iloc[-1]['close']
                proceeds = position * final_price * (1 - self.commission_rate)
                cash += proceeds
                trades.append({
                    'date': data.index[-1],
                    'type': 'SELL',
                    'price': final_price,
                    'shares': position,
                    'proceeds': proceeds,
                    'reason': 'END_BACKTEST'
                })
            
            # Calculate metrics
            final_value = cash
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            # Sharpe ratio
            portfolio_series = pd.Series(portfolio_value, index=data.index)
            daily_returns = portfolio_series.pct_change().dropna()
            sharpe_ratio = 0.0
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            
            # Win rate and trade analysis
            buy_trades = [t for t in trades if t['type'] == 'BUY']
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            
            wins = 0
            losses = 0
            total_pnl = 0
            
            for i, buy in enumerate(buy_trades):
                if i < len(sell_trades):
                    sell = sell_trades[i]
                    pnl = sell['proceeds'] - buy['cost']
                    total_pnl += pnl
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
            
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            
            # Analyze trade reasons
            reason_stats = {}
            for trade in trades:
                if trade['type'] == 'BUY' and 'reason' in trade:
                    reasons = trade['reason'].split(',')
                    for reason in reasons:
                        reason_stats[reason] = reason_stats.get(reason, 0) + 1
            
            return {
                'symbol': symbol,
                'fast_ema': fast_period,
                'slow_ema': slow_period,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'num_trades': len(trades),
                'win_rate': win_rate,
                'wins': wins,
                'losses': losses,
                'final_value': final_value,
                'max_drawdown': self._calculate_max_drawdown(portfolio_series),
                'trade_reasons': reason_stats,
                'trades': trades
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced backtest for {symbol}: {e}")
            return self._empty_result(symbol, fast_period, slow_period, str(e))
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = portfolio_values / portfolio_values.iloc[0]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _empty_result(self, symbol: str, fast: int, slow: int, error: str) -> Dict:
        """Return empty result structure"""
        return {
            'symbol': symbol,
            'fast_ema': fast,
            'slow_ema': slow,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'num_trades': 0,
            'win_rate': 0.0,
            'wins': 0,
            'losses': 0,
            'final_value': self.initial_capital,
            'max_drawdown': 0.0,
            'trade_reasons': {},
            'error': error
        }
    
    def compare_strategies(self, symbol: str, fast_period: int, slow_period: int,
                          start_date: str = None, end_date: str = None) -> Dict:
        """Compare simple vs enhanced strategy"""
        
        # Run simple EMA cross (no filters)
        simple_result = self.backtest_simple_ema(symbol, fast_period, slow_period, start_date, end_date)
        
        # Run enhanced strategy
        enhanced_result = self.backtest_enhanced(symbol, fast_period, slow_period, start_date, end_date)
        
        comparison = {
            'symbol': symbol,
            'parameters': f"EMA {fast_period}/{slow_period}",
            'simple': {
                'return': simple_result['total_return'],
                'sharpe': simple_result['sharpe_ratio'],
                'trades': simple_result['num_trades'],
                'win_rate': simple_result['win_rate']
            },
            'enhanced': {
                'return': enhanced_result['total_return'],
                'sharpe': enhanced_result['sharpe_ratio'],
                'trades': enhanced_result['num_trades'],
                'win_rate': enhanced_result['win_rate'],
                'trade_reasons': enhanced_result['trade_reasons']
            },
            'improvement': {
                'return_diff': enhanced_result['total_return'] - simple_result['total_return'],
                'sharpe_diff': enhanced_result['sharpe_ratio'] - simple_result['sharpe_ratio'],
                'trade_reduction': simple_result['num_trades'] - enhanced_result['num_trades'],
                'win_rate_diff': enhanced_result['win_rate'] - simple_result['win_rate']
            }
        }
        
        return comparison
    
    def backtest_simple_ema(self, symbol: str, fast_period: int, slow_period: int,
                           start_date: str = None, end_date: str = None) -> Dict:
        """Simple EMA crossover without any filters (for comparison)"""
        
        try:
            # Load data
            data = self.csv_manager.load_raw_data(symbol, '1d')
            if data is None or len(data) < slow_period + 50:
                return self._empty_result(symbol, fast_period, slow_period, "Insufficient data")
            
            # Filter date range
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            # Calculate EMAs
            data['ema_fast'] = self.calculate_ema(data['close'], fast_period)
            data['ema_slow'] = self.calculate_ema(data['close'], slow_period)
            
            # Skip warm-up
            data = data.iloc[slow_period:]
            
            # Simple trading logic
            portfolio_value = []
            cash = self.initial_capital
            position = 0
            trades = []
            
            for idx, row in data.iterrows():
                current_value = cash + (position * row['close'] if position > 0 else 0)
                portfolio_value.append(current_value)
                
                # Simple crossover
                if idx > data.index[0]:
                    prev_idx = data.index[data.index.get_loc(idx) - 1]
                    prev_row = data.loc[prev_idx]
                    
                    # Buy signal
                    if prev_row['ema_fast'] <= prev_row['ema_slow'] and row['ema_fast'] > row['ema_slow'] and position == 0:
                        shares = int(cash / (row['close'] * (1 + self.commission_rate)))
                        if shares > 0:
                            cost = shares * row['close'] * (1 + self.commission_rate)
                            cash -= cost
                            position = shares
                            trades.append({'type': 'BUY', 'price': row['close'], 'cost': cost})
                    
                    # Sell signal
                    elif prev_row['ema_fast'] >= prev_row['ema_slow'] and row['ema_fast'] < row['ema_slow'] and position > 0:
                        proceeds = position * row['close'] * (1 - self.commission_rate)
                        cash += proceeds
                        trades.append({'type': 'SELL', 'price': row['close'], 'proceeds': proceeds})
                        position = 0
            
            # Close final position
            if position > 0:
                final_price = data.iloc[-1]['close']
                proceeds = position * final_price * (1 - self.commission_rate)
                cash += proceeds
                trades.append({'type': 'SELL', 'price': final_price, 'proceeds': proceeds})
            
            # Calculate metrics
            final_value = cash
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            # Calculate win rate
            wins = 0
            losses = 0
            for i in range(0, len(trades)-1, 2):
                if i+1 < len(trades):
                    if trades[i+1]['proceeds'] > trades[i]['cost']:
                        wins += 1
                    else:
                        losses += 1
            
            win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
            
            # Sharpe ratio
            portfolio_series = pd.Series(portfolio_value, index=data.index)
            daily_returns = portfolio_series.pct_change().dropna()
            sharpe_ratio = 0.0
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            
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
            return self._empty_result(symbol, fast_period, slow_period, str(e))


def main():
    """Test enhanced EMA strategy"""
    print("Starting Enhanced EMA Strategy Test...")
    print("="*60)
    
    strategy = EMAEnhancedStrategy()
    
    # Load ML system (optional)
    print("Loading ML system...")
    strategy.load_ml_system('1d')
    
    # Test symbols - using the best performers from previous optimization
    test_cases = [
        ('ASELS', 11, 166),  # Best performer
        ('KLRHO', 26, 162),  # Second best
        ('EKGYO', 46, 140),  # Good performer
        ('AKBNK', 9, 97),    # Poor performer for comparison
    ]
    
    results = []
    
    print(f"Testing {len(test_cases)} symbols...")
    
    for symbol, fast, slow in test_cases:
        # Compare simple vs enhanced
        comparison = strategy.compare_strategies(
            symbol, fast, slow,
            start_date='2024-01-01',
            end_date='2025-06-26'
        )
        
        results.append(comparison)
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Results for {symbol} (EMA {fast}/{slow}):")
        print(f"{'='*60}")
        
        print("\nSimple Strategy:")
        print(f"  Return: {comparison['simple']['return']:.1%}")
        print(f"  Sharpe: {comparison['simple']['sharpe']:.2f}")
        print(f"  Trades: {comparison['simple']['trades']}")
        print(f"  Win Rate: {comparison['simple']['win_rate']:.1%}")
        
        print("\nEnhanced Strategy:")
        print(f"  Return: {comparison['enhanced']['return']:.1%}")
        print(f"  Sharpe: {comparison['enhanced']['sharpe']:.2f}")
        print(f"  Trades: {comparison['enhanced']['trades']}")
        print(f"  Win Rate: {comparison['enhanced']['win_rate']:.1%}")
        
        if comparison['enhanced']['trade_reasons']:
            print("\nTrade Confirmations Used:")
            for reason, count in comparison['enhanced']['trade_reasons'].items():
                print(f"    {reason}: {count}")
        
        print("\nImprovement:")
        print(f"  Return: {comparison['improvement']['return_diff']:+.1%}")
        print(f"  Sharpe: {comparison['improvement']['sharpe_diff']:+.2f}")
        print(f"  Trade Reduction: {comparison['improvement']['trade_reduction']}")
        print(f"  Win Rate: {comparison['improvement']['win_rate_diff']:+.1%}")
    
    # Save results
    import pandas as pd
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df.to_json(f'backtest/ema_enhanced_comparison_{timestamp}.json', orient='records', indent=2)
    
    print(f"\nResults saved to backtest/ema_enhanced_comparison_{timestamp}.json")
    print("\nTest completed!")


if __name__ == "__main__":
    main()