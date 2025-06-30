"""
Backtest Engine for Hybrid Trading System
Historical performance testing with realistic conditions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import argparse
from loguru import logger
import sys

# Add parent path
sys.path.append(str(Path(__file__).parent.parent))

from core.csv_data_manager import CSVDataManager
from indicators.indicator_calculator import IndicatorCalculator
from core.portfolio_manager import PortfolioManager


class BacktestEngine:
    """Backtest engine with realistic trading conditions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.csv_manager = CSVDataManager()
        self.indicator_calc = IndicatorCalculator()
        
        # Backtest parameters
        self.initial_capital = config.get('initial_capital', 100000)
        self.commission = config.get('commission', 0.0002)  # 0.02%
        self.slippage = config.get('slippage', 0.001)  # 0.1%
        self.min_position_size = 5000  # Min 5000 TRY per position
        
        # Results storage
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.daily_returns = []
        
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str, 
                     strategy: str = 'macd_multi_timeframe'):
        """Run backtest for given symbols and period"""
        logger.info(f"Starting backtest: {len(symbols)} symbols, {start_date} to {end_date}")
        
        # Convert dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Portfolio state
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'total_value': self.initial_capital
        }
        
        # Process each symbol
        results_by_symbol = {}
        
        for symbol in symbols:
            logger.info(f"Backtesting {symbol}...")
            
            # Get data for all timeframes
            data = self._prepare_data(symbol, start, end)
            if not data:
                logger.warning(f"No data for {symbol}, skipping")
                continue
            
            # Run strategy
            if strategy == 'macd_multi_timeframe':
                symbol_trades = self._backtest_macd_strategy(symbol, data, portfolio)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            results_by_symbol[symbol] = symbol_trades
            self.trades.extend(symbol_trades)
        
        # Calculate final metrics
        metrics = self._calculate_metrics(portfolio)
        
        return {
            'metrics': metrics,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'trades_by_symbol': results_by_symbol
        }
    
    def _prepare_data(self, symbol: str, start: datetime, end: datetime) -> Dict:
        """Prepare multi-timeframe data with indicators"""
        data = {}
        
        # Get data for each timeframe
        for timeframe in ['1h', '4h', '1d']:
            df = self.csv_manager.get_raw_data(symbol, timeframe)
            if df is None:
                continue
            
            # Filter date range
            df = df[(df.index >= start) & (df.index <= end)]
            if len(df) < 50:  # Need enough data for indicators
                continue
            
            # Calculate indicators
            indicators = self.indicator_calc.calculate_all_indicators(symbol, timeframe, save=False)
            
            # Merge with price data
            if not indicators.empty:
                df = pd.concat([df, indicators], axis=1)
            
            data[timeframe] = df
        
        return data
    
    def _backtest_macd_strategy(self, symbol: str, data: Dict, portfolio: Dict) -> List[Dict]:
        """MACD multi-timeframe strategy backtest"""
        trades = []
        position = None
        
        # Use 1h as primary timeframe
        if '1h' not in data:
            return trades
        
        df_1h = data['1h']
        df_4h = data.get('4h')
        df_1d = data.get('1d')
        
        # Iterate through 1h bars
        for i in range(50, len(df_1h)):
            current_bar = df_1h.iloc[i]
            current_date = df_1h.index[i]
            
            # Skip if already in position
            if position is not None:
                # Check exit conditions
                exit_signal = self._check_exit_conditions(position, current_bar)
                
                if exit_signal:
                    # Close position
                    exit_price = current_bar['close'] * (1 - self.slippage)
                    profit = self._calculate_profit(position, exit_price)
                    
                    # Update portfolio
                    portfolio['cash'] += position['quantity'] * exit_price - self.commission * position['quantity'] * exit_price
                    portfolio['positions'].pop(symbol, None)
                    
                    # Record trade
                    trade = {
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'quantity': position['quantity'],
                        'profit': profit,
                        'profit_pct': profit / (position['entry_price'] * position['quantity']) * 100,
                        'exit_reason': exit_signal
                    }
                    trades.append(trade)
                    position = None
                    
                    logger.debug(f"{symbol} closed: {trade['profit_pct']:.2f}% - {exit_signal}")
                
                continue
            
            # Check entry conditions
            entry_signal = self._check_entry_conditions(df_1h, df_4h, df_1d, i)
            
            if entry_signal and portfolio['cash'] > self.min_position_size:
                # Calculate position size (1% risk)
                atr = current_bar.get('atr', current_bar['close'] * 0.02)
                
                # Check for NaN in ATR
                if pd.isna(atr) or atr <= 0:
                    atr = current_bar['close'] * 0.02  # Default 2%
                
                stop_distance = 2 * atr  # 2x ATR stop
                
                risk_amount = portfolio['total_value'] * 0.01  # 1% risk
                position_size = min(risk_amount / (stop_distance / current_bar['close']), portfolio['cash'] * 0.9)
                
                if position_size < self.min_position_size:
                    continue
                
                # Open position
                entry_price = current_bar['close'] * (1 + self.slippage)
                
                # Check for NaN values
                if pd.isna(entry_price) or pd.isna(position_size) or entry_price <= 0:
                    continue
                    
                quantity = int(position_size / entry_price)
                
                position = {
                    'symbol': symbol,
                    'entry_date': current_date,
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'stop_loss': entry_price - stop_distance,
                    'take_profit': entry_price + stop_distance * 2,  # 2:1 R/R
                    'atr': atr
                }
                
                # Update portfolio
                portfolio['cash'] -= quantity * entry_price + self.commission * quantity * entry_price
                portfolio['positions'][symbol] = position
                
                logger.debug(f"{symbol} entry: {quantity} @ {entry_price:.2f}")
            
            # Update portfolio value
            self._update_portfolio_value(portfolio, df_1h.iloc[i])
        
        # Close any remaining position
        if position is not None:
            exit_price = df_1h.iloc[-1]['close']
            profit = self._calculate_profit(position, exit_price)
            
            trade = {
                'symbol': symbol,
                'entry_date': position['entry_date'],
                'exit_date': df_1h.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'quantity': position['quantity'],
                'profit': profit,
                'profit_pct': profit / (position['entry_price'] * position['quantity']) * 100,
                'exit_reason': 'end_of_data'
            }
            trades.append(trade)
        
        return trades
    
    def _check_entry_conditions(self, df_1h: pd.DataFrame, df_4h: Optional[pd.DataFrame], 
                               df_1d: Optional[pd.DataFrame], i: int) -> bool:
        """Check MACD entry conditions across timeframes"""
        # 1H MACD crossover
        if 'macd' not in df_1h.columns or 'macd_signal' not in df_1h.columns:
            return False
        
        macd_current = df_1h['macd'].iloc[i]
        signal_current = df_1h['macd_signal'].iloc[i]
        macd_prev = df_1h['macd'].iloc[i-1]
        signal_prev = df_1h['macd_signal'].iloc[i-1]
        
        # Bullish crossover
        bullish_cross_1h = macd_prev <= signal_prev and macd_current > signal_current
        
        if not bullish_cross_1h:
            return False
        
        # Multi-timeframe confirmation
        confirmations = 1
        
        # 4H confirmation
        if df_4h is not None and 'macd' in df_4h.columns:
            # Find corresponding 4H bar
            current_time = df_1h.index[i]
            mask = df_4h.index <= current_time
            if mask.any():
                idx_4h = mask.sum() - 1
                if idx_4h >= 0 and idx_4h < len(df_4h):
                    if df_4h['macd'].iloc[idx_4h] > df_4h['macd_signal'].iloc[idx_4h]:
                        confirmations += 1
        
        # 1D confirmation
        if df_1d is not None and 'macd' in df_1d.columns:
            current_time = df_1h.index[i]
            mask = df_1d.index <= current_time
            if mask.any():
                idx_1d = mask.sum() - 1
                if idx_1d >= 0 and idx_1d < len(df_1d):
                    if df_1d['macd'].iloc[idx_1d] > df_1d['macd_signal'].iloc[idx_1d]:
                        confirmations += 1
        
        # Need at least 2 timeframe confirmations
        return confirmations >= 2
    
    def _check_exit_conditions(self, position: Dict, current_bar: pd.Series) -> Optional[str]:
        """Check exit conditions for position"""
        current_price = current_bar['close']
        
        # Stop loss
        if current_price <= position['stop_loss']:
            return 'stop_loss'
        
        # Take profit
        if current_price >= position['take_profit']:
            return 'take_profit'
        
        # Trailing stop (if in profit)
        if current_price > position['entry_price'] * 1.02:  # 2% in profit
            trailing_stop = current_price - 2 * position['atr']
            if trailing_stop > position['stop_loss']:
                position['stop_loss'] = trailing_stop
        
        # MACD bearish crossover
        if 'macd' in current_bar and 'macd_signal' in current_bar:
            if current_bar['macd'] < current_bar['macd_signal']:
                return 'macd_bearish'
        
        return None
    
    def _calculate_profit(self, position: Dict, exit_price: float) -> float:
        """Calculate profit for a trade"""
        entry_cost = position['quantity'] * position['entry_price']
        exit_value = position['quantity'] * exit_price
        
        # Include commission
        total_commission = self.commission * (entry_cost + exit_value)
        
        return exit_value - entry_cost - total_commission
    
    def _update_portfolio_value(self, portfolio: Dict, current_prices: pd.Series):
        """Update portfolio total value"""
        total = portfolio['cash']
        
        for symbol, position in portfolio['positions'].items():
            total += position['quantity'] * current_prices['close']
        
        portfolio['total_value'] = total
        self.equity_curve.append(total)
    
    def _calculate_metrics(self, portfolio: Dict) -> Dict:
        """Calculate backtest performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        # Win rate
        winning_trades = [t for t in self.trades if t['profit'] > 0]
        win_rate = len(winning_trades) / len(self.trades) * 100
        
        # Profit factor
        gross_profit = sum(t['profit'] for t in self.trades if t['profit'] > 0)
        gross_loss = abs(sum(t['profit'] for t in self.trades if t['profit'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Total return
        final_value = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Calculate daily returns from equity curve
        equity_series = pd.Series(self.equity_curve)
        daily_returns = equity_series.pct_change().dropna()
        
        # Sharpe ratio (assuming 252 trading days)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100
        
        # Additional metrics
        avg_win = np.mean([t['profit_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['profit_pct'] for t in self.trades if t['profit'] < 0]) if len(self.trades) > len(winning_trades) else 0
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(self.trades) - len(winning_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_capital': final_value
        }


def main():
    """Main backtest function"""
    parser = argparse.ArgumentParser(description='Run backtest for hybrid trading system')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='Symbols to backtest (default: all available symbols)')
    parser.add_argument('--start', default='2025-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2025-06-30', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--output', default='backtest_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Get all available symbols if not specified
    if args.symbols is None:
        csv_manager = CSVDataManager()
        args.symbols = [s for s in csv_manager.get_available_symbols() if not s.endswith('.IS')]
        logger.info(f"Using all {len(args.symbols)} available symbols")
    
    # Configuration
    config = {
        'initial_capital': args.capital,
        'commission': 0.0002,
        'slippage': 0.001
    }
    
    # Run backtest
    engine = BacktestEngine(config)
    
    logger.info("=" * 60)
    logger.info("HYBRID TRADING SYSTEM - BACKTEST")
    logger.info("=" * 60)
    logger.info(f"Symbols: {', '.join(args.symbols)}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Initial Capital: {args.capital:,.0f} TRY")
    
    # Run backtest
    results = engine.run_backtest(args.symbols, args.start, args.end)
    
    # Display results
    metrics = results['metrics']
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Average Win: {metrics['avg_win']:.2f}%")
    print(f"Average Loss: {metrics['avg_loss']:.2f}%")
    print(f"Final Capital: {metrics['final_capital']:,.0f} TRY")
    
    # Monthly returns
    if results['equity_curve']:
        equity_df = pd.DataFrame({'equity': results['equity_curve']})
        equity_df['returns'] = equity_df['equity'].pct_change()
        monthly_return = equity_df['returns'].mean() * 30 * 100  # Approximate
        print(f"Avg Monthly Return: {monthly_return:.1f}%")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {args.output}")
    
    # Trade summary by symbol
    print("\n" + "="*60)
    print("TRADES BY SYMBOL")
    print("="*60)
    
    for symbol, trades in results['trades_by_symbol'].items():
        if trades:
            symbol_wins = [t for t in trades if t['profit'] > 0]
            symbol_wr = len(symbol_wins) / len(trades) * 100 if trades else 0
            total_profit = sum(t['profit'] for t in trades)
            
            print(f"{symbol}: {len(trades)} trades, WR: {symbol_wr:.0f}%, P&L: {total_profit:,.0f} TRY")


if __name__ == "__main__":
    main()