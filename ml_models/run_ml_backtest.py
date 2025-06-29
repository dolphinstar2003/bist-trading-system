#!/usr/bin/env python3
"""
ML Trading System Backtest
Test the ML models with historical data
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.ml_trading_system_fixed import MLTradingSystem
from ml_models.ml_signal_generator import MLSignalGenerator
from config.assets import ASSETS
from utils.csv_data_manager import CSVDataManager


class MLBacktester:
    """Backtest ML trading strategies"""
    
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
        
    def run_backtest(self, symbols: list, timeframe: str, start_date: str = None, end_date: str = None):
        """Run backtest on multiple symbols"""
        logger.info(f"Starting ML backtest for {len(symbols)} symbols on {timeframe}")
        
        # Load ML models
        success = self.ml_system.load_models(timeframe, timestamp=None)
        if not success:
            logger.error("Failed to load ML models")
            return None
            
        # Set signal generator ML system
        self.signal_generator.ml_system = self.ml_system
        self.signal_generator.min_confidence = 0.4  # 40% minimum confidence
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'history': [],
            'trades': []
        }
        
        # Get date range from first symbol
        sample_data = self.csv_manager.load_raw_data(symbols[0], timeframe)
        if sample_data is None:
            logger.error("No data available")
            return None
            
        # Filter date range
        if start_date:
            sample_data = sample_data[sample_data.index >= start_date]
        if end_date:
            sample_data = sample_data[sample_data.index <= end_date]
            
        dates = sample_data.index[200:]  # Skip first 200 for indicators
        
        logger.info(f"Backtesting from {dates[0]} to {dates[-1]}")
        
        # Run backtest day by day
        for current_date in dates:
            # Get signals for all symbols
            daily_signals = []
            
            for symbol in symbols:
                # Get data up to current date
                data = self.csv_manager.load_raw_data(symbol, timeframe)
                if data is None:
                    continue
                    
                # Filter to current date
                data = data[data.index <= current_date]
                if len(data) < 200:
                    continue
                    
                # Get current price
                current_price = data.iloc[-1]['close']
                
                # Generate signal
                signal = self.signal_generator.generate_signal(symbol, timeframe)
                if signal and signal.signal != 0:
                    daily_signals.append({
                        'symbol': symbol,
                        'signal': signal.signal,
                        'confidence': signal.confidence,
                        'price': current_price
                    })
            
            # Sort by confidence
            daily_signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Execute trades
            # First, check for exit signals and risk management
            positions_to_close = []
            for symbol, position in portfolio['positions'].items():
                # Get current price for this symbol
                symbol_data = self.csv_manager.load_raw_data(symbol, timeframe)
                if symbol_data is None or current_date not in symbol_data.index:
                    continue
                    
                current_price = symbol_data.loc[current_date, 'close']
                days_held = (current_date - position['entry_date']).days
                
                # Update highest price for trailing stop
                if current_price > position['highest_price']:
                    position['highest_price'] = current_price
                    # Update trailing stop when price reaches near target
                    if current_price >= position['entry_price'] * (1 + self.ml_target_return * 0.8):  # 80% of target
                        # Tighten trailing stop to protect profits
                        position['stop_loss'] = position['highest_price'] * (1 - self.trailing_stop_pct * 0.5)  # %2.5 tight stop
                    else:
                        # Normal trailing stop
                        position['stop_loss'] = max(
                            position['stop_loss'],
                            position['highest_price'] * (1 - self.trailing_stop_pct)
                        )
                
                # Check exit conditions
                exit_reason = None
                
                # 1. ML sell signal
                exit_signal = next((s for s in daily_signals if s['symbol'] == symbol and s['signal'] == -1), None)
                if exit_signal:
                    exit_reason = 'ML_SELL_SIGNAL'
                    
                # 2. Target reached (%2 or more)
                elif current_price >= position['target_price']:
                    exit_reason = 'TARGET_REACHED'
                    
                # 3. Stop loss hit
                elif current_price <= position['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                    
                # 4. Maximum holding period (30 days)
                elif days_held >= self.max_holding_days:
                    exit_reason = 'MAX_HOLDING_PERIOD'
                    
                # 5. Monthly target achieved - if position contributed to %9 monthly return
                elif days_held >= 20:  # At least 20 days
                    position_return = (current_price - position['entry_price']) / position['entry_price']
                    if position_return >= self.target_monthly_return * 0.7:  # 70% of monthly target
                        exit_reason = 'MONTHLY_TARGET'
                
                if exit_reason:
                    positions_to_close.append((symbol, exit_reason))
            
            # Close positions
            for symbol, exit_reason in positions_to_close:
                position = portfolio['positions'].pop(symbol)
                
                # Get exit price
                symbol_data = self.csv_manager.load_raw_data(symbol, timeframe)
                if symbol_data is None or current_date not in symbol_data.index:
                    continue
                exit_price = symbol_data.loc[current_date, 'close']
                
                # Calculate P&L
                shares = position['shares']
                entry_cost = position['cost']
                exit_value = shares * exit_price * (1 - self.commission_rate)
                pnl = exit_value - entry_cost
                pnl_pct = pnl / entry_cost
                
                # Update cash
                portfolio['cash'] += exit_value
                
                # Record trade
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
                
                logger.info(f"SELL {symbol}: {shares} shares @ {exit_price:.2f} | PnL: {pnl:.2f} ({pnl_pct:.1%}) | Reason: {exit_reason}")
            
            # Open new positions (only BUY signals)
            buy_signals = [s for s in daily_signals if s['signal'] == 1]
            
            for signal in buy_signals:
                # Check if we can open more positions
                if len(portfolio['positions']) >= self.max_positions:
                    break
                    
                # Check if already have position
                if signal['symbol'] in portfolio['positions']:
                    continue
                    
                # Calculate position size
                position_value = portfolio['cash'] * self.position_size
                if position_value < 1000:  # Minimum position size
                    continue
                    
                # Calculate shares (round down to 1 lot = 1 share for BIST)
                shares = int(position_value / (signal['price'] * (1 + self.commission_rate)))
                if shares < 1:
                    continue
                    
                # Calculate actual cost
                cost = shares * signal['price'] * (1 + self.commission_rate)
                
                # Check if we have enough cash
                if cost > portfolio['cash']:
                    continue
                    
                # Open position
                portfolio['positions'][signal['symbol']] = {
                    'entry_date': current_date,
                    'entry_price': signal['price'],
                    'shares': shares,
                    'cost': cost,
                    'confidence': signal['confidence'],
                    'highest_price': signal['price'],  # For trailing stop
                    'lowest_price': signal['price'],   # For risk tracking
                    'target_price': signal['price'] * (1 + self.ml_target_return),  # %2 target
                    'stop_loss': signal['price'] * (1 - self.ml_stop_loss)  # %5 stop loss
                }
                
                # Update cash
                portfolio['cash'] -= cost
                
                logger.info(f"BUY {signal['symbol']}: {shares} shares @ {signal['price']:.2f} | Confidence: {signal['confidence']:.1%}")
            
            # Calculate portfolio value
            total_value = portfolio['cash']
            for symbol, position in portfolio['positions'].items():
                # Get current price
                symbol_data = self.csv_manager.load_raw_data(symbol, timeframe)
                if symbol_data is not None and current_date in symbol_data.index:
                    current_price = symbol_data.loc[current_date, 'close']
                else:
                    current_price = position['entry_price']
                total_value += position['shares'] * current_price
            
            # Record history
            portfolio['history'].append({
                'date': current_date,
                'value': total_value,
                'cash': portfolio['cash'],
                'positions': len(portfolio['positions'])
            })
        
        # Close remaining positions at end
        logger.info("\nClosing remaining positions...")
        for symbol, position in portfolio['positions'].items():
            # Get final price
            data = self.csv_manager.load_raw_data(symbol, timeframe)
            if data is None:
                continue
                
            final_price = data.iloc[-1]['close']
            
            # Calculate P&L
            shares = position['shares']
            entry_cost = position['cost']
            exit_value = shares * final_price * (1 - self.commission_rate)
            pnl = exit_value - entry_cost
            pnl_pct = pnl / entry_cost
            
            # Record trade
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
        
        # Calculate final metrics
        return self.calculate_metrics(portfolio, dates)
    
    def calculate_metrics(self, portfolio: dict, dates: pd.DatetimeIndex) -> dict:
        """Calculate backtest metrics"""
        if not portfolio['history']:
            return None
            
        # Convert to DataFrame
        history_df = pd.DataFrame(portfolio['history'])
        trades_df = pd.DataFrame(portfolio['trades']) if portfolio['trades'] else pd.DataFrame()
        
        # Calculate returns
        history_df['returns'] = history_df['value'].pct_change()
        
        # Basic metrics
        initial_value = self.initial_capital
        final_value = history_df['value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Risk metrics
        sharpe_ratio = 0
        if len(history_df['returns'].dropna()) > 0:
            daily_returns = history_df['returns'].dropna()
            if daily_returns.std() > 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        
        # Drawdown
        cumulative_returns = (1 + history_df['returns'].fillna(0)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade statistics
        num_trades = len(trades_df)
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        
        if num_trades > 0:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] < 0]
            
            win_rate = len(winning_trades) / num_trades
            
            if len(winning_trades) > 0:
                avg_win = winning_trades['pnl'].mean()
                
            if len(losing_trades) > 0:
                avg_loss = losing_trades['pnl'].mean()
                
            gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 0
            
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
        
        # Calculate additional metrics
        avg_holding_days = 0
        exit_reasons = {}
        monthly_return = 0
        
        if num_trades > 0:
            # Average holding days
            if 'days_held' in trades_df.columns:
                avg_holding_days = trades_df['days_held'].mean()
            
            # Exit reasons breakdown
            if 'exit_reason' in trades_df.columns:
                exit_reasons = trades_df['exit_reason'].value_counts().to_dict()
            
            # Monthly return estimate
            trading_months = len(dates) / 21  # Approximate trading days per month
            if trading_months > 0:
                monthly_return = (1 + total_return) ** (1 / trading_months) - 1
        
        # Create report
        metrics = {
            'period': {
                'start': str(dates[0]),
                'end': str(dates[-1]),
                'days': len(dates)
            },
            'returns': {
                'total_return': total_return,
                'monthly_return': monthly_return,
                'annualized_return': total_return * 252 / len(dates),
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'trades': {
                'total_trades': num_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'avg_trade_pnl': trades_df['pnl'].mean() if num_trades > 0 else 0,
                'avg_holding_days': avg_holding_days,
                'exit_reasons': exit_reasons
            },
            'capital': {
                'initial': initial_value,
                'final': final_value,
                'max_value': history_df['value'].max(),
                'min_value': history_df['value'].min()
            }
        }
        
        # Save results
        self.save_results(metrics, portfolio, history_df, trades_df)
        
        return metrics
    
    def save_results(self, metrics: dict, portfolio: dict, history_df: pd.DataFrame, trades_df: pd.DataFrame):
        """Save backtest results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save metrics
        with open(f'ml_models/backtest_results_{timestamp}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save trade history
        if not trades_df.empty:
            trades_df.to_csv(f'ml_models/backtest_trades_{timestamp}.csv', index=False)
        
        # Save portfolio history
        history_df.to_csv(f'ml_models/backtest_history_{timestamp}.csv', index=False)
        
        # Print summary
        self.print_summary(metrics)
    
    def print_summary(self, metrics: dict):
        """Print backtest summary"""
        print("\n" + "="*60)
        print("ML BACKTEST RESULTS")
        print("="*60)
        print(f"Period: {metrics['period']['start']} to {metrics['period']['end']}")
        print(f"Days: {metrics['period']['days']}")
        
        print(f"\nRETURNS:")
        print(f"  Total Return: {metrics['returns']['total_return']:.1%}")
        print(f"  Monthly Return: {metrics['returns'].get('monthly_return', 0):.1%}")
        print(f"  Annualized Return: {metrics['returns']['annualized_return']:.1%}")
        print(f"  Sharpe Ratio: {metrics['returns']['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['returns']['max_drawdown']:.1%}")
        
        print(f"\nTRADES:")
        print(f"  Total Trades: {metrics['trades']['total_trades']}")
        print(f"  Win Rate: {metrics['trades']['win_rate']:.1%}")
        print(f"  Average Win: ${metrics['trades']['avg_win']:.2f}")
        print(f"  Average Loss: ${metrics['trades']['avg_loss']:.2f}")
        print(f"  Profit Factor: {metrics['trades']['profit_factor']:.2f}")
        print(f"  Avg Trade PnL: ${metrics['trades']['avg_trade_pnl']:.2f}")
        print(f"  Avg Holding Days: {metrics['trades'].get('avg_holding_days', 0):.1f}")
        
        # Exit reasons breakdown
        if 'exit_reasons' in metrics['trades']:
            print(f"\nEXIT REASONS:")
            for reason, count in metrics['trades']['exit_reasons'].items():
                print(f"  {reason}: {count}")
        
        print(f"\nCAPITAL:")
        print(f"  Initial: ${metrics['capital']['initial']:,.0f}")
        print(f"  Final: ${metrics['capital']['final']:,.0f}")
        print(f"  P&L: ${metrics['capital']['final'] - metrics['capital']['initial']:,.0f}")
        print("="*60)


def main():
    """Run ML backtest"""
    # Initialize backtester
    backtester = MLBacktester(initial_capital=100000)
    
    # Select symbols to test
    test_symbols = ASSETS[:20]  # Test with first 20 symbols
    
    # Run backtest
    logger.info("Starting ML Backtest...")
    results = backtester.run_backtest(
        symbols=test_symbols,
        timeframe='1d',
        start_date='2024-01-01',
        end_date='2025-06-26'
    )
    
    if results:
        logger.info("Backtest completed successfully!")
    else:
        logger.error("Backtest failed!")


if __name__ == "__main__":
    main()