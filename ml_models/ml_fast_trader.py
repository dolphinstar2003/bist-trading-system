#!/usr/bin/env python3
"""
Fast ML Trading System - Enhanced with optimal stop loss/take profit parameters
Targets high returns with stock-specific risk management
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')


class FastMLTrader:
    """Fast ML trading system with optimal risk management"""
    
    def __init__(self, initial_capital: float = 50000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Load optimal trading parameters
        self.load_optimal_parameters()
        
        # Trading parameters
        self.max_positions = 5
        self.commission = 0.002
        
        # Portfolio
        self.positions = {}
        self.transactions = []
        self.portfolio_history = []
        
        # Top liquid stocks only
        self.symbols = ['THYAO', 'ASELS', 'SISE', 'EREGL', 'ARCLK',
                       'SAHOL', 'KCHOL', 'AKBNK', 'GARAN', 'FROTO']
        
    def load_optimal_parameters(self):
        """Load optimal trading parameters from analysis"""
        try:
            # Load trading strategies
            strategies_path = 'data/analysis/trading_strategies.csv'
            if os.path.exists(strategies_path):
                strategies_df = pd.read_csv(strategies_path, index_col=0)
                self.stock_params = strategies_df.to_dict('index')
                logger.info(f"Loaded optimal parameters for {len(self.stock_params)} stocks")
            else:
                # Default parameters if file not found
                self.stock_params = {}
                logger.warning("Optimal parameters not found, using defaults")
                
            # Load advanced parameters for more details
            params_path = 'data/analysis/advanced_trading_parameters.csv'
            if os.path.exists(params_path):
                params_df = pd.read_csv(params_path)
                self.advanced_params = params_df.set_index('symbol').to_dict('index')
            else:
                self.advanced_params = {}
                
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            self.stock_params = {}
            self.advanced_params = {}
    
    def get_stock_parameters(self, symbol: str) -> dict:
        """Get stock-specific trading parameters"""
        if symbol in self.stock_params:
            params = self.stock_params[symbol]
            return {
                'stop_loss': params['stop_loss'] / 100,
                'take_profit': params['take_profit'] / 100,
                'trailing_stop': params['trailing_stop'] / 100,
                'position_size': params['position_size'],
                'risk_reward': params['risk_reward_ratio']
            }
        else:
            # Default parameters
            return {
                'stop_loss': 0.15,
                'take_profit': 0.25,
                'trailing_stop': 0.10,
                'position_size': 0.15,
                'risk_reward': 1.5
            }
        
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load stock data"""
        try:
            path = f"data/raw/{symbol}_1d_raw.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df
        except:
            pass
        return pd.DataFrame()
        
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate simple but effective features"""
        # Returns
        df['return_1'] = df['close'].pct_change()
        df['return_5'] = df['close'].pct_change(5)
        df['return_20'] = df['close'].pct_change(20)
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['sma_ratio'] = df['close'] / df['sma_20']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_sma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_sma + 2 * bb_std
        df['bb_lower'] = bb_sma - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volatility
        df['volatility'] = df['return_1'].rolling(20).std()
        
        # Momentum
        df['momentum'] = df['close'] / df['close'].shift(10) - 1
        
        return df.dropna()
        
    def generate_signals(self, date: pd.Timestamp) -> list:
        """Generate trading signals for all symbols"""
        signals = []
        
        for symbol in self.symbols:
            df = self.load_data(symbol)
            if df.empty or date not in df.index:
                continue
                
            # Get data up to current date
            hist_data = df[df.index <= date].tail(200)
            if len(hist_data) < 100:
                continue
                
            # Calculate features
            features_df = self.calculate_features(hist_data)
            if len(features_df) < 60:
                continue
                
            # Simple ML prediction
            X = features_df[['return_5', 'return_20', 'sma_ratio', 'rsi', 
                           'bb_position', 'volume_ratio', 'momentum']].iloc[:-1]
            y = features_df['return_1'].shift(-1).iloc[:-1]
            
            if len(X) < 50:
                continue
                
            # Train simple model
            model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            scaler = StandardScaler()
            
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)
            
            # Predict
            last_X = X.iloc[[-1]]
            last_X_scaled = scaler.transform(last_X)
            prediction = model.predict(last_X_scaled)[0]
            
            # Feature importances for confidence
            importances = model.feature_importances_
            confidence = np.mean(importances) * 10  # Simple confidence metric
            
            # Current metrics
            current_price = hist_data['close'].iloc[-1]
            current_rsi = features_df['rsi'].iloc[-1]
            current_momentum = features_df['momentum'].iloc[-1]
            
            signals.append({
                'symbol': symbol,
                'prediction': prediction,
                'confidence': min(confidence, 1.0),
                'price': current_price,
                'rsi': current_rsi,
                'momentum': current_momentum,
                'volatility': features_df['volatility'].iloc[-1]
            })
            
        # Sort by expected return
        signals.sort(key=lambda x: x['prediction'] * x['confidence'], reverse=True)
        return signals
        
    def execute_trades(self, signals: list, date: pd.Timestamp):
        """Execute trades based on signals with stock-specific parameters"""
        # Check existing positions
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            current_signal = next((s for s in signals if s['symbol'] == symbol), None)
            
            if current_signal:
                current_price = current_signal['price']
                pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
                
                # Update trailing stop if applicable
                if 'trailing_stop' in pos and pos['trailing_stop'] > 0:
                    if current_price > pos['highest_price']:
                        pos['highest_price'] = current_price
                        pos['stop_price'] = current_price * (1 - pos['trailing_stop'])
                
                # Exit conditions with stock-specific parameters
                should_exit = False
                exit_reason = ""
                
                # Check trailing stop
                if 'stop_price' in pos and current_price <= pos['stop_price']:
                    should_exit = True
                    exit_reason = "trailing_stop"
                # Check fixed stop loss
                elif pnl_pct <= -pos['stop_loss']:
                    should_exit = True
                    exit_reason = "stop_loss"
                # Check take profit
                elif pnl_pct >= pos['take_profit']:
                    should_exit = True
                    exit_reason = "take_profit"
                # Check signal reversal
                elif current_signal['prediction'] < -0.005:
                    should_exit = True
                    exit_reason = "signal_reversal"
                
                if should_exit:
                    self.close_position(symbol, current_price, date, pnl_pct, exit_reason)
                    
        # Open new positions
        if len(self.positions) < self.max_positions:
            for signal in signals:
                if len(self.positions) >= self.max_positions:
                    break
                    
                symbol = signal['symbol']
                if symbol not in self.positions:
                    # Get stock-specific parameters
                    params = self.get_stock_parameters(symbol)
                    
                    # Entry criteria with risk/reward consideration
                    min_prediction = 0.005 * (1 / params['risk_reward'])  # Adjust by RR ratio
                    min_confidence = max(0.4, 1 - params['risk_reward'] / 3)  # Higher RR = lower confidence needed
                    
                    if (signal['prediction'] > min_prediction and 
                        signal['confidence'] > min_confidence):
                        
                        # Additional filters based on volatility
                        if symbol in self.advanced_params:
                            volatility = self.advanced_params[symbol]['annual_volatility']
                            # More selective for high volatility stocks
                            if volatility > 50 and signal['rsi'] > 70:
                                continue
                        
                        # Buy signal with good momentum
                        if signal['rsi'] < 70 and signal['momentum'] > -0.1:
                            self.open_position(symbol, signal['price'], date, params)
                        
    def open_position(self, symbol: str, price: float, date: pd.Timestamp, params: dict = None):
        """Open a new position with stock-specific parameters"""
        if params is None:
            params = self.get_stock_parameters(symbol)
            
        position_size = self.capital * params['position_size']
        shares = int(position_size / price)
        
        if shares > 0:
            cost = shares * price * (1 + self.commission)
            
            self.positions[symbol] = {
                'shares': shares,
                'entry_price': price,
                'entry_date': date,
                'cost': cost,
                'stop_loss': params['stop_loss'],
                'take_profit': params['take_profit'],
                'trailing_stop': params['trailing_stop'],
                'highest_price': price,
                'stop_price': price * (1 - params['stop_loss'])
            }
            
            self.transactions.append({
                'date': date,
                'type': 'BUY',
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'stop_loss': params['stop_loss'] * 100,
                'take_profit': params['take_profit'] * 100,
                'risk_reward': params['risk_reward']
            })
            
            logger.info(f"{date.date()} BUY {shares} {symbol} @ {price:.2f} " +
                       f"(SL: {params['stop_loss']*100:.1f}%, TP: {params['take_profit']*100:.1f}%, RR: {params['risk_reward']:.2f})")
            
    def close_position(self, symbol: str, price: float, date: pd.Timestamp, pnl_pct: float, reason: str = ""):
        """Close a position"""
        if symbol not in self.positions:
            return
            
        pos = self.positions[symbol]
        revenue = pos['shares'] * price * (1 - self.commission)
        profit = revenue - pos['cost']
        
        self.capital += profit
        
        self.transactions.append({
            'date': date,
            'type': 'SELL',
            'symbol': symbol,
            'shares': pos['shares'],
            'price': price,
            'profit': profit,
            'profit_pct': pnl_pct * 100,
            'exit_reason': reason
        })
        
        del self.positions[symbol]
        
        logger.info(f"{date.date()} SELL {symbol} @ {price:.2f} (P&L: {pnl_pct*100:.1f}%, Reason: {reason})")
        
    def calculate_portfolio_value(self, signals: list) -> float:
        """Calculate total portfolio value"""
        positions_value = 0
        
        for symbol, pos in self.positions.items():
            current_price = next((s['price'] for s in signals if s['symbol'] == symbol), 
                               pos['entry_price'])
            positions_value += pos['shares'] * current_price
            
        return self.capital + positions_value
        
    def run_backtest(self, start_date: str = '2024-01-01', end_date: str = '2024-12-31'):
        """Run backtest"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Get trading dates from first symbol
        df_sample = self.load_data(self.symbols[0])
        if df_sample.empty:
            logger.error("No data available")
            return
            
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        trading_dates = df_sample[(df_sample.index >= start) & (df_sample.index <= end)].index
        
        logger.info(f"Testing {len(trading_dates)} trading days")
        
        # Skip first 60 days for model training
        for i, date in enumerate(trading_dates[60:], 60):
            # Generate signals
            signals = self.generate_signals(date)
            
            # Execute trades
            self.execute_trades(signals, date)
            
            # Record portfolio value
            portfolio_value = self.calculate_portfolio_value(signals)
            self.portfolio_history.append({
                'date': date,
                'value': portfolio_value,
                'positions': len(self.positions),
                'capital': self.capital
            })
            
            # Progress update
            if (i - 60 + 1) % 20 == 0:
                pct_return = (portfolio_value - self.initial_capital) / self.initial_capital * 100
                logger.info(f"Day {i-60+1}: Portfolio {portfolio_value:,.0f} ({pct_return:+.1f}%)")
                
    def print_results(self):
        """Print enhanced results with exit reason analysis"""
        if not self.portfolio_history:
            logger.error("No results to display")
            return
            
        df = pd.DataFrame(self.portfolio_history)
        
        initial = self.initial_capital
        final = df['value'].iloc[-1]
        total_return = (final - initial) / initial * 100
        
        # Calculate monthly returns
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_returns = df.groupby('month')['value'].agg(['first', 'last'])
        monthly_returns['return'] = (monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first'] * 100
        
        # Trading stats
        sells = [t for t in self.transactions if t['type'] == 'SELL']
        if sells:
            wins = [t for t in sells if t['profit'] > 0]
            win_rate = len(wins) / len(sells) * 100
            avg_profit = np.mean([t['profit_pct'] for t in sells])
            
            # Exit reason analysis
            exit_reasons = {}
            for sell in sells:
                reason = sell.get('exit_reason', 'unknown')
                if reason not in exit_reasons:
                    exit_reasons[reason] = {'count': 0, 'avg_pnl': [], 'win_count': 0}
                exit_reasons[reason]['count'] += 1
                exit_reasons[reason]['avg_pnl'].append(sell['profit_pct'])
                if sell['profit'] > 0:
                    exit_reasons[reason]['win_count'] += 1
        else:
            win_rate = avg_profit = 0
            exit_reasons = {}
            
        print(f"\n{'='*80}")
        print(f"Enhanced ML Trading Results with Optimal Parameters")
        print(f"{'='*80}")
        print(f"Initial Capital: {initial:,.0f} TL")
        print(f"Final Value: {final:,.0f} TL") 
        print(f"Total Return: {total_return:.2f}%")
        print(f"Total Trades: {len(sells)}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Avg Profit/Trade: {avg_profit:.1f}%")
        
        # Exit reason breakdown
        if exit_reasons:
            print(f"\n{'='*50}")
            print("Exit Reason Analysis:")
            print(f"{'-'*50}")
            print(f"{'Reason':<15} {'Count':>8} {'Win Rate':>10} {'Avg P&L':>10}")
            print(f"{'-'*50}")
            for reason, stats in exit_reasons.items():
                avg_pnl = np.mean(stats['avg_pnl'])
                win_pct = (stats['win_count'] / stats['count'] * 100) if stats['count'] > 0 else 0
                print(f"{reason:<15} {stats['count']:>8} {win_pct:>9.1f}% {avg_pnl:>9.1f}%")
        
        print(f"\n{'='*50}")
        print("Monthly Returns:")
        print(f"{'-'*50}")
        
        for month, ret in monthly_returns['return'].items():
            status = "✓✓" if ret >= 8 else "✓" if ret >= 5 else "✗" if ret >= 0 else "✗✗"
            print(f"{month}: {ret:>6.1f}% {status}")
            
        achieved = sum(1 for r in monthly_returns['return'] if r >= 8)
        print(f"\nMonths ≥8%: {achieved}/{len(monthly_returns)} ({achieved/len(monthly_returns)*100:.0f}%)")
        
        # Average monthly return
        avg_monthly = monthly_returns['return'].mean()
        print(f"Average Monthly: {avg_monthly:.1f}%")
        print(f"Annualized: {avg_monthly * 12:.1f}%")
        print(f"{'='*80}")
        
    def plot_results(self):
        """Plot enhanced results with trade markers"""
        if not self.portfolio_history:
            return
            
        df = pd.DataFrame(self.portfolio_history)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
        
        # Portfolio value
        ax1.plot(df['date'], df['value'], 'b-', linewidth=2, label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Add buy/sell markers
        buys = [t for t in self.transactions if t['type'] == 'BUY']
        sells = [t for t in self.transactions if t['type'] == 'SELL']
        
        if buys:
            buy_dates = [t['date'] for t in buys]
            buy_values = []
            for bd in buy_dates:
                idx = df[df['date'] == bd].index
                if len(idx) > 0:
                    buy_values.append(df.loc[idx[0], 'value'])
                else:
                    buy_values.append(self.initial_capital)
            ax1.scatter(buy_dates, buy_values, color='green', marker='^', s=100, alpha=0.7, label='Buy')
        
        if sells:
            sell_dates = [t['date'] for t in sells]
            sell_values = []
            for sd in sell_dates:
                idx = df[df['date'] == sd].index
                if len(idx) > 0:
                    sell_values.append(df.loc[idx[0], 'value'])
                else:
                    sell_values.append(self.initial_capital)
            ax1.scatter(sell_dates, sell_values, color='red', marker='v', s=100, alpha=0.7, label='Sell')
        
        ax1.fill_between(df['date'], self.initial_capital, df['value'],
                        where=(df['value'] >= self.initial_capital),
                        color='green', alpha=0.2)
        ax1.fill_between(df['date'], self.initial_capital, df['value'],
                        where=(df['value'] < self.initial_capital),
                        color='red', alpha=0.2)
        ax1.set_title('Portfolio Value - Enhanced ML Strategy with Optimal Parameters', fontsize=14)
        ax1.set_ylabel('Value (TL)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Positions
        ax2.bar(df['date'], df['positions'], color='purple', alpha=0.7)
        ax2.set_title('Active Positions')
        ax2.set_ylabel('Number of Positions')
        ax2.set_ylim(0, self.max_positions + 1)
        ax2.axhline(y=self.max_positions, color='orange', linestyle='--', alpha=0.5, label='Max Positions')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Monthly returns bar chart
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly_returns = df.groupby('month')['value'].agg(['first', 'last'])
        monthly_returns['return'] = (monthly_returns['last'] - monthly_returns['first']) / monthly_returns['first'] * 100
        
        months = [str(m) for m in monthly_returns.index]
        returns = monthly_returns['return'].values
        colors = ['green' if r >= 8 else 'orange' if r >= 0 else 'red' for r in returns]
        
        ax3.bar(months, returns, color=colors, alpha=0.7)
        ax3.axhline(y=8, color='green', linestyle='--', alpha=0.5, label='Target 8%')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title('Monthly Returns')
        ax3.set_ylabel('Return (%)')
        ax3.set_xlabel('Month')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Format dates
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('ml_fast_trader_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast ML Trading')
    parser.add_argument('--capital', type=float, default=50000)
    parser.add_argument('--start-date', type=str, default='2024-01-01')
    parser.add_argument('--end-date', type=str, default='2024-12-31')
    
    args = parser.parse_args()
    
    trader = FastMLTrader(initial_capital=args.capital)
    
    try:
        trader.run_backtest(args.start_date, args.end_date)
        trader.print_results()
        trader.plot_results()
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()