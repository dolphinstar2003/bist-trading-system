#!/usr/bin/env python3
"""
Realistic ML Portfolio Backtest
Tests portfolio strategy on historical data with proper walk-forward analysis
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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import json
import logging
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PortfolioBacktest:
    """Realistic portfolio backtest using historical data"""
    
    def __init__(self, initial_capital: float = 50000, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.commission_rate = 0.002  # 0.2% commission
        self.min_position_size = 5000  # Minimum 5000 TL per position
        
        # Portfolio state
        self.portfolio = {}
        self.cash = initial_capital
        self.portfolio_history = []
        self.transaction_history = []
        self.daily_returns = []
        
        # Load stock list
        self.load_stock_list()
        
        # Cache for stock data
        self.stock_data_cache = {}
        
    def load_stock_list(self):
        """Load BIST stock list from settings.json"""
        settings_path = 'settings.json'
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                self.symbols = settings.get('trading', {}).get('symbols', [])
                # Use top 20 most liquid stocks for faster backtesting
                self.symbols = self.symbols[:20]
                logger.info(f"Using {len(self.symbols)} symbols for backtest")
        else:
            self.symbols = ['THYAO', 'ASELS', 'SISE', 'TUPRS', 'EREGL']
            
    def load_all_stock_data(self) -> Dict[str, pd.DataFrame]:
        """Load historical data for all stocks"""
        logger.info("Loading historical data for all stocks...")
        
        for symbol in self.symbols:
            try:
                raw_path = f"data/raw/{symbol}_1d_raw.csv"
                if os.path.exists(raw_path):
                    df = pd.read_csv(raw_path)
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                    df.sort_index(inplace=True)
                    
                    # Keep only last 2 years
                    if len(df) > 500:
                        df = df.iloc[-500:]
                        
                    self.stock_data_cache[symbol] = df
                    logger.info(f"Loaded {len(df)} days of data for {symbol}")
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
                
        return self.stock_data_cache
        
    def create_features(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Create technical features"""
        features = pd.DataFrame(index=df.index)
        
        # Price momentum
        features['return_1d'] = df['close'].pct_change()
        features['return_5d'] = df['close'].pct_change(5)
        features['return_20d'] = df['close'].pct_change(20)
        
        # Moving averages
        features['sma_ratio'] = df['close'] / df['close'].rolling(20).mean()
        features['ema_ratio'] = df['close'] / df['close'].ewm(span=20).mean()
        
        # Volatility
        features['volatility'] = df['close'].pct_change().rolling(20).std()
        
        # Volume
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Price position
        features['price_position'] = (df['close'] - df['low'].rolling(20).min()) / (
            df['high'].rolling(20).max() - df['low'].rolling(20).min()
        )
        
        return features.dropna()
        
    def train_predict_model(self, train_data: pd.DataFrame, test_date: pd.Timestamp) -> float:
        """Train model and predict next day return"""
        # Create features
        features = self.create_features(train_data)
        if len(features) < 50:
            return 0.0
            
        # Target: Next day return
        target = train_data['close'].pct_change().shift(-1)
        
        # Align features and target
        valid_idx = features.index.intersection(target.dropna().index)
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]
        
        if len(X) < 50:
            return 0.0
            
        # Use all data for training (up to test_date)
        train_mask = X.index < test_date
        if train_mask.sum() < 50:
            return 0.0
            
        X_train = X[train_mask]
        y_train = y[train_mask]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train simple model
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predict for test date
        if test_date in features.index:
            X_test = features.loc[[test_date]]
            X_test_scaled = scaler.transform(X_test)
            prediction = model.predict(X_test_scaled)[0]
            return prediction
        else:
            return 0.0
            
    def calculate_rankings(self, date: pd.Timestamp) -> List[Tuple[str, float]]:
        """Calculate stock rankings for given date"""
        rankings = []
        
        for symbol, df in self.stock_data_cache.items():
            if date not in df.index:
                continue
                
            # Get data up to current date
            historical_data = df[df.index <= date]
            if len(historical_data) < 100:
                continue
                
            # Predict next day return
            predicted_return = self.train_predict_model(historical_data.iloc[:-1], date)
            
            # Add momentum factor
            momentum = historical_data['close'].pct_change(20).iloc[-1]
            
            # Combined score
            score = predicted_return + 0.3 * momentum
            
            rankings.append((symbol, score, df.loc[date, 'close']))
            
        # Sort by score
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
        
    def execute_trades(self, rankings: List[Tuple[str, float, float]], date: pd.Timestamp):
        """Execute trades based on rankings"""
        # Get current portfolio symbols
        current_symbols = set(self.portfolio.keys())
        
        # Get top symbols
        top_symbols = {r[0] for r in rankings[:self.max_positions] if r[1] > 0}
        
        # Sell positions not in top symbols
        for symbol in current_symbols - top_symbols:
            self.sell_position(symbol, date)
            
        # Buy new positions
        if len(self.portfolio) < self.max_positions:
            available_cash = self.cash * 0.95  # Keep 5% cash buffer
            positions_to_buy = min(
                self.max_positions - len(self.portfolio),
                int(available_cash / self.min_position_size)
            )
            
            for symbol, score, price in rankings:
                if positions_to_buy <= 0:
                    break
                    
                if symbol not in self.portfolio and score > 0:
                    position_size = min(
                        available_cash / positions_to_buy,
                        self.cash * 0.15  # Max 15% per position
                    )
                    
                    if position_size >= self.min_position_size:
                        self.buy_position(symbol, position_size, price, date)
                        positions_to_buy -= 1
                        available_cash = self.cash * 0.95
                        
    def buy_position(self, symbol: str, amount: float, price: float, date: pd.Timestamp):
        """Buy a position"""
        commission = amount * self.commission_rate
        shares = int((amount - commission) / price)
        
        if shares > 0:
            total_cost = shares * price + commission
            
            if total_cost <= self.cash:
                self.cash -= total_cost
                self.portfolio[symbol] = {
                    'shares': shares,
                    'entry_price': price,
                    'entry_date': date
                }
                
                self.transaction_history.append({
                    'date': date,
                    'type': 'BUY',
                    'symbol': symbol,
                    'shares': shares,
                    'price': price,
                    'amount': total_cost
                })
                
                logger.info(f"{date.date()} - BUY {shares} {symbol} @ {price:.2f}")
                
    def sell_position(self, symbol: str, date: pd.Timestamp):
        """Sell a position"""
        if symbol not in self.portfolio:
            return
            
        position = self.portfolio[symbol]
        current_price = self.get_price(symbol, date)
        
        if current_price is None:
            return
            
        shares = position['shares']
        gross_amount = shares * current_price
        commission = gross_amount * self.commission_rate
        net_amount = gross_amount - commission
        
        # Calculate return
        entry_cost = shares * position['entry_price']
        profit = net_amount - entry_cost
        profit_pct = (profit / entry_cost) * 100
        
        self.cash += net_amount
        del self.portfolio[symbol]
        
        self.transaction_history.append({
            'date': date,
            'type': 'SELL',
            'symbol': symbol,
            'shares': shares,
            'price': current_price,
            'amount': net_amount,
            'profit': profit,
            'profit_pct': profit_pct,
            'holding_days': (date - position['entry_date']).days
        })
        
        logger.info(f"{date.date()} - SELL {shares} {symbol} @ {current_price:.2f} "
                   f"(Profit: {profit_pct:.2f}%)")
                   
    def get_price(self, symbol: str, date: pd.Timestamp) -> float:
        """Get price for symbol on date"""
        if symbol in self.stock_data_cache:
            df = self.stock_data_cache[symbol]
            if date in df.index:
                return df.loc[date, 'close']
        return None
        
    def calculate_portfolio_value(self, date: pd.Timestamp) -> float:
        """Calculate total portfolio value"""
        stock_value = 0
        for symbol, position in self.portfolio.items():
            current_price = self.get_price(symbol, date)
            if current_price:
                stock_value += position['shares'] * current_price
                
        return self.cash + stock_value
        
    def run_backtest(self, start_date: str = '2024-01-01', end_date: str = '2024-12-31'):
        """Run portfolio backtest"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Load all stock data
        self.load_all_stock_data()
        
        if not self.stock_data_cache:
            logger.error("No stock data loaded")
            return
            
        # Get date range
        all_dates = []
        for df in self.stock_data_cache.values():
            all_dates.extend(df.index.tolist())
        all_dates = sorted(list(set(all_dates)))
        
        # Filter date range
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        trading_dates = [d for d in all_dates if start <= d <= end]
        
        logger.info(f"Backtesting {len(trading_dates)} trading days")
        
        # Run backtest
        for i, date in enumerate(trading_dates):
            # Skip first 100 days (need history for training)
            if i < 100:
                continue
                
            # Calculate rankings
            rankings = self.calculate_rankings(date)
            
            # Execute trades
            self.execute_trades(rankings, date)
            
            # Record portfolio value
            portfolio_value = self.calculate_portfolio_value(date)
            self.portfolio_history.append({
                'date': date,
                'value': portfolio_value,
                'cash': self.cash,
                'positions': len(self.portfolio)
            })
            
            # Calculate daily return
            if len(self.portfolio_history) > 1:
                prev_value = self.portfolio_history[-2]['value']
                daily_return = (portfolio_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
                
            # Log progress
            if (i + 1) % 20 == 0:
                logger.info(f"Completed {i + 1}/{len(trading_dates)} days - "
                           f"Portfolio Value: {portfolio_value:,.2f} TL")
                           
    def calculate_metrics(self) -> dict:
        """Calculate performance metrics"""
        if not self.portfolio_history:
            return {}
            
        initial_value = self.initial_capital
        final_value = self.portfolio_history[-1]['value']
        
        # Returns
        total_return = (final_value - initial_value) / initial_value
        
        # Convert to DataFrame for easier calculation
        df = pd.DataFrame(self.portfolio_history)
        df['returns'] = df['value'].pct_change()
        
        # Annualized metrics
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        annual_factor = 365 / days if days > 0 else 1
        
        annual_return = (1 + total_return) ** annual_factor - 1
        annual_volatility = df['returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Drawdown
        df['cummax'] = df['value'].cummax()
        df['drawdown'] = (df['value'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()
        
        # Win rate
        trades = [t for t in self.transaction_history if t['type'] == 'SELL']
        if trades:
            profitable_trades = [t for t in trades if t['profit'] > 0]
            win_rate = len(profitable_trades) / len(trades)
            avg_profit = np.mean([t['profit_pct'] for t in trades])
            avg_holding_days = np.mean([t['holding_days'] for t in trades])
        else:
            win_rate = 0
            avg_profit = 0
            avg_holding_days = 0
            
        return {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'annual_volatility': annual_volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'total_trades': len(trades),
            'win_rate': win_rate * 100,
            'avg_profit_per_trade': avg_profit,
            'avg_holding_days': avg_holding_days
        }
        
    def print_results(self):
        """Print backtest results"""
        metrics = self.calculate_metrics()
        
        print(f"\n{'='*60}")
        print(f"Portfolio Backtest Results")
        print(f"{'='*60}")
        print(f"Initial Capital: {metrics['initial_capital']:,.2f} TL")
        print(f"Final Value: {metrics['final_value']:,.2f} TL")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Annual Return: {metrics['annual_return']:.2f}%")
        print(f"Annual Volatility: {metrics['annual_volatility']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"\nTrading Statistics:")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Avg Profit per Trade: {metrics['avg_profit_per_trade']:.2f}%")
        print(f"Avg Holding Days: {metrics['avg_holding_days']:.0f}")
        
        # Monthly returns
        if self.portfolio_history:
            df = pd.DataFrame(self.portfolio_history)
            df['month'] = df['date'].dt.to_period('M')
            monthly_returns = df.groupby('month').apply(
                lambda x: (x['value'].iloc[-1] - x['value'].iloc[0]) / x['value'].iloc[0] * 100
            )
            
            print(f"\nMonthly Returns:")
            print(f"{'-'*30}")
            for month, ret in monthly_returns.items():
                print(f"{month}: {ret:>6.2f}%")
                
        print(f"{'='*60}")
        
    def plot_results(self, save_path: str = None):
        """Plot backtest results"""
        if not self.portfolio_history:
            logger.error("No data to plot")
            return
            
        df = pd.DataFrame(self.portfolio_history)
        df['returns'] = df['value'].pct_change().fillna(0)
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Portfolio value
        ax1 = axes[0]
        ax1.plot(df['date'], df['value'], 'b-', linewidth=2)
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5)
        ax1.fill_between(df['date'], self.initial_capital, df['value'],
                        where=(df['value'] >= self.initial_capital),
                        color='green', alpha=0.3)
        ax1.fill_between(df['date'], self.initial_capital, df['value'],
                        where=(df['value'] < self.initial_capital),
                        color='red', alpha=0.3)
        ax1.set_title('Portfolio Value', fontsize=14)
        ax1.set_ylabel('Value (TL)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Cumulative returns
        ax2 = axes[1]
        ax2.plot(df['date'], df['cumulative_returns'] * 100, 'g-', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Cumulative Returns', fontsize=14)
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Drawdown
        ax3 = axes[2]
        df['cummax'] = df['value'].cummax()
        df['drawdown'] = (df['value'] - df['cummax']) / df['cummax'] * 100
        ax3.fill_between(df['date'], 0, df['drawdown'], color='red', alpha=0.5)
        ax3.plot(df['date'], df['drawdown'], 'r-', linewidth=1)
        ax3.set_title('Drawdown', fontsize=14)
        ax3.set_ylabel('Drawdown (%)', fontsize=12)
        ax3.set_xlabel('Date', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Portfolio Backtest')
    parser.add_argument('--capital', type=float, default=50000, help='Initial capital')
    parser.add_argument('--max-positions', type=int, default=10, help='Maximum positions')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--save-plot', type=str, help='Path to save plot')
    
    args = parser.parse_args()
    
    # Create backtest
    backtest = PortfolioBacktest(
        initial_capital=args.capital,
        max_positions=args.max_positions
    )
    
    try:
        # Run backtest
        backtest.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Print results
        backtest.print_results()
        
        # Plot results
        backtest.plot_results(save_path=args.save_plot)
        
        # Save transaction history
        if backtest.transaction_history:
            trans_df = pd.DataFrame(backtest.transaction_history)
            trans_df.to_csv('data/portfolio/backtest_transactions.csv', index=False)
            logger.info("Transaction history saved")
            
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        raise


if __name__ == "__main__":
    main()