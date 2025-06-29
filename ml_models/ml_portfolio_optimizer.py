#!/usr/bin/env python3
"""
ML-Based Portfolio Optimizer
Manages a portfolio of up to 10 stocks using ML predictions
Targets 10% monthly returns with 50,000 TL initial capital
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
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MLPortfolioOptimizer:
    """Portfolio optimizer using ML predictions"""
    
    def __init__(self, initial_capital: float = 50000, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.commission_rate = 0.002  # 0.2% commission
        self.min_position_size = 5000  # Minimum 5000 TL per position
        
        # Portfolio state
        self.portfolio = {}  # {symbol: {'shares': x, 'avg_price': y, 'current_price': z}}
        self.cash = initial_capital
        self.transaction_history = []
        self.portfolio_value_history = []
        
        # ML models for each stock
        self.models = {}
        self.scalers = {}
        self.predictions = {}
        
        # Load BIST stock list from settings
        self.load_stock_list()
        
    def load_stock_list(self):
        """Load BIST stock list from settings.json"""
        settings_path = 'settings.json'
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                # Get symbols from trading.symbols
                self.symbols = settings.get('trading', {}).get('symbols', [])
                logger.info(f"Loaded {len(self.symbols)} symbols from settings")
        else:
            # Default list if settings not found
            self.symbols = ['THYAO', 'ASELS', 'SISE', 'TUPRS', 'EREGL', 
                          'SAHOL', 'KCHOL', 'TOASO', 'AKBNK', 'GARAN']
            logger.warning("Settings.json not found, using default symbols")
            
    def load_stock_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Load historical data for a stock"""
        try:
            raw_path = f"data/raw/{symbol}_1d_raw.csv"
            if not os.path.exists(raw_path):
                logger.warning(f"Data not found for {symbol}")
                return pd.DataFrame()
                
            df = pd.read_csv(raw_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Get last N days
            end_date = df.index[-1]
            start_date = end_date - timedelta(days=days)
            df = df[df.index >= start_date]
            
            return df
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
            
    def create_features(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Create technical features for ML"""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        for i in range(1, lookback + 1):
            features[f'close_lag_{i}'] = df['close'].shift(i)
            features[f'volume_lag_{i}'] = df['volume'].shift(i)
            
        # Returns
        for period in [1, 5, 10, 20]:
            if period <= lookback:
                features[f'return_{period}'] = df['close'].pct_change(period)
                
        # Technical indicators
        for window in [5, 10, 20]:
            if window <= lookback:
                features[f'sma_{window}'] = df['close'].rolling(window).mean()
                features[f'ema_{window}'] = df['close'].ewm(span=window).mean()
                features[f'std_{window}'] = df['close'].rolling(window).std()
                
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Momentum
        features['momentum'] = df['close'] / df['close'].shift(10) - 1
        
        # Volume indicators
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # OBV
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ema'] = obv.ewm(span=20).mean()
        
        return features
        
    def train_model_for_stock(self, symbol: str) -> dict:
        """Train ML model for a single stock"""
        logger.info(f"Training model for {symbol}")
        
        # Load data
        df = self.load_stock_data(symbol)
        if df.empty:
            return None
            
        # Create features
        features = self.create_features(df)
        
        # Target: Next day return (for ranking)
        target = df['close'].pct_change().shift(-1)
        
        # Remove NaN
        valid_idx = features.dropna().index.intersection(target.dropna().index)
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]
        
        if len(X) < 100:
            logger.warning(f"Insufficient data for {symbol}")
            return None
            
        # Split data (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble
        models = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'ridge': Ridge(alpha=1.0, random_state=42)
        }
        
        predictions = []
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            predictions.append(pred)
            
        # Ensemble prediction
        ensemble_pred = np.mean(predictions, axis=0)
        test_mae = mean_absolute_error(y_test, ensemble_pred)
        
        # Make prediction for tomorrow
        if len(X) > 0:
            last_features = X.iloc[[-1]]
            last_scaled = scaler.transform(last_features)
            
            tomorrow_predictions = []
            for model in models.values():
                pred = model.predict(last_scaled)[0]
                tomorrow_predictions.append(pred)
                
            tomorrow_return = np.mean(tomorrow_predictions)
            
            # Calculate confidence (inverse of prediction std)
            pred_std = np.std(tomorrow_predictions)
            confidence = 1 / (1 + pred_std) if pred_std > 0 else 1
            
            return {
                'symbol': symbol,
                'models': models,
                'scaler': scaler,
                'test_mae': test_mae,
                'last_price': df['close'].iloc[-1],
                'predicted_return': tomorrow_return,
                'confidence': confidence,
                'last_date': df.index[-1]
            }
            
        return None
        
    def train_all_models(self):
        """Train models for all stocks in parallel"""
        logger.info(f"Training models for {len(self.symbols)} stocks...")
        
        # Train models in parallel
        with ProcessPoolExecutor(max_workers=8) as executor:
            future_to_symbol = {
                executor.submit(self.train_model_for_stock, symbol): symbol 
                for symbol in self.symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        self.models[symbol] = result['models']
                        self.scalers[symbol] = result['scaler']
                        self.predictions[symbol] = {
                            'predicted_return': result['predicted_return'],
                            'confidence': result['confidence'],
                            'last_price': result['last_price'],
                            'test_mae': result['test_mae']
                        }
                        logger.info(f"{symbol}: Predicted return: {result['predicted_return']*100:.2f}%, "
                                  f"Confidence: {result['confidence']:.2f}")
                except Exception as e:
                    logger.error(f"Error training model for {symbol}: {e}")
                    
    def rank_stocks(self) -> list:
        """Rank stocks by expected return adjusted for confidence"""
        rankings = []
        
        for symbol, pred in self.predictions.items():
            # Skip if already at max position size
            current_position_value = 0
            if symbol in self.portfolio:
                current_position_value = (self.portfolio[symbol]['shares'] * 
                                        self.portfolio[symbol]['current_price'])
                
            # Skip if position is already large
            if current_position_value > self.current_capital * 0.2:  # Max 20% per stock
                continue
                
            # Calculate score: predicted return * confidence
            score = pred['predicted_return'] * pred['confidence']
            
            # Adjust for momentum (favor stocks already performing well)
            if pred['predicted_return'] > 0:
                score *= 1.2
                
            rankings.append({
                'symbol': symbol,
                'score': score,
                'predicted_return': pred['predicted_return'],
                'confidence': pred['confidence'],
                'last_price': pred['last_price'],
                'current_position': current_position_value
            })
            
        # Sort by score
        rankings.sort(key=lambda x: x['score'], reverse=True)
        return rankings
        
    def execute_trades(self, rankings: list, date: datetime):
        """Execute buy/sell trades based on rankings"""
        # Sell positions not in top stocks or with negative predictions
        for symbol in list(self.portfolio.keys()):
            if symbol not in [r['symbol'] for r in rankings[:self.max_positions]]:
                # Sell if not in top stocks
                self.sell_stock(symbol, date, "Not in top stocks")
            elif symbol in self.predictions and self.predictions[symbol]['predicted_return'] < -0.01:
                # Sell if expecting loss > 1%
                self.sell_stock(symbol, date, "Negative prediction")
                
        # Buy top-ranked stocks
        current_positions = len(self.portfolio)
        available_cash = self.cash
        
        for rank in rankings[:self.max_positions]:
            if current_positions >= self.max_positions:
                break
                
            symbol = rank['symbol']
            if symbol not in self.portfolio and rank['predicted_return'] > 0.005:  # Buy if expecting > 0.5% return
                # Calculate position size
                position_size = min(
                    available_cash / (self.max_positions - current_positions),
                    self.current_capital * 0.15,  # Max 15% per new position
                    available_cash * 0.9  # Keep some cash
                )
                
                if position_size >= self.min_position_size:
                    self.buy_stock(symbol, position_size, rank['last_price'], date, 
                                 f"Predicted return: {rank['predicted_return']*100:.2f}%")
                    current_positions += 1
                    available_cash = self.cash
                    
    def buy_stock(self, symbol: str, amount: float, price: float, date: datetime, reason: str):
        """Execute buy order"""
        commission = amount * self.commission_rate
        net_amount = amount - commission
        shares = int(net_amount / price)
        
        if shares > 0:
            total_cost = shares * price + commission
            
            if total_cost <= self.cash:
                self.cash -= total_cost
                
                if symbol in self.portfolio:
                    # Add to existing position
                    old_shares = self.portfolio[symbol]['shares']
                    old_avg = self.portfolio[symbol]['avg_price']
                    new_shares = old_shares + shares
                    new_avg = (old_shares * old_avg + shares * price) / new_shares
                    
                    self.portfolio[symbol]['shares'] = new_shares
                    self.portfolio[symbol]['avg_price'] = new_avg
                else:
                    # New position
                    self.portfolio[symbol] = {
                        'shares': shares,
                        'avg_price': price,
                        'current_price': price
                    }
                    
                self.transaction_history.append({
                    'date': date,
                    'type': 'BUY',
                    'symbol': symbol,
                    'shares': shares,
                    'price': price,
                    'commission': commission,
                    'reason': reason
                })
                
                logger.info(f"BUY {shares} shares of {symbol} at {price:.2f} TL ({reason})")
                
    def sell_stock(self, symbol: str, date: datetime, reason: str):
        """Execute sell order"""
        if symbol not in self.portfolio:
            return
            
        shares = self.portfolio[symbol]['shares']
        price = self.portfolio[symbol]['current_price']
        gross_amount = shares * price
        commission = gross_amount * self.commission_rate
        net_amount = gross_amount - commission
        
        # Calculate profit/loss
        cost_basis = shares * self.portfolio[symbol]['avg_price']
        profit = net_amount - cost_basis
        profit_pct = (profit / cost_basis) * 100
        
        self.cash += net_amount
        del self.portfolio[symbol]
        
        self.transaction_history.append({
            'date': date,
            'type': 'SELL',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'commission': commission,
            'profit': profit,
            'profit_pct': profit_pct,
            'reason': reason
        })
        
        logger.info(f"SELL {shares} shares of {symbol} at {price:.2f} TL "
                   f"(Profit: {profit:.2f} TL / {profit_pct:.2f}%) - {reason}")
                   
    def update_portfolio_prices(self):
        """Update current prices for all holdings"""
        for symbol in self.portfolio:
            if symbol in self.predictions:
                self.portfolio[symbol]['current_price'] = self.predictions[symbol]['last_price']
                
    def calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        stock_value = sum(
            position['shares'] * position['current_price'] 
            for position in self.portfolio.values()
        )
        return self.cash + stock_value
        
    def run_backtest(self, start_date: str = None, end_date: str = None):
        """Run portfolio backtest"""
        logger.info("Starting portfolio backtest...")
        
        # Train initial models
        self.train_all_models()
        
        if not self.predictions:
            logger.error("No models trained successfully")
            return
            
        # Simulate trading for 180 days
        current_date = datetime.now()
        
        for day in range(180):
            logger.info(f"\n--- Day {day + 1} ---")
            
            # Update portfolio prices
            self.update_portfolio_prices()
            
            # Calculate and log portfolio value
            portfolio_value = self.calculate_portfolio_value()
            self.portfolio_value_history.append({
                'date': current_date,
                'value': portfolio_value,
                'cash': self.cash,
                'positions': len(self.portfolio)
            })
            
            self.current_capital = portfolio_value
            
            # Rank stocks
            rankings = self.rank_stocks()
            
            # Execute trades
            self.execute_trades(rankings, current_date)
            
            # Re-train models weekly
            if day % 7 == 6:
                logger.info("Re-training models...")
                self.train_all_models()
                
            current_date += timedelta(days=1)
            
        # Final results
        self.print_results()
        
    def print_results(self):
        """Print backtest results"""
        initial_value = self.initial_capital
        final_value = self.calculate_portfolio_value()
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        print(f"\n{'='*60}")
        print(f"Portfolio Backtest Results")
        print(f"{'='*60}")
        print(f"Initial Capital: {initial_value:,.2f} TL")
        print(f"Final Value: {final_value:,.2f} TL")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Monthly Return (30 days): {total_return:.2f}%")
        
        # Transaction summary
        buys = [t for t in self.transaction_history if t['type'] == 'BUY']
        sells = [t for t in self.transaction_history if t['type'] == 'SELL']
        
        print(f"\nTransactions:")
        print(f"Total Buys: {len(buys)}")
        print(f"Total Sells: {len(sells)}")
        
        if sells:
            profitable_trades = [s for s in sells if s['profit'] > 0]
            avg_profit = np.mean([s['profit_pct'] for s in sells])
            win_rate = len(profitable_trades) / len(sells) * 100
            
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Average Profit per Trade: {avg_profit:.2f}%")
            
        # Current holdings
        print(f"\nCurrent Holdings:")
        print(f"{'Symbol':<10} {'Shares':<10} {'Avg Price':<12} {'Current':<12} {'P&L %':<10}")
        print(f"{'-'*60}")
        
        for symbol, position in self.portfolio.items():
            pnl = ((position['current_price'] - position['avg_price']) / position['avg_price']) * 100
            print(f"{symbol:<10} {position['shares']:<10} {position['avg_price']:<12.2f} "
                  f"{position['current_price']:<12.2f} {pnl:<10.2f}")
                  
        print(f"\nCash Balance: {self.cash:,.2f} TL")
        print(f"{'='*60}")
        
    def plot_results(self, save_path: str = None):
        """Plot portfolio performance"""
        if not self.portfolio_value_history:
            logger.error("No data to plot")
            return
            
        # Create DataFrame
        df = pd.DataFrame(self.portfolio_value_history)
        df.set_index('date', inplace=True)
        
        # Calculate returns
        df['return'] = (df['value'] / self.initial_capital - 1) * 100
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Portfolio value
        ax1.plot(df.index, df['value'], 'b-', linewidth=2)
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5)
        ax1.fill_between(df.index, self.initial_capital, df['value'], 
                        where=(df['value'] >= self.initial_capital), 
                        color='green', alpha=0.3, label='Profit')
        ax1.fill_between(df.index, self.initial_capital, df['value'], 
                        where=(df['value'] < self.initial_capital), 
                        color='red', alpha=0.3, label='Loss')
        ax1.set_title('Portfolio Value Over Time', fontsize=14)
        ax1.set_ylabel('Value (TL)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Returns
        ax2.plot(df.index, df['return'], 'g-', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='10% Target')
        ax2.fill_between(df.index, 0, df['return'], 
                        where=(df['return'] >= 0), 
                        color='green', alpha=0.3)
        ax2.fill_between(df.index, 0, df['return'], 
                        where=(df['return'] < 0), 
                        color='red', alpha=0.3)
        ax2.set_title('Portfolio Returns (%)', fontsize=14)
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Number of positions and cash
        ax3.plot(df.index, df['positions'], 'purple', linewidth=2, label='Positions')
        ax3.set_ylabel('Number of Positions', fontsize=12)
        ax3.set_ylim(0, self.max_positions + 1)
        
        ax3_twin = ax3.twinx()
        ax3_twin.plot(df.index, df['cash'], 'orange', linewidth=2, label='Cash')
        ax3_twin.set_ylabel('Cash (TL)', fontsize=12)
        
        ax3.set_title('Positions and Cash Balance', fontsize=14)
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def save_results(self, output_dir: str = 'data/portfolio'):
        """Save portfolio results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save portfolio history
        df = pd.DataFrame(self.portfolio_value_history)
        df.to_csv(os.path.join(output_dir, 'portfolio_history.csv'), index=False)
        
        # Save transactions
        if self.transaction_history:
            trans_df = pd.DataFrame(self.transaction_history)
            trans_df.to_csv(os.path.join(output_dir, 'transactions.csv'), index=False)
            
        # Save current holdings
        holdings = []
        for symbol, position in self.portfolio.items():
            holdings.append({
                'symbol': symbol,
                'shares': position['shares'],
                'avg_price': position['avg_price'],
                'current_price': position['current_price'],
                'value': position['shares'] * position['current_price'],
                'pnl': (position['current_price'] - position['avg_price']) * position['shares'],
                'pnl_pct': ((position['current_price'] - position['avg_price']) / position['avg_price']) * 100
            })
            
        if holdings:
            holdings_df = pd.DataFrame(holdings)
            holdings_df.to_csv(os.path.join(output_dir, 'current_holdings.csv'), index=False)
            
        logger.info(f"Results saved to {output_dir}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Portfolio Optimizer')
    parser.add_argument('--capital', type=float, default=50000, help='Initial capital')
    parser.add_argument('--max-positions', type=int, default=10, help='Maximum positions')
    parser.add_argument('--save-plot', type=str, help='Path to save plot')
    parser.add_argument('--save-results', action='store_true', help='Save results to files')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = MLPortfolioOptimizer(
        initial_capital=args.capital,
        max_positions=args.max_positions
    )
    
    try:
        # Run backtest
        optimizer.run_backtest()
        
        # Save results if requested
        if args.save_results:
            optimizer.save_results()
            
        # Plot results
        optimizer.plot_results(save_path=args.save_plot)
        
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        raise


if __name__ == "__main__":
    main()