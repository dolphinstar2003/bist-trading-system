#!/usr/bin/env python3
"""
ML Profit Accumulator System
- Fixed 10,000 TL positions
- Take profit at every 10% gain
- Add new positions every 50,000 TL capital milestone
- Focus on highest potential stocks
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
from sklearn.preprocessing import StandardScaler
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


class ProfitAccumulator:
    """Profit accumulation system with dynamic targets based on historical patterns"""
    
    def __init__(self, initial_capital: float = 80000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.total_value = initial_capital
        
        # Fixed parameters
        self.position_size = 8000  # Always 8K TL per position
        self.default_profit_target = 0.10   # Default 10% if no data
        self.capital_milestone = 50000  # New position slot every 50K
        self.commission = 0.002
        
        # Load optimal parameters including dynamic targets
        self.load_optimal_parameters()
        
        # Portfolio state
        self.positions = {}  # Active positions
        self.pending_orders = []  # Orders waiting for capital
        self.profit_history = []  # Track all profit taking events
        self.transaction_history = []
        self.portfolio_history = []
        self.partial_profits = 0  # Accumulated profits from partial sales
        
        # Stock universe - focus on high potential stocks
        self.load_stock_universe()
        
    def load_optimal_parameters(self):
        """Load optimal trading parameters including dynamic targets"""
        try:
            # Load strategies for stop losses
            strategies_path = 'data/analysis/trading_strategies.csv'
            advanced_params_path = 'data/analysis/advanced_trading_parameters.csv'
            
            if os.path.exists(strategies_path):
                strategies_df = pd.read_csv(strategies_path, index_col=0)
                self.stock_params = strategies_df.to_dict('index')
                
                # Sort by risk/reward ratio
                self.sorted_stocks = sorted(
                    [(symbol, params['risk_reward_ratio']) for symbol, params in self.stock_params.items()],
                    key=lambda x: x[1],
                    reverse=True
                )
                logger.info(f"Loaded {len(self.stock_params)} stocks sorted by R/R ratio")
            else:
                self.stock_params = {}
                self.sorted_stocks = []
                
            # Load advanced parameters for dynamic targets
            if os.path.exists(advanced_params_path):
                advanced_df = pd.read_csv(advanced_params_path)
                
                # Create dynamic profit targets based on historical rallies
                self.dynamic_targets = {}
                self.all_stocks_rr = []  # Track all stocks with R/R
                for _, row in advanced_df.iterrows():
                    symbol = row['symbol']
                    
                    # Calculate dynamic profit target based on rally patterns
                    # Use conservative target (50th percentile) for consistent profits
                    conservative_target = row['conservative_take_profit'] / 100
                    optimal_target = row['optimal_take_profit'] / 100
                    
                    # Adjust based on stop loss - wider stop = can aim for higher target
                    stop_loss = row['optimal_stop_loss'] / 100
                    risk_reward = row['risk_reward_ratio']
                    
                    # Dynamic target calculation
                    if risk_reward > 1.5:
                        # High R/R - use optimal target
                        target = min(optimal_target * 0.8, 0.25)  # Cap at 25%
                    elif risk_reward > 1.2:
                        # Medium R/R - between conservative and optimal
                        target = (conservative_target + optimal_target) / 2 * 0.8
                    else:
                        # Low R/R - use conservative target
                        target = min(conservative_target * 0.8, 0.15)  # Cap at 15%
                    
                    # Ensure minimum 8% target
                    target = max(target, 0.08)
                    
                    self.dynamic_targets[symbol] = {
                        'profit_target': target,
                        'stop_loss': stop_loss,
                        'risk_reward': risk_reward,
                        'avg_rally': row['avg_rally'] / 100,
                        'avg_drawdown': row['avg_drawdown'] / 100,
                        'volatility': row['annual_volatility'] / 100
                    }
                    
                    # Track all stocks with their R/R
                    self.all_stocks_rr.append((symbol, risk_reward))
                    
                # Sort all stocks by R/R ratio
                self.all_stocks_rr.sort(key=lambda x: x[1], reverse=True)
                
                logger.info(f"Loaded dynamic targets for {len(self.dynamic_targets)} stocks")
                
                # Show R/R > 1.2 stocks count
                high_rr_count = sum(1 for dt in self.dynamic_targets.values() if dt['risk_reward'] > 1.2)
                logger.info(f"Stocks with R/R > 1.2: {high_rr_count}")
                
                # Show sample targets
                sample_stocks = ['GARAN', 'THYAO', 'ASELS', 'SASA', 'BIMAS']
                for symbol in sample_stocks:
                    if symbol in self.dynamic_targets:
                        dt = self.dynamic_targets[symbol]
                        logger.info(f"{symbol}: Target={dt['profit_target']*100:.1f}%, " +
                                  f"SL={dt['stop_loss']*100:.1f}%, RR={dt['risk_reward']:.2f}")
                                  
                # Show all stocks being traded
                logger.info(f"\nALL TRADEABLE STOCKS ({len(self.dynamic_targets)}):")
                sorted_targets = sorted(self.dynamic_targets.items(), 
                                      key=lambda x: x[1]['risk_reward'], 
                                      reverse=True)
                for i, (symbol, dt) in enumerate(sorted_targets):
                    if i < 10:  # First 10
                        logger.info(f"{i+1}. {symbol}: RR={dt['risk_reward']:.2f}, Target={dt['profit_target']*100:.0f}%")
                if len(sorted_targets) > 10:
                    logger.info(f"... and {len(sorted_targets)-10} more stocks")
                
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            self.stock_params = {}
            self.sorted_stocks = []
            self.dynamic_targets = {}
    
    def load_stock_universe(self):
        """Load all available stocks from analysis"""
        if self.all_stocks_rr:
            # Use all 58 stocks from the analysis
            self.symbols = [stock[0] for stock in self.all_stocks_rr]
            
            logger.info(f"Trading universe: {len(self.symbols)} stocks")
            logger.info(f"Top 10 by R/R: {', '.join(self.symbols[:10])}")
            
            # Show R/R distribution
            rr_ranges = {
                'RR > 1.5': sum(1 for _, rr in self.all_stocks_rr if rr > 1.5),
                'RR 1.2-1.5': sum(1 for _, rr in self.all_stocks_rr if 1.2 <= rr <= 1.5),
                'RR 1.0-1.2': sum(1 for _, rr in self.all_stocks_rr if 1.0 <= rr < 1.2),
                'RR < 1.0': sum(1 for _, rr in self.all_stocks_rr if rr < 1.0)
            }
            
            logger.info("R/R Distribution:")
            for range_name, count in rr_ranges.items():
                logger.info(f"  {range_name}: {count} stocks")
                
            # Show some examples with dynamic targets
            logger.info("\nSample dynamic targets:")
            for i, (symbol, rr) in enumerate(self.all_stocks_rr[:5]):
                if symbol in self.dynamic_targets:
                    dt = self.dynamic_targets[symbol]
                    logger.info(f"  {symbol}: RR={rr:.2f}, Target={dt['profit_target']*100:.0f}%, SL={dt['stop_loss']*100:.0f}%")
        else:
            # Fallback to default stocks
            self.symbols = ['GARAN', 'THYAO', 'ASELS', 'SISE', 'EREGL']
            logger.warning("Using default stock list")
            
    def calculate_max_positions(self) -> int:
        """Calculate maximum positions based on total capital"""
        # Start with 10 positions for 80K
        # Add 1 position for every additional 50K
        base_positions = 10
        additional_capital = max(0, self.total_value - self.initial_capital)
        additional_positions = int(additional_capital / self.capital_milestone)
        
        max_positions = base_positions + additional_positions
        return max_positions
        
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
        """Calculate ML features"""
        # Price momentum
        df['return_1'] = df['close'].pct_change()
        df['return_5'] = df['close'].pct_change(5)
        df['return_10'] = df['close'].pct_change(10)
        df['return_20'] = df['close'].pct_change(20)
        
        # Moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['price_to_sma20'] = df['close'] / df['sma_20']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # Volatility
        df['volatility'] = df['return_1'].rolling(20).std()
        df['atr'] = self.calculate_atr(df)
        
        # Trend strength
        df['trend_strength'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        return df.dropna()
        
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
        
    def generate_ml_signals(self, date: pd.Timestamp) -> Dict[str, float]:
        """Generate ML predictions for all symbols"""
        predictions = {}
        
        for symbol in self.symbols:
            df = self.load_data(symbol)
            if df.empty or date not in df.index:
                continue
                
            # Get historical data up to date
            hist_data = df[df.index <= date].tail(100)
            if len(hist_data) < 60:
                continue
                
            # Calculate features
            features_df = self.calculate_features(hist_data)
            if len(features_df) < 50:
                continue
                
            # Prepare ML data
            feature_cols = ['return_5', 'return_10', 'return_20', 'price_to_sma20', 
                          'rsi', 'volume_ratio', 'volatility', 'trend_strength']
            
            X = features_df[feature_cols].iloc[:-1]
            y = features_df['return_5'].shift(-5).iloc[:-1]  # 5-day forward return
            
            # Remove NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            if len(X) < 30:
                continue
                
            # Train ensemble model
            rf_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            gb_model = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train models
            rf_model.fit(X_scaled, y)
            gb_model.fit(X_scaled, y)
            
            # Make prediction
            last_features = features_df[feature_cols].iloc[-1].values.reshape(1, -1)
            last_scaled = scaler.transform(last_features)
            
            # Ensemble prediction
            rf_pred = rf_model.predict(last_scaled)[0]
            gb_pred = gb_model.predict(last_scaled)[0]
            ensemble_pred = (rf_pred + gb_pred) / 2
            
            # Add risk/reward adjustment
            if symbol in self.stock_params:
                rr_ratio = self.stock_params[symbol]['risk_reward_ratio']
                # Higher R/R ratio = more attractive
                ensemble_pred *= (1 + rr_ratio / 10)
            
            predictions[symbol] = {
                'prediction': ensemble_pred,
                'current_price': hist_data['close'].iloc[-1],
                'rsi': features_df['rsi'].iloc[-1],
                'volume_ratio': features_df['volume_ratio'].iloc[-1],
                'trend': features_df['trend_strength'].iloc[-1]
            }
            
        return predictions
        
    def check_profit_taking(self, symbol: str, current_price: float, date: pd.Timestamp):
        """Check if we should take profits on a position using dynamic targets"""
        if symbol not in self.positions:
            return
            
        pos = self.positions[symbol]
        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
        
        # Get dynamic profit target for this stock
        profit_target = pos.get('profit_target', self.default_profit_target)
        
        # Take profit at dynamic target
        if pnl_pct >= profit_target:
            # Calculate profit
            shares = pos['shares']
            revenue = shares * current_price * (1 - self.commission)
            cost = pos['cost']
            profit = revenue - cost
            
            # Update cash and remove position
            self.cash += revenue
            self.partial_profits += profit
            
            # Record profit taking
            self.profit_history.append({
                'date': date,
                'symbol': symbol,
                'entry_price': pos['entry_price'],
                'exit_price': current_price,
                'shares': shares,
                'profit': profit,
                'profit_pct': pnl_pct * 100,
                'holding_days': (date - pos['entry_date']).days
            })
            
            self.transaction_history.append({
                'date': date,
                'type': 'PROFIT_TAKE',
                'symbol': symbol,
                'shares': shares,
                'price': current_price,
                'profit': profit
            })
            
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"{date.date()} PROFIT TAKE {symbol} @ {current_price:.2f} " +
                       f"(+{pnl_pct*100:.1f}%, Target: {profit_target*100:.1f}%, Profit: {profit:.0f} TL)")
            
    def check_stop_loss(self, symbol: str, current_price: float, date: pd.Timestamp):
        """Check stop loss based on optimal parameters"""
        if symbol not in self.positions:
            return
            
        pos = self.positions[symbol]
        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
        
        # Use stock-specific stop loss
        stop_loss = pos.get('stop_loss', 0.15)
        
        if pnl_pct <= -stop_loss:
            # Calculate loss
            shares = pos['shares']
            revenue = shares * current_price * (1 - self.commission)
            cost = pos['cost']
            loss = revenue - cost
            
            # Update cash
            self.cash += revenue
            
            self.transaction_history.append({
                'date': date,
                'type': 'STOP_LOSS',
                'symbol': symbol,
                'shares': shares,
                'price': current_price,
                'profit': loss
            })
            
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"{date.date()} STOP LOSS {symbol} @ {current_price:.2f} " +
                       f"({pnl_pct*100:.1f}%, Loss: {loss:.0f} TL)")
                       
    def open_position(self, symbol: str, price: float, date: pd.Timestamp):
        """Open a new position with fixed 10K TL and dynamic targets"""
        # Check if we have enough cash
        required_cash = self.position_size * (1 + self.commission)
        if self.cash < required_cash:
            return False
            
        # Check if we're at max positions
        current_positions = len(self.positions)
        max_positions = self.calculate_max_positions()
        
        if current_positions >= max_positions:
            return False
            
        # Calculate shares for position size
        shares = int(self.position_size / price)
        if shares == 0:
            return False
            
        cost = shares * price * (1 + self.commission)
        
        # Get dynamic parameters for this stock
        if symbol in self.dynamic_targets:
            dt = self.dynamic_targets[symbol]
            stop_loss = dt['stop_loss']
            profit_target = dt['profit_target']
            risk_reward = dt['risk_reward']
        else:
            # Fallback to defaults
            stop_loss = 0.15
            profit_target = self.default_profit_target
            risk_reward = 1.0
            if symbol in self.stock_params:
                stop_loss = self.stock_params[symbol]['stop_loss'] / 100
            
        # Create position with dynamic parameters
        self.positions[symbol] = {
            'shares': shares,
            'entry_price': price,
            'entry_date': date,
            'cost': cost,
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'risk_reward': risk_reward
        }
        
        # Update cash
        self.cash -= cost
        
        self.transaction_history.append({
            'date': date,
            'type': 'BUY',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'cost': cost,
            'profit_target': profit_target * 100,
            'stop_loss': stop_loss * 100
        })
        
        logger.info(f"{date.date()} BUY {shares} {symbol} @ {price:.2f} " +
                   f"(Cost: {cost:.0f} TL, Target: {profit_target*100:.1f}%, SL: {stop_loss*100:.1f}%, RR: {risk_reward:.2f})")
        
        return True
        
    def execute_daily_trading(self, date: pd.Timestamp, predictions: Dict):
        """Execute daily trading decisions"""
        # First, check all existing positions for profit taking or stop loss
        for symbol in list(self.positions.keys()):
            if symbol in predictions:
                current_price = predictions[symbol]['current_price']
                self.check_profit_taking(symbol, current_price, date)
                self.check_stop_loss(symbol, current_price, date)
                
        # Calculate portfolio value
        self.calculate_portfolio_value(predictions)
        
        # Check if we can open new positions
        current_positions = len(self.positions)
        max_positions = self.calculate_max_positions()
        available_slots = max_positions - current_positions
        
        if available_slots > 0 and self.cash >= self.position_size * (1 + self.commission):
            # Sort predictions by expected return
            sorted_predictions = sorted(
                [(symbol, data) for symbol, data in predictions.items()],
                key=lambda x: x[1]['prediction'],
                reverse=True
            )
            
            # Try to open positions in best opportunities
            for symbol, data in sorted_predictions:
                if symbol not in self.positions:
                    # Additional filters
                    if (data['prediction'] > 0.01 and  # Positive prediction
                        data['rsi'] < 70 and           # Not overbought
                        data['volume_ratio'] > 0.8 and  # Decent volume
                        data['trend'] > -0.1):          # Not in strong downtrend
                        
                        if self.open_position(symbol, data['current_price'], date):
                            available_slots -= 1
                            if available_slots == 0:
                                break
                                
    def calculate_portfolio_value(self, predictions: Dict):
        """Calculate total portfolio value"""
        positions_value = 0
        
        for symbol, pos in self.positions.items():
            if symbol in predictions:
                current_price = predictions[symbol]['current_price']
                positions_value += pos['shares'] * current_price
            else:
                # Use entry price if no current price
                positions_value += pos['shares'] * pos['entry_price']
                
        self.total_value = self.cash + positions_value + self.partial_profits
        
    def run_backtest(self, start_date: str = '2024-01-01', end_date: str = '2024-12-31'):
        """Run backtest simulation"""
        logger.info(f"Starting Profit Accumulator backtest from {start_date} to {end_date}")
        logger.info(f"Initial capital: {self.initial_capital:,.0f} TL")
        logger.info(f"Position size: {self.position_size:,.0f} TL")
        logger.info(f"Max initial positions: {self.calculate_max_positions()}")
        logger.info(f"Using dynamic profit targets based on historical patterns")
        
        # Get trading dates
        sample_df = self.load_data(self.symbols[0])
        if sample_df.empty:
            logger.error("No data available")
            return
            
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        trading_dates = sample_df[(sample_df.index >= start) & (sample_df.index <= end)].index
        
        logger.info(f"Trading days: {len(trading_dates)}")
        
        # Main backtest loop
        for i, date in enumerate(trading_dates):
            # Skip first 60 days for ML training
            if i < 60:
                continue
                
            # Generate ML predictions
            predictions = self.generate_ml_signals(date)
            
            # Execute trading
            self.execute_daily_trading(date, predictions)
            
            # Record portfolio state
            self.portfolio_history.append({
                'date': date,
                'cash': self.cash,
                'positions_value': self.total_value - self.cash - self.partial_profits,
                'partial_profits': self.partial_profits,
                'total_value': self.total_value,
                'num_positions': len(self.positions),
                'max_positions': self.calculate_max_positions()
            })
            
            # Progress update
            if (i - 60 + 1) % 20 == 0:
                logger.info(f"Day {i-60+1}: Portfolio {self.total_value:,.0f} TL " +
                           f"(+{(self.total_value/self.initial_capital-1)*100:.1f}%), " +
                           f"Positions: {len(self.positions)}/{self.calculate_max_positions()}")
                           
    def print_results(self):
        """Print comprehensive results"""
        if not self.portfolio_history:
            logger.error("No results to display")
            return
            
        df = pd.DataFrame(self.portfolio_history)
        
        initial = self.initial_capital
        final = self.total_value
        total_return = (final - initial) / initial * 100
        
        print(f"\n{'='*80}")
        print(f"PROFIT ACCUMULATOR RESULTS")
        print(f"{'='*80}")
        print(f"Initial Capital: {initial:,.0f} TL")
        print(f"Final Value: {final:,.0f} TL")
        print(f"Total Return: {total_return:.1f}%")
        print(f"Accumulated Profits: {self.partial_profits:,.0f} TL")
        
        # Profit taking analysis
        if self.profit_history:
            profit_df = pd.DataFrame(self.profit_history)
            total_profits = profit_df['profit'].sum()
            avg_holding = profit_df['holding_days'].mean()
            
            print(f"\n{'='*50}")
            print("PROFIT TAKING ANALYSIS:")
            print(f"Total profit takes: {len(profit_df)}")
            print(f"Total profits realized: {total_profits:,.0f} TL")
            print(f"Average profit per take: {profit_df['profit'].mean():,.0f} TL")
            print(f"Average holding period: {avg_holding:.1f} days")
            
            # Top profit takes
            print(f"\n{'='*50}")
            print("TOP 5 PROFIT TAKES:")
            print(f"{'Symbol':<8} {'Entry':>8} {'Exit':>8} {'Days':>6} {'Profit':>10}")
            print(f"{'-'*50}")
            
            top_profits = profit_df.nlargest(5, 'profit')
            for _, row in top_profits.iterrows():
                print(f"{row['symbol']:<8} {row['entry_price']:>8.2f} {row['exit_price']:>8.2f} " +
                      f"{row['holding_days']:>6} {row['profit']:>10,.0f}")
                      
            # Dynamic target analysis
            print(f"\n{'='*50}")
            print("DYNAMIC TARGET PERFORMANCE:")
            print(f"{'-'*50}")
            
            # Group by symbol to show average performance
            symbol_stats = {}
            for _, row in profit_df.iterrows():
                symbol = row['symbol']
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {'count': 0, 'avg_profit_pct': [], 'total_profit': 0}
                symbol_stats[symbol]['count'] += 1
                symbol_stats[symbol]['avg_profit_pct'].append(row['profit_pct'])
                symbol_stats[symbol]['total_profit'] += row['profit']
                
            print(f"{'Symbol':<8} {'Trades':>8} {'Avg %':>8} {'Total Profit':>12}")
            print(f"{'-'*50}")
            for symbol, stats in sorted(symbol_stats.items(), key=lambda x: x[1]['total_profit'], reverse=True)[:10]:
                avg_pct = np.mean(stats['avg_profit_pct'])
                print(f"{symbol:<8} {stats['count']:>8} {avg_pct:>7.1f}% {stats['total_profit']:>12,.0f}")
                      
        # Monthly returns
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly = df.groupby('month')['total_value'].agg(['first', 'last'])
        monthly['return'] = (monthly['last'] - monthly['first']) / monthly['first'] * 100
        
        print(f"\n{'='*50}")
        print("MONTHLY RETURNS:")
        print(f"{'-'*50}")
        
        for month, ret in monthly['return'].items():
            status = "✓✓" if ret >= 8 else "✓" if ret >= 4 else "○"
            print(f"{month}: {ret:>6.1f}% {status}")
            
        avg_monthly = monthly['return'].mean()
        print(f"\nAverage Monthly: {avg_monthly:.1f}%")
        print(f"Annualized: {avg_monthly * 12:.1f}%")
        
        # Position slot utilization
        avg_positions = df['num_positions'].mean()
        avg_max_positions = df['max_positions'].mean()
        utilization = (avg_positions / avg_max_positions * 100) if avg_max_positions > 0 else 0
        
        print(f"\n{'='*50}")
        print("POSITION MANAGEMENT:")
        print(f"Average positions held: {avg_positions:.1f}")
        print(f"Average max positions: {avg_max_positions:.1f}")
        print(f"Slot utilization: {utilization:.1f}%")
        print(f"Final max positions: {self.calculate_max_positions()}")
        
        # Transaction summary
        buys = [t for t in self.transaction_history if t['type'] == 'BUY']
        profit_takes = [t for t in self.transaction_history if t['type'] == 'PROFIT_TAKE']
        stop_losses = [t for t in self.transaction_history if t['type'] == 'STOP_LOSS']
        
        print(f"\n{'='*50}")
        print("TRANSACTION SUMMARY:")
        print(f"Total buys: {len(buys)}")
        print(f"Profit takes: {len(profit_takes)}")
        print(f"Stop losses: {len(stop_losses)}")
        if len(profit_takes) + len(stop_losses) > 0:
            win_rate = len(profit_takes) / (len(profit_takes) + len(stop_losses)) * 100
            print(f"Win rate: {win_rate:.1f}%")
            
        print(f"{'='*80}")
        
    def plot_results(self):
        """Plot comprehensive results"""
        if not self.portfolio_history:
            return
            
        df = pd.DataFrame(self.portfolio_history)
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # 1. Portfolio value with components
        ax1 = axes[0]
        ax1.plot(df['date'], df['total_value'], 'b-', linewidth=2, label='Total Value')
        ax1.plot(df['date'], df['cash'], 'g--', alpha=0.7, label='Cash')
        ax1.plot(df['date'], df['partial_profits'], 'orange', alpha=0.7, label='Accumulated Profits')
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Mark profit taking events
        if self.profit_history:
            profit_dates = [p['date'] for p in self.profit_history]
            profit_values = []
            for pd_date in profit_dates:
                idx = df[df['date'] == pd_date].index
                if len(idx) > 0:
                    profit_values.append(df.loc[idx[0], 'total_value'])
            if profit_values:
                ax1.scatter(profit_dates, profit_values, color='green', marker='$₺$', s=200, 
                           alpha=0.8, label='Profit Take', zorder=5)
        
        ax1.set_title('Portfolio Value Evolution with Profit Accumulation', fontsize=14)
        ax1.set_ylabel('Value (TL)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Position slots utilization
        ax2 = axes[1]
        ax2.fill_between(df['date'], 0, df['max_positions'], alpha=0.3, color='gray', label='Available Slots')
        ax2.bar(df['date'], df['num_positions'], color='purple', alpha=0.7, label='Used Slots')
        ax2.set_title('Position Slot Utilization')
        ax2.set_ylabel('Number of Positions')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Profit accumulation over time
        ax3 = axes[2]
        ax3.plot(df['date'], df['partial_profits'], 'green', linewidth=2)
        ax3.fill_between(df['date'], 0, df['partial_profits'], alpha=0.3, color='green')
        ax3.set_title('Accumulated Profits from 10% Profit Taking')
        ax3.set_ylabel('Accumulated Profits (TL)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly returns
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly = df.groupby('month')['total_value'].agg(['first', 'last'])
        monthly['return'] = (monthly['last'] - monthly['first']) / monthly['first'] * 100
        
        months = [str(m) for m in monthly.index]
        returns = monthly['return'].values
        colors = ['darkgreen' if r >= 8 else 'green' if r >= 4 else 'orange' if r >= 0 else 'red' for r in returns]
        
        ax4 = axes[3]
        bars = ax4.bar(months, returns, color=colors, alpha=0.7)
        ax4.axhline(y=8, color='darkgreen', linestyle='--', alpha=0.5, label='Target 8%')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ret:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        ax4.set_title('Monthly Returns')
        ax4.set_ylabel('Return (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Format x-axis
        for ax in [ax1, ax2, ax3]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle(f'Profit Accumulator Strategy - {self.position_size:,} TL Fixed Positions', fontsize=16)
        plt.tight_layout()
        
        # Save figure
        plt.savefig('profit_accumulator_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Profit Accumulator')
    parser.add_argument('--capital', type=float, default=80000, help='Initial capital')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date')
    
    args = parser.parse_args()
    
    # Create and run system
    accumulator = ProfitAccumulator(initial_capital=args.capital)
    
    try:
        accumulator.run_backtest(args.start_date, args.end_date)
        accumulator.print_results()
        accumulator.plot_results()
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        raise


if __name__ == "__main__":
    main()