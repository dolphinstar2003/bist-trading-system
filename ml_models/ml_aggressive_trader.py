#!/usr/bin/env python3
"""
Aggressive ML Trading System
Targets 8-9% monthly returns (70%+ annually) using advanced strategies
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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import json
import logging
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
# Note: talib requires system dependencies. If not installed, we'll use custom implementations

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AggressiveMLTrader:
    """Aggressive trading system targeting high monthly returns"""
    
    def __init__(self, initial_capital: float = 50000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Aggressive parameters
        self.max_positions = 5  # Concentrate capital
        self.max_position_pct = 0.4  # Up to 40% per position
        self.leverage = 2.0  # 2x leverage (margin trading)
        self.commission_rate = 0.002
        self.min_confidence = 0.65  # Lower threshold for more trades
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.15  # 15% take profit
        
        # Portfolio state
        self.positions = {}  # Active positions
        self.cash = initial_capital
        self.margin_used = 0
        self.transaction_history = []
        self.daily_pnl = []
        self.portfolio_history = []
        
        # ML models
        self.models = {}
        self.predictions_cache = {}
        
        # Load stock list
        self.load_stock_list()
        
        # Technical indicator settings
        self.indicator_params = {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2,
            'atr_period': 14,
            'adx_period': 14,
            'stoch_period': 14,
            'cci_period': 20,
            'mfi_period': 14,
            'willr_period': 14
        }
        
    def load_stock_list(self):
        """Load high volatility BIST stocks"""
        settings_path = 'settings.json'
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                all_symbols = settings.get('trading', {}).get('symbols', [])
                # Focus on most liquid stocks for better execution
                self.symbols = all_symbols[:30]
                logger.info(f"Trading {len(self.symbols)} liquid stocks")
        else:
            self.symbols = ['THYAO', 'ASELS', 'SISE', 'TUPRS', 'EREGL']
            
    def load_stock_data(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Load stock data with given timeframe"""
        try:
            # Try to load hourly data for more signals
            raw_path = f"data/raw/{symbol}_{timeframe}_raw.csv"
            if not os.path.exists(raw_path) and timeframe == '1h':
                # Fallback to daily
                raw_path = f"data/raw/{symbol}_1d_raw.csv"
                
            if os.path.exists(raw_path):
                df = pd.read_csv(raw_path)
                # Check if Date column exists
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)
                elif df.index.name == 'Date':
                    df.index = pd.to_datetime(df.index)
                else:
                    # Try to use the first column as date
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                    df.set_index(df.columns[0], inplace=True)
                    
                df.sort_index(inplace=True)
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
            return pd.DataFrame()
            
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        # Price and volume
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']
        
        # Trend indicators - Simple implementations
        df['sma_10'] = close.rolling(window=10).mean()
        df['sma_20'] = close.rolling(window=20).mean()
        df['sma_50'] = close.rolling(window=50).mean()
        df['ema_10'] = close.ewm(span=10, adjust=False).mean()
        df['ema_20'] = close.ewm(span=20, adjust=False).mean()
        
        # MACD - Custom implementation
        exp1 = close.ewm(span=self.indicator_params['macd_fast'], adjust=False).mean()
        exp2 = close.ewm(span=self.indicator_params['macd_slow'], adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=self.indicator_params['macd_signal'], adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI - Custom implementation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.indicator_params['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.indicator_params['rsi_period']).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_sma = close.rolling(window=self.indicator_params['bb_period']).mean()
        bb_std = close.rolling(window=self.indicator_params['bb_period']).std()
        df['bb_upper'] = bb_sma + (self.indicator_params['bb_std'] * bb_std)
        df['bb_middle'] = bb_sma
        df['bb_lower'] = bb_sma - (self.indicator_params['bb_std'] * bb_std)
        
        # ATR - Custom implementation
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=self.indicator_params['atr_period']).mean()
        
        # ADX - Simplified implementation
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = true_range.rolling(window=self.indicator_params['adx_period']).sum()
        df['plus_di'] = 100 * (plus_dm.rolling(window=self.indicator_params['adx_period']).sum() / tr)
        df['minus_di'] = 100 * (minus_dm.rolling(window=self.indicator_params['adx_period']).sum() / tr)
        
        dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = dx.rolling(window=self.indicator_params['adx_period']).mean()
        
        # Stochastic
        low_min = low.rolling(window=self.indicator_params['stoch_period']).min()
        high_max = high.rolling(window=self.indicator_params['stoch_period']).max()
        df['stoch_k'] = 100 * ((close - low_min) / (high_max - low_min))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # CCI - Commodity Channel Index
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(window=self.indicator_params['cci_period']).mean()
        mad = typical_price.rolling(window=self.indicator_params['cci_period']).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        # MFI - Money Flow Index
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=self.indicator_params['mfi_period']).sum()
        negative_mf = negative_flow.rolling(window=self.indicator_params['mfi_period']).sum()
        mfi_ratio = positive_mf / negative_mf
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        # Williams %R
        df['willr'] = -100 * ((high_max - close) / (high_max - low_min))
        
        # OBV - On Balance Volume
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        df['obv'] = obv
        
        # Simple pattern detection (not as sophisticated as TA-Lib)
        df['cdl_doji'] = ((abs(df['open'] - close) / (high - low)) < 0.1).astype(int)
        df['cdl_hammer'] = (((high - low) > 3 * abs(df['open'] - close)) & 
                           ((close - low) / (high - low) > 0.6) & 
                           ((df['open'] - low) / (high - low) > 0.6)).astype(int)
        df['cdl_engulfing'] = 0  # Simplified - would need more complex logic
        df['cdl_harami'] = 0  # Simplified - would need more complex logic
        
        # Custom features
        df['price_position'] = (close - low) / (high - low + 1e-10)
        df['volume_ratio'] = volume / volume.rolling(window=20).mean()
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = high / (low + 1e-10)
        df['close_open_ratio'] = close / (df['open'] + 1e-10)
        
        # Volatility features
        df['volatility'] = df['close'].pct_change().rolling(20).std()
        df['atr_ratio'] = df['atr'] / df['close']
        
        # Support/Resistance levels
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['distance_to_resistance'] = (df['resistance'] - close) / close
        df['distance_to_support'] = (close - df['support']) / close
        
        return df.dropna()
        
    def create_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML models"""
        features = pd.DataFrame(index=df.index)
        
        # All technical indicators
        indicator_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        for col in indicator_cols:
            features[col] = df[col]
            
        # Lag features for time series
        for lag in [1, 2, 3, 5, 10]:
            features[f'return_lag_{lag}'] = df['close'].pct_change(lag)
            features[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            
        # Rolling statistics
        for window in [5, 10, 20]:
            features[f'return_mean_{window}'] = df['close'].pct_change().rolling(window).mean()
            features[f'return_std_{window}'] = df['close'].pct_change().rolling(window).std()
            features[f'volume_mean_{window}'] = df['volume'].rolling(window).mean()
            
        # Technical indicator derivatives
        features['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        features['macd_signal_cross'] = np.sign(df['macd'] - df['macd_signal'])
        features['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        features['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        features['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
        
        # Trend features
        features['trend_strength'] = df['adx']
        features['trend_direction'] = np.sign(df['plus_di'] - df['minus_di'])
        features['ema_trend'] = np.sign(df['ema_10'] - df['ema_20'])
        
        return features.dropna()
        
    def train_ensemble_model(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Train ensemble of ML models for a symbol"""
        # Calculate indicators
        df_indicators = self.calculate_technical_indicators(df)
        if len(df_indicators) < 200:
            return None
            
        # Create features
        features = self.create_ml_features(df_indicators)
        
        # Target: Next period return (classification for direction + regression for magnitude)
        target_direction = np.sign(df_indicators['close'].pct_change().shift(-1))
        target_magnitude = df_indicators['close'].pct_change().shift(-1)
        
        # Remove NaN
        valid_idx = features.dropna().index.intersection(target_direction.dropna().index)
        X = features.loc[valid_idx]
        y_direction = target_direction.loc[valid_idx]
        y_magnitude = target_magnitude.loc[valid_idx]
        
        if len(X) < 200:
            return None
            
        # Train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_dir_train, y_dir_test = y_direction.iloc[:split_idx], y_direction.iloc[split_idx:]
        y_mag_train, y_mag_test = y_magnitude.iloc[:split_idx], y_magnitude.iloc[split_idx:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble for direction prediction
        models_direction = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
        }
        
        predictions = []
        for name, model in models_direction.items():
            model.fit(X_train_scaled, y_dir_train)
            pred = model.predict(X_test_scaled)
            predictions.append(pred)
            
        # Ensemble prediction
        ensemble_direction = np.mean(predictions, axis=0)
        
        # Train magnitude predictor only for predicted positive returns
        magnitude_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
        positive_mask = y_mag_train > 0
        if positive_mask.sum() > 50:
            magnitude_model.fit(X_train_scaled[positive_mask], y_mag_train[positive_mask])
        else:
            magnitude_model.fit(X_train_scaled, np.abs(y_mag_train))
            
        return {
            'symbol': symbol,
            'models_direction': models_direction,
            'magnitude_model': magnitude_model,
            'scaler': scaler,
            'features': list(X.columns),
            'test_score': np.mean(np.sign(ensemble_direction) == np.sign(y_dir_test))
        }
        
    def generate_signals(self, date: pd.Timestamp) -> List[Dict]:
        """Generate trading signals for all symbols"""
        signals = []
        
        # Train/update models in parallel
        with ProcessPoolExecutor(max_workers=8) as executor:
            future_to_symbol = {}
            
            for symbol in self.symbols:
                # Load data
                df = self.load_stock_data(symbol, '1h')
                if df.empty or date not in df.index:
                    df = self.load_stock_data(symbol, '1d')
                    
                if not df.empty and date in df.index:
                    # Get historical data up to current date
                    historical = df[df.index <= date]
                    if len(historical) > 200:
                        future = executor.submit(self.train_ensemble_model, symbol, historical)
                        future_to_symbol[future] = symbol
                        
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        self.models[symbol] = result
                        
                        # Generate prediction
                        df = self.load_stock_data(symbol, '1h')
                        if df.empty:
                            df = self.load_stock_data(symbol, '1d')
                            
                        if date in df.index:
                            historical = df[df.index <= date]
                            prediction = self.predict_next_return(symbol, historical)
                            
                            if prediction:
                                signals.append({
                                    'symbol': symbol,
                                    'date': date,
                                    'direction': prediction['direction'],
                                    'magnitude': prediction['magnitude'],
                                    'confidence': prediction['confidence'],
                                    'current_price': historical['close'].iloc[-1],
                                    'volatility': historical['close'].pct_change().rolling(20).std().iloc[-1]
                                })
                                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    
        # Sort by expected return (direction * magnitude * confidence)
        signals.sort(key=lambda x: x['direction'] * x['magnitude'] * x['confidence'], reverse=True)
        return signals
        
    def predict_next_return(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Predict next return for a symbol"""
        if symbol not in self.models:
            return None
            
        model_data = self.models[symbol]
        
        # Calculate indicators
        df_indicators = self.calculate_technical_indicators(df)
        if df_indicators.empty:
            return None
            
        # Create features
        features = self.create_ml_features(df_indicators)
        if features.empty:
            return None
            
        # Get last row
        X_last = features.iloc[[-1]]
        X_scaled = model_data['scaler'].transform(X_last)
        
        # Predict direction
        direction_preds = []
        for model in model_data['models_direction'].values():
            pred = model.predict(X_scaled)[0]
            direction_preds.append(pred)
            
        ensemble_direction = np.mean(direction_preds)
        direction_std = np.std(direction_preds)
        
        # Predict magnitude
        magnitude = model_data['magnitude_model'].predict(X_scaled)[0]
        
        # Calculate confidence
        confidence = 1 / (1 + direction_std) * model_data['test_score']
        
        return {
            'direction': np.sign(ensemble_direction),
            'magnitude': abs(magnitude),
            'confidence': confidence
        }
        
    def execute_trades(self, signals: List[Dict], date: pd.Timestamp):
        """Execute trades based on signals with aggressive position sizing"""
        # Close positions with stop loss or take profit
        for symbol, position in list(self.positions.items()):
            current_price = next((s['current_price'] for s in signals if s['symbol'] == symbol), None)
            if current_price:
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
                
                # Stop loss or take profit
                if pnl_pct <= -self.stop_loss_pct or pnl_pct >= self.take_profit_pct:
                    self.close_position(symbol, current_price, date, 
                                      'Stop Loss' if pnl_pct < 0 else 'Take Profit')
                    
                # Close if signal reversed
                signal = next((s for s in signals if s['symbol'] == symbol), None)
                if signal and signal['direction'] * position['direction'] < 0:
                    self.close_position(symbol, current_price, date, 'Signal Reversed')
                    
        # Open new positions
        available_capital = self.capital * self.leverage - self.margin_used
        
        for signal in signals:
            if len(self.positions) >= self.max_positions:
                break
                
            symbol = signal['symbol']
            if symbol not in self.positions and signal['confidence'] >= self.min_confidence:
                # Aggressive position sizing based on confidence and volatility
                base_size = available_capital * self.max_position_pct
                confidence_mult = signal['confidence'] ** 2  # Square for more aggressive sizing
                volatility_mult = min(signal['volatility'] * 10, 2.0)  # Higher size for volatile stocks
                
                position_size = base_size * confidence_mult * volatility_mult
                position_size = min(position_size, available_capital * 0.8)  # Max 80% of available
                
                if position_size > 5000:  # Minimum position size
                    self.open_position(
                        symbol=symbol,
                        direction=signal['direction'],
                        size=position_size,
                        price=signal['current_price'],
                        date=date,
                        confidence=signal['confidence']
                    )
                    
                    available_capital = self.capital * self.leverage - self.margin_used
                    
    def open_position(self, symbol: str, direction: int, size: float, price: float, 
                     date: pd.Timestamp, confidence: float):
        """Open a new position"""
        shares = int(size / price)
        if shares == 0:
            return
            
        actual_size = shares * price
        commission = actual_size * self.commission_rate
        
        self.positions[symbol] = {
            'direction': direction,
            'shares': shares,
            'entry_price': price,
            'entry_date': date,
            'size': actual_size,
            'confidence': confidence
        }
        
        self.margin_used += actual_size
        
        self.transaction_history.append({
            'date': date,
            'type': 'BUY' if direction > 0 else 'SHORT',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'size': actual_size,
            'commission': commission,
            'confidence': confidence
        })
        
        logger.info(f"{date.date()} - {'BUY' if direction > 0 else 'SHORT'} {shares} {symbol} "
                   f"@ {price:.2f} (Size: {actual_size:,.0f} TL, Confidence: {confidence:.2f})")
                   
    def close_position(self, symbol: str, price: float, date: pd.Timestamp, reason: str):
        """Close a position"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        
        # Calculate P&L
        if position['direction'] > 0:  # Long position
            pnl = (price - position['entry_price']) * position['shares']
        else:  # Short position
            pnl = (position['entry_price'] - price) * position['shares']
            
        pnl_pct = pnl / position['size'] * 100
        commission = position['shares'] * price * self.commission_rate
        net_pnl = pnl - commission
        
        # Update capital
        self.capital += net_pnl
        self.margin_used -= position['size']
        
        # Record transaction
        self.transaction_history.append({
            'date': date,
            'type': 'SELL' if position['direction'] > 0 else 'COVER',
            'symbol': symbol,
            'shares': position['shares'],
            'price': price,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'commission': commission,
            'reason': reason,
            'holding_days': (date - position['entry_date']).days
        })
        
        del self.positions[symbol]
        
        logger.info(f"{date.date()} - CLOSE {symbol} @ {price:.2f} "
                   f"(P&L: {net_pnl:,.0f} TL / {pnl_pct:.2f}%) - {reason}")
                   
    def calculate_portfolio_value(self, date: pd.Timestamp, signals: List[Dict]) -> float:
        """Calculate total portfolio value"""
        # Cash + unrealized P&L
        unrealized_pnl = 0
        
        for symbol, position in self.positions.items():
            current_price = next((s['current_price'] for s in signals if s['symbol'] == symbol), None)
            if current_price:
                if position['direction'] > 0:  # Long
                    unrealized_pnl += (current_price - position['entry_price']) * position['shares']
                else:  # Short
                    unrealized_pnl += (position['entry_price'] - current_price) * position['shares']
                    
        return self.capital + unrealized_pnl
        
    def run_backtest(self, start_date: str = '2024-01-01', end_date: str = '2024-12-31'):
        """Run aggressive backtest"""
        logger.info(f"Running aggressive backtest from {start_date} to {end_date}")
        
        # Get all trading dates
        all_dates = set()
        for symbol in self.symbols[:10]:  # Sample from first 10 symbols
            df = self.load_stock_data(symbol, '1d')
            if not df.empty:
                all_dates.update(df.index.tolist())
                
        # Filter date range
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        trading_dates = sorted([d for d in all_dates if start <= d <= end])
        
        logger.info(f"Backtesting {len(trading_dates)} trading days")
        
        # Run backtest
        for i, date in enumerate(trading_dates):
            # Skip first 60 days for model training
            if i < 60:
                continue
                
            # Generate signals
            signals = self.generate_signals(date)
            
            # Execute trades
            if signals:
                self.execute_trades(signals, date)
                
            # Record portfolio value
            portfolio_value = self.calculate_portfolio_value(date, signals)
            self.portfolio_history.append({
                'date': date,
                'value': portfolio_value,
                'capital': self.capital,
                'positions': len(self.positions),
                'margin_used': self.margin_used
            })
            
            # Log progress
            if (i + 1) % 20 == 0:
                returns = (portfolio_value - self.initial_capital) / self.initial_capital * 100
                logger.info(f"Day {i + 1}: Portfolio {portfolio_value:,.0f} TL ({returns:+.2f}%), "
                           f"Positions: {len(self.positions)}")
                           
    def calculate_performance_metrics(self) -> Dict:
        """Calculate detailed performance metrics"""
        if not self.portfolio_history:
            return {}
            
        df = pd.DataFrame(self.portfolio_history)
        df['returns'] = df['value'].pct_change()
        
        # Overall metrics
        total_return = (df['value'].iloc[-1] - self.initial_capital) / self.initial_capital
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        
        # Monthly returns
        df['month'] = df['date'].dt.to_period('M')
        monthly_returns = df.groupby('month')['value'].apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) > 0 else 0
        )
        
        # Annualized metrics
        annual_factor = 365 / days if days > 0 else 1
        annual_return = (1 + total_return) ** annual_factor - 1
        annual_volatility = df['returns'].std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Max drawdown
        df['cummax'] = df['value'].cummax()
        df['drawdown'] = (df['value'] - df['cummax']) / df['cummax']
        max_drawdown = df['drawdown'].min()
        
        # Trading statistics
        trades = [t for t in self.transaction_history if t['type'] in ['SELL', 'COVER']]
        if trades:
            profitable_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(profitable_trades) / len(trades)
            avg_win = np.mean([t['pnl_pct'] for t in profitable_trades]) if profitable_trades else 0
            avg_loss = np.mean([t['pnl_pct'] for t in trades if t['pnl'] <= 0]) if any(t['pnl'] <= 0 for t in trades) else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
            
        return {
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'monthly_avg_return': monthly_returns.mean() * 100,
            'monthly_returns': monthly_returns * 100,
            'annual_volatility': annual_volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'max_consecutive_wins': self.max_consecutive_wins(),
            'max_consecutive_losses': self.max_consecutive_losses()
        }
        
    def max_consecutive_wins(self) -> int:
        """Calculate maximum consecutive winning trades"""
        trades = [t for t in self.transaction_history if t['type'] in ['SELL', 'COVER']]
        if not trades:
            return 0
            
        max_wins = current_wins = 0
        for trade in trades:
            if trade['pnl'] > 0:
                current_wins += 1
                max_wins = max(max_wins, current_wins)
            else:
                current_wins = 0
                
        return max_wins
        
    def max_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losing trades"""
        trades = [t for t in self.transaction_history if t['type'] in ['SELL', 'COVER']]
        if not trades:
            return 0
            
        max_losses = current_losses = 0
        for trade in trades:
            if trade['pnl'] <= 0:
                current_losses += 1
                max_losses = max(max_losses, current_losses)
            else:
                current_losses = 0
                
        return max_losses
        
    def print_results(self):
        """Print detailed results"""
        metrics = self.calculate_performance_metrics()
        
        print(f"\n{'='*70}")
        print(f"AGGRESSIVE ML TRADING RESULTS")
        print(f"{'='*70}")
        print(f"Initial Capital: {self.initial_capital:,.0f} TL")
        print(f"Final Capital: {self.capital:,.0f} TL")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Annual Return: {metrics['annual_return']:.2f}%")
        print(f"Average Monthly Return: {metrics['monthly_avg_return']:.2f}%")
        print(f"\nRisk Metrics:")
        print(f"Annual Volatility: {metrics['annual_volatility']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"\nTrading Statistics:")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Average Win: {metrics['avg_win']:.2f}%")
        print(f"Average Loss: {metrics['avg_loss']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Max Consecutive Wins: {metrics['max_consecutive_wins']}")
        print(f"Max Consecutive Losses: {metrics['max_consecutive_losses']}")
        
        print(f"\nMonthly Returns:")
        print(f"{'-'*35}")
        for month, ret in metrics['monthly_returns'].items():
            status = "✓" if ret >= 8 else "✗"
            print(f"{month}: {ret:>6.2f}% {status}")
            
        # Target achievement
        months_above_target = sum(1 for r in metrics['monthly_returns'] if r >= 8)
        achievement_rate = months_above_target / len(metrics['monthly_returns']) * 100
        
        print(f"\nTarget Achievement (8% monthly):")
        print(f"Months above target: {months_above_target}/{len(metrics['monthly_returns'])} ({achievement_rate:.1f}%)")
        print(f"{'='*70}")
        
    def plot_results(self, save_path: str = None):
        """Plot comprehensive results"""
        if not self.portfolio_history:
            return
            
        df = pd.DataFrame(self.portfolio_history)
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
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
        ax1.set_title('Portfolio Value - Aggressive Strategy', fontsize=14)
        ax1.set_ylabel('Value (TL)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Monthly returns bar chart
        ax2 = axes[1]
        metrics = self.calculate_performance_metrics()
        monthly_returns = metrics['monthly_returns']
        months = list(monthly_returns.index)
        returns = list(monthly_returns.values)
        colors = ['green' if r >= 8 else 'red' if r < 0 else 'orange' for r in returns]
        
        ax2.bar(range(len(months)), returns, color=colors, alpha=0.7)
        ax2.axhline(y=8, color='blue', linestyle='--', alpha=0.5, label='8% Target')
        ax2.set_title('Monthly Returns', fontsize=14)
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.set_xticks(range(len(months)))
        ax2.set_xticklabels([str(m) for m in months], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Drawdown
        ax3 = axes[2]
        df['cummax'] = df['value'].cummax()
        df['drawdown'] = (df['value'] - df['cummax']) / df['cummax'] * 100
        ax3.fill_between(df['date'], 0, df['drawdown'], color='red', alpha=0.5)
        ax3.plot(df['date'], df['drawdown'], 'r-', linewidth=1)
        ax3.set_title('Drawdown', fontsize=14)
        ax3.set_ylabel('Drawdown (%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Position count and margin usage
        ax4 = axes[3]
        ax4.plot(df['date'], df['positions'], 'purple', linewidth=2, label='Active Positions')
        ax4.set_ylabel('Positions', fontsize=12)
        ax4.set_ylim(0, self.max_positions + 1)
        
        ax4_twin = ax4.twinx()
        margin_usage = df['margin_used'] / (self.initial_capital * self.leverage) * 100
        ax4_twin.plot(df['date'], margin_usage, 'orange', linewidth=2, label='Margin Usage %')
        ax4_twin.set_ylabel('Margin Usage (%)', fontsize=12)
        
        ax4.set_title('Positions and Margin Usage', fontsize=14)
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
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
    
    parser = argparse.ArgumentParser(description='Aggressive ML Trading System')
    parser.add_argument('--capital', type=float, default=50000, help='Initial capital')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date')
    parser.add_argument('--save-plot', type=str, help='Path to save plot')
    
    args = parser.parse_args()
    
    # Create trader
    trader = AggressiveMLTrader(initial_capital=args.capital)
    
    try:
        # Run backtest
        trader.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Print results
        trader.print_results()
        
        # Plot results
        trader.plot_results(save_path=args.save_plot)
        
    except Exception as e:
        logger.error(f"Error in aggressive trading: {e}")
        raise


if __name__ == "__main__":
    main()