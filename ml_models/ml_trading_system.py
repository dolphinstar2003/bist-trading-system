#!/usr/bin/env python3
"""
ML Trading System - CSV verileri ile gelişmiş trading sistemi
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import json
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Proje imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS


class MLTradingSystem:
    """ML-based trading system using indicator CSVs"""
    
    def __init__(self):
        self.csv_manager = CSVDataManager()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Feature groups
        self.feature_groups = {
            'price_features': [
                'returns_1', 'returns_5', 'returns_10', 'returns_20',
                'volatility_20', 'volatility_50',
                'rsi_14', 'rsi_30',
                'price_to_sma20', 'price_to_sma50',
                'volume_ratio'
            ],
            'trend_indicators': [
                'supertrend_signal', 'adx_value', 'di_diff',
                'trend_strength_20', 'trend_strength_50'
            ],
            'momentum_indicators': [
                'squeeze_momentum', 'squeeze_on',
                'macd_signal', 'macd_histogram_value',
                'wavetrend_signal', 'wt_value'
            ],
            'ml_indicators': [
                'lorentzian_signal', 'lor_confidence',
                'trend_vanguard_signal', 'tv_strength', 'tv_confidence'
            ],
            'pattern_features': [
                'support_distance', 'resistance_distance',
                'pivot_high', 'pivot_low',
                'candle_pattern'
            ]
        }
        
        # Model parameters
        self.model_params = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 150,
                'learning_rate': 0.05,
                'max_depth': 5,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'subsample': 0.8,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 6,
                'min_child_weight': 10,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'random_state': 42
            }
        }
        
        # Target parameters
        self.target_return = 0.02  # %2 hedef getiri
        self.target_horizon = 10   # 10 bar sonrası
        self.stop_loss = 0.05     # %5 stop loss
        
    def load_and_prepare_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load all data for a symbol and prepare features"""
        logger.info(f"Loading data for {symbol} {timeframe}")
        
        # Load price data
        price_data = self.csv_manager.load_raw_data(symbol, timeframe)
        if price_data is None or len(price_data) < 200:
            return None
        
        df = price_data.copy()
        
        # Price features
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['returns_20'] = df['close'].pct_change(20)
        
        df['volatility_20'] = df['returns_1'].rolling(20).std()
        df['volatility_50'] = df['returns_1'].rolling(50).std()
        
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['rsi_30'] = self.calculate_rsi(df['close'], 30)
        
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma50'] = df['close'].rolling(50).mean()
        df['price_to_sma20'] = df['close'] / df['sma20'] - 1
        df['price_to_sma50'] = df['close'] / df['sma50'] - 1
        
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Trend strength
        df['trend_strength_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
        df['trend_strength_50'] = (df['close'] - df['close'].shift(50)) / df['close'].shift(50)
        
        # Support/Resistance
        df['support_distance'], df['resistance_distance'] = self.calculate_support_resistance(df)
        
        # Candle patterns
        df['candle_pattern'] = self.identify_candle_patterns(df)
        
        # Load indicators
        indicators_loaded = self.load_all_indicators(symbol, timeframe, df)
        if not indicators_loaded:
            logger.warning(f"Failed to load indicators for {symbol}")
            return None
        
        # Create target
        df['future_return'] = df['close'].shift(-self.target_horizon) / df['close'] - 1
        df['target'] = 0  # Hold
        
        # Buy signal: future return > target and no stop loss hit
        future_max_loss = df['low'].rolling(self.target_horizon).min().shift(-self.target_horizon) / df['close'] - 1
        df.loc[(df['future_return'] > self.target_return) & 
               (future_max_loss > -self.stop_loss), 'target'] = 1
        
        # Sell signal
        df.loc[df['future_return'] < -self.target_return, 'target'] = -1
        
        # Drop NaN rows
        df = df.dropna()
        
        return df
    
    def load_all_indicators(self, symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
        """Load all indicator data and merge with main dataframe"""
        try:
            # Supertrend
            st_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'supertrend')
            if st_data is not None and 'buy_signal' in st_data.columns:
                df['supertrend_signal'] = st_data['buy_signal'].astype(int) - st_data.get('sell_signal', 0).astype(int)
                df['supertrend_signal'] = df['supertrend_signal'].reindex(df.index, fill_value=0)
            
            # ADX/DI
            adx_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'adx_di')
            if adx_data is not None:
                if 'adx' in adx_data.columns:
                    df['adx_value'] = adx_data['adx'].reindex(df.index)
                if 'plus_di' in adx_data.columns and 'minus_di' in adx_data.columns:
                    df['di_diff'] = (adx_data['plus_di'] - adx_data['minus_di']).reindex(df.index)
            
            # Squeeze Momentum
            sqz_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'squeeze_momentum')
            if sqz_data is not None:
                if 'momentum' in sqz_data.columns:
                    df['squeeze_momentum'] = sqz_data['momentum'].reindex(df.index)
                if 'squeeze_on' in sqz_data.columns:
                    df['squeeze_on'] = sqz_data['squeeze_on'].astype(int).reindex(df.index, fill_value=0)
            
            # MACD
            macd_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'macd')
            if macd_data is not None:
                if 'macd_buy_signal' in macd_data.columns:
                    df['macd_signal'] = (macd_data['macd_buy_signal'].astype(int) - 
                                       macd_data.get('macd_sell_signal', 0).astype(int)).reindex(df.index, fill_value=0)
                if 'macd_hist' in macd_data.columns:
                    df['macd_histogram_value'] = macd_data['macd_hist'].reindex(df.index)
                elif 'macd_histogram' in macd_data.columns:
                    df['macd_histogram_value'] = macd_data['macd_histogram'].reindex(df.index)
            
            # WaveTrend
            wt_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'wavetrend')
            if wt_data is not None:
                if 'wt_buy_signal' in wt_data.columns:
                    df['wavetrend_signal'] = (wt_data['wt_buy_signal'].astype(int) - 
                                            wt_data.get('wt_sell_signal', 0).astype(int)).reindex(df.index, fill_value=0)
                if 'wt1' in wt_data.columns:
                    df['wt_value'] = wt_data['wt1'].reindex(df.index)
            
            # Lorentzian
            lor_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'lorentzian')
            if lor_data is not None:
                if 'signal' in lor_data.columns:
                    df['lorentzian_signal'] = lor_data['signal'].reindex(df.index, fill_value=0)
                if 'confidence' in lor_data.columns:
                    df['lor_confidence'] = lor_data['confidence'].reindex(df.index, fill_value=0.5)
            
            # Trend Vanguard
            tv_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'trend_vanguard')
            if tv_data is not None:
                if 'signal' in tv_data.columns:
                    df['trend_vanguard_signal'] = tv_data['signal'].reindex(df.index, fill_value=0)
                if 'strength' in tv_data.columns:
                    df['tv_strength'] = tv_data['strength'].reindex(df.index, fill_value=0)
                if 'confidence' in tv_data.columns:
                    df['tv_confidence'] = tv_data['confidence'].reindex(df.index, fill_value=0.5)
            
            # Pivot points from Trend Vanguard
            if tv_data is not None:
                if 'is_pivot_high' in tv_data.columns:
                    df['pivot_high'] = tv_data['is_pivot_high'].astype(int).reindex(df.index, fill_value=0)
                if 'is_pivot_low' in tv_data.columns:
                    df['pivot_low'] = tv_data['is_pivot_low'].astype(int).reindex(df.index, fill_value=0)
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading indicators: {e}")
            return False
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 50) -> Tuple[pd.Series, pd.Series]:
        """Calculate distance to support and resistance levels"""
        # Rolling min/max as simple support/resistance
        support = df['low'].rolling(window).min()
        resistance = df['high'].rolling(window).max()
        
        # Distance as percentage
        support_distance = (df['close'] - support) / df['close']
        resistance_distance = (resistance - df['close']) / df['close']
        
        return support_distance, resistance_distance
    
    def identify_candle_patterns(self, df: pd.DataFrame) -> pd.Series:
        """Identify basic candle patterns"""
        pattern = pd.Series(0, index=df.index)
        
        body = abs(df['close'] - df['open'])
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        
        # Hammer: small body, long lower wick
        hammer = (lower_wick > 2 * body) & (df['close'] > df['open'])
        pattern[hammer] = 1
        
        # Shooting star: small body, long upper wick
        shooting_star = (upper_wick > 2 * body) & (df['close'] < df['open'])
        pattern[shooting_star] = -1
        
        # Doji: very small body
        doji = body < (df['high'] - df['low']) * 0.1
        pattern[doji] = 2
        
        return pattern
    
    def get_feature_list(self) -> List[str]:
        """Get all feature names"""
        features = []
        for group in self.feature_groups.values():
            features.extend(group)
        return features
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target"""
        features = self.get_feature_list()
        
        # Check which features are available
        available_features = [f for f in features if f in df.columns]
        missing_features = [f for f in features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        
        # Create feature matrix
        X = df[available_features].copy()
        y = df['target'].copy()
        
        # Fill NaN values
        X = X.fillna(0)
        
        return X, y
    
    def train_ensemble_models(self, symbols: List[str], timeframe: str, test_size: float = 0.2):
        """Train ensemble of models on multiple symbols"""
        logger.info(f"Training ensemble models for {len(symbols)} symbols on {timeframe}")
        
        all_X = []
        all_y = []
        
        # Load and prepare data for all symbols
        for symbol in symbols:
            df = self.load_and_prepare_data(symbol, timeframe)
            if df is not None and len(df) > 100:
                X, y = self.prepare_features(df)
                all_X.append(X)
                all_y.append(y)
                logger.info(f"Loaded {len(X)} samples for {symbol}")
        
        if not all_X:
            logger.error("No data loaded for training")
            return
        
        # Combine all data
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        
        logger.info(f"Total samples: {len(X_combined)}")
        logger.info(f"Target distribution: {y_combined.value_counts().to_dict()}")
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        self.scalers[timeframe] = scaler
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # For binary classification when filtering out hold signals
        # Map -1 (sell) to 0, 1 (buy) to 1
        y_binary = y_combined.copy()
        y_binary = y_binary.map({-1: 0, 0: np.nan, 1: 1})
        
        # Train multiple models
        models_to_train = {
            'random_forest': RandomForestClassifier(**self.model_params['random_forest']),
            'gradient_boosting': GradientBoostingClassifier(**self.model_params['gradient_boosting']),
            'xgboost': xgb.XGBClassifier(**self.model_params['xgboost'], use_label_encoder=False)
        }
        
        self.models[timeframe] = {}
        self.feature_importance[timeframe] = {}
        
        for model_name, model in models_to_train.items():
            logger.info(f"Training {model_name}...")
            
            # Cross-validation scores
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_combined.iloc[train_idx], y_combined.iloc[val_idx]
                
                # Balance classes for training
                # Only use samples where we have clear signals (not hold)
                train_mask = y_train != 0
                X_train = X_train[train_mask]
                y_train = y_train[train_mask]
                
                # For XGBoost, convert -1 to 0
                if model_name == 'xgboost' and len(y_train) > 0:
                    y_train = y_train.copy()
                    y_train[y_train == -1] = 0
                
                if len(np.unique(y_train)) < 2:
                    continue
                
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                cv_scores.append(score)
            
            # Final training on all data
            # Balance the dataset
            train_mask = y_combined != 0
            X_train_final = X_scaled[train_mask]
            y_train_final = y_combined[train_mask]
            
            # For XGBoost, convert -1 to 0
            if model_name == 'xgboost':
                y_train_final = y_train_final.copy()
                y_train_final[y_train_final == -1] = 0
            
            model.fit(X_train_final, y_train_final)
            self.models[timeframe][model_name] = model
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': X_combined.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[timeframe][model_name] = importance
                
                logger.info(f"{model_name} - Top 10 features:")
                print(importance.head(10))
            
            logger.info(f"{model_name} - Average CV score: {np.mean(cv_scores):.4f}")
        
        # Save models
        self.save_models(timeframe)
    
    def predict_ensemble(self, symbol: str, timeframe: str, current_data: pd.DataFrame = None) -> Dict[str, Any]:
        """Make ensemble predictions for a symbol"""
        if timeframe not in self.models:
            logger.error(f"No models trained for {timeframe}")
            return None
        
        # Load data if not provided
        if current_data is None:
            current_data = self.load_and_prepare_data(symbol, timeframe)
            if current_data is None:
                return None
        
        # Get latest data point
        X, _ = self.prepare_features(current_data)
        if len(X) == 0:
            return None
        
        X_latest = X.iloc[-1:].values
        
        # Scale features
        if timeframe in self.scalers:
            X_scaled = self.scalers[timeframe].transform(X_latest)
        else:
            logger.error(f"No scaler found for {timeframe}")
            return None
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models[timeframe].items():
            pred = model.predict(X_scaled)[0]
            # Convert back from XGBoost mapping (0 -> -1)
            if model_name == 'xgboost' and pred == 0:
                pred = -1
            predictions[model_name] = pred
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)[0]
                # Get probability for the predicted class
                if pred == -1 and len(proba) > 0:
                    probabilities[model_name] = proba[0]  # Probability of class -1
                elif pred == 1 and len(proba) > 1:
                    probabilities[model_name] = proba[-1]  # Probability of class 1
                else:
                    probabilities[model_name] = 0.5
        
        # Ensemble decision
        votes = list(predictions.values())
        ensemble_prediction = 1 if sum(votes) > 0 else (-1 if sum(votes) < 0 else 0)
        
        # Confidence based on agreement
        agreement = abs(sum(votes)) / len(votes)
        
        # Get feature values for context
        feature_values = {
            'rsi_14': X_latest[0][X.columns.get_loc('rsi_14')] if 'rsi_14' in X.columns else None,
            'supertrend_signal': X_latest[0][X.columns.get_loc('supertrend_signal')] if 'supertrend_signal' in X.columns else None,
            'squeeze_momentum': X_latest[0][X.columns.get_loc('squeeze_momentum')] if 'squeeze_momentum' in X.columns else None,
            'trend_strength_20': X_latest[0][X.columns.get_loc('trend_strength_20')] if 'trend_strength_20' in X.columns else None
        }
        
        result = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': current_data.index[-1],
            'ensemble_prediction': ensemble_prediction,
            'confidence': agreement,
            'individual_predictions': predictions,
            'probabilities': probabilities,
            'feature_context': feature_values,
            'current_price': current_data['close'].iloc[-1]
        }
        
        return result
    
    def backtest_ml_strategy(self, symbols: List[str], timeframe: str, start_date: str = None):
        """Backtest the ML strategy"""
        if timeframe not in self.models:
            logger.error(f"No models trained for {timeframe}")
            return
        
        results = []
        
        for symbol in symbols:
            logger.info(f"Backtesting {symbol}")
            
            # Load data
            df = self.load_and_prepare_data(symbol, timeframe)
            if df is None:
                continue
            
            # Prepare features
            X, y = self.prepare_features(df)
            if len(X) < 100:
                continue
            
            # Scale features
            X_scaled = self.scalers[timeframe].transform(X)
            
            # Get predictions for each model
            predictions = pd.DataFrame(index=df.index)
            
            for model_name, model in self.models[timeframe].items():
                predictions[model_name] = model.predict(X_scaled)
            
            # Ensemble prediction
            predictions['ensemble'] = predictions.mode(axis=1)[0]
            
            # Calculate returns
            df['signal'] = predictions['ensemble']
            df['position'] = df['signal'].shift(1)  # Enter next bar
            df['strategy_return'] = df['position'] * df['returns_1']
            
            # Calculate metrics
            total_return = (1 + df['strategy_return']).prod() - 1
            sharpe_ratio = df['strategy_return'].mean() / df['strategy_return'].std() * np.sqrt(252)
            max_drawdown = (df['strategy_return'].cumsum().cummax() - df['strategy_return'].cumsum()).max()
            win_rate = (df['strategy_return'] > 0).sum() / (df['strategy_return'] != 0).sum()
            
            result = {
                'symbol': symbol,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'num_trades': (df['signal'] != 0).sum()
            }
            
            results.append(result)
            logger.info(f"{symbol}: Return={total_return:.2%}, Sharpe={sharpe_ratio:.2f}")
        
        # Summary
        results_df = pd.DataFrame(results)
        logger.info("\nBacktest Summary:")
        logger.info(f"Average Return: {results_df['total_return'].mean():.2%}")
        logger.info(f"Average Sharpe: {results_df['sharpe_ratio'].mean():.2f}")
        logger.info(f"Average Win Rate: {results_df['win_rate'].mean():.2%}")
        
        return results_df
    
    def save_models(self, timeframe: str):
        """Save trained models"""
        models_dir = Path("ml_models/saved_models")
        models_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save each model
        for model_name, model in self.models[timeframe].items():
            filename = models_dir / f"{model_name}_{timeframe}_{timestamp}.joblib"
            joblib.dump(model, filename)
            logger.info(f"Saved {model_name} to {filename}")
        
        # Save scaler
        scaler_file = models_dir / f"scaler_{timeframe}_{timestamp}.joblib"
        joblib.dump(self.scalers[timeframe], scaler_file)
        
        # Save feature importance
        importance_file = models_dir / f"feature_importance_{timeframe}_{timestamp}.json"
        importance_data = {}
        for model_name, importance_df in self.feature_importance[timeframe].items():
            importance_data[model_name] = importance_df.to_dict('records')
        
        with open(importance_file, 'w') as f:
            json.dump(importance_data, f, indent=2)
        
        logger.info(f"Models saved with timestamp {timestamp}")
    
    def load_models(self, timeframe: str, timestamp: str):
        """Load previously trained models"""
        models_dir = Path("ml_models/saved_models")
        
        self.models[timeframe] = {}
        
        # Load models
        for model_name in ['random_forest', 'gradient_boosting', 'xgboost']:
            filename = models_dir / f"{model_name}_{timeframe}_{timestamp}.joblib"
            if filename.exists():
                self.models[timeframe][model_name] = joblib.load(filename)
                logger.info(f"Loaded {model_name}")
        
        # Load scaler
        scaler_file = models_dir / f"scaler_{timeframe}_{timestamp}.joblib"
        if scaler_file.exists():
            self.scalers[timeframe] = joblib.load(scaler_file)
        
        logger.info(f"Models loaded for {timeframe}")


def main():
    """Train and test ML trading system"""
    ml_system = MLTradingSystem()
    
    # Select symbols for training
    training_symbols = ['THYAO', 'GARAN', 'SAHOL', 'EREGL', 'AKBNK', 'SISE', 'TUPRS', 'ARCLK']
    timeframe = '1h'
    
    # Train models
    logger.info("Training ML models...")
    ml_system.train_ensemble_models(training_symbols, timeframe)
    
    # Test predictions
    logger.info("\nTesting predictions...")
    for symbol in training_symbols[:3]:
        prediction = ml_system.predict_ensemble(symbol, timeframe)
        if prediction:
            logger.info(f"\n{symbol} Prediction:")
            logger.info(f"  Signal: {prediction['ensemble_prediction']}")
            logger.info(f"  Confidence: {prediction['confidence']:.2f}")
            logger.info(f"  Models agree: {prediction['individual_predictions']}")
    
    # Backtest
    logger.info("\nRunning backtest...")
    backtest_results = ml_system.backtest_ml_strategy(training_symbols, timeframe)
    
    # Save results
    backtest_results.to_csv('ml_models/backtest_results.csv', index=False)
    logger.info("Results saved to ml_models/backtest_results.csv")


if __name__ == "__main__":
    main()