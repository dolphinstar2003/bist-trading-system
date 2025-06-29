#!/usr/bin/env python3
"""
ML Trading System with Advanced Features - Fixed Version
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import joblib
from pathlib import Path
from loguru import logger

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# Local imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.csv_data_manager import CSVDataManager


class MLTradingSystem:
    """ML-based trading system with ensemble models"""
    
    def __init__(self):
        self.csv_manager = CSVDataManager()
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Feature groups
        self.feature_groups = {
            'price_features': [
                'returns_1', 'returns_5', 'returns_20',
                'volatility_20', 'rsi_14', 'distance_to_ma20',
                'distance_to_ma50', 'price_position', 'candle_pattern'
            ],
            'indicator_features': [
                'supertrend_signal', 'adx_value', 'di_diff',
                'squeeze_momentum', 'squeeze_on', 'macd_signal',
                'macd_histogram_value', 'wavetrend_signal', 'wt_value'
            ],
            'ml_indicators': [
                'lorentzian_signal', 'lor_confidence',
                'trend_vanguard_signal', 'tv_strength', 'tv_confidence'
            ],
            'market_structure': [
                'support_distance', 'resistance_distance'
            ]
        }
        
        # Model parameters
        self.model_params = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
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
        
        # Load raw data
        df = self.csv_manager.load_raw_data(symbol, timeframe)
        if df is None:
            logger.error(f"Could not load raw data for {symbol}")
            return None
        
        # Calculate basic features
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_20'] = df['close'].pct_change(20)
        df['volatility_20'] = df['returns_1'].rolling(20).std()
        
        # RSI
        df['rsi_14'] = self.calculate_rsi(df['close'], 14) / 100
        
        # Moving averages
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma50'] = df['close'].rolling(50).mean()
        df['distance_to_ma20'] = (df['close'] - df['ma20']) / df['ma20']
        df['distance_to_ma50'] = (df['close'] - df['ma50']) / df['ma50']
        
        # Price position
        rolling_high = df['high'].rolling(50).max()
        rolling_low = df['low'].rolling(50).min()
        df['price_position'] = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-10)
        
        # Load all indicators
        try:
            df = self.load_all_indicators(df, symbol, timeframe)
        except Exception as e:
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
    
    def load_all_indicators(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load all indicator data"""
        try:
            # Supertrend
            st_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'supertrend')
            if st_data is not None:
                df['supertrend_signal'] = st_data.get('trend', 0).reindex(df.index, fill_value=0)
            
            # ADX/DI
            adx_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'adx_di')
            if adx_data is not None:
                if 'adx' in adx_data.columns:
                    df['adx_value'] = adx_data['adx'].reindex(df.index) / 100
                if 'plus_di' in adx_data.columns and 'minus_di' in adx_data.columns:
                    df['di_diff'] = (adx_data['plus_di'] - adx_data['minus_di']).reindex(df.index)
            
            # Squeeze Momentum
            sqz_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'squeeze_momentum')
            if sqz_data is not None:
                if 'momentum' in sqz_data.columns:
                    df['squeeze_momentum'] = sqz_data['momentum'].reindex(df.index)
                if 'squeeze_on' in sqz_data.columns:
                    # Fix: Handle NaN values before converting to int
                    squeeze_values = sqz_data['squeeze_on'].fillna(0).astype(int)
                    df['squeeze_on'] = squeeze_values.reindex(df.index, fill_value=0)
            
            # MACD
            macd_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'macd')
            if macd_data is not None:
                if 'macd_buy_signal' in macd_data.columns:
                    # Fix: Handle NaN values before converting to int
                    buy_signal = macd_data['macd_buy_signal'].fillna(0).astype(int)
                    sell_signal = macd_data.get('macd_sell_signal', pd.Series(0, index=macd_data.index)).fillna(0).astype(int)
                    df['macd_signal'] = (buy_signal - sell_signal).reindex(df.index, fill_value=0)
                if 'macd_hist' in macd_data.columns:
                    df['macd_histogram_value'] = macd_data['macd_hist'].reindex(df.index)
                elif 'macd_histogram' in macd_data.columns:
                    df['macd_histogram_value'] = macd_data['macd_histogram'].reindex(df.index)
            
            # WaveTrend
            wt_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'wavetrend')
            if wt_data is not None:
                if 'wt_buy_signal' in wt_data.columns:
                    # Fix: Handle NaN values before converting to int
                    buy_signal = wt_data['wt_buy_signal'].fillna(0).astype(int)
                    sell_signal = wt_data.get('wt_sell_signal', pd.Series(0, index=wt_data.index)).fillna(0).astype(int)
                    df['wavetrend_signal'] = (buy_signal - sell_signal).reindex(df.index, fill_value=0)
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
            if tv_data is not None and 'is_pivot_high' in tv_data.columns:
                # Use the main dataframe's close prices for pivot values
                pivot_high_mask = tv_data['is_pivot_high'].reindex(df.index, fill_value=False)
                pivot_low_mask = tv_data['is_pivot_low'].reindex(df.index, fill_value=False)
                
                pivot_highs = df.loc[pivot_high_mask == True, 'close']
                pivot_lows = df.loc[pivot_low_mask == True, 'close']
                
                # Support/Resistance distances
                support_dist, resistance_dist = self.calculate_sr_distances(df, pivot_highs, pivot_lows)
                df['support_distance'] = support_dist
                df['resistance_distance'] = resistance_dist
            else:
                # Default values if no pivot data
                df['support_distance'] = 0.05
                df['resistance_distance'] = 0.05
            
            # Candle patterns
            df['candle_pattern'] = self.identify_candle_patterns(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading indicators: {e}")
            raise
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_sr_distances(self, df: pd.DataFrame, pivot_highs: pd.Series, pivot_lows: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate distance to support and resistance"""
        support_distance = pd.Series(index=df.index, dtype=float)
        resistance_distance = pd.Series(index=df.index, dtype=float)
        
        for idx in df.index:
            current_price = df.loc[idx, 'close']
            
            # Find nearest support (pivot low below current price)
            valid_supports = pivot_lows[pivot_lows < current_price]
            if len(valid_supports) > 0:
                nearest_support = valid_supports.iloc[-1]
                support_distance.loc[idx] = (current_price - nearest_support) / current_price
            else:
                support_distance.loc[idx] = 0.1  # Default if no support found
            
            # Find nearest resistance (pivot high above current price)
            valid_resistances = pivot_highs[pivot_highs > current_price]
            if len(valid_resistances) > 0:
                nearest_resistance = valid_resistances.iloc[0]
                resistance_distance.loc[idx] = (nearest_resistance - current_price) / current_price
            else:
                resistance_distance.loc[idx] = 0.1  # Default if no resistance found
        
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
                logger.info(f"Top 5 features for {model_name}:")
                logger.info(importance.head())
            
            if cv_scores:
                logger.info(f"{model_name} CV score: {np.mean(cv_scores):.3f}")
    
    def predict_ensemble(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Make ensemble prediction for a symbol"""
        if timeframe not in self.models:
            logger.error(f"No models trained for {timeframe}")
            return None
        
        # Load and prepare current data
        df = self.load_and_prepare_data(symbol, timeframe)
        if df is None or len(df) == 0:
            return None
        
        # Get latest data point
        X, _ = self.prepare_features(df.iloc[-1:])
        
        if timeframe not in self.scalers:
            logger.error(f"No scaler found for {timeframe}")
            return None
        
        X_scaled = self.scalers[timeframe].transform(X)
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models[timeframe].items():
            pred = model.predict(X_scaled)[0]
            
            # Convert back from XGBoost format
            if model_name == 'xgboost' and pred == 0:
                pred = -1
            
            predictions[model_name] = int(pred)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)[0]
                probabilities[model_name] = proba.max()
        
        # Ensemble decision (majority vote)
        votes = list(predictions.values())
        ensemble_pred = int(np.median(votes))
        
        # Calculate confidence
        agreement = sum(1 for v in votes if v == ensemble_pred) / len(votes)
        avg_proba = np.mean(list(probabilities.values())) if probabilities else 0.5
        confidence = agreement * avg_proba
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'ensemble_prediction': ensemble_pred,
            'individual_predictions': predictions,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
    
    def save_models(self, timeframe: str):
        """Save trained models"""
        model_dir = Path('ml_models/saved_models')
        model_dir.mkdir(exist_ok=True, parents=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save models
        for model_name, model in self.models.get(timeframe, {}).items():
            filename = model_dir / f"{model_name}_{timeframe}_{timestamp}.joblib"
            joblib.dump(model, filename)
            logger.info(f"Saved {model_name} to {filename}")
        
        # Save scaler
        if timeframe in self.scalers:
            scaler_file = model_dir / f"scaler_{timeframe}_{timestamp}.joblib"
            joblib.dump(self.scalers[timeframe], scaler_file)
            logger.info(f"Saved scaler to {scaler_file}")
        
        # Save feature importance
        if timeframe in self.feature_importance:
            for model_name, importance in self.feature_importance[timeframe].items():
                imp_file = model_dir / f"feature_importance_{model_name}_{timeframe}_{timestamp}.csv"
                importance.to_csv(imp_file, index=False)
                logger.info(f"Saved feature importance to {imp_file}")
    
    def load_models(self, timeframe: str, timestamp: str = None):
        """Load saved models"""
        model_dir = Path('ml_models/saved_models')
        
        if timestamp is None:
            # Find latest models
            model_files = list(model_dir.glob(f"*_{timeframe}_*.joblib"))
            if not model_files:
                logger.error(f"No saved models found for {timeframe}")
                return False
            
            # Get latest timestamp
            timestamps = [f.stem.split('_')[-2] + '_' + f.stem.split('_')[-1] for f in model_files]
            timestamp = max(timestamps)
        
        # Load models
        self.models[timeframe] = {}
        for model_name in ['random_forest', 'gradient_boosting', 'xgboost']:
            model_file = model_dir / f"{model_name}_{timeframe}_{timestamp}.joblib"
            if model_file.exists():
                self.models[timeframe][model_name] = joblib.load(model_file)
                logger.info(f"Loaded {model_name} from {model_file}")
        
        # Load scaler
        scaler_file = model_dir / f"scaler_{timeframe}_{timestamp}.joblib"
        if scaler_file.exists():
            self.scalers[timeframe] = joblib.load(scaler_file)
            logger.info(f"Loaded scaler from {scaler_file}")
        
        return True


if __name__ == "__main__":
    # Test the system
    ml_system = MLTradingSystem()
    
    # Train on some symbols
    symbols = ['THYAO', 'GARAN', 'SAHOL', 'AKBNK', 'EREGL']
    timeframe = '1d'
    
    ml_system.train_ensemble_models(symbols, timeframe)
    
    # Test prediction
    for symbol in symbols[:3]:
        prediction = ml_system.predict_ensemble(symbol, timeframe)
        if prediction:
            print(f"\nPrediction for {symbol}:")
            print(f"Signal: {prediction['ensemble_prediction']}")
            print(f"Confidence: {prediction['confidence']:.2%}")
            print(f"Models agree: {prediction['individual_predictions']}")
    
    # Save models
    ml_system.save_models(timeframe)