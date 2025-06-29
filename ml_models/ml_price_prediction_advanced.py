#!/usr/bin/env python3
"""
Advanced Multi-Model Price Prediction Ensemble System
Uses all technical indicators from data/indicators folder with optimizations
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

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import joblib
from typing import Dict, List, Tuple, Any, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import multiprocessing as mp
from functools import lru_cache
import pickle
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional imports with fallback
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed, will use alternative models")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not installed, will use alternative models")

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    logger.warning("CatBoost not installed, will use alternative models")


class AdvancedPricePredictionEnsemble:
    """Advanced ensemble with all technical indicators and optimizations"""
    
    def __init__(self, symbol: str, timeframe: str = '1d', use_cache: bool = True):
        self.symbol = symbol
        self.timeframe = timeframe
        self.use_cache = use_cache
        self.cache_dir = ".cache/ml_predictions"
        self.models = {}
        self.scalers = {}
        self.optimal_lookbacks = {}
        self.feature_importance = {}
        self.predictions = {}
        self.model_weights = {}
        
        # Define lookback candidates
        self.lookback_candidates = [5, 10, 20, 30, 50, 60, 90, 120]
        
        # Available indicators in data/indicators folder
        self.indicator_list = [
            'lorentzian', 'trend_vanguard', 'supertrend', 'squeeze_momentum',
            'macd', 'wavetrend', 'adx', 'di'
        ]
        
        # Initialize cache
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Initialize models
        self._initialize_models()
        
        # Set number of parallel workers
        self.n_workers = mp.cpu_count() - 1
        logger.info(f"Using {self.n_workers} parallel workers")
        
    def _get_cache_key(self, data_type: str, **kwargs) -> str:
        """Generate unique cache key"""
        key_data = f"{data_type}_{self.symbol}_{self.timeframe}_{kwargs}"
        return hashlib.md5(key_data.encode()).hexdigest()
        
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache if exists"""
        if not self.use_cache:
            return None
            
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    logger.info(f"Loading from cache: {cache_key}")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")
                return None
        return None
        
    def _save_to_cache(self, cache_key: str, data: Any):
        """Save data to cache"""
        if not self.use_cache:
            return
            
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                logger.info(f"Saved to cache: {cache_key}")
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")
            
    def _initialize_models(self):
        """Initialize all ML models with base parameters"""
        # Always available models
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            ),
            'ridge': Ridge(
                alpha=1.0,
                random_state=42
            ),
            'lasso': Lasso(
                alpha=0.001,
                random_state=42,
                max_iter=1000
            ),
            'elastic_net': ElasticNet(
                alpha=0.001,
                l1_ratio=0.5,
                random_state=42,
                max_iter=1000
            ),
            'svr': SVR(
                kernel='rbf',
                C=1.0,
                epsilon=0.1,
                gamma='scale'
            ),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
        }
        
        # Add optional models if available
        if HAS_XGBOOST:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'  # Faster training
            )
            
        if HAS_LIGHTGBM:
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                force_col_wise=True  # Optimization
            )
            
        if HAS_CATBOOST:
            self.models['catboost'] = CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                verbose=False,
                thread_count=-1
            )
            
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data for the symbol and timeframe"""
        cache_key = self._get_cache_key('raw_data')
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        data_path = f"data/raw/{self.symbol}_{self.timeframe}_raw.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"Loaded {len(df)} rows for {self.symbol} {self.timeframe}")
        
        self._save_to_cache(cache_key, df)
        return df
        
    def _load_indicator_parallel(self, indicator_name: str) -> Optional[pd.DataFrame]:
        """Load a single indicator data"""
        try:
            indicator_path = f"data/indicators/{indicator_name}/{self.symbol}_{self.timeframe}.csv"
            if os.path.exists(indicator_path):
                df = pd.read_csv(indicator_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                # Rename columns to include indicator prefix
                df.columns = [f"{indicator_name}_{col}" if col != 'Date' else col for col in df.columns]
                logger.info(f"Loaded {indicator_name} indicator for {self.symbol}")
                return df
            else:
                logger.warning(f"Indicator file not found: {indicator_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading {indicator_name}: {e}")
            return None
            
    def load_all_indicators(self) -> pd.DataFrame:
        """Load all technical indicators from data/indicators folder using parallel processing"""
        cache_key = self._get_cache_key('all_indicators')
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        # Load indicators in parallel
        indicator_dfs = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_indicator = {
                executor.submit(self._load_indicator_parallel, ind): ind 
                for ind in self.indicator_list
            }
            
            for future in as_completed(future_to_indicator):
                result = future.result()
                if result is not None:
                    indicator_dfs.append(result)
                    
        # Combine all indicators
        if indicator_dfs:
            combined_df = indicator_dfs[0]
            for df in indicator_dfs[1:]:
                combined_df = combined_df.join(df, how='outer')
                
            logger.info(f"Loaded {len(combined_df.columns)} indicator features")
            self._save_to_cache(cache_key, combined_df)
            return combined_df
        else:
            return pd.DataFrame()
            
    def create_volume_features(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Create advanced volume features including OBV and VWAP"""
        volume_features = pd.DataFrame(index=df.index)
        
        # On-Balance Volume (OBV)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        volume_features['obv'] = obv
        volume_features['obv_ma'] = obv.rolling(window=min(20, lookback)).mean()
        
        # Volume Weighted Average Price (VWAP)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(window=lookback).sum() / df['volume'].rolling(window=lookback).sum()
        volume_features['vwap'] = vwap
        volume_features['price_vwap_ratio'] = df['close'] / vwap
        
        # Volume Moving Averages
        for period in [5, 10, 20]:
            if period <= lookback:
                volume_features[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
                volume_features[f'volume_ratio_{period}'] = df['volume'] / volume_features[f'volume_ma_{period}']
                
        # Volume Rate of Change
        volume_features['volume_roc'] = df['volume'].pct_change(min(10, lookback))
        
        # Money Flow Index components
        money_flow = typical_price * df['volume']
        volume_features['money_flow'] = money_flow
        volume_features['money_flow_ma'] = money_flow.rolling(window=min(14, lookback)).mean()
        
        # Accumulation/Distribution Line
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)
        adl = (clv * df['volume']).cumsum()
        volume_features['adl'] = adl
        volume_features['adl_ma'] = adl.rolling(window=min(20, lookback)).mean()
        
        return volume_features
        
    def create_features(self, df: pd.DataFrame, indicators_df: pd.DataFrame, lookback: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Create comprehensive features including all indicators"""
        cache_key = self._get_cache_key('features', lookback=lookback)
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
            
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        for i in range(1, min(lookback + 1, 21)):  # Limit to avoid too many features
            features[f'close_lag_{i}'] = df['close'].shift(i)
            features[f'volume_lag_{i}'] = df['volume'].shift(i)
            
        # Price returns and changes
        for period in [1, 5, 10, 20]:
            if period <= lookback:
                features[f'return_{period}'] = df['close'].pct_change(period)
                features[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
                
        # Rolling statistics
        for window in [5, 10, 20, 30]:
            if window <= lookback:
                # Price statistics
                features[f'sma_{window}'] = df['close'].rolling(window).mean()
                features[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
                features[f'std_{window}'] = df['close'].rolling(window).std()
                features[f'skew_{window}'] = df['close'].rolling(window).skew()
                features[f'kurt_{window}'] = df['close'].rolling(window).kurt()
                
                # High-Low statistics
                features[f'high_roll_max_{window}'] = df['high'].rolling(window).max()
                features[f'low_roll_min_{window}'] = df['low'].rolling(window).min()
                features[f'range_{window}'] = features[f'high_roll_max_{window}'] - features[f'low_roll_min_{window}']
                
        # Bollinger Bands
        for period in [20, 30]:
            if period <= lookback:
                sma = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                features[f'bb_upper_{period}'] = sma + (2 * std)
                features[f'bb_lower_{period}'] = sma - (2 * std)
                features[f'bb_width_{period}'] = features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']
                features[f'bb_position_{period}'] = (df['close'] - features[f'bb_lower_{period}']) / features[f'bb_width_{period}']
                
        # Add volume features
        volume_features = self.create_volume_features(df, lookback)
        features = features.join(volume_features)
        
        # Add all technical indicators
        if not indicators_df.empty:
            # Align indices and join
            features = features.join(indicators_df, how='left')
            
        # Technical features
        features['price_position'] = (df['close'] - df['low'].rolling(lookback).min()) / (
            df['high'].rolling(lookback).max() - df['low'].rolling(lookback).min()
        )
        
        # Volatility features
        features['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        features['atr'] = features['true_range'].rolling(min(14, lookback)).mean()
        
        # Price ratios
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # Time-based features
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        # Target: Next period close price
        target = df['close'].shift(-1)
        
        # Drop NaN values
        valid_idx = features.dropna().index.intersection(target.dropna().index)
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]
        
        logger.info(f"Created {len(features.columns)} features with lookback {lookback}")
        
        result = (features, target)
        self._save_to_cache(cache_key, result)
        return result
        
    def _train_model_parallel(self, args):
        """Train a single model (for parallel processing)"""
        model_name, model, df, indicators_df, lookback = args
        try:
            logger.info(f"Training {model_name} with lookback {lookback}")
            
            # Create features
            X, y = self.create_features(df, indicators_df, lookback)
            
            # Split data (80% train, 20% test)
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            # Get feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                feature_importance = dict(zip(X.columns, np.abs(model.coef_)))
                
            results = {
                'model': model,
                'scaler': scaler,
                'lookback': lookback,
                'features': list(X.columns),
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'feature_importance': feature_importance,
                'test_predictions': y_test_pred,
                'test_actual': y_test,
                'test_dates': X_test.index
            }
            
            logger.info(f"{model_name} - Test MAE: {test_mae:.6f}, Test R2: {test_r2:.4f}")
            return model_name, results
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return model_name, None
            
    def find_optimal_lookback(self, df: pd.DataFrame, indicators_df: pd.DataFrame) -> Dict[str, int]:
        """Find optimal lookback for each model using parallel processing"""
        optimal_lookbacks = {}
        
        # Prepare tasks for parallel processing
        tasks = []
        for model_name, model in self.models.items():
            best_score = -np.inf
            best_lookback = self.lookback_candidates[0]
            
            # Quick evaluation using subset of data
            for lookback in self.lookback_candidates:
                try:
                    X, y = self.create_features(df, indicators_df, lookback)
                    if len(X) < 100:
                        continue
                        
                    # Use only last 500 samples for quick evaluation
                    X_eval = X.tail(500)
                    y_eval = y.tail(500)
                    
                    # Simple train/test split
                    split_idx = int(len(X_eval) * 0.8)
                    X_train = X_eval.iloc[:split_idx]
                    y_train = y_eval.iloc[:split_idx]
                    X_test = X_eval.iloc[split_idx:]
                    y_test = y_eval.iloc[split_idx:]
                    
                    # Quick training
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model_copy = model.__class__(**model.get_params())
                    model_copy.fit(X_train_scaled, y_train)
                    
                    y_pred = model_copy.predict(X_test_scaled)
                    score = -mean_absolute_error(y_test, y_pred)
                    
                    if score > best_score:
                        best_score = score
                        best_lookback = lookback
                        
                except Exception as e:
                    logger.warning(f"Error evaluating {model_name} with lookback {lookback}: {e}")
                    continue
                    
            optimal_lookbacks[model_name] = best_lookback
            logger.info(f"Optimal lookback for {model_name}: {best_lookback}")
            
        return optimal_lookbacks
        
    def train_all_models(self, df: pd.DataFrame, indicators_df: pd.DataFrame):
        """Train all models with optimal lookback periods using parallel processing"""
        logger.info("Starting parallel training for all models...")
        
        # Find optimal lookbacks
        self.optimal_lookbacks = self.find_optimal_lookback(df, indicators_df)
        
        # Prepare tasks for parallel training
        tasks = [
            (model_name, model, df, indicators_df, self.optimal_lookbacks[model_name])
            for model_name, model in self.models.items()
        ]
        
        # Train models in parallel
        trained_models = {}
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_model = {
                executor.submit(self._train_model_parallel, task): task[0]
                for task in tasks
            }
            
            for future in as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    name, results = future.result()
                    if results is not None:
                        trained_models[name] = results
                        self.scalers[name] = results['scaler']
                        self.feature_importance[name] = results['feature_importance']
                except Exception as e:
                    logger.error(f"Error getting results for {model_name}: {e}")
                    
        # Calculate model weights based on test performance
        test_scores = {name: -results['test_mae'] for name, results in trained_models.items()}
        total_score = sum(np.exp(score) for score in test_scores.values())
        self.model_weights = {name: np.exp(score) / total_score for name, score in test_scores.items()}
        
        logger.info(f"Model weights: {self.model_weights}")
        
        # Store trained models
        self.models = {name: results['model'] for name, results in trained_models.items()}
        
        return trained_models
        
    def predict(self, df: pd.DataFrame, indicators_df: pd.DataFrame, periods: int = 1, 
                use_best_models_only: bool = True, top_n_models: int = 4) -> pd.DataFrame:
        """Make predictions for next periods using ensemble"""
        predictions = []
        current_df = df.copy()
        current_indicators = indicators_df.copy()
        
        # Filter to use only best models if requested
        if use_best_models_only:
            # Sort models by weight (higher is better)
            model_performances = [(name, self.model_weights[name]) for name in self.models.keys()]
            model_performances.sort(key=lambda x: x[1], reverse=True)
            best_models = [name for name, _ in model_performances[:top_n_models]]
            
            # Recalculate weights for best models only
            best_weights = {name: self.model_weights[name] for name in best_models}
            total_weight = sum(best_weights.values())
            best_weights = {name: w/total_weight for name, w in best_weights.items()}
            
            logger.info(f"Using top {top_n_models} models: {best_models}")
            logger.info(f"Adjusted weights: {best_weights}")
        else:
            best_models = list(self.models.keys())
            best_weights = self.model_weights
            
        for period in range(periods):
            period_predictions = {}
            
            for model_name in best_models:
                model = self.models[model_name]
                lookback = self.optimal_lookbacks[model_name]
                scaler = self.scalers[model_name]
                
                # Create features
                X, _ = self.create_features(current_df, current_indicators, lookback)
                if len(X) == 0:
                    logger.warning(f"No valid features for {model_name}")
                    continue
                    
                # Use last row for prediction
                X_last = X.iloc[[-1]]
                X_scaled = scaler.transform(X_last)
                
                # Predict
                pred = model.predict(X_scaled)[0]
                period_predictions[model_name] = pred
                
            # Weighted ensemble prediction
            if period_predictions:
                ensemble_pred = sum(
                    best_weights.get(name, 0) * pred 
                    for name, pred in period_predictions.items()
                )
                
                # Calculate prediction confidence
                pred_values = list(period_predictions.values())
                pred_std = np.std(pred_values) if len(pred_values) > 1 else 0
                
                # Add prediction to dataframe
                next_date = current_df.index[-1] + timedelta(days=1)
                predictions.append({
                    'Date': next_date,
                    'predicted_close': ensemble_pred,
                    'prediction_std': pred_std,
                    'prediction_min': min(pred_values),
                    'prediction_max': max(pred_values),
                    'confidence': 1 / (1 + pred_std),  # Higher std = lower confidence
                    **{f'{name}_prediction': pred for name, pred in period_predictions.items()}
                })
                
                # Update current_df with prediction
                new_row = pd.DataFrame({
                    'open': ensemble_pred,
                    'high': ensemble_pred * 1.01,
                    'low': ensemble_pred * 0.99,
                    'close': ensemble_pred,
                    'volume': current_df['volume'].mean()
                }, index=[next_date])
                current_df = pd.concat([current_df, new_row])
                
        return pd.DataFrame(predictions)
        
    def evaluate_ensemble(self, trained_models: Dict) -> pd.DataFrame:
        """Evaluate ensemble performance"""
        results = []
        
        for model_name, model_results in trained_models.items():
            results.append({
                'Model': model_name,
                'Lookback': model_results['lookback'],
                'Train MAE': model_results['train_mae'],
                'Test MAE': model_results['test_mae'],
                'Train RMSE': model_results['train_rmse'],
                'Test RMSE': model_results['test_rmse'],
                'Train R²': model_results['train_r2'],
                'Test R²': model_results['test_r2'],
                'Weight': self.model_weights.get(model_name, 0)
            })
            
        return pd.DataFrame(results).sort_values('Test MAE')
        
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get aggregated feature importance across all models"""
        all_features = {}
        
        for model_name, importance in self.feature_importance.items():
            if importance is not None:
                weight = self.model_weights.get(model_name, 0)
                for feature, value in importance.items():
                    if feature not in all_features:
                        all_features[feature] = 0
                    all_features[feature] += value * weight
                    
        # Sort by importance
        sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
        
        # Create dataframe
        importance_df = pd.DataFrame(sorted_features[:top_n], columns=['Feature', 'Importance'])
        importance_df['Importance_Normalized'] = importance_df['Importance'] / importance_df['Importance'].sum()
        
        return importance_df
        
    def save_models(self, path: str = 'models/advanced_prediction'):
        """Save trained models and parameters"""
        os.makedirs(path, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = os.path.join(path, f'{self.symbol}_{self.timeframe}_{model_name}.pkl')
            joblib.dump(model, model_path)
            
        # Save scalers
        for model_name, scaler in self.scalers.items():
            scaler_path = os.path.join(path, f'{self.symbol}_{self.timeframe}_{model_name}_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            
        # Save parameters
        params = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'optimal_lookbacks': self.optimal_lookbacks,
            'model_weights': self.model_weights,
            'feature_importance': self.feature_importance
        }
        
        params_path = os.path.join(path, f'{self.symbol}_{self.timeframe}_params.json')
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=4, default=str)
            
        logger.info(f"Models saved to {path}")


def main():
    """Main function to run advanced price prediction ensemble"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced ML Price Prediction Ensemble')
    parser.add_argument('--symbol', type=str, default='THYAO', help='Stock symbol')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe (1d, 1h, etc)')
    parser.add_argument('--predict-days', type=int, default=5, help='Number of days to predict')
    parser.add_argument('--save-models', action='store_true', help='Save trained models')
    parser.add_argument('--export-csv', action='store_true', help='Export results to CSV')
    parser.add_argument('--use-all-models', action='store_true', help='Use all models instead of best only')
    parser.add_argument('--top-n-models', type=int, default=4, help='Number of top models to use')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    
    args = parser.parse_args()
    
    # Create ensemble
    ensemble = AdvancedPricePredictionEnsemble(
        args.symbol, 
        args.timeframe, 
        use_cache=not args.no_cache
    )
    
    try:
        # Load data
        df = ensemble.load_data()
        indicators_df = ensemble.load_all_indicators()
        
        # Train all models
        trained_models = ensemble.train_all_models(df, indicators_df)
        
        # Evaluate ensemble
        evaluation = ensemble.evaluate_ensemble(trained_models)
        print("\nModel Performance Summary:")
        print(evaluation.to_string(index=False))
        
        # Save evaluation to CSV
        if args.export_csv:
            eval_csv_path = f"data/predictions/{args.symbol}_{args.timeframe}_advanced_model_evaluation.csv"
            os.makedirs("data/predictions", exist_ok=True)
            evaluation.to_csv(eval_csv_path, index=False)
            logger.info(f"Model evaluation saved to {eval_csv_path}")
            
        # Get feature importance
        feature_importance = ensemble.get_feature_importance(top_n=30)
        print("\nTop 30 Most Important Features:")
        print(feature_importance.to_string(index=False))
        
        if args.export_csv:
            importance_csv_path = f"data/predictions/{args.symbol}_{args.timeframe}_feature_importance.csv"
            feature_importance.to_csv(importance_csv_path, index=False)
            
        # Make predictions
        predictions = ensemble.predict(
            df, 
            indicators_df,
            periods=args.predict_days,
            use_best_models_only=not args.use_all_models,
            top_n_models=args.top_n_models
        )
        print(f"\n{args.predict_days}-Day Price Predictions for {args.symbol}:")
        print(predictions.to_string(index=False))
        
        # Save predictions to CSV
        if args.export_csv:
            pred_csv_path = f"data/predictions/{args.symbol}_{args.timeframe}_advanced_predictions.csv"
            predictions.to_csv(pred_csv_path, index=False)
            logger.info(f"Predictions saved to {pred_csv_path}")
            
            # Save detailed test predictions
            for model_name, results in trained_models.items():
                test_preds_df = pd.DataFrame({
                    'Date': results['test_dates'],
                    'Actual': results['test_actual'],
                    'Predicted': results['test_predictions'],
                    'Error': results['test_actual'] - results['test_predictions'],
                    'Error_Pct': ((results['test_actual'] - results['test_predictions']) / results['test_actual'] * 100)
                })
                test_csv_path = f"data/predictions/{args.symbol}_{args.timeframe}_{model_name}_advanced_test_predictions.csv"
                test_preds_df.to_csv(test_csv_path, index=False)
                
        # Calculate ensemble statistics
        print("\nEnsemble Statistics:")
        print(f"Average Test MAE: {evaluation['Test MAE'].mean():.6f}")
        print(f"Average Test R²: {evaluation['Test R²'].mean():.4f}")
        print(f"Best Model: {evaluation.iloc[0]['Model']} (MAE: {evaluation.iloc[0]['Test MAE']:.6f})")
        
        # Show hybrid ensemble info
        if not args.use_all_models:
            print(f"\nHybrid Ensemble using top {args.top_n_models} models:")
            top_models = evaluation.head(args.top_n_models)
            print(top_models[['Model', 'Test MAE', 'Test R²', 'Weight']].to_string(index=False))
            print(f"Hybrid Ensemble Average MAE: {top_models['Test MAE'].mean():.6f}")
            
        # Save models if requested
        if args.save_models:
            ensemble.save_models()
            
    except Exception as e:
        logger.error(f"Error in advanced price prediction: {e}")
        raise


if __name__ == "__main__":
    main()