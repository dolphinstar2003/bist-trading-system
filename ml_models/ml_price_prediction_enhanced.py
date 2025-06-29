#!/usr/bin/env python3
"""
Enhanced Multi-Model Price Prediction Ensemble System
Uses all available indicators from data/indicators folder with optimizations
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
from typing import Dict, List, Tuple, Any
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import pickle
from functools import lru_cache
import multiprocessing as mp
from numba import jit, njit

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


class EnhancedPricePredictionEnsemble:
    """Enhanced ensemble with all indicators and optimizations"""
    
    def __init__(self, symbol: str, timeframe: str = '1d', use_cache: bool = True):
        self.symbol = symbol
        self.timeframe = timeframe
        self.use_cache = use_cache
        self.cache_dir = '.cache/ml_predictions'
        self.models = {}
        self.scalers = {}
        self.optimal_lookbacks = {}
        self.feature_importance = {}
        self.predictions = {}
        self.model_weights = {}
        
        # Available indicators in data/indicators folder
        self.available_indicators = [
            'lorentzian', 'trend_vanguard', 'supertrend', 'squeeze_momentum',
            'macd', 'wavetrend', 'adx_di', 'williams_vix_fix'
        ]
        
        # Define lookback candidates to test
        self.lookback_candidates = [5, 10, 20, 30, 50, 60, 90, 120]
        
        # Initialize cache
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        # Initialize models
        self._initialize_models()
        
        # Set up parallel processing
        self.n_jobs = mp.cpu_count() - 1
        logger.info(f"Using {self.n_jobs} CPU cores for parallel processing")
        
    def _initialize_models(self):
        """Initialize all ML models with optimized parameters"""
        # Always available models
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=False,
                random_state=42,
                n_jobs=-1
            ),
            'ridge': Ridge(
                alpha=1.0,
                random_state=42,
                solver='auto'
            ),
            'lasso': Lasso(
                alpha=0.001,
                random_state=42,
                max_iter=2000,
                warm_start=True
            ),
            'elastic_net': ElasticNet(
                alpha=0.001,
                l1_ratio=0.5,
                random_state=42,
                max_iter=2000,
                warm_start=True
            )
        }
        
        # Add optional models if available
        if HAS_XGBOOST:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1,
                tree_method='hist'  # Faster than exact
            )
            
        if HAS_LIGHTGBM:
            self.models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=5,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
                force_col_wise=True  # Faster for many features
            )
            
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        
    @lru_cache(maxsize=128)
    def _load_indicator_data(self, indicator: str) -> pd.DataFrame:
        """Load indicator data from CSV with caching"""
        file_path = f"data/indicators/{self.symbol}_{self.timeframe}_{indicator}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                # Remove timezone info to match raw data
                if df['Date'].dt.tz is not None:
                    df['Date'] = df['Date'].dt.tz_localize(None)
                df.set_index('Date', inplace=True)
            return df
        else:
            logger.warning(f"Indicator file not found: {file_path}")
            return pd.DataFrame()
            
    def load_all_data(self) -> pd.DataFrame:
        """Load raw data and all indicators using parallel processing"""
        # Load raw data
        raw_path = f"data/raw/{self.symbol}_{self.timeframe}_raw.csv"
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
            
        df = pd.read_csv(raw_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Load indicators in parallel
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_indicator = {
                executor.submit(self._load_indicator_data, indicator): indicator
                for indicator in self.available_indicators
            }
            
            for future in as_completed(future_to_indicator):
                indicator = future_to_indicator[future]
                try:
                    indicator_df = future.result()
                    if not indicator_df.empty:
                        # Merge indicator data
                        df = df.join(indicator_df, how='left', rsuffix=f'_{indicator}')
                        logger.info(f"Loaded {indicator} indicator")
                except Exception as e:
                    logger.error(f"Error loading {indicator}: {e}")
                    
        logger.info(f"Loaded {len(df)} rows with {len(df.columns)} features")
        return df
        
    def create_enhanced_features(self, df: pd.DataFrame, lookback: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Create comprehensive features including all technical indicators"""
        features = pd.DataFrame(index=df.index)
        
        # Handle categorical columns by encoding them
        categorical_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                categorical_columns.append(col)
                # Simple label encoding for categorical features
                unique_vals = df[col].dropna().unique()
                mapping = {val: i for i, val in enumerate(unique_vals)}
                features[f'{col}_encoded'] = df[col].map(mapping).fillna(-1)
                
        # Remove categorical columns from main dataframe
        df_numeric = df.drop(columns=categorical_columns)
        
        # Price-based features with multiple lags
        for i in range(1, lookback + 1):
            features[f'close_lag_{i}'] = df['close'].shift(i)
            features[f'volume_lag_{i}'] = df['volume'].shift(i)
            features[f'high_lag_{i}'] = df['high'].shift(i)
            features[f'low_lag_{i}'] = df['low'].shift(i)
            features[f'open_lag_{i}'] = df['open'].shift(i)
            
            # Price ratios
            features[f'close_open_ratio_lag_{i}'] = (df['close'] / df['open']).shift(i)
            features[f'high_low_ratio_lag_{i}'] = (df['high'] / df['low']).shift(i)
            
        # Returns and price changes
        for period in [1, 5, 10, 20, 30]:
            if period <= lookback:
                features[f'return_{period}'] = df['close'].pct_change(period)
                features[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
                features[f'price_change_{period}'] = df['close'].diff(period)
                
        # Volume features
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_ema_ratio'] = df['volume'] / df['volume'].ewm(span=20).mean()
        
        # Calculate OBV (On Balance Volume)
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_sma_ratio'] = obv / obv.rolling(20).mean()
        
        # Calculate VWAP (Volume Weighted Average Price)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        features['vwap'] = vwap
        features['price_vwap_ratio'] = df['close'] / vwap
        
        # Rolling statistics with multiple windows
        for window in [5, 10, 20, 30, 50]:
            if window <= lookback:
                # Price statistics
                features[f'sma_{window}'] = df['close'].rolling(window).mean()
                features[f'ema_{window}'] = df['close'].ewm(span=window).mean()
                features[f'std_{window}'] = df['close'].rolling(window).std()
                features[f'skew_{window}'] = df['close'].rolling(window).skew()
                features[f'kurt_{window}'] = df['close'].rolling(window).kurt()
                
                # Price position
                features[f'price_position_{window}'] = (df['close'] - df['low'].rolling(window).min()) / (
                    df['high'].rolling(window).max() - df['low'].rolling(window).min()
                )
                
                # Volume statistics
                features[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
                features[f'volume_std_{window}'] = df['volume'].rolling(window).std()
                
        # Bollinger Bands
        for window in [20, 30]:
            if window <= lookback:
                sma = df['close'].rolling(window).mean()
                std = df['close'].rolling(window).std()
                features[f'bb_upper_{window}'] = sma + 2 * std
                features[f'bb_lower_{window}'] = sma - 2 * std
                features[f'bb_width_{window}'] = features[f'bb_upper_{window}'] - features[f'bb_lower_{window}']
                features[f'bb_position_{window}'] = (df['close'] - features[f'bb_lower_{window}']) / features[f'bb_width_{window}']
                
        # Add all loaded indicators (numeric only)
        for col in df_numeric.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                features[col] = df_numeric[col]
                
        # Time-based features
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        features['is_month_start'] = df.index.is_month_start.astype(int)
        features['is_month_end'] = df.index.is_month_end.astype(int)
        features['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        features['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Target: Next period close price
        target = df['close'].shift(-1)
        
        # Drop NaN values
        valid_idx = features.dropna().index.intersection(target.dropna().index)
        features = features.loc[valid_idx]
        target = target.loc[valid_idx]
        
        logger.info(f"Created {len(features.columns)} features for lookback {lookback}")
        return features, target
        
    def find_optimal_lookback_parallel(self, df: pd.DataFrame) -> Dict[str, int]:
        """Find optimal lookback for all models using parallel processing"""
        logger.info("Finding optimal lookbacks for all models in parallel...")
        
        # Check cache first
        cache_key = f"{self.symbol}_{self.timeframe}_lookbacks"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
        
        if self.use_cache and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.optimal_lookbacks = pickle.load(f)
                logger.info(f"Loaded optimal lookbacks from cache: {self.optimal_lookbacks}")
                return self.optimal_lookbacks
                
        # Find optimal lookbacks in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_model = {
                executor.submit(self._find_model_lookback, df, model_name, model): (model_name, model)
                for model_name, model in self.models.items()
            }
            
            for future in as_completed(future_to_model):
                model_name, _ = future_to_model[future]
                try:
                    optimal_lookback = future.result()
                    self.optimal_lookbacks[model_name] = optimal_lookback
                    logger.info(f"Optimal lookback for {model_name}: {optimal_lookback}")
                except Exception as e:
                    logger.error(f"Error finding lookback for {model_name}: {e}")
                    self.optimal_lookbacks[model_name] = 20  # Default
                    
        # Cache results
        if self.use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.optimal_lookbacks, f)
                
        return self.optimal_lookbacks
        
    def _find_model_lookback(self, df: pd.DataFrame, model_name: str, model: Any) -> int:
        """Find optimal lookback for a single model"""
        best_score = -np.inf
        best_lookback = self.lookback_candidates[0]
        
        for lookback in self.lookback_candidates:
            try:
                # Create features
                X, y = self.create_enhanced_features(df, lookback)
                
                if len(X) < 100:  # Need minimum samples
                    continue
                    
                # Time series split
                tscv = TimeSeriesSplit(n_splits=3)
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Scale features
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train model
                    model_copy = model.__class__(**model.get_params())
                    model_copy.fit(X_train_scaled, y_train)
                    
                    # Evaluate
                    y_pred = model_copy.predict(X_val_scaled)
                    score = -mean_absolute_error(y_val, y_pred)  # Negative MAE
                    scores.append(score)
                    
                avg_score = np.mean(scores)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_lookback = lookback
                    
            except Exception as e:
                logger.warning(f"Error with lookback {lookback} for {model_name}: {e}")
                continue
                
        return best_lookback
        
    def train_models_parallel(self, df: pd.DataFrame) -> Dict:
        """Train all models in parallel with optimal lookbacks"""
        logger.info("Training all models in parallel...")
        
        # First find optimal lookbacks
        self.find_optimal_lookback_parallel(df)
        
        # Train models in parallel
        trained_models = {}
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_model = {
                executor.submit(
                    self._train_single_model, 
                    df, 
                    model_name, 
                    model, 
                    self.optimal_lookbacks[model_name]
                ): (model_name, model)
                for model_name, model in self.models.items()
            }
            
            for future in as_completed(future_to_model):
                model_name, _ = future_to_model[future]
                try:
                    results = future.result()
                    trained_models[model_name] = results
                    self.scalers[model_name] = results['scaler']
                    self.feature_importance[model_name] = results['feature_importance']
                    logger.info(f"Trained {model_name} - Test MAE: {results['test_mae']:.4f}")
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    
        # Calculate model weights based on performance
        test_scores = {name: -results['test_mae'] for name, results in trained_models.items()}
        total_score = sum(np.exp(score) for score in test_scores.values())
        self.model_weights = {name: np.exp(score) / total_score for name, score in test_scores.items()}
        
        logger.info(f"Model weights: {self.model_weights}")
        
        # Update models with trained versions
        self.models = {name: results['model'] for name, results in trained_models.items()}
        
        return trained_models
        
    def _train_single_model(self, df: pd.DataFrame, model_name: str, model: Any, lookback: int) -> Dict:
        """Train a single model with given lookback"""
        # Create features
        X, y = self.create_enhanced_features(df, lookback)
        
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
            # Get top 20 features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            feature_importance = dict(top_features)
        elif hasattr(model, 'coef_'):
            feature_importance = dict(zip(X.columns, np.abs(model.coef_)))
            # Get top 20 features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            feature_importance = dict(top_features)
            
        return {
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
        
    def predict_enhanced(self, df: pd.DataFrame, periods: int = 1, use_best_models_only: bool = True, top_n_models: int = 4) -> pd.DataFrame:
        """Make predictions with enhanced features"""
        predictions = []
        current_df = df.copy()
        
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
        else:
            best_models = list(self.models.keys())
            best_weights = self.model_weights
            
        for period in range(periods):
            period_predictions = {}
            
            # Parallel predictions for all models
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                future_to_model = {
                    executor.submit(
                        self._predict_single_model,
                        current_df,
                        model_name,
                        self.models[model_name],
                        self.optimal_lookbacks[model_name],
                        self.scalers[model_name]
                    ): model_name
                    for model_name in best_models
                }
                
                for future in as_completed(future_to_model):
                    model_name = future_to_model[future]
                    try:
                        pred = future.result()
                        if pred is not None:
                            period_predictions[model_name] = pred
                    except Exception as e:
                        logger.error(f"Error predicting with {model_name}: {e}")
                        
            # Weighted ensemble prediction
            if period_predictions:
                ensemble_pred = sum(
                    best_weights.get(name, 0) * pred 
                    for name, pred in period_predictions.items()
                )
                
                # Calculate prediction statistics
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
                    'confidence': 1 - (pred_std / ensemble_pred),  # Confidence score
                    **{f'{name}_prediction': pred for name, pred in period_predictions.items()}
                })
                
                # Update current_df with prediction for next iteration
                new_row = pd.DataFrame({
                    'open': ensemble_pred,
                    'high': ensemble_pred * 1.01,
                    'low': ensemble_pred * 0.99,
                    'close': ensemble_pred,
                    'volume': current_df['volume'].mean()
                }, index=[next_date])
                current_df = pd.concat([current_df, new_row])
                
        return pd.DataFrame(predictions)
        
    def _predict_single_model(self, df: pd.DataFrame, model_name: str, model: Any, lookback: int, scaler: Any) -> float:
        """Make prediction with a single model"""
        try:
            # Create features
            X, _ = self.create_enhanced_features(df, lookback)
            if len(X) == 0:
                return None
                
            # Use last row for prediction
            X_last = X.iloc[[-1]]
            X_scaled = scaler.transform(X_last)
            
            # Predict
            pred = model.predict(X_scaled)[0]
            return pred
        except Exception as e:
            logger.error(f"Error in prediction for {model_name}: {e}")
            return None
            
    def evaluate_ensemble(self, trained_models: Dict) -> pd.DataFrame:
        """Evaluate ensemble performance with additional metrics"""
        results = []
        
        for model_name, model_results in trained_models.items():
            # Calculate additional metrics
            test_mape = np.mean(np.abs((model_results['test_actual'] - model_results['test_predictions']) / model_results['test_actual'])) * 100
            
            results.append({
                'Model': model_name,
                'Lookback': model_results['lookback'],
                'Train MAE': model_results['train_mae'],
                'Test MAE': model_results['test_mae'],
                'Train RMSE': model_results['train_rmse'],
                'Test RMSE': model_results['test_rmse'],
                'Train R²': model_results['train_r2'],
                'Test R²': model_results['test_r2'],
                'Test MAPE': test_mape,
                'Weight': self.model_weights.get(model_name, 0),
                'Features': len(model_results['features'])
            })
            
        return pd.DataFrame(results).sort_values('Test MAE')
        
    def save_models(self, path: str = 'models/price_prediction_enhanced'):
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
        
    def get_feature_importance_summary(self) -> pd.DataFrame:
        """Get aggregated feature importance across all models"""
        all_features = {}
        
        for model_name, importance_dict in self.feature_importance.items():
            if importance_dict:
                weight = self.model_weights.get(model_name, 0)
                for feature, importance in importance_dict.items():
                    if feature not in all_features:
                        all_features[feature] = 0
                    all_features[feature] += importance * weight
                    
        # Sort by importance
        sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
        
        return pd.DataFrame(sorted_features[:30], columns=['Feature', 'Weighted_Importance'])


def main():
    """Main function to run enhanced price prediction ensemble"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced ML Price Prediction Ensemble')
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
    ensemble = EnhancedPricePredictionEnsemble(
        args.symbol, 
        args.timeframe, 
        use_cache=not args.no_cache
    )
    
    try:
        # Load all data including indicators
        df = ensemble.load_all_data()
        
        # Train all models in parallel
        trained_models = ensemble.train_models_parallel(df)
        
        # Evaluate ensemble
        evaluation = ensemble.evaluate_ensemble(trained_models)
        print("\nModel Performance Summary:")
        print(evaluation.to_string(index=False))
        
        # Save evaluation to CSV if requested
        if args.export_csv:
            eval_csv_path = f"data/predictions/{args.symbol}_{args.timeframe}_enhanced_model_evaluation.csv"
            os.makedirs("data/predictions", exist_ok=True)
            evaluation.to_csv(eval_csv_path, index=False)
            logger.info(f"Model evaluation saved to {eval_csv_path}")
            
        # Get feature importance summary
        feature_importance = ensemble.get_feature_importance_summary()
        print("\nTop 30 Most Important Features:")
        print(feature_importance.to_string(index=False))
        
        if args.export_csv:
            feat_csv_path = f"data/predictions/{args.symbol}_{args.timeframe}_feature_importance.csv"
            feature_importance.to_csv(feat_csv_path, index=False)
            
        # Make predictions using best models
        predictions = ensemble.predict_enhanced(
            df, 
            periods=args.predict_days,
            use_best_models_only=not args.use_all_models,
            top_n_models=args.top_n_models
        )
        print(f"\n{args.predict_days}-Day Price Predictions for {args.symbol}:")
        print(predictions.to_string(index=False))
        
        # Save predictions to CSV if requested
        if args.export_csv:
            pred_csv_path = f"data/predictions/{args.symbol}_{args.timeframe}_enhanced_predictions.csv"
            predictions.to_csv(pred_csv_path, index=False)
            logger.info(f"Predictions saved to {pred_csv_path}")
            
        # Calculate ensemble statistics
        print("\nEnsemble Statistics:")
        print(f"Average Test MAE: {evaluation['Test MAE'].mean():.6f}")
        print(f"Average Test R²: {evaluation['Test R²'].mean():.4f}")
        print(f"Average Test MAPE: {evaluation['Test MAPE'].mean():.2f}%")
        print(f"Best Model: {evaluation.iloc[0]['Model']} (MAE: {evaluation.iloc[0]['Test MAE']:.6f})")
        
        # Show hybrid ensemble info
        if not args.use_all_models:
            print(f"\nHybrid Ensemble using top {args.top_n_models} models:")
            top_models = evaluation.head(args.top_n_models)
            print(top_models[['Model', 'Test MAE', 'Test R²', 'Test MAPE', 'Weight']].to_string(index=False))
            print(f"Hybrid Ensemble Average MAE: {top_models['Test MAE'].mean():.6f}")
            print(f"Hybrid Ensemble Average MAPE: {top_models['Test MAPE'].mean():.2f}%")
            
        # Save models if requested
        if args.save_models:
            ensemble.save_models()
            
    except Exception as e:
        logger.error(f"Error in enhanced price prediction: {e}")
        raise


if __name__ == "__main__":
    main()