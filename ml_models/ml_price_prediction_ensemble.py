#!/usr/bin/env python3
"""
Multi-Model Price Prediction Ensemble System
Uses 8 different ML models with optimal lookback periods for price prediction
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

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


class PricePredictionEnsemble:
    """Ensemble of 8 ML models for price prediction with optimal lookback"""
    
    def __init__(self, symbol: str, timeframe: str = '1d'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.models = {}
        self.scalers = {}
        self.optimal_lookbacks = {}
        self.feature_importance = {}
        self.predictions = {}
        self.model_weights = {}
        
        # Define lookback candidates to test
        self.lookback_candidates = [5, 10, 20, 30, 50, 60, 90, 120, 150, 200]
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize all 8 ML models with base parameters"""
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
                n_jobs=-1
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
                verbose=-1
            )
            
        if HAS_CATBOOST:
            self.models['catboost'] = CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            )
            
        logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
        
    def load_data(self) -> pd.DataFrame:
        """Load raw data for the symbol and timeframe"""
        data_path = f"data/raw/{self.symbol}_{self.timeframe}_raw.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        logger.info(f"Loaded {len(df)} rows for {self.symbol} {self.timeframe}")
        return df
        
    def create_features(self, df: pd.DataFrame, lookback: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Create time series features with given lookback period"""
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        for i in range(1, lookback + 1):
            features[f'close_lag_{i}'] = df['close'].shift(i)
            features[f'volume_lag_{i}'] = df['volume'].shift(i)
            features[f'high_lag_{i}'] = df['high'].shift(i)
            features[f'low_lag_{i}'] = df['low'].shift(i)
            
        # Rolling statistics
        for window in [5, 10, 20, 30]:
            if window <= lookback:
                features[f'sma_{window}'] = df['close'].rolling(window).mean()
                features[f'std_{window}'] = df['close'].rolling(window).std()
                features[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
                features[f'high_roll_max_{window}'] = df['high'].rolling(window).max()
                features[f'low_roll_min_{window}'] = df['low'].rolling(window).min()
                
        # Price changes and returns
        for period in [1, 5, 10, 20]:
            if period <= lookback:
                features[f'return_{period}'] = df['close'].pct_change(period)
                features[f'price_change_{period}'] = df['close'].diff(period)
                
        # Technical features
        features['rsi'] = self._calculate_rsi(df['close'], min(14, lookback))
        features['price_position'] = (df['close'] - df['low'].rolling(lookback).min()) / (
            df['high'].rolling(lookback).max() - df['low'].rolling(lookback).min()
        )
        
        # Volume features
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(min(20, lookback)).mean()
        features['price_volume'] = df['close'] * df['volume']
        
        # Volatility features
        features['true_range'] = df['high'] - df['low']
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
        
        return features, target
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def find_optimal_lookback(self, df: pd.DataFrame, model_name: str, model: Any) -> int:
        """Find optimal lookback period for a specific model"""
        logger.info(f"Finding optimal lookback for {model_name}")
        
        best_score = -np.inf
        best_lookback = self.lookback_candidates[0]
        
        for lookback in self.lookback_candidates:
            try:
                # Create features
                X, y = self.create_features(df, lookback)
                
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
                logger.info(f"{model_name} - Lookback {lookback}: Score {avg_score:.6f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_lookback = lookback
                    
            except Exception as e:
                logger.warning(f"Error with lookback {lookback} for {model_name}: {e}")
                continue
                
        logger.info(f"Optimal lookback for {model_name}: {best_lookback} days")
        return best_lookback
        
    def train_model(self, df: pd.DataFrame, model_name: str, model: Any, lookback: int) -> Dict:
        """Train a single model with optimal lookback"""
        logger.info(f"Training {model_name} with lookback {lookback}")
        
        # Create features
        X, y = self.create_features(df, lookback)
        
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
        return results
        
    def train_all_models(self, df: pd.DataFrame):
        """Train all models with optimal lookback periods"""
        logger.info("Starting training for all models...")
        
        # Find optimal lookback for each model
        for model_name, model in self.models.items():
            self.optimal_lookbacks[model_name] = self.find_optimal_lookback(df, model_name, model)
            
        # Train models with optimal lookbacks
        trained_models = {}
        for model_name, model in self.models.items():
            lookback = self.optimal_lookbacks[model_name]
            results = self.train_model(df, model_name, model, lookback)
            trained_models[model_name] = results
            self.scalers[model_name] = results['scaler']
            self.feature_importance[model_name] = results['feature_importance']
            
        # Calculate model weights based on test performance
        test_scores = {name: -results['test_mae'] for name, results in trained_models.items()}
        total_score = sum(np.exp(score) for score in test_scores.values())
        self.model_weights = {name: np.exp(score) / total_score for name, score in test_scores.items()}
        
        logger.info(f"Model weights: {self.model_weights}")
        
        # Store trained models
        self.models = {name: results['model'] for name, results in trained_models.items()}
        
        return trained_models
        
    def predict(self, df: pd.DataFrame, periods: int = 1, use_best_models_only: bool = True, top_n_models: int = 4) -> pd.DataFrame:
        """Make predictions for next periods using ensemble"""
        predictions = []
        current_df = df.copy()
        
        # Filter to use only best models if requested
        if use_best_models_only:
            # Sort models by test MAE (lower is better)
            model_performances = [(name, self.model_weights[name]) for name in self.models.keys()]
            model_performances.sort(key=lambda x: x[1], reverse=True)  # Higher weight = better
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
                # Get model specific parameters
                lookback = self.optimal_lookbacks[model_name]
                scaler = self.scalers[model_name]
                
                # Create features
                X, _ = self.create_features(current_df, lookback)
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
                
                # Calculate prediction confidence (std dev of predictions)
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
        
    def save_models(self, path: str = 'models/price_prediction'):
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
            json.dump(params, f, indent=4)
            
        logger.info(f"Models saved to {path}")


def main():
    """Main function to run price prediction ensemble"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Price Prediction Ensemble')
    parser.add_argument('--symbol', type=str, default='THYAO', help='Stock symbol')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe (1d, 1h, etc)')
    parser.add_argument('--predict-days', type=int, default=5, help='Number of days to predict')
    parser.add_argument('--save-models', action='store_true', help='Save trained models')
    parser.add_argument('--export-csv', action='store_true', help='Export results to CSV')
    parser.add_argument('--use-all-models', action='store_true', help='Use all models instead of best only')
    parser.add_argument('--top-n-models', type=int, default=4, help='Number of top models to use')
    
    args = parser.parse_args()
    
    # Create ensemble
    ensemble = PricePredictionEnsemble(args.symbol, args.timeframe)
    
    try:
        # Load data
        df = ensemble.load_data()
        
        # Train all models
        trained_models = ensemble.train_all_models(df)
        
        # Evaluate ensemble
        evaluation = ensemble.evaluate_ensemble(trained_models)
        print("\nModel Performance Summary:")
        print(evaluation.to_string(index=False))
        
        # Save evaluation to CSV if requested
        if args.export_csv:
            eval_csv_path = f"data/predictions/{args.symbol}_{args.timeframe}_model_evaluation.csv"
            os.makedirs("data/predictions", exist_ok=True)
            evaluation.to_csv(eval_csv_path, index=False)
            logger.info(f"Model evaluation saved to {eval_csv_path}")
        
        # Make predictions using best models
        predictions = ensemble.predict(
            df, 
            periods=args.predict_days,
            use_best_models_only=not args.use_all_models,
            top_n_models=args.top_n_models
        )
        print(f"\n{args.predict_days}-Day Price Predictions for {args.symbol}:")
        print(predictions.to_string(index=False))
        
        # Save predictions to CSV if requested
        if args.export_csv:
            pred_csv_path = f"data/predictions/{args.symbol}_{args.timeframe}_predictions.csv"
            predictions.to_csv(pred_csv_path, index=False)
            logger.info(f"Predictions saved to {pred_csv_path}")
            
            # Also save detailed test predictions for each model
            for model_name, results in trained_models.items():
                test_preds_df = pd.DataFrame({
                    'Date': results['test_dates'],
                    'Actual': results['test_actual'],
                    'Predicted': results['test_predictions'],
                    'Error': results['test_actual'] - results['test_predictions'],
                    'Error_Pct': ((results['test_actual'] - results['test_predictions']) / results['test_actual'] * 100)
                })
                test_csv_path = f"data/predictions/{args.symbol}_{args.timeframe}_{model_name}_test_predictions.csv"
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
        logger.error(f"Error in price prediction: {e}")
        raise


if __name__ == "__main__":
    main()