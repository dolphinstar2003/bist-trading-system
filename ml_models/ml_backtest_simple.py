#!/usr/bin/env python3
"""
Simple ML Backtest - Uses only price data and basic indicators for testing
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SimpleMLBacktest:
    """Simple backtest using only price data"""
    
    def __init__(self, symbol: str, timeframe: str = '1d'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'ridge': Ridge(alpha=1.0, random_state=42),
            'lasso': Lasso(alpha=0.01, random_state=42)
        }
        self.trained_models = {}
        self.scalers = {}
        self.predictions_df = None
        self.metrics = {}
        
    def load_data(self, days: int = 365) -> pd.DataFrame:
        """Load raw price data"""
        raw_path = f"data/raw/{self.symbol}_{self.timeframe}_raw.csv"
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
            
        df = pd.read_csv(raw_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Get last N days
        end_date = df.index[-1]
        start_date = end_date - timedelta(days=days)
        df = df[df.index >= start_date]
        
        logger.info(f"Loaded {len(df)} days of data from {df.index[0]} to {df.index[-1]}")
        return df
        
    def create_features(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Create simple features from price data"""
        features = pd.DataFrame(index=df.index)
        
        # Price lags
        for i in range(1, lookback + 1):
            features[f'close_lag_{i}'] = df['close'].shift(i)
            features[f'volume_lag_{i}'] = df['volume'].shift(i)
            
        # Returns
        for period in [1, 5, 10, 20]:
            if period <= lookback:
                features[f'return_{period}'] = df['close'].pct_change(period)
                
        # Moving averages
        for window in [5, 10, 20]:
            if window <= lookback:
                features[f'sma_{window}'] = df['close'].rolling(window).mean()
                features[f'ema_{window}'] = df['close'].ewm(span=window).mean()
                features[f'volume_sma_{window}'] = df['volume'].rolling(window).mean()
                
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        features['bb_upper'] = sma20 + 2 * std20
        features['bb_lower'] = sma20 - 2 * std20
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # OBV
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_sma'] = obv.rolling(20).mean()
        
        # VWAP
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        features['vwap'] = vwap
        features['price_vwap_ratio'] = df['close'] / vwap
        
        # Time features
        features['day_of_week'] = df.index.dayofweek
        features['month'] = df.index.month
        
        return features
        
    def run_backtest(self, test_ratio: float = 0.2):
        """Run simple backtest"""
        logger.info("Starting simple ML backtest...")
        
        # Load data
        df = self.load_data()
        
        # Create features
        features = self.create_features(df)
        
        # Prepare target
        target = df['close'].shift(-1)
        
        # Remove NaN
        valid_idx = features.dropna().index.intersection(target.dropna().index)
        X = features.loc[valid_idx]
        y = target.loc[valid_idx]
        
        # Split data
        split_idx = int(len(X) * (1 - test_ratio))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train models
        predictions = pd.DataFrame(index=X_test.index)
        predictions['actual'] = y_test
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            predictions[f'{name}_pred'] = y_pred
            
            # Store
            self.trained_models[name] = model
            self.scalers[name] = scaler
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
        # Ensemble prediction
        model_columns = [col for col in predictions.columns if col.endswith('_pred')]
        predictions['ensemble_pred'] = predictions[model_columns].mean(axis=1)
        
        # Add OHLCV data
        predictions['open'] = df.loc[predictions.index, 'open']
        predictions['high'] = df.loc[predictions.index, 'high']
        predictions['low'] = df.loc[predictions.index, 'low']
        predictions['volume'] = df.loc[predictions.index, 'volume']
        
        self.predictions_df = predictions
        
        # Calculate metrics
        self._calculate_metrics()
        
        return predictions
        
    def _calculate_metrics(self):
        """Calculate performance metrics"""
        df = self.predictions_df
        
        for model in ['random_forest', 'gradient_boosting', 'ridge', 'lasso', 'ensemble']:
            pred_col = f'{model}_pred'
            
            mae = mean_absolute_error(df['actual'], df[pred_col])
            rmse = np.sqrt(mean_squared_error(df['actual'], df[pred_col]))
            mape = np.mean(np.abs((df['actual'] - df[pred_col]) / df['actual'])) * 100
            r2 = r2_score(df['actual'], df[pred_col])
            
            # Directional accuracy
            actual_direction = df['actual'].diff()
            pred_direction = df[pred_col].diff()
            dir_acc = np.mean((actual_direction * pred_direction) > 0) * 100
            
            self.metrics[model] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2,
                'directional_accuracy': dir_acc
            }
            
    def plot_results(self, save_path: str = None):
        """Plot backtest results"""
        if self.predictions_df is None:
            logger.error("No predictions to plot")
            return
            
        df = self.predictions_df
        
        # Create figure
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Main price chart
        ax1 = plt.subplot(2, 1, 1)
        
        # Plot candlesticks
        for idx, row in df.iterrows():
            color = 'green' if row['actual'] >= row['open'] else 'red'
            # Body
            height = abs(row['actual'] - row['open'])
            bottom = min(row['actual'], row['open'])
            ax1.add_patch(Rectangle((mdates.date2num(idx) - 0.3, bottom), 0.6, height, 
                                   facecolor=color, alpha=0.7))
            # Wick
            ax1.plot([mdates.date2num(idx), mdates.date2num(idx)], 
                    [row['low'], row['high']], 
                    color=color, linewidth=1, alpha=0.7)
                    
        # Plot predictions
        ax1.plot(df.index, df['ensemble_pred'], 'b-', label='Ensemble Prediction', linewidth=2)
        ax1.plot(df.index, df['ridge_pred'], 'g--', label='Ridge', alpha=0.7)
        ax1.plot(df.index, df['lasso_pred'], 'r--', label='Lasso', alpha=0.7)
        
        ax1.set_title(f'{self.symbol} - Actual vs ML Predictions', fontsize=14)
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Error analysis
        ax2 = plt.subplot(2, 2, 3)
        errors = ((df['actual'] - df['ensemble_pred']) / df['actual']) * 100
        ax2.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_title('Ensemble Prediction Error Distribution', fontsize=12)
        ax2.set_xlabel('Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # 3. Model comparison
        ax3 = plt.subplot(2, 2, 4)
        model_names = list(self.metrics.keys())
        mae_values = [self.metrics[m]['mae'] for m in model_names]
        colors = ['blue' if m == 'ensemble' else 'gray' for m in model_names]
        
        bars = ax3.bar(model_names, mae_values, color=colors, alpha=0.7)
        ax3.set_title('Model MAE Comparison', fontsize=12)
        ax3.set_ylabel('MAE')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, mae_values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{val:.2f}', ha='center', va='bottom')
                    
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def print_summary(self):
        """Print summary of results"""
        print(f"\n{'='*60}")
        print(f"Simple ML Backtest Results for {self.symbol}")
        print(f"{'='*60}")
        print(f"Test Period: {self.predictions_df.index[0]} to {self.predictions_df.index[-1]}")
        print(f"Total Predictions: {len(self.predictions_df)}")
        print(f"\nModel Performance:")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<10} {'Dir Acc':<10}")
        print(f"{'-'*60}")
        
        for model, metrics in self.metrics.items():
            print(f"{model:<20} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} "
                  f"{metrics['mape']:<10.2f} {metrics['r2']:<10.4f} {metrics['directional_accuracy']:<10.2f}")
        print(f"{'='*60}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple ML Backtest')
    parser.add_argument('--symbol', type=str, default='THYAO', help='Stock symbol')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Test data ratio')
    parser.add_argument('--save-plot', type=str, help='Path to save plot')
    
    args = parser.parse_args()
    
    # Run backtest
    backtest = SimpleMLBacktest(args.symbol, args.timeframe)
    
    try:
        predictions = backtest.run_backtest(test_ratio=args.test_ratio)
        
        # Print summary
        backtest.print_summary()
        
        # Save results
        output_path = f"data/predictions/{args.symbol}_{args.timeframe}_simple_backtest.csv"
        os.makedirs("data/predictions", exist_ok=True)
        predictions.to_csv(output_path)
        logger.info(f"Results saved to {output_path}")
        
        # Plot results
        backtest.plot_results(save_path=args.save_plot)
        
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        raise


if __name__ == "__main__":
    main()