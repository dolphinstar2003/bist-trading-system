#!/usr/bin/env python3
"""
ML Model Backtest Visualization
Tests ML models on 1 year of historical data and visualizes predictions vs actual OHLCV
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

# Import our ML models
from ml_models.ml_price_prediction_enhanced import EnhancedPricePredictionEnsemble

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class MLBacktestVisualizer:
    """Backtest ML models and visualize predictions vs actual prices"""
    
    def __init__(self, symbol: str, timeframe: str = '1d'):
        self.symbol = symbol
        self.timeframe = timeframe
        self.ensemble = None
        self.predictions_df = None
        self.metrics = {}
        
    def prepare_data(self, lookback_days: int = 365) -> pd.DataFrame:
        """Load and prepare data for backtesting"""
        # Load raw data
        raw_path = f"data/raw/{self.symbol}_{self.timeframe}_raw.csv"
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
            
        df = pd.read_csv(raw_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
        # Get last N days of data
        end_date = df.index[-1]
        start_date = end_date - timedelta(days=lookback_days)
        df = df[df.index >= start_date]
        
        logger.info(f"Loaded {len(df)} days of data from {df.index[0]} to {df.index[-1]}")
        return df
        
    def run_backtest(self, test_ratio: float = 0.2) -> pd.DataFrame:
        """Run backtest on historical data"""
        logger.info("Starting ML backtest...")
        
        # Load data
        df = self.prepare_data()
        
        # Split data into train and test
        split_idx = int(len(df) * (1 - test_ratio))
        train_end_date = df.index[split_idx]
        
        logger.info(f"Training on data up to {train_end_date}")
        logger.info(f"Testing on {len(df) - split_idx} days")
        
        # Initialize ensemble
        self.ensemble = EnhancedPricePredictionEnsemble(self.symbol, self.timeframe, use_cache=True)
        
        # Load all data with indicators
        full_df = self.ensemble.load_all_data()
        
        # Train models on training period only
        train_df = full_df[full_df.index <= train_end_date]
        self.ensemble.train_models_parallel(train_df)
        
        # Make predictions for test period
        predictions = []
        test_dates = df.index[split_idx + 1:]
        
        for i, test_date in enumerate(test_dates):
            # Use data up to the day before test_date
            current_df = full_df[full_df.index < test_date]
            
            if len(current_df) < 100:  # Need minimum data
                continue
                
            # Predict next day
            pred_result = self.ensemble.predict_enhanced(
                current_df, 
                periods=1,
                use_best_models_only=True,
                top_n_models=4
            )
            
            if not pred_result.empty:
                pred_row = pred_result.iloc[0]
                predictions.append({
                    'Date': test_date,
                    'actual_open': df.loc[test_date, 'open'],
                    'actual_high': df.loc[test_date, 'high'],
                    'actual_low': df.loc[test_date, 'low'],
                    'actual_close': df.loc[test_date, 'close'],
                    'actual_volume': df.loc[test_date, 'volume'],
                    'predicted_close': pred_row['predicted_close'],
                    'prediction_std': pred_row['prediction_std'],
                    'prediction_min': pred_row['prediction_min'],
                    'prediction_max': pred_row['prediction_max'],
                    'confidence': pred_row['confidence']
                })
                
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{len(test_dates)} predictions")
                
        self.predictions_df = pd.DataFrame(predictions)
        
        if self.predictions_df.empty:
            logger.error("No predictions were made. Check if models were trained successfully.")
            return self.predictions_df
            
        self.predictions_df.set_index('Date', inplace=True)
        
        # Calculate metrics
        self._calculate_metrics()
        
        return self.predictions_df
        
    def _calculate_metrics(self):
        """Calculate prediction accuracy metrics"""
        if self.predictions_df is None or self.predictions_df.empty:
            return
            
        actual = self.predictions_df['actual_close']
        predicted = self.predictions_df['predicted_close']
        
        self.metrics = {
            'mae': mean_absolute_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'mape': np.mean(np.abs((actual - predicted) / actual)) * 100,
            'r2': r2_score(actual, predicted),
            'directional_accuracy': np.mean((actual.diff() * predicted.diff()) > 0) * 100,
            'mean_confidence': self.predictions_df['confidence'].mean()
        }
        
        # Calculate profit if trading based on predictions
        returns = actual.pct_change()
        predicted_direction = predicted.diff()
        strategy_returns = returns * np.sign(predicted_direction.shift(1))
        
        self.metrics['cumulative_return'] = (1 + returns).cumprod().iloc[-1] - 1
        self.metrics['strategy_return'] = (1 + strategy_returns).cumprod().iloc[-1] - 1
        self.metrics['sharpe_ratio'] = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        
    def plot_predictions(self, save_path: str = None):
        """Create comprehensive visualization of predictions vs actual"""
        if self.predictions_df is None or self.predictions_df.empty:
            logger.error("No predictions to plot")
            return
            
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Main price chart with predictions
        ax1 = plt.subplot(3, 1, 1)
        self._plot_price_comparison(ax1)
        
        # Prediction error analysis
        ax2 = plt.subplot(3, 2, 3)
        self._plot_error_distribution(ax2)
        
        ax3 = plt.subplot(3, 2, 4)
        self._plot_error_over_time(ax3)
        
        # Performance metrics
        ax4 = plt.subplot(3, 2, 5)
        self._plot_cumulative_returns(ax4)
        
        ax5 = plt.subplot(3, 2, 6)
        self._plot_metrics_summary(ax5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        else:
            plt.show()
            
    def _plot_price_comparison(self, ax):
        """Plot actual OHLC vs predicted prices"""
        df = self.predictions_df
        
        # Plot candlestick chart
        for idx, row in df.iterrows():
            color = 'green' if row['actual_close'] >= row['actual_open'] else 'red'
            # Body
            height = abs(row['actual_close'] - row['actual_open'])
            bottom = min(row['actual_close'], row['actual_open'])
            ax.add_patch(Rectangle((mdates.date2num(idx) - 0.3, bottom), 0.6, height, 
                                 facecolor=color, alpha=0.7))
            # Wick
            ax.plot([mdates.date2num(idx), mdates.date2num(idx)], 
                   [row['actual_low'], row['actual_high']], 
                   color=color, linewidth=1, alpha=0.7)
                   
        # Plot predictions
        ax.plot(df.index, df['predicted_close'], 'b-', label='ML Predictions', linewidth=2)
        
        # Plot confidence bands
        ax.fill_between(df.index, 
                       df['prediction_min'], 
                       df['prediction_max'],
                       alpha=0.2, color='blue', label='Prediction Range')
                       
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax.set_title(f'{self.symbol} - Actual OHLC vs ML Predictions', fontsize=14)
        ax.set_ylabel('Price', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
    def _plot_error_distribution(self, ax):
        """Plot distribution of prediction errors"""
        errors = self.predictions_df['actual_close'] - self.predictions_df['predicted_close']
        errors_pct = (errors / self.predictions_df['actual_close']) * 100
        
        ax.hist(errors_pct, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax.axvline(x=errors_pct.mean(), color='green', linestyle='--', linewidth=2, 
                  label=f'Mean: {errors_pct.mean():.2f}%')
        
        ax.set_title('Prediction Error Distribution', fontsize=12)
        ax.set_xlabel('Error (%)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_error_over_time(self, ax):
        """Plot prediction errors over time"""
        df = self.predictions_df
        errors_pct = ((df['actual_close'] - df['predicted_close']) / df['actual_close']) * 100
        
        ax.scatter(df.index, errors_pct, c=df['confidence'], cmap='RdYlGn', alpha=0.6)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Add rolling mean of errors
        rolling_mean = errors_pct.rolling(window=20).mean()
        ax.plot(df.index, rolling_mean, 'b-', linewidth=2, label='20-day MA')
        
        ax.set_title('Prediction Errors Over Time', fontsize=12)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Error (%)', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=df['confidence'].min(), 
                                                                     vmax=df['confidence'].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Confidence', fontsize=10)
        
    def _plot_cumulative_returns(self, ax):
        """Plot cumulative returns comparison"""
        df = self.predictions_df
        
        # Calculate returns
        actual_returns = df['actual_close'].pct_change()
        
        # Strategy: Buy when predicted to go up, sell when predicted to go down
        predicted_direction = df['predicted_close'].diff()
        position = np.sign(predicted_direction.shift(1))
        strategy_returns = actual_returns * position
        
        # Calculate cumulative returns
        cum_actual = (1 + actual_returns).cumprod()
        cum_strategy = (1 + strategy_returns).cumprod()
        
        ax.plot(df.index, cum_actual, 'g-', label='Buy & Hold', linewidth=2)
        ax.plot(df.index, cum_strategy, 'b-', label='ML Strategy', linewidth=2)
        
        ax.set_title('Cumulative Returns Comparison', fontsize=12)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Cumulative Return', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
    def _plot_metrics_summary(self, ax):
        """Plot summary metrics table"""
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare metrics data
        metrics_data = [
            ['Metric', 'Value'],
            ['Mean Absolute Error', f'{self.metrics["mae"]:.4f}'],
            ['Root Mean Square Error', f'{self.metrics["rmse"]:.4f}'],
            ['Mean Absolute % Error', f'{self.metrics["mape"]:.2f}%'],
            ['R² Score', f'{self.metrics["r2"]:.4f}'],
            ['Directional Accuracy', f'{self.metrics["directional_accuracy"]:.2f}%'],
            ['Mean Confidence', f'{self.metrics["mean_confidence"]:.4f}'],
            ['Buy & Hold Return', f'{self.metrics["cumulative_return"]*100:.2f}%'],
            ['ML Strategy Return', f'{self.metrics["strategy_return"]*100:.2f}%'],
            ['Sharpe Ratio', f'{self.metrics["sharpe_ratio"]:.4f}']
        ]
        
        # Create table
        table = ax.table(cellText=metrics_data, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
            
        # Color code performance metrics
        for i in range(1, len(metrics_data)):
            if 'Return' in metrics_data[i][0]:
                value = float(metrics_data[i][1].replace('%', ''))
                if value > 0:
                    table[(i, 1)].set_facecolor('#90EE90')
                else:
                    table[(i, 1)].set_facecolor('#FFB6C1')
                    
        ax.set_title('Performance Metrics Summary', fontsize=12, pad=20)
        
    def save_results(self, output_dir: str = 'data/predictions'):
        """Save backtest results to CSV"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save predictions
        pred_path = os.path.join(output_dir, f'{self.symbol}_{self.timeframe}_backtest_predictions.csv')
        self.predictions_df.to_csv(pred_path)
        logger.info(f"Predictions saved to {pred_path}")
        
        # Save metrics
        metrics_df = pd.DataFrame([self.metrics])
        metrics_path = os.path.join(output_dir, f'{self.symbol}_{self.timeframe}_backtest_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to {metrics_path}")
        
    def create_detailed_report(self, save_path: str = None):
        """Create a detailed HTML report with interactive charts"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        df = self.predictions_df
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('OHLC vs ML Predictions', 'Prediction Errors', 
                          'Error Distribution', 'Cumulative Returns',
                          'Confidence vs Accuracy', 'Model Performance'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]],
            vertical_spacing=0.1
        )
        
        # 1. OHLC Chart with predictions
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['actual_open'],
                high=df['actual_high'],
                low=df['actual_low'],
                close=df['actual_close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['predicted_close'],
                mode='lines',
                name='ML Predictions',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add confidence bands
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['prediction_max'],
                mode='lines',
                name='Upper Band',
                line=dict(width=0),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['prediction_min'],
                mode='lines',
                name='Lower Band',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(0,100,250,0.2)',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Prediction Errors
        errors_pct = ((df['actual_close'] - df['predicted_close']) / df['actual_close']) * 100
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=errors_pct,
                mode='markers',
                name='Errors',
                marker=dict(
                    color=df['confidence'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Confidence")
                )
            ),
            row=1, col=2
        )
        
        # 3. Error Distribution
        fig.add_trace(
            go.Histogram(
                x=errors_pct,
                nbinsx=30,
                name='Error Distribution'
            ),
            row=2, col=1
        )
        
        # 4. Cumulative Returns
        actual_returns = df['actual_close'].pct_change()
        predicted_direction = df['predicted_close'].diff()
        position = np.sign(predicted_direction.shift(1))
        strategy_returns = actual_returns * position
        
        cum_actual = (1 + actual_returns).cumprod()
        cum_strategy = (1 + strategy_returns).cumprod()
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=cum_actual,
                mode='lines',
                name='Buy & Hold',
                line=dict(color='green', width=2)
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=cum_strategy,
                mode='lines',
                name='ML Strategy',
                line=dict(color='blue', width=2)
            ),
            row=2, col=2
        )
        
        # 5. Confidence vs Accuracy scatter
        abs_errors = np.abs(errors_pct)
        fig.add_trace(
            go.Scatter(
                x=df['confidence'],
                y=abs_errors,
                mode='markers',
                name='Confidence vs Error',
                marker=dict(
                    size=8,
                    color=abs_errors,
                    colorscale='Reds_r',
                    showscale=True,
                    colorbar=dict(title="Abs Error %")
                )
            ),
            row=3, col=1
        )
        
        # 6. Metrics Table
        metrics_data = [
            ['<b>Metric</b>', '<b>Value</b>'],
            ['MAE', f'{self.metrics["mae"]:.4f}'],
            ['RMSE', f'{self.metrics["rmse"]:.4f}'],
            ['MAPE', f'{self.metrics["mape"]:.2f}%'],
            ['R²', f'{self.metrics["r2"]:.4f}'],
            ['Directional Accuracy', f'{self.metrics["directional_accuracy"]:.2f}%'],
            ['Buy & Hold Return', f'{self.metrics["cumulative_return"]*100:.2f}%'],
            ['ML Strategy Return', f'{self.metrics["strategy_return"]*100:.2f}%'],
            ['Sharpe Ratio', f'{self.metrics["sharpe_ratio"]:.4f}']
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                          fill_color='paleturquoise',
                          align='left'),
                cells=dict(values=[[row[0] for row in metrics_data[1:]],
                                 [row[1] for row in metrics_data[1:]]],
                         fill_color='lavender',
                         align='left')
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f'{self.symbol} ML Model Backtest Results',
            height=1200,
            showlegend=True
        )
        
        # Save as HTML
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive report saved to {save_path}")
        else:
            fig.show()


def main():
    """Main function to run backtest visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Model Backtest Visualization')
    parser.add_argument('--symbol', type=str, default='THYAO', help='Stock symbol')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe')
    parser.add_argument('--test-ratio', type=float, default=0.2, help='Test data ratio')
    parser.add_argument('--save-plot', type=str, help='Path to save plot')
    parser.add_argument('--save-report', type=str, help='Path to save HTML report')
    parser.add_argument('--no-display', action='store_true', help='Do not display plots')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = MLBacktestVisualizer(args.symbol, args.timeframe)
    
    try:
        # Run backtest
        predictions = visualizer.run_backtest(test_ratio=args.test_ratio)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"ML Backtest Results for {args.symbol}")
        print(f"{'='*60}")
        print(f"Test Period: {predictions.index[0]} to {predictions.index[-1]}")
        print(f"Total Predictions: {len(predictions)}")
        print(f"\nPerformance Metrics:")
        print(f"{'='*60}")
        for metric, value in visualizer.metrics.items():
            if isinstance(value, float):
                if 'return' in metric or 'accuracy' in metric:
                    print(f"{metric.replace('_', ' ').title()}: {value*100:.2f}%")
                else:
                    print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        print(f"{'='*60}")
        
        # Save results
        visualizer.save_results()
        
        # Create visualizations
        if args.save_plot:
            visualizer.plot_predictions(save_path=args.save_plot)
        elif not args.no_display:
            visualizer.plot_predictions()
            
        # Create interactive report if requested
        if args.save_report:
            try:
                visualizer.create_detailed_report(save_path=args.save_report)
            except ImportError:
                logger.warning("Plotly not installed. Skipping interactive report.")
                
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        raise


if __name__ == "__main__":
    main()