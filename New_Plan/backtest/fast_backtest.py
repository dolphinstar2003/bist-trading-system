"""
Fast Backtest with Parallel Processing and Caching
Optimized for speed while maintaining accuracy
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import torch
from typing import Dict, List, Tuple, Optional
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from functools import lru_cache
import pickle
import joblib
from numba import jit, njit
import time

from core.csv_data_manager import CSVDataManager
from core.feature_engineering import FeatureEngineering
from indicators.indicator_calculator import IndicatorCalculator
from models.simple_gru_model import SimpleMultiTimeframeGRU
from core.portfolio_manager import PortfolioManager


class FastBacktestEngine:
    """Fast backtest engine with optimizations"""
    
    def __init__(self, config_path: str = 'config.json', num_workers: int = None):
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set number of workers
        self.num_workers = num_workers or mp.cpu_count() - 1
        logger.info(f"Using {self.num_workers} workers for parallel processing")
        
        # Initialize components
        self.csv_manager = CSVDataManager()
        self.indicator_calc = IndicatorCalculator()
        self.feature_engineer = FeatureEngineering(self.config)
        self.portfolio_manager = PortfolioManager(self.config)
        
        # Cache setup
        self.cache_dir = Path('.cache/backtest')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Model loading (cached)
        self.model = self._load_model_cached()
        
        # Pre-load all data into memory
        self.data_cache = {}
        self.indicator_cache = {}
        self.feature_cache = {}
        
        # Results storage
        self.execution_history = []
        
        logger.info("Fast Backtest engine initialized")
    
    @lru_cache(maxsize=1)
    def _load_model_cached(self) -> SimpleMultiTimeframeGRU:
        """Load and cache model"""
        cache_file = self.cache_dir / 'model_cache.pkl'
        
        # Try to load from cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    model = cached_data['model']
                    self.scalers = cached_data['scalers']
                    logger.info("Model loaded from cache")
                    return model
            except:
                pass
        
        # Load model normally
        model_path = Path('models/saved/gru_multi_timeframe.pth')
        if not model_path.exists():
            raise FileNotFoundError("Model not found. Please train the model first.")
        
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Create model
        model = SimpleMultiTimeframeGRU(
            input_size=checkpoint['config']['input_size'],
            hidden_size=checkpoint['config']['hidden_size'],
            num_layers=checkpoint['config']['num_layers']
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Store scalers
        self.scalers = checkpoint['scalers']
        
        # Cache for next time
        with open(cache_file, 'wb') as f:
            pickle.dump({'model': model, 'scalers': self.scalers}, f)
        
        logger.info("Model loaded and cached")
        return model
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str):
        """Run fast backtest with parallel processing"""
        start_time = time.time()
        
        logger.info(f"Starting fast backtest from {start_date} to {end_date}")
        logger.info(f"Processing {len(symbols)} symbols")
        
        # Convert dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Pre-load all data in parallel
        logger.info("Pre-loading all data...")
        self._preload_all_data(symbols, start, end)
        
        # Get trading days
        all_dates = pd.date_range(start, end, freq='D')
        trading_days = [d for d in all_dates if d.weekday() < 5]
        
        # Process each day
        for current_date in tqdm(trading_days, desc="Backtesting"):
            # Update existing positions
            self._update_positions_fast(current_date)
            
            # Generate signals for all symbols in parallel
            daily_predictions = self._generate_daily_predictions_parallel(symbols, current_date)
            
            # Rank and select best signals
            selected_signals = self._select_best_signals_fast(daily_predictions)
            
            # Execute trades
            self._execute_signals_fast(selected_signals, current_date)
        
        # Generate final report
        results = self._generate_report()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Backtest completed in {elapsed_time:.2f} seconds")
        
        return results
    
    def _preload_all_data(self, symbols: List[str], start: pd.Timestamp, end: pd.Timestamp):
        """Pre-load all data into memory for fast access"""
        
        def load_symbol_data(symbol: str):
            """Load data for a single symbol"""
            symbol_data = {}
            
            for tf in self.config['timeframes']['analysis']:
                df = self.csv_manager.get_raw_data(symbol, tf)
                if df is not None:
                    # Filter date range
                    df = df[(df.index >= start) & (df.index <= end)]
                    if len(df) >= 50:  # Lower threshold for data
                        # Calculate indicators
                        indicators = self.indicator_calc.calculate_all_indicators(symbol, tf, save=False)
                        if not indicators.empty:
                            df = pd.concat([df, indicators], axis=1)
                        symbol_data[tf] = df
                    else:
                        logger.debug(f"{symbol} {tf}: Only {len(df)} bars in date range")
            
            return symbol, symbol_data
        
        # Load data in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(load_symbol_data, symbol) for symbol in symbols]
            
            for future in tqdm(futures, desc="Loading data"):
                symbol, data = future.result()
                if data:
                    self.data_cache[symbol] = data
        
        logger.info(f"Loaded data for {len(self.data_cache)} symbols")
        
        # Pre-calculate features for all symbols
        self._precompute_features()
    
    def _precompute_features(self):
        """Pre-compute features for all symbols and cache them"""
        logger.info("Pre-computing features...")
        
        for symbol, data in tqdm(self.data_cache.items(), desc="Computing features"):
            try:
                # Prepare data structure
                symbol_data = {'indicators': data}
                symbol_data['macro'] = {'vix': 20, 'usdtry': 30}
                symbol_data['sentiment'] = {'score': 0, 'count': 0}
                
                # Generate features
                features = self.feature_engineer.create_features(symbol_data, symbol)
                if features:
                    self.feature_cache[symbol] = features
            except Exception as e:
                logger.error(f"Error computing features for {symbol}: {e}")
    
    def _generate_daily_predictions_parallel(self, symbols: List[str], current_date: pd.Timestamp) -> List[Dict]:
        """Generate predictions in parallel"""
        
        def process_symbol(symbol: str):
            """Process a single symbol"""
            try:
                # Skip if already in position
                if symbol in self.portfolio_manager.positions:
                    return None
                
                # Get cached features
                if symbol not in self.feature_cache:
                    return None
                
                features = self.feature_cache[symbol]
                
                # Get data up to current date
                trimmed_features = {}
                for tf, df in features.items():
                    trimmed_features[tf] = df[df.index <= current_date]
                
                # Get ML prediction
                prediction = self._get_ml_prediction_fast(trimmed_features)
                if prediction is None:
                    return None
                
                # Get current price and indicators
                data = self.data_cache.get(symbol, {})
                if '1h' not in data:
                    return None
                
                df_1h = data['1h']
                current_data = df_1h[df_1h.index <= current_date]
                if current_data.empty:
                    return None
                
                current_price = current_data.iloc[-1]['close']
                atr = current_data.iloc[-1].get('atr', current_price * 0.02)
                
                # Check MACD signal
                macd_signal = self._check_macd_signal_fast(data, current_date)
                
                # Create signal if conditions met
                if prediction['probability'] > 0.65 and macd_signal:
                    return {
                        'symbol': symbol,
                        'date': current_date,
                        'ml_probability': prediction['probability'],
                        'ml_confidence': prediction['confidence'],
                        'direction': 'buy',
                        'entry_price': current_price,
                        'stop_loss': current_price - 2 * atr,
                        'target_1': current_price + 2 * atr,
                        'target_2': current_price + 3 * atr,
                        'atr': atr,
                        'macd_confirmed': macd_signal,
                        'signal_strength': prediction['probability'] * (1 if macd_signal else 0.8)
                    }
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                
            return None
        
        # Process symbols in parallel
        predictions = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(process_symbol, symbol) for symbol in symbols]
            
            for future in futures:
                result = future.result()
                if result:
                    predictions.append(result)
        
        return predictions
    
    @lru_cache(maxsize=10000)
    def _get_ml_prediction_fast(self, features_tuple: tuple) -> Optional[Dict]:
        """Get ML prediction with caching (features must be hashable)"""
        # Convert tuple back to dict for processing
        features = dict(features_tuple) if isinstance(features_tuple, tuple) else features_tuple
        
        try:
            with torch.no_grad():
                # Prepare sequences
                sequences = {}
                
                for tf in ['1h', '4h', '1d']:
                    if tf not in features:
                        continue
                    
                    seq_length = self.config['model']['sequence_length']
                    df = features[tf]
                    
                    if len(df) < seq_length:
                        pad_length = seq_length - len(df)
                        padding = pd.DataFrame(
                            np.zeros((pad_length, df.shape[1])),
                            columns=df.columns
                        )
                        df = pd.concat([padding, df])
                    else:
                        df = df.tail(seq_length)
                    
                    # Normalize
                    if tf in self.scalers:
                        normalized = self.scalers[tf].transform(df)
                    else:
                        normalized = (df - df.mean()) / (df.std() + 1e-8)
                    
                    sequences[tf] = torch.FloatTensor(normalized).unsqueeze(0)
                
                # Model prediction
                x_15m = None
                x_1h = sequences.get('1h', None)
                x_4h = sequences.get('4h', None)
                x_1d = sequences.get('1d', None)
                x_1w = None
                
                output, attention_weights = self.model(x_15m, x_1h, x_4h, x_1d, x_1w)
                
                # Convert to probability
                probability = torch.sigmoid(output).item()
                confidence = 0.8  # Simplified confidence
                
                return {
                    'probability': probability,
                    'confidence': confidence,
                    'raw_output': output.item()
                }
                
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return None
    
    def _check_macd_signal_fast(self, data: Dict, current_date: pd.Timestamp) -> bool:
        """Fast MACD signal check"""
        confirmations = 0
        
        for tf in ['1h', '4h', '1d']:
            if tf not in data:
                continue
            
            df = data[tf]
            current_data = df[df.index <= current_date]
            
            if len(current_data) < 2:
                continue
            
            if 'macd' in current_data.columns and 'macd_signal' in current_data.columns:
                current = current_data.iloc[-1]
                if current['macd'] > current['macd_signal']:
                    confirmations += 1
        
        return confirmations >= 2
    
    def _select_best_signals_fast(self, predictions: List[Dict]) -> List[Dict]:
        """Fast signal selection using numpy operations"""
        if not predictions:
            return []
        
        # Convert to numpy for fast operations
        signal_strengths = np.array([p['signal_strength'] for p in predictions])
        
        # Get top N indices
        portfolio_status = self.portfolio_manager.get_portfolio_status()
        available_slots = self.config['portfolio']['max_positions'] - portfolio_status['open_positions']
        
        if available_slots <= 0:
            return []
        
        # Get top signals using argpartition (faster than full sort)
        if len(signal_strengths) > available_slots:
            top_indices = np.argpartition(signal_strengths, -available_slots)[-available_slots:]
            top_indices = top_indices[np.argsort(signal_strengths[top_indices])[::-1]]
        else:
            top_indices = np.argsort(signal_strengths)[::-1]
        
        # Filter and return
        selected = []
        for idx in top_indices:
            signal = predictions[idx]
            if signal['ml_probability'] >= self.config['signals']['confidence_threshold']:
                risk = abs(signal['entry_price'] - signal['stop_loss'])
                reward = signal['target_1'] - signal['entry_price']
                if reward / risk >= 2.0:
                    selected.append(signal)
        
        return selected
    
    def _update_positions_fast(self, current_date: pd.Timestamp):
        """Fast position update"""
        for symbol in list(self.portfolio_manager.positions.keys()):
            try:
                if symbol not in self.data_cache:
                    continue
                
                df = self.data_cache[symbol].get('1h')
                if df is None:
                    continue
                
                current_data = df[df.index <= current_date]
                if current_data.empty:
                    continue
                
                current_price = current_data.iloc[-1]['close']
                result = self.portfolio_manager.update_position(symbol, current_price)
                
                if result and result.get('status') == 'closed':
                    self.execution_history.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'reason': result.get('close_reason', 'unknown')
                    })
                    
            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")
    
    def _execute_signals_fast(self, signals: List[Dict], current_date: pd.Timestamp):
        """Fast signal execution"""
        for signal in signals:
            try:
                # Create order
                order = asyncio.run(self.portfolio_manager.process_signal(signal))
                
                if order:
                    execution_price = signal['entry_price'] * (1 + self.config['backtest']['slippage'])
                    
                    success = self.portfolio_manager.execute_order(order, execution_price)
                    
                    if success:
                        self.execution_history.append({
                            'date': current_date,
                            'symbol': signal['symbol'],
                            'action': 'BUY',
                            'price': execution_price,
                            'ml_probability': signal['ml_probability'],
                            'signal_strength': signal['signal_strength']
                        })
                        
            except Exception as e:
                logger.error(f"Error executing signal for {signal['symbol']}: {e}")
    
    def _generate_report(self) -> Dict:
        """Generate report (same as original)"""
        final_status = self.portfolio_manager.get_portfolio_status()
        
        # Calculate metrics
        execution_df = pd.DataFrame(self.execution_history)
        
        # Monthly returns
        if len(self.portfolio_manager.equity_curve) > 30:
            equity_series = pd.Series(self.portfolio_manager.equity_curve)
            monthly_returns = []
            
            for i in range(30, len(equity_series), 30):
                month_return = (equity_series.iloc[i] - equity_series.iloc[i-30]) / equity_series.iloc[i-30]
                monthly_returns.append(month_return)
            
            avg_monthly_return = np.mean(monthly_returns) * 100 if monthly_returns else 0
        else:
            avg_monthly_return = 0
        
        # Generate plots
        self._plot_results(final_status, execution_df)
        
        # Compile results
        results = {
            'portfolio_status': final_status,
            'total_trades': self.portfolio_manager.performance['total_trades'],
            'win_rate': final_status['win_rate'],
            'total_return': final_status['total_return'],
            'avg_monthly_return': avg_monthly_return,
            'sharpe_ratio': final_status['sharpe_ratio'],
            'max_drawdown': final_status['max_drawdown'],
            'profit_factor': final_status['profit_factor'],
            'execution_history': self.execution_history,
            'final_capital': final_status['total_equity']
        }
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("FAST BACKTEST RESULTS")
        logger.info("="*60)
        logger.info(f"Total Return: {results['total_return']:.2f}%")
        logger.info(f"Avg Monthly Return: {results['avg_monthly_return']:.2f}%")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.1f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        logger.info(f"Final Capital: {results['final_capital']:,.0f} TRY")
        logger.info("="*60)
        
        # Save results
        with open('fast_backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _plot_results(self, portfolio_status: Dict, execution_df: pd.DataFrame):
        """Generate result plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity Curve
        ax = axes[0, 0]
        equity_curve = self.portfolio_manager.equity_curve
        ax.plot(equity_curve, label='Portfolio Value')
        ax.axhline(y=self.config['backtest']['initial_capital'], color='r', linestyle='--', label='Initial Capital')
        ax.set_title('Equity Curve')
        ax.set_ylabel('Portfolio Value (TRY)')
        ax.legend()
        ax.grid(True)
        
        # 2. Drawdown
        ax = axes[0, 1]
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        
        # 3. Trade Distribution
        ax = axes[1, 0]
        if not execution_df.empty:
            buys = execution_df[execution_df['action'] == 'BUY']
            sells = execution_df[execution_df['action'] == 'SELL']
            
            ax.bar(['BUY', 'SELL'], [len(buys), len(sells)], color=['green', 'red'])
            ax.set_title('Trade Distribution')
            ax.set_ylabel('Number of Trades')
        ax.grid(True)
        
        # 4. Performance Metrics
        ax = axes[1, 1]
        metrics_text = f"""
Total Return: {portfolio_status['total_return']:.2f}%
Win Rate: {portfolio_status['win_rate']:.1f}%
Sharpe Ratio: {portfolio_status['sharpe_ratio']:.2f}
Max Drawdown: {portfolio_status['max_drawdown']:.2f}%
Profit Factor: {portfolio_status['profit_factor']:.2f}
"""
        ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes, fontsize=12, verticalalignment='center')
        ax.axis('off')
        ax.set_title('Performance Summary')
        
        plt.tight_layout()
        plt.savefig('fast_backtest_results.png', dpi=300)
        logger.info("Plots saved to fast_backtest_results.png")


def main():
    """Run fast backtest"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run fast backtest for hybrid trading system')
    parser.add_argument('--symbols', nargs='+', help='Symbols to backtest (default: all available)')
    parser.add_argument('--start', default='2025-01-01', help='Start date')
    parser.add_argument('--end', default='2025-06-30', help='End date')
    parser.add_argument('--config', default='config.json', help='Config file path')
    parser.add_argument('--workers', type=int, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Initialize backtest engine
    engine = FastBacktestEngine(args.config, args.workers)
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        # Get all available symbols
        symbols = [s for s in engine.csv_manager.get_available_symbols() if not s.endswith('.IS')]
        logger.info(f"Using all {len(symbols)} available symbols")
    
    # First run the backtest with all symbols - data will be loaded in _preload_all_data
    logger.info(f"Processing {len(symbols)} symbols")
    
    # Run backtest
    results = engine.run_backtest(symbols, args.start, args.end)
    
    logger.info("\nFast Backtest completed successfully!")


if __name__ == "__main__":
    main()