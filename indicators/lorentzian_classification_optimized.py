"""
Optimized Lorentzian Classification
k-NN based machine learning for price direction prediction with performance optimizations
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger
import warnings
from numba import jit, prange
from concurrent.futures import ThreadPoolExecutor
import functools
from sklearn.preprocessing import StandardScaler
from collections import deque

warnings.filterwarnings('ignore')


@jit(nopython=True)
def lorentzian_distance_numba(x1: float, x2: float) -> float:
    """Optimized Lorentzian distance calculation"""
    return np.log(1 + abs(x1 - x2))


@jit(nopython=True)
def calculate_distances_numba(current_features: np.ndarray, 
                            historical_features: np.ndarray) -> np.ndarray:
    """Calculate all distances using numba"""
    n_historical = historical_features.shape[0]
    distances = np.zeros(n_historical)
    
    for i in prange(n_historical):
        dist = 0.0
        for j in range(len(current_features)):
            dist += lorentzian_distance_numba(current_features[j], 
                                             historical_features[i, j])
        distances[i] = dist
    
    return distances


@jit(nopython=True)
def weighted_voting_numba(predictions: np.ndarray, distances: np.ndarray) -> float:
    """Fast weighted voting using numba"""
    if len(predictions) == 0:
        return 0.0
    
    weights = 1.0 / (distances + 1e-10)
    weighted_sum = np.sum(predictions * weights)
    total_weight = np.sum(weights)
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


class OptimizedLorentzianClassification:
    """Optimized Lorentzian distance k-NN classifier"""
    
    def __init__(self, neighbors_count: int = 8, max_bars_back: int = 500,
                 feature_count: int = 5, show_predictions: bool = True,
                 use_kernel_filter: bool = True, use_kernel_smoothing: bool = False,
                 lookback_window: int = 8, relative_weighting: float = 8.0,
                 regression_level: int = 25, kernel_regression_lookback: int = 50,
                 use_cache: bool = True):
        """
        Optimized parameters for faster calculation
        """
        self.neighbors_count = neighbors_count
        self.max_bars_back = min(max_bars_back, 500)  # Limit for performance
        self.feature_count = min(feature_count, 5)  # Limit features
        self.show_predictions = show_predictions
        self.use_kernel_filter = use_kernel_filter
        self.use_kernel_smoothing = use_kernel_smoothing
        self.lookback_window = lookback_window
        self.relative_weighting = relative_weighting
        self.regression_level = regression_level
        self.kernel_regression_lookback = kernel_regression_lookback
        self.use_cache = use_cache
        
        # Cache for features
        self.feature_cache = {}
        self.distance_cache = {}
        
        # Pre-calculate constants
        self.min_required_bars = self.neighbors_count + 4
        
    @functools.lru_cache(maxsize=128)
    def _get_rolling_stats(self, length: int, period: int) -> Tuple[int, int]:
        """Cache rolling window calculations"""
        return max(0, length - period), length
    
    def calculate_rsi_vectorized(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Vectorized RSI calculation"""
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
        
        # Avoid division by zero
        rs = gain / loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_wt_vectorized(self, df: pd.DataFrame, n1: int = 10, n2: int = 21) -> pd.Series:
        """Vectorized WaveTrend calculation"""
        ap = (df['high'] + df['low'] + df['close']) / 3
        esa = ap.ewm(span=n1, adjust=False, min_periods=1).mean()
        d = (ap - esa).abs().ewm(span=n1, adjust=False, min_periods=1).mean()
        ci = (ap - esa) / (0.015 * d + 1e-10)
        wt = ci.ewm(span=n2, adjust=False, min_periods=1).mean()
        return wt.fillna(0)
    
    def calculate_cci_vectorized(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Vectorized CCI calculation"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period, min_periods=1).mean()
        mad = (tp - sma).abs().rolling(window=period, min_periods=1).mean()
        cci = (tp - sma) / (0.015 * mad + 1e-10)
        return cci.fillna(0)
    
    def calculate_adx_simplified(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Simplified ADX for performance"""
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        # Simplified ADX proxy
        adx = (atr / df['close']) * 100
        adx = adx.rolling(window=period, min_periods=1).mean()
        
        return adx.fillna(25)
    
    def normalize_series_fast(self, series: pd.Series, min_val: float = 0, 
                            max_val: float = 1) -> pd.Series:
        """Fast normalization using rolling windows"""
        # Use smaller window for performance
        window = min(self.max_bars_back, 200)
        
        series_min = series.rolling(window=window, min_periods=20).min()
        series_max = series.rolling(window=window, min_periods=20).max()
        
        # Avoid division by zero
        range_val = series_max - series_min
        range_val = range_val.replace(0, 1)
        
        normalized = (series - series_min) / range_val
        normalized = normalized * (max_val - min_val) + min_val
        
        return normalized.fillna((min_val + max_val) / 2)
    
    def calculate_features_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features in batch for better performance"""
        features = pd.DataFrame(index=df.index)
        
        # Pre-calculate all base indicators
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all calculations
            futures = {
                'rsi_14': executor.submit(self.calculate_rsi_vectorized, df['close'], 14),
                'rsi_9': executor.submit(self.calculate_rsi_vectorized, df['close'], 9),
                'wt': executor.submit(self.calculate_wt_vectorized, df),
                'cci': executor.submit(self.calculate_cci_vectorized, df, 20),
                'adx': executor.submit(self.calculate_adx_simplified, df)
            }
            
            # Get results
            results = {name: future.result() for name, future in futures.items()}
        
        # Normalize all at once
        features['f1'] = self.normalize_series_fast(results['rsi_14'], -1, 1)
        features['f2'] = self.normalize_series_fast(results['wt'], -1, 1)
        features['f3'] = self.normalize_series_fast(results['cci'], -1, 1)
        features['f4'] = self.normalize_series_fast(results['adx'], -1, 1)
        features['f5'] = self.normalize_series_fast(results['rsi_9'], -1, 1)
        
        return features
    
    def kernel_regression_fast(self, series: pd.Series, lookback: int, 
                             relative_weight: float) -> pd.Series:
        """Optimized kernel regression using vectorization"""
        result = pd.Series(index=series.index, dtype=float)
        values = series.values
        
        if len(values) < lookback:
            return result
        
        # Pre-calculate weights
        weights = np.exp(-np.power(np.arange(lookback), 2) / (2 * relative_weight ** 2))
        weights = weights / weights.sum()
        
        # Vectorized convolution
        for i in range(lookback, len(values)):
            window = values[i-lookback+1:i+1][::-1]  # Reverse for proper alignment
            result.iloc[i] = np.dot(window, weights)
        
        return result.fillna(method='ffill')
    
    def predict_batch(self, features_array: np.ndarray, close_prices: np.ndarray,
                     start_idx: int, end_idx: int) -> List[Tuple[int, float]]:
        """Batch prediction for multiple bars"""
        results = []
        
        for current_idx in range(start_idx, end_idx):
            if current_idx < self.min_required_bars:
                results.append((current_idx, 0.0))
                continue
            
            # Get current features
            current_features = features_array[current_idx]
            
            # Define historical range
            hist_start = max(0, current_idx - self.max_bars_back)
            hist_end = current_idx - 4
            
            if hist_end - hist_start < self.neighbors_count:
                results.append((current_idx, 0.0))
                continue
            
            # Get historical features
            historical_features = features_array[hist_start:hist_end]
            
            # Calculate distances using numba
            distances = calculate_distances_numba(current_features, historical_features)
            
            # Get k nearest neighbors
            if len(distances) < self.neighbors_count:
                results.append((current_idx, 0.0))
                continue
                
            k_nearest_indices = np.argpartition(distances, self.neighbors_count-1)[:self.neighbors_count]
            k_nearest_distances = distances[k_nearest_indices]
            
            # Calculate directions for nearest neighbors
            predictions = []
            for idx in k_nearest_indices:
                actual_idx = hist_start + idx
                future_idx = actual_idx + 4
                
                if future_idx < len(close_prices):
                    direction = 1 if close_prices[future_idx] > close_prices[actual_idx] else -1
                    predictions.append(direction)
            
            if predictions:
                predictions_array = np.array(predictions, dtype=np.float32)
                prediction = weighted_voting_numba(predictions_array, k_nearest_distances)
            else:
                prediction = 0.0
            
            results.append((current_idx, prediction))
        
        return results
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized Lorentzian Classification calculation"""
        try:
            # Use only last N bars for performance
            if len(df) > self.max_bars_back * 2:
                df = df.iloc[-self.max_bars_back * 2:].copy()
            else:
                df = df.copy()
            
            # Calculate features in batch
            logger.info("Calculating features...")
            features = self.calculate_features_batch(df)
            
            # Convert to numpy for faster computation
            features_array = features.values
            close_prices = df['close'].values
            
            # Initialize result columns
            data = pd.DataFrame(index=df.index)
            data['prediction'] = 0.0
            data['signal'] = 0
            data['confidence'] = 0.5
            
            # Batch predictions
            logger.info("Making predictions...")
            batch_size = 100
            
            for i in range(0, len(df), batch_size):
                start_idx = i
                end_idx = min(i + batch_size, len(df))
                
                batch_results = self.predict_batch(
                    features_array, close_prices, start_idx, end_idx
                )
                
                for idx, prediction in batch_results:
                    data.loc[data.index[idx], 'prediction'] = prediction
                    
                    # Generate signal
                    if prediction > 0.2:
                        data.loc[data.index[idx], 'signal'] = 1
                        data.loc[data.index[idx], 'confidence'] = min(0.5 + abs(prediction), 1.0)
                    elif prediction < -0.2:
                        data.loc[data.index[idx], 'signal'] = -1
                        data.loc[data.index[idx], 'confidence'] = min(0.5 + abs(prediction), 1.0)
            
            # Apply kernel filter if enabled
            if self.use_kernel_filter and len(data) > self.kernel_regression_lookback:
                logger.info("Applying kernel filter...")
                kernel_series = self.kernel_regression_fast(
                    df['close'], 
                    self.kernel_regression_lookback, 
                    self.relative_weighting
                )
                
                data['kernel_estimate'] = kernel_series
                
                # Simple filter: predictions must align with kernel trend
                kernel_trend = (df['close'] > kernel_series).astype(int) * 2 - 1
                data['signal'] = data['signal'].where(
                    data['signal'] == kernel_trend, 0
                )
            
            # Add some basic columns for compatibility
            data['close'] = df['close']
            data['timestamp'] = df.index
            
            logger.info(f"Lorentzian calculation completed for {len(data)} bars")
            
            return data[['prediction', 'signal', 'confidence', 'close', 'timestamp']]
            
        except Exception as e:
            logger.error(f"Error in optimized Lorentzian calculation: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_signal(self, latest_values: Dict[str, Any]) -> str:
        """Get signal from latest values"""
        try:
            signal = latest_values.get('signal', 0)
            confidence = latest_values.get('confidence', 0.5)
            
            if signal == 1:
                if confidence > 0.7:
                    return 'STRONG_BUY'
                else:
                    return 'BUY'
            elif signal == -1:
                if confidence > 0.7:
                    return 'STRONG_SELL'
                else:
                    return 'SELL'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Error getting Lorentzian signal: {e}")
            return 'ERROR'


# For backward compatibility
LorentzianClassification = OptimizedLorentzianClassification