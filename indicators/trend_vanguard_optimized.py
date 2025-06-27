"""
Optimized Trend Vanguard Strategy
Performance-optimized version with vectorization and caching
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger
import warnings
from numba import jit, prange
from functools import lru_cache
import scipy.signal

warnings.filterwarnings('ignore')


@jit(nopython=True)
def find_pivot_points_numba(high: np.ndarray, low: np.ndarray, depth: int) -> Tuple[np.ndarray, np.ndarray]:
    """Find pivot points using numba for speed"""
    n = len(high)
    pivot_highs = np.zeros(n, dtype=np.bool_)
    pivot_lows = np.zeros(n, dtype=np.bool_)
    
    for i in prange(depth, n - depth):
        # Check for pivot high
        is_high = True
        for j in range(1, depth + 1):
            if high[i] <= high[i - j] or high[i] <= high[i + j]:
                is_high = False
                break
        
        if is_high:
            pivot_highs[i] = True
        
        # Check for pivot low
        is_low = True
        for j in range(1, depth + 1):
            if low[i] >= low[i - j] or low[i] >= low[i + j]:
                is_low = False
                break
        
        if is_low:
            pivot_lows[i] = True
    
    return pivot_highs, pivot_lows


@jit(nopython=True)
def lorentzian_distance_batch(features1: np.ndarray, features2: np.ndarray) -> np.ndarray:
    """Calculate Lorentzian distances for batch of features"""
    n_samples = features2.shape[0]
    distances = np.zeros(n_samples)
    
    for i in range(n_samples):
        dist = 0.0
        for j in range(features1.shape[0]):
            dist += np.log(1 + abs(features1[j] - features2[i, j]))
        distances[i] = dist
    
    return distances


class OptimizedTrendVanguard:
    """Optimized Trend Vanguard Strategy"""
    
    def __init__(self, 
                 # ZigZag parameters
                 zigzag_depth: int = 12,  # Reduced for performance
                 zigzag_deviation: float = 6.0,
                 zigzag_backstep: int = 4,
                 # k-NN parameters
                 knn_lookback: int = 50,  # Reduced for performance
                 k_value: int = 5,  # Reduced for performance
                 confidence_threshold: float = 0.55,
                 # Risk management
                 risk_per_trade: float = 0.005,
                 atr_multiplier: float = 1.5,
                 # Pattern recognition
                 pattern_lookback: int = 30,  # Reduced for performance
                 min_pattern_bars: int = 10,
                 # Performance options
                 use_cache: bool = True,
                 max_bars: int = 1000):  # Process only last N bars
        """Initialize with optimized parameters"""
        self.zigzag_depth = zigzag_depth
        self.zigzag_deviation = zigzag_deviation
        self.zigzag_backstep = zigzag_backstep
        self.knn_lookback = knn_lookback
        self.k_value = k_value
        self.confidence_threshold = confidence_threshold
        self.risk_per_trade = risk_per_trade
        self.atr_multiplier = atr_multiplier
        self.pattern_lookback = pattern_lookback
        self.min_pattern_bars = min_pattern_bars
        self.use_cache = use_cache
        self.max_bars = max_bars
        
        # Cache
        self._pivot_cache = {}
        self._feature_cache = {}
    
    def calculate_atr_vectorized(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Vectorized ATR calculation"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # True Range calculation
        high_low = high - low
        high_close = np.abs(np.roll(close, 1) - high)
        low_close = np.abs(np.roll(close, 1) - low)
        
        # Stack and take max
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        tr[0] = high_low[0]  # First value doesn't have previous close
        
        # Fast moving average using convolution
        kernel = np.ones(period) / period
        atr = np.convolve(tr, kernel, mode='same')
        
        # Adjust edges
        for i in range(period):
            if i < len(tr):
                atr[i] = np.mean(tr[:i+1])
        
        return pd.Series(atr, index=df.index)
    
    def calculate_zigzag_optimized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized ZigZag calculation"""
        # Check cache
        cache_key = f"{len(df)}_{df.index[-1]}"
        if self.use_cache and cache_key in self._pivot_cache:
            return self._pivot_cache[cache_key]
        
        # Use numba function
        high = df['high'].values
        low = df['low'].values
        
        pivot_highs, pivot_lows = find_pivot_points_numba(high, low, self.zigzag_depth)
        
        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        result['zigzag'] = np.nan
        result['is_pivot_high'] = pivot_highs
        result['is_pivot_low'] = pivot_lows
        
        # Fill zigzag values
        result.loc[pivot_highs, 'zigzag'] = df.loc[pivot_highs, 'high']
        result.loc[pivot_lows, 'zigzag'] = df.loc[pivot_lows, 'low']
        
        # Cache result
        if self.use_cache:
            self._pivot_cache[cache_key] = result
        
        return result
    
    def extract_pattern_features_vectorized(self, df: pd.DataFrame, indices: List[int],
                                          zigzag_data: pd.DataFrame) -> np.ndarray:
        """Extract features for multiple indices at once"""
        n_features = 10
        features = np.zeros((len(indices), n_features))
        
        for idx, i in enumerate(indices):
            if i < self.pattern_lookback:
                continue
                
            # Find pivot points in lookback window
            window_start = max(0, i - self.pattern_lookback)
            window = zigzag_data.iloc[window_start:i]
            
            pivot_mask = window['is_pivot_high'] | window['is_pivot_low']
            pivot_indices = window.index[pivot_mask].tolist()
            
            if len(pivot_indices) < 4:
                continue
            
            # Use last 4 pivots
            last_pivots = pivot_indices[-4:]
            
            try:
                # Feature 1: Last pivot type
                last_pivot_idx = last_pivots[-1]
                features[idx, 0] = 1 if zigzag_data.loc[last_pivot_idx, 'is_pivot_high'] else -1
                
                # Features 2-5: Normalized pivot levels
                current_price = df['close'].iloc[i]
                for j, pivot_idx in enumerate(last_pivots):
                    pivot_value = zigzag_data.loc[pivot_idx, 'zigzag']
                    features[idx, 1 + j] = (pivot_value - current_price) / current_price
                
                # Features 6-8: Distances between pivots
                pivot_positions = [df.index.get_loc(idx) for idx in last_pivots]
                for j in range(len(pivot_positions) - 1):
                    distance = pivot_positions[j + 1] - pivot_positions[j]
                    features[idx, 5 + j] = distance / self.pattern_lookback
                
                # Feature 9: Overall trend
                first_value = zigzag_data.loc[last_pivots[0], 'zigzag']
                last_value = zigzag_data.loc[last_pivots[-1], 'zigzag']
                features[idx, 9] = (last_value - first_value) / first_value
                
            except Exception:
                pass
        
        return features
    
    def predict_batch_knn(self, df: pd.DataFrame, start_idx: int, end_idx: int,
                         zigzag_data: pd.DataFrame) -> List[Tuple[float, float]]:
        """Batch k-NN predictions"""
        predictions = []
        
        # Extract features for all indices at once
        current_indices = list(range(start_idx, end_idx))
        current_features = self.extract_pattern_features_vectorized(df, current_indices, zigzag_data)
        
        # Historical indices
        hist_start = max(0, start_idx - self.knn_lookback)
        hist_end = start_idx - 10
        
        if hist_end - hist_start < self.k_value:
            return [(0, 0) for _ in current_indices]
        
        historical_indices = list(range(hist_start, hist_end))
        historical_features = self.extract_pattern_features_vectorized(df, historical_indices, zigzag_data)
        
        # Calculate future returns for historical data
        close_prices = df['close'].values
        historical_returns = np.zeros(len(historical_indices))
        
        for i, idx in enumerate(historical_indices):
            if idx + 10 < len(close_prices):
                historical_returns[i] = (close_prices[idx + 10] - close_prices[idx]) / close_prices[idx]
        
        # For each current index, find k nearest neighbors
        for i, current_feat in enumerate(current_features):
            if np.all(current_feat == 0):
                predictions.append((0, 0))
                continue
            
            # Calculate distances
            distances = lorentzian_distance_batch(current_feat, historical_features)
            
            # Find k nearest
            k_nearest_idx = np.argpartition(distances, min(self.k_value, len(distances)-1))[:self.k_value]
            k_distances = distances[k_nearest_idx]
            k_returns = historical_returns[k_nearest_idx]
            
            # Weighted prediction
            weights = 1.0 / (k_distances + 1e-10)
            prediction = np.sum(weights * k_returns) / np.sum(weights)
            
            # Confidence based on consistency
            confidence = 1.0 / (1.0 + np.std(k_returns))
            
            predictions.append((prediction, confidence))
        
        return predictions
    
    def calculate_regime_filter_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Vectorized market regime filter"""
        close = df['close']
        
        # Fast moving averages
        sma_20 = close.rolling(window=20, min_periods=1).mean()
        sma_50 = close.rolling(window=50, min_periods=1).mean()
        sma_200 = close.rolling(window=200, min_periods=1).mean()
        
        # Vectorized regime detection
        regime = pd.Series('neutral', index=df.index)
        
        bullish_mask = (sma_20 > sma_50) & (sma_50 > sma_200) & (close > sma_20)
        bearish_mask = (sma_20 < sma_50) & (sma_50 < sma_200) & (close < sma_20)
        
        regime[bullish_mask] = 'bullish'
        regime[bearish_mask] = 'bearish'
        
        return regime
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimized Trend Vanguard calculation"""
        try:
            # Limit data size for performance
            if len(df) > self.max_bars:
                df = df.iloc[-self.max_bars:].copy()
            else:
                df = df.copy()
            
            logger.info(f"Processing {len(df)} bars")
            
            # Vectorized ATR
            atr = self.calculate_atr_vectorized(df)
            
            # Optimized ZigZag
            zigzag_data = self.calculate_zigzag_optimized(df)
            
            # Vectorized regime filter
            market_regime = self.calculate_regime_filter_vectorized(df)
            
            # Initialize result columns
            result = pd.DataFrame(index=df.index)
            result['atr'] = atr
            result['market_regime'] = market_regime
            result['signal'] = 0
            result['strength'] = 0.5
            result['prediction'] = 0.0
            result['confidence'] = 0.0
            
            # Batch predictions
            batch_size = 50
            min_idx = self.knn_lookback + 10
            
            for i in range(min_idx, len(df), batch_size):
                start_idx = i
                end_idx = min(i + batch_size, len(df))
                
                batch_predictions = self.predict_batch_knn(df, start_idx, end_idx, zigzag_data)
                
                for j, (pred, conf) in enumerate(batch_predictions):
                    idx = start_idx + j
                    result.iloc[idx, result.columns.get_loc('prediction')] = pred
                    result.iloc[idx, result.columns.get_loc('confidence')] = conf
                    
                    # Generate signals
                    if conf >= self.confidence_threshold:
                        if pred > 0.01:
                            result.iloc[idx, result.columns.get_loc('signal')] = 1
                            result.iloc[idx, result.columns.get_loc('strength')] = min(0.5 + pred * 10, 1.0)
                        elif pred < -0.01:
                            result.iloc[idx, result.columns.get_loc('signal')] = -1
                            result.iloc[idx, result.columns.get_loc('strength')] = min(0.5 + abs(pred) * 10, 1.0)
            
            # Add ZigZag data
            result['zigzag'] = zigzag_data['zigzag']
            result['is_pivot_high'] = zigzag_data['is_pivot_high']
            result['is_pivot_low'] = zigzag_data['is_pivot_low']
            
            # Risk management
            result['position_size'] = self.risk_per_trade / (atr / df['close'])
            result['position_size'] = result['position_size'].clip(0, 0.1)
            
            # Simple filtering
            sma_fast = df['close'].rolling(20).mean()
            sma_slow = df['close'].rolling(50).mean()
            
            bullish_filter = sma_fast > sma_slow
            bearish_filter = sma_fast < sma_slow
            
            result.loc[(result['signal'] == 1) & ~bullish_filter, 'signal'] = 0
            result.loc[(result['signal'] == -1) & ~bearish_filter, 'signal'] = 0
            
            logger.info(f"Trend Vanguard calculation completed")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in optimized Trend Vanguard: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_signal(self, latest_values: Dict[str, Any]) -> str:
        """Get signal from latest values"""
        try:
            signal = latest_values.get('signal', 0)
            confidence = latest_values.get('confidence', 0)
            strength = latest_values.get('strength', 0.5)
            regime = latest_values.get('market_regime', 'neutral')
            
            if signal == 1:
                if confidence > 0.7 and strength > 0.7:
                    return 'STRONG_BUY'
                else:
                    return 'BUY'
            elif signal == -1:
                if confidence > 0.7 and strength > 0.7:
                    return 'STRONG_SELL'
                else:
                    return 'SELL'
            
            # Check pivot signals
            if latest_values.get('is_pivot_low', False):
                return 'POTENTIAL_BOTTOM'
            elif latest_values.get('is_pivot_high', False):
                return 'POTENTIAL_TOP'
            
            return 'NEUTRAL'
            
        except Exception as e:
            logger.error(f"Error getting signal: {e}")
            return 'ERROR'


# For backward compatibility
TrendVanguard = OptimizedTrendVanguard