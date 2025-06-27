#!/usr/bin/env python3
"""
Advanced Feature Engineering for ML Trading System
Gelişmiş özellik mühendisliği
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from loguru import logger
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureEngineering:
    """Advanced feature engineering for trading ML models"""
    
    def __init__(self):
        self.feature_stats = {}
        self.pca_models = {}
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced price-based features"""
        
        # Log returns for different periods
        for period in [1, 2, 3, 5, 10, 20, 50]:
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        
        # Price position in range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Volatility features
        df['volatility_5'] = df['log_return_1'].rolling(5).std()
        df['volatility_20'] = df['log_return_1'].rolling(20).std()
        df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
        
        # Volatility regime
        df['volatility_percentile'] = df['volatility_20'].rolling(252).rank(pct=True)
        
        # Price acceleration
        df['price_acceleration'] = df['close'].diff().diff()
        
        # Distance from high/low
        df['distance_from_high_20'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['distance_from_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
        
        # Price efficiency
        df['efficiency_ratio'] = self.calculate_efficiency_ratio(df['close'], 10)
        
        # Fractal dimension
        df['fractal_dimension'] = self.calculate_fractal_dimension(df['close'], 20)
        
        return df
    
    def create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced volume features"""
        
        # Volume moving averages
        df['volume_sma_5'] = df['volume'].rolling(5).mean()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        
        # Volume ratios
        df['volume_ratio_5_20'] = df['volume_sma_5'] / (df['volume_sma_20'] + 1e-10)
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        df['obv_divergence'] = df['obv'] - df['obv_sma']
        
        # Volume-Price Trend (VPT)
        df['vpt'] = (df['close'].pct_change() * df['volume']).cumsum()
        
        # Accumulation/Distribution Line
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        df['ad_line'] = (clv * df['volume']).cumsum()
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(10)
        
        # Price-Volume correlation
        df['pv_correlation'] = df['close'].rolling(20).corr(df['volume'])
        
        # Volume spikes
        volume_mean = df['volume'].rolling(20).mean()
        volume_std = df['volume'].rolling(20).std()
        df['volume_spike'] = (df['volume'] - volume_mean) / (volume_std + 1e-10)
        
        return df
    
    def create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        
        # Spread
        df['spread'] = df['high'] - df['low']
        df['spread_pct'] = df['spread'] / df['close']
        df['avg_spread_20'] = df['spread_pct'].rolling(20).mean()
        
        # True Range (simplified)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        df['true_range'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = df['true_range'].rolling(14).mean()
        df['atr_pct'] = df['atr_14'] / df['close']
        
        # Garman-Klass volatility
        df['gk_volatility'] = np.sqrt(
            (0.5 * np.log(df['high'] / df['low']) ** 2 - 
             (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2).rolling(20).mean()
        )
        
        # Parkinson volatility
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df['high'] / df['low']) ** 2).rolling(20).mean()
        )
        
        # Close location value (CLV)
        df['clv'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        
        # Money Flow Index components
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        mfi_ratio = positive_flow.rolling(14).sum() / (negative_flow.rolling(14).sum() + 1e-10)
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        return df
    
    def create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create pattern recognition features"""
        
        # Japanese candlestick patterns
        df['doji'] = (abs(df['close'] - df['open']) < (df['high'] - df['low']) * 0.1).astype(int)
        
        # Hammer
        body = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        df['hammer'] = ((lower_shadow > 2 * body) & (df['close'] > df['open'])).astype(int)
        
        # Shooting star
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        df['shooting_star'] = ((upper_shadow > 2 * body) & (df['close'] < df['open'])).astype(int)
        
        # Engulfing patterns
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) & 
            (df['close'].shift() < df['open'].shift()) &
            (df['open'] < df['close'].shift()) &
            (df['close'] > df['open'].shift())
        ).astype(int)
        
        df['bearish_engulfing'] = (
            (df['close'] < df['open']) & 
            (df['close'].shift() > df['open'].shift()) &
            (df['open'] > df['close'].shift()) &
            (df['close'] < df['open'].shift())
        ).astype(int)
        
        # Support/Resistance levels
        df['pivot_point'] = (df['high'] + df['low'] + df['close']) / 3
        df['resistance_1'] = 2 * df['pivot_point'] - df['low']
        df['support_1'] = 2 * df['pivot_point'] - df['high']
        
        # Distance to pivot levels
        df['distance_to_pivot'] = (df['close'] - df['pivot_point']) / df['close']
        df['distance_to_resistance'] = (df['resistance_1'] - df['close']) / df['close']
        df['distance_to_support'] = (df['close'] - df['support_1']) / df['close']
        
        return df
    
    def create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features"""
        
        # Rolling statistics
        for window in [5, 10, 20]:
            # Skewness
            df[f'return_skew_{window}'] = df['log_return_1'].rolling(window).skew()
            
            # Kurtosis
            df[f'return_kurt_{window}'] = df['log_return_1'].rolling(window).kurt()
            
            # Entropy
            df[f'price_entropy_{window}'] = df['close'].rolling(window).apply(
                lambda x: stats.entropy(np.histogram(x, bins=10)[0] + 1e-10)
            )
        
        # Autocorrelation
        df['return_autocorr_1'] = df['log_return_1'].rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
        )
        
        # Hurst exponent
        df['hurst_exponent'] = df['close'].rolling(50).apply(
            lambda x: self.calculate_hurst_exponent(x) if len(x) == 50 else 0.5
        )
        
        # Z-score of price
        price_mean = df['close'].rolling(20).mean()
        price_std = df['close'].rolling(20).std()
        df['price_zscore'] = (df['close'] - price_mean) / (price_std + 1e-10)
        
        return df
    
    def create_indicator_combinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create combinations of technical indicators"""
        
        # Moving average features
        for short, long in [(5, 20), (10, 30), (20, 50)]:
            ma_short = df['close'].rolling(short).mean()
            ma_long = df['close'].rolling(long).mean()
            
            # Crossover
            df[f'ma_cross_{short}_{long}'] = (ma_short > ma_long).astype(int)
            
            # Distance
            df[f'ma_spread_{short}_{long}'] = (ma_short - ma_long) / ma_long
            
            # Convergence rate
            df[f'ma_conv_rate_{short}_{long}'] = df[f'ma_spread_{short}_{long}'].diff()
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma = df['close'].rolling(bb_period).mean()
        std = df['close'].rolling(bb_period).std()
        
        df['bb_upper'] = sma + (bb_std * std)
        df['bb_lower'] = sma - (bb_std * std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # RSI divergence
        df['rsi_14'] = self.calculate_rsi(df['close'], 14)
        df['rsi_price_divergence'] = df['rsi_14'].pct_change(5) - df['close'].pct_change(5)
        
        # MACD features
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_line'] = exp1 - exp2
        df['macd_signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_histogram_calc'] = df['macd_line'] - df['macd_signal_line']
        df['macd_histogram_slope'] = df['macd_histogram_calc'].diff()
        
        return df
    
    def create_ml_predictions_as_features(self, df: pd.DataFrame, 
                                        indicator_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Use ML indicator predictions as features"""
        
        # Lorentzian features
        if 'lorentzian' in indicator_data:
            lor_data = indicator_data['lorentzian']
            if 'signal' in lor_data.columns:
                df['lor_signal'] = lor_data['signal'].reindex(df.index, fill_value=0)
            if 'confidence' in lor_data.columns:
                df['lor_confidence'] = lor_data['confidence'].reindex(df.index, fill_value=0.5)
            if 'prediction' in lor_data.columns:
                df['lor_prediction'] = lor_data['prediction'].reindex(df.index, fill_value=0)
        
        # Trend Vanguard features
        if 'trend_vanguard' in indicator_data:
            tv_data = indicator_data['trend_vanguard']
            if 'signal' in tv_data.columns:
                df['tv_signal'] = tv_data['signal'].reindex(df.index, fill_value=0)
            if 'strength' in tv_data.columns:
                df['tv_strength'] = tv_data['strength'].reindex(df.index, fill_value=0)
            if 'confidence' in tv_data.columns:
                df['tv_confidence'] = tv_data['confidence'].reindex(df.index, fill_value=0.5)
            if 'market_regime' in tv_data.columns:
                # Encode market regime
                regime_map = {'bullish': 1, 'bearish': -1, 'neutral': 0}
                df['tv_regime'] = tv_data['market_regime'].map(regime_map).reindex(df.index, fill_value=0)
        
        # Combined ML signals
        if 'lor_signal' in df.columns and 'tv_signal' in df.columns:
            df['ml_signal_agreement'] = (df['lor_signal'] == df['tv_signal']).astype(int)
            df['ml_signal_strength'] = (df['lor_signal'] + df['tv_signal']) / 2
            df['ml_confidence_avg'] = (df.get('lor_confidence', 0.5) + df.get('tv_confidence', 0.5)) / 2
        
        return df
    
    def create_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market regime features"""
        
        # Trend strength
        df['trend_strength_short'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        df['trend_strength_medium'] = (df['close'] - df['close'].shift(30)) / df['close'].shift(30)
        df['trend_strength_long'] = (df['close'] - df['close'].shift(60)) / df['close'].shift(60)
        
        # Trend consistency
        df['up_days_pct_10'] = (df['close'] > df['close'].shift()).rolling(10).mean()
        df['up_days_pct_20'] = (df['close'] > df['close'].shift()).rolling(20).mean()
        
        # Volatility regime
        vol_percentile = df['volatility_20'].rolling(252).rank(pct=True)
        df['high_vol_regime'] = (vol_percentile > 0.7).astype(int)
        df['low_vol_regime'] = (vol_percentile < 0.3).astype(int)
        
        # Trend regime
        sma_50 = df['close'].rolling(50).mean()
        sma_200 = df['close'].rolling(200).mean()
        df['bull_market'] = ((df['close'] > sma_50) & (sma_50 > sma_200)).astype(int)
        df['bear_market'] = ((df['close'] < sma_50) & (sma_50 < sma_200)).astype(int)
        
        # Market breadth (if we had sector data)
        # For now, use rolling correlation with index as proxy
        df['market_correlation'] = df['close'].pct_change().rolling(20).corr(
            df['volume'].pct_change()  # Simplified proxy
        )
        
        return df
    
    def calculate_efficiency_ratio(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Kaufman's Efficiency Ratio"""
        direction = abs(prices - prices.shift(period))
        volatility = prices.diff().abs().rolling(period).sum()
        return direction / (volatility + 1e-10)
    
    def calculate_fractal_dimension(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate fractal dimension using box-counting method"""
        def fractal_dim(x):
            if len(x) < period:
                return 1.5
            n = len(x)
            max_val = x.max()
            min_val = x.min()
            if max_val == min_val:
                return 1.5
            
            # Normalize
            normalized = (x - min_val) / (max_val - min_val)
            
            # Box counting
            boxes = []
            for scale in [2, 4, 8, 16]:
                if scale > n:
                    continue
                box_size = 1.0 / scale
                box_count = 0
                
                for i in range(0, n, n//scale):
                    segment = normalized.iloc[i:i+n//scale]
                    if len(segment) > 0:
                        box_count += np.ceil((segment.max() - segment.min()) / box_size)
                
                if box_count > 0:
                    boxes.append((np.log(scale), np.log(box_count)))
            
            if len(boxes) < 2:
                return 1.5
            
            # Linear regression
            scales = np.array([b[0] for b in boxes])
            counts = np.array([b[1] for b in boxes])
            
            slope = np.polyfit(scales, counts, 1)[0]
            return abs(slope)
        
        return prices.rolling(period).apply(fractal_dim)
    
    def calculate_hurst_exponent(self, prices: np.ndarray) -> float:
        """Calculate Hurst exponent"""
        if len(prices) < 20:
            return 0.5
        
        # Create range series
        lags = range(2, min(20, len(prices)//2))
        tau = []
        
        for lag in lags:
            # Calculate standard deviation of differenced series
            std_dev = np.std(np.subtract(prices[lag:], prices[:-lag]))
            tau.append(std_dev)
        
        # Perform linear regression
        if len(tau) < 2:
            return 0.5
        
        reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return reg[0]
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def reduce_dimensionality(self, df: pd.DataFrame, n_components: int = 50) -> pd.DataFrame:
        """Reduce feature dimensionality using PCA"""
        
        # Select numerical features
        feature_cols = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] 
                       and col not in ['open', 'high', 'low', 'close', 'volume']]
        
        if len(feature_cols) < n_components:
            return df
        
        # Prepare data
        X = df[feature_cols].fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create PCA features
        pca_df = pd.DataFrame(
            X_pca,
            columns=[f'pca_{i}' for i in range(n_components)],
            index=df.index
        )
        
        # Add to original dataframe
        df = pd.concat([df, pca_df], axis=1)
        
        # Store PCA model and explained variance
        self.pca_models['main'] = pca
        logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_[:10]}")
        logger.info(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
        
        return df
    
    def engineer_all_features(self, df: pd.DataFrame, 
                            indicator_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Apply all feature engineering"""
        
        logger.info("Starting feature engineering...")
        
        # Basic features
        df = self.create_price_features(df)
        logger.info(f"Price features created. Total features: {len(df.columns)}")
        
        df = self.create_volume_features(df)
        logger.info(f"Volume features created. Total features: {len(df.columns)}")
        
        df = self.create_microstructure_features(df)
        logger.info(f"Microstructure features created. Total features: {len(df.columns)}")
        
        df = self.create_pattern_features(df)
        logger.info(f"Pattern features created. Total features: {len(df.columns)}")
        
        df = self.create_statistical_features(df)
        logger.info(f"Statistical features created. Total features: {len(df.columns)}")
        
        df = self.create_indicator_combinations(df)
        logger.info(f"Indicator combinations created. Total features: {len(df.columns)}")
        
        # ML predictions as features
        if indicator_data:
            df = self.create_ml_predictions_as_features(df, indicator_data)
            logger.info(f"ML features created. Total features: {len(df.columns)}")
        
        df = self.create_market_regime_features(df)
        logger.info(f"Market regime features created. Total features: {len(df.columns)}")
        
        # Optional: Reduce dimensionality
        # df = self.reduce_dimensionality(df)
        
        # Store feature statistics
        self.feature_stats = {
            'total_features': len(df.columns),
            'price_features': len([c for c in df.columns if 'price' in c or 'return' in c]),
            'volume_features': len([c for c in df.columns if 'volume' in c or 'obv' in c]),
            'statistical_features': len([c for c in df.columns if 'skew' in c or 'kurt' in c or 'entropy' in c]),
            'ml_features': len([c for c in df.columns if 'lor_' in c or 'tv_' in c])
        }
        
        logger.info(f"Feature engineering complete. Final features: {len(df.columns)}")
        
        return df


def main():
    """Test feature engineering"""
    from utils.csv_data_manager import CSVDataManager
    
    # Initialize
    csv_manager = CSVDataManager()
    fe = AdvancedFeatureEngineering()
    
    # Load sample data
    symbol = 'THYAO'
    timeframe = '1h'
    
    df = csv_manager.load_raw_data(symbol, timeframe)
    if df is None:
        logger.error("Could not load data")
        return
    
    # Load indicator data
    indicator_data = {
        'lorentzian': csv_manager.load_indicator_data(symbol, timeframe, 'lorentzian'),
        'trend_vanguard': csv_manager.load_indicator_data(symbol, timeframe, 'trend_vanguard')
    }
    
    # Apply feature engineering
    df_features = fe.engineer_all_features(df.copy(), indicator_data)
    
    # Show results
    print(f"\nOriginal columns: {len(df.columns)}")
    print(f"After feature engineering: {len(df_features.columns)}")
    print(f"\nFeature statistics: {fe.feature_stats}")
    
    # Sample features
    print("\nSample features:")
    feature_samples = [
        'log_return_5', 'volatility_ratio', 'efficiency_ratio',
        'volume_spike', 'mfi', 'price_entropy_20', 'bb_position',
        'ml_signal_strength', 'bull_market'
    ]
    
    for feat in feature_samples:
        if feat in df_features.columns:
            print(f"{feat}: mean={df_features[feat].mean():.4f}, std={df_features[feat].std():.4f}")
    
    # Save engineered features
    output_file = f"ml_models/features/{symbol}_{timeframe}_features.csv"
    df_features.to_csv(output_file)
    logger.info(f"Features saved to {output_file}")


if __name__ == "__main__":
    main()