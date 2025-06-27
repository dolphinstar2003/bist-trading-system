#!/usr/bin/env python3
"""
Feature Engineering for ML Models
Teknik indikatörleri ve fiyat verilerini ML için özellik setine dönüştürür
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.csv_data_manager import CSVDataManager
from indicators.indicator_calculator import IndicatorCalculator


class FeatureEngineering:
    """ML için özellik mühendisliği"""
    
    def __init__(self):
        self.csv_manager = CSVDataManager()
        self.indicator_calculator = IndicatorCalculator()
        
        # Özellik grupları
        self.price_features = ['return_1', 'return_5', 'return_10', 'return_20',
                              'high_low_ratio', 'close_open_ratio', 'volume_ratio']
        
        self.technical_features = ['rsi_14', 'macd_signal', 'bb_position', 
                                  'atr_ratio', 'adx_value', 'stoch_k']
        
        self.pattern_features = ['support_distance', 'resistance_distance',
                               'trend_strength', 'volatility_rank']
        
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fiyat tabanlı özellikler oluştur"""
        features = pd.DataFrame(index=df.index)
        
        # Getiri özellikleri
        for period in [1, 5, 10, 20]:
            features[f'return_{period}'] = df['close'].pct_change(period)
            features[f'return_{period}_abs'] = features[f'return_{period}'].abs()
        
        # Fiyat oranları
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['close_open_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Volume özellikleri
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_change'] = df['volume'].pct_change()
        
        # Momentum özellikleri
        features['price_momentum'] = df['close'] - df['close'].shift(10)
        features['price_acceleration'] = features['price_momentum'].diff()
        
        # Volatilite özellikleri
        features['volatility_20'] = df['close'].pct_change().rolling(20).std()
        features['volatility_ratio'] = features['volatility_20'] / features['volatility_20'].rolling(60).mean()
        
        # Trend özellikleri
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        features['trend_sma_20'] = (df['close'] - sma_20) / sma_20
        features['trend_sma_50'] = (df['close'] - sma_50) / sma_50
        features['sma_20_50_ratio'] = sma_20 / sma_50
        
        # Price levels
        features['close_to_high_20'] = df['close'] / df['high'].rolling(20).max()
        features['close_to_low_20'] = df['close'] / df['low'].rolling(20).min()
        
        return features
    
    def create_indicator_features(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """İndikatör tabanlı özellikler"""
        features = pd.DataFrame()
        
        # Her indikatör için özellikleri al (string kolonları hariç tut)
        indicator_mapping = {
            'williams_vix_fix': ['wvf', 'vix_fix_signal'],
            'wavetrend': ['wt1', 'wt2', 'wt_buy_signal', 'wt_sell_signal'],
            'squeeze_momentum': ['squeeze_on', 'momentum', 'momentum_increasing'],
            'adx_di': ['adx', 'plus_di', 'minus_di', 'adx_above_threshold'],  # trend_strength string olduğu için çıkardık
            'supertrend': ['trend', 'buy_signal', 'sell_signal'],
            'macd': ['macd', 'macd_signal', 'macd_hist', 'macd_cross_up'],
            'lorentzian': ['prediction', 'signal_strength', 'is_bullish', 'is_bearish'],
            'trend_vanguard': ['knn_prediction', 'knn_confidence', 'signal', 'signal_strength', 'filtered_signal']
        }
        
        for indicator_name, columns in indicator_mapping.items():
            try:
                # İndikatör verisini yükle
                indicator_data = self.indicator_calculator.get_indicator_data(
                    symbol, timeframe, indicator_name
                )
                
                if indicator_data is not None:
                    # Seçili sütunları al
                    for col in columns:
                        if col in indicator_data.columns:
                            feature_name = f'{indicator_name}_{col}'
                            features[feature_name] = indicator_data[col]
                            
            except Exception as e:
                logger.warning(f"Could not load {indicator_name}: {e}")
        
        return features
    
    def create_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market rejimi özellikleri"""
        features = pd.DataFrame(index=df.index)
        
        # Trend rejimi
        sma_50 = df['close'].rolling(50).mean()
        sma_200 = df['close'].rolling(200).mean()
        
        features['regime_bullish'] = ((df['close'] > sma_50) & (sma_50 > sma_200)).astype(int)
        features['regime_bearish'] = ((df['close'] < sma_50) & (sma_50 < sma_200)).astype(int)
        features['regime_neutral'] = (~features['regime_bullish'] & ~features['regime_bearish']).astype(int)
        
        # Volatilite rejimi
        volatility = df['close'].pct_change().rolling(20).std()
        vol_percentile = volatility.rolling(252).rank(pct=True)
        
        features['vol_regime_low'] = (vol_percentile < 0.25).astype(int)
        features['vol_regime_normal'] = ((vol_percentile >= 0.25) & (vol_percentile < 0.75)).astype(int)
        features['vol_regime_high'] = (vol_percentile >= 0.75).astype(int)
        
        # Volume rejimi
        volume_sma = df['volume'].rolling(20).mean()
        volume_ratio = df['volume'] / volume_sma
        
        features['volume_regime_low'] = (volume_ratio < 0.8).astype(int)
        features['volume_regime_normal'] = ((volume_ratio >= 0.8) & (volume_ratio <= 1.2)).astype(int)
        features['volume_regime_high'] = (volume_ratio > 1.2).astype(int)
        
        return features
    
    def create_target_variables(self, df: pd.DataFrame, 
                              target_periods: List[int] = [1, 5, 20]) -> pd.DataFrame:
        """Hedef değişkenleri oluştur"""
        targets = pd.DataFrame(index=df.index)
        
        for period in target_periods:
            # Gelecek getiri
            future_return = df['close'].shift(-period) / df['close'] - 1
            
            # Regresyon hedefi (sürekli)
            targets[f'target_return_{period}'] = future_return
            
            # Sınıflandırma hedefleri
            # Binary: Yukarı/Aşağı
            targets[f'target_direction_{period}'] = (future_return > 0).astype(int)
            
            # Multi-class: Güçlü Düşüş, Düşüş, Nötr, Yükseliş, Güçlü Yükseliş
            conditions = [
                future_return < -0.02,
                (future_return >= -0.02) & (future_return < -0.005),
                (future_return >= -0.005) & (future_return <= 0.005),
                (future_return > 0.005) & (future_return <= 0.02),
                future_return > 0.02
            ]
            choices = [0, 1, 2, 3, 4]  # Sınıf etiketleri
            targets[f'target_class_{period}'] = np.select(conditions, choices, default=2)
            
            # Risk-adjusted hedef (Sharpe benzeri)
            returns_std = df['close'].pct_change().rolling(period).std()
            targets[f'target_sharpe_{period}'] = future_return / (returns_std + 1e-10)
        
        return targets
    
    def create_feature_matrix(self, symbol: str, timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Tam özellik matrisi oluştur"""
        logger.info(f"Creating feature matrix for {symbol} {timeframe}")
        
        # Ham veriyi yükle
        df = self.csv_manager.load_raw_data(symbol, timeframe)
        if df is None or len(df) < 200:
            logger.error(f"Insufficient data for {symbol} {timeframe}")
            return pd.DataFrame(), pd.DataFrame()
        
        # Özellik gruplarını oluştur
        price_features = self.create_price_features(df)
        indicator_features = self.create_indicator_features(symbol, timeframe)
        regime_features = self.create_market_regime_features(df)
        
        # Tüm özellikleri birleştir
        all_features = pd.concat([price_features, indicator_features, regime_features], axis=1)
        
        # Hedef değişkenleri oluştur
        targets = self.create_target_variables(df)
        
        # NaN değerleri temizle
        # İlk N satırı at (moving average'lar için)
        min_periods = 200
        all_features = all_features.iloc[min_periods:]
        targets = targets.iloc[min_periods:]
        
        # Kalan NaN'ları doldur
        all_features = all_features.fillna(method='ffill').fillna(0)
        targets = targets.fillna(0)
        
        # Kategorik sütunları kontrol et ve dönüştür
        for col in all_features.columns:
            if all_features[col].dtype == 'object':
                # String değerleri numerik değerlere dönüştür
                if 'trend' in col.lower():
                    # Trend sütunları için mapping
                    all_features[col] = all_features[col].map({
                        'uptrend': 1,
                        'downtrend': -1,
                        'no_trend': 0
                    }).fillna(0)
                else:
                    # Diğer kategorik sütunlar için
                    try:
                        all_features[col] = pd.to_numeric(all_features[col], errors='coerce').fillna(0)
                    except:
                        # Label encoding
                        all_features[col] = pd.Categorical(all_features[col]).codes
        
        # Özellik normalizasyonu (opsiyonel - model tipine göre)
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # features_scaled = pd.DataFrame(
        #     scaler.fit_transform(all_features),
        #     index=all_features.index,
        #     columns=all_features.columns
        # )
        
        logger.info(f"Feature matrix created: {all_features.shape[0]} samples, {all_features.shape[1]} features")
        
        return all_features, targets
    
    def get_feature_importance(self, features: pd.DataFrame, target: pd.Series, 
                             method: str = 'correlation') -> pd.Series:
        """Özellik önemini hesapla"""
        if method == 'correlation':
            # Basit korelasyon analizi
            importance = features.corrwith(target).abs().sort_values(ascending=False)
        
        elif method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_regression
            importance = pd.Series(
                mutual_info_regression(features, target),
                index=features.columns
            ).sort_values(ascending=False)
        
        elif method == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(features, target)
            importance = pd.Series(
                rf.feature_importances_,
                index=features.columns
            ).sort_values(ascending=False)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return importance
    
    def save_features(self, features: pd.DataFrame, targets: pd.DataFrame,
                     symbol: str, timeframe: str):
        """Özellikleri kaydet"""
        output_dir = Path("data/processed")
        output_dir.mkdir(exist_ok=True)
        
        # Features
        features_file = output_dir / f"{symbol}_{timeframe}_features.parquet"
        features.to_parquet(features_file, compression='snappy')
        
        # Targets
        targets_file = output_dir / f"{symbol}_{timeframe}_targets.parquet"
        targets.to_parquet(targets_file, compression='snappy')
        
        logger.info(f"Features saved: {features_file}")
        logger.info(f"Targets saved: {targets_file}")
    
    def load_features(self, symbol: str, timeframe: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Kaydedilmiş özellikleri yükle"""
        output_dir = Path("data/processed")
        
        features_file = output_dir / f"{symbol}_{timeframe}_features.parquet"
        targets_file = output_dir / f"{symbol}_{timeframe}_targets.parquet"
        
        if features_file.exists() and targets_file.exists():
            features = pd.read_parquet(features_file)
            targets = pd.read_parquet(targets_file)
            return features, targets
        else:
            logger.warning(f"Feature files not found for {symbol} {timeframe}")
            return pd.DataFrame(), pd.DataFrame()


def main():
    """Test feature engineering"""
    fe = FeatureEngineering()
    
    # Test için bir sembol
    symbol = "AKBNK"
    timeframe = "1h"
    
    # Özellik matrisi oluştur
    features, targets = fe.create_feature_matrix(symbol, timeframe)
    
    if not features.empty:
        print(f"\nFeature Matrix Shape: {features.shape}")
        print(f"Target Matrix Shape: {targets.shape}")
        
        print("\nFeature Columns:")
        for i, col in enumerate(features.columns[:20]):
            print(f"  {i+1}. {col}")
        print(f"  ... and {len(features.columns) - 20} more features")
        
        print("\nTarget Columns:")
        for col in targets.columns:
            print(f"  - {col}")
        
        # Özellik önemi
        print("\nFeature Importance (correlation with 1-day return):")
        importance = fe.get_feature_importance(
            features, 
            targets['target_return_1'],
            method='correlation'
        )
        print(importance.head(10))
        
        # Kaydet
        fe.save_features(features, targets, symbol, timeframe)
        
        # Test yükleme
        loaded_features, loaded_targets = fe.load_features(symbol, timeframe)
        print(f"\nLoaded features shape: {loaded_features.shape}")


if __name__ == "__main__":
    main()