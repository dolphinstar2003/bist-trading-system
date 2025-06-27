"""
Lorentzian Classification
k-NN tabanlı makine öğrenmesi ile fiyat yönü tahmini
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class LorentzianClassification:
    """Lorentzian mesafe metriki kullanan k-NN sınıflandırıcı"""
    
    def __init__(self, neighbors_count: int = 8, max_bars_back: int = 2000,
                 feature_count: int = 5, show_predictions: bool = True,
                 use_kernel_filter: bool = True, use_kernel_smoothing: bool = False,
                 lookback_window: int = 8, relative_weighting: float = 8.0,
                 regression_level: int = 25, kernel_regression_lookback: int = 50):
        """
        Args:
            neighbors_count: k-NN için komşu sayısı
            max_bars_back: Maksimum geriye bakış
            feature_count: Kullanılacak özellik sayısı
            show_predictions: Tahminleri göster
            use_kernel_filter: Kernel regresyon filtresi kullan
            use_kernel_smoothing: Kernel smoothing kullan
            lookback_window: Kernel için bakış penceresi
            relative_weighting: Kernel ağırlıklandırma
            regression_level: Regresyon seviyesi
            kernel_regression_lookback: Kernel regresyon geriye bakış
        """
        self.neighbors_count = neighbors_count
        self.max_bars_back = max_bars_back
        self.feature_count = feature_count
        self.show_predictions = show_predictions
        self.use_kernel_filter = use_kernel_filter
        self.use_kernel_smoothing = use_kernel_smoothing
        self.lookback_window = lookback_window
        self.relative_weighting = relative_weighting
        self.regression_level = regression_level
        self.kernel_regression_lookback = kernel_regression_lookback
    
    def calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """RSI hesapla"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_wt(self, df: pd.DataFrame, n1: int = 10, n2: int = 21) -> pd.Series:
        """WaveTrend hesapla"""
        ap = (df['high'] + df['low'] + df['close']) / 3
        esa = ap.ewm(span=n1, adjust=False).mean()
        d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
        ci = (ap - esa) / (0.015 * d)
        wt = ci.ewm(span=n2, adjust=False).mean()
        return wt
    
    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """CCI hesapla"""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = (tp - sma).abs().rolling(window=period).mean()
        cci = (tp - sma) / (0.015 * mad)
        return cci
    
    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Basit ADX hesapla"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Basitleştirilmiş ADX
        adx = (atr / df['close']) * 100
        adx = adx.rolling(window=period).mean()
        
        return adx
    
    def normalize_series(self, series: pd.Series, min_val: float = 0, max_val: float = 1) -> pd.Series:
        """Seriyi normalize et"""
        series_min = series.rolling(window=self.max_bars_back, min_periods=1).min()
        series_max = series.rolling(window=self.max_bars_back, min_periods=1).max()
        normalized = (series - series_min) / (series_max - series_min)
        normalized = normalized * (max_val - min_val) + min_val
        return normalized.fillna(0.5)
    
    def lorentzian_distance(self, x1: float, x2: float) -> float:
        """Lorentzian mesafe hesapla"""
        return np.log(1 + abs(x1 - x2))
    
    def kernel_regression(self, series: pd.Series, lookback: int, 
                         relative_weight: float) -> pd.Series:
        """Nadaraya-Watson Kernel Regression"""
        result = pd.Series(index=series.index, dtype=float)
        
        for i in range(lookback, len(series)):
            weights = []
            values = []
            
            for j in range(lookback):
                weight = np.exp(-np.power(j, 2) / (2 * np.power(relative_weight, 2)))
                weights.append(weight)
                values.append(series.iloc[i - j])
            
            weights = np.array(weights)
            values = np.array(values)
            
            result.iloc[i] = np.sum(weights * values) / np.sum(weights)
        
        return result
    
    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tüm özellikleri hesapla"""
        features = pd.DataFrame(index=df.index)
        
        # Özellik 1: RSI
        features['f1'] = self.normalize_series(self.calculate_rsi(df['close'], 14), -1, 1)
        
        # Özellik 2: WaveTrend
        features['f2'] = self.normalize_series(self.calculate_wt(df), -1, 1)
        
        # Özellik 3: CCI
        features['f3'] = self.normalize_series(self.calculate_cci(df, 20), -1, 1)
        
        # Özellik 4: ADX
        features['f4'] = self.normalize_series(self.calculate_adx(df), -1, 1)
        
        # Özellik 5: RSI (9 periyot)
        features['f5'] = self.normalize_series(self.calculate_rsi(df['close'], 9), -1, 1)
        
        # İsteğe bağlı ek özellikler
        if self.feature_count > 5:
            # Özellik 6: WaveTrend (alternatif parametreler)
            features['f6'] = self.normalize_series(self.calculate_wt(df), -1, 1)
            
        if self.feature_count > 6:
            # Özellik 7: CCI (alternatif periyot)
            features['f7'] = self.normalize_series(self.calculate_cci(df, 10), -1, 1)
            
        if self.feature_count > 7:
            # Özellik 8: ADX (alternatif periyot)
            features['f8'] = self.normalize_series(self.calculate_adx(df), -1, 1)
        
        return features
    
    def predict_direction(self, df: pd.DataFrame, features: pd.DataFrame, 
                         current_idx: int) -> Tuple[float, List[float]]:
        """k-NN ile yön tahmini yap"""
        if current_idx < self.neighbors_count + 4:
            return 0, []
        
        # Mevcut özellikleri al
        current_features = []
        for i in range(1, self.feature_count + 1):
            if f'f{i}' in features.columns:
                current_features.append(features[f'f{i}'].iloc[current_idx])
        
        # Mesafeleri hesapla
        distances = []
        directions = []
        
        for i in range(self.neighbors_count, min(current_idx - 4, self.max_bars_back)):
            past_idx = current_idx - i
            
            # Geçmiş özellikleri al
            past_features = []
            for j in range(1, self.feature_count + 1):
                if f'f{j}' in features.columns:
                    past_features.append(features[f'f{j}'].iloc[past_idx])
            
            # Lorentzian mesafe hesapla
            dist = 0
            for curr_f, past_f in zip(current_features, past_features):
                dist += self.lorentzian_distance(curr_f, past_f)
            
            # 4 bar sonraki yön
            future_idx = past_idx + 4
            if future_idx < len(df):
                direction = 1 if df['close'].iloc[future_idx] > df['close'].iloc[past_idx] else -1
                distances.append((dist, direction))
        
        # En yakın k komşuyu bul
        distances.sort(key=lambda x: x[0])
        nearest_neighbors = distances[:self.neighbors_count]
        
        # Ağırlıklı oylama
        if nearest_neighbors:
            predictions = [n[1] for n in nearest_neighbors]
            weights = [1 / (n[0] + 1e-10) for n in nearest_neighbors]
            
            weighted_sum = sum(p * w for p, w in zip(predictions, weights))
            total_weight = sum(weights)
            
            prediction = weighted_sum / total_weight if total_weight > 0 else 0
            
            return prediction, predictions
        
        return 0, []
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lorentzian Classification hesapla"""
        try:
            # Kopya al
            data = df.copy()
            
            # Özellikleri hesapla
            features = self.calculate_features(data)
            
            # Tahmin sonuçları için sütunlar
            data['prediction'] = 0.0
            data['prediction_raw'] = 0.0
            data['signal'] = 0
            data['is_bullish'] = False
            data['is_bearish'] = False
            data['is_neutral'] = True
            
            # Kernel regression için hazırlık
            if self.use_kernel_filter:
                data['kernel_estimate'] = 0.0
                data['kernel_upper'] = 0.0
                data['kernel_lower'] = 0.0
            
            # Her bar için tahmin yap
            for i in range(self.neighbors_count + 4, len(data)):
                prediction, _ = self.predict_direction(data, features, i)
                data.loc[data.index[i], 'prediction_raw'] = prediction
                
                # Sinyal üret
                if prediction > 0:
                    data.loc[data.index[i], 'signal'] = 1
                    data.loc[data.index[i], 'is_bullish'] = True
                    data.loc[data.index[i], 'is_neutral'] = False
                elif prediction < 0:
                    data.loc[data.index[i], 'signal'] = -1
                    data.loc[data.index[i], 'is_bearish'] = True
                    data.loc[data.index[i], 'is_neutral'] = False
            
            # Kernel filter uygula
            if self.use_kernel_filter and len(data) > self.kernel_regression_lookback:
                # Kernel regression
                kernel_series = self.kernel_regression(
                    data['close'], 
                    self.kernel_regression_lookback, 
                    self.relative_weighting
                )
                
                data['kernel_estimate'] = kernel_series
                
                # Kernel bantları
                kernel_dev = (data['close'] - kernel_series).abs().rolling(
                    window=self.kernel_regression_lookback
                ).std()
                
                data['kernel_upper'] = kernel_series + kernel_dev
                data['kernel_lower'] = kernel_series - kernel_dev
                
                # Filtreleme
                kernel_filter = (
                    (data['is_bullish'] & (data['close'] > data['kernel_estimate'])) |
                    (data['is_bearish'] & (data['close'] < data['kernel_estimate']))
                )
                
                data['prediction'] = data['prediction_raw'].where(kernel_filter, 0)
            else:
                data['prediction'] = data['prediction_raw']
            
            # EMA filtreleri (opsiyonel)
            data['ema_200'] = data['close'].ewm(span=200, adjust=False).mean()
            data['above_ema_200'] = (data['close'] > data['ema_200']).astype(int)
            
            # Sinyal gücü
            data['signal_strength'] = abs(data['prediction'])
            
            # Sadece hesaplanan sütunları döndür
            result_columns = ['prediction', 'prediction_raw', 'signal', 
                            'is_bullish', 'is_bearish', 'is_neutral',
                            'signal_strength', 'ema_200', 'above_ema_200']
            
            if self.use_kernel_filter:
                result_columns.extend(['kernel_estimate', 'kernel_upper', 'kernel_lower'])
            
            return data[result_columns]
            
        except Exception as e:
            logger.error(f"Error calculating Lorentzian Classification: {e}")
            return pd.DataFrame()
    
    def get_signal(self, latest_values: Dict[str, Any]) -> str:
        """Son değerlere göre sinyal üret"""
        try:
            prediction = latest_values.get('prediction', 0)
            signal_strength = latest_values.get('signal_strength', 0)
            above_ema = latest_values.get('above_ema_200', 0)
            
            # Güçlü sinyaller
            if prediction > 0 and signal_strength > 0.5:
                if above_ema:
                    return 'STRONG_BUY'
                else:
                    return 'BUY'
            elif prediction < 0 and signal_strength > 0.5:
                if not above_ema:
                    return 'STRONG_SELL'
                else:
                    return 'SELL'
            
            # Zayıf sinyaller
            if prediction > 0:
                return 'WEAK_BUY'
            elif prediction < 0:
                return 'WEAK_SELL'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Error getting Lorentzian signal: {e}")
            return 'ERROR'