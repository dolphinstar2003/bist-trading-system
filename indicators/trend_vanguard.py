"""
Trend Vanguard Strategy
ZigZag pattern tanıma ve k-NN makine öğrenmesi ile gelişmiş trading stratejisi
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class TrendVanguard:
    """Trend Vanguard Strategy - Gelişmiş pattern tanıma ve ML"""
    
    def __init__(self, 
                 # ZigZag parametreleri
                 zigzag_depth: int = 21,
                 zigzag_deviation: float = 6.0,
                 zigzag_backstep: int = 4,
                 # k-NN parametreleri
                 knn_lookback: int = 100,
                 k_value: int = 9,
                 confidence_threshold: float = 0.55,
                 # Risk yönetimi
                 risk_per_trade: float = 0.005,
                 atr_multiplier: float = 1.5,
                 # Pattern tanıma
                 pattern_lookback: int = 50,
                 min_pattern_bars: int = 10):
        """
        Args:
            zigzag_depth: ZigZag derinlik parametresi
            zigzag_deviation: ZigZag sapma yüzdesi
            zigzag_backstep: ZigZag geri adım
            knn_lookback: k-NN için geriye bakış
            k_value: k-NN komşu sayısı
            confidence_threshold: Güven eşiği
            risk_per_trade: İşlem başına risk
            atr_multiplier: Stop loss için ATR çarpanı
            pattern_lookback: Pattern arama penceresi
            min_pattern_bars: Minimum pattern bar sayısı
        """
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
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR hesapla"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_zigzag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basitleştirilmiş ZigZag hesapla"""
        result = pd.DataFrame(index=df.index)
        result['zigzag'] = np.nan
        result['is_pivot_high'] = False
        result['is_pivot_low'] = False
        
        # Pivot noktaları bul
        for i in range(self.zigzag_depth, len(df) - self.zigzag_depth):
            # Yüksek pivot kontrolü
            is_high = True
            for j in range(1, self.zigzag_depth + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i - j] or \
                   df['high'].iloc[i] <= df['high'].iloc[i + j]:
                    is_high = False
                    break
            
            if is_high:
                result.loc[result.index[i], 'is_pivot_high'] = True
                result.loc[result.index[i], 'zigzag'] = df['high'].iloc[i]
            
            # Düşük pivot kontrolü
            is_low = True
            for j in range(1, self.zigzag_depth + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i - j] or \
                   df['low'].iloc[i] >= df['low'].iloc[i + j]:
                    is_low = False
                    break
            
            if is_low:
                result.loc[result.index[i], 'is_pivot_low'] = True
                result.loc[result.index[i], 'zigzag'] = df['low'].iloc[i]
        
        return result
    
    def extract_pattern_features(self, df: pd.DataFrame, idx: int, 
                               zigzag_data: pd.DataFrame) -> List[float]:
        """Pattern özelliklerini çıkar"""
        features = []
        
        try:
            # Son pivot noktalarını bul
            pivot_indices = []
            for i in range(idx - self.pattern_lookback, idx):
                if zigzag_data['is_pivot_high'].iloc[i] or zigzag_data['is_pivot_low'].iloc[i]:
                    pivot_indices.append(i)
            
            if len(pivot_indices) < 4:
                return [0] * 10  # Yeterli pivot yok
            
            # Son 4 pivot'u al
            last_pivots = pivot_indices[-4:]
            
            # Özellik 1: Son pivot'un türü (high=1, low=-1)
            last_pivot_idx = last_pivots[-1]
            features.append(1 if zigzag_data['is_pivot_high'].iloc[last_pivot_idx] else -1)
            
            # Özellik 2-5: Pivot seviyeleri (normalize edilmiş)
            for pivot_idx in last_pivots:
                pivot_value = zigzag_data['zigzag'].iloc[pivot_idx]
                normalized = (pivot_value - df['close'].iloc[idx]) / df['close'].iloc[idx]
                features.append(normalized)
            
            # Özellik 6-9: Pivot arası mesafeler (bar sayısı)
            for i in range(len(last_pivots) - 1):
                distance = last_pivots[i + 1] - last_pivots[i]
                features.append(distance / self.pattern_lookback)
            
            # Özellik 10: Genel trend (ilk ve son pivot farkı)
            first_value = zigzag_data['zigzag'].iloc[last_pivots[0]]
            last_value = zigzag_data['zigzag'].iloc[last_pivots[-1]]
            trend = (last_value - first_value) / first_value
            features.append(trend)
            
        except Exception:
            features = [0] * 10
        
        return features
    
    def lorentzian_distance(self, x1: List[float], x2: List[float]) -> float:
        """Lorentzian mesafe hesapla"""
        distance = 0
        for a, b in zip(x1, x2):
            distance += np.log(1 + abs(a - b))
        return distance
    
    def predict_with_knn(self, df: pd.DataFrame, current_idx: int,
                        zigzag_data: pd.DataFrame) -> Tuple[float, float]:
        """k-NN ile tahmin yap"""
        if current_idx < self.knn_lookback + 10:
            return 0, 0
        
        # Mevcut pattern özelliklerini çıkar
        current_features = self.extract_pattern_features(df, current_idx, zigzag_data)
        
        # Geçmiş pattern'leri ve sonuçlarını topla
        patterns = []
        
        for i in range(current_idx - self.knn_lookback, current_idx - 10):
            past_features = self.extract_pattern_features(df, i, zigzag_data)
            
            # 10 bar sonraki sonuç
            future_return = (df['close'].iloc[i + 10] - df['close'].iloc[i]) / df['close'].iloc[i]
            
            # Mesafe hesapla
            distance = self.lorentzian_distance(current_features, past_features)
            
            patterns.append((distance, future_return))
        
        if not patterns:
            return 0, 0
        
        # En yakın k pattern'i bul
        patterns.sort(key=lambda x: x[0])
        nearest = patterns[:self.k_value]
        
        # Ağırlıklı tahmin
        total_weight = 0
        weighted_prediction = 0
        
        for distance, return_val in nearest:
            weight = 1 / (distance + 1e-10)
            weighted_prediction += weight * return_val
            total_weight += weight
        
        if total_weight > 0:
            prediction = weighted_prediction / total_weight
            
            # Güven skoru (en yakın komşuların tutarlılığı)
            returns = [r for _, r in nearest]
            std_dev = np.std(returns) if len(returns) > 1 else 1
            confidence = 1 / (1 + std_dev)
            
            return prediction, confidence
        
        return 0, 0
    
    def calculate_regime_filter(self, df: pd.DataFrame) -> pd.Series:
        """Market rejim filtresi"""
        # Basit trend filtresi
        sma_20 = df['close'].rolling(window=20).mean()
        sma_50 = df['close'].rolling(window=50).mean()
        sma_200 = df['close'].rolling(window=200).mean()
        
        regime = pd.Series(index=df.index, dtype=str)
        regime[:] = 'neutral'
        
        # Yükseliş trendi
        bullish = (sma_20 > sma_50) & (sma_50 > sma_200) & (df['close'] > sma_20)
        regime[bullish] = 'bullish'
        
        # Düşüş trendi
        bearish = (sma_20 < sma_50) & (sma_50 < sma_200) & (df['close'] < sma_20)
        regime[bearish] = 'bearish'
        
        return regime
    
    def calculate_volatility_filter(self, df: pd.DataFrame, atr: pd.Series) -> pd.Series:
        """Volatilite filtresi"""
        # ATR yüzdesi
        atr_pct = atr / df['close'] * 100
        
        # Volatilite seviyeleri
        volatility = pd.Series(index=df.index, dtype=str)
        volatility[:] = 'normal'
        
        # Düşük volatilite
        volatility[atr_pct < atr_pct.rolling(window=100).quantile(0.25)] = 'low'
        
        # Yüksek volatilite
        volatility[atr_pct > atr_pct.rolling(window=100).quantile(0.75)] = 'high'
        
        # Ekstrem volatilite
        volatility[atr_pct > atr_pct.rolling(window=100).quantile(0.95)] = 'extreme'
        
        return volatility
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend Vanguard hesapla"""
        try:
            # Kopya al
            data = df.copy()
            
            # ATR hesapla
            data['atr'] = self.calculate_atr(data)
            
            # ZigZag hesapla
            zigzag_data = self.calculate_zigzag(data)
            
            # Market rejimi
            data['market_regime'] = self.calculate_regime_filter(data)
            
            # Volatilite filtresi
            data['volatility_state'] = self.calculate_volatility_filter(data, data['atr'])
            
            # Tahmin ve sinyaller için sütunlar
            data['knn_prediction'] = 0.0
            data['knn_confidence'] = 0.0
            data['signal'] = 0
            data['signal_strength'] = 0.0
            
            # Stop loss ve take profit seviyeleri
            data['stop_loss_long'] = np.nan
            data['stop_loss_short'] = np.nan
            data['take_profit_long'] = np.nan
            data['take_profit_short'] = np.nan
            
            # Her bar için tahmin yap
            for i in range(self.knn_lookback + 10, len(data)):
                prediction, confidence = self.predict_with_knn(data, i, zigzag_data)
                
                data.loc[data.index[i], 'knn_prediction'] = prediction
                data.loc[data.index[i], 'knn_confidence'] = confidence
                
                # Sinyal üretimi
                if confidence >= self.confidence_threshold:
                    current_price = data['close'].iloc[i]
                    current_atr = data['atr'].iloc[i]
                    
                    if prediction > 0.01:  # %1'den fazla yükseliş beklentisi
                        data.loc[data.index[i], 'signal'] = 1
                        data.loc[data.index[i], 'signal_strength'] = min(prediction * 10, 1.0)
                        
                        # Stop loss ve take profit
                        data.loc[data.index[i], 'stop_loss_long'] = current_price - (current_atr * self.atr_multiplier)
                        data.loc[data.index[i], 'take_profit_long'] = current_price + (current_atr * self.atr_multiplier * 2)
                        
                    elif prediction < -0.01:  # %1'den fazla düşüş beklentisi
                        data.loc[data.index[i], 'signal'] = -1
                        data.loc[data.index[i], 'signal_strength'] = min(abs(prediction) * 10, 1.0)
                        
                        # Stop loss ve take profit
                        data.loc[data.index[i], 'stop_loss_short'] = current_price + (current_atr * self.atr_multiplier)
                        data.loc[data.index[i], 'take_profit_short'] = current_price - (current_atr * self.atr_multiplier * 2)
            
            # Multi-timeframe onayı (basitleştirilmiş)
            data['mtf_confirmation'] = 0
            sma_fast = data['close'].rolling(window=20).mean()
            sma_slow = data['close'].rolling(window=50).mean()
            
            data.loc[(data['signal'] == 1) & (sma_fast > sma_slow), 'mtf_confirmation'] = 1
            data.loc[(data['signal'] == -1) & (sma_fast < sma_slow), 'mtf_confirmation'] = 1
            
            # Filtreli sinyaller
            data['filtered_signal'] = data['signal'] * data['mtf_confirmation']
            
            # Quantum-ilhamlı olasılık düzeltmesi (basitleştirilmiş)
            data['quantum_adjustment'] = np.random.normal(1.0, 0.05, len(data))
            data['adjusted_confidence'] = data['knn_confidence'] * data['quantum_adjustment']
            data['adjusted_confidence'] = data['adjusted_confidence'].clip(0, 1)
            
            # Risk yönetimi metrikleri
            data['position_size'] = self.risk_per_trade / (data['atr'] / data['close'])
            data['position_size'] = data['position_size'].clip(0, 0.1)  # Maksimum %10
            
            # Pattern tanıma sonuçları - zigzag_data sütunlarını data'ya ekle
            for col in zigzag_data.columns:
                data[col] = zigzag_data[col]
            
            # Sadece hesaplanan sütunları döndür
            result_columns = ['knn_prediction', 'knn_confidence', 'signal', 'signal_strength',
                            'market_regime', 'volatility_state', 'atr',
                            'stop_loss_long', 'stop_loss_short', 
                            'take_profit_long', 'take_profit_short',
                            'mtf_confirmation', 'filtered_signal',
                            'adjusted_confidence', 'position_size',
                            'zigzag', 'is_pivot_high', 'is_pivot_low']
            
            return data[result_columns]
            
        except Exception as e:
            logger.error(f"Error calculating Trend Vanguard: {e}")
            return pd.DataFrame()
    
    def get_signal(self, latest_values: Dict[str, Any]) -> str:
        """Son değerlere göre sinyal üret"""
        try:
            filtered_signal = latest_values.get('filtered_signal', 0)
            confidence = latest_values.get('adjusted_confidence', 0)
            signal_strength = latest_values.get('signal_strength', 0)
            regime = latest_values.get('market_regime', 'neutral')
            volatility = latest_values.get('volatility_state', 'normal')
            
            # Filtreli sinyal varsa
            if filtered_signal == 1:
                if confidence > 0.7 and signal_strength > 0.7:
                    if regime == 'bullish' and volatility != 'extreme':
                        return 'STRONG_BUY'
                    else:
                        return 'BUY'
                else:
                    return 'WEAK_BUY'
                    
            elif filtered_signal == -1:
                if confidence > 0.7 and signal_strength > 0.7:
                    if regime == 'bearish' and volatility != 'extreme':
                        return 'STRONG_SELL'
                    else:
                        return 'SELL'
                else:
                    return 'WEAK_SELL'
            
            # Pivot sinyalleri
            if latest_values.get('is_pivot_low', False):
                return 'POTENTIAL_BOTTOM'
            elif latest_values.get('is_pivot_high', False):
                return 'POTENTIAL_TOP'
            
            # Rejim bazlı değerlendirme
            if regime == 'bullish' and volatility == 'low':
                return 'BULLISH_REGIME'
            elif regime == 'bearish' and volatility == 'low':
                return 'BEARISH_REGIME'
            elif volatility == 'extreme':
                return 'HIGH_VOLATILITY'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Error getting Trend Vanguard signal: {e}")
            return 'ERROR'