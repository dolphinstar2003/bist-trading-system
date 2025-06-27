"""
ADX and DI (Average Directional Index)
Trend gücü ve yönü tespiti
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from loguru import logger


class ADX_DI:
    """ADX ve Directional Indicators"""
    
    def __init__(self, length: int = 14, threshold: int = 20):
        """
        Args:
            length: ADX hesaplama periyodu
            threshold: Güçlü trend eşik değeri
        """
        self.length = length
        self.threshold = threshold
    
    def calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """True Range hesapla"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr
    
    def wilder_smooth(self, series: pd.Series, period: int) -> pd.Series:
        """Wilder's Smoothing (RMA)"""
        # İlk değer için SMA kullan
        wilder = series.rolling(window=period).mean()
        
        # Sonraki değerler için Wilder's smoothing formülü
        for i in range(period, len(series)):
            if pd.notna(wilder.iloc[i-1]):
                wilder.iloc[i] = (wilder.iloc[i-1] * (period - 1) + series.iloc[i]) / period
        
        return wilder
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """ADX ve DI değerlerini hesapla"""
        try:
            # Kopya al
            data = df.copy()
            
            # True Range
            data['tr'] = self.calculate_true_range(data)
            
            # Directional Movement
            high_diff = data['high'] - data['high'].shift(1)
            low_diff = data['low'].shift(1) - data['low']
            
            # +DM ve -DM
            data['plus_dm'] = np.where(
                (high_diff > low_diff) & (high_diff > 0), 
                high_diff, 
                0
            )
            data['minus_dm'] = np.where(
                (low_diff > high_diff) & (low_diff > 0), 
                low_diff, 
                0
            )
            
            # Smoothed True Range, +DM ve -DM
            data['atr'] = self.wilder_smooth(data['tr'], self.length)
            data['plus_dm_smooth'] = self.wilder_smooth(data['plus_dm'], self.length)
            data['minus_dm_smooth'] = self.wilder_smooth(data['minus_dm'], self.length)
            
            # +DI ve -DI
            data['plus_di'] = 100 * data['plus_dm_smooth'] / data['atr']
            data['minus_di'] = 100 * data['minus_dm_smooth'] / data['atr']
            
            # DX
            di_sum = data['plus_di'] + data['minus_di']
            di_diff = abs(data['plus_di'] - data['minus_di'])
            data['dx'] = 100 * di_diff / di_sum
            
            # ADX (DX'in smoothed hali)
            data['adx'] = self.wilder_smooth(data['dx'], self.length)
            
            # Trend gücü
            data['trend_strength'] = 'no_trend'
            data.loc[data['adx'] > self.threshold, 'trend_strength'] = 'trending'
            data.loc[data['adx'] > 25, 'trend_strength'] = 'strong_trend'
            data.loc[data['adx'] > 50, 'trend_strength'] = 'very_strong_trend'
            data.loc[data['adx'] > 75, 'trend_strength'] = 'extremely_strong_trend'
            
            # Trend yönü
            data['trend_direction'] = 'neutral'
            data.loc[data['plus_di'] > data['minus_di'], 'trend_direction'] = 'bullish'
            data.loc[data['minus_di'] > data['plus_di'], 'trend_direction'] = 'bearish'
            
            # DI kesişimleri
            data['di_bullish_cross'] = (
                (data['plus_di'] > data['minus_di']) & 
                (data['plus_di'].shift(1) <= data['minus_di'].shift(1))
            ).astype(int)
            
            data['di_bearish_cross'] = (
                (data['plus_di'] < data['minus_di']) & 
                (data['plus_di'].shift(1) >= data['minus_di'].shift(1))
            ).astype(int)
            
            # ADX yükseliyor mu?
            data['adx_rising'] = (data['adx'] > data['adx'].shift(1)).astype(int)
            data['adx_falling'] = (data['adx'] < data['adx'].shift(1)).astype(int)
            
            # Threshold üstünde/altında
            data['adx_above_threshold'] = (data['adx'] > self.threshold).astype(int)
            
            # Sinyaller
            # Güçlü trend + DI kesişimi
            data['adx_buy_signal'] = (
                data['di_bullish_cross'] & 
                data['adx_above_threshold']
            ).astype(int)
            
            data['adx_sell_signal'] = (
                data['di_bearish_cross'] & 
                data['adx_above_threshold']
            ).astype(int)
            
            # Sadece hesaplanan sütunları döndür
            result_columns = ['adx', 'plus_di', 'minus_di', 'atr',
                            'trend_strength', 'trend_direction',
                            'di_bullish_cross', 'di_bearish_cross',
                            'adx_rising', 'adx_falling', 'adx_above_threshold',
                            'adx_buy_signal', 'adx_sell_signal']
            
            return data[result_columns]
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {e}")
            return pd.DataFrame()
    
    def get_signal(self, latest_values: Dict[str, Any]) -> str:
        """Son değerlere göre sinyal üret"""
        try:
            # Direkt alım/satım sinyalleri
            if latest_values.get('adx_buy_signal', 0) == 1:
                return 'BUY'
            elif latest_values.get('adx_sell_signal', 0) == 1:
                return 'SELL'
            
            # Trend analizi
            adx = latest_values.get('adx', 0)
            trend_strength = latest_values.get('trend_strength', 'no_trend')
            trend_direction = latest_values.get('trend_direction', 'neutral')
            adx_rising = latest_values.get('adx_rising', 0)
            
            # Trend yok
            if adx < self.threshold:
                return 'NO_TREND'
            
            # Güçlü trend var
            if trend_strength in ['strong_trend', 'very_strong_trend', 'extremely_strong_trend']:
                if trend_direction == 'bullish' and adx_rising:
                    return 'STRONG_BULLISH_TREND'
                elif trend_direction == 'bearish' and adx_rising:
                    return 'STRONG_BEARISH_TREND'
                elif not adx_rising:
                    return 'TREND_WEAKENING'
            
            # Normal trend
            if trend_direction == 'bullish':
                return 'BULLISH_TREND'
            elif trend_direction == 'bearish':
                return 'BEARISH_TREND'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Error getting ADX signal: {e}")
            return 'ERROR'