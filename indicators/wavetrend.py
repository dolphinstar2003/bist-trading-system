"""
WaveTrend with Crosses [LazyBear]
Momentum osilatörü - aşırı alım/satım bölgeleri ve momentum dönüşleri
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from loguru import logger


class WaveTrend:
    """WaveTrend osilatör indikatörü"""
    
    def __init__(self, n1: int = 10, n2: int = 21, 
                 ob_level1: int = 60, ob_level2: int = 53,
                 os_level1: int = -60, os_level2: int = -53):
        """
        Args:
            n1: Kanal uzunluğu (Channel Length)
            n2: Ortalama uzunluğu (Average Length)
            ob_level1: Aşırı alım seviyesi 1
            ob_level2: Aşırı alım seviyesi 2
            os_level1: Aşırı satım seviyesi 1
            os_level2: Aşırı satım seviyesi 2
        """
        self.n1 = n1
        self.n2 = n2
        self.ob_level1 = ob_level1
        self.ob_level2 = ob_level2
        self.os_level1 = os_level1
        self.os_level2 = os_level2
    
    def ema(self, series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average hesapla"""
        return series.ewm(span=period, adjust=False).mean()
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """WaveTrend hesapla"""
        try:
            # Kopya al
            data = df.copy()
            
            # Tipik fiyat (hlc3)
            ap = (data['high'] + data['low'] + data['close']) / 3
            
            # ESA - Exponentially Smoothed Average of ap
            esa = self.ema(ap, self.n1)
            
            # d - EMA of absolute difference
            d = self.ema(abs(ap - esa), self.n1)
            
            # CI - Channel Index
            ci = (ap - esa) / (0.015 * d)
            # NaN ve inf değerlerini temizle
            ci = ci.replace([np.inf, -np.inf], 0).fillna(0)
            
            # TCI - WaveTrend Line 1
            tci = self.ema(ci, self.n2)
            data['wt1'] = tci
            
            # WaveTrend Line 2 (Signal Line)
            data['wt2'] = data['wt1'].rolling(window=4).mean()
            
            # Aşırı alım/satım seviyeleri
            data['ob_level1'] = self.ob_level1
            data['ob_level2'] = self.ob_level2
            data['os_level1'] = self.os_level1
            data['os_level2'] = self.os_level2
            
            # Kesişimleri tespit et
            data['wt_cross_up'] = ((data['wt1'] > data['wt2']) & 
                                  (data['wt1'].shift(1) <= data['wt2'].shift(1))).astype(int)
            data['wt_cross_down'] = ((data['wt1'] < data['wt2']) & 
                                    (data['wt1'].shift(1) >= data['wt2'].shift(1))).astype(int)
            
            # Aşırı alım/satım bölgelerindeki kesişimler
            data['wt_buy_signal'] = (data['wt_cross_up'] & 
                                    (data['wt1'] < self.os_level2)).astype(int)
            data['wt_sell_signal'] = (data['wt_cross_down'] & 
                                     (data['wt1'] > self.ob_level2)).astype(int)
            
            # Bölge tespiti
            data['wt_zone'] = 'neutral'
            data.loc[data['wt1'] > self.ob_level1, 'wt_zone'] = 'extreme_overbought'
            data.loc[(data['wt1'] > self.ob_level2) & (data['wt1'] <= self.ob_level1), 'wt_zone'] = 'overbought'
            data.loc[data['wt1'] < self.os_level1, 'wt_zone'] = 'extreme_oversold'
            data.loc[(data['wt1'] < self.os_level2) & (data['wt1'] >= self.os_level1), 'wt_zone'] = 'oversold'
            
            # Momentum yönü
            data['wt_momentum'] = 'neutral'
            data.loc[data['wt1'] > data['wt1'].shift(1), 'wt_momentum'] = 'bullish'
            data.loc[data['wt1'] < data['wt1'].shift(1), 'wt_momentum'] = 'bearish'
            
            # Sadece hesaplanan sütunları döndür
            result_columns = ['wt1', 'wt2', 'ob_level1', 'ob_level2', 'os_level1', 'os_level2',
                            'wt_cross_up', 'wt_cross_down', 'wt_buy_signal', 'wt_sell_signal',
                            'wt_zone', 'wt_momentum']
            
            return data[result_columns]
            
        except Exception as e:
            logger.error(f"Error calculating WaveTrend: {e}")
            return pd.DataFrame()
    
    def get_signal(self, latest_values: Dict[str, Any]) -> str:
        """Son değerlere göre sinyal üret"""
        try:
            # Alım/satım sinyalleri
            if latest_values.get('wt_buy_signal', 0) == 1:
                return 'BUY'
            elif latest_values.get('wt_sell_signal', 0) == 1:
                return 'SELL'
            
            # Bölge bazlı değerlendirme
            zone = latest_values.get('wt_zone', 'neutral')
            momentum = latest_values.get('wt_momentum', 'neutral')
            
            if zone == 'extreme_oversold' and momentum == 'bullish':
                return 'STRONG_BUY_ZONE'
            elif zone == 'extreme_overbought' and momentum == 'bearish':
                return 'STRONG_SELL_ZONE'
            elif zone == 'oversold':
                return 'BUY_ZONE'
            elif zone == 'overbought':
                return 'SELL_ZONE'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Error getting WaveTrend signal: {e}")
            return 'ERROR'