"""
MACD Custom Indicator - Multi Time Frame
Klasik MACD göstergesi ile trend yönü ve momentum değişimleri
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from loguru import logger


class MACDCustom:
    """MACD Custom indikatörü"""
    
    def __init__(self, fast_length: int = 12, slow_length: int = 26, 
                 signal_length: int = 9, source: str = 'close'):
        """
        Args:
            fast_length: Hızlı EMA periyodu
            slow_length: Yavaş EMA periyodu
            signal_length: Sinyal hattı SMA periyodu
            source: Kaynak fiyat (close, open, high, low, hl2, hlc3, ohlc4)
        """
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.signal_length = signal_length
        self.source = source
    
    def get_source_price(self, df: pd.DataFrame) -> pd.Series:
        """Kaynak fiyatı hesapla"""
        if self.source == 'close':
            return df['close']
        elif self.source == 'open':
            return df['open']
        elif self.source == 'high':
            return df['high']
        elif self.source == 'low':
            return df['low']
        elif self.source == 'hl2':
            return (df['high'] + df['low']) / 2
        elif self.source == 'hlc3':
            return (df['high'] + df['low'] + df['close']) / 3
        elif self.source == 'ohlc4':
            return (df['open'] + df['high'] + df['low'] + df['close']) / 4
        else:
            return df['close']  # Varsayılan
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD hesapla"""
        try:
            # Kopya al
            data = df.copy()
            
            # Kaynak fiyat
            src = self.get_source_price(data)
            
            # EMA hesapla
            data['ema_fast'] = src.ewm(span=self.fast_length, adjust=False).mean()
            data['ema_slow'] = src.ewm(span=self.slow_length, adjust=False).mean()
            
            # MACD hattı
            data['macd'] = data['ema_fast'] - data['ema_slow']
            
            # Sinyal hattı (MACD'nin SMA'sı)
            data['macd_signal'] = data['macd'].rolling(window=self.signal_length).mean()
            
            # Histogram
            data['macd_hist'] = data['macd'] - data['macd_signal']
            
            # Histogram rengi ve yönü
            data['hist_color'] = 'gray'
            data.loc[data['macd_hist'] > 0, 'hist_color'] = 'green'
            data.loc[data['macd_hist'] < 0, 'hist_color'] = 'red'
            
            # Histogram momentum (artan/azalan)
            data['hist_increasing'] = (data['macd_hist'] > data['macd_hist'].shift(1)).astype(int)
            data['hist_decreasing'] = (data['macd_hist'] < data['macd_hist'].shift(1)).astype(int)
            
            # Gelişmiş histogram renklendirme
            data.loc[(data['macd_hist'] > 0) & data['hist_increasing'], 'hist_color'] = 'lime'
            data.loc[(data['macd_hist'] > 0) & data['hist_decreasing'], 'hist_color'] = 'green'
            data.loc[(data['macd_hist'] < 0) & data['hist_decreasing'], 'hist_color'] = 'maroon'
            data.loc[(data['macd_hist'] < 0) & data['hist_increasing'], 'hist_color'] = 'red'
            
            # MACD ve Sinyal hattı kesişimleri
            data['macd_cross_up'] = ((data['macd'] > data['macd_signal']) & 
                                    (data['macd'].shift(1) <= data['macd_signal'].shift(1))).astype(int)
            data['macd_cross_down'] = ((data['macd'] < data['macd_signal']) & 
                                      (data['macd'].shift(1) >= data['macd_signal'].shift(1))).astype(int)
            
            # Zero line kesişimleri
            data['macd_zero_cross_up'] = ((data['macd'] > 0) & 
                                         (data['macd'].shift(1) <= 0)).astype(int)
            data['macd_zero_cross_down'] = ((data['macd'] < 0) & 
                                           (data['macd'].shift(1) >= 0)).astype(int)
            
            # Divergence tespiti için hazırlık (basit versiyon)
            # Fiyat ve MACD zirvelerini/diplerini karşılaştır
            price_highs = src.rolling(window=5).max() == src
            price_lows = src.rolling(window=5).min() == src
            macd_highs = data['macd'].rolling(window=5).max() == data['macd']
            macd_lows = data['macd'].rolling(window=5).min() == data['macd']
            
            # Bullish divergence: Fiyat düşük dip, MACD yüksek dip
            data['potential_bullish_div'] = (price_lows & macd_lows & 
                                           (src < src.shift(10)) & 
                                           (data['macd'] > data['macd'].shift(10))).astype(int)
            
            # Bearish divergence: Fiyat yüksek zirve, MACD düşük zirve
            data['potential_bearish_div'] = (price_highs & macd_highs & 
                                            (src > src.shift(10)) & 
                                            (data['macd'] < data['macd'].shift(10))).astype(int)
            
            # Trend durumu
            data['macd_trend'] = 'neutral'
            data.loc[data['macd'] > 0, 'macd_trend'] = 'bullish'
            data.loc[data['macd'] < 0, 'macd_trend'] = 'bearish'
            
            # Momentum gücü
            data['macd_momentum'] = 'neutral'
            data.loc[(data['macd'] > data['macd_signal']) & data['hist_increasing'], 'macd_momentum'] = 'strong_bullish'
            data.loc[(data['macd'] > data['macd_signal']) & ~data['hist_increasing'], 'macd_momentum'] = 'bullish'
            data.loc[(data['macd'] < data['macd_signal']) & data['hist_decreasing'], 'macd_momentum'] = 'strong_bearish'
            data.loc[(data['macd'] < data['macd_signal']) & ~data['hist_decreasing'], 'macd_momentum'] = 'bearish'
            
            # Sinyaller
            data['macd_buy_signal'] = data['macd_cross_up'].astype(int)
            data['macd_sell_signal'] = data['macd_cross_down'].astype(int)
            
            # Sadece hesaplanan sütunları döndür
            result_columns = ['macd', 'macd_signal', 'macd_hist', 'hist_color',
                            'hist_increasing', 'hist_decreasing',
                            'macd_cross_up', 'macd_cross_down',
                            'macd_zero_cross_up', 'macd_zero_cross_down',
                            'potential_bullish_div', 'potential_bearish_div',
                            'macd_trend', 'macd_momentum',
                            'macd_buy_signal', 'macd_sell_signal']
            
            return data[result_columns]
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return pd.DataFrame()
    
    def get_signal(self, latest_values: Dict[str, Any]) -> str:
        """Son değerlere göre sinyal üret"""
        try:
            # Direkt alım/satım sinyalleri
            if latest_values.get('macd_buy_signal', 0) == 1:
                return 'BUY'
            elif latest_values.get('macd_sell_signal', 0) == 1:
                return 'SELL'
            
            # Divergence sinyalleri
            if latest_values.get('potential_bullish_div', 0) == 1:
                return 'BULLISH_DIVERGENCE'
            elif latest_values.get('potential_bearish_div', 0) == 1:
                return 'BEARISH_DIVERGENCE'
            
            # Zero line kesişimleri
            if latest_values.get('macd_zero_cross_up', 0) == 1:
                return 'BULLISH_MOMENTUM'
            elif latest_values.get('macd_zero_cross_down', 0) == 1:
                return 'BEARISH_MOMENTUM'
            
            # Genel momentum durumu
            momentum = latest_values.get('macd_momentum', 'neutral')
            if momentum == 'strong_bullish':
                return 'STRONG_BULLISH'
            elif momentum == 'strong_bearish':
                return 'STRONG_BEARISH'
            elif momentum == 'bullish':
                return 'BULLISH'
            elif momentum == 'bearish':
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Error getting MACD signal: {e}")
            return 'ERROR'