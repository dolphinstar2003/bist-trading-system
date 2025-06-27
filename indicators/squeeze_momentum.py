"""
Squeeze Momentum Indicator [LazyBear]
Volatilite sıkışması ve momentum patlaması tespiti
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from loguru import logger


class SqueezeMomentum:
    """Squeeze Momentum indikatörü"""
    
    def __init__(self, bb_length: int = 20, bb_mult: float = 2.0,
                 kc_length: int = 20, kc_mult: float = 1.5,
                 use_true_range: bool = True):
        """
        Args:
            bb_length: Bollinger Band periyodu
            bb_mult: Bollinger Band standart sapma çarpanı
            kc_length: Keltner Channel periyodu
            kc_mult: Keltner Channel ATR çarpanı
            use_true_range: Keltner için True Range kullan
        """
        self.bb_length = bb_length
        self.bb_mult = bb_mult
        self.kc_length = kc_length
        self.kc_mult = kc_mult
        self.use_true_range = use_true_range
    
    def calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """True Range hesapla"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Squeeze Momentum hesapla"""
        try:
            # Kopya al
            data = df.copy()
            
            # Kaynak fiyat (close)
            source = data['close']
            
            # Bollinger Bands
            bb_basis = source.rolling(window=self.bb_length).mean()
            bb_std = source.rolling(window=self.bb_length).std()
            bb_upper = bb_basis + (bb_std * self.bb_mult)
            bb_lower = bb_basis - (bb_std * self.bb_mult)
            
            # Keltner Channels
            kc_basis = source.rolling(window=self.kc_length).mean()
            
            if self.use_true_range:
                # True Range kullan
                tr = self.calculate_true_range(data)
                range_ma = tr.rolling(window=self.kc_length).mean()
            else:
                # High-Low range kullan
                range_ma = (data['high'] - data['low']).rolling(window=self.kc_length).mean()
            
            kc_upper = kc_basis + (range_ma * self.kc_mult)
            kc_lower = kc_basis - (range_ma * self.kc_mult)
            
            # Squeeze tespiti
            # Squeeze ON: BB tamamen KC'nin içinde
            data['squeeze_on'] = ((bb_lower > kc_lower) & (bb_upper < kc_upper)).astype(int)
            
            # Squeeze OFF: BB, KC'nin dışında
            data['squeeze_off'] = ((bb_lower < kc_lower) | (bb_upper > kc_upper)).astype(int)
            
            # No Squeeze
            data['no_squeeze'] = ((data['squeeze_on'] == 0) & (data['squeeze_off'] == 0)).astype(int)
            
            # Momentum hesaplama (Linear Regression değeri)
            # Basitleştirilmiş momentum: (close - n periyot önceki ortalama)
            momentum_period = self.bb_length
            data['momentum'] = source - source.rolling(window=momentum_period).mean()
            
            # Momentum histogramı için renk kodlama
            data['momentum_color'] = 'gray'
            # Pozitif momentum
            data.loc[data['momentum'] > 0, 'momentum_color'] = 'green'
            data.loc[(data['momentum'] > 0) & (data['momentum'] > data['momentum'].shift(1)), 'momentum_color'] = 'lime'
            # Negatif momentum
            data.loc[data['momentum'] < 0, 'momentum_color'] = 'red'
            data.loc[(data['momentum'] < 0) & (data['momentum'] < data['momentum'].shift(1)), 'momentum_color'] = 'maroon'
            
            # Momentum değişim yönü
            data['momentum_increasing'] = (data['momentum'] > data['momentum'].shift(1)).astype(int)
            data['momentum_decreasing'] = (data['momentum'] < data['momentum'].shift(1)).astype(int)
            
            # Squeeze durumu değişimleri
            data['squeeze_release'] = ((data['squeeze_on'].shift(1) == 1) & 
                                      (data['squeeze_on'] == 0)).astype(int)
            data['squeeze_start'] = ((data['squeeze_on'].shift(1) == 0) & 
                                    (data['squeeze_on'] == 1)).astype(int)
            
            # Sinyaller
            # Squeeze sonrası momentum yönüne göre
            data['sqz_buy_signal'] = (data['squeeze_release'] & 
                                      (data['momentum'] > 0)).astype(int)
            data['sqz_sell_signal'] = (data['squeeze_release'] & 
                                       (data['momentum'] < 0)).astype(int)
            
            # Bollinger ve Keltner değerlerini de sakla (analiz için)
            data['bb_upper'] = bb_upper
            data['bb_lower'] = bb_lower
            data['bb_basis'] = bb_basis
            data['kc_upper'] = kc_upper
            data['kc_lower'] = kc_lower
            data['kc_basis'] = kc_basis
            
            # Sadece hesaplanan sütunları döndür
            result_columns = ['squeeze_on', 'squeeze_off', 'no_squeeze', 'momentum', 
                            'momentum_color', 'momentum_increasing', 'momentum_decreasing',
                            'squeeze_release', 'squeeze_start', 'sqz_buy_signal', 'sqz_sell_signal',
                            'bb_upper', 'bb_lower', 'bb_basis', 'kc_upper', 'kc_lower', 'kc_basis']
            
            return data[result_columns]
            
        except Exception as e:
            logger.error(f"Error calculating Squeeze Momentum: {e}")
            return pd.DataFrame()
    
    def get_signal(self, latest_values: Dict[str, Any]) -> str:
        """Son değerlere göre sinyal üret"""
        try:
            # Direkt alım/satım sinyalleri
            if latest_values.get('sqz_buy_signal', 0) == 1:
                return 'BUY'
            elif latest_values.get('sqz_sell_signal', 0) == 1:
                return 'SELL'
            
            # Squeeze durumu
            squeeze_on = latest_values.get('squeeze_on', 0)
            momentum = latest_values.get('momentum', 0)
            momentum_color = latest_values.get('momentum_color', 'gray')
            
            if squeeze_on:
                return 'SQUEEZE_ON'  # Sıkışma devam ediyor, bekle
            else:
                # Momentum yönüne göre
                if momentum > 0:
                    if momentum_color == 'lime':
                        return 'STRONG_BULLISH'  # Güçlü yükseliş momentumu
                    else:
                        return 'BULLISH'  # Yükseliş momentumu
                elif momentum < 0:
                    if momentum_color == 'maroon':
                        return 'STRONG_BEARISH'  # Güçlü düşüş momentumu
                    else:
                        return 'BEARISH'  # Düşüş momentumu
                else:
                    return 'NEUTRAL'
                    
        except Exception as e:
            logger.error(f"Error getting Squeeze signal: {e}")
            return 'ERROR'