"""
Supertrend Indicator
ATR tabanlı dinamik destek/direnç seviyeleri ile trend takibi
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from loguru import logger


class Supertrend:
    """Supertrend indikatörü"""
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        Args:
            period: ATR hesaplama periyodu
            multiplier: ATR çarpanı
        """
        self.period = period
        self.multiplier = multiplier
    
    def calculate_atr(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Average True Range hesapla"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Supertrend hesapla"""
        try:
            # Kopya al
            data = df.copy()
            
            # HL2 (Kaynak fiyat)
            hl2 = (data['high'] + data['low']) / 2
            
            # ATR hesapla
            data['atr'] = self.calculate_atr(data, self.period)
            
            # Temel üst ve alt bantlar
            data['basic_upper_band'] = hl2 + (self.multiplier * data['atr'])
            data['basic_lower_band'] = hl2 - (self.multiplier * data['atr'])
            
            # Final bantları hesaplamak için boş sütunlar
            data['final_upper_band'] = data['basic_upper_band']
            data['final_lower_band'] = data['basic_lower_band']
            data['supertrend'] = np.nan
            data['trend'] = 1  # 1: Yükseliş, -1: Düşüş
            
            # Supertrend hesaplama
            for i in range(self.period, len(data)):
                # Final Upper Band
                if data['basic_upper_band'].iloc[i] < data['final_upper_band'].iloc[i-1] or \
                   data['close'].iloc[i-1] > data['final_upper_band'].iloc[i-1]:
                    data.loc[data.index[i], 'final_upper_band'] = data['basic_upper_band'].iloc[i]
                else:
                    data.loc[data.index[i], 'final_upper_band'] = data['final_upper_band'].iloc[i-1]
                
                # Final Lower Band
                if data['basic_lower_band'].iloc[i] > data['final_lower_band'].iloc[i-1] or \
                   data['close'].iloc[i-1] < data['final_lower_band'].iloc[i-1]:
                    data.loc[data.index[i], 'final_lower_band'] = data['basic_lower_band'].iloc[i]
                else:
                    data.loc[data.index[i], 'final_lower_band'] = data['final_lower_band'].iloc[i-1]
                
                # Trend belirleme
                if i == self.period:  # İlk değer
                    if data['close'].iloc[i] <= data['final_upper_band'].iloc[i]:
                        data.loc[data.index[i], 'trend'] = -1
                        data.loc[data.index[i], 'supertrend'] = data['final_upper_band'].iloc[i]
                    else:
                        data.loc[data.index[i], 'trend'] = 1
                        data.loc[data.index[i], 'supertrend'] = data['final_lower_band'].iloc[i]
                else:
                    # Önceki trend yükseliş ise
                    if data['trend'].iloc[i-1] == 1:
                        if data['close'].iloc[i] <= data['final_lower_band'].iloc[i]:
                            data.loc[data.index[i], 'trend'] = -1
                            data.loc[data.index[i], 'supertrend'] = data['final_upper_band'].iloc[i]
                        else:
                            data.loc[data.index[i], 'trend'] = 1
                            data.loc[data.index[i], 'supertrend'] = data['final_lower_band'].iloc[i]
                    # Önceki trend düşüş ise
                    else:
                        if data['close'].iloc[i] >= data['final_upper_band'].iloc[i]:
                            data.loc[data.index[i], 'trend'] = 1
                            data.loc[data.index[i], 'supertrend'] = data['final_lower_band'].iloc[i]
                        else:
                            data.loc[data.index[i], 'trend'] = -1
                            data.loc[data.index[i], 'supertrend'] = data['final_upper_band'].iloc[i]
            
            # Trend değişim sinyalleri
            data['trend_changed'] = (data['trend'] != data['trend'].shift(1)).astype(int)
            data['buy_signal'] = ((data['trend'] == 1) & (data['trend'].shift(1) == -1)).astype(int)
            data['sell_signal'] = ((data['trend'] == -1) & (data['trend'].shift(1) == 1)).astype(int)
            
            # Trend yönü açıklaması
            data['trend_direction'] = data['trend'].map({1: 'bullish', -1: 'bearish'})
            
            # Stop loss seviyeleri
            data['stop_loss'] = data['supertrend']
            
            # Pozisyondan çıkış koşulu (fiyat supertrend'i kırdığında)
            data['exit_long'] = ((data['close'] < data['supertrend']) & 
                               (data['trend'] == -1)).astype(int)
            data['exit_short'] = ((data['close'] > data['supertrend']) & 
                                (data['trend'] == 1)).astype(int)
            
            # Supertrend rengi (görselleştirme için)
            data['supertrend_color'] = data['trend'].map({1: 'green', -1: 'red'})
            
            # Sadece hesaplanan sütunları döndür
            result_columns = ['supertrend', 'trend', 'trend_direction', 'atr',
                            'final_upper_band', 'final_lower_band',
                            'trend_changed', 'buy_signal', 'sell_signal',
                            'stop_loss', 'exit_long', 'exit_short', 'supertrend_color']
            
            return data[result_columns]
            
        except Exception as e:
            logger.error(f"Error calculating Supertrend: {e}")
            return pd.DataFrame()
    
    def get_signal(self, latest_values: Dict[str, Any]) -> str:
        """Son değerlere göre sinyal üret"""
        try:
            # Direkt alım/satım sinyalleri
            if latest_values.get('buy_signal', 0) == 1:
                return 'BUY'
            elif latest_values.get('sell_signal', 0) == 1:
                return 'SELL'
            
            # Çıkış sinyalleri
            if latest_values.get('exit_long', 0) == 1:
                return 'EXIT_LONG'
            elif latest_values.get('exit_short', 0) == 1:
                return 'EXIT_SHORT'
            
            # Trend durumu
            trend = latest_values.get('trend', 0)
            if trend == 1:
                return 'BULLISH_TREND'
            elif trend == -1:
                return 'BEARISH_TREND'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Error getting Supertrend signal: {e}")
            return 'ERROR'