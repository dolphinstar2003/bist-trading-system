"""
Williams VIX Fix - Market Bottoms Finder
Piyasa diplerini tespit etmek için volatilite tabanlı gösterge
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from loguru import logger


class WilliamsVixFix:
    """Williams VIX Fix indikatörü"""
    
    def __init__(self, pd: int = 22, bbl: int = 20, mult: float = 2.0, 
                 lb: int = 50, ph: float = 0.85, pl: float = 1.01):
        """
        Args:
            pd: Highest close için bakılan periyot
            bbl: Bollinger Band uzunluğu
            mult: Bollinger Band standart sapma çarpanı
            lb: Yüzdelik hesaplama için bakılan periyot
            ph: En yüksek yüzdelik eşik değeri
            pl: En düşük yüzdelik eşik değeri
        """
        self.pd = pd
        self.bbl = bbl
        self.mult = mult
        self.lb = lb
        self.ph = ph
        self.pl = pl
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Williams VIX Fix hesapla"""
        try:
            # Kopya al
            data = df.copy()
            
            # Williams VIX Fix formülü
            # wvf = ((highest(close, pd)-low)/(highest(close, pd)))*100
            highest_close = data['close'].rolling(window=self.pd).max()
            data['wvf'] = ((highest_close - data['low']) / highest_close) * 100
            
            # Bollinger Bands hesapla
            data['wvf_sma'] = data['wvf'].rolling(window=self.bbl).mean()
            data['wvf_std'] = data['wvf'].rolling(window=self.bbl).std()
            data['wvf_upper'] = data['wvf_sma'] + (self.mult * data['wvf_std'])
            data['wvf_lower'] = data['wvf_sma'] - (self.mult * data['wvf_std'])
            
            # Yüzdelik değerleri hesapla
            data['range_high'] = data['wvf'].rolling(window=self.lb).max() * self.ph
            data['range_low'] = data['wvf'].rolling(window=self.lb).min() * self.pl
            
            # Sinyalleri belirle
            # VIX Fix değeri üst band veya yüksek yüzdelik seviyeyi aştığında
            data['vix_fix_signal'] = (
                (data['wvf'] >= data['wvf_upper']) | 
                (data['wvf'] >= data['range_high'])
            ).astype(int)
            
            # Renk kodları (görselleştirme için)
            data['vix_fix_color'] = 'gray'  # Varsayılan
            data.loc[data['vix_fix_signal'] == 1, 'vix_fix_color'] = 'lime'  # Sinyal var
            data.loc[data['wvf'] >= data['wvf_upper'], 'vix_fix_color'] = 'red'  # Ekstrem
            
            # Sadece hesaplanan sütunları döndür
            result_columns = ['wvf', 'wvf_sma', 'wvf_upper', 'wvf_lower', 
                            'range_high', 'range_low', 'vix_fix_signal', 'vix_fix_color']
            
            return data[result_columns]
            
        except Exception as e:
            logger.error(f"Error calculating Williams VIX Fix: {e}")
            return pd.DataFrame()
    
    def get_signal(self, latest_values: Dict[str, Any]) -> str:
        """Son değerlere göre sinyal üret"""
        try:
            if latest_values.get('vix_fix_signal', 0) == 1:
                if latest_values.get('vix_fix_color') == 'red':
                    return 'EXTREME_FEAR'  # Ekstrem korku - güçlü dip sinyali
                else:
                    return 'FEAR'  # Korku - potansiyel dip
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Error getting VIX Fix signal: {e}")
            return 'ERROR'