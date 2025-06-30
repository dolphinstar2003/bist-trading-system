"""
CSV Data Manager - Mevcut sistemle uyumlu
/home/yunus/Belgeler/New_Start/data/ yapısını kullanır
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger
import os
import sys

# Add parent path
sys.path.append(str(Path(__file__).parent.parent))
from utils.data_utils import read_csv_with_date_index, align_timezone


class CSVDataManager:
    """Mevcut CSV veri yapısı ile uyumlu data manager"""
    
    def __init__(self, base_path: str = "/home/yunus/Belgeler/New_Start"):
        self.base_path = Path(base_path)
        self.raw_data_path = self.base_path / "data" / "raw"
        self.indicators_path = self.base_path / "data" / "indicators"
        self.analysis_path = self.base_path / "data" / "analysis"
        
        # New_Plan altındaki indicators path
        self.new_indicators_path = self.base_path / "New_Plan" / "data" / "indicators"
        self.new_indicators_path.mkdir(parents=True, exist_ok=True)
        
        # Cache for loaded data
        self.cache = {}
        self.cache_ttl = 900  # 15 dakika
        self.cache_timestamps = {}
        
        logger.info(f"CSVDataManager başlatıldı - Base: {self.base_path}")
        logger.info(f"Raw data: {self.raw_data_path}")
        logger.info(f"Indicators: {self.indicators_path}")
    
    def get_raw_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Mevcut raw data'yı oku"""
        cache_key = f"raw_{symbol}_{timeframe}"
        
        # Cache kontrolü
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Dosya yolu
            filename = f"{symbol}_{timeframe}_raw.csv"
            filepath = self.raw_data_path / filename
            
            if not filepath.exists():
                logger.warning(f"Raw data bulunamadı: {filepath}")
                return None
            
            # CSV oku with proper date handling
            df = read_csv_with_date_index(filepath)
            
            # Kolon isimlerini küçük harf yap
            df.columns = [col.lower() for col in df.columns]
            
            # Temel kontroller
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Eksik kolonlar: {filepath}")
                return None
            
            # Cache'e kaydet
            self._update_cache(cache_key, df)
            
            logger.debug(f"Raw data yüklendi: {symbol} {timeframe} - {len(df)} satır")
            return df
            
        except Exception as e:
            logger.error(f"Raw data okuma hatası {symbol} {timeframe}: {e}")
            return None
    
    def get_indicator_data(self, symbol: str, timeframe: str, indicator: str) -> Optional[pd.DataFrame]:
        """Mevcut indicator data'yı oku"""
        cache_key = f"ind_{symbol}_{timeframe}_{indicator}"
        
        # Cache kontrolü
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]
        
        try:
            # Dosya yolu
            filename = f"{symbol}_{timeframe}_{indicator}.csv"
            filepath = self.indicators_path / filename
            
            if not filepath.exists():
                logger.debug(f"Indicator data bulunamadı: {filepath}")
                return None
            
            # CSV oku with proper date handling
            df = read_csv_with_date_index(filepath)
            
            # Cache'e kaydet
            self._update_cache(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Indicator data okuma hatası: {e}")
            return None
    
    def save_indicator_data(self, symbol: str, timeframe: str, indicator: str, 
                          data: pd.DataFrame, use_new_path: bool = True) -> bool:
        """Indicator data'yı kaydet"""
        try:
            # Hangi path kullanılacak
            if use_new_path:
                save_path = self.new_indicators_path
            else:
                save_path = self.indicators_path
            
            # Dosya yolu
            filename = f"{symbol}_{timeframe}_{indicator}.csv"
            filepath = save_path / filename
            
            # Kaydet
            data.to_csv(filepath)
            
            logger.debug(f"Indicator kaydedildi: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Indicator kaydetme hatası: {e}")
            return False
    
    def get_multi_timeframe_data(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """Bir sembol için multiple timeframe verisi"""
        data = {}
        
        for tf in timeframes:
            df = self.get_raw_data(symbol, tf)
            if df is not None:
                data[tf] = df
            else:
                logger.warning(f"{symbol} {tf} verisi bulunamadı")
        
        return data
    
    def get_all_indicators(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Bir sembol için tüm indicator'ları birleştir"""
        # Raw data'yı al
        raw_data = self.get_raw_data(symbol, timeframe)
        if raw_data is None:
            return pd.DataFrame()
        
        # Base dataframe
        combined = raw_data.copy()
        
        # Mevcut indicator dosyalarını bul
        pattern = f"{symbol}_{timeframe}_*.csv"
        indicator_files = list(self.indicators_path.glob(pattern))
        
        for file_path in indicator_files:
            # Indicator adını çıkar
            indicator_name = file_path.stem.replace(f"{symbol}_{timeframe}_", "")
            
            # Indicator verisini oku
            ind_data = self.get_indicator_data(symbol, timeframe, indicator_name)
            
            if ind_data is not None:
                # Kolon isimlerini prefix ile güncelle
                ind_data.columns = [f"{indicator_name}_{col}" if col not in ['open', 'high', 'low', 'close', 'volume'] 
                                   else col for col in ind_data.columns]
                
                # Ensure timezone consistency before joining
                combined = align_timezone(combined)
                ind_data = align_timezone(ind_data)
                
                # Birleştir
                combined = combined.join(ind_data, how='left', rsuffix='_dup')
                
                # Duplicate kolonları kaldır
                combined = combined.loc[:, ~combined.columns.str.endswith('_dup')]
        
        logger.info(f"{symbol} {timeframe} için {len(combined.columns)} kolon birleştirildi")
        return combined
    
    def get_analysis_data(self, filename: str) -> Optional[pd.DataFrame]:
        """Analysis klasöründen veri oku (optimal parameters vs.)"""
        try:
            filepath = self.analysis_path / filename
            
            if not filepath.exists():
                logger.warning(f"Analysis dosyası bulunamadı: {filepath}")
                return None
            
            df = pd.read_csv(filepath)
            return df
            
        except Exception as e:
            logger.error(f"Analysis data okuma hatası: {e}")
            return None
    
    def align_multi_timeframe_data(self, data_dict: Dict[str, pd.DataFrame], 
                                  base_timeframe: str = '15m') -> pd.DataFrame:
        """Farklı timeframe verilerini hizala"""
        if base_timeframe not in data_dict:
            raise ValueError(f"Base timeframe {base_timeframe} bulunamadı")
        
        # Base dataframe
        aligned = data_dict[base_timeframe].copy()
        
        # Diğer timeframe'leri ekle
        for tf, df in data_dict.items():
            if tf == base_timeframe:
                continue
            
            # Prefix ekle
            df_prefixed = df.add_prefix(f"{tf}_")
            
            # Forward fill ile hizala
            aligned = aligned.join(df_prefixed, how='left')
            aligned.fillna(method='ffill', inplace=True)
        
        return aligned
    
    def get_latest_data(self, symbol: str, timeframe: str, periods: int = 100) -> pd.DataFrame:
        """Son N periyot veriyi getir"""
        df = self.get_raw_data(symbol, timeframe)
        
        if df is not None:
            return df.tail(periods)
        
        return pd.DataFrame()
    
    def _is_cache_valid(self, key: str) -> bool:
        """Cache geçerli mi kontrol et"""
        if key not in self.cache:
            return False
        
        # TTL kontrolü
        if key in self.cache_timestamps:
            age = (datetime.now() - self.cache_timestamps[key]).total_seconds()
            return age < self.cache_ttl
        
        return False
    
    def _update_cache(self, key: str, data: pd.DataFrame):
        """Cache güncelle"""
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()
    
    def clear_cache(self):
        """Cache temizle"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Cache temizlendi")
    
    def get_available_symbols(self) -> List[str]:
        """Mevcut sembol listesini al"""
        symbols = set()
        
        # Raw data klasöründeki dosyalardan sembolleri çıkar
        for file_path in self.raw_data_path.glob("*_raw.csv"):
            # Dosya adından sembolü çıkar
            parts = file_path.stem.split('_')
            if len(parts) >= 2:
                symbol = parts[0]
                symbols.add(symbol)
        
        return sorted(list(symbols))
    
    def get_available_timeframes(self, symbol: str) -> List[str]:
        """Bir sembol için mevcut timeframe'leri al"""
        timeframes = []
        
        # Raw data dosyalarını kontrol et
        for file_path in self.raw_data_path.glob(f"{symbol}_*_raw.csv"):
            parts = file_path.stem.split('_')
            if len(parts) >= 2:
                tf = parts[1]
                timeframes.append(tf)
        
        return sorted(timeframes)
    
    def check_data_integrity(self, symbol: str, timeframe: str) -> Dict[str, any]:
        """Veri bütünlüğünü kontrol et"""
        df = self.get_raw_data(symbol, timeframe)
        
        if df is None:
            return {'exists': False}
        
        integrity = {
            'exists': True,
            'rows': len(df),
            'start_date': df.index[0],
            'end_date': df.index[-1],
            'missing_dates': 0,
            'null_values': df.isnull().sum().to_dict(),
            'duplicates': df.index.duplicated().sum()
        }
        
        # Tarih boşluklarını kontrol et
        date_range = pd.date_range(start=df.index[0], end=df.index[-1], freq='D')
        missing_dates = len(date_range) - len(df)
        integrity['missing_dates'] = missing_dates
        
        return integrity