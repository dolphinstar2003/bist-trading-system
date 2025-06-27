import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Union
from utils.logger import get_logger

logger = get_logger(__name__)


class CSVDataManager:
    """CSV dosyaları ile veri yönetimi"""
    
    def __init__(self):
        # Settings'den yapılandırmayı yükle
        with open("settings.json", "r") as f:
            self.settings = json.load(f)
        
        # Veri dizinlerini ayarla
        self.raw_data_path = Path(self.settings["paths"]["data_raw"])
        self.processed_data_path = Path(self.settings["paths"]["data_processed"])
        self.indicators_data_path = Path(self.settings["paths"]["data_indicators"])
        
        # Dizinleri oluştur
        for path in [self.raw_data_path, self.processed_data_path, self.indicators_data_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Hisse listesi
        self.symbols = self.settings["trading"]["symbols"]
        
        logger.info(f"CSV Data Manager başlatıldı. {len(self.symbols)} hisse takip ediliyor.")
    
    def save_raw_data(self, symbol: str, data: pd.DataFrame, timeframe: str = "1d"):
        """Ham veriyi CSV olarak kaydet"""
        try:
            filename = f"{symbol}_{timeframe}_raw.csv"
            filepath = self.raw_data_path / filename
            
            # Mevcut veriyi yükle
            if filepath.exists():
                existing_data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # Timezone uyumluluğunu sağla
                # hasattr ile tz attribute'ünün varlığını kontrol et
                has_existing_tz = hasattr(existing_data.index, 'tz') and existing_data.index.tz is not None
                has_data_tz = hasattr(data.index, 'tz') and data.index.tz is not None
                
                if not has_existing_tz and has_data_tz:
                    # Existing data tz-naive, new data tz-aware
                    existing_data.index = existing_data.index.tz_localize('UTC')
                elif has_existing_tz and not has_data_tz:
                    # Existing data tz-aware, new data tz-naive
                    data.index = data.index.tz_localize('UTC')
                elif has_existing_tz and has_data_tz:
                    # Both tz-aware, convert to same timezone if different
                    if existing_data.index.tz != data.index.tz:
                        data.index = data.index.tz_convert('UTC')
                        existing_data.index = existing_data.index.tz_convert('UTC')
                
                # Yeni veriyi ekle (duplicate'leri kaldır)
                data = pd.concat([existing_data, data])
                data = data[~data.index.duplicated(keep='last')]
                data = data.sort_index()
            
            # Kaydet
            data.to_csv(filepath)
            logger.info(f"Raw data saved: {symbol} {timeframe} - {len(data)} rows")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving raw data for {symbol}: {str(e)}")
            return False
    
    def load_raw_data(self, symbol: str, timeframe: str = "1d", 
                      start_date: Optional[str] = None, 
                      end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Ham veriyi CSV'den yükle"""
        try:
            filename = f"{symbol}_{timeframe}_raw.csv"
            filepath = self.raw_data_path / filename
            
            if not filepath.exists():
                logger.warning(f"Raw data file not found: {filename}")
                return None
            
            # Veriyi yükle
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Eğer timezone bilgisi yoksa UTC olarak ayarla
            try:
                if hasattr(data.index, 'tz') and data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
            except AttributeError:
                # Index datetime değilse, önce datetime'a çevir
                data.index = pd.to_datetime(data.index)
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
            
            # Tarih filtreleme
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            logger.info(f"Raw data loaded: {symbol} {timeframe} - {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Error loading raw data for {symbol}: {str(e)}")
            return None
    
    def save_indicators(self, symbol: str, indicators: pd.DataFrame, timeframe: str = "1d"):
        """İndikatör verilerini kaydet"""
        try:
            filename = f"{symbol}_{timeframe}_indicators.csv"
            filepath = self.indicators_data_path / filename
            
            indicators.to_csv(filepath)
            logger.info(f"Indicators saved: {symbol} {timeframe} - {indicators.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving indicators for {symbol}: {str(e)}")
            return False
    
    def load_indicators(self, symbol: str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        """İndikatör verilerini yükle"""
        try:
            filename = f"{symbol}_{timeframe}_indicators.csv"
            filepath = self.indicators_data_path / filename
            
            if not filepath.exists():
                logger.warning(f"Indicators file not found: {filename}")
                return None
            
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Eğer timezone bilgisi yoksa UTC olarak ayarla
            try:
                if hasattr(data.index, 'tz') and data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
            except AttributeError:
                # Index datetime değilse, önce datetime'a çevir
                data.index = pd.to_datetime(data.index)
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
                
            logger.info(f"Indicators loaded: {symbol} {timeframe} - {data.shape}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading indicators for {symbol}: {str(e)}")
            return None
    
    def load_indicator_data(self, symbol: str, timeframe: str, indicator_name: str) -> Optional[pd.DataFrame]:
        """Tek bir indikatör dosyasını yükle"""
        file_path = self.indicators_data_path / f"{symbol}_{timeframe}_{indicator_name}.csv"
        
        if not file_path.exists():
            logger.debug(f"Indicator file not found: {file_path.name}")
            return None
        
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            
            # Eğer timezone bilgisi yoksa UTC olarak ayarla
            if hasattr(df.index, 'tz') and df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            logger.debug(f"Indicator loaded: {symbol} {timeframe} {indicator_name} - Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading indicator {indicator_name} for {symbol}: {str(e)}")
            return None
    
    def combine_data(self, symbol: str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        """Ham veri ve indikatörleri birleştir"""
        try:
            # Ham veriyi yükle
            raw_data = self.load_raw_data(symbol, timeframe)
            if raw_data is None:
                return None
            
            # İndikatörleri yükle
            indicators = self.load_indicators(symbol, timeframe)
            
            # Birleştir
            if indicators is not None:
                combined = pd.concat([raw_data, indicators], axis=1)
                # Duplicate sütunları kaldır
                combined = combined.loc[:, ~combined.columns.duplicated()]
            else:
                combined = raw_data
            
            logger.info(f"Combined data: {symbol} {timeframe} - {combined.shape}")
            return combined
            
        except Exception as e:
            logger.error(f"Error combining data for {symbol}: {str(e)}")
            return None
    
    def get_latest_data(self, symbol: str, timeframe: str = "1d", 
                       periods: int = 100) -> Optional[pd.DataFrame]:
        """Son N periyot veriyi getir"""
        try:
            data = self.combine_data(symbol, timeframe)
            if data is None:
                return None
            
            # Son N periyodu al
            latest_data = data.tail(periods)
            
            return latest_data
            
        except Exception as e:
            logger.error(f"Error getting latest data for {symbol}: {str(e)}")
            return None
    
    def save_trade_history(self, trades: pd.DataFrame):
        """İşlem geçmişini kaydet"""
        try:
            filepath = self.processed_data_path / "trade_history.csv"
            
            if filepath.exists():
                existing_trades = pd.read_csv(filepath, index_col=0, parse_dates=True)
                
                # Timezone uyumluluğunu sağla
                if existing_trades.index.tz is None and trades.index.tz is not None:
                    existing_trades.index = existing_trades.index.tz_localize('UTC')
                elif existing_trades.index.tz is not None and trades.index.tz is None:
                    trades.index = trades.index.tz_localize('UTC')
                elif existing_trades.index.tz is not None and trades.index.tz is not None:
                    if existing_trades.index.tz != trades.index.tz:
                        trades.index = trades.index.tz_convert('UTC')
                        existing_trades.index = existing_trades.index.tz_convert('UTC')
                
                trades = pd.concat([existing_trades, trades])
                trades = trades[~trades.index.duplicated(keep='last')]
            
            trades.to_csv(filepath)
            logger.info(f"Trade history saved: {len(trades)} trades")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving trade history: {str(e)}")
            return False
    
    def load_trade_history(self, start_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """İşlem geçmişini yükle"""
        try:
            filepath = self.processed_data_path / "trade_history.csv"
            
            if not filepath.exists():
                logger.warning("Trade history file not found")
                return pd.DataFrame()
            
            trades = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            # Eğer timezone bilgisi yoksa UTC olarak ayarla
            if trades.index.tz is None:
                trades.index = trades.index.tz_localize('UTC')
            
            if start_date:
                trades = trades[trades.index >= start_date]
            
            logger.info(f"Trade history loaded: {len(trades)} trades")
            return trades
            
        except Exception as e:
            logger.error(f"Error loading trade history: {str(e)}")
            return None
    
    def get_data_info(self) -> Dict:
        """Mevcut veri durumu hakkında bilgi"""
        info = {
            "symbols": {},
            "total_files": 0,
            "total_size_mb": 0
        }
        
        for symbol in self.symbols:
            symbol_info = {
                "raw_data": {},
                "indicators": {}
            }
            
            # Her timeframe için kontrol
            for timeframe in self.settings["trading"]["timeframes"]:
                # Raw data
                raw_file = self.raw_data_path / f"{symbol}_{timeframe}_raw.csv"
                if raw_file.exists():
                    size_mb = raw_file.stat().st_size / (1024 * 1024)
                    symbol_info["raw_data"][timeframe] = {
                        "exists": True,
                        "size_mb": round(size_mb, 2),
                        "modified": datetime.fromtimestamp(raw_file.stat().st_mtime).isoformat()
                    }
                    info["total_files"] += 1
                    info["total_size_mb"] += size_mb
                
                # Indicators
                ind_file = self.indicators_data_path / f"{symbol}_{timeframe}_indicators.csv"
                if ind_file.exists():
                    size_mb = ind_file.stat().st_size / (1024 * 1024)
                    symbol_info["indicators"][timeframe] = {
                        "exists": True,
                        "size_mb": round(size_mb, 2),
                        "modified": datetime.fromtimestamp(ind_file.stat().st_mtime).isoformat()
                    }
                    info["total_files"] += 1
                    info["total_size_mb"] += size_mb
            
            info["symbols"][symbol] = symbol_info
        
        info["total_size_mb"] = round(info["total_size_mb"], 2)
        
        return info
    
    def cleanup_old_data(self, days: int = 365):
        """Eski verileri temizle"""
        try:
            # UTC timezone ile cutoff date oluştur
            cutoff_date = pd.Timestamp.now(tz='UTC') - timedelta(days=days)
            cleaned_count = 0
            
            for symbol in self.symbols:
                for timeframe in self.settings["trading"]["timeframes"]:
                    # Raw data
                    raw_data = self.load_raw_data(symbol, timeframe)
                    if raw_data is not None:
                        before_len = len(raw_data)
                        raw_data = raw_data[raw_data.index >= cutoff_date]
                        after_len = len(raw_data)
                        
                        if before_len > after_len:
                            self.save_raw_data(symbol, raw_data, timeframe)
                            cleaned_count += before_len - after_len
                            logger.info(f"Cleaned {before_len - after_len} old records from {symbol} {timeframe}")
            
            logger.info(f"Total cleaned records: {cleaned_count}")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning old data: {str(e)}")
            return 0


if __name__ == "__main__":
    # Test
    manager = CSVDataManager()
    
    # Test data oluştur
    test_data = pd.DataFrame({
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=pd.date_range('2024-01-01', periods=100, freq='D', tz='UTC'))
    
    # Test kaydet
    manager.save_raw_data("THYAO", test_data, "1d")
    
    # Test yükle
    loaded_data = manager.load_raw_data("THYAO", "1d")
    print(f"Loaded data shape: {loaded_data.shape}")
    
    # Veri bilgisi
    info = manager.get_data_info()
    print(f"Data info: {json.dumps(info, indent=2)}")