#!/usr/bin/env python3
"""
Gelişmiş Veri İndirme Sistemi
- Paralel işleme (multi-threading)
- Cache mekanizması
- Incremental download (kaldığı yerden devam)
- Progress tracking
- Rate limiting
"""

import os
import sys
import time
import json
import threading
from queue import Queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle

from algolab_wrapper import AlgoLabWrapper
from utils.csv_data_manager import CSVDataManager


class DataDownloadManager:
    """Gelişmiş veri indirme yöneticisi"""
    
    def __init__(self, max_workers: int = 5):
        """
        Args:
            max_workers: Maksimum paralel işlem sayısı
        """
        self.api = AlgoLabWrapper()
        self.csv_manager = CSVDataManager()
        self.max_workers = max_workers
        
        # Cache dizini
        self.cache_dir = Path("data/.cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking
        self.progress_file = self.cache_dir / "download_progress.json"
        self.progress = self.load_progress()
        
        # Rate limiting
        self.rate_limiter = threading.Semaphore(1)  # 5 saniyede 1 istek
        self.last_request_time = 0
        self.min_request_interval = 5  # saniye
        
    def load_progress(self) -> Dict:
        """İndirme ilerlemesini yükle"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_progress(self):
        """İndirme ilerlemesini kaydet"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2, default=str)
    
    def get_last_data_date(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Bir sembol için en son indirilen verinin tarihini al"""
        try:
            df = self.csv_manager.load_raw_data(symbol, timeframe)
            if df is not None and len(df) > 0:
                # DataFrame index'i datetime olmalı
                return df.index[-1].to_pydatetime()
        except Exception as e:
            logger.debug(f"No existing data for {symbol} {timeframe}: {e}")
        return None
    
    def calculate_date_ranges(self, symbol: str, timeframe: str, 
                            start_date: datetime, end_date: datetime) -> List[Tuple[datetime, datetime]]:
        """
        İndirilecek tarih aralıklarını hesapla
        - Mevcut veriyi kontrol et
        - Eksik aralıkları belirle
        - Chunk'lara böl (her chunk max 250 bar)
        """
        ranges = []
        
        # Mevcut en son veriyi kontrol et
        last_date = self.get_last_data_date(symbol, timeframe)
        
        if last_date:
            # Kaldığı yerden devam et
            start_date = max(start_date, last_date + timedelta(minutes=1))
            logger.info(f"{symbol} {timeframe}: Continuing from {last_date}")
        
        # Eğer başlangıç bitiş tarihinden büyükse, veri güncel
        if start_date >= end_date:
            logger.info(f"{symbol} {timeframe}: Data is up to date")
            return []
        
        # Tarih aralığını chunk'lara böl
        # Her timeframe için farklı chunk boyutu
        chunk_sizes = {
            '15m': timedelta(days=30),   # 15m için 30 günlük chunk
            '1h': timedelta(days=60),    # 1h için 60 günlük chunk
            '4h': timedelta(days=120),   # 4h için 120 günlük chunk
            '1d': timedelta(days=250)    # 1d için 250 günlük chunk
        }
        
        chunk_size = chunk_sizes.get(timeframe, timedelta(days=30))
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + chunk_size, end_date)
            ranges.append((current_start, current_end))
            current_start = current_end + timedelta(minutes=1)
        
        return ranges
    
    def download_chunk(self, symbol: str, timeframe: str, 
                      start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Tek bir veri chunk'ını indir"""
        
        # Rate limiting
        with self.rate_limiter:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
            
            # Cache key
            cache_key = f"{symbol}_{timeframe}_{start_date.date()}_{end_date.date()}"
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            # Cache'i kontrol et
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        df = pickle.load(f)
                    logger.debug(f"Loaded from cache: {cache_key}")
                    return df
                except Exception as e:
                    logger.warning(f"Cache read error: {e}")
            
            # API'den veri çek
            try:
                # AlgoLab API'si start/end date parametresi almıyor
                # Sadece bar sayısı ile çalışıyor, bu yüzden get_market_data kullanıyoruz
                df = self.api.get_market_data(symbol, timeframe)
                
                if df is not None and len(df) > 0:
                    # Tarih filtreleme yap
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    df = df[mask]
                    
                    if len(df) > 0:
                        # Cache'e kaydet
                        with open(cache_file, 'wb') as f:
                            pickle.dump(df, f)
                        
                        return df
                
            except Exception as e:
                logger.error(f"Error downloading {symbol} {timeframe} {start_date}-{end_date}: {e}")
            
            return None
    
    def download_symbol_data(self, symbol: str, timeframe: str, 
                           start_date: datetime, end_date: datetime) -> int:
        """
        Bir sembol için veri indir
        Returns: İndirilen bar sayısı
        """
        logger.info(f"Downloading {symbol} {timeframe} data...")
        
        # Tarih aralıklarını hesapla
        date_ranges = self.calculate_date_ranges(symbol, timeframe, start_date, end_date)
        
        if not date_ranges:
            return 0
        
        total_bars = 0
        all_data = []
        
        # Her chunk'ı sırayla indir
        for i, (chunk_start, chunk_end) in enumerate(date_ranges):
            logger.info(f"{symbol} {timeframe}: Chunk {i+1}/{len(date_ranges)} "
                       f"({chunk_start.date()} to {chunk_end.date()})")
            
            df = self.download_chunk(symbol, timeframe, chunk_start, chunk_end)
            
            if df is not None and len(df) > 0:
                all_data.append(df)
                total_bars += len(df)
                
                # Her chunk sonrası progress güncelle
                progress_key = f"{symbol}_{timeframe}"
                self.progress[progress_key] = {
                    'last_date': str(df.index[-1]),
                    'total_bars': total_bars,
                    'last_update': str(datetime.now())
                }
                self.save_progress()
        
        # Tüm chunk'ları birleştir ve kaydet
        if all_data:
            combined_df = pd.concat(all_data)
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df.sort_index(inplace=True)
            
            # CSV'ye kaydet (append mode)
            self.csv_manager.save_raw_data(symbol, combined_df, timeframe)
            logger.success(f"{symbol} {timeframe}: {total_bars} bars downloaded")
        
        return total_bars
    
    def download_all_parallel(self, symbols: List[str], timeframes: List[str],
                            start_date: datetime, end_date: datetime):
        """Tüm sembolleri paralel olarak indir"""
        
        # API bağlantısı
        if not self.api.connect():
            logger.error("API connection failed!")
            return
        
        # İndirme görevlerini oluştur
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                tasks.append((symbol, timeframe, start_date, end_date))
        
        logger.info(f"Total download tasks: {len(tasks)}")
        
        # Progress tracking
        completed = 0
        total = len(tasks)
        start_time = time.time()
        
        # Paralel indirme
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Görevleri submit et
            future_to_task = {
                executor.submit(self.download_symbol_data, *task): task 
                for task in tasks
            }
            
            # Sonuçları bekle
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                symbol, timeframe = task[0], task[1]
                
                try:
                    bars = future.result()
                    completed += 1
                    
                    # Progress göster
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = (total - completed) * avg_time
                    
                    logger.info(f"Progress: {completed}/{total} "
                              f"({completed/total*100:.1f}%) "
                              f"- ETA: {remaining/60:.1f} min")
                    
                except Exception as e:
                    logger.error(f"Error in {symbol} {timeframe}: {e}")
                    completed += 1
        
        # Özet
        total_time = time.time() - start_time
        logger.success(f"Download completed in {total_time/60:.1f} minutes")
        
        # Cache temizliği (7 günden eski cache dosyalarını sil)
        self.cleanup_old_cache()
        
        # API bağlantısını kapat
        self.api.disconnect()
    
    def cleanup_old_cache(self, days: int = 7):
        """Eski cache dosyalarını temizle"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_date.timestamp():
                cache_file.unlink()
                logger.debug(f"Deleted old cache: {cache_file.name}")


def main():
    """Ana program"""
    
    # Settings dosyasını yükle
    with open('settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    
    symbols = settings['trading']['symbols']
    
    # Timeframe'ler
    timeframes = ['15m', '1h', '4h', '1d']
    
    # Tarih aralığı (4 yıl geriye)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=4*365)
    
    logger.info(f"Download period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Symbols: {len(symbols)}")
    logger.info(f"Timeframes: {timeframes}")
    
    # Download manager
    manager = DataDownloadManager(max_workers=3)  # 3 paralel işlem
    
    # Kullanıcıdan onay al
    print("\nBu işlem uzun sürebilir. Devam etmek istiyor musunuz? (E/H): ", end='')
    if input().upper() != 'E':
        print("İptal edildi.")
        return
    
    # İndirmeyi başlat
    manager.download_all_parallel(symbols, timeframes, start_date, end_date)
    
    logger.success("Tüm veri indirme işlemi tamamlandı!")


if __name__ == "__main__":
    main()