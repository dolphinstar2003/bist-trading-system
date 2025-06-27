#!/usr/bin/env python3
"""
Incremental Veri İndirme Sistemi
- Geriye doğru tarih bazlı indirme
- Her seferinde belirli sayıda bar indir
- Kaldığı yerden devam et
- Otomatik retry mekanizması
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algolab.algolab_wrapper import AlgoLabWrapper
from utils.csv_data_manager import CSVDataManager


class IncrementalDownloader:
    """Geriye doğru incremental veri indirme"""
    
    def __init__(self):
        self.api = AlgoLabWrapper()
        self.csv_manager = CSVDataManager()
        
        # Progress tracking
        self.progress_dir = Path("data/.progress")
        self.progress_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.api_lock = threading.Lock()
        self.last_api_call = 0
        self.min_interval = 5.0  # saniye
        
    def get_progress(self, symbol: str, timeframe: str) -> Dict:
        """İndirme ilerlemesini al"""
        progress_file = self.progress_dir / f"{symbol}_{timeframe}.json"
        
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                return json.load(f)
        
        return {
            'oldest_date': None,
            'newest_date': None,
            'total_bars': 0,
            'last_update': None,
            'target_reached': False
        }
    
    def save_progress(self, symbol: str, timeframe: str, progress: Dict):
        """İndirme ilerlemesini kaydet"""
        progress_file = self.progress_dir / f"{symbol}_{timeframe}.json"
        
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2, default=str)
    
    def rate_limited_api_call(self, func, *args, **kwargs):
        """Rate limited API çağrısı"""
        with self.api_lock:
            # Rate limiting
            elapsed = time.time() - self.last_api_call
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                logger.debug(f"Rate limiting: waiting {sleep_time:.1f}s")
                time.sleep(sleep_time)
            
            result = func(*args, **kwargs)
            self.last_api_call = time.time()
            return result
    
    def download_backwards(self, symbol: str, timeframe: str, 
                         target_date: datetime, batch_size: int = 250) -> bool:
        """
        Geriye doğru veri indir
        
        Args:
            symbol: Hisse kodu
            timeframe: Zaman dilimi
            target_date: Hedef tarih (bu tarihe kadar indir)
            batch_size: Her seferde indirilecek bar sayısı
            
        Returns:
            bool: Hedef tarihe ulaşıldı mı?
        """
        progress = self.get_progress(symbol, timeframe)
        
        # Hedef tarihe ulaşmış mı kontrol et
        if progress.get('target_reached'):
            logger.info(f"{symbol} {timeframe}: Already reached target date")
            return True
        
        # Mevcut veriyi kontrol et
        existing_df = self.csv_manager.load_raw_data(symbol, timeframe)
        
        if existing_df is not None and len(existing_df) > 0:
            oldest_existing = existing_df.index[0]
            newest_existing = existing_df.index[-1]
            
            # Hedef tarihe ulaşmış mı?
            if oldest_existing <= target_date:
                progress['target_reached'] = True
                progress['oldest_date'] = str(oldest_existing)
                self.save_progress(symbol, timeframe, progress)
                logger.success(f"{symbol} {timeframe}: Target date reached! "
                             f"Oldest: {oldest_existing.date()}")
                return True
        else:
            oldest_existing = None
            newest_existing = None
        
        # API'den veri çek
        try:
            logger.info(f"{symbol} {timeframe}: Downloading {batch_size} bars...")
            
            # Rate limited API call
            df = self.rate_limited_api_call(
                self.api.get_market_data, 
                symbol, 
                timeframe,
                bar_count=batch_size
            )
            
            if df is None or len(df) == 0:
                logger.warning(f"{symbol} {timeframe}: No data received")
                return False
            
            # Yeni verinin tarih aralığı
            new_oldest = df.index[0]
            new_newest = df.index[-1]
            
            logger.info(f"{symbol} {timeframe}: Got data from "
                       f"{new_oldest.date()} to {new_newest.date()} "
                       f"({len(df)} bars)")
            
            # Mevcut veri ile birleştir
            if existing_df is not None:
                # Yeni veri daha eski mi kontrol et
                if new_newest < oldest_existing:
                    # Tamamen eski veri, başa ekle
                    combined_df = pd.concat([df, existing_df])
                else:
                    # Overlap var, birleştir
                    combined_df = pd.concat([existing_df, df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                
                combined_df.sort_index(inplace=True)
            else:
                combined_df = df
            
            # CSV'ye kaydet
            self.csv_manager.save_raw_data(symbol, combined_df, timeframe)
            
            # Progress güncelle
            progress['oldest_date'] = str(combined_df.index[0])
            progress['newest_date'] = str(combined_df.index[-1])
            progress['total_bars'] = len(combined_df)
            progress['last_update'] = str(datetime.now())
            
            # Hedef tarihe ulaştı mı?
            if combined_df.index[0] <= target_date:
                progress['target_reached'] = True
                logger.success(f"{symbol} {timeframe}: Target date reached!")
                
            self.save_progress(symbol, timeframe, progress)
            
            return progress['target_reached']
            
        except Exception as e:
            logger.error(f"{symbol} {timeframe}: Download error: {e}")
            return False
    
    def download_symbol_full_history(self, symbol: str, timeframe: str, 
                                   target_date: datetime, max_retries: int = 3):
        """Bir sembol için tüm geçmişi indir"""
        
        retries = 0
        
        while retries < max_retries:
            try:
                # Target'a ulaşana kadar indir
                iterations = 0
                max_iterations = 100  # Sonsuz döngüyü önle
                
                while iterations < max_iterations:
                    reached = self.download_backwards(symbol, timeframe, target_date)
                    
                    if reached:
                        return True
                    
                    iterations += 1
                    
                    # Her 10 iterasyonda bir durum raporu
                    if iterations % 10 == 0:
                        progress = self.get_progress(symbol, timeframe)
                        logger.info(f"{symbol} {timeframe}: {progress['total_bars']} bars, "
                                  f"oldest: {progress['oldest_date']}")
                
                logger.warning(f"{symbol} {timeframe}: Max iterations reached")
                return False
                
            except Exception as e:
                retries += 1
                logger.error(f"{symbol} {timeframe}: Error (retry {retries}/{max_retries}): {e}")
                
                if retries < max_retries:
                    time.sleep(30)  # 30 saniye bekle
                    
        return False
    
    def download_all_incremental(self, symbols: List[str], timeframes: List[str],
                               years_back: int = 4):
        """Tüm sembolleri incremental olarak indir"""
        
        # API bağlantısı
        if not self.api.connect():
            logger.error("API connection failed!")
            return
        
        # Hedef tarih
        target_date = datetime.now() - timedelta(days=years_back*365)
        logger.info(f"Target date: {target_date.date()} ({years_back} years back)")
        
        # İstatistikler
        stats = {
            'total': len(symbols) * len(timeframes),
            'completed': 0,
            'failed': 0,
            'already_done': 0
        }
        
        start_time = time.time()
        
        # Her sembol ve timeframe için sırayla indir
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing: {symbol} {timeframe}")
                logger.info(f"Progress: {stats['completed']}/{stats['total']} "
                          f"({stats['completed']/stats['total']*100:.1f}%)")
                
                # Önceden tamamlanmış mı?
                progress = self.get_progress(symbol, timeframe)
                if progress.get('target_reached'):
                    logger.info(f"{symbol} {timeframe}: Already completed")
                    stats['already_done'] += 1
                    stats['completed'] += 1
                    continue
                
                # İndir
                success = self.download_symbol_full_history(
                    symbol, timeframe, target_date
                )
                
                if success:
                    stats['completed'] += 1
                else:
                    stats['failed'] += 1
                    stats['completed'] += 1
                
                # Durum raporu
                elapsed = time.time() - start_time
                avg_time = elapsed / stats['completed'] if stats['completed'] > 0 else 0
                remaining = (stats['total'] - stats['completed']) * avg_time
                
                logger.info(f"Stats: {stats['completed']} done, "
                          f"{stats['failed']} failed, "
                          f"{stats['already_done']} already done")
                logger.info(f"ETA: {remaining/60:.1f} minutes")
        
        # Özet
        total_time = time.time() - start_time
        logger.success(f"\nDownload completed in {total_time/60:.1f} minutes")
        logger.info(f"Total: {stats['total']}")
        logger.info(f"Completed: {stats['completed']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info(f"Already done: {stats['already_done']}")
        
        # API bağlantısını kapat
        self.api.disconnect()


def main():
    """Ana program"""
    
    # Settings dosyasını yükle
    with open('settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    
    symbols = settings['trading']['symbols']
    
    # Kullanıcı seçimi
    print("\nIncremental Data Download")
    print("=" * 50)
    print("1. Tüm timeframe'ler (15m, 1h, 4h, 1d)")
    print("2. Sadece 15m")
    print("3. Sadece 1h")
    print("4. Sadece 4h")
    print("5. Sadece 1d")
    print("6. Özel seçim")
    
    choice = input("\nSeçiminiz (1-6): ")
    
    timeframe_options = {
        '1': ['15m', '1h', '4h', '1d'],
        '2': ['15m'],
        '3': ['1h'],
        '4': ['4h'],
        '5': ['1d']
    }
    
    if choice in timeframe_options:
        timeframes = timeframe_options[choice]
    elif choice == '6':
        print("\nTimeframe'leri virgülle ayırarak girin (örn: 15m,1h): ")
        timeframes = [t.strip() for t in input().split(',')]
    else:
        print("Geçersiz seçim!")
        return
    
    # Yıl seçimi
    years = int(input("\nKaç yıl geriye gidilsin? (varsayılan: 4): ") or "4")
    
    # Sembol seçimi
    print(f"\nToplam {len(symbols)} sembol var.")
    use_all = input("Hepsini indir? (E/H): ").upper() == 'E'
    
    if not use_all:
        print("\nSembolleri virgülle ayırarak girin (örn: THYAO,GARAN): ")
        selected = [s.strip().upper() for s in input().split(',')]
        symbols = [s for s in symbols if s in selected]
    
    # Özet
    print(f"\nİndirilecek:")
    print(f"- Semboller: {len(symbols)} adet")
    print(f"- Timeframe'ler: {timeframes}")
    print(f"- Süre: {years} yıl geriye")
    print(f"- Toplam görev: {len(symbols) * len(timeframes)}")
    
    if input("\nDevam? (E/H): ").upper() != 'E':
        print("İptal edildi.")
        return
    
    # İndirmeyi başlat
    downloader = IncrementalDownloader()
    downloader.download_all_incremental(symbols, timeframes, years)


if __name__ == "__main__":
    main()