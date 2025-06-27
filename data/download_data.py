#!/usr/bin/env python3
"""
BIST Veri İndirme - Tek Dosya
AlgoLab API ile son 250 bar veri indirir
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import pandas as pd
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algolab.algolab_wrapper import AlgoLabWrapper
from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS


class DataDownloader:
    """Veri indirme sınıfı"""
    
    def __init__(self):
        self.api = AlgoLabWrapper()
        self.csv_manager = CSVDataManager()
        
        # Progress tracking
        self.progress_file = Path("data/.download_progress.json")
        self.progress = self.load_progress()
        
        # Rate limiting
        self.rate_limit_delay = 5  # Saniye
        self.last_request_time = 0
        
    def load_progress(self) -> dict:
        """İlerleme durumunu yükle"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_progress(self):
        """İlerleme durumunu kaydet"""
        try:
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save progress: {e}")
    
    def wait_rate_limit(self):
        """Rate limit için bekle"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        self.last_request_time = time.time()
    
    def download_symbol(self, symbol: str, timeframe: str) -> bool:
        """Tek bir sembol için veri indir"""
        try:
            # Progress kontrolü
            key = f"{symbol}_{timeframe}"
            if self.progress.get(key, {}).get('completed', False):
                logger.debug(f"Skipping {key} - already downloaded")
                return True
            
            logger.info(f"Downloading {symbol} {timeframe}...")
            
            # Rate limit
            self.wait_rate_limit()
            
            # API'den veri al
            df = self.api.get_market_data(symbol, timeframe, bar_count=250)
            
            if df is not None and len(df) > 0:
                # CSV'ye kaydet
                if self.csv_manager.save_raw_data(symbol, df, timeframe):
                    logger.success(f"✅ {symbol} {timeframe}: {len(df)} bars saved")
                    
                    # Progress güncelle
                    self.progress[key] = {
                        'completed': True,
                        'bars': len(df),
                        'timestamp': datetime.now().isoformat(),
                        'date_range': f"{df.index[0]} to {df.index[-1]}"
                    }
                    self.save_progress()
                    return True
                else:
                    logger.error(f"❌ {symbol} {timeframe}: Failed to save")
            else:
                logger.warning(f"⚠️  {symbol} {timeframe}: No data received")
                
        except Exception as e:
            logger.error(f"❌ {symbol} {timeframe}: Error - {e}")
        
        return False
    
    def download_all(self, symbols: Optional[List[str]] = None, 
                    timeframes: Optional[List[str]] = None):
        """Tüm sembolleri indir"""
        if symbols is None:
            symbols = ASSETS
        if timeframes is None:
            timeframes = ['15m', '1h', '4h', '1d']
        
        total = len(symbols) * len(timeframes)
        completed = 0
        successful = 0
        
        logger.info(f"\nStarting download: {len(symbols)} symbols × {len(timeframes)} timeframes = {total} tasks")
        logger.info(f"Rate limit: {self.rate_limit_delay}s between requests")
        logger.info(f"Expected time: ~{total * self.rate_limit_delay / 60:.1f} minutes\n")
        
        start_time = time.time()
        
        for symbol in symbols:
            for timeframe in timeframes:
                completed += 1
                
                # Progress göster
                progress_pct = (completed / total) * 100
                logger.info(f"[{completed}/{total}] ({progress_pct:.1f}%) {symbol} {timeframe}")
                
                # İndir
                if self.download_symbol(symbol, timeframe):
                    successful += 1
                
                # Her 10 işlemde bir özet göster
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (total - completed) / rate if rate > 0 else 0
                    logger.info(f"Progress: {successful}/{completed} successful, "
                              f"ETA: {eta/60:.1f} min")
        
        # Final özet
        elapsed = time.time() - start_time
        logger.info("\n" + "="*60)
        logger.info("DOWNLOAD COMPLETED")
        logger.info("="*60)
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Tasks: {completed}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {completed - successful}")
        logger.info(f"Success rate: {(successful/completed*100):.1f}%")
        logger.info("="*60)
        
        # Veri özeti
        self.print_data_summary()
    
    def print_data_summary(self):
        """İndirilen veri özetini göster"""
        raw_dir = Path("data/raw")
        if not raw_dir.exists():
            return
        
        csv_files = list(raw_dir.glob("*.csv"))
        logger.info(f"\nData files: {len(csv_files)}")
        
        # Timeframe bazında özet
        summary = {'15m': 0, '1h': 0, '4h': 0, '1d': 0}
        
        for csv_file in csv_files:
            for tf in summary.keys():
                if f"_{tf}_" in csv_file.name:
                    summary[tf] += 1
        
        logger.info("\nBy timeframe:")
        for tf, count in summary.items():
            logger.info(f"  {tf}: {count} files")
    
    def clean_start(self):
        """Temiz başlangıç - progress'i sıfırla"""
        if self.progress_file.exists():
            self.progress_file.unlink()
        self.progress = {}
        logger.info("Progress cleared")


def main():
    """Ana fonksiyon"""
    downloader = DataDownloader()
    
    # Komut satırı argümanları
    if len(sys.argv) > 1:
        if sys.argv[1] == '--clean':
            # Progress'i temizle ve baştan başla
            downloader.clean_start()
            downloader.download_all()
            
        elif sys.argv[1] == '--symbol' and len(sys.argv) >= 4:
            # Tek sembol
            symbol = sys.argv[2]
            timeframe = sys.argv[3]
            downloader.download_symbol(symbol, timeframe)
            
        elif sys.argv[1] == '--test':
            # Test - ilk 3 sembol
            test_symbols = ASSETS[:3]
            test_timeframes = ['1h', '1d']
            downloader.download_all(test_symbols, test_timeframes)
            
        elif sys.argv[1] == '--resume':
            # Kaldığı yerden devam
            downloader.download_all()
            
        else:
            print("Usage:")
            print("  python download_data.py           # Download all")
            print("  python download_data.py --resume  # Resume from last")
            print("  python download_data.py --clean   # Clean start")
            print("  python download_data.py --test    # Test with 3 symbols")
            print("  python download_data.py --symbol THYAO 1d  # Single symbol")
    else:
        # Kullanıcıya sor
        print("\nBIST Data Downloader")
        print("="*40)
        print(f"Symbols: {len(ASSETS)}")
        print("Timeframes: 15m, 1h, 4h, 1d")
        print(f"Total tasks: {len(ASSETS) * 4}")
        print(f"Estimated time: ~{len(ASSETS) * 4 * 5 / 60:.0f} minutes")
        print("\nOptions:")
        print("1. Download all (fresh start)")
        print("2. Resume from last position")
        print("3. Test with 3 symbols")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ")
        
        if choice == '1':
            downloader.clean_start()
            downloader.download_all()
        elif choice == '2':
            downloader.download_all()
        elif choice == '3':
            test_symbols = ASSETS[:3]
            test_timeframes = ['1h', '1d']
            downloader.download_all(test_symbols, test_timeframes)
        else:
            print("Exiting...")


if __name__ == "__main__":
    main()