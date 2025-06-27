#!/usr/bin/env python3
"""
Basit veri indirme scripti
- Tüm hisseler için veri indir
- 4 timeframe: 15m, 1h, 4h, 1d
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from loguru import logger

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algolab.algolab_wrapper import AlgoLabWrapper
from utils.csv_data_manager import CSVDataManager


def download_all_data():
    """Tüm sembolleri sırayla indir"""
    
    # Settings yükle
    with open('settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    
    symbols = settings['trading']['symbols']
    timeframes = ['15m', '1h', '4h', '1d']
    
    # API ve CSV manager
    api = AlgoLabWrapper()
    csv_manager = CSVDataManager()
    
    # API bağlantısı
    logger.info("Connecting to API...")
    result = api.connect()
    
    if not result:
        # SMS kodu gerekiyor
        logger.info("SMS code sent to your phone.")
        sms_code = input("\nEnter SMS code: ")
        
        # SMS kodu ile tekrar dene
        result = api.connect(sms_code=sms_code)
        
        if not result:
            logger.error("API connection failed!")
            return
    
    logger.success("API connected successfully!")
    
    # İstatistikler
    total_tasks = len(symbols) * len(timeframes)
    completed = 0
    failed = 0
    start_time = time.time()
    
    # Her sembol için
    for i, symbol in enumerate(symbols):
        logger.info(f"\n{'='*60}")
        logger.info(f"Symbol {i+1}/{len(symbols)}: {symbol}")
        
        # Her timeframe için
        for timeframe in timeframes:
            completed += 1
            
            try:
                logger.info(f"  Downloading {symbol} {timeframe}...")
                
                # Rate limiting (5 saniye bekle)
                time.sleep(5.1)
                
                # Veri çek
                df = api.get_market_data(symbol, timeframe)
                
                if df is not None and len(df) > 0:
                    # CSV'ye kaydet
                    csv_manager.save_raw_data(symbol, df, timeframe)
                    logger.success(f"  ✓ {symbol} {timeframe}: {len(df)} bars saved")
                else:
                    logger.warning(f"  ✗ {symbol} {timeframe}: No data received")
                    failed += 1
                    
            except Exception as e:
                logger.error(f"  ✗ {symbol} {timeframe}: Error - {str(e)}")
                failed += 1
            
            # Progress
            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            remaining = (total_tasks - completed) * avg_time
            
            progress_pct = (completed / total_tasks) * 100
            logger.info(f"  Progress: {completed}/{total_tasks} ({progress_pct:.1f}%) "
                       f"- ETA: {remaining/60:.1f} min")
    
    # Özet
    total_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.success(f"Download completed in {total_time/60:.1f} minutes")
    logger.info(f"Total: {total_tasks} tasks")
    logger.info(f"Success: {completed - failed}")
    logger.info(f"Failed: {failed}")
    
    # Bağlantıyı kapat
    api.disconnect()


def main():
    """Ana program"""
    
    print("\nBIST Data Downloader")
    print("="*50)
    print("This will download data for all symbols and timeframes.")
    print("Estimated time: 2-3 hours")
    print("\nTimeframes: 15m, 1h, 4h, 1d")
    print("Symbols: 59 stocks from BIST")
    
    if input("\nContinue? (Y/N): ").upper() != 'Y':
        print("Cancelled.")
        return
    
    download_all_data()


if __name__ == "__main__":
    main()