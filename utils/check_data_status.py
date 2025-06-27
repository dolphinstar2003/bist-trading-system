#!/usr/bin/env python3
"""
Veri durumunu kontrol et
- Her sembol için mevcut veri aralığını göster
- Eksik verileri listele
- İstatistikler
"""

import sys
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from tabulate import tabulate
from loguru import logger

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.csv_data_manager import CSVDataManager


def check_data_status():
    """Tüm sembollerin veri durumunu kontrol et"""
    
    # Settings yükle
    with open('settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)
    
    symbols = settings['trading']['symbols']
    timeframes = ['15m', '1h', '4h', '1d']
    
    csv_manager = CSVDataManager()
    
    # Durum tablosu
    status_data = []
    
    for symbol in symbols:
        row = {'Symbol': symbol}
        
        for tf in timeframes:
            try:
                df = csv_manager.load_raw_data(symbol, tf)
                
                if df is not None and len(df) > 0:
                    start_date = df.index[0]
                    end_date = df.index[-1]
                    bars = len(df)
                    
                    # Tarih aralığını formatla
                    date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    row[tf] = f"{bars:,} bars\n{date_range}"
                else:
                    row[tf] = "No data"
                    
            except Exception as e:
                row[tf] = f"Error: {str(e)[:20]}"
        
        status_data.append(row)
    
    # Tabloyu yazdır
    print("\n" + "="*100)
    print("DATA STATUS REPORT")
    print("="*100)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total symbols: {len(symbols)}")
    print("\n")
    
    # Tabulate ile güzel görünüm
    headers = ['Symbol'] + timeframes
    table_data = []
    
    for row in status_data:
        table_row = [row.get(col, '') for col in headers]
        table_data.append(table_row)
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # İstatistikler
    print("\n" + "="*50)
    print("STATISTICS")
    print("="*50)
    
    for tf in timeframes:
        has_data = sum(1 for row in status_data if row[tf] != "No data" and not row[tf].startswith("Error"))
        no_data = len(symbols) - has_data
        
        print(f"\n{tf}:")
        print(f"  - Has data: {has_data} symbols")
        print(f"  - No data: {no_data} symbols")
        
        if has_data > 0:
            # En eski ve en yeni tarihleri bul
            oldest_dates = []
            newest_dates = []
            
            for symbol in symbols:
                try:
                    df = csv_manager.load_raw_data(symbol, tf)
                    if df is not None and len(df) > 0:
                        oldest_dates.append(df.index[0])
                        newest_dates.append(df.index[-1])
                except:
                    pass
            
            if oldest_dates:
                print(f"  - Oldest data: {min(oldest_dates).strftime('%Y-%m-%d')}")
                print(f"  - Newest data: {max(newest_dates).strftime('%Y-%m-%d')}")
    
    # Progress dosyalarını kontrol et
    progress_dir = Path("data/.progress")
    if progress_dir.exists():
        progress_files = list(progress_dir.glob("*.json"))
        
        print("\n" + "="*50)
        print("DOWNLOAD PROGRESS")
        print("="*50)
        
        if progress_files:
            print(f"\nFound {len(progress_files)} progress files:")
            
            completed = 0
            in_progress = 0
            
            for pf in progress_files:
                try:
                    with open(pf, 'r') as f:
                        progress = json.load(f)
                    
                    if progress.get('target_reached'):
                        completed += 1
                    else:
                        in_progress += 1
                        
                except:
                    pass
            
            print(f"  - Completed: {completed}")
            print(f"  - In progress: {in_progress}")
        else:
            print("\nNo progress files found.")
    
    # Disk kullanımı
    data_dir = Path("data")
    total_size = 0
    file_count = 0
    
    for csv_file in data_dir.rglob("*.csv"):
        total_size += csv_file.stat().st_size
        file_count += 1
    
    print("\n" + "="*50)
    print("DISK USAGE")
    print("="*50)
    print(f"Total CSV files: {file_count}")
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    
    # Öneri
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    
    missing_data = []
    for symbol in symbols:
        for tf in timeframes:
            try:
                df = csv_manager.load_raw_data(symbol, tf)
                if df is None or len(df) == 0:
                    missing_data.append(f"{symbol} - {tf}")
            except:
                missing_data.append(f"{symbol} - {tf}")
    
    if missing_data:
        print(f"\n⚠️  Missing data for {len(missing_data)} symbol-timeframe combinations:")
        print("\nTo download missing data, run:")
        print("  python data_download_incremental.py")
    else:
        print("\n✅ All data is available!")
    
    # En son güncelleme zamanları
    print("\n" + "="*50)
    print("LAST UPDATE TIMES")
    print("="*50)
    
    update_times = []
    for symbol in symbols[:10]:  # İlk 10 sembol
        for tf in timeframes:
            try:
                df = csv_manager.load_raw_data(symbol, tf)
                if df is not None and len(df) > 0:
                    last_time = df.index[-1]
                    update_times.append((symbol, tf, last_time))
            except:
                pass
    
    if update_times:
        # En son 5 güncellemeyi göster
        update_times.sort(key=lambda x: x[2], reverse=True)
        print("\nMost recent updates:")
        for symbol, tf, last_time in update_times[:5]:
            print(f"  {symbol} ({tf}): {last_time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    check_data_status()