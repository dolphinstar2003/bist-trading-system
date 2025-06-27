#!/usr/bin/env python3
"""
Eksik Verileri İndir
"""

import sys
from pathlib import Path

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.download_data_multi_source import MultiSourceDownloader
from config.assets import ASSETS

print("Eksik Veri Analizi\n" + "="*50)

# Mevcut verileri kontrol et
raw_dir = Path("data/raw")
existing_symbols = set()

if raw_dir.exists():
    for file in raw_dir.glob("*_1d_raw.csv"):
        symbol = file.stem.split('_')[0]
        existing_symbols.add(symbol)

print(f"Toplam sembol: {len(ASSETS)}")
print(f"Mevcut sembol: {len(existing_symbols)}")
print(f"Eksik sembol: {len(ASSETS) - len(existing_symbols)}")

# Eksik sembolleri bul
missing_symbols = [s for s in ASSETS if s not in existing_symbols]

if missing_symbols:
    print(f"\nEksik semboller ({len(missing_symbols)}):")
    for i, symbol in enumerate(missing_symbols[:20], 1):  # İlk 20'yi göster
        print(f"{i:2d}. {symbol}")
    
    if len(missing_symbols) > 20:
        print(f"... ve {len(missing_symbols) - 20} sembol daha")
    
    # İndirme işlemi
    print("\n" + "="*50)
    print("EKSİK VERİLER İNDİRİLİYOR")
    print("="*50)
    
    downloader = MultiSourceDownloader()
    downloader.download_all(missing_symbols)
    
else:
    print("\n✅ Tüm semboller mevcut!")

# Her durumda veri detaylarını göster
print("\n" + "="*80)
print("VERİ DETAYLARI")
print("="*80)

import pandas as pd
from utils.csv_data_manager import CSVDataManager

csv_manager = CSVDataManager()
timeframes = ['1d', '4h', '1h', '15m']

# Her sembol için bilgi topla
all_data = []

for symbol in sorted(existing_symbols):
    symbol_info = {'symbol': symbol}
    
    for tf in timeframes:
        try:
            df = csv_manager.load_raw_data(symbol, tf)
            if df is not None and len(df) > 0:
                symbol_info[f'{tf}_start'] = df.index[0].strftime('%Y-%m-%d')
                symbol_info[f'{tf}_end'] = df.index[-1].strftime('%Y-%m-%d')
                symbol_info[f'{tf}_bars'] = len(df)
            else:
                symbol_info[f'{tf}_start'] = '-'
                symbol_info[f'{tf}_end'] = '-'
                symbol_info[f'{tf}_bars'] = 0
        except Exception as e:
            symbol_info[f'{tf}_start'] = 'ERROR'
            symbol_info[f'{tf}_end'] = 'ERROR'
            symbol_info[f'{tf}_bars'] = 0
    
    all_data.append(symbol_info)

# Tablo olarak yazdır
print(f"\n{'Sembol':<8} {'1D Başlangıç':<12} {'1D Bitiş':<12} {'Bars':<6} | "
      f"{'4H Başlangıç':<12} {'4H Bitiş':<12} {'Bars':<6}")
print("-" * 90)

for data in all_data[:20]:  # İlk 20 sembol
    print(f"{data['symbol']:<8} "
          f"{data['1d_start']:<12} {data['1d_end']:<12} {data['1d_bars']:<6} | "
          f"{data['4h_start']:<12} {data['4h_end']:<12} {data['4h_bars']:<6}")

if len(all_data) > 20:
    print(f"\n... ve {len(all_data) - 20} sembol daha")

# Özet istatistikler
print("\n" + "="*50)
print("ÖZET İSTATİSTİKLER")
print("="*50)

total_files = 0
total_bars = 0
oldest_date = None
newest_date = None

for tf in timeframes:
    tf_files = 0
    tf_bars = 0
    
    for data in all_data:
        if data[f'{tf}_bars'] > 0:
            tf_files += 1
            tf_bars += data[f'{tf}_bars']
            
            # En eski ve en yeni tarihleri bul
            if data[f'{tf}_start'] != '-' and data[f'{tf}_start'] != 'ERROR':
                if oldest_date is None or data[f'{tf}_start'] < oldest_date:
                    oldest_date = data[f'{tf}_start']
                if newest_date is None or data[f'{tf}_end'] > newest_date:
                    newest_date = data[f'{tf}_end']
    
    total_files += tf_files
    total_bars += tf_bars
    
    print(f"{tf:>4}: {tf_files:>3} dosya, {tf_bars:>8,} bar")

print(f"\nToplam: {total_files} dosya, {total_bars:,} bar")
print(f"Tarih aralığı: {oldest_date} - {newest_date}")

# En fazla veriye sahip semboller
print("\n" + "="*50)
print("EN FAZLA VERİYE SAHİP 10 SEMBOL (1D)")
print("="*50)

sorted_by_bars = sorted(all_data, key=lambda x: x['1d_bars'], reverse=True)
for i, data in enumerate(sorted_by_bars[:10], 1):
    years = data['1d_bars'] / 252 if data['1d_bars'] > 0 else 0  # Yaklaşık yıl
    print(f"{i:2}. {data['symbol']:<8} {data['1d_bars']:>6,} bar (~{years:>4.1f} yıl) "
          f"[{data['1d_start']} - {data['1d_end']}]")