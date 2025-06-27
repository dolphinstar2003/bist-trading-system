#!/usr/bin/env python3
"""
Cache dosyalarını CSV'ye aktar
AlgoLab API 250 bar limiti olduğu için chunk'lar halinde indirilen verileri birleştir
"""

import os
import sys
import pickle
import pandas as pd
from pathlib import Path
from loguru import logger
import re

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def process_cache_files():
    """Cache dosyalarını CSV'ye dönüştür"""
    cache_dir = Path("data/.cache")
    raw_dir = Path("data/raw")
    
    if not cache_dir.exists():
        logger.error("Cache directory not found")
        return
    
    # Tüm pickle dosyalarını bul
    pkl_files = list(cache_dir.glob("*.pkl"))
    logger.info(f"Found {len(pkl_files)} cache files")
    
    # Symbol-timeframe bazında grupla
    data_dict = {}
    
    for pkl_file in pkl_files:
        try:
            # Dosya adından symbol ve timeframe çıkar
            # Format: SYMBOL_TIMEFRAME_DATE1_DATE2.pkl
            filename = pkl_file.stem
            parts = filename.split("_")
            
            if len(parts) >= 2:
                symbol = parts[0]
                timeframe = parts[1]
                key = f"{symbol}_{timeframe}"
                
                # Pickle dosyasını yükle
                with open(pkl_file, 'rb') as f:
                    df = pickle.load(f)
                
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    if key not in data_dict:
                        data_dict[key] = []
                    data_dict[key].append(df)
                    logger.debug(f"Loaded {len(df)} rows from {filename}")
        
        except Exception as e:
            logger.error(f"Error processing {pkl_file}: {e}")
    
    # Her symbol-timeframe için verileri birleştir
    for key, df_list in data_dict.items():
        try:
            symbol, timeframe = key.split("_", 1)
            
            # Tüm DataFrame'leri birleştir
            combined_df = pd.concat(df_list)
            
            # Duplicate'leri kaldır ve sırala
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
            
            # CSV dosya adı
            csv_filename = f"{symbol}_{timeframe}_raw.csv"
            csv_path = raw_dir / csv_filename
            
            # Mevcut CSV varsa yükle ve birleştir
            if csv_path.exists():
                try:
                    existing_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    
                    # Timezone uyumluluğu
                    if hasattr(existing_df.index, 'tz'):
                        if existing_df.index.tz is None and combined_df.index.tz is not None:
                            existing_df.index = existing_df.index.tz_localize('UTC')
                        elif existing_df.index.tz is not None and combined_df.index.tz is None:
                            combined_df.index = combined_df.index.tz_localize('UTC')
                    
                    # Birleştir
                    final_df = pd.concat([existing_df, combined_df])
                    final_df = final_df[~final_df.index.duplicated(keep='last')]
                    final_df = final_df.sort_index()
                    
                    logger.info(f"Merged with existing data: {csv_filename}")
                except Exception as e:
                    logger.warning(f"Could not merge with existing {csv_filename}: {e}")
                    final_df = combined_df
            else:
                final_df = combined_df
            
            # CSV'ye kaydet
            final_df.to_csv(csv_path)
            logger.success(f"Saved {csv_filename}: {len(final_df)} rows, "
                         f"Date range: {final_df.index[0]} to {final_df.index[-1]}")
            
        except Exception as e:
            logger.error(f"Error combining data for {key}: {e}")
    
    # Cache dosyalarını temizle (opsiyonel)
    if input("\nDelete cache files? (y/N): ").lower() == 'y':
        for pkl_file in pkl_files:
            pkl_file.unlink()
        logger.info("Cache files deleted")


def check_data_quality():
    """Veri kalitesini kontrol et"""
    raw_dir = Path("data/raw")
    csv_files = list(raw_dir.glob("*.csv"))
    
    print("\n" + "="*80)
    print("DATA QUALITY REPORT")
    print("="*80)
    
    summary = []
    
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            # Symbol ve timeframe
            parts = csv_file.stem.replace("_raw", "").split("_")
            symbol = parts[0]
            timeframe = parts[1] if len(parts) > 1 else "?"
            
            # İstatistikler
            stats = {
                'Symbol': symbol,
                'Timeframe': timeframe,
                'Rows': len(df),
                'Start': str(df.index[0])[:19] if len(df) > 0 else "N/A",
                'End': str(df.index[-1])[:19] if len(df) > 0 else "N/A",
                'Days': (df.index[-1] - df.index[0]).days if len(df) > 1 else 0,
                'Missing': df.isnull().sum().sum()
            }
            
            summary.append(stats)
            
        except Exception as e:
            logger.error(f"Error checking {csv_file}: {e}")
    
    # DataFrame olarak göster
    import pandas as pd
    summary_df = pd.DataFrame(summary)
    
    # Timeframe'e göre grupla
    for tf in ['15m', '1h', '4h', '1d']:
        tf_data = summary_df[summary_df['Timeframe'] == tf]
        if len(tf_data) > 0:
            print(f"\n{tf} Timeframe:")
            print(tf_data.to_string(index=False))
            print(f"  Total symbols: {len(tf_data)}")
            print(f"  Total rows: {tf_data['Rows'].sum():,}")
            print(f"  Average rows per symbol: {tf_data['Rows'].mean():.0f}")


def main():
    """Ana fonksiyon"""
    print("Cache to CSV Converter")
    print("="*50)
    
    # Cache dosyalarını işle
    process_cache_files()
    
    # Veri kalitesi raporu
    check_data_quality()


if __name__ == "__main__":
    main()