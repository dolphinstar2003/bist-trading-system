#!/usr/bin/env python3
"""
Yahoo Finance Proper Downloader
Son 5 yıl için düzgün timeframe verileri indirir
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pandas as pd
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf

# Proje imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.assets import ASSETS
from utils.csv_data_manager import CSVDataManager


class YahooProperDownloader:
    """Yahoo Finance'den düzgün timeframe verileri indiren sınıf"""
    
    def __init__(self, years: int = 5):
        self.csv_manager = CSVDataManager()
        self.years = years
        self.start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Progress tracking
        self.progress_file = Path("data/.yahoo_proper_progress.json")
        self.progress = self.load_progress()
        
        # Yahoo Finance interval mapping
        self.interval_mapping = {
            '1d': '1d',   # Günlük
            '4h': '1h',   # Yahoo'da 4h yok, 1h'den hesaplayacağız
            '1h': '1h',   # Saatlik
            '15m': '15m'  # 15 dakikalık (max 60 gün)
        }
        
        # Period limits for different intervals
        self.period_limits = {
            '1d': '5y',    # 5 yıl
            '1h': '730d',  # 2 yıl (730 gün)
            '15m': '60d'   # 60 gün
        }
        
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'start_time': datetime.now()
        }
    
    def load_progress(self) -> Dict:
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_progress(self):
        try:
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Progress kaydedilemedi: {e}")
    
    def download_timeframe(self, symbol: str, timeframe: str) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Belirli bir timeframe için veri indir"""
        yahoo_symbol = f"{symbol}.IS"
        
        try:
            if timeframe == '1d':
                # Günlük veri - 5 yıl
                df = yf.download(
                    yahoo_symbol,
                    start=self.start_date,
                    end=self.end_date,
                    interval='1d',
                    progress=False,
                    auto_adjust=True
                )
                
            elif timeframe == '4h':
                # 4 saatlik - Yahoo'da yok, 1h'den hesapla
                df_1h = yf.download(
                    yahoo_symbol,
                    period='730d',  # Max 2 yıl
                    interval='1h',
                    progress=False,
                    auto_adjust=True
                )
                
                if not df_1h.empty:
                    # Multi-level columns durumunu kontrol et
                    if isinstance(df_1h.columns, pd.MultiIndex):
                        df_1h.columns = df_1h.columns.get_level_values(0)
                    
                    # Column isimlerini küçült
                    df_1h.columns = [col.lower() if isinstance(col, str) else str(col).lower() for col in df_1h.columns]
                    
                    # 1h -> 4h resample
                    df = df_1h.resample('4H').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                else:
                    return False, None
                    
            elif timeframe == '1h':
                # Saatlik veri - max 2 yıl
                df = yf.download(
                    yahoo_symbol,
                    period='730d',
                    interval='1h',
                    progress=False,
                    auto_adjust=True
                )
                
            elif timeframe == '15m':
                # 15 dakikalık - max 60 gün
                df = yf.download(
                    yahoo_symbol,
                    period='60d',
                    interval='15m',
                    progress=False,
                    auto_adjust=True
                )
            else:
                logger.error(f"Geçersiz timeframe: {timeframe}")
                return False, None
            
            if df is not None and not df.empty:
                # Multi-level columns durumunu kontrol et
                if isinstance(df.columns, pd.MultiIndex):
                    # Multi-level ise sadece ilk seviyeyi al
                    df.columns = df.columns.get_level_values(0)
                
                # Column isimlerini küçült
                df.columns = [col.lower() if isinstance(col, str) else str(col).lower() for col in df.columns]
                
                # Sadece OHLCV
                if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    df['volume'] = df['volume'].fillna(0).astype(int)
                    
                    # Boş satırları temizle
                    df = df.dropna()
                    
                    logger.debug(f"{symbol} {timeframe}: {len(df)} bar indirildi")
                    return True, df
                else:
                    logger.warning(f"{symbol} {timeframe}: Eksik kolonlar")
                    return False, None
            else:
                return False, None
                
        except Exception as e:
            logger.error(f"{symbol} {timeframe} hatası: {e}")
            return False, None
    
    def download_symbol(self, symbol: str) -> Dict[str, bool]:
        """Bir sembol için tüm timeframe'leri indir"""
        results = {}
        
        # Progress kontrolü
        if self.progress.get(symbol, {}).get('completed', False):
            logger.debug(f"{symbol} zaten indirilmiş")
            return {'skipped': True}
        
        logger.info(f"\nİndiriliyor: {symbol}")
        
        for timeframe in ['1d', '4h', '1h', '15m']:
            success, df = self.download_timeframe(symbol, timeframe)
            
            if success and df is not None:
                # CSV'ye kaydet
                if self.csv_manager.save_raw_data(symbol, df, timeframe):
                    results[timeframe] = True
                    logger.success(f"  ✓ {timeframe}: {len(df)} bar")
                else:
                    results[timeframe] = False
                    logger.error(f"  ✗ {timeframe}: Kaydetme hatası")
            else:
                results[timeframe] = False
                logger.warning(f"  ✗ {timeframe}: Veri yok")
            
            # Timeframe'ler arası kısa bekleme
            time.sleep(0.5)
        
        # Progress güncelle
        success_count = sum(1 for v in results.values() if v)
        if success_count > 0:
            self.progress[symbol] = {
                'completed': True,
                'timestamp': datetime.now().isoformat(),
                'timeframes': results,
                'success_count': success_count
            }
            self.save_progress()
            self.stats['success'] += 1
        else:
            self.stats['failed'] += 1
        
        return results
    
    def download_all(self, symbols: Optional[List[str]] = None):
        """Tüm sembolleri indir"""
        if symbols is None:
            symbols = ASSETS
        
        self.stats['total'] = len(symbols)
        
        print(f"""
╔══════════════════════════════════════════════════╗
║    YAHOO FINANCE PROPER DOWNLOADER       ║
╚══════════════════════════════════════════════════╝

Timeframe Limitleri:
  1d:  Son 5 yıl
  4h:  Son 2 yıl (1h'den hesaplanır)
  1h:  Son 2 yıl
  15m: Son 60 gün

Sembol sayısı: {len(symbols)}
Tahmini süre: {len(symbols) * 2} - {len(symbols) * 3} dakika

Başlıyor...
""")
        
        for i, symbol in enumerate(symbols, 1):
            # İlerleme
            progress_pct = (i / len(symbols)) * 100
            print(f"\n[{i}/{len(symbols)}] ({progress_pct:.1f}%)", end=' ')
            
            # İndir
            results = self.download_symbol(symbol)
            
            # Her 10 sembolde özet
            if i % 10 == 0:
                self.print_progress_summary(i)
            
            # Rate limiting
            time.sleep(1)
        
        # Final özet
        self.print_final_summary()
    
    def print_progress_summary(self, current: int):
        """Ara ilerleme özeti"""
        elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
        rate = current / elapsed if elapsed > 0 else 0
        eta = (self.stats['total'] - current) / rate if rate > 0 else 0
        
        print(f"\n\n--- İlerleme ---")
        print(f"Başarılı: {self.stats['success']}")
        print(f"Başarısız: {self.stats['failed']}")
        print(f"Kalan süre: ~{eta/60:.1f} dakika\n")
    
    def print_final_summary(self):
        """Final özet"""
        elapsed = datetime.now() - self.stats['start_time']
        
        print(f"""
\n{'='*50}
İNDİRME TAMAMLANDI
{'='*50}

Toplam süre: {elapsed}
Toplam sembol: {self.stats['total']}
Başarılı: {self.stats['success']}
Başarısız: {self.stats['failed']}

Timeframe Sınırları:
  1d:  5 yıl ({self.start_date} - {self.end_date})
  4h:  2 yıl
  1h:  2 yıl  
  15m: 60 gün

Veriler: data/raw/
{'='*50}
""")


def main():
    downloader = YahooProperDownloader(years=5)
    
    print("""
1. Tüm sembolleri indir
2. Test (ilk 3 sembol)
3. Progress'i sıfırla
0. Çıkış
""")
    
    choice = input("Seçiminiz (0-3): ")
    
    if choice == '1':
        downloader.download_all()
        
    elif choice == '2':
        # Test
        test_symbols = ASSETS[:3]
        print(f"\nTest sembolleri: {', '.join(test_symbols)}")
        downloader.download_all(test_symbols)
        
    elif choice == '3':
        # Progress sıfırla
        if downloader.progress_file.exists():
            downloader.progress_file.unlink()
            print("Progress sıfırlandı")
        else:
            print("Progress dosyası zaten yok")


if __name__ == "__main__":
    main()