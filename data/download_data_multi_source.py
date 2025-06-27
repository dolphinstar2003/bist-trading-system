#!/usr/bin/env python3
"""
Multi-Source Data Downloader
Yahoo Finance, Alpha Vantage ve diğer kaynaklardan veri indirici
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

# Veri kaynakları
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance yüklü değil. Yüklemek için: pip install yfinance")

try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    logger.warning("alpha_vantage yüklü değil. Yüklemek için: pip install alpha-vantage")

try:
    import investpy
    INVESTPY_AVAILABLE = True
except ImportError:
    INVESTPY_AVAILABLE = False
    logger.warning("investpy yüklü değil. Yüklemek için: pip install investpy")

# Proje imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.assets import ASSETS
from utils.csv_data_manager import CSVDataManager


class MultiSourceDownloader:
    """Birden fazla kaynaktan veri indirici"""
    
    def __init__(self):
        self.csv_manager = CSVDataManager()
        
        # Progress tracking
        self.progress_file = Path("data/.multi_source_progress.json")
        self.progress = self.load_progress()
        
        # Yahoo Finance için sembol formatları
        self.yahoo_suffixes = [".IS", ".E", ""]  # BIST için .IS
        
        # Alpha Vantage API key
        self.alpha_vantage_key = "3D0PCT614PAOUXAL"
        
        # İstatistikler
        self.stats = {
            'yahoo': {'success': 0, 'failed': 0},
            'alpha_vantage': {'success': 0, 'failed': 0},
            'investpy': {'success': 0, 'failed': 0},
            'total_success': 0,
            'total_failed': 0
        }
    
    def load_progress(self) -> Dict:
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
            logger.error(f"Progress kaydedilemedi: {e}")
    
    def download_yahoo(self, symbol: str) -> Tuple[bool, pd.DataFrame]:
        """Yahoo Finance'den veri indir"""
        if not YFINANCE_AVAILABLE:
            return False, pd.DataFrame()
        
        logger.debug(f"Yahoo Finance deneniyor: {symbol}")
        
        # Farklı suffix'leri dene
        for suffix in self.yahoo_suffixes:
            yahoo_symbol = f"{symbol}{suffix}"
            
            try:
                # Yahoo Finance ticker
                ticker = yf.Ticker(yahoo_symbol)
                
                # Maksimum veri al (max 10 yıl)
                df = ticker.history(period="max")
                
                if df is not None and not df.empty:
                    # Column isimlerini standartlaştır
                    df.columns = [col.lower() for col in df.columns]
                    
                    # Gereksiz kolonları çıkar
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    
                    # Volume'u integer yap
                    df['volume'] = df['volume'].fillna(0).astype(int)
                    
                    logger.success(f"Yahoo: {symbol} ({len(df)} bar) indirildi")
                    self.stats['yahoo']['success'] += 1
                    return True, df
                    
            except Exception as e:
                logger.debug(f"Yahoo {yahoo_symbol} hatası: {e}")
                continue
        
        self.stats['yahoo']['failed'] += 1
        return False, pd.DataFrame()
    
    def download_alpha_vantage(self, symbol: str) -> Tuple[bool, pd.DataFrame]:
        """Alpha Vantage'den veri indir"""
        if not ALPHA_VANTAGE_AVAILABLE:
            return False, pd.DataFrame()
        
        logger.debug(f"Alpha Vantage deneniyor: {symbol}")
        
        try:
            ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
            
            # Türkiye için .IS suffix
            av_symbol = f"{symbol}.IS"
            
            # 15 saniye bekle (rate limit)
            time.sleep(15)
            
            df, meta = ts.get_daily_adjusted(symbol=av_symbol, outputsize='full')
            
            if df is not None and not df.empty:
                # Column mapping
                column_mapping = {
                    '1. open': 'open',
                    '2. high': 'high',
                    '3. low': 'low',
                    '4. close': 'close',
                    '5. adjusted close': 'adj_close',
                    '6. volume': 'volume'
                }
                
                df = df.rename(columns=column_mapping)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df['volume'] = df['volume'].astype(int)
                
                logger.success(f"Alpha Vantage: {symbol} ({len(df)} bar) indirildi")
                self.stats['alpha_vantage']['success'] += 1
                return True, df
                
        except Exception as e:
            logger.debug(f"Alpha Vantage {symbol} hatası: {e}")
        
        self.stats['alpha_vantage']['failed'] += 1
        return False, pd.DataFrame()
    
    def download_investpy(self, symbol: str) -> Tuple[bool, pd.DataFrame]:
        """Investing.com'dan veri indir"""
        if not INVESTPY_AVAILABLE:
            return False, pd.DataFrame()
        
        logger.debug(f"Investpy deneniyor: {symbol}")
        
        try:
            # Türkiye hisseleri için
            stocks = investpy.get_stocks_list(country='turkey')
            
            # Sembolü bul
            if symbol in stocks:
                # Son 10 yıl
                end_date = datetime.now().strftime('%d/%m/%Y')
                start_date = (datetime.now() - timedelta(days=3650)).strftime('%d/%m/%Y')
                
                df = investpy.get_stock_historical_data(
                    stock=symbol,
                    country='turkey',
                    from_date=start_date,
                    to_date=end_date
                )
                
                if df is not None and not df.empty:
                    # Column isimlerini standartlaştır
                    df.columns = [col.lower() for col in df.columns]
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    df['volume'] = df['volume'].fillna(0).astype(int)
                    
                    logger.success(f"Investpy: {symbol} ({len(df)} bar) indirildi")
                    self.stats['investpy']['success'] += 1
                    return True, df
                    
        except Exception as e:
            logger.debug(f"Investpy {symbol} hatası: {e}")
        
        self.stats['investpy']['failed'] += 1
        return False, pd.DataFrame()
    
    def download_symbol(self, symbol: str, sources: List[str] = None) -> bool:
        """Bir sembolü mevcut kaynaklardan indir"""
        # Varsayılan sıralama: Yahoo -> Alpha Vantage -> Investpy
        if sources is None:
            sources = []
            if YFINANCE_AVAILABLE:
                sources.append('yahoo')
            if ALPHA_VANTAGE_AVAILABLE:
                sources.append('alpha_vantage')
            if INVESTPY_AVAILABLE:
                sources.append('investpy')
        
        # Progress kontrolü
        if self.progress.get(symbol, {}).get('completed', False):
            logger.debug(f"{symbol} zaten indirilmiş, atlanıyor")
            return True
        
        logger.info(f"\nİndiriliyor: {symbol}")
        
        # Her kaynaktan dene
        for source in sources:
            if source == 'yahoo' and 'yahoo' in sources:
                success, df = self.download_yahoo(symbol)
            elif source == 'alpha_vantage' and 'alpha_vantage' in sources:
                success, df = self.download_alpha_vantage(symbol)
            elif source == 'investpy' and 'investpy' in sources:
                success, df = self.download_investpy(symbol)
            else:
                continue
            
            if success and not df.empty:
                # Timeframe'lere ayır ve kaydet
                saved_count = self.save_timeframes(symbol, df)
                
                if saved_count > 0:
                    # Progress güncelle
                    self.progress[symbol] = {
                        'completed': True,
                        'source': source,
                        'timestamp': datetime.now().isoformat(),
                        'bars': len(df),
                        'timeframes_saved': saved_count,
                        'date_range': f"{df.index[0]} to {df.index[-1]}"
                    }
                    self.save_progress()
                    
                    self.stats['total_success'] += 1
                    logger.success(f"✅ {symbol} başarıyla indirildi ({source})")
                    return True
        
        # Hiçbir kaynak başarılı olmadı
        self.stats['total_failed'] += 1
        logger.error(f"❌ {symbol} hiçbir kaynaktan indirilemedi")
        return False
    
    def save_timeframes(self, symbol: str, df: pd.DataFrame) -> int:
        """Farklı timeframe'lere böl ve kaydet"""
        timeframes = {
            '1d': 'D',    # Günlük
            '4h': '4H',   # 4 saatlik
            '1h': '1H',   # 1 saatlik  
            '15m': '15T'  # 15 dakikalık
        }
        
        saved_count = 0
        
        for tf_name, tf_code in timeframes.items():
            try:
                if tf_name == '1d':
                    # Günlük veri zaten var
                    df_resampled = df.copy()
                else:
                    # Diğer timeframe'ler için resample
                    # Sadece iş günleri varsa, boşlukları atla
                    df_resampled = df.resample(tf_code).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                
                # Kaydet
                if len(df_resampled) > 0:
                    if self.csv_manager.save_raw_data(symbol, df_resampled, tf_name):
                        saved_count += 1
                        logger.debug(f"  ✓ {tf_name}: {len(df_resampled)} bar")
                
            except Exception as e:
                logger.error(f"  ✗ {tf_name} kaydetme hatası: {e}")
        
        return saved_count
    
    def download_all(self, symbols: Optional[List[str]] = None):
        """Tüm sembolleri indir"""
        if symbols is None:
            symbols = ASSETS
        
        start_time = datetime.now()
        
        print(f"""
╔════════════════════════════════════════════════╗
║      MULTI-SOURCE DATA DOWNLOADER        ║
╚════════════════════════════════════════════════╝

Mevcut Kaynaklar:
""")
        
        available_sources = []
        if YFINANCE_AVAILABLE:
            print("✅ Yahoo Finance")
            available_sources.append('yahoo')
        if ALPHA_VANTAGE_AVAILABLE:
            print("✅ Alpha Vantage")
            available_sources.append('alpha_vantage')
        if INVESTPY_AVAILABLE:
            print("✅ Investing.com")
            available_sources.append('investpy')
        
        if not available_sources:
            print("❌ Hiçbir veri kaynağı yüklenmemiş!")
            print("Yüklemek için: pip install yfinance alpha-vantage investpy")
            return
        
        print(f"\nSembol sayısı: {len(symbols)}")
        print(f"Başlıyor...\n")
        
        # Tüm sembolleri indir
        for i, symbol in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] {symbol}", end=' ')
            
            success = self.download_symbol(symbol, available_sources)
            
            # Her 10 sembolde özet
            if i % 10 == 0:
                self.print_progress_summary()
        
        # Final özet
        self.print_final_summary(start_time)
    
    def print_progress_summary(self):
        """Ara ilerleme özeti"""
        print(f"\n--- Özet ---")
        for source, stats in self.stats.items():
            if isinstance(stats, dict) and 'success' in stats:
                print(f"{source}: Başarılı={stats['success']}, Başarısız={stats['failed']}")
        print()
    
    def print_final_summary(self, start_time: datetime):
        """Final özet"""
        elapsed = datetime.now() - start_time
        
        print(f"""
\n{'='*50}
TAMAMLANDI
{'='*50}

Toplam süre: {elapsed}
Başarılı: {self.stats['total_success']}
Başarısız: {self.stats['total_failed']}

Kaynak başına:
""")
        
        for source, stats in self.stats.items():
            if isinstance(stats, dict) and 'success' in stats:
                total = stats['success'] + stats['failed']
                if total > 0:
                    success_rate = (stats['success'] / total) * 100
                    print(f"  {source}: {stats['success']}/{total} ({success_rate:.1f}%)")
        
        print(f"\nVeriler: data/raw/")
        print("="*50)
    
    def check_available_sources(self):
        """Mevcut veri kaynaklarını kontrol et"""
        print("\nVeri Kaynakları Kontrolü:")
        print("-" * 30)
        
        sources = {
            'Yahoo Finance': YFINANCE_AVAILABLE,
            'Alpha Vantage': ALPHA_VANTAGE_AVAILABLE,
            'Investing.com': INVESTPY_AVAILABLE
        }
        
        available_count = 0
        for name, available in sources.items():
            if available:
                print(f"✅ {name} - Yüklendi")
                available_count += 1
            else:
                print(f"❌ {name} - Yüklenmemiş")
        
        if available_count == 0:
            print("\n⚠️  Hiçbir kaynak yüklenmemiş!")
            print("Yüklemek için:")
            print("pip install yfinance alpha-vantage investpy")
        
        return available_count > 0


def main():
    """Ana fonksiyon"""
    downloader = MultiSourceDownloader()
    
    # Kaynakları kontrol et
    if not downloader.check_available_sources():
        return
    
    print("""
\n1. Tüm sembolleri indir
2. Eksik sembolleri indir  
3. Test (ilk 5 sembol)
4. Mevcut verileri kontrol et
0. Çıkış
""")
    
    choice = input("Seçiminiz (0-4): ")
    
    if choice == '1':
        downloader.download_all()
        
    elif choice == '2':
        # Eksikleri bul
        existing = set()
        raw_dir = Path("data/raw")
        
        for file in raw_dir.glob("*_1d_raw.csv"):
            symbol = file.stem.split('_')[0]
            existing.add(symbol)
        
        missing = [s for s in ASSETS if s not in existing]
        
        if missing:
            print(f"\n{len(missing)} eksik sembol bulundu")
            downloader.download_all(missing)
        else:
            print("\nTüm semboller mevcut!")
    
    elif choice == '3':
        # Test
        test_symbols = ASSETS[:5]
        print(f"\nTest sembolleri: {', '.join(test_symbols)}")
        downloader.download_all(test_symbols)
    
    elif choice == '4':
        # Mevcut verileri kontrol et
        raw_dir = Path("data/raw")
        if not raw_dir.exists():
            print("\nHiç veri yok!")
            return
        
        files = list(raw_dir.glob("*.csv"))
        symbols = set()
        
        for file in files:
            symbol = file.stem.split('_')[0]
            symbols.add(symbol)
        
        print(f"\nToplam dosya: {len(files)}")
        print(f"Sembol sayısı: {len(symbols)}")
        print(f"\nMevcut semboller: {', '.join(sorted(symbols))}")


if __name__ == "__main__":
    main()