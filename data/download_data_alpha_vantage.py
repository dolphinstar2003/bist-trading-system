#!/usr/bin/env python3
"""
Alpha Vantage Data Downloader
Türkiye hisse senetleri için geçmiş veri indirici
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict
import pandas as pd
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Alpha Vantage
try:
    from alpha_vantage.timeseries import TimeSeries
except ImportError:
    logger.error("alpha_vantage paketi yüklü değil!")
    logger.info("Yüklemek için: pip install alpha-vantage")
    sys.exit(1)

# Proje imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.assets import ASSETS
from utils.csv_data_manager import CSVDataManager


class AlphaVantageDownloader:
    """Alpha Vantage API ile veri indirici"""
    
    def __init__(self):
        # API Key (kod içinde)
        self.api_key = "3D0PCT614PAOUXAL"
        
        # Alpha Vantage client
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        
        # CSV manager
        self.csv_manager = CSVDataManager()
        
        # Progress tracking
        self.progress_file = Path("data/.alpha_vantage_progress.json")
        self.progress = self.load_progress()
        
        # Rate limiting
        self.calls_per_minute = 5
        self.wait_time = 15  # saniye (güvenli taraf)
        self.last_call_time = 0
        
        # İstatistikler
        self.stats = {
            'total_symbols': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'start_time': datetime.now()
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
    
    def wait_for_rate_limit(self):
        """Rate limit için bekle"""
        elapsed = time.time() - self.last_call_time
        if elapsed < self.wait_time:
            wait = self.wait_time - elapsed
            logger.debug(f"Rate limit: {wait:.1f} saniye bekleniyor...")
            time.sleep(wait)
        self.last_call_time = time.time()
    
    def convert_to_ist_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alpha Vantage verisini IST hisse formatına dönüştür"""
        try:
            # Sütun isimlerini düzenle
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. adjusted close': 'adj_close',
                '6. volume': 'volume',
                '7. dividend amount': 'dividend',
                '8. split coefficient': 'split'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Sadece OHLCV sütunlarını al
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            if 'adj_close' in df.columns:
                ohlcv_columns.append('adj_close')
            
            df = df[ohlcv_columns]
            
            # Index'i datetime yap
            df.index = pd.to_datetime(df.index)
            
            # Sırala (eski->yeni)
            df = df.sort_index()
            
            # Volume'ü integer yap
            df['volume'] = df['volume'].astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Veri dönüştürme hatası: {e}")
            return pd.DataFrame()
    
    def download_symbol(self, symbol: str) -> bool:
        """Tek bir sembol için veri indir"""
        # IST formatına çevir
        alpha_symbol = f"{symbol}.IS"
        
        # Progress kontrolü
        if self.progress.get(symbol, {}).get('completed', False):
            logger.debug(f"{symbol} zaten indirilmiş, atlanıyor")
            self.stats['skipped'] += 1
            return True
        
        logger.info(f"İndiriliyor: {symbol} ({alpha_symbol})...")
        
        # Rate limit
        self.wait_for_rate_limit()
        
        try:
            # Alpha Vantage'dan veri çek
            df, meta_data = self.ts.get_daily_adjusted(
                symbol=alpha_symbol,
                outputsize='full'  # Tüm geçmiş
            )
            
            if df is None or df.empty:
                logger.warning(f"{symbol} için veri bulunamadı")
                self.stats['failed'] += 1
                return False
            
            # Formatı dönüştür
            df_converted = self.convert_to_ist_format(df)
            
            if df_converted.empty:
                logger.error(f"{symbol} veri dönüştürme başarısız")
                self.stats['failed'] += 1
                return False
            
            # Her timeframe için resample et ve kaydet
            timeframes = {
                '1d': 'D',    # Günlük (zaten günlük)
                '4h': '4H',   # 4 saatlik
                '1h': '1H',   # 1 saatlik
                '15m': '15T'  # 15 dakikalık
            }
            
            saved_count = 0
            
            for tf_name, tf_code in timeframes.items():
                try:
                    if tf_name == '1d':
                        # Günlük veri zaten var
                        df_resampled = df_converted.copy()
                    else:
                        # Diğer timeframe'ler için resample
                        df_resampled = df_converted.resample(tf_code).agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum'
                        }).dropna()
                    
                    # Kaydet
                    if self.csv_manager.save_raw_data(symbol, df_resampled, tf_name):
                        saved_count += 1
                        logger.debug(f"  ✓ {tf_name}: {len(df_resampled)} bar kaydedildi")
                    
                except Exception as e:
                    logger.error(f"  ✗ {tf_name} resample hatası: {e}")
            
            # Progress güncelle
            self.progress[symbol] = {
                'completed': True,
                'timestamp': datetime.now().isoformat(),
                'timeframes_saved': saved_count,
                'total_bars': len(df_converted),
                'date_range': f"{df_converted.index[0]} to {df_converted.index[-1]}"
            }
            self.save_progress()
            
            self.stats['successful'] += 1
            logger.success(f"✅ {symbol}: {saved_count} timeframe kaydedildi")
            return True
            
        except Exception as e:
            logger.error(f"❌ {symbol} indirme hatası: {e}")
            self.stats['failed'] += 1
            
            # Eğer limit hatası ise ekstra bekle
            if "API call frequency" in str(e) or "rate limit" in str(e).lower():
                logger.warning("API limit aşıldı, 60 saniye bekleniyor...")
                time.sleep(60)
            
            return False
    
    def download_all(self, symbols: Optional[List[str]] = None):
        """Tüm sembolleri indir"""
        if symbols is None:
            symbols = ASSETS
        
        self.stats['total_symbols'] = len(symbols)
        
        print(f"""
╔════════════════════════════════════════════╗
║       ALPHA VANTAGE DATA DOWNLOADER        ║
╚════════════════════════════════════════════╝

API Key: {self.api_key[:8]}...
Sembol sayısı: {len(symbols)}
Rate limit: {self.calls_per_minute} çağrı/dakika
Bekleme süresi: {self.wait_time} saniye/çağrı
Tahmini süre: ~{len(symbols) * self.wait_time / 60:.1f} dakika

Başlıyor...
""")
        
        for i, symbol in enumerate(symbols, 1):
            # İlerleme göster
            progress_pct = ((i - 1) / len(symbols)) * 100
            elapsed = (datetime.now() - self.stats['start_time']).total_seconds()
            
            if i > 1:
                eta = (elapsed / (i - 1)) * (len(symbols) - i + 1)
                eta_str = str(timedelta(seconds=int(eta)))
            else:
                eta_str = "Hesaplanıyor..."
            
            print(f"\n[{i}/{len(symbols)}] ({progress_pct:.1f}%) - ETA: {eta_str}")
            
            # İndir
            self.download_symbol(symbol)
            
            # Her 10 sembolde bir özet
            if i % 10 == 0:
                self.print_progress_summary()
        
        # Final özet
        self.print_final_summary()
    
    def print_progress_summary(self):
        """Ara ilerleme özeti"""
        total_processed = self.stats['successful'] + self.stats['failed'] + self.stats['skipped']
        success_rate = (self.stats['successful'] / total_processed * 100) if total_processed > 0 else 0
        
        print(f"""
--- İlerleme Özeti ---
Başarılı: {self.stats['successful']}
Başarısız: {self.stats['failed']}
Atlanan: {self.stats['skipped']}
Başarı oranı: {success_rate:.1f}%
""")
    
    def print_final_summary(self):
        """Final özet raporu"""
        elapsed = datetime.now() - self.stats['start_time']
        total_processed = self.stats['successful'] + self.stats['failed'] + self.stats['skipped']
        
        print(f"""
\n{'='*50}
ALPHA VANTAGE İNDİRME TAMAMLANDI
{'='*50}

Toplam süre: {elapsed}
Toplam sembol: {self.stats['total_symbols']}
İşlenen: {total_processed}
  - Başarılı: {self.stats['successful']}
  - Başarısız: {self.stats['failed']}
  - Atlanan: {self.stats['skipped']}

Başarı oranı: {(self.stats['successful'] / total_processed * 100) if total_processed > 0 else 0:.1f}%

Veriler kaydedildi: data/raw/
{'='*50}
""")
        
        # Başarısız sembolleri listele
        if self.stats['failed'] > 0:
            print("\nBaşarısız semboller:")
            for symbol in ASSETS:
                if symbol not in self.progress or not self.progress[symbol].get('completed', False):
                    if symbol not in [s for s in ASSETS if self.progress.get(s, {}).get('completed', False)]:
                        print(f"  - {symbol}")
    
    def check_existing_data(self):
        """Mevcut verileri kontrol et"""
        print("\nMevcut veriler kontrol ediliyor...")
        
        existing = {}
        for symbol in ASSETS:
            existing[symbol] = {'1d': False, '4h': False, '1h': False, '15m': False}
            
            for tf in ['1d', '4h', '1h', '15m']:
                file_path = Path(f"data/raw/{symbol}_{tf}_raw.csv")
                if file_path.exists():
                    existing[symbol][tf] = True
        
        # Özet
        symbols_with_data = sum(1 for s in existing if any(existing[s].values()))
        print(f"\nVeri olan sembol sayısı: {symbols_with_data}/{len(ASSETS)}")
        
        return existing


def main():
    """Ana fonksiyon"""
    downloader = AlphaVantageDownloader()
    
    # Menü
    print("""
1. Tüm sembolleri indir
2. Eksik sembolleri indir
3. Belirli sembolleri indir
4. Mevcut verileri kontrol et
5. Progress'i sıfırla
0. Çıkış
""")
    
    choice = input("Seçiminiz (0-5): ")
    
    if choice == '1':
        # Tümünü indir
        downloader.download_all()
        
    elif choice == '2':
        # Eksikleri indir
        existing = downloader.check_existing_data()
        missing = [s for s in ASSETS if not any(existing[s].values())]
        
        if missing:
            print(f"\n{len(missing)} eksik sembol bulundu")
            downloader.download_all(missing)
        else:
            print("\nTüm semboller mevcut!")
    
    elif choice == '3':
        # Belirli sembolleri indir
        symbols_input = input("\nSembolleri girin (virgülle ayırın): ")
        symbols = [s.strip().upper() for s in symbols_input.split(',')]
        
        valid_symbols = [s for s in symbols if s in ASSETS]
        if valid_symbols:
            downloader.download_all(valid_symbols)
        else:
            print("Geçersiz sembol!")
    
    elif choice == '4':
        # Kontrol et
        downloader.check_existing_data()
        
    elif choice == '5':
        # Progress sıfırla
        if downloader.progress_file.exists():
            downloader.progress_file.unlink()
            print("Progress sıfırlandı")
        downloader.progress = {}
    
    elif choice == '0':
        print("Çıkış yapılıyor...")
        sys.exit(0)


if __name__ == "__main__":
    main()