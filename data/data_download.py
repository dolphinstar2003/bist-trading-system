#!/usr/bin/env python3
"""
Veri İndirme Modülü

Bu modül AlgoLab API'sinden hisse verilerini indirir ve CSV dosyalarına kaydeder.
Her çalıştırmada CSV'deki son veri tarihini kontrol eder ve sadece yeni verileri indirir.
"""

import os
import sys
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Proje kök dizinini Python path'e ekle
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from algolab_wrapper import AlgoLabWrapper
from utils.logger import get_logger
from utils.csv_data_manager import CSVDataManager

logger = get_logger("data_download")


class DataDownloader:
    """Veri indirme ve güncelleme sınıfı"""
    
    def __init__(self):
        """DataDownloader başlat"""
        self.wrapper = AlgoLabWrapper()
        self.csv_manager = CSVDataManager()
        
        # Settings'den hisse listesini al
        settings_path = os.path.join(project_root, "settings.json")
        with open(settings_path, "r") as f:
            self.settings = json.load(f)
        
        self.symbols = self.settings["trading"]["symbols"]
        self.timeframes = ["15m", "1h", "4h", "1d"]  # İndirilecek timeframe'ler
        
        # Rate limit ayarları
        self.rate_limit_delay = 5.1  # API rate limit: 5 saniyede 1 istek
        
        logger.info(f"DataDownloader başlatıldı. {len(self.symbols)} hisse için veri indirilecek")
    
    def get_last_data_date(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """CSV'deki son veri tarihini al"""
        try:
            # Raw data dosyasını kontrol et
            file_path = self.csv_manager.raw_data_path / f"{symbol}_{timeframe}_raw.csv"
            
            if not file_path.exists():
                logger.debug(f"{symbol} {timeframe} için veri dosyası bulunamadı")
                return None
            
            # Son satırdaki tarihi al
            df = pd.read_csv(file_path)
            if df.empty:
                return None
            
            # Tarih sütununu datetime'a çevir
            df['Tarih'] = pd.to_datetime(df['Tarih'])
            last_date = df['Tarih'].max()
            
            logger.debug(f"{symbol} {timeframe} son veri tarihi: {last_date}")
            return last_date
            
        except Exception as e:
            logger.error(f"Son veri tarihi alınamadı {symbol} {timeframe}: {e}")
            return None
    
    def calculate_bars_needed(self, last_date: Optional[datetime], timeframe: str) -> int:
        """İndirilmesi gereken bar sayısını hesapla"""
        if last_date is None:
            # İlk indirme: maksimum veri al
            if timeframe == "1d":
                return 500  # 2 yıllık günlük veri
            elif timeframe == "4h":
                return 420  # 70 günlük 4 saatlik veri
            elif timeframe == "1h":
                return 720  # 30 günlük saatlik veri
            else:  # 15m
                return 480  # 5 günlük 15 dakikalık veri
        
        # Güncelleme: son tarihten bugüne kadar
        now = datetime.now()
        time_diff = now - last_date
        
        if timeframe == "1d":
            bars_needed = time_diff.days + 1
        elif timeframe == "4h":
            bars_needed = int(time_diff.total_seconds() / 14400) + 1  # 4 saat = 14400 saniye
        elif timeframe == "1h":
            bars_needed = int(time_diff.total_seconds() / 3600) + 1
        else:  # 15m
            bars_needed = int(time_diff.total_seconds() / 900) + 1
        
        # Minimum 10 bar, maksimum limitleri aş
        bars_needed = max(10, bars_needed)
        
        if timeframe == "1d":
            bars_needed = min(bars_needed, 250)
        elif timeframe == "4h":
            bars_needed = min(bars_needed, 300)
        elif timeframe == "1h":
            bars_needed = min(bars_needed, 500)
        else:  # 15m
            bars_needed = min(bars_needed, 200)
        
        return bars_needed
    
    def download_symbol_data(self, symbol: str) -> Dict[str, bool]:
        """Bir hisse için tüm timeframe verilerini indir"""
        results = {}
        
        for timeframe in self.timeframes:
            try:
                # Son veri tarihini kontrol et
                last_date = self.get_last_data_date(symbol, timeframe)
                
                # İndirilecek bar sayısını hesapla
                bars_needed = self.calculate_bars_needed(last_date, timeframe)
                
                if last_date and bars_needed < 5:
                    logger.info(f"{symbol} {timeframe} zaten güncel, atlanıyor")
                    results[timeframe] = True
                    continue
                
                logger.info(f"{symbol} {timeframe} için {bars_needed} bar indiriliyor...")
                
                # Veriyi indir
                data = self.wrapper.get_market_data(
                    symbol=symbol,
                    period=timeframe,
                    bar_count=bars_needed
                )
                
                if data is not None and not data.empty:
                    # CSV'ye kaydet (doğru parametre sırası: symbol, data, timeframe)
                    saved = self.csv_manager.save_raw_data(symbol, data, timeframe)
                    results[timeframe] = saved
                    
                    if saved:
                        logger.info(f"✅ {symbol} {timeframe} başarıyla indirildi")
                    else:
                        logger.error(f"❌ {symbol} {timeframe} kaydedilemedi")
                else:
                    logger.error(f"❌ {symbol} {timeframe} verisi alınamadı")
                    results[timeframe] = False
                
                # Rate limit bekle
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Hata {symbol} {timeframe}: {e}")
                results[timeframe] = False
                time.sleep(self.rate_limit_delay)
        
        return results
    
    def download_all_data(self):
        """Tüm hisseler için veri indir"""
        logger.info("=" * 60)
        logger.info("VERİ İNDİRME BAŞLIYOR")
        logger.info(f"Hisse sayısı: {len(self.symbols)}")
        logger.info(f"Timeframe'ler: {', '.join(self.timeframes)}")
        logger.info("=" * 60)
        
        # API bağlantısı kur
        result = self.wrapper.connect()
        
        if not result:
            # SMS kodu gerekiyor
            logger.info("SMS code sent to your phone.")
            sms_code = input("\nEnter SMS code: ")
            
            # SMS kodu ile tekrar dene
            result = self.wrapper.connect(sms_code=sms_code)
            
            if not result:
                logger.error("API bağlantısı kurulamadı!")
                return False
        
        # İstatistikler
        total_symbols = len(self.symbols)
        successful_symbols = 0
        failed_symbols = []
        
        start_time = datetime.now()
        
        try:
            for i, symbol in enumerate(self.symbols, 1):
                logger.info(f"\n[{i}/{total_symbols}] {symbol} işleniyor...")
                
                results = self.download_symbol_data(symbol)
                
                # Başarı kontrolü
                if all(results.values()):
                    successful_symbols += 1
                else:
                    failed_symbols.append(symbol)
                
                # İlerleme durumu
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / i
                remaining = avg_time * (total_symbols - i)
                
                logger.info(f"İlerleme: {i}/{total_symbols} ({i/total_symbols*100:.1f}%)")
                logger.info(f"Tahmini kalan süre: {int(remaining/60)} dakika {int(remaining%60)} saniye")
                
        except KeyboardInterrupt:
            logger.warning("\nİndirme kullanıcı tarafından durduruldu")
        except Exception as e:
            logger.error(f"Beklenmeyen hata: {e}")
        finally:
            # Bağlantıyı kapat
            self.wrapper.disconnect()
        
        # Özet rapor
        total_time = (datetime.now() - start_time).total_seconds()
        logger.info("\n" + "=" * 60)
        logger.info("İNDİRME TAMAMLANDI")
        logger.info(f"Toplam süre: {int(total_time/60)} dakika {int(total_time%60)} saniye")
        logger.info(f"Başarılı: {successful_symbols}/{total_symbols}")
        
        if failed_symbols:
            logger.warning(f"Başarısız hisseler: {', '.join(failed_symbols)}")
        
        logger.info("=" * 60)
        
        return successful_symbols == total_symbols
    
    def show_data_status(self):
        """Mevcut veri durumunu göster"""
        logger.info("\n" + "=" * 60)
        logger.info("MEVCUT VERİ DURUMU")
        logger.info("=" * 60)
        
        status_data = []
        
        for symbol in self.symbols[:10]:  # İlk 10 hisse için özet
            symbol_status = {"Hisse": symbol}
            
            for timeframe in self.timeframes:
                last_date = self.get_last_data_date(symbol, timeframe)
                if last_date:
                    days_old = (datetime.now() - last_date).days
                    symbol_status[timeframe] = f"{days_old} gün önce"
                else:
                    symbol_status[timeframe] = "Veri yok"
            
            status_data.append(symbol_status)
        
        # DataFrame olarak göster
        df = pd.DataFrame(status_data)
        print(df.to_string(index=False))
        
        logger.info(f"\n(İlk 10 hisse gösteriliyor. Toplam: {len(self.symbols)} hisse)")


def main():
    """Ana fonksiyon"""
    downloader = DataDownloader()
    
    # Komut satırı argümanları
    if len(sys.argv) > 1:
        if sys.argv[1] == "status":
            # Sadece durum göster
            downloader.show_data_status()
        elif sys.argv[1] == "download":
            # Veri indir
            downloader.download_all_data()
        else:
            print("Kullanım:")
            print("  python data_download.py status    # Veri durumunu göster")
            print("  python data_download.py download  # Verileri indir/güncelle")
            print("  python data_download.py          # Verileri indir/güncelle")
    else:
        # Varsayılan: veri indir
        downloader.download_all_data()


if __name__ == "__main__":
    main()