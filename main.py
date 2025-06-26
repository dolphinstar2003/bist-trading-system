#!/usr/bin/env python3
"""
BIST Algoritmik Trading Sistemi - Ana Program
"""

import argparse
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Proje modülleri
from utils.logger import get_logger, log_performance
from utils.csv_data_manager import CSVDataManager
from algolab_wrapper import AlgoLabWrapper

logger = get_logger(__name__)


class TradingSystem:
    """Ana trading sistemi sınıfı"""
    
    def __init__(self, mode='paper'):
        self.mode = mode
        self.running = False
        
        # Yapılandırma yükle
        with open("settings.json", "r") as f:
            self.settings = json.load(f)
        
        # Bileşenleri başlat
        self.csv_manager = CSVDataManager()
        self.algolab = AlgoLabWrapper()
        
        logger.info(f"Trading System initialized in {mode} mode")
    
    def connect(self):
        """API bağlantılarını kur"""
        logger.info("Connecting to AlgoLab API...")
        
        # İlk bağlantı denemesi
        connected = self.algolab.connect()
        
        if not connected and not self.algolab.is_connected:
            # SMS doğrulama gerekiyor
            if self.algolab.token:
                # SMS gönderildi, kullanıcıdan SMS kodu iste
                sms_code = input("\nPlease enter the SMS code sent to your phone: ")
                
                # SMS ile tekrar dene
                if self.algolab.connect(sms_code=sms_code):
                    logger.info("API connection successful with SMS verification")
                    return True
                else:
                    logger.error("Failed to connect with SMS code")
                    return False
            else:
                logger.error("Failed to send SMS")
                return False
        elif connected:
            logger.info("API connection successful")
            return True
        else:
            logger.error("Failed to connect to AlgoLab API")
            return False
    
    def update_market_data(self):
        """Tüm hisseler için market verilerini güncelle"""
        logger.info("Updating market data for all symbols...")
        
        results = self.algolab.update_all_market_data(self.csv_manager)
        
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Market data update completed: {success_count}/{len(results)} successful")
        
        return results
    
    def show_account_info(self):
        """Hesap bilgilerini göster"""
        # Nakit bakiye bilgisi
        cash_info = self.algolab.get_account_info()
        
        if cash_info and cash_info.get("success"):
            content = cash_info.get("content", {})
            logger.info("=== Nakit Bakiye ===")
            logger.info(f"T+0 (Bugün): {content.get('t0', '0')} TL")
            logger.info(f"T+1 (Yarın): {content.get('t1', '0')} TL")
            logger.info(f"T+2: {content.get('t2', '0')} TL")
            
            # Kullanılabilir bakiye (en düşük değer)
            try:
                t0 = float(content.get('t0', 0))
                t1 = float(content.get('t1', 0))
                t2 = float(content.get('t2', 0))
                available = min(t0, t1, t2)
                logger.info(f"Kullanılabilir: {available:,.2f} TL")
            except:
                pass
        
        # Pozisyonları göster
        positions = self.algolab.get_positions()
        if positions and isinstance(positions, list) and len(positions) > 0:
            logger.info(f"\n=== Açık Pozisyonlar ({len(positions)}) ===")
            
            total_value = 0
            for pos in positions:
                try:
                    code = pos.get('code', 'N/A')
                    quantity = float(pos.get('totalstock', 0))
                    cost = float(pos.get('cost', 0))
                    current = float(pos.get('unitprice', 0))
                    value = float(pos.get('tlamaount', 0))
                    profit = float(pos.get('profit', 0))
                    
                    logger.info(f"{code}: {quantity:,.0f} adet")
                    logger.info(f"  Maliyet: {cost:.2f} - Güncel: {current:.2f} TL")
                    logger.info(f"  Değer: {value:,.2f} TL - K/Z: {profit:+,.2f} TL")
                    
                    total_value += value
                except Exception as e:
                    logger.error(f"Error parsing position: {e}")
            
            if total_value > 0:
                logger.info(f"\nToplam Portföy Değeri: {total_value:,.2f} TL")
        else:
            logger.info("\nAçık pozisyon yok")
    
    def run_paper_trading(self):
        """Paper trading modunu çalıştır"""
        logger.info("Starting paper trading mode...")
        self.running = True
        
        while self.running:
            try:
                # Her döngüde yapılacaklar
                current_time = datetime.now()
                
                # Market açık mı kontrol et (09:10 - 18:00)
                # Hafta sonu kontrolü
                if current_time.weekday() >= 5:  # Cumartesi=5, Pazar=6
                    logger.info("Hafta sonu - Market kapalı")
                    time.sleep(300)  # 5 dakika bekle
                    continue
                    
                if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 10) or current_time.hour >= 18:
                    logger.info("Market kapalı. Bekleniyor...")
                    time.sleep(60)  # 1 dakika bekle
                    continue
                
                # TODO: Strateji çalıştır
                logger.info(f"Running strategies at {current_time.strftime('%H:%M:%S')}...")
                
                # TODO: Sinyalleri değerlendir
                
                # TODO: Emirleri gönder
                
                # Performans logla
                log_performance("heartbeat", 1.0, mode=self.mode)
                
                # Bekleme süresi (varsayılan 30 saniye)
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("Paper trading stopped by user")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in paper trading loop: {str(e)}")
                time.sleep(60)  # Hata durumunda 1 dakika bekle
    
    def run_live_trading(self):
        """Canlı trading modunu çalıştır"""
        logger.warning("LIVE TRADING MODE - Real money will be used!")
        
        # Onay iste
        confirm = input("Are you sure you want to start LIVE trading? (yes/no): ")
        if confirm.lower() != 'yes':
            logger.info("Live trading cancelled")
            return
        
        # Paper trading ile aynı mantık, gerçek emirlerle
        self.run_paper_trading()
    
    def run_backtest(self, start_date, end_date):
        """Backtest modunu çalıştır"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # TODO: Backtest implementasyonu
        logger.info("Backtest functionality not yet implemented")
    
    def stop(self):
        """Sistemi durdur"""
        logger.info("Stopping trading system...")
        self.running = False
        
        if self.algolab:
            self.algolab.disconnect()
        
        logger.info("Trading system stopped")


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='BIST Algoritmik Trading Sistemi')
    
    parser.add_argument(
        '--mode',
        choices=['live', 'paper', 'backtest', 'update-data'],
        default='paper',
        help='Çalışma modu (default: paper)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Backtest başlangıç tarihi (YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='Backtest bitiş tarihi (YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    # Trading sistemi oluştur
    system = TradingSystem(mode=args.mode)
    
    try:
        # API bağlantısı kur
        if not system.connect():
            logger.error("Failed to initialize system")
            sys.exit(1)
        
        # Modlara göre çalıştır
        if args.mode == 'update-data':
            # Sadece veri güncelle
            system.update_market_data()
            
        elif args.mode == 'backtest':
            # Backtest modu
            if not args.start or not args.end:
                logger.error("Backtest requires --start and --end dates")
                sys.exit(1)
            system.run_backtest(args.start, args.end)
            
        elif args.mode == 'paper':
            # Paper trading
            system.show_account_info()
            system.run_paper_trading()
            
        elif args.mode == 'live':
            # Canlı trading
            system.show_account_info()
            system.run_live_trading()
    
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.error(f"System error: {str(e)}")
    finally:
        system.stop()


if __name__ == "__main__":
    main()