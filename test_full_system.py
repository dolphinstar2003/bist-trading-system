#!/usr/bin/env python3
"""
AlgoLab Trading System - Full Test
"""

import sys
import time
from datetime import datetime
from algolab_wrapper import AlgoLabWrapper
from utils.logger import get_logger
from utils.csv_data_manager import CSVDataManager

logger = get_logger(__name__)


def test_connection_and_data():
    """Bağlantı ve veri testi"""
    print("=" * 60)
    print("AlgoLab Trading System - Full Test")
    print("=" * 60)
    
    # 1. Wrapper oluştur
    wrapper = AlgoLabWrapper()
    csv_manager = CSVDataManager()
    
    # 2. Bağlantı test
    print("\n1. API Bağlantısı:")
    connected = wrapper.connect()
    
    if not connected and wrapper.token:
        # SMS doğrulaması gerekiyor
        sms_code = input("\nSMS kodunu girin: ")
        connected = wrapper.connect(sms_code=sms_code)
    
    if not connected:
        print("❌ Bağlantı başarısız!")
        return False
    
    print("✅ Bağlantı başarılı!")
    
    # 3. Hesap bilgileri
    print("\n2. Hesap Bilgileri:")
    cash_info = wrapper.get_account_info()
    if cash_info and cash_info.get("success"):
        content = cash_info["content"]
        print(f"   T+0: {content.get('t0', '0')} TL")
        print(f"   T+1: {content.get('t1', '0')} TL")
        print(f"   T+2: {content.get('t2', '0')} TL")
    
    # 4. Pozisyonlar
    print("\n3. Açık Pozisyonlar:")
    positions = wrapper.get_positions()
    if positions:
        print(f"   {len(positions)} pozisyon bulundu")
        for pos in positions[:3]:  # İlk 3 pozisyon
            print(f"   - {pos.get('code')}: {pos.get('totalstock')} adet @ {pos.get('cost')} TL")
    else:
        print("   Açık pozisyon yok")
    
    # 5. Market data test
    print("\n4. Market Data Test:")
    test_symbols = ["THYAO", "GARAN", "AKBNK"]
    
    for symbol in test_symbols:
        print(f"\n   {symbol} verisi çekiliyor...")
        
        # Sembol bilgisi
        info = wrapper.api.GetEquityInfo(symbol)
        if info and info.get("success"):
            content = info["content"]
            print(f"   - Son: {content.get('lst')} TL")
            print(f"   - Taban: {content.get('flr')} - Tavan: {content.get('clg')}")
        
        # Günlük veri
        df = wrapper.get_market_data(symbol, period="1d", bar_count=5)
        if df is not None:
            print(f"   - {len(df)} günlük veri alındı")
            print(f"   - Son kapanış: {df['close'].iloc[-1] if 'close' in df.columns else 'N/A'}")
            
            # CSV'ye kaydet
            csv_manager.save_raw_data(symbol, df, "1d")
            print(f"   - CSV'ye kaydedildi")
        else:
            print(f"   ❌ Veri alınamadı")
        
        time.sleep(5.1)  # Rate limit
    
    # 6. Session refresh testi
    print("\n5. Session Refresh Testi:")
    result = wrapper.api.SessionRefresh()
    if result and result.get("success"):
        print("   ✅ Session refresh çalışıyor")
        print("   - Otomatik yenileme 14 dakikada bir yapılacak")
    else:
        print("   ❌ Session refresh başarısız")
    
    # 7. Order history
    print("\n6. İşlem Geçmişi:")
    orders = wrapper.get_equity_order_history()
    if orders:
        print(f"   {len(orders)} işlem bulundu")
    else:
        print("   İşlem geçmişi boş")
    
    # Bağlantıyı kapat
    print("\n7. Bağlantı kapatılıyor...")
    wrapper.disconnect()
    print("✅ Test tamamlandı!")
    
    return True


def test_trading_simulation():
    """Trading simülasyonu"""
    print("\n" + "=" * 60)
    print("Trading Simülasyonu (SADECE TEST)")
    print("=" * 60)
    
    confirm = input("\nBu test GERÇEK EMİR göndermez. Devam? (E/H): ")
    if confirm.upper() != 'E':
        return
    
    wrapper = AlgoLabWrapper()
    
    # Bağlan
    if not wrapper.connect():
        if wrapper.token:
            sms_code = input("\nSMS kodu: ")
            if not wrapper.connect(sms_code=sms_code):
                return
        else:
            return
    
    # Simülasyon parametreleri
    test_symbol = "THYAO"
    
    # Sembol bilgisi al
    info = wrapper.api.GetEquityInfo(test_symbol)
    if info and info.get("success"):
        content = info["content"]
        current_price = float(content["lst"])
        floor_price = float(content["flr"])
        ceiling_price = float(content["clg"])
        
        print(f"\n{test_symbol} Bilgileri:")
        print(f"Son Fiyat: {current_price} TL")
        print(f"Taban: {floor_price} - Tavan: {ceiling_price}")
        
        # Simüle edilecek emirler
        print("\n--- SİMÜLASYON (Gerçek emir gönderilmeyecek) ---")
        
        # Alım emri
        buy_price = round(current_price * 0.99, 2)  # %1 altından
        print(f"\n1. Alım Emri:")
        print(f"   Sembol: {test_symbol}")
        print(f"   Fiyat: {buy_price} TL (Limit)")
        print(f"   Adet: 100")
        print(f"   Tutar: {buy_price * 100:.2f} TL")
        
        # Satım emri  
        sell_price = round(current_price * 1.01, 2)  # %1 üstünden
        print(f"\n2. Satım Emri:")
        print(f"   Sembol: {test_symbol}")
        print(f"   Fiyat: {sell_price} TL (Limit)")
        print(f"   Adet: 100")
        print(f"   Tutar: {sell_price * 100:.2f} TL")
        
        # Risk yönetimi
        print(f"\n3. Risk Yönetimi:")
        print(f"   Stop Loss: {round(buy_price * 0.98, 2)} TL (%2 zarar)")
        print(f"   Take Profit: {round(buy_price * 1.03, 2)} TL (%3 kar)")
    
    wrapper.disconnect()
    print("\n✅ Simülasyon tamamlandı!")


def main():
    """Ana test fonksiyonu"""
    try:
        # Temel test
        success = test_connection_and_data()
        
        if success:
            # Trading simülasyonu
            test_trading_simulation()
        
    except KeyboardInterrupt:
        print("\n\nTest kullanıcı tarafından durduruldu.")
    except Exception as e:
        logger.error(f"Test hatası: {e}")
        import traceback
        traceback.print_exc()
    
    # Programı temiz kapat
    import os
    os._exit(0)


if __name__ == "__main__":
    main()