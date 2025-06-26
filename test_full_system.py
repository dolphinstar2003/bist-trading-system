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
    
    # 3. Alt hesapları kontrol et
    print("\n2. Alt Hesaplar:")
    try:
        sub_accounts = wrapper.api.GetEquitySubAccounts()
        if sub_accounts and sub_accounts.get("success"):
            accounts = sub_accounts.get("content", [])
            print(f"   {len(accounts)} alt hesap bulundu")
            for acc in accounts:
                if isinstance(acc, dict):
                    print(f"   - {acc.get('name', 'N/A')}: {acc.get('code', 'N/A')}")
        time.sleep(5.1)
    except Exception as e:
        print(f"   Alt hesap bilgisi alınamadı: {e}")
    
    # 4. Hesap bilgileri
    print("\n3. Hesap Bilgileri:")
    cash_info = wrapper.get_account_info()
    if cash_info and cash_info.get("success"):
        content = cash_info["content"]
        print(f"   T+0: {content.get('t0', '0')} TL")
        print(f"   T+1: {content.get('t1', '0')} TL")
        print(f"   T+2: {content.get('t2', '0')} TL")
        print(f"\n   Tüm Hesap Detayları:")
        for key, value in content.items():
            if value != '0' and value != 0 and value != '0.00':  # Sadece 0 olmayanları göster
                print(f"   - {key}: {value}")
    
    # Rate limit için bekle
    time.sleep(5.1)
    
    # 5. Pozisyonlar
    print("\n4. Açık Pozisyonlar:")
    positions = wrapper.get_positions()
    if positions and isinstance(positions, list):
        # Total satırını filtrele
        real_positions = [p for p in positions if p.get('explanation') != 'total']
        
        if real_positions:
            print(f"   {len(real_positions)} pozisyon bulundu")
            for i, pos in enumerate(real_positions):
                if isinstance(pos, dict):
                    print(f"\n   Pozisyon {i+1}:")
                    print(f"   - Hisse: {pos.get('code', 'N/A')}")
                    print(f"   - Adet: {pos.get('totalstock', 0)}")
                    print(f"   - Maliyet: {pos.get('cost', 0)} TL")
                    print(f"   - Güncel Fiyat: {pos.get('price', 0)} TL")
                    print(f"   - Kar/Zarar: {pos.get('profit', 0)} TL")
        else:
            print("   Gerçek pozisyon bulunmuyor (sadece total satırı var)")
    elif positions and isinstance(positions, dict):
        print(f"   {len(positions)} pozisyon bulundu (dict formatında)")
        print(f"   Dict keys: {list(positions.keys())}")
    else:
        print("   Açık pozisyon yok")
    
    # 6. Market data test
    print("\n5. Market Data Test:")
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
    
    # 7. Session refresh testi
    print("\n6. Session Refresh Testi:")
    result = wrapper.api.SessionRefresh()
    if result:  # SessionRefresh bool döndürüyor
        print("   ✅ Session refresh çalışıyor")
        print("   - Otomatik yenileme 14 dakikada bir yapılacak")
    else:
        print("   ❌ Session refresh başarısız")
    
    # 8. Order history
    print("\n7. İşlem Geçmişi:")
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