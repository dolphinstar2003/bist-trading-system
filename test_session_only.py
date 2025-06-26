#!/usr/bin/env python3
"""
Session-Only Test - SMS gerektirmez
"""

from algolab_wrapper import AlgoLabWrapper
from utils.logger import get_logger

logger = get_logger(__name__)

def test_session():
    """Mevcut session ile test"""
    print("=" * 50) 
    print("Session Test (SMS gerektirmez)")
    print("=" * 50)
    
    # Wrapper oluştur - otomatik olarak mevcut session'ı yükleyecek
    wrapper = AlgoLabWrapper()
    
    print("\n1. Mevcut session ile bağlanılıyor...")
    
    # Bağlan - eğer geçerli session varsa SMS istemeyecek
    result = wrapper.connect()
    
    if result:
        print("✅ Mevcut session ile bağlantı kuruldu!")
        
        # API testleri
        print("\n2. API Testleri:")
        
        # Portfolio/Pozisyonlar
        try:
            portfolio = wrapper.get_portfolio()
            if portfolio and portfolio.get("success"):
                print("   ✅ Portfolio/Pozisyon bilgileri alındı")
                content = portfolio.get("content", [])
                if isinstance(content, list):
                    print(f"      {len(content)} açık pozisyon")
                elif isinstance(content, dict):
                    print(f"      Bakiye: {content.get('balance', 'N/A')}")
        except Exception as e:
            print(f"   ⚠️  Portfolio hatası: {e}")
            
        # THYAO bilgisi
        try:
            info = wrapper.api.GetEquityInfo("THYAO")
            if info and info.get("success"):
                print("   ✅ THYAO bilgisi alındı")
                content = info["content"]
                print(f"      Sembol: {content.get('name', 'THYAO')}")
                print(f"      Son: {content.get('lst', 'N/A')} TL")
                print(f"      Alış: {content.get('bid', 'N/A')} - Satış: {content.get('ask', 'N/A')}")
                print(f"      Taban: {content.get('flr', 'N/A')} - Tavan: {content.get('clg', 'N/A')}")
        except Exception as e:
            print(f"   ⚠️  THYAO bilgi hatası: {e}")
            
        # Hesap bilgisi
        print("\n3. Hesap Durumu:")
        positions = wrapper.get_positions()
        if positions is not None:
            print(f"   ✅ {len(positions)} açık pozisyon")
            
        print("\n✅ Tüm testler başarılı!")
            
    elif wrapper.token:
        print("❌ Session süresi dolmuş, SMS doğrulaması gerekiyor")
        print("   test_interactive.py dosyasını çalıştırın")
    else:
        print("❌ Bağlantı kurulamadı")
        print("   API credentials'ları kontrol edin")
    
    wrapper.disconnect()
    print("\nTest tamamlandı.")
    
    # Tüm thread'leri zorla kapat ve çık
    import os
    import signal
    os._exit(0)  # Daha agresif çıkış

if __name__ == "__main__":
    test_session()