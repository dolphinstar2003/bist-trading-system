#!/usr/bin/env python3
"""
Basit AlgoLab Login Script
SMS kodu sorununu çözmek için
"""

import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# AlgoLab kütüphanesini import et
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from algolab.algolab import AlgoLab


def login_and_save_session():
    """AlgoLab'a login ol ve session'ı kaydet"""
    
    # .env dosyasını yükle
    load_dotenv('config/.env')
    
    username = os.getenv('ALGOLAB_USERNAME')
    password = os.getenv('ALGOLAB_PASSWORD')
    api_key = os.getenv('ALGOLAB_API_KEY')
    
    if not all([username, password, api_key]):
        logger.error("config/.env dosyasında eksik bilgiler var!")
        return False
    
    logger.info("AlgoLab API'ye bağlanılıyor...")
    
    # AlgoLab nesnesini oluştur (auto_login=False)
    api = AlgoLab(
        api_key=api_key,
        username=username,
        password=password,
        auto_login=False,  # Otomatik login yapmasın
        keep_alive=False,  # Keep alive thread başlatmasın
        verbose=True
    )
    
    # Manuel login yap
    logger.info("Login işlemi başlatılıyor...")
    login_result = api.LoginUser()
    
    if not login_result:
        logger.error("Login başarısız!")
        return False
    
    logger.success("SMS gönderildi! Telefonunuzu kontrol edin.")
    
    # LoginUserControl otomatik olarak input() ile SMS kodunu isteyecek
    # Kullanıcı SMS kodunu girecek
    control_result = api.LoginUserControl()
    
    if not control_result:
        logger.error("SMS doğrulama başarısız!")
        return False
    
    logger.success("Login başarılı!")
    
    # Session bilgilerini manuel kaydet
    session_data = {
        'hash': api.hash,
        'token': api.token,
        'username': username,
        'created_at': datetime.now().isoformat(),
        'expires_at': (datetime.now() + timedelta(minutes=15)).isoformat()
    }
    
    session_file = Path('.algolab_session.json')
    with open(session_file, 'w') as f:
        json.dump(session_data, f, indent=2)
    
    logger.success(f"Session kaydedildi: {session_file}")
    
    # Test et
    try:
        logger.info("Bağlantı test ediliyor...")
        
        # Hesap bilgisi al
        cash_flow = api.CashFlow()
        if cash_flow and cash_flow.get('success'):
            content = cash_flow['content']
            logger.info(f"T+0: {content.get('t0', '0')} TL")
            logger.info(f"T+1: {content.get('t1', '0')} TL")
            logger.info(f"T+2: {content.get('t2', '0')} TL")
        
        # Bir hisse bilgisi al
        thyao = api.GetEquityInfo("THYAO")
        if thyao and thyao.get('success'):
            content = thyao['content']
            logger.info(f"THYAO - Son: {content['lst']} TL")
    except Exception as e:
        logger.warning(f"Test sırasında hata: {e}")
    
    return True


def check_session():
    """Mevcut session'ı kontrol et"""
    
    session_file = Path('.algolab_session.json')
    
    if not session_file.exists():
        logger.info("Session dosyası bulunamadı.")
        return False
    
    with open(session_file, 'r') as f:
        session_data = json.load(f)
    
    expires_at = datetime.fromisoformat(session_data['expires_at'])
    remaining = expires_at - datetime.now()
    
    if remaining.total_seconds() > 0:
        logger.info(f"Session geçerli ({remaining.total_seconds()/60:.1f} dakika kaldı)")
        logger.info(f"Hash: {session_data['hash'][:50]}...")
        return True
    else:
        logger.warning("Session süresi dolmuş.")
        return False


def main():
    """Ana program"""
    
    print("\n" + "="*60)
    print("ALGOLAB SIMPLE LOGIN")
    print("="*60)
    
    # Mevcut session kontrol
    if check_session():
        print("\nMevcut bir session var.")
        choice = input("Yeni login yapmak ister misiniz? (e/H): ")
        if choice.lower() != 'e':
            return
    
    # Yeni login
    print("\nYeni login işlemi başlatılıyor...")
    print("SMS kodu geldiğinde, kodu girip ENTER'a basın.")
    print("="*60)
    
    success = login_and_save_session()
    
    if success:
        print("\n✅ Login başarılı!")
        print("\nŞimdi şu komutları çalıştırabilirsiniz:")
        print("- python data_download.py")
        print("- python data_download_incremental.py")
        print("- python data_download_advanced.py")
        print("\nSession 15 dakika geçerli olacak.")
    else:
        print("\n❌ Login başarısız!")
        print("Lütfen credentials'ları kontrol edin.")


if __name__ == "__main__":
    main()