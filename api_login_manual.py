#!/usr/bin/env python3
"""
Manuel API Login
SMS kodu sorununu çözmek için alternatif yöntem
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from AlgoLab import AlgoLab


def manual_login():
    """Manuel login işlemi"""
    
    # Credentials
    from dotenv import load_dotenv
    load_dotenv('config/.env')
    
    username = os.getenv('ALGOLAB_USERNAME')
    password = os.getenv('ALGOLAB_PASSWORD')
    api_key = os.getenv('ALGOLAB_API_KEY')
    
    if not all([username, password, api_key]):
        logger.error("Missing credentials in config/.env")
        return False
    
    # API nesnesi
    api = AlgoLab(api_key=api_key, username=username, password=password, keep_alive=True)
    
    # Login - SMS gönder
    logger.info("Sending login request...")
    login_result = api.LoginUser()
    
    if login_result:
        logger.success("Login successful, SMS sent!")
        
        # Token'ı sakla
        token = api.token
        logger.info(f"Token received: {token[:10]}...")
        
        # SMS kodunu manuel al
        print("\n" + "="*50)
        print("SMS CODE REQUIRED")
        print("="*50)
        print("Check your phone for SMS code")
        sms_code = input("Enter SMS code here: ")
        
        # SMS ile doğrula
        logger.info("Verifying SMS code...")
        api.sms_code = sms_code
        
        # Login control
        control_result = api.LoginUserControl()
        
        if control_result:
            logger.success("SMS verification successful!")
            
            # Hash'i sakla
            if hasattr(api, 'hash') and api.hash:
                session_data = {
                    'hash': api.hash,
                    'username': username,
                    'created_at': datetime.now().isoformat(),
                    'expires_at': (datetime.now() + timedelta(minutes=15)).isoformat()
                }
                
                # Session dosyasına kaydet
                session_file = Path('.algolab_session.json')
                with open(session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                
                logger.success(f"Session saved to {session_file}")
                logger.info(f"Hash: {api.hash[:20]}...")
                
                # Test - hesap bilgisi
                try:
                    cash_flow = api.CashFlow()
                    logger.info(f"Account test - CashFlow: {cash_flow}")
                except Exception as e:
                    logger.warning(f"CashFlow test failed: {e}")
                
                return True
            else:
                logger.error("No hash received")
                return False
        else:
            logger.error("SMS verification failed")
            return False
    else:
        logger.error("Initial login failed")
        return False


def check_existing_session():
    """Mevcut session'ı kontrol et"""
    
    session_file = Path('.algolab_session.json')
    
    if session_file.exists():
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        expires_at = datetime.fromisoformat(session_data['expires_at'])
        remaining = expires_at - datetime.now()
        
        if remaining.total_seconds() > 0:
            logger.info(f"Existing session found ({remaining.total_seconds()/60:.1f} minutes remaining)")
            logger.info(f"Hash: {session_data['hash'][:20]}...")
            return True
        else:
            logger.warning("Session expired")
            return False
    else:
        logger.info("No existing session found")
        return False


def main():
    """Ana program"""
    
    print("\nAlgoLab Manual Login")
    print("="*50)
    
    # Mevcut session kontrol
    if check_existing_session():
        logger.info("You can use the existing session")
        choice = input("\nCreate new session anyway? (y/N): ")
        if choice.lower() != 'y':
            return
    
    # Yeni login
    success = manual_login()
    
    if success:
        print("\n✅ Login successful!")
        print("You can now run data download scripts")
    else:
        print("\n❌ Login failed!")
        print("Please check your credentials and try again")


if __name__ == "__main__":
    main()