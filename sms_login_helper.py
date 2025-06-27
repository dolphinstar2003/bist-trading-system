#!/usr/bin/env python3
"""
SMS Login Helper
Kolay SMS kodu girişi için
"""

import os
import sys
import time
from pathlib import Path
from loguru import logger
from algolab_wrapper import AlgoLabWrapper


def clear_screen():
    """Ekranı temizle"""
    os.system('clear' if os.name == 'posix' else 'cls')


def animated_waiting(duration=5):
    """Animasyonlu bekleme"""
    for i in range(duration):
        for char in ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']:
            print(f'\r{char} Waiting... ({duration-i}s)', end='', flush=True)
            time.sleep(0.1)
    print('\r✓ Ready!                    ')


def login_with_sms():
    """SMS ile login işlemi"""
    
    api = AlgoLabWrapper()
    
    # İlk login denemesi
    clear_screen()
    print("="*60)
    print("ALGOLAB API LOGIN")
    print("="*60)
    print("\n1. Attempting to connect...")
    
    result = api.connect()
    
    if result:
        print("\n✅ Connected with existing session!")
        api.disconnect()
        return True
    
    # SMS gönderildi
    print("\n📱 SMS code sent to your registered phone!")
    print("\n" + "="*60)
    print("IMPORTANT:")
    print("- Check your phone for SMS code")
    print("- The code is valid for 3 minutes")
    print("- You have 3 attempts")
    print("="*60)
    
    # SMS kodunu al
    print("\n")
    sms_code = input("Enter SMS code (6 digits): ").strip()
    
    # SMS kodu ile bağlan
    print(f"\n2. Verifying SMS code: {sms_code}")
    animated_waiting(3)
    
    result = api.connect(sms_code=sms_code)
    
    if result:
        print("\n✅ SUCCESS! API connected with SMS verification")
        
        # Test
        print("\n3. Testing connection...")
        account = api.get_account_info()
        if account:
            print("✅ Connection test successful!")
        
        api.disconnect()
        return True
    else:
        print("\n❌ FAILED! SMS verification failed")
        print("\nPossible reasons:")
        print("- Wrong SMS code")
        print("- Code expired (3 minutes)")
        print("- Too many failed attempts (2 hour block after 3 fails)")
        return False


def main():
    """Ana program"""
    
    success = login_with_sms()
    
    if success:
        print("\n" + "="*60)
        print("✅ LOGIN SUCCESSFUL!")
        print("="*60)
        print("\nYou can now run:")
        print("- python data_download.py")
        print("- python data_download_incremental.py")
        print("- python data_download_advanced.py")
        print("\nSession will remain active for ~15 minutes")
    else:
        print("\n" + "="*60)
        print("❌ LOGIN FAILED!")
        print("="*60)
        print("\nTroubleshooting:")
        print("1. Wait 2 hours if you're blocked")
        print("2. Check your credentials in config/.env")
        print("3. Contact Denizbank: 0850 222 0 800")


if __name__ == "__main__":
    main()