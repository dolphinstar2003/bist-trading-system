#!/usr/bin/env python3
"""
API Bağlantı Helper
SMS kodu sorununu çözmek için
"""

import sys
from pathlib import Path

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algolab.algolab_wrapper import AlgoLabWrapper
from loguru import logger


def connect_with_sms():
    """SMS kodu ile API'ye bağlan"""
    
    api = AlgoLabWrapper()
    
    # İlk adım: Login ve SMS gönder
    logger.info("Connecting to API...")
    result = api.connect()
    
    if not result:
        logger.info("SMS sent. Please check your phone.")
        
        # SMS kodunu al
        sms_code = input("\nEnter SMS code: ")
        
        # SMS kodu ile tekrar bağlan
        result = api.connect(sms_code=sms_code)
        
        if result:
            logger.success("Connected successfully!")
            
            # Test için hesap bilgisi al
            account_info = api.get_account_info()
            if account_info:
                logger.info(f"Account info: {account_info}")
            
            # Bağlantıyı kapat
            api.disconnect()
            return True
        else:
            logger.error("Connection failed with SMS code")
            return False
    else:
        logger.success("Connected with existing session!")
        api.disconnect()
        return True


def main():
    """Ana program"""
    
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python connect_api.py")
        print("This will connect to API and handle SMS verification")
        return
    
    success = connect_with_sms()
    
    if success:
        logger.success("\n✅ API connection established successfully!")
        logger.info("You can now run data download scripts.")
    else:
        logger.error("\n❌ API connection failed!")
        logger.info("Please check your credentials and try again.")


if __name__ == "__main__":
    main()