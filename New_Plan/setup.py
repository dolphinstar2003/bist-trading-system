#!/usr/bin/env python3
"""
Hibrit Trading Sistemi - Kurulum Script'i
"""

import os
import sys
import json
import shutil
from pathlib import Path
from loguru import logger


def setup_directories():
    """Gerekli dizinleri oluştur"""
    directories = [
        'logs',
        'data/cache',
        'data/models',
        'data/backtest',
        'reports',
        'notebooks'
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Dizin oluşturuldu: {dir_path}")


def setup_config():
    """Config dosyasını oluştur"""
    template_path = Path('configs/config_template.json')
    config_path = Path('configs/config.json')
    
    if not config_path.exists():
        shutil.copy(template_path, config_path)
        logger.info("Config dosyası oluşturuldu: configs/config.json")
        logger.warning("API anahtarlarını configs/config.json dosyasına eklemeyi unutmayın!")
    else:
        logger.info("Config dosyası zaten mevcut")


def check_dependencies():
    """Bağımlılıkları kontrol et"""
    try:
        import pandas
        import numpy
        import torch
        import xgboost
        import yfinance
        logger.success("Tüm temel bağımlılıklar yüklü")
    except ImportError as e:
        logger.error(f"Eksik bağımlılık: {e}")
        logger.info("Lütfen 'pip install -r requirements.txt' komutunu çalıştırın")
        return False
    
    return True


def setup_redis():
    """Redis bağlantısını kontrol et"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        logger.success("Redis bağlantısı başarılı")
    except Exception as e:
        logger.warning(f"Redis bağlantısı başarısız: {e}")
        logger.info("Redis kurulu değilse: sudo apt-get install redis-server")


def create_example_notebook():
    """Örnek Jupyter notebook oluştur"""
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Hibrit Trading Sistemi - Örnek Kullanım\n"]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "import sys\n",
                    "sys.path.append('..')\n",
                    "\n",
                    "from core.data_collector import UnifiedDataCollector\n",
                    "from core.signal_generator import MultiTimeframeSignalGenerator\n",
                    "from risk_management.position_sizer import AdvancedPositionSizer\n",
                    "\n",
                    "# Config yükle\n",
                    "import json\n",
                    "with open('../configs/config.json', 'r') as f:\n",
                    "    config = json.load(f)\n",
                    "\n",
                    "print('Sistem hazır!')"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Veri toplama örneği\n",
                    "collector = UnifiedDataCollector(config)\n",
                    "data = await collector.collect_multi_timeframe_data('THYAO')\n",
                    "\n",
                    "print(f'Toplanan timeframe\\'ler: {list(data.keys())}')"
                ]
            }
        ]
    }
    
    notebook_path = Path('notebooks/example_usage.ipynb')
    with open(notebook_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    logger.info("Örnek notebook oluşturuldu: notebooks/example_usage.ipynb")


def main():
    """Ana kurulum fonksiyonu"""
    logger.info("Hibrit Trading Sistemi kurulumu başlıyor...")
    
    # 1. Dizinleri oluştur
    setup_directories()
    
    # 2. Config dosyasını oluştur
    setup_config()
    
    # 3. Bağımlılıkları kontrol et
    if not check_dependencies():
        logger.error("Kurulum tamamlanamadı! Bağımlılıkları yükleyin.")
        sys.exit(1)
    
    # 4. Redis kontrolü
    setup_redis()
    
    # 5. Örnek notebook
    create_example_notebook()
    
    logger.success("\n✅ Kurulum tamamlandı!")
    logger.info("\nSonraki adımlar:")
    logger.info("1. configs/config.json dosyasına API anahtarlarını ekleyin")
    logger.info("2. Redis servisini başlatın: redis-server")
    logger.info("3. Paper trading ile test edin: python main.py --mode paper")
    logger.info("4. Dokümantasyon için README.md dosyasını inceleyin")
    
    print("\n" + "="*50)
    print("Hibrit Trading Sistemi kurulumu tamamlandı!")
    print("Aylık %8-9 getiri hedefi ile başarılar!")
    print("="*50)


if __name__ == "__main__":
    main()