#!/usr/bin/env python3
"""
Alpha Vantage Test Script
"""

import sys
from pathlib import Path

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.download_data_alpha_vantage import AlphaVantageDownloader

print("Alpha Vantage Test Başlıyor...\n")

# Downloader oluştur
downloader = AlphaVantageDownloader()

# Mevcut verileri kontrol et
print("=== MEVCUT VERİLER KONTROL EDİLİYOR ===")
existing = downloader.check_existing_data()

# Sadece THYAO'yu test edelim
print("\n=== TEST: THYAO İNDİRİLİYOR ===")
success = downloader.download_symbol('THYAO')

if success:
    print("\n✅ Test başarılı! Alpha Vantage çalışıyor.")
    print("\nTüm sembolleri indirmek için:")
    print("python download_data_alpha_vantage.py")
    print("ve menüden 1 veya 2'yi seçin.")
else:
    print("\n❌ Test başarısız! API key veya bağlantı kontrol edin.")