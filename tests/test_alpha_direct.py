#!/usr/bin/env python3
"""
Alpha Vantage Direct API Test
"""

import sys
from pathlib import Path

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
import json

api_key = "3D0PCT614PAOUXAL"

# Test 1: API key geçerli mi?
print("Test 1: API Key Kontrolu")
test_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey={api_key}"
response = requests.get(test_url)
print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    if "Error Message" in data:
        print(f"API Hatası: {data['Error Message']}")
    elif "Note" in data:
        print(f"API Notu: {data['Note']}")
    elif "Time Series (Daily)" in data:
        print("✅ API Key geçerli!")
    else:
        print("Bilinmeyen yanıt:", list(data.keys())[:3])

# Test 2: Türkiye hissesi deneyelim
print("\nTest 2: Türkiye Hissesi (THYAO.IS)")
turkish_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=THYAO.IS&outputsize=compact&apikey={api_key}"
response = requests.get(turkish_url)
print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    if "Error Message" in data:
        print(f"Hata: {data['Error Message']}")
    elif "Time Series (Daily)" in data:
        print("✅ THYAO.IS bulundu!")
        dates = list(data["Time Series (Daily)"].keys())[:5]
        print(f"Son 5 gün: {dates}")
    else:
        print("Yanıt:", json.dumps(data, indent=2)[:500])

# Test 3: Farklı sembol formatları
print("\nTest 3: Farklı Formatlar")
test_symbols = ["THYAO.IS", "THYAO.IST", "THYAO", "THYAO.E"]

for symbol in test_symbols:
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "Global Quote" in data and data["Global Quote"]:
            print(f"✅ {symbol}: Bulundu!")
        else:
            print(f"❌ {symbol}: Bulunamadı")