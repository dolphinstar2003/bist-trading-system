#!/usr/bin/env python3
"""
Yahoo Finance Test
"""

import sys
from pathlib import Path

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yfinance as yf
import pandas as pd

print("Yahoo Finance Test\n" + "="*30)

# Test sembolleri
test_symbols = [
    ("THYAO", "THYAO.IS"),  # Turkish Airlines
    ("GARAN", "GARAN.IS"),  # Garanti Bank
    ("AKBNK", "AKBNK.IS"),  # Akbank
]

for local_symbol, yahoo_symbol in test_symbols:
    print(f"\nTest: {local_symbol} -> {yahoo_symbol}")
    
    try:
        # Ticker oluştur
        ticker = yf.Ticker(yahoo_symbol)
        
        # Son 1 yıllık veri
        df = ticker.history(period="1y")
        
        if not df.empty:
            print(f"✅ Başarılı! {len(df)} gün veri")
            print(f"   Tarih aralığı: {df.index[0].date()} - {df.index[-1].date()}")
            print(f"   Son kapanış: {df['Close'].iloc[-1]:.2f}")
        else:
            print(f"❌ Veri bulunamadı")
            
    except Exception as e:
        print(f"❌ Hata: {e}")

# Toplu indirme testi
print("\n\nToplu İndirme Testi\n" + "="*30)

symbols = ["THYAO.IS", "GARAN.IS", "AKBNK.IS", "EREGL.IS", "ASELS.IS"]
print(f"Semboller: {', '.join(symbols)}")

try:
    data = yf.download(symbols, period="1mo", group_by="ticker")
    
    if not data.empty:
        print(f"\n✅ Toplu indirme başarılı!")
        print(f"Veri boyutu: {data.shape}")
        print(f"Tarih aralığı: {data.index[0].date()} - {data.index[-1].date()}")
except Exception as e:
    print(f"\n❌ Toplu indirme hatası: {e}")

print("\n\nSonuç: Yahoo Finance Türkiye hisselerini .IS uzantısıyla destekliyor!")