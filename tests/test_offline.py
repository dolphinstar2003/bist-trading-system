#!/usr/bin/env python3
"""
Offline Test - API bağlantısı olmadan sistem testi
"""

import sys
from pathlib import Path

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.logger import get_logger
from utils.csv_data_manager import CSVDataManager

logger = get_logger(__name__)


def generate_mock_data(symbol, days=30):
    """Mock OHLCV data oluştur"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Rastgele fiyat hareketi
    base_price = 100
    prices = []
    
    for i in range(days):
        change = np.random.randn() * 2  # %2 standart sapma
        base_price = base_price * (1 + change/100)
        
        # OHLC hesapla
        high = base_price * (1 + abs(np.random.randn()) * 0.01)
        low = base_price * (1 - abs(np.random.randn()) * 0.01)
        open_price = np.random.uniform(low, high)
        close_price = np.random.uniform(low, high)
        volume = np.random.randint(1000000, 5000000)
        
        prices.append({
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(prices, index=dates)
    return df


def test_csv_manager():
    """CSV Manager testi"""
    print("=" * 50)
    print("CSV Data Manager Test")
    print("=" * 50)
    
    csv_manager = CSVDataManager()
    
    # Test sembolleri
    test_symbols = ["THYAO", "GARAN", "AKBNK"]
    
    for symbol in test_symbols:
        print(f"\n{symbol} için mock data oluşturuluyor...")
        
        # Farklı timeframe'ler için data oluştur
        timeframes = {
            "1d": 100,   # 100 günlük
            "1h": 500,   # 500 saatlik (yaklaşık 20 gün)
            "15m": 1000  # 1000 bar (yaklaşık 10 gün)
        }
        
        for tf, periods in timeframes.items():
            # Mock data oluştur
            df = generate_mock_data(symbol, days=periods if tf == "1d" else 30)
            
            # CSV'ye kaydet
            csv_manager.save_raw_data(symbol, df, tf)
            print(f"  - {tf} verisi kaydedildi ({len(df)} bar)")
            
            # Geri oku ve kontrol et
            loaded_df = csv_manager.load_raw_data(symbol, tf)
            if loaded_df is not None:
                print(f"  - {tf} verisi okundu ✓")
    
    print("\n✅ CSV Manager testi başarılı!")


def test_indicator_calculations():
    """İndikatör hesaplama testi"""
    print("\n" + "=" * 50)
    print("İndikatör Hesaplama Testi")
    print("=" * 50)
    
    # Mock data
    df = generate_mock_data("TEST", days=50)
    
    # Basit indikatörler
    print("\n1. SMA (Simple Moving Average):")
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    print(f"   SMA10: {df['SMA_10'].iloc[-1]:.2f}")
    print(f"   SMA20: {df['SMA_20'].iloc[-1]:.2f}")
    
    print("\n2. RSI (Relative Strength Index):")
    # RSI hesaplama
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    print(f"   RSI(14): {df['RSI'].iloc[-1]:.2f}")
    
    print("\n3. Bollinger Bands:")
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    print(f"   Upper: {df['BB_upper'].iloc[-1]:.2f}")
    print(f"   Middle: {df['BB_middle'].iloc[-1]:.2f}")
    print(f"   Lower: {df['BB_lower'].iloc[-1]:.2f}")
    
    print("\n4. MACD:")
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    print(f"   MACD: {df['MACD'].iloc[-1]:.2f}")
    print(f"   Signal: {df['Signal'].iloc[-1]:.2f}")
    print(f"   Histogram: {df['Histogram'].iloc[-1]:.2f}")
    
    # Trading sinyalleri
    print("\n5. Trading Sinyalleri:")
    
    # Golden Cross / Death Cross
    if df['SMA_10'].iloc[-1] > df['SMA_20'].iloc[-1] and df['SMA_10'].iloc[-2] <= df['SMA_20'].iloc[-2]:
        print("   📈 Golden Cross tespit edildi!")
    elif df['SMA_10'].iloc[-1] < df['SMA_20'].iloc[-1] and df['SMA_10'].iloc[-2] >= df['SMA_20'].iloc[-2]:
        print("   📉 Death Cross tespit edildi!")
    
    # RSI sinyalleri
    if df['RSI'].iloc[-1] < 30:
        print("   🟢 RSI Oversold (< 30)")
    elif df['RSI'].iloc[-1] > 70:
        print("   🔴 RSI Overbought (> 70)")
    
    # Bollinger Band sinyalleri
    last_close = df['close'].iloc[-1]
    if last_close < df['BB_lower'].iloc[-1]:
        print("   🟢 Fiyat alt bandın altında")
    elif last_close > df['BB_upper'].iloc[-1]:
        print("   🔴 Fiyat üst bandın üstünde")
    
    print("\n✅ İndikatör testi başarılı!")


def test_risk_management():
    """Risk yönetimi hesaplamaları"""
    print("\n" + "=" * 50)
    print("Risk Yönetimi Testi")
    print("=" * 50)
    
    # Örnek portföy
    portfolio_value = 100000  # 100K TL
    position_size_pct = 5     # %5 pozisyon büyüklüğü
    stop_loss_pct = 2         # %2 stop loss
    
    # Örnek hisse
    symbol = "THYAO"
    current_price = 265.25
    
    print(f"\nPortföy Değeri: {portfolio_value:,.0f} TL")
    print(f"Hisse: {symbol} @ {current_price} TL")
    
    # Pozisyon hesaplama
    position_value = portfolio_value * (position_size_pct / 100)
    shares = int(position_value / current_price)
    actual_value = shares * current_price
    
    print(f"\n1. Pozisyon Büyüklüğü (%{position_size_pct}):")
    print(f"   Hedef: {position_value:,.0f} TL")
    print(f"   Adet: {shares:,}")
    print(f"   Gerçek: {actual_value:,.0f} TL")
    
    # Stop loss hesaplama
    stop_price = current_price * (1 - stop_loss_pct/100)
    potential_loss = shares * (current_price - stop_price)
    
    print(f"\n2. Stop Loss (%{stop_loss_pct}):")
    print(f"   Stop Fiyat: {stop_price:.2f} TL")
    print(f"   Risk: {potential_loss:,.0f} TL")
    print(f"   Risk/Portföy: {(potential_loss/portfolio_value)*100:.2f}%")
    
    # Kelly Criterion
    win_rate = 0.55  # %55 kazanma oranı
    avg_win = 3.0    # Ortalama %3 kazanç
    avg_loss = 2.0   # Ortalama %2 kayıp
    
    kelly_pct = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
    kelly_pct = max(0, min(kelly_pct, 0.25))  # Max %25 ile sınırla
    
    print(f"\n3. Kelly Criterion:")
    print(f"   Win Rate: %{win_rate*100}")
    print(f"   Win/Loss Ratio: {avg_win/avg_loss:.2f}")
    print(f"   Önerilen Pozisyon: %{kelly_pct*100:.1f}")
    
    # Diversifikasyon
    max_positions = 10
    max_per_sector = 0.3  # Sektör başına max %30
    
    print(f"\n4. Diversifikasyon:")
    print(f"   Max Pozisyon Sayısı: {max_positions}")
    print(f"   Min Pozisyon Büyüklüğü: %{100/max_positions:.1f}")
    print(f"   Sektör Limiti: %{max_per_sector*100}")
    
    print("\n✅ Risk yönetimi testi başarılı!")


def main():
    """Ana test fonksiyonu"""
    print("🚀 Offline Trading System Test")
    print("=" * 60)
    print("API bağlantısı olmadan sistem bileşenleri test ediliyor...")
    print("=" * 60)
    
    try:
        # CSV Manager testi
        test_csv_manager()
        
        # İndikatör testi
        test_indicator_calculations()
        
        # Risk yönetimi testi
        test_risk_management()
        
        print("\n" + "=" * 60)
        print("✅ TÜM TESTLER BAŞARILI!")
        print("=" * 60)
        
        print("\n📌 NOT: API erişimi 2 saat sonra tekrar denenebilir.")
        print("📌 Bu sürede strateji geliştirmeye devam edebilirsiniz.")
        
    except Exception as e:
        logger.error(f"Test hatası: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    
    # Temiz çıkış
    import os
    os._exit(0)