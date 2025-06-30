# Hibrit Trading Sistemi - Proje Özeti

## 🎯 Hedefler
- **Aylık Getiri**: %8-9
- **Maksimum Aylık Drawdown**: %8
- **Başlangıç Sermayesi**: 50,000 TL
- **Platform**: Algolab
- **Piyasa**: BIST + Opsiyonel Kripto

## 🏗️ Sistem Mimarisi

### 1. Veri Toplama Katmanı
- **Multi-timeframe**: 15m, 1h, 4h, 1d, 1w
- **Kaynaklar**: Yahoo Finance, Finnhub, Alpha Vantage
- **Cache**: Redis (15 dakika TTL)
- **Makro Veriler**: USD/TRY, Altın, VIX, XU100

### 2. ML/DL Model Katmanı
- **Ana Model**: Multi-timeframe GRU (CPU optimized)
  - 5 ayrı GRU (her timeframe için)
  - Attention mekanizması
  - 50 hidden units, 1 layer
- **Yardımcı Model**: XGBoost Ensemble
- **Özellikler**: 30+ teknik indikatör + sentiment

### 3. Strateji Katmanı
- **MACD Stratejisi**: Literatürde en güvenli bulunmuş
- **Multi-TF Confirmation**: Üst timeframe onayı
- **Confidence Threshold**: >%65

### 4. Risk Yönetimi
- **Position Sizing**: %1 risk kuralı (dinamik)
- **Stop Loss**: 2x ATR
- **Trailing Stop**: 3x ATR
- **Kelly Criterion**: %25 fraction
- **Aylık Drawdown Limiti**: %8'de sistem durur

### 5. Execution Katmanı
- **Algolab WebSocket**: Gerçek zamanlı bağlantı
- **Order Types**: Limit orders (%0.2 spread kazancı)
- **Partial Close**: %3 karda %50 pozisyon kapatma

## 📁 Klasör Yapısı
```
New_Plan/
├── core/               # Çekirdek modüller
├── models/             # ML/DL modeller
├── strategies/         # Trading stratejileri
├── risk_management/    # Risk yönetimi
├── indicators/         # Teknik indikatörler
├── execution/          # Algolab entegrasyonu
├── monitoring/         # İzleme ve raporlama
├── configs/           # Konfigürasyon dosyaları
├── tests/             # Test modülleri
├── docs/              # Dokümantasyon
└── notebooks/         # Jupyter notebooks
```

## 🚀 Başlangıç

1. **Kurulum**:
```bash
cd New_Plan
pip install -r requirements.txt
python setup.py
```

2. **Konfigürasyon**:
```bash
cp configs/config_template.json configs/config.json
# API anahtarlarını ekle
```

3. **Paper Trading**:
```bash
python main.py --mode paper
```

## 📊 Performans Metrikleri
- **Target Monthly Return**: %8
- **Expected Win Rate**: >%50
- **Risk/Reward Ratio**: >1.5:1
- **Max Positions**: 5
- **Trade Frequency**: 15-25/ay

## 🔑 Kritik Başarı Faktörleri

1. **Disiplin**: %1 risk kuralına mutlak uyum
2. **Multi-TF Onay**: Yanlış sinyalleri azaltır
3. **MACD Güvenliği**: En stabil indikatör
4. **Drawdown Kontrolü**: %8'de dur
5. **Adaptasyon**: Piyasa rejimine göre ayarlama

## 📈 Gelişim Yol Haritası

### Faz 1 (2 Hafta): Temel Sistem
- ✅ Veri toplama altyapısı
- ✅ GRU model implementasyonu
- ✅ Risk yönetimi modülleri
- ✅ Algolab entegrasyonu

### Faz 2 (2 Hafta): Optimizasyon
- [ ] Backtest framework
- [ ] Hyperparameter tuning
- [ ] Walk-forward optimization
- [ ] Performance analytics

### Faz 3 (1 Ay): Test
- [ ] Paper trading
- [ ] Model fine-tuning
- [ ] Risk parametre ayarı
- [ ] Canlı test (küçük sermaye)

### Faz 4: Production
- [ ] Full deployment
- [ ] 7/24 monitoring
- [ ] Otomatik raporlama
- [ ] Sürekli iyileştirme

## ⚠️ Risk Uyarıları
- Finansal piyasalarda işlem yapmak risk içerir
- Geçmiş performans gelecek garantisi değildir
- Sadece kaybetmeyi göze alabileceğiniz parayla işlem yapın
- Sistem tamamen test edilmeden canlı kullanmayın

## 📞 İletişim
- Sistem logları: `logs/` klasörü
- Performans raporları: `reports/` klasörü
- Hata durumunda: `logs/errors.log`

---
*Bu sistem, akademik araştırmalar ve en iyi pratikler temel alınarak geliştirilmiştir.*