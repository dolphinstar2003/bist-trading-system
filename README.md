# BIST Trading System

BIST hisse senetleri için gelişmiş ML/DL tabanlı alım-satım sistemi.

## 🚀 Özellikler

- **Hibrit Yaklaşım**: Teknik analiz + Makine öğrenmesi + Derin öğrenme
- **Multi-timeframe Analiz**: 1h, 4h, 1d zaman dilimlerinde analiz
- **GRU Model**: Attention mekanizmalı LSTM/GRU sinir ağları
- **122 BIST Hissesi**: Tüm major BIST hisseleri için destek
- **Risk Yönetimi**: %1 risk kuralı, Kelly Criterion, trailing stop
- **Gerçek Zamanlı Trading**: AlgoLab entegrasyonu

## 📊 Performans

- Model Doğruluğu: %81.9 (train) / %81.3 (validation)
- Hedef: Aylık %8-9 getiri
- Maksimum Drawdown: %8
- Risk/Reward: Minimum 2:1

## 🛠️ Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/dolphinstar2003/bist-trading-system.git
cd bist-trading-system

# Virtual environment oluşturun
python -m venv trading
source trading/bin/activate  # Linux/Mac
# veya
trading\Scripts\activate  # Windows

# Gereksinimleri yükleyin
pip install -r requirements.txt
```

## 📁 Proje Yapısı

```
bist-trading-system/
├── New_Plan/           # Ana trading sistemi
│   ├── core/          # Çekirdek modüller
│   ├── models/        # ML/DL modeller
│   ├── indicators/    # Teknik indikatörler
│   ├── backtest/      # Backtest modülleri
│   └── data/          # Veri dizini
├── ml_models/         # Diğer ML modeller
├── dl_models/         # Deep Learning modeller
└── strategies/        # Trading stratejileri
```

## 🚦 Kullanım

### 1. Veri İndirme
```bash
python download_data_yahoo_proper.py
```

### 2. İndikatör Hesaplama
```bash
cd New_Plan
python calculate_indicators.py
```

### 3. Model Eğitimi
```bash
python train_model.py --start 2023-01-01 --end 2025-06-30
```

### 4. Backtest
```bash
python backtest/advanced_backtest.py --start 2025-01-01 --end 2025-06-30
```

### 5. Canlı Trading
```bash
python main.py --mode paper  # Kağıt trading
python main.py --mode live   # Gerçek trading
```

## 📈 Stratejiler

1. **MACD Multi-timeframe**: Ana strateji
2. **ML Ensemble**: Random Forest + XGBoost + LightGBM
3. **Deep Learning**: LSTM/GRU + CNN + Transformer
4. **Hybrid System**: Tüm modellerin kombinasyonu

## ⚠️ Risk Uyarısı

Bu sistem eğitim ve araştırma amaçlıdır. Gerçek parayla işlem yapmadan önce:
- Paper trading ile test edin
- Risk yönetimi kurallarına uyun
- Sadece kaybetmeyi göze alabileceğiniz parayla işlem yapın

## 📝 Lisans

MIT License

## 👤 İletişim

GitHub: [@dolphinstar2003](https://github.com/dolphinstar2003)