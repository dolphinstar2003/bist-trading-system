README.md
markdown# 🚀 BIST Algoritmik Trading Sistemi

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AlgoLab](https://img.shields.io/badge/AlgoLab-API-orange.svg)](https://algolab.com.tr)
[![Status](https://img.shields.io/badge/Status-Development-yellow.svg)]()

Profesyonel algoritmik trading sistemi - BIST için makine öğrenmesi destekli, çoklu zaman dilimi analizi yapan otomatik trading botu.

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Sistem Gereksinimleri](#sistem-gereksinimleri)
- [Kurulum](#kurulum)
- [Yapılandırma](#yapılandırma)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Trading Stratejileri](#trading-stratejileri)
- [Performans Metrikleri](#performans-metrikleri)
- [Risk Yönetimi](#risk-yönetimi)
- [Katkıda Bulunma](#katkıda-bulunma)

## 🎯 Özellikler

### Veri Yönetimi
- ✅ **Multi-Timeframe Analiz**: 15m, 1h, 4h, 1d
- ✅ **50 Hisse Desteği**: BIST30 + En çok işlem gören 20 hisse
- ✅ **Gerçek Zamanlı Veri**: AlgoLab WebSocket entegrasyonu
- ✅ **Otomatik Veri Güncelleme**: Periyodik veri senkronizasyonu

### Teknik Analiz
- 📊 **8 Gelişmiş İndikatör**: ADX/DI, Williams Vix Fix, Lorentzian ML, MACD MTF, Squeeze Momentum, Supertrend, Trend Vanguard, WaveTrend
- 📈 **Price Action Patterns**: Engulfing, Hammer, Doji tanıma
- 📉 **Dinamik Destek/Direnç**: Otomatik seviye tespiti

### Makine Öğrenmesi
- 🤖 **4 ML Modeli**: XGBoost, Random Forest, LSTM, LightGBM
- 🎯 **Hisseye Özel Eğitim**: Her hisse için optimize edilmiş modeller
- 🔄 **Ensemble Learning**: Model birleştirme ile güçlü sinyaller
- 📊 **Feature Engineering**: 50+ özellik ile detaylı analiz

### Piyasa Analizi
- 💹 **Makro Göstergeler**: USD/TRY, EUR/TRY, XAU/USD takibi
- 📰 **KAP Entegrasyonu**: Önemli haberlerin otomatik takibi
- 🔍 **Sentiment Analizi**: Türkçe NLP ile haber yorumlama
- 🔗 **Korelasyon Matrisi**: Hisseler arası ilişki analizi

### Risk Yönetimi
- ⚠️ **ATR-Based Stop Loss**: Volatiliteye göre dinamik stop
- 💰 **Position Sizing**: Kelly Criterion ile optimal pozisyon
- 🛡️ **Maximum Drawdown Koruması**: %15 limit
- 🚨 **Emergency Stop**: Anormal durumlarda otomatik çıkış

## 💻 Sistem Gereksinimleri

- Python 3.9+
- RAM: Minimum 8GB (16GB önerilir)
- İşlemci: 4+ çekirdek
- Disk: 50GB+ boş alan
- İnternet: Stabil bağlantı (min 10Mbps)
- OS: Windows/Linux/MacOS

## 🔧 Kurulum

### 1. Repoyu Klonlayın
```bash
git clone https://github.com/yourusername/bist-algo-trading.git
cd bist-algo-trading
2. Virtual Environment Oluşturun
bashpython -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
3. Bağımlılıkları Yükleyin
bashpip install -r requirements.txt
4. AlgoLab API Kurulumu
bash# AlgoLab klasörünü kopyalayın (DOKUNMAYIN)
cp -r /path/to/algolab ./algolab
⚙️ Yapılandırma
1. API Anahtarlarını Ayarlayın
bashcp config/.env.example config/.env
.env dosyasını düzenleyin:
env# AlgoLab API
ALGOLAB_API_KEY=your_api_key
ALGOLAB_USERNAME=your_tc_kimlik
ALGOLAB_PASSWORD=your_password

# Database
DB_CONNECTION=sqlite:///trading.db

# Risk Management
MAX_POSITION_SIZE=0.05  # %5
MAX_DAILY_LOSS=0.02     # %2
STOP_LOSS_MULTIPLIER=1.5

# Trading
CONFIDENCE_THRESHOLD=0.7
MIN_VOLUME=1000000
2. Hisse Listesini Yapılandırın
python# config/symbols.json
{
  "symbols": [
    "THYAO", "GARAN", "SISE", "TUPRS", "EREGL",
    // ... diğer hisseler
  ],
  "excluded": ["HALKB"],  // İşlem yapmak istemediğiniz hisseler
  "favorites": ["THYAO", "GARAN"]  // Öncelikli hisseler
}
🚀 Kullanım
Hızlı Başlangıç
bash# Sistemi başlat
python main.py --mode live

# Paper trading
python main.py --mode paper

# Backtest
python main.py --mode backtest --start 2024-01-01 --end 2024-12-31
Veri Güncelleme
bash# Tüm veriyi güncelle
python data_manager.py --update all

# Belirli hisse güncelle
python data_manager.py --update THYAO

# İndikatörleri hesapla
python indicator_calculator.py --symbols all
Model Eğitimi
bash# Tüm modelleri eğit
python ml_trainer.py --train all

# Belirli hisse için eğit
python ml_trainer.py --train THYAO --model xgboost
📁 Proje Yapısı
bist-algo-trading/
├── algolab/                 # AlgoLab API (DOKUNMAYIN)
│   ├── algolab.py
│   ├── algolab_socket.py
│   └── config.py
├── calisan/                 # Başarılı sürümler
├── config/                  # Yapılandırma dosyaları
│   ├── .env
│   ├── symbols.json
│   └── strategies.yaml
├── data/                    # Veri depolama
│   ├── raw/                # Ham veriler
│   ├── processed/          # İşlenmiş veriler
│   └── indicators/         # Hesaplanmış indikatörler
├── indicators/             # Teknik indikatörler
│   ├── adx_di.py
│   ├── williams_vix_fix.py
│   └── ...
├── ml_models/              # ML modelleri
│   ├── trainers/
│   ├── predictors/
│   └── trained/
├── strategies/             # Trading stratejileri
│   ├── base_strategy.py
│   ├── momentum.py
│   └── mean_reversion.py
├── backtest/               # Backtest modülü
├── paper_trade/            # Paper trading
├── live_trade/             # Canlı trading
├── market_analysis/        # Piyasa analizi
├── risk/                   # Risk yönetimi
├── utils/                  # Yardımcı fonksiyonlar
├── logs/                   # Log dosyaları
├── tests/                  # Test dosyaları
├── main.py                 # Ana program
├── requirements.txt        # Bağımlılıklar
└── README.md              # Bu dosya
📊 Trading Stratejileri
1. Momentum Strategy

RSI + MACD kombinasyonu
Volume konfirmasyonu
Trend takibi

2. Mean Reversion

Bollinger Bands
Z-Score analizi
Pairs trading

3. ML Ensemble

4 model kombinasyonu
Confidence scoring
Dynamic weighting

4. Market Making

Bid-ask spread analizi
Likidite sağlama
Mikro arbitraj

📈 Performans Metrikleri
MetrikHedefMevcutSharpe Ratio> 1.5-Max Drawdown< 15%-Win Rate> 60%-Profit Factor> 1.8-Daily Trades5-10-Risk/Reward> 1:2-
⚠️ Risk Yönetimi
Position Sizing
pythonposition_size = account_balance * 0.05  # Max %5
adjusted_size = position_size * confidence_score
final_size = min(adjusted_size, max_position_limit)
Stop Loss Stratejisi

ATR-based dynamic stops
Time-based stops
Trailing stops
Circuit breaker

Risk Limitleri

Günlük maksimum kayıp: %2
Haftalık maksimum kayıp: %5
Maksimum açık pozisyon: 5
Korelasyon limiti: 0.7

🤝 Katkıda Bulunma

Fork yapın
Feature branch oluşturun (git checkout -b feature/amazing-feature)
Değişikliklerinizi commit edin (git commit -m 'Add amazing feature')
Branch'e push yapın (git push origin feature/amazing-feature)
Pull Request açın

Kod Standartları

PEP 8 uyumlu
Type hints kullanın
Docstring zorunlu
Unit test yazın

📝 Lisans
Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasına bakın.
⚡ Sorumluluk Reddi
Bu yazılım yalnızca eğitim amaçlıdır. Finansal kayıplardan yazılım geliştiricileri sorumlu değildir. Yatırım tavsiyesi değildir.
📞 İletişim

GitHub: @yourusername
Email: your.email@example.com


💡 Not: AlgoLab klasörü içindeki dosyalara kesinlikle dokunmayın!