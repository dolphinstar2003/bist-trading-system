# Hybrid Trading System - Detaylı Kullanım Kılavuzu

## 🎯 Sistem Özeti

Bu sistem, "Hibrit Alım-Satım Algoritması Araştırma Raporu"nda önerilen stratejileri implemente eder:
- Multi-timeframe GRU neural network
- MACD odaklı trading stratejisi
- %1 risk yönetimi kuralı
- Aylık %8-9 getiri hedefi

## 🚀 Hızlı Başlangıç

### 1. Sistem Testi
```bash
cd /home/yunus/Belgeler/New_Start/New_Plan
python test_system.py
```

### 2. İndikatör Hesaplama
```bash
# Tüm semboller için indikatörleri hesapla
python -c "
from indicators.indicator_calculator import IndicatorCalculator
calc = IndicatorCalculator()
symbols = calc.csv_manager.get_available_symbols()
timeframes = ['15m', '1h', '4h', '1d']
calc.process_all_symbols(symbols[:10], timeframes)  # İlk 10 sembol
"
```

### 3. Konfigurasyon
```bash
# config.json dosyasını düzenleyin
# API anahtarları ekleyin (opsiyonel)
# Algolab bilgilerini ekleyin
```

## 📊 Veri Yapısı

### Mevcut Veri Kullanımı
Sistem, `/home/yunus/Belgeler/New_Start/data/raw` klasöründeki CSV dosyalarını kullanır:
```
data/raw/
├── AKBNK_1d_raw.csv
├── AKBNK_1h_raw.csv
├── AKBNK_4h_raw.csv
└── ...
```

### Yeni İndikatörler
Hesaplanan indikatörler `New_Plan/data/indicators/` altına kaydedilir:
```
New_Plan/data/indicators/
├── AKBNK_1h_macd.csv
├── AKBNK_1h_rsi.csv
├── AKBNK_1h_atr.csv
└── ...
```

## 🤖 Model Kullanımı

### Model Eğitimi (TODO)
```python
from models.train_gru import train_model

# Model eğit
model = train_model(
    symbols=['AKBNK', 'GARAN', 'THYAO'],
    start_date='2022-01-01',
    end_date='2023-12-31',
    epochs=100
)
```

### Sinyal Üretimi
```python
from core.signal_generator import SignalGenerator
import asyncio

async def generate_signals():
    config = json.load(open('config.json'))
    generator = SignalGenerator(config)
    
    symbols = ['AKBNK', 'THYAO', 'EREGL']
    signals = await generator.generate_signals(symbols)
    
    for signal in signals:
        print(generator.get_signal_summary(signal))

asyncio.run(generate_signals())
```

## 💼 Portfolio Yönetimi

### Portfolio Durumu
```python
from core.portfolio_manager import PortfolioManager

config = json.load(open('config.json'))
pm = PortfolioManager(config)

# Durum kontrolü
status = pm.get_portfolio_status()
print(f"Capital: {status['capital']:,.0f} TRY")
print(f"Open Positions: {status['open_positions']}")
print(f"Total PnL: {status['total_pnl']:,.2f} TRY")
print(f"Win Rate: {status['win_rate']:.1f}%")

# Risk kontrolü
risk = pm.risk_check()
print(f"Portfolio Risk: {risk['portfolio_risk']:.1f}%")
print(f"Can Open New Positions: {risk['can_open_positions']}")
```

## 📈 Strateji Detayları

### MACD Stratejisi
1. **Multi-Timeframe Confirmation**:
   - 1h: Ana sinyal
   - 4h: Trend doğrulama
   - 1d: Genel yön

2. **Entry Conditions**:
   - MACD bullish cross
   - ML model confidence > 65%
   - Volume ratio > 0.8
   - ATR volatility normal range

3. **Exit Conditions**:
   - 2x ATR stop loss
   - Target 1: 2:1 R/R (%50 pozisyon)
   - Target 2: 3:1 R/R (kalan %50)
   - Trailing stop: 2.5x ATR

### Risk Parametreleri
```json
{
  "max_risk_per_trade": 0.01,      // %1
  "stop_loss_atr_multiplier": 2.0,  // 2x ATR
  "kelly_fraction": 0.25,           // Kelly %25
  "max_monthly_drawdown": 0.08      // %8
}
```

## 🔧 Özelleştirme

### Yeni İndikatör Ekleme
```python
# indicators/indicator_calculator.py içine ekleyin
def calculate_custom_indicator(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
    results = {}
    
    # Özel indikatör hesaplama
    results['my_indicator'] = talib.SMA(df['close'], timeperiod=20)
    
    return results
```

### Feature Ekleme
```python
# core/feature_engineering.py içinde
def _create_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    
    # Özel feature
    features['price_range'] = (df['high'] - df['low']) / df['close']
    
    return features
```

## 📊 Backtest Çalıştırma

```python
# TODO: Backtest modülü eklenecek
from backtest.engine import BacktestEngine

engine = BacktestEngine(config)
results = engine.run(
    symbols=['AKBNK', 'GARAN', 'THYAO'],
    start_date='2023-01-01',
    end_date='2023-12-31'
)

print(f"Total Return: {results['total_return']:.1f}%")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.1f}%")
```

## 🚨 Önemli Uyarılar

1. **Paper Trading**: Her zaman önce paper trading ile test edin
2. **Küçük Başlayın**: Live trading'de min sermaye ile başlayın
3. **Veri Kalitesi**: CSV verilerinin güncel olduğundan emin olun
4. **Model Güncelleme**: Modeli düzenli olarak yeniden eğitin
5. **Risk Limitleri**: Drawdown limitlerini kesinlikle aşmayın

## 🐛 Yaygın Sorunlar

### 1. "Module not found" Hatası
```bash
# PYTHONPATH ekleyin
export PYTHONPATH=/home/yunus/Belgeler/New_Start/New_Plan:$PYTHONPATH
```

### 2. Redis Bağlantı Hatası
```bash
# Redis kurulumu (opsiyonel)
sudo apt-get install redis-server
sudo service redis-server start
```

### 3. Yetersiz Veri
```bash
# Eksik verileri kontrol et
python -c "
from core.csv_data_manager import CSVDataManager
cm = CSVDataManager()
for symbol in cm.get_available_symbols()[:5]:
    print(f'{symbol}: {cm.get_available_timeframes(symbol)}')
"
```

## 📞 Destek

Sorularınız için:
1. Dokümantasyonu kontrol edin
2. test_system.py çıktısını inceleyin
3. Log dosyalarını kontrol edin (logs/ dizini)