# Hibrit Alım-Satım Sistemi (Algolab Entegrasyonlu)

## Hedefler
- Aylık %8-9 getiri
- Maksimum aylık %8 çekilme (drawdown)
- 50,000 TL başlangıç sermayesi
- CPU üzerinde çalışan ML/DL modeller
- Multi-timeframe analiz (15m, 1h, 4h, 1d, 1w)

## Sistem Mimarisi

### 1. Core (Çekirdek Modüller)
- `data_collector.py`: Veri toplama ve yönetimi
- `feature_engineering.py`: Özellik mühendisliği
- `signal_generator.py`: Sinyal üretimi
- `portfolio_manager.py`: Portföy yönetimi

### 2. Models (ML/DL Modeller)
- `gru_multi_timeframe.py`: Ana GRU modeli
- `xgboost_ensemble.py`: Yardımcı XGBoost modeli
- `model_trainer.py`: Model eğitim pipeline
- `model_evaluator.py`: Model değerlendirme

### 3. Strategies (Strateji Modülleri)
- `macd_strategy.py`: MACD tabanlı strateji
- `multi_tf_confirmation.py`: Multi-timeframe onaylama
- `hybrid_strategy.py`: ML + Teknik analiz birleşimi

### 4. Risk Management
- `position_sizer.py`: Pozisyon boyutlandırma (%1 risk kuralı)
- `stop_loss_manager.py`: Dinamik stop-loss yönetimi
- `portfolio_risk_monitor.py`: Portföy risk takibi

### 5. Indicators (Teknik İndikatörler)
- `trend_indicators.py`: MACD, EMA, ADX
- `momentum_indicators.py`: RSI, DMI, Stochastic
- `volatility_indicators.py`: ATR, Bollinger Bands
- `volume_indicators.py`: OBV, Volume Ratio

### 6. Execution (İşlem Yürütme)
- `algolab_connector.py`: Algolab API bağlantısı
- `order_manager.py`: Emir yönetimi
- `trade_executor.py`: İşlem yürütücü

### 7. Monitoring (İzleme)
- `performance_tracker.py`: Performans takibi
- `risk_monitor.py`: Risk izleme
- `alert_system.py`: Uyarı sistemi

## Kurulum

```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Konfigürasyonu ayarla
cp configs/config_template.json configs/config.json
# config.json dosyasını düzenle

# Sistemi başlat
python main.py --mode paper  # Paper trading
python main.py --mode live   # Canlı trading
```

## Kullanım

### 1. Veri Toplama
```python
from core.data_collector import UnifiedDataCollector
collector = UnifiedDataCollector()
data = collector.collect_multi_timeframe_data('THYAO')
```

### 2. Sinyal Üretimi
```python
from core.signal_generator import MultiTimeframeSignalGenerator
generator = MultiTimeframeSignalGenerator()
signal, confidence = generator.generate_signal(data)
```

### 3. Risk Yönetimi
```python
from risk_management.position_sizer import AdvancedPositionSizer
sizer = AdvancedPositionSizer(capital=50000)
position_size = sizer.calculate_position_size(signal)
```

## Performans Hedefleri
- Win Rate: >%50
- Risk/Reward Ratio: >1.5:1
- Aylık İşlem Sayısı: 15-25
- Maksimum Açık Pozisyon: 5

## Kritik Kurallar
1. İşlem başına maksimum %1 risk
2. Aylık %8 drawdown'da sistem durur
3. MACD onayı olmadan işlem yapılmaz
4. Multi-timeframe onay gereklidir
5. Korelasyon >0.7 olan pozisyonlardan kaçınılır