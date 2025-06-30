# 📚 Hybrid Trading System - Kullanım Kılavuzu

## 🚀 Hızlı Başlangıç

### 1. Backtest Çalıştırma
```bash
# Varsayılan sembollerle (THYAO, GARAN, AKBNK, EREGL, ASELS)
python backtest/run_backtest.py

# Özel parametrelerle
python backtest/run_backtest.py --symbols THYAO GARAN AKBNK --start 2023-01-01 --end 2023-12-31 --capital 100000

# Sonuçları görüntüle
cat backtest_results.json
```

### 2. Model Eğitimi
```bash
# Varsayılan ayarlarla eğit
python train_model.py

# Özel parametrelerle
python train_model.py --symbols THYAO GARAN AKBNK EREGL ASELS SASA KCHOL --epochs 100 --lr 0.001

# Tüm sembollerle eğit
python train_model.py --symbols $(python -c "from core.csv_data_manager import CSVDataManager; print(' '.join(CSVDataManager().get_available_symbols()[:20]))")
```

### 3. Paper Trading Başlatma
```bash
# Paper trading modunda başlat
python main.py --mode paper

# Belirli sembollerle
python main.py --mode paper --symbols THYAO GARAN AKBNK

# Farklı config dosyasıyla
python main.py --mode paper --config my_config.json
```

### 4. Performance Monitoring

#### Basit Konsol Monitör:
```bash
python monitoring/simple_monitor.py
```

#### Streamlit Dashboard (opsiyonel):
```bash
# Streamlit kurulumu
pip install streamlit plotly

# Dashboard'u başlat
streamlit run monitoring/dashboard.py
```

## 📊 Sistem Akışı

### 1. Veri Hazırlama
- Sistem mevcut CSV verilerinizi kullanır: `/home/yunus/Belgeler/New_Start/data/raw/`
- İndikatörler otomatik hesaplanır ve `New_Plan/data/indicators/` altına kaydedilir

### 2. Sinyal Üretimi
- Multi-timeframe analiz (15m, 1h, 4h, 1d, 1w)
- MACD crossover stratejisi
- GRU model tahmini ile doğrulama
- Minimum %65 confidence threshold

### 3. Risk Yönetimi
- Pozisyon büyüklüğü: Sermayenin %1'i risk
- Stop Loss: 2x ATR
- Take Profit: 2:1 ve 3:1 R/R
- Trailing Stop: Kar'da 2.5x ATR
- Max 5 eş zamanlı pozisyon

### 4. Portfolio Yönetimi
- Kelly Criterion ile pozisyon boyutlandırma (%25 fraction)
- Sektör konsantrasyon kontrolü
- Korelasyon limitleri
- Aylık %8 drawdown limiti

## 🎯 Strateji Özeti

### MACD Multi-Timeframe Strategy
1. **Entry Conditions**:
   - 1H MACD bullish crossover
   - En az 2 timeframe'de MACD onayı (4H, 1D)
   - Volume ratio > 0.8
   - RSI not overbought (< 70)

2. **Exit Conditions**:
   - Stop Loss: Entry - 2x ATR
   - Take Profit 1: Entry + 2x ATR (%50 pozisyon)
   - Take Profit 2: Entry + 3x ATR (kalan %50)
   - MACD bearish crossover
   - Trailing stop aktif (2x ATR)

## 📈 Beklenen Performans

Araştırma raporuna göre:
- **Hedef Aylık Getiri**: %8-9
- **Beklenen Win Rate**: %45-55
- **Profit Factor**: >1.5
- **Max Drawdown**: <%15
- **Sharpe Ratio**: >2.0

## ⚙️ Konfigürasyon

`config.json` dosyasında önemli parametreler:

```json
{
  "portfolio": {
    "initial_capital": 100000,
    "max_positions": 5
  },
  "risk": {
    "max_risk_per_trade": 0.01,      // %1
    "stop_loss_atr_multiplier": 2.0,  // 2x ATR
    "kelly_fraction": 0.25            // %25
  },
  "signals": {
    "confidence_threshold": 0.65,     // %65
    "min_profit_target": 0.02        // %2
  }
}
```

## 🔍 Debugging

### Log Dosyaları
- Trading logs: `logs/trading.log`
- Error logs: `logs/errors.log`
- Trade history: `data/trade_history.csv`

### Yaygın Sorunlar

1. **"No data found" hatası**:
   ```bash
   # Veri durumunu kontrol et
   python simple_test.py
   ```

2. **Model not found**:
   ```bash
   # Modeli eğit
   python train_model.py --epochs 50
   ```

3. **Low signal generation**:
   - Confidence threshold'u düşür (config.json)
   - Daha fazla sembol ekle
   - Timeframe sayısını artır

## 🚨 Güvenlik Notları

1. **Paper Trading**: Her zaman önce paper trading ile test edin
2. **Backtest**: Stratejinizi historical data ile doğrulayın
3. **Risk Limitleri**: Drawdown limitlerini aşmayın
4. **Model Güncelleme**: Modeli düzenli olarak yeniden eğitin

## 📞 Destek

Sorunlarınız için:
1. `simple_test.py` ile sistem kontrolü yapın
2. Log dosyalarını inceleyin
3. `demo.py` ile örnek kullanımı görün

---

**Not**: Live trading için Algolab kredilerini `config.json` dosyasına eklemeyi unutmayın!