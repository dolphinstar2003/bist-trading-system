#!/usr/bin/env python3
"""
Quick ML Test - Hızlı ML testi
Tek komutla tüm ML pipeline'ı test eder
"""

import sys
from pathlib import Path
from loguru import logger

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level} | {message}")

print("""
===================================
ML TRADING SYSTEM - QUICK TEST
===================================
""")

# 1. Veri kontrolü
print("1. Checking data...")
data_dir = Path("data/raw")
if not data_dir.exists():
    print("   ❌ No data found! First run: python download_data.py")
    sys.exit(1)

csv_files = list(data_dir.glob("*.csv"))
print(f"   ✓ Found {len(csv_files)} data files")

# 2. İndikatör kontrolü
print("\n2. Checking indicators...")
indicator_dir = Path("data/indicators")
if indicator_dir.exists():
    indicator_files = list(indicator_dir.glob("*.csv"))
    print(f"   ✓ Found {len(indicator_files)} indicator files")
else:
    print("   ⚠️  No indicators found, will calculate")

# 3. Model eğitimi
print("\n3. Training model...")
from ml_models.model_trainer import ModelTrainer

trainer = ModelTrainer(model_type='classification')
result = trainer.train_and_save('AKBNK', '1h', target_column='target_direction_1')

if result:
    print(f"   ✓ Best model: {result['best_model_name']}")
    
    # Model sonuçları
    for model_name, model_result in result['results'].items():
        f1_score = model_result['test_metrics']['f1_score']
        print(f"     - {model_name}: F1={f1_score:.3f}")
else:
    print("   ❌ Model training failed!")
    sys.exit(1)

# 4. Tahmin
print("\n4. Making prediction...")
from ml_models.predictor import Predictor

predictor = Predictor()
prediction = predictor.predict_and_save('AKBNK', '1h', use_ensemble=False)

if prediction:
    print(f"   ✓ Signal: {prediction['action']}")
    print(f"   ✓ Confidence: {prediction['confidence']:.1%}")
    print(f"   ✓ Strength: {prediction['signal_strength']:.3f}")
else:
    print("   ❌ Prediction failed!")

# 5. Backtest
print("\n5. Running backtest...")
from ml_models.backtester import Backtester

backtester = Backtester(initial_capital=100000)
backtest = backtester.run_backtest('AKBNK', '1h', start_date='2024-01-01')

if backtest and backtest.get('total_trades', 0) > 0:
    print(f"   ✓ Total return: {backtest['total_return']:.1%}")
    print(f"   ✓ Win rate: {backtest['win_rate']:.1%}")
    print(f"   ✓ Sharpe ratio: {backtest['sharpe_ratio']:.2f}")
    print(f"   ✓ Total trades: {backtest['total_trades']}")
else:
    print("   ⚠️  No trades in backtest")

print("""
===================================
TEST COMPLETED!
===================================

Next steps:
1. Train more models: python run_ml_pipeline.py
2. Check predictions: cat data/predictions/*_signals_*.csv
3. View backtest results: cat data/backtest_results/*.json
""")