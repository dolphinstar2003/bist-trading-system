#!/usr/bin/env python3
"""
ML Pipeline Runner
Tüm ML sürecini yöneten ana script
"""

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ML modülleri
from ml_models.feature_engineering import FeatureEngineering
from ml_models.model_trainer import ModelTrainer
from ml_models.predictor import Predictor
from ml_models.backtester import Backtester
from config.assets import ASSETS


def setup_logger():
    """Logger ayarları"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    logger.add(
        "logs/ml_pipeline_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )


def check_data_availability():
    """Veri varlığını kontrol et"""
    data_dir = Path("data/raw")
    if not data_dir.exists():
        logger.error("Data directory not found! Run download_data.py first.")
        return False
    
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        logger.error("No data files found! Run download_data.py first.")
        return False
    
    logger.info(f"Found {len(csv_files)} data files")
    return True


def train_models(symbols=None, timeframes=None, target_column='target_direction_1'):
    """Modelleri eğit"""
    if symbols is None:
        symbols = ['AKBNK', 'THYAO', 'GARAN', 'ASELS', 'SISE']
    if timeframes is None:
        timeframes = ['1h', '4h']
    
    trainer = ModelTrainer(model_type='classification')
    results = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training models for {symbol} {timeframe}")
            logger.info(f"{'='*60}")
            
            result = trainer.train_and_save(symbol, timeframe, target_column)
            if result:
                results.append(result)
                logger.success(f"✓ Model trained: {result['best_model_name']}")
            else:
                logger.error(f"✗ Training failed for {symbol} {timeframe}")
    
    return results


def make_predictions(symbols=None, timeframes=None):
    """Tahminler yap"""
    if symbols is None:
        symbols = ['AKBNK', 'THYAO', 'GARAN']
    if timeframes is None:
        timeframes = ['1h', '4h']
    
    predictor = Predictor()
    predictions = []
    
    logger.info("\nMaking predictions...")
    
    for symbol in symbols:
        for timeframe in timeframes:
            result = predictor.predict_and_save(symbol, timeframe, use_ensemble=True)
            if result:
                predictions.append(result)
                logger.info(f"{symbol} {timeframe}: {result['action']} "
                          f"(confidence: {result['confidence']:.1%})")
    
    return predictions


def run_backtests(symbols=None, timeframes=None):
    """Backtest çalıştır"""
    if symbols is None:
        symbols = ['AKBNK', 'THYAO', 'GARAN']
    if timeframes is None:
        timeframes = ['1h', '4h']
    
    backtester = Backtester(initial_capital=100000)
    
    logger.info("\nRunning backtests...")
    
    # Multi backtest
    results = backtester.run_multi_backtest(
        symbols, timeframes,
        start_date="2024-01-01",
        end_date=None  # Günümüze kadar
    )
    
    return results


def main():
    """Ana pipeline"""
    setup_logger()
    
    logger.info("ML Trading Pipeline Started")
    
    # Veri kontrolü
    if not check_data_availability():
        return
    
    # Menü
    while True:
        print("\n" + "="*60)
        print("ML TRADING PIPELINE")
        print("="*60)
        print("1. Train Models (Modelleri Eğit)")
        print("2. Make Predictions (Tahmin Yap)")
        print("3. Run Backtests (Backtest Çalıştır)")
        print("4. Full Pipeline (Tüm Süreç)")
        print("5. Quick Test (Hızlı Test - 1 sembol)")
        print("0. Exit")
        print("="*60)
        
        choice = input("\nSelect option (0-5): ")
        
        if choice == '0':
            logger.info("Exiting...")
            break
            
        elif choice == '1':
            # Model eğitimi
            print("\nSelect symbols (comma separated, or press Enter for defaults):")
            print(f"Available: {', '.join(ASSETS[:10])}...")
            symbols_input = input().strip()
            
            if symbols_input:
                symbols = [s.strip().upper() for s in symbols_input.split(',')]
            else:
                symbols = None
            
            train_models(symbols)
            
        elif choice == '2':
            # Tahmin
            predictions = make_predictions()
            
            if predictions:
                print("\n" + "="*60)
                print("CURRENT SIGNALS")
                print("="*60)
                
                # Güçlü sinyalleri göster
                strong_signals = [p for p in predictions if p['confidence'] > 0.7]
                if strong_signals:
                    print("\nSTRONG SIGNALS (>70% confidence):")
                    for signal in sorted(strong_signals, key=lambda x: x['confidence'], reverse=True):
                        print(f"  {signal['symbol']} {signal['timeframe']}: "
                              f"{signal['action']} ({signal['confidence']:.1%})")
                else:
                    print("No strong signals found.")
                    
        elif choice == '3':
            # Backtest
            results = run_backtests()
            
        elif choice == '4':
            # Full pipeline
            logger.info("Running full pipeline...")
            
            # 1. Eğit
            train_results = train_models()
            
            # 2. Tahmin
            predictions = make_predictions()
            
            # 3. Backtest
            backtest_results = run_backtests()
            
            logger.success("Full pipeline completed!")
            
        elif choice == '5':
            # Hızlı test
            logger.info("Running quick test with AKBNK 1h...")
            
            # Eğit
            trainer = ModelTrainer(model_type='classification')
            result = trainer.train_and_save('AKBNK', '1h')
            
            if result:
                # Tahmin
                predictor = Predictor()
                prediction = predictor.predict_and_save('AKBNK', '1h')
                
                if prediction:
                    print(f"\nPrediction: {prediction['action']} "
                          f"(confidence: {prediction['confidence']:.1%})")
                    
                    # Backtest
                    backtester = Backtester()
                    backtest = backtester.run_backtest('AKBNK', '1h')


if __name__ == "__main__":
    main()