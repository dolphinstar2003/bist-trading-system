#!/usr/bin/env python3
"""
Mevcut verilerle indikatörleri test et
API'ye ihtiyaç duymaz, sadece data/raw klasöründeki verileri kullanır
"""

import sys
from pathlib import Path
from loguru import logger
import pandas as pd

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from indicators.indicator_calculator import IndicatorCalculator
from utils.csv_data_manager import CSVDataManager


def test_single_symbol():
    """Tek bir sembol için test"""
    # İlk mevcut dosyayı bul
    raw_data_path = Path("data/raw")
    csv_files = list(raw_data_path.glob("*.csv"))
    
    if not csv_files:
        logger.error("No data files found in data/raw/")
        return
    
    # İlk dosyadan sembol ve timeframe bilgisini çıkar
    first_file = csv_files[0].name
    parts = first_file.replace("_raw.csv", "").split("_")
    symbol = parts[0]
    timeframe = parts[1]
    
    logger.info(f"Testing with {symbol} {timeframe}")
    
    # CSV manager ile veriyi yükle
    csv_manager = CSVDataManager()
    df = csv_manager.load_raw_data(symbol, timeframe)
    
    if df is None:
        logger.error(f"Could not load data for {symbol} {timeframe}")
        return
    
    logger.info(f"Loaded {len(df)} rows of data")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # İndikatör hesaplayıcıyı başlat
    calculator = IndicatorCalculator()
    
    # Tüm indikatörleri hesapla
    logger.info(f"\nCalculating indicators for {symbol} {timeframe}...")
    results = calculator.calculate_all_indicators(symbol, timeframe)
    
    logger.info(f"\nCompleted {len(results)} indicators:")
    for indicator_name, data in results.items():
        logger.info(f"- {indicator_name}: {len(data)} rows")
        if len(data) > 0:
            logger.info(f"  Columns: {data.columns.tolist()[:5]}...")  # İlk 5 sütun


def process_all_available():
    """Mevcut tüm veriler için indikatörleri hesapla"""
    raw_data_path = Path("data/raw")
    csv_files = list(raw_data_path.glob("*.csv"))
    
    if not csv_files:
        logger.error("No data files found in data/raw/")
        return
    
    logger.info(f"Found {len(csv_files)} data files")
    
    # Benzersiz sembol-timeframe kombinasyonlarını bul
    combinations = set()
    for csv_file in csv_files:
        parts = csv_file.name.replace("_raw.csv", "").split("_")
        if len(parts) >= 2:
            symbol = parts[0]
            timeframe = parts[1]
            combinations.add((symbol, timeframe))
    
    logger.info(f"Found {len(combinations)} unique symbol-timeframe combinations")
    
    # İndikatör hesaplayıcıyı başlat
    calculator = IndicatorCalculator()
    
    # Her kombinasyon için hesapla
    for i, (symbol, timeframe) in enumerate(sorted(combinations), 1):
        logger.info(f"\n[{i}/{len(combinations)}] Processing {symbol} {timeframe}")
        
        try:
            results = calculator.calculate_all_indicators(symbol, timeframe)
            if results:
                logger.success(f"✓ Calculated {len(results)} indicators for {symbol} {timeframe}")
            else:
                logger.warning(f"✗ No indicators calculated for {symbol} {timeframe}")
                
        except Exception as e:
            logger.error(f"✗ Error processing {symbol} {timeframe}: {e}")
    
    logger.info("\n" + "="*50)
    logger.info("Indicator calculation completed!")
    
    # Sonuçları kontrol et
    indicators_path = Path("data/indicators")
    if indicators_path.exists():
        indicator_files = list(indicators_path.glob("*.csv"))
        logger.info(f"Created {len(indicator_files)} indicator files")
        
        # İlk birkaç dosyayı listele
        for file in indicator_files[:10]:
            logger.info(f"- {file.name}")
        if len(indicator_files) > 10:
            logger.info(f"... and {len(indicator_files) - 10} more files")


def check_single_indicator(symbol: str, timeframe: str, indicator_name: str):
    """Tek bir indikatörü detaylı kontrol et"""
    calculator = IndicatorCalculator()
    
    # İndikatör verisini yükle
    indicator_data = calculator.get_indicator_data(symbol, timeframe, indicator_name)
    
    if indicator_data is None:
        logger.error(f"No data found for {indicator_name}")
        return
    
    logger.info(f"\n{indicator_name} for {symbol} {timeframe}:")
    logger.info(f"Shape: {indicator_data.shape}")
    logger.info(f"Columns: {indicator_data.columns.tolist()}")
    logger.info(f"\nLast 5 rows:")
    print(indicator_data.tail())
    
    # Son sinyali al
    signals = calculator.get_latest_signals(symbol, timeframe)
    if indicator_name in signals:
        logger.info(f"\nLatest signal: {signals[indicator_name]['signal']}")
        logger.info(f"Signal timestamp: {signals[indicator_name]['timestamp']}")


def main():
    """Ana fonksiyon"""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            # Sadece tek bir sembol test et
            test_single_symbol()
        elif sys.argv[1] == '--all':
            # Tüm mevcut verileri işle
            process_all_available()
        elif sys.argv[1] == '--check' and len(sys.argv) >= 5:
            # Belirli bir indikatörü kontrol et
            symbol = sys.argv[2]
            timeframe = sys.argv[3]
            indicator = sys.argv[4]
            check_single_indicator(symbol, timeframe, indicator)
        else:
            print("Usage:")
            print("  python test_indicators.py --test              # Test single symbol")
            print("  python test_indicators.py --all               # Process all available data")
            print("  python test_indicators.py --check SYMBOL TF INDICATOR  # Check specific indicator")
            print("\nExample:")
            print("  python test_indicators.py --check AKBNK 1h supertrend")
    else:
        # Varsayılan: tek sembol test
        test_single_symbol()


if __name__ == "__main__":
    main()