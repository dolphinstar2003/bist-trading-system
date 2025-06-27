#!/usr/bin/env python3
"""
Ana İndikatör Hesaplama Sistemi
Tüm indikatörleri hesaplar ve data/indicators altına kaydeder
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from loguru import logger

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS

# İndikatör modüllerini import et
from indicators.williams_vix_fix import WilliamsVixFix
from indicators.wavetrend import WaveTrend
from indicators.squeeze_momentum import SqueezeMomentum
from indicators.adx_di import ADX_DI
from indicators.supertrend import Supertrend
from indicators.macd_custom import MACDCustom
from indicators.lorentzian_classification_optimized import OptimizedLorentzianClassification as LorentzianClassification
from indicators.trend_vanguard_optimized import OptimizedTrendVanguard as TrendVanguard


class IndicatorCalculator:
    """Ana indikatör hesaplama sınıfı"""
    
    def __init__(self):
        self.csv_manager = CSVDataManager()
        self.indicators_path = Path("data/indicators")
        self.indicators_path.mkdir(parents=True, exist_ok=True)
        
        # Tüm indikatör sınıfları
        self.indicator_classes = {
            'williams_vix_fix': WilliamsVixFix(),
            'wavetrend': WaveTrend(),
            'squeeze_momentum': SqueezeMomentum(),
            'adx_di': ADX_DI(),
            'supertrend': Supertrend(),
            'macd': MACDCustom(),
            'lorentzian': LorentzianClassification(),
            'trend_vanguard': TrendVanguard()
        }
        
        # İndikatör listesi (hız için ayarlanabilir)
        self.indicator_list = list(self.indicator_classes.keys())
        
        logger.info(f"Indicator Calculator initialized with {len(self.indicator_classes)} indicators")
    
    def calculate_single_indicator(self, symbol: str, timeframe: str, 
                                 indicator_name: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Tek bir indikatör hesapla"""
        try:
            if indicator_name not in self.indicator_classes:
                logger.error(f"Unknown indicator: {indicator_name}")
                return None
            
            indicator = self.indicator_classes[indicator_name]
            result = indicator.calculate(df)
            
            if result is not None:
                logger.info(f"Calculated {indicator_name} for {symbol} {timeframe}")
                return result
            else:
                logger.warning(f"No result from {indicator_name} for {symbol} {timeframe}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating {indicator_name} for {symbol} {timeframe}: {e}")
            return None
    
    def save_indicator_data(self, symbol: str, timeframe: str, 
                          indicator_name: str, data: pd.DataFrame) -> bool:
        """İndikatör verisini kaydet"""
        try:
            # Dosya yolu oluştur
            filename = f"{symbol}_{timeframe}_{indicator_name}.csv"
            filepath = self.indicators_path / filename
            
            # Veriyi kaydet
            data.to_csv(filepath)
            logger.info(f"Saved {indicator_name} data for {symbol} {timeframe}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving {indicator_name} data: {e}")
            return False
    
    def calculate_all_indicators(self, symbol: str, timeframe: str) -> Dict[str, pd.DataFrame]:
        """Bir sembol için tüm indikatörleri hesapla"""
        results = {}
        
        # Ham veriyi yükle
        df = self.csv_manager.load_raw_data(symbol, timeframe)
        if df is None or len(df) < 100:  # Minimum veri kontrolü
            logger.warning(f"Insufficient data for {symbol} {timeframe}")
            return results
        
        # Her indikatör için hesapla (sadece aktif olanlar)
        for indicator_name in self.indicator_list:
            try:
                logger.info(f"Calculating {indicator_name} for {symbol} {timeframe}")
                
                # İndikatörü hesapla
                indicator_data = self.calculate_single_indicator(
                    symbol, timeframe, indicator_name, df.copy()
                )
                
                if indicator_data is not None:
                    # Sonucu sakla
                    results[indicator_name] = indicator_data
                    
                    # Dosyaya kaydet
                    self.save_indicator_data(symbol, timeframe, indicator_name, indicator_data)
                    
            except Exception as e:
                logger.error(f"Error in {indicator_name} calculation: {e}")
                continue
        
        return results
    
    def process_all_symbols(self, timeframes: List[str] = None):
        """Tüm semboller için indikatörleri hesapla"""
        if timeframes is None:
            timeframes = ['15m', '1h', '4h', '1d']
        
        total_combinations = len(ASSETS) * len(timeframes)
        processed = 0
        
        logger.info(f"Processing {len(ASSETS)} symbols for {len(timeframes)} timeframes")
        
        for symbol in ASSETS:
            for timeframe in timeframes:
                processed += 1
                logger.info(f"[{processed}/{total_combinations}] Processing {symbol} {timeframe}")
                
                try:
                    results = self.calculate_all_indicators(symbol, timeframe)
                    logger.info(f"Calculated {len(results)} indicators for {symbol} {timeframe}")
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol} {timeframe}: {e}")
                    continue
        
        logger.info("All indicators calculated successfully")
    
    def get_indicator_data(self, symbol: str, timeframe: str, 
                         indicator_name: str) -> Optional[pd.DataFrame]:
        """Kaydedilmiş indikatör verisini yükle"""
        try:
            filename = f"{symbol}_{timeframe}_{indicator_name}.csv"
            filepath = self.indicators_path / filename
            
            if not filepath.exists():
                logger.warning(f"Indicator data not found: {filename}")
                return None
            
            # Veriyi yükle
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return data
            
        except Exception as e:
            logger.error(f"Error loading indicator data: {e}")
            return None
    
    def get_latest_signals(self, symbol: str, timeframe: str) -> Dict[str, any]:
        """Bir sembol için en son sinyalleri al"""
        signals = {}
        
        for indicator_name in self.indicator_classes:
            try:
                # İndikatör verisini yükle
                data = self.get_indicator_data(symbol, timeframe, indicator_name)
                if data is None or len(data) == 0:
                    continue
                
                # Son değerleri al
                latest = data.iloc[-1].to_dict()
                
                # İndikatöre özel sinyal çıkarma
                indicator = self.indicator_classes[indicator_name]
                signal = indicator.get_signal(latest)
                
                signals[indicator_name] = {
                    'signal': signal,
                    'values': latest,
                    'timestamp': data.index[-1].isoformat() if hasattr(data.index[-1], 'isoformat') else str(data.index[-1])
                }
                
            except Exception as e:
                logger.error(f"Error getting signal for {indicator_name}: {e}")
                continue
        
        return signals


def main():
    """Ana fonksiyon"""
    calculator = IndicatorCalculator()
    
    # Argümanları kontrol et
    if len(sys.argv) > 1:
        if sys.argv[1] == '--symbol' and len(sys.argv) >= 4:
            symbol = sys.argv[2]
            timeframe = sys.argv[3]
            calculator.calculate_all_indicators(symbol, timeframe)
        elif sys.argv[1] == '--all':
            calculator.process_all_symbols()
        else:
            print("Usage:")
            print("  python indicator_calculator.py --symbol AKBNK 1h")
            print("  python indicator_calculator.py --all")
    else:
        # Varsayılan olarak tüm sembolleri işle
        calculator.process_all_symbols()


if __name__ == "__main__":
    main()