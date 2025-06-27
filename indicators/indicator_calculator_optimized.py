#!/usr/bin/env python3
"""
Optimized Indicator Calculator with Caching and Parallel Processing
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from loguru import logger
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import lru_cache, partial
import multiprocessing as mp
from collections import defaultdict
import time

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
from indicators.lorentzian_classification import LorentzianClassification
from indicators.trend_vanguard import TrendVanguard


class IndicatorCache:
    """İndikatör hesaplama cache sistemi"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.cache_stats = defaultdict(int)
        
    def _get_cache_key(self, symbol: str, timeframe: str, indicator: str, 
                      data_hash: str) -> str:
        """Cache key oluştur"""
        return f"{symbol}_{timeframe}_{indicator}_{data_hash}"
    
    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """DataFrame'den hash oluştur"""
        # Son 5 satırın hash'ini al (performans için)
        hash_data = pd.util.hash_pandas_object(df.tail(5)).values
        return hashlib.md5(hash_data.tobytes()).hexdigest()[:8]
    
    def get(self, symbol: str, timeframe: str, indicator: str, 
            df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Cache'den veri al"""
        data_hash = self._get_data_hash(df)
        cache_key = self._get_cache_key(symbol, timeframe, indicator, data_hash)
        
        # Önce memory cache'e bak
        if cache_key in self.memory_cache:
            self.cache_stats['memory_hits'] += 1
            return self.memory_cache[cache_key]
        
        # Disk cache'e bak
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                # Cache dosyası 24 saatten eskiyse kullanma
                if time.time() - cache_file.stat().st_mtime > 86400:
                    cache_file.unlink()
                    self.cache_stats['expired'] += 1
                    return None
                
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.memory_cache[cache_key] = data
                    self.cache_stats['disk_hits'] += 1
                    return data
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
                
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, symbol: str, timeframe: str, indicator: str, 
            df: pd.DataFrame, result: pd.DataFrame):
        """Cache'e veri yaz"""
        data_hash = self._get_data_hash(df)
        cache_key = self._get_cache_key(symbol, timeframe, indicator, data_hash)
        
        # Memory cache'e yaz
        self.memory_cache[cache_key] = result
        
        # Disk cache'e yaz
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
            self.cache_stats['writes'] += 1
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def clear_old_cache(self, days: int = 7):
        """Eski cache dosyalarını temizle"""
        cutoff_time = time.time() - (days * 86400)
        cleared = 0
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
                cleared += 1
        
        logger.info(f"Cleared {cleared} old cache files")
        return cleared
    
    def get_stats(self) -> Dict[str, int]:
        """Cache istatistiklerini al"""
        return dict(self.cache_stats)


class OptimizedIndicatorCalculator:
    """Optimized indicator calculator with caching and parallel processing"""
    
    def __init__(self, num_workers: int = None):
        self.csv_manager = CSVDataManager()
        self.indicators_path = Path("data/indicators")
        self.indicators_path.mkdir(parents=True, exist_ok=True)
        
        # İndikatör sınıfları - tek instance
        self.indicator_classes = {
            'williams_vix_fix': WilliamsVixFix,
            'wavetrend': WaveTrend,
            'squeeze_momentum': SqueezeMomentum,
            'adx_di': ADX_DI,
            'supertrend': Supertrend,
            'macd': MACDCustom,
            'lorentzian': LorentzianClassification,
            'trend_vanguard': TrendVanguard
        }
        
        # Cache sistemi
        self.cache = IndicatorCache()
        
        # Worker sayısı
        self.num_workers = num_workers or min(mp.cpu_count() - 1, 8)
        
        # Hızlı ve yavaş indikatörleri ayır
        self.fast_indicators = ['williams_vix_fix', 'wavetrend', 'squeeze_momentum', 
                               'adx_di', 'supertrend', 'macd']
        self.slow_indicators = ['lorentzian', 'trend_vanguard']
        
        logger.info(f"Optimized Calculator initialized with {self.num_workers} workers")
    
    def _calculate_indicator_batch(self, tasks: List[Tuple[str, str, str, pd.DataFrame]]) -> List[Tuple[str, Optional[pd.DataFrame]]]:
        """Birden fazla indikatörü batch olarak hesapla"""
        results = []
        
        for symbol, timeframe, indicator_name, df in tasks:
            try:
                # Cache kontrolü
                cached_result = self.cache.get(symbol, timeframe, indicator_name, df)
                if cached_result is not None:
                    results.append((f"{symbol}_{timeframe}_{indicator_name}", cached_result))
                    continue
                
                # İndikatör hesapla
                indicator_class = self.indicator_classes[indicator_name]
                indicator = indicator_class()
                result = indicator.calculate(df)
                
                if result is not None:
                    # Cache'e kaydet
                    self.cache.set(symbol, timeframe, indicator_name, df, result)
                    results.append((f"{symbol}_{timeframe}_{indicator_name}", result))
                else:
                    results.append((f"{symbol}_{timeframe}_{indicator_name}", None))
                    
            except Exception as e:
                logger.error(f"Error calculating {indicator_name} for {symbol}: {e}")
                results.append((f"{symbol}_{timeframe}_{indicator_name}", None))
        
        return results
    
    def calculate_all_indicators_parallel(self, symbol: str, timeframe: str) -> Dict[str, pd.DataFrame]:
        """Bir sembol için tüm indikatörleri paralel hesapla"""
        results = {}
        
        # Ham veriyi yükle
        df = self.csv_manager.load_raw_data(symbol, timeframe)
        if df is None or len(df) < 100:
            logger.warning(f"Insufficient data for {symbol} {timeframe}")
            return results
        
        # Görevleri hazırla
        fast_tasks = [(symbol, timeframe, ind, df.copy()) for ind in self.fast_indicators]
        slow_tasks = [(symbol, timeframe, ind, df.copy()) for ind in self.slow_indicators]
        
        # Thread pool ile hızlı indikatörleri hesapla
        with ThreadPoolExecutor(max_workers=min(len(fast_tasks), 4)) as executor:
            future_to_task = {
                executor.submit(self._calculate_single_indicator, task): task 
                for task in fast_tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                indicator_name = task[2]
                try:
                    result = future.result()
                    if result is not None:
                        results[indicator_name] = result
                        self.save_indicator_data(symbol, timeframe, indicator_name, result)
                except Exception as e:
                    logger.error(f"Error in {indicator_name}: {e}")
        
        # Process pool ile yavaş indikatörleri hesapla
        if slow_tasks:
            with ProcessPoolExecutor(max_workers=min(len(slow_tasks), 2)) as executor:
                future_to_task = {
                    executor.submit(calculate_indicator_process, task): task 
                    for task in slow_tasks
                }
                
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    indicator_name = task[2]
                    try:
                        result = future.result()
                        if result is not None:
                            results[indicator_name] = result
                            self.save_indicator_data(symbol, timeframe, indicator_name, result)
                    except Exception as e:
                        logger.error(f"Error in {indicator_name}: {e}")
        
        return results
    
    def _calculate_single_indicator(self, task: Tuple[str, str, str, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Tek bir indikatör hesapla (thread-safe)"""
        symbol, timeframe, indicator_name, df = task
        
        try:
            # Cache kontrolü
            cached_result = self.cache.get(symbol, timeframe, indicator_name, df)
            if cached_result is not None:
                logger.debug(f"Using cached {indicator_name} for {symbol}")
                return cached_result
            
            # İndikatör hesapla
            indicator_class = self.indicator_classes[indicator_name]
            indicator = indicator_class()
            result = indicator.calculate(df)
            
            if result is not None:
                # Cache'e kaydet
                self.cache.set(symbol, timeframe, indicator_name, df, result)
                logger.info(f"Calculated {indicator_name} for {symbol} {timeframe}")
                return result
                
        except Exception as e:
            logger.error(f"Error calculating {indicator_name}: {e}")
            
        return None
    
    def save_indicator_data(self, symbol: str, timeframe: str, 
                          indicator_name: str, data: pd.DataFrame) -> bool:
        """İndikatör verisini kaydet"""
        try:
            filename = f"{symbol}_{timeframe}_{indicator_name}.csv"
            filepath = self.indicators_path / filename
            data.to_csv(filepath)
            logger.info(f"Saved {indicator_name} data for {symbol} {timeframe}")
            return True
        except Exception as e:
            logger.error(f"Error saving {indicator_name} data: {e}")
            return False
    
    def process_all_symbols_optimized(self, timeframes: List[str] = None, 
                                    symbols: List[str] = None):
        """Tüm sembolleri optimize edilmiş şekilde işle"""
        if timeframes is None:
            timeframes = ['1h', '4h', '1d']
        if symbols is None:
            symbols = ASSETS
        
        total_combinations = len(symbols) * len(timeframes)
        logger.info(f"Processing {len(symbols)} symbols for {len(timeframes)} timeframes")
        
        # İlk önce cache'i temizle
        self.cache.clear_old_cache(days=1)
        
        # Batch processing için görevleri grupla
        all_tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                all_tasks.append((symbol, timeframe))
        
        # Batch olarak işle
        batch_size = 10
        processed = 0
        
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i+batch_size]
            
            # Her batch'i paralel işle
            with ThreadPoolExecutor(max_workers=min(len(batch), self.num_workers)) as executor:
                future_to_task = {
                    executor.submit(self.calculate_all_indicators_parallel, task[0], task[1]): task 
                    for task in batch
                }
                
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    processed += 1
                    
                    try:
                        results = future.result()
                        logger.info(f"[{processed}/{total_combinations}] Processed {task[0]} {task[1]} - {len(results)} indicators")
                    except Exception as e:
                        logger.error(f"Error processing {task[0]} {task[1]}: {e}")
        
        # Cache istatistiklerini göster
        stats = self.cache.get_stats()
        logger.info(f"Cache stats: {stats}")
        logger.info("All indicators calculated successfully")
    
    @lru_cache(maxsize=128)
    def get_indicator_data(self, symbol: str, timeframe: str, 
                         indicator_name: str) -> Optional[pd.DataFrame]:
        """Kaydedilmiş indikatör verisini yükle (cached)"""
        try:
            filename = f"{symbol}_{timeframe}_{indicator_name}.csv"
            filepath = self.indicators_path / filename
            
            if not filepath.exists():
                return None
            
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            return data
            
        except Exception as e:
            logger.error(f"Error loading indicator data: {e}")
            return None
    
    def get_latest_signals_batch(self, symbols: List[str], timeframe: str) -> Dict[str, Dict]:
        """Birden fazla sembol için sinyalleri batch olarak al"""
        all_signals = {}
        
        with ThreadPoolExecutor(max_workers=min(len(symbols), 8)) as executor:
            future_to_symbol = {
                executor.submit(self.get_latest_signals, symbol, timeframe): symbol 
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    signals = future.result()
                    all_signals[symbol] = signals
                except Exception as e:
                    logger.error(f"Error getting signals for {symbol}: {e}")
                    all_signals[symbol] = {}
        
        return all_signals
    
    def get_latest_signals(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Bir sembol için en son sinyalleri al"""
        signals = {}
        
        for indicator_name in self.indicator_classes:
            try:
                data = self.get_indicator_data(symbol, timeframe, indicator_name)
                if data is None or len(data) == 0:
                    continue
                
                latest = data.iloc[-1].to_dict()
                
                # İndikatöre özel sinyal çıkarma
                indicator_class = self.indicator_classes[indicator_name]
                indicator = indicator_class()
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


def calculate_indicator_process(task: Tuple[str, str, str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Process pool için indikatör hesaplama fonksiyonu"""
    symbol, timeframe, indicator_name, df = task
    
    try:
        # İndikatör sınıflarını tekrar import et (process için)
        if indicator_name == 'lorentzian':
            from indicators.lorentzian_classification import LorentzianClassification
            indicator = LorentzianClassification()
        elif indicator_name == 'trend_vanguard':
            from indicators.trend_vanguard import TrendVanguard
            indicator = TrendVanguard()
        else:
            return None
        
        result = indicator.calculate(df)
        return result
        
    except Exception as e:
        logger.error(f"Process error calculating {indicator_name}: {e}")
        return None


def main():
    """Ana fonksiyon"""
    calculator = OptimizedIndicatorCalculator()
    
    # Argümanları kontrol et
    if len(sys.argv) > 1:
        if sys.argv[1] == '--symbol' and len(sys.argv) >= 4:
            symbol = sys.argv[2]
            timeframe = sys.argv[3]
            start_time = time.time()
            results = calculator.calculate_all_indicators_parallel(symbol, timeframe)
            elapsed = time.time() - start_time
            logger.info(f"Calculated {len(results)} indicators in {elapsed:.2f} seconds")
            
        elif sys.argv[1] == '--all':
            start_time = time.time()
            calculator.process_all_symbols_optimized()
            elapsed = time.time() - start_time
            logger.info(f"Total processing time: {elapsed:.2f} seconds")
            
        elif sys.argv[1] == '--batch' and len(sys.argv) >= 3:
            symbols = sys.argv[2].split(',')
            start_time = time.time()
            calculator.process_all_symbols_optimized(symbols=symbols)
            elapsed = time.time() - start_time
            logger.info(f"Batch processing time: {elapsed:.2f} seconds")
            
        else:
            print("Usage:")
            print("  python indicator_calculator_optimized.py --symbol AKBNK 1h")
            print("  python indicator_calculator_optimized.py --all")
            print("  python indicator_calculator_optimized.py --batch THYAO,GARAN,AKBNK")
    else:
        # Test için birkaç sembol
        test_symbols = ['THYAO', 'GARAN', 'AKBNK']
        start_time = time.time()
        calculator.process_all_symbols_optimized(symbols=test_symbols, timeframes=['1h'])
        elapsed = time.time() - start_time
        logger.info(f"Test processing time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()