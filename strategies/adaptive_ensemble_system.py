#!/usr/bin/env python3
"""
Adaptif Ensemble Trading Sistemi
Dinamik ağırlıklandırma ve performans takibi ile
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque, defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
from loguru import logger

# Proje imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.csv_data_manager import CSVDataManager
from indicators.indicator_calculator import IndicatorCalculator


class AdaptiveEnsembleSystem:
    """Dinamik ağırlıklandırmalı adaptif ensemble trading sistemi"""
    
    def __init__(self, initial_capital: float = 50000):
        self.csv_manager = CSVDataManager()
        self.indicator_calculator = IndicatorCalculator()
        self.initial_capital = initial_capital
        
        # İndikatör kategorileri ve başlangıç ağırlıkları
        self.indicator_categories = {
            'momentum': {
                'indicators': ['wavetrend', 'squeeze_momentum', 'macd'],
                'weight': 0.3,
                'performance_score': 1.0,
                'trades': deque(maxlen=100)
            },
            'trend': {
                'indicators': ['supertrend', 'adx_di'],
                'weight': 0.3,
                'performance_score': 1.0,
                'trades': deque(maxlen=100)
            },
            'ml_based': {
                'indicators': ['lorentzian', 'trend_vanguard'],
                'weight': 0.25,
                'performance_score': 1.0,
                'trades': deque(maxlen=100)
            },
            'volatility': {
                'indicators': ['williams_vix_fix'],
                'weight': 0.15,
                'performance_score': 1.0,
                'trades': deque(maxlen=100)
            }
        }
        
        # Performans takibi
        self.performance_history = deque(maxlen=1000)
        self.signal_memory = deque(maxlen=1000)
        self.last_weight_update = datetime.now()
        
    def load_all_indicators(self, symbol: str, timeframe: str) -> Dict:
        """Tüm indikatör verilerini yükle"""
        indicators_data = {}
        
        for category, config in self.indicator_categories.items():
            for indicator in config['indicators']:
                try:
                    # İndikatör verisini yükle
                    df = self.csv_manager.load_indicator_data(symbol, timeframe, indicator)
                    if df is not None and not df.empty:
                        indicators_data[indicator] = df
                        logger.debug(f"{indicator} yüklendi: {len(df)} bar")
                except Exception as e:
                    logger.error(f"{indicator} yüklenemedi: {e}")
                    
        return indicators_data
    
    def calculate_category_signal(self, category: str, indicators_data: Dict) -> Tuple[str, float]:
        """Kategori bazında sinyal hesapla"""
        config = self.indicator_categories[category]
        signals = []
        strengths = []
        
        for indicator in config['indicators']:
            if indicator not in indicators_data:
                continue
                
            df = indicators_data[indicator]
            signal, strength = self._get_indicator_signal(indicator, df)
            
            if signal != 'HOLD':
                signals.append(signal)
                strengths.append(strength)
        
        if not signals:
            return 'HOLD', 0.0
        
        # Çoğunluk oylaması
        buy_count = sum(1 for s in signals if s == 'BUY')
        sell_count = sum(1 for s in signals if s == 'SELL')
        
        if buy_count > sell_count:
            return 'BUY', np.mean([s for s, sig in zip(strengths, signals) if sig == 'BUY'])
        elif sell_count > buy_count:
            return 'SELL', np.mean([s for s, sig in zip(strengths, signals) if sig == 'SELL'])
        else:
            return 'HOLD', 0.0
    
    def _get_indicator_signal(self, indicator: str, df: pd.DataFrame) -> Tuple[str, float]:
        """Tek bir indikatörden sinyal al"""
        if df.empty:
            return 'HOLD', 0.0
            
        last_row = df.iloc[-1]
        signal = 'HOLD'
        strength = 0.0
        
        # Her indikatör için özel sinyal mantığı
        if indicator == 'supertrend':
            if 'trend' in last_row and last_row['trend'] == 1:
                signal = 'BUY'
                strength = abs(last_row.get('value', 0) - last_row.get('close', 0)) / last_row.get('close', 1)
            elif 'trend' in last_row and last_row['trend'] == -1:
                signal = 'SELL'
                strength = abs(last_row.get('value', 0) - last_row.get('close', 0)) / last_row.get('close', 1)
                
        elif indicator == 'macd':
            if 'signal' in last_row:
                if last_row['signal'] == 1:
                    signal = 'BUY'
                    strength = abs(last_row.get('histogram', 0))
                elif last_row['signal'] == -1:
                    signal = 'SELL'
                    strength = abs(last_row.get('histogram', 0))
                    
        elif indicator == 'wavetrend':
            if 'signal' in last_row:
                if last_row['signal'] == 1:
                    signal = 'BUY'
                    strength = min(abs(last_row.get('wt1', 0) + 60) / 40, 1.0)
                elif last_row['signal'] == -1:
                    signal = 'SELL'
                    strength = min(abs(last_row.get('wt1', 0) - 60) / 40, 1.0)
                    
        elif indicator == 'adx_di':
            if 'signal' in last_row and last_row.get('adx', 0) > 25:
                if last_row['signal'] == 1:
                    signal = 'BUY'
                    strength = min(last_row.get('adx', 0) / 50, 1.0)
                elif last_row['signal'] == -1:
                    signal = 'SELL'
                    strength = min(last_row.get('adx', 0) / 50, 1.0)
                    
        elif indicator == 'squeeze_momentum':
            if 'signal' in last_row:
                if last_row['signal'] == 1:
                    signal = 'BUY'
                    strength = min(abs(last_row.get('value', 0)) / 10, 1.0)
                elif last_row['signal'] == -1:
                    signal = 'SELL'
                    strength = min(abs(last_row.get('value', 0)) / 10, 1.0)
                    
        elif indicator == 'williams_vix_fix':
            if 'signal' in last_row:
                if last_row['signal'] == 1:  # Yüksek VIX = potansiyel dip
                    signal = 'BUY'
                    strength = min(last_row.get('value', 0) / 20, 1.0)
                    
        elif indicator == 'lorentzian':
            if 'signal' in last_row:
                if last_row['signal'] == 1:
                    signal = 'BUY'
                    strength = last_row.get('confidence', 0.5)
                elif last_row['signal'] == -1:
                    signal = 'SELL'
                    strength = last_row.get('confidence', 0.5)
                    
        elif indicator == 'trend_vanguard':
            if 'signal' in last_row:
                if last_row['signal'] == 1:
                    signal = 'BUY'
                    strength = last_row.get('strength', 0.5)
                elif last_row['signal'] == -1:
                    signal = 'SELL'
                    strength = last_row.get('strength', 0.5)
        
        return signal, strength
    
    def generate_ensemble_signal(self, symbol: str, timeframe: str) -> Dict:
        """Ensemble sinyal üret"""
        # İndikatörleri yükle
        indicators_data = self.load_all_indicators(symbol, timeframe)
        
        if not indicators_data:
            logger.warning(f"{symbol} için indikatör verisi bulunamadı")
            return {'signal': 'HOLD', 'confidence': 0.0, 'details': {}}
        
        # Her kategori için sinyal al
        category_signals = {}
        weighted_buy_score = 0.0
        weighted_sell_score = 0.0
        
        for category, config in self.indicator_categories.items():
            signal, strength = self.calculate_category_signal(category, indicators_data)
            category_signals[category] = {
                'signal': signal,
                'strength': strength,
                'weight': config['weight']
            }
            
            # Ağırlıklı skorları hesapla
            if signal == 'BUY':
                weighted_buy_score += strength * config['weight']
            elif signal == 'SELL':
                weighted_sell_score += strength * config['weight']
        
        # Final sinyal
        final_signal = 'HOLD'
        confidence = 0.0
        
        if weighted_buy_score > weighted_sell_score and weighted_buy_score > 0.3:
            final_signal = 'BUY'
            confidence = weighted_buy_score
        elif weighted_sell_score > weighted_buy_score and weighted_sell_score > 0.3:
            final_signal = 'SELL'
            confidence = weighted_sell_score
        
        # Detaylı sonuç
        result = {
            'signal': final_signal,
            'confidence': confidence,
            'buy_score': weighted_buy_score,
            'sell_score': weighted_sell_score,
            'category_signals': category_signals,
            'timestamp': datetime.now().isoformat()
        }
        
        # Sinyal hafızasına ekle
        self.signal_memory.append(result)
        
        return result
    
    def update_weights(self, performance_window: int = 30):
        """Performansa göre kategori ağırlıklarını güncelle"""
        # Son güncelleme üzerinden yeterli zaman geçti mi?
        if (datetime.now() - self.last_weight_update).days < 7:
            return
        
        # Her kategori için Sharpe Ratio hesapla
        for category in self.indicator_categories:
            sharpe = self.calculate_category_sharpe(category, performance_window)
            
            # Performans skorunu güncelle (EMA ile smooth)
            alpha = 0.1
            old_score = self.indicator_categories[category]['performance_score']
            new_score = alpha * sharpe + (1 - alpha) * old_score
            self.indicator_categories[category]['performance_score'] = max(0.1, new_score)
        
        # Ağırlıkları normalize et
        total_score = sum(cat['performance_score'] for cat in self.indicator_categories.values())
        
        for category in self.indicator_categories:
            self.indicator_categories[category]['weight'] = (
                self.indicator_categories[category]['performance_score'] / total_score
            )
        
        self.last_weight_update = datetime.now()
        logger.info("Kategori ağırlıkları güncellendi")
        self._log_weights()
    
    def calculate_category_sharpe(self, category: str, window: int) -> float:
        """Kategori için Sharpe Ratio hesapla"""
        trades = list(self.indicator_categories[category]['trades'])
        
        if len(trades) < 10:
            return 1.0  # Yeterli veri yok, nötr skor
        
        # Son window günlük işlemleri al
        recent_trades = trades[-window:] if len(trades) > window else trades
        
        if not recent_trades:
            return 1.0
        
        # Günlük getirileri hesapla
        returns = [t['return'] for t in recent_trades if 'return' in t]
        
        if len(returns) < 5:
            return 1.0
        
        # Sharpe Ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 1.0
        
        # Yıllıklandırılmış Sharpe (252 işlem günü)
        sharpe = (mean_return * np.sqrt(252)) / std_return
        
        # -2 ile 2 arasında sınırla
        return max(-2, min(2, sharpe))
    
    def record_trade_result(self, category: str, trade_result: Dict):
        """İşlem sonucunu kaydet"""
        self.indicator_categories[category]['trades'].append(trade_result)
        self.performance_history.append({
            'category': category,
            'result': trade_result,
            'timestamp': datetime.now().isoformat()
        })
    
    def _log_weights(self):
        """Mevcut ağırlıkları logla"""
        logger.info("Kategori Ağırlıkları:")
        for category, config in self.indicator_categories.items():
            logger.info(f"  {category}: {config['weight']:.3f} (score: {config['performance_score']:.3f})")
    
    def get_system_stats(self) -> Dict:
        """Sistem istatistiklerini al"""
        stats = {
            'total_signals': len(self.signal_memory),
            'category_weights': {},
            'category_performance': {},
            'last_update': self.last_weight_update.isoformat()
        }
        
        for category, config in self.indicator_categories.items():
            stats['category_weights'][category] = config['weight']
            stats['category_performance'][category] = {
                'score': config['performance_score'],
                'trades': len(config['trades'])
            }
        
        return stats
    
    def save_state(self, filepath: str):
        """Sistem durumunu kaydet"""
        state = {
            'indicator_categories': {
                cat: {
                    'weight': config['weight'],
                    'performance_score': config['performance_score'],
                    'trades': list(config['trades'])
                }
                for cat, config in self.indicator_categories.items()
            },
            'performance_history': list(self.performance_history),
            'last_weight_update': self.last_weight_update.isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Sistem durumunu yükle"""
        if not Path(filepath).exists():
            return
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Durumu geri yükle
        for cat, data in state['indicator_categories'].items():
            if cat in self.indicator_categories:
                self.indicator_categories[cat]['weight'] = data['weight']
                self.indicator_categories[cat]['performance_score'] = data['performance_score']
                self.indicator_categories[cat]['trades'] = deque(data['trades'], maxlen=100)
        
        self.performance_history = deque(state['performance_history'], maxlen=1000)
        self.last_weight_update = datetime.fromisoformat(state['last_weight_update'])


def main():
    """Test fonksiyonu"""
    system = AdaptiveEnsembleSystem()
    
    # Test sinyali
    symbol = 'THYAO'
    timeframe = '1h'
    
    signal = system.generate_ensemble_signal(symbol, timeframe)
    
    print(f"\nEnsemble Sinyal: {symbol} - {timeframe}")
    print(f"Sinyal: {signal['signal']}")
    print(f"Güven: {signal['confidence']:.3f}")
    print(f"Buy Score: {signal['buy_score']:.3f}")
    print(f"Sell Score: {signal['sell_score']:.3f}")
    
    print("\nKategori Detayları:")
    for cat, data in signal['category_signals'].items():
        print(f"  {cat}: {data['signal']} (strength: {data['strength']:.3f}, weight: {data['weight']:.3f})")
    
    # Sistem istatistikleri
    stats = system.get_system_stats()
    print(f"\nSistem İstatistikleri:")
    print(f"Toplam Sinyal: {stats['total_signals']}")
    print(f"Son Güncelleme: {stats['last_update']}")


if __name__ == "__main__":
    main()