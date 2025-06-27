#!/usr/bin/env python3
"""
Market Rejim Tanıma Sistemi
Piyasa koşullarını analiz edip rejim tespit eder
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from datetime import datetime
from loguru import logger

# Proje imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.csv_data_manager import CSVDataManager


class MarketRegimeDetector:
    """Piyasa rejimlerini tespit eden sistem"""
    
    def __init__(self):
        self.csv_manager = CSVDataManager()
        
        # Rejim tanımları
        self.regimes = {
            'strong_trend_up': {
                'description': 'Güçlü yükseliş trendi',
                'color': 'green',
                'risk_level': 0.7,
                'position_size_multiplier': 1.2
            },
            'strong_trend_down': {
                'description': 'Güçlü düşüş trendi',
                'color': 'red',
                'risk_level': 0.9,
                'position_size_multiplier': 0.5
            },
            'weak_trend': {
                'description': 'Zayıf trend',
                'color': 'yellow',
                'risk_level': 0.5,
                'position_size_multiplier': 0.8
            },
            'ranging': {
                'description': 'Yatay piyasa',
                'color': 'blue',
                'risk_level': 0.4,
                'position_size_multiplier': 0.6
            },
            'volatile': {
                'description': 'Yüksek volatilite',
                'color': 'orange',
                'risk_level': 0.8,
                'position_size_multiplier': 0.4
            },
            'squeeze': {
                'description': 'Sıkışma/Konsolidasyon',
                'color': 'purple',
                'risk_level': 0.3,
                'position_size_multiplier': 0.5
            }
        }
        
        # Rejim geçmişi
        self.regime_history = []
        
    def detect_regime(self, symbol: str, timeframe: str = '1h') -> Tuple[str, Dict]:
        """Mevcut piyasa rejimini tespit et"""
        # Veri yükle
        df = self.csv_manager.load_raw_data(symbol, timeframe)
        
        if df is None or len(df) < 100:
            logger.warning(f"{symbol} için yeterli veri yok")
            return 'unknown', {}
        
        # Teknik göstergeleri hesapla
        indicators = self._calculate_regime_indicators(df)
        
        # Rejim skorlarını hesapla
        regime_scores = {
            'strong_trend_up': self._check_strong_trend_up(indicators),
            'strong_trend_down': self._check_strong_trend_down(indicators),
            'weak_trend': self._check_weak_trend(indicators),
            'ranging': self._check_ranging(indicators),
            'volatile': self._check_volatile(indicators),
            'squeeze': self._check_squeeze(indicators)
        }
        
        # En yüksek skorlu rejimi seç
        current_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
        
        # Detaylı analiz
        analysis = {
            'regime': current_regime,
            'scores': regime_scores,
            'indicators': indicators,
            'confidence': regime_scores[current_regime],
            'timestamp': datetime.now().isoformat()
        }
        
        # Geçmişe ekle
        self.regime_history.append(analysis)
        
        return current_regime, analysis
    
    def _calculate_regime_indicators(self, df: pd.DataFrame) -> Dict:
        """Rejim tespiti için gerekli göstergeleri hesapla"""
        indicators = {}
        
        # ADX - Trend gücü (basit yaklaşım)
        indicators['adx'] = self._calculate_adx(df)
        indicators['plus_di'] = 50  # Dummy değer
        indicators['minus_di'] = 50  # Dummy değer
        
        # ATR - Volatilite
        indicators['atr'] = self._calculate_atr(df)
        indicators['atr_ratio'] = indicators['atr'] / df['close']
        
        # Bollinger Bands - Volatilite ve squeeze
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        upper = sma_20 + (2 * std_20)
        lower = sma_20 - (2 * std_20)
        indicators['bb_width'] = (upper - lower) / sma_20
        indicators['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        # Moving Averages - Trend
        indicators['sma_20'] = df['close'].rolling(20).mean()
        indicators['sma_50'] = df['close'].rolling(50).mean()
        indicators['sma_200'] = df['close'].rolling(200).mean()
        
        # Price action
        indicators['price_change_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
        indicators['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Volume
        indicators['volume_sma'] = df['volume'].rolling(20).mean()
        indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(df['close'])
        
        # Linear Regression - Trend yönü ve gücü
        if len(df) >= 20:
            x = np.arange(20)
            y = df['close'].iloc[-20:].values
            slope, intercept = np.polyfit(x, y, 1)
            indicators['trend_slope'] = slope / df['close'].iloc[-1]  # Normalize edilmiş eğim
        else:
            indicators['trend_slope'] = 0
        
        return indicators
    
    def _check_strong_trend_up(self, indicators: Dict) -> float:
        """Güçlü yükseliş trendi skoru"""
        score = 0.0
        
        # ADX > 30 ve +DI > -DI
        if indicators['adx'].iloc[-1] > 30:
            score += 0.3
        # Dummy değerler için basit kontrol
        if isinstance(indicators['plus_di'], (int, float)) and isinstance(indicators['minus_di'], (int, float)):
            if indicators['plus_di'] > indicators['minus_di']:
                score += 0.2
        else:
            if indicators['plus_di'].iloc[-1] > indicators['minus_di'].iloc[-1]:
                score += 0.2
        
        # Fiyat MA'ların üzerinde
        current_price = indicators['sma_20'].index[-1] if hasattr(indicators['sma_20'], 'index') else 0
        if indicators['sma_20'].iloc[-1] > 0:  # Basit kontrol
            if indicators['bb_position'].iloc[-1] > 0.5:  # BB'nin üst yarısında
                score += 0.15
        if indicators['sma_50'].iloc[-1] > 0:
            if indicators['bb_position'].iloc[-1] > 0.6:  # Daha da yukarıda
                score += 0.15
        
        # Pozitif trend eğimi
        if indicators['trend_slope'] > 0.001:
            score += 0.2
        
        return min(score, 1.0)
    
    def _check_strong_trend_down(self, indicators: Dict) -> float:
        """Güçlü düşüş trendi skoru"""
        score = 0.0
        
        # ADX > 30 ve -DI > +DI
        if indicators['adx'].iloc[-1] > 30:
            score += 0.3
        # Dummy değerler için basit kontrol
        if isinstance(indicators['plus_di'], (int, float)) and isinstance(indicators['minus_di'], (int, float)):
            if indicators['minus_di'] > indicators['plus_di']:
                score += 0.2
        else:
            if indicators['minus_di'].iloc[-1] > indicators['plus_di'].iloc[-1]:
                score += 0.2
        
        # Fiyat MA'ların altında
        if indicators['sma_20'].iloc[-1] > 0:  # Basit kontrol
            if indicators['bb_position'].iloc[-1] < 0.5:  # BB'nin alt yarısında
                score += 0.15
        if indicators['sma_50'].iloc[-1] > 0:
            if indicators['bb_position'].iloc[-1] < 0.4:  # Daha da aşağıda
                score += 0.15
        
        # Negatif trend eğimi
        if indicators['trend_slope'] < -0.001:
            score += 0.2
        
        return min(score, 1.0)
    
    def _check_weak_trend(self, indicators: Dict) -> float:
        """Zayıf trend skoru"""
        score = 0.0
        
        # ADX 20-30 arası
        adx = indicators['adx'].iloc[-1]
        if 20 < adx < 30:
            score += 0.4
        
        # Hafif eğim
        slope = abs(indicators['trend_slope'])
        if 0.0005 < slope < 0.001:
            score += 0.3
        
        # Orta volatilite
        if 0.01 < indicators['atr_ratio'].iloc[-1] < 0.02:
            score += 0.3
        
        return min(score, 1.0)
    
    def _check_ranging(self, indicators: Dict) -> float:
        """Yatay piyasa skoru"""
        score = 0.0
        
        # ADX < 20
        if indicators['adx'].iloc[-1] < 20:
            score += 0.3
        
        # Düşük trend eğimi
        if abs(indicators['trend_slope']) < 0.0005:
            score += 0.3
        
        # RSI 40-60 arası
        rsi = indicators['rsi'].iloc[-1]
        if 40 < rsi < 60:
            score += 0.2
        
        # BB ortasına yakın
        bb_pos = indicators['bb_position'].iloc[-1]
        if 0.3 < bb_pos < 0.7:
            score += 0.2
        
        return min(score, 1.0)
    
    def _check_volatile(self, indicators: Dict) -> float:
        """Yüksek volatilite skoru"""
        score = 0.0
        
        # Yüksek ATR
        if indicators['atr_ratio'].iloc[-1] > 0.025:
            score += 0.3
        
        # Geniş Bollinger Bands
        if indicators['bb_width'].iloc[-1] > 0.04:
            score += 0.3
        
        # Yüksek intraday range
        if indicators['high_low_ratio'].iloc[-1] > 0.03:
            score += 0.2
        
        # Volume spike
        if indicators['volume_ratio'].iloc[-1] > 1.5:
            score += 0.2
        
        return min(score, 1.0)
    
    def _check_squeeze(self, indicators: Dict) -> float:
        """Sıkışma/Konsolidasyon skoru"""
        score = 0.0
        
        # Dar Bollinger Bands
        bb_width = indicators['bb_width'].iloc[-1]
        if bb_width < 0.02:
            score += 0.4
        
        # Düşük ATR
        if indicators['atr_ratio'].iloc[-1] < 0.01:
            score += 0.3
        
        # Düşük volume
        if indicators['volume_ratio'].iloc[-1] < 0.8:
            score += 0.2
        
        # ADX düşüş trendinde
        if len(indicators['adx']) > 5:
            adx_trend = indicators['adx'].iloc[-1] < indicators['adx'].iloc[-5]
            if adx_trend:
                score += 0.1
        
        return min(score, 1.0)
    
    def get_regime_recommendation(self, regime: str) -> Dict:
        """Rejim bazında işlem önerileri"""
        recommendations = {
            'strong_trend_up': {
                'strategy': 'Trend takibi',
                'indicators': ['supertrend', 'macd', 'adx_di'],
                'entry': 'Pullback alımları',
                'exit': 'Trailing stop',
                'avoid': 'Kısa pozisyon'
            },
            'strong_trend_down': {
                'strategy': 'Defansif',
                'indicators': ['supertrend', 'williams_vix_fix'],
                'entry': 'Dikkatli ve küçük pozisyon',
                'exit': 'Hızlı çıkış',
                'avoid': 'Agresif alım'
            },
            'weak_trend': {
                'strategy': 'Momentum',
                'indicators': ['wavetrend', 'squeeze_momentum'],
                'entry': 'Sinyal onayı bekle',
                'exit': 'Hedef fiyat',
                'avoid': 'Büyük pozisyon'
            },
            'ranging': {
                'strategy': 'Mean reversion',
                'indicators': ['wavetrend', 'squeeze_momentum'],
                'entry': 'Destek/direnç seviyeleri',
                'exit': 'Karşı taraf',
                'avoid': 'Trend takibi'
            },
            'volatile': {
                'strategy': 'Risk yönetimi öncelikli',
                'indicators': ['williams_vix_fix', 'supertrend'],
                'entry': 'Küçük pozisyon',
                'exit': 'Geniş stop',
                'avoid': 'Yüksek kaldıraç'
            },
            'squeeze': {
                'strategy': 'Kırılım bekle',
                'indicators': ['squeeze_momentum', 'adx_di'],
                'entry': 'Kırılım onayı',
                'exit': 'Momentum kaybı',
                'avoid': 'Erken giriş'
            }
        }
        
        return recommendations.get(regime, {})
    
    def analyze_regime_transitions(self, lookback: int = 50) -> Dict:
        """Rejim geçişlerini analiz et"""
        if len(self.regime_history) < 2:
            return {}
        
        recent_history = self.regime_history[-lookback:] if len(self.regime_history) > lookback else self.regime_history
        
        transitions = []
        for i in range(1, len(recent_history)):
            prev_regime = recent_history[i-1]['regime']
            curr_regime = recent_history[i]['regime']
            
            if prev_regime != curr_regime:
                transitions.append({
                    'from': prev_regime,
                    'to': curr_regime,
                    'timestamp': recent_history[i]['timestamp']
                })
        
        # Geçiş matrisi
        transition_matrix = {}
        for t in transitions:
            key = f"{t['from']}_to_{t['to']}"
            transition_matrix[key] = transition_matrix.get(key, 0) + 1
        
        return {
            'transitions': transitions,
            'matrix': transition_matrix,
            'total_transitions': len(transitions)
        }
    
    def get_current_market_conditions(self, symbols: List[str], timeframe: str = '1h') -> Dict:
        """Birden fazla sembol için piyasa koşullarını özetle"""
        conditions = {}
        regime_counts = {}
        
        for symbol in symbols:
            try:
                regime, analysis = self.detect_regime(symbol, timeframe)
                conditions[symbol] = {
                    'regime': regime,
                    'confidence': analysis['confidence']
                }
                
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            except Exception as e:
                logger.error(f"{symbol} rejim analizi hatası: {e}")
        
        # Genel piyasa durumu
        dominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0] if regime_counts else 'unknown'
        
        return {
            'symbol_conditions': conditions,
            'regime_distribution': regime_counts,
            'dominant_regime': dominant_regime,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range hesapla"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI hesapla"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Basit ADX hesaplaması"""
        # Basitleştirilmiş ADX - gerçek hesaplama karmaşık
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.DataFrame({
            'hl': high_low,
            'hc': high_close,
            'lc': low_close
        }).max(axis=1)
        
        # Basit bir trend gücü metriği
        price_change = df['close'].pct_change(period)
        volatility = true_range.rolling(period).mean() / df['close']
        
        # ADX benzeri metrik (0-100 arası)
        adx = (abs(price_change) / volatility).rolling(period).mean() * 100
        adx = adx.clip(0, 100).fillna(25)  # 25 nötr değer
        
        return adx


def main():
    """Test fonksiyonu"""
    detector = MarketRegimeDetector()
    
    # Tek sembol testi
    symbol = 'THYAO'
    regime, analysis = detector.detect_regime(symbol)
    
    print(f"\n{symbol} Piyasa Rejimi: {regime}")
    print(f"Açıklama: {detector.regimes[regime]['description']}")
    print(f"Güven: {analysis['confidence']:.3f}")
    
    print("\nRejim Skorları:")
    for r, score in analysis['scores'].items():
        print(f"  {r}: {score:.3f}")
    
    # Öneriler
    recommendations = detector.get_regime_recommendation(regime)
    print(f"\nÖneriler:")
    for key, value in recommendations.items():
        print(f"  {key}: {value}")
    
    # Çoklu sembol analizi
    symbols = ['THYAO', 'GARAN', 'AKBNK']
    market_conditions = detector.get_current_market_conditions(symbols)
    
    print(f"\n\nGenel Piyasa Durumu:")
    print(f"Baskın Rejim: {market_conditions['dominant_regime']}")
    print(f"\nRejim Dağılımı:")
    for regime, count in market_conditions['regime_distribution'].items():
        print(f"  {regime}: {count} sembol")


if __name__ == "__main__":
    main()