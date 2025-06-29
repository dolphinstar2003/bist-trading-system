#!/usr/bin/env python3
"""
ML Quality Upgrader System - Kalite Yükseltme Sistemi

Temel Felsefe:
- Boğa piyasasında: Kaliteli hisseleri tut, trendi sonuna kadar kullan
- Ayı piyasasında: Hızla kalitesizleri ele, defansif pozisyon al
- Yatay piyasada: Sadece net kalite farkı varsa rotasyon yap

Her piyasa koşulunda çalışacak şekilde tasarlandı.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')


class MarketRegime(Enum):
    """Piyasa rejimleri"""
    BULL = "BOĞA"
    BEAR = "AYI"
    SIDEWAYS = "YATAY"
    

@dataclass
class PositionQuality:
    """Pozisyon kalite metrikleri"""
    symbol: str
    quality_score: float
    momentum_score: float
    trend_health: float
    risk_score: float
    days_held: int
    current_pnl: float
    potential_score: float  # Gelecek potansiyeli
    

@dataclass
class OpportunityScore:
    """Yeni fırsat skorları"""
    symbol: str
    setup_quality: float
    ml_prediction: float
    technical_strength: float
    relative_performance: float
    risk_reward: float
    total_score: float


class QualityUpgrader:
    """Kalite Yükseltme Sistemi - Her piyasa koşuluna adapte olur"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.total_value = initial_capital
        
        # Sistem parametreleri
        self.position_size_pct = 0.10  # %10 pozisyon büyüklüğü
        self.commission = 0.002
        
        # Piyasa rejimi bazlı parametreler
        self.regime_params = {
            MarketRegime.BULL: {
                'max_positions': 10,
                'quality_threshold': 1.2,  # %20 daha iyi olmalı (daha agresif)
                'min_holding_days': 2,     # 5'ten 2'ye
                'stop_loss_multiplier': 1.5,
                'cash_reserve': 0.0  # Full invested
            },
            MarketRegime.BEAR: {
                'max_positions': 5,
                'quality_threshold': 1.3,  # %30 daha iyi olmalı (1.5'ten düşük)
                'min_holding_days': 1,     # 3'ten 1'e
                'stop_loss_multiplier': 0.8,
                'cash_reserve': 0.4  # %40 nakit tut
            },
            MarketRegime.SIDEWAYS: {
                'max_positions': 8,        # 7'den 8'e
                'quality_threshold': 1.15, # %15 daha iyi olmalı (1.2'den düşük)
                'min_holding_days': 2,     # 4'ten 2'ye
                'stop_loss_multiplier': 1.0,
                'cash_reserve': 0.1  # %10 nakit tut (0.2'den düşük)
            }
        }
        
        # Portföy durumu
        self.positions = {}
        self.blacklist = {}  # Satılan hisseler için bekleme listesi
        self.stop_loss_history = {}  # Stop yiyen hisseler
        
        # Daha agresif parametreler
        self.blacklist_days = 3  # 30 yerine 3 gün
        self.stop_loss_blacklist_days = 7  # 60 yerine 7 gün
        self.max_daily_upgrades = 5  # 2 yerine 5
        self.min_quality_score = 5.0  # 5.5 yerine 5.0
        
        # Performans takibi
        self.transaction_history = []
        self.portfolio_history = []
        self.quality_upgrades = []  # Kalite yükseltme logları
        
        # Piyasa ve analiz verileri
        self.load_market_data()
        self.load_stock_parameters()
        
    def load_market_data(self):
        """BIST100 ve diğer piyasa verilerini yükle"""
        try:
            # BIST100 verisi (örnek olarak XU100.IS)
            self.market_data = self.load_data('XU100', index=True)
            logger.info("Market data loaded successfully")
        except:
            logger.warning("Could not load market data, using mock data")
            self.market_data = None
            
    def load_stock_parameters(self):
        """Hisse parametrelerini ve kalite metriklerini yükle"""
        try:
            # RR oranları ve diğer metrikler
            strategies_path = 'data/analysis/trading_strategies.csv'
            if os.path.exists(strategies_path):
                strategies_df = pd.read_csv(strategies_path, index_col=0)
                self.stock_quality = {}
                
                for symbol, row in strategies_df.iterrows():
                    self.stock_quality[symbol] = {
                        'risk_reward': row['risk_reward_ratio'],
                        'base_quality': self._calculate_base_quality(row),
                        'volatility': row.get('annual_volatility', 40) / 100
                    }
                    
                logger.info(f"Loaded quality metrics for {len(self.stock_quality)} stocks")
                
                # Kalite sınıflarına ayır
                self._categorize_stocks()
            else:
                self.stock_quality = {}
                
        except Exception as e:
            logger.error(f"Error loading stock parameters: {e}")
            self.stock_quality = {}
            
    def _calculate_base_quality(self, row) -> float:
        """Temel kalite skoru hesapla"""
        rr = row['risk_reward_ratio']
        
        # RR bazlı temel skor
        if rr > 1.5:
            base = 8
        elif rr > 1.2:
            base = 6
        elif rr > 1.0:
            base = 4
        else:
            base = 2
            
        # Stop loss ve take profit dengesi
        sl = row['stop_loss']
        tp = row['take_profit']
        
        # Dar stop loss iyidir
        if sl < 15:
            base += 1
        elif sl > 25:
            base -= 1
            
        return min(10, max(0, base))
        
    def _categorize_stocks(self):
        """Hisseleri kalite sınıflarına ayır"""
        self.quality_tiers = {
            'A': [],  # Premium kalite (skor >= 7)
            'B': [],  # Orta kalite (skor 4-7)
            'C': []   # Düşük kalite (skor < 4)
        }
        
        for symbol, metrics in self.stock_quality.items():
            score = metrics['base_quality']
            if score >= 7:
                self.quality_tiers['A'].append(symbol)
            elif score >= 4:
                self.quality_tiers['B'].append(symbol)
            else:
                self.quality_tiers['C'].append(symbol)
                
        logger.info(f"Stock tiers - A: {len(self.quality_tiers['A'])}, " +
                   f"B: {len(self.quality_tiers['B'])}, C: {len(self.quality_tiers['C'])}")
        
    def detect_market_regime(self, date: pd.Timestamp) -> MarketRegime:
        """Piyasa rejimini tespit et - Basitleştirilmiş"""
        # BIST100 verisi yoksa veya yüklenemiyorsa, tarih bazlı basit karar ver
        
        # 2024 yılı için bilinen dönemler (yaklaşık)
        if date.year == 2024:
            if date.month in [1, 2]:
                return MarketRegime.SIDEWAYS
            elif date.month in [3, 4, 5]:
                return MarketRegime.BULL  # Q2 genelde güçlü
            elif date.month in [6, 7, 8]:
                return MarketRegime.SIDEWAYS
            elif date.month in [9, 10, 11]:
                return MarketRegime.BULL  # Q4 başı güçlü
            else:
                return MarketRegime.SIDEWAYS
        
        # 2025 için agresif varsayım
        elif date.year == 2025:
            if date.month in [1, 2, 3]:
                return MarketRegime.BULL  # Yılbaşı rallisi
            elif date.month in [4, 5, 6]:
                return MarketRegime.SIDEWAYS
            else:
                return MarketRegime.SIDEWAYS
                
        return MarketRegime.SIDEWAYS
            
    def calculate_position_quality(self, symbol: str, current_price: float, 
                                 entry_date: pd.Timestamp, current_date: pd.Timestamp) -> PositionQuality:
        """Mevcut pozisyonun kalite skorunu hesapla"""
        pos = self.positions[symbol]
        days_held = (current_date - entry_date).days
        current_pnl = (current_price - pos['entry_price']) / pos['entry_price']
        
        # Trend sağlığı kontrolü
        df = self.load_data(symbol)
        if df.empty or current_date not in df.index:
            trend_health = 5.0
        else:
            recent_data = df[df.index <= current_date].tail(20)
            if len(recent_data) >= 20:
                sma20 = recent_data['close'].mean()
                trend_health = 8.0 if current_price > sma20 else 3.0
            else:
                trend_health = 5.0
                
        # Momentum skoru
        momentum_score = self._calculate_momentum_score(symbol, current_date)
        
        # Risk skoru (volatilite bazlı)
        risk_score = self._calculate_risk_score(symbol, current_date)
        
        # Gelecek potansiyeli (ML tahmin + teknik)
        potential_score = self._calculate_future_potential(symbol, current_date)
        
        # Toplam kalite skoru
        quality_score = (
            self.stock_quality.get(symbol, {}).get('base_quality', 5) * 0.3 +
            trend_health * 0.2 +
            momentum_score * 0.2 +
            (10 - risk_score) * 0.15 +
            potential_score * 0.15
        )
        
        # Kar durumuna göre ayarlama
        if current_pnl > 0.15:  # %15+ kar
            quality_score *= 0.9  # Olgunlaşmış pozisyon
        elif current_pnl < -0.05:  # %5+ zarar
            quality_score *= 0.8  # Zayıf pozisyon
            
        return PositionQuality(
            symbol=symbol,
            quality_score=quality_score,
            momentum_score=momentum_score,
            trend_health=trend_health,
            risk_score=risk_score,
            days_held=days_held,
            current_pnl=current_pnl,
            potential_score=potential_score
        )
        
    def scan_opportunities(self, date: pd.Timestamp, 
                         excluded_symbols: List[str]) -> List[OpportunityScore]:
        """Yeni fırsatları tara ve skorla"""
        opportunities = []
        
        # Tüm hisseleri tara (portföyde olmayanlar)
        all_symbols = list(self.stock_quality.keys())
        
        for symbol in all_symbols:
            # Hariç tutulanları atla
            if symbol in excluded_symbols:
                continue
                
            # Blacklist kontrolü
            if symbol in self.blacklist:
                if (date - self.blacklist[symbol]).days < self.blacklist_days:
                    continue  # 7 gün bekle
                    
            # Stop loss geçmişi kontrolü
            if symbol in self.stop_loss_history:
                if (date - self.stop_loss_history[symbol]).days < self.stop_loss_blacklist_days:
                    continue  # 14 gün bekle
                    
            # Fırsat skorunu hesapla
            opp_score = self._calculate_opportunity_score(symbol, date)
            if opp_score:
                opportunities.append(opp_score)
                
        # En iyi fırsatları sırala
        opportunities.sort(key=lambda x: x.total_score, reverse=True)
        
        return opportunities
        
    def _calculate_opportunity_score(self, symbol: str, date: pd.Timestamp) -> Optional[OpportunityScore]:
        """Bir hisse için fırsat skoru hesapla"""
        df = self.load_data(symbol)
        if df.empty or date not in df.index:
            return None
            
        # Son 60 günlük veri
        hist_data = df[df.index <= date].tail(60)
        if len(hist_data) < 30:
            return None
            
        current_price = hist_data['close'].iloc[-1]
        
        # 1. Setup Kalitesi (Teknik analiz)
        setup_quality = self._evaluate_setup_quality(hist_data)
        
        # 2. ML Tahmin Skoru
        ml_prediction = self._get_ml_prediction(symbol, hist_data)
        
        # 3. Teknik Güç
        technical_strength = self._calculate_technical_strength(hist_data)
        
        # 4. Göreli Performans (Sektör/Endeks)
        relative_perf = self._calculate_relative_performance(symbol, date)
        
        # 5. Risk/Reward
        risk_reward = self.stock_quality.get(symbol, {}).get('risk_reward', 1.0)
        
        # Toplam skor (ağırlıklı ortalama)
        total_score = (
            setup_quality * 0.25 +
            ml_prediction * 0.25 +
            technical_strength * 0.20 +
            relative_perf * 0.15 +
            min(risk_reward * 3, 10) * 0.15  # RR'yi 0-10 skalasına çevir
        )
        
        return OpportunityScore(
            symbol=symbol,
            setup_quality=setup_quality,
            ml_prediction=ml_prediction,
            technical_strength=technical_strength,
            relative_performance=relative_perf,
            risk_reward=risk_reward,
            total_score=total_score
        )
        
    def should_upgrade_position(self, current_pos: PositionQuality, 
                              new_opp: OpportunityScore, 
                              market_regime: MarketRegime) -> bool:
        """Pozisyon yükseltilmeli mi?"""
        params = self.regime_params[market_regime]
        
        # Minimum tutma süresi kontrolü
        if current_pos.days_held < params['min_holding_days']:
            return False
            
        # Kalite farkı eşiği
        quality_ratio = new_opp.total_score / current_pos.quality_score
        
        # Piyasa rejimine göre eşik
        required_ratio = params['quality_threshold']
        
        # Ek filtreler
        if current_pos.current_pnl > 0.20:  # %20+ karda
            required_ratio *= 0.8  # Daha kolay değiştir
        elif current_pos.current_pnl > 0 and current_pos.trend_health > 7:
            required_ratio *= 1.2  # Karda ve trend güçlüyse zor değiştir
            
        # Karar
        if quality_ratio >= required_ratio:
            logger.info(f"Quality upgrade: {current_pos.symbol} (score: {current_pos.quality_score:.1f}) -> " +
                       f"{new_opp.symbol} (score: {new_opp.total_score:.1f}), ratio: {quality_ratio:.2f}")
            return True
            
        return False
        
    def execute_daily_trading(self, date: pd.Timestamp):
        """Günlük trading kararlarını uygula"""
        # Piyasa rejimini tespit et
        market_regime = self.detect_market_regime(date)
        params = self.regime_params[market_regime]
        
        logger.info(f"{date.date()} Market Regime: {market_regime.value}")
        
        # Mevcut pozisyonları değerlendir
        position_qualities = []
        for symbol in list(self.positions.keys()):
            df = self.load_data(symbol)
            if not df.empty and date in df.index:
                current_price = df.loc[date, 'close']
                
                # Stop loss kontrolü
                self._check_stop_loss(symbol, current_price, date, params)
                
                # Hala pozisyondaysa kalite hesapla
                if symbol in self.positions:
                    quality = self.calculate_position_quality(
                        symbol, current_price, 
                        self.positions[symbol]['entry_date'], date
                    )
                    position_qualities.append(quality)
                    
        # En düşük kaliteli pozisyonları bul
        position_qualities.sort(key=lambda x: x.quality_score)
        
        # Yeni fırsatları tara
        excluded = list(self.positions.keys())
        opportunities = self.scan_opportunities(date, excluded)
        
        # Portföy doluluk durumu
        current_positions = len(self.positions)
        max_positions = params['max_positions']
        
        # Nakit rezervi kontrolü
        target_cash_pct = params['cash_reserve']
        current_cash_pct = self.cash / self.total_value
        
        # Kalite yükseltme kararları
        upgrades_made = 0
        max_daily_upgrades = self.max_daily_upgrades  # Günlük maksimum değişim
        
        # Eğer boş slot varsa, en iyi fırsatları al
        while current_positions < max_positions and opportunities:
            if current_cash_pct <= target_cash_pct:
                break  # Nakit rezervini koru
                
            best_opp = opportunities.pop(0)
            if best_opp.total_score >= self.min_quality_score:  # Minimum kalite eşiği
                if self._open_position(best_opp.symbol, date):
                    current_positions += 1
                    upgrades_made += 1
                    
        # Pozisyon yükseltme (düşük kaliteli -> yüksek kaliteli)
        for low_quality_pos in position_qualities:
            if upgrades_made >= max_daily_upgrades:
                break
                
            if not opportunities:
                break
                
            # En iyi fırsatla karşılaştır
            best_opp = opportunities[0]
            
            if self.should_upgrade_position(low_quality_pos, best_opp, market_regime):
                # Eski pozisyonu kapat
                if self._close_position(low_quality_pos.symbol, date, reason="QUALITY_UPGRADE"):
                    # Yeni pozisyon aç
                    if self._open_position(best_opp.symbol, date):
                        # Upgrade logla
                        self.quality_upgrades.append({
                            'date': date,
                            'old_symbol': low_quality_pos.symbol,
                            'old_score': low_quality_pos.quality_score,
                            'new_symbol': best_opp.symbol,
                            'new_score': best_opp.total_score,
                            'regime': market_regime.value
                        })
                        
                        upgrades_made += 1
                        opportunities.pop(0)
                        
        # Portfolio değerini güncelle
        self._update_portfolio_value(date)
        
    def _check_stop_loss(self, symbol: str, current_price: float, 
                        date: pd.Timestamp, params: dict):
        """Stop loss kontrolü"""
        pos = self.positions[symbol]
        entry_price = pos['entry_price']
        
        # Dinamik stop loss
        base_stop = self.stock_quality.get(symbol, {}).get('stop_loss', 0.15)
        adjusted_stop = base_stop * params['stop_loss_multiplier']
        
        # Trailing stop (karda ise)
        pnl_pct = (current_price - entry_price) / entry_price
        if pnl_pct > 0.10:  # %10+ karda
            # Trailing stop aktif
            highest_price = pos.get('highest_price', entry_price)
            if current_price > highest_price:
                pos['highest_price'] = current_price
                
            trailing_stop_price = highest_price * (1 - adjusted_stop * 0.5)
            if current_price < trailing_stop_price:
                self._close_position(symbol, date, reason="TRAILING_STOP")
                
        elif pnl_pct < -adjusted_stop:
            # Normal stop loss
            self._close_position(symbol, date, reason="STOP_LOSS")
            self.stop_loss_history[symbol] = date
            
    def _open_position(self, symbol: str, date: pd.Timestamp) -> bool:
        """Yeni pozisyon aç"""
        df = self.load_data(symbol)
        if df.empty or date not in df.index:
            return False
            
        price = df.loc[date, 'close']
        position_size = self.total_value * self.position_size_pct
        
        if self.cash < position_size * (1 + self.commission):
            return False
            
        shares = int(position_size / price)
        if shares == 0:
            return False
            
        cost = shares * price * (1 + self.commission)
        
        self.positions[symbol] = {
            'shares': shares,
            'entry_price': price,
            'entry_date': date,
            'cost': cost,
            'highest_price': price
        }
        
        self.cash -= cost
        
        self.transaction_history.append({
            'date': date,
            'type': 'BUY',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'cost': cost
        })
        
        quality = self.stock_quality.get(symbol, {})
        logger.info(f"{date.date()} BUY {shares} {symbol} @ {price:.2f} " +
                   f"(Cost: {cost:.0f} TL, RR: {quality.get('risk_reward', 'N/A')})")
        
        return True
        
    def _close_position(self, symbol: str, date: pd.Timestamp, reason: str) -> bool:
        """Pozisyonu kapat"""
        if symbol not in self.positions:
            return False
            
        df = self.load_data(symbol)
        if df.empty or date not in df.index:
            return False
            
        pos = self.positions[symbol]
        price = df.loc[date, 'close']
        revenue = pos['shares'] * price * (1 - self.commission)
        profit = revenue - pos['cost']
        pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
        
        self.cash += revenue
        
        self.transaction_history.append({
            'date': date,
            'type': 'SELL',
            'symbol': symbol,
            'shares': pos['shares'],
            'price': price,
            'profit': profit,
            'reason': reason
        })
        
        # Blacklist'e ekle (stop loss değilse)
        if reason != "STOP_LOSS":
            self.blacklist[symbol] = date
            
        del self.positions[symbol]
        
        logger.info(f"{date.date()} {reason} {symbol} @ {price:.2f} " +
                   f"({pnl_pct:+.1f}%, Profit: {profit:+.0f} TL)")
        
        return True
        
    def _update_portfolio_value(self, date: pd.Timestamp):
        """Portfolio değerini güncelle"""
        positions_value = 0
        
        for symbol, pos in self.positions.items():
            df = self.load_data(symbol)
            if not df.empty and date in df.index:
                current_price = df.loc[date, 'close']
                positions_value += pos['shares'] * current_price
            else:
                positions_value += pos['shares'] * pos['entry_price']
                
        self.total_value = self.cash + positions_value
        
        self.portfolio_history.append({
            'date': date,
            'cash': self.cash,
            'positions_value': positions_value,
            'total_value': self.total_value,
            'num_positions': len(self.positions),
            'regime': self.detect_market_regime(date).value
        })
        
    def _calculate_momentum_score(self, symbol: str, date: pd.Timestamp) -> float:
        """Momentum skoru hesapla"""
        df = self.load_data(symbol)
        if df.empty or date not in df.index:
            return 5.0
            
        hist = df[df.index <= date].tail(20)
        if len(hist) < 10:
            return 5.0
            
        # RSI
        rsi = self._calculate_rsi(hist['close'])
        
        # Price momentum
        returns_5d = (hist['close'].iloc[-1] / hist['close'].iloc[-5] - 1) if len(hist) >= 5 else 0
        
        # Volume momentum
        vol_ratio = hist['volume'].iloc[-5:].mean() / hist['volume'].mean() if len(hist) >= 5 else 1
        
        # Skor hesapla
        score = 5.0
        
        if 40 <= rsi <= 70:
            score += 2
        elif rsi > 70:
            score -= 1
            
        if returns_5d > 0.05:
            score += 2
        elif returns_5d < -0.05:
            score -= 2
            
        if vol_ratio > 1.5:
            score += 1
            
        return min(10, max(0, score))
        
    def _calculate_risk_score(self, symbol: str, date: pd.Timestamp) -> float:
        """Risk skoru hesapla (0-10, yüksek = riskli)"""
        base_volatility = self.stock_quality.get(symbol, {}).get('volatility', 0.4)
        
        # Volatilite bazlı risk
        if base_volatility < 0.3:
            return 3
        elif base_volatility < 0.5:
            return 5
        else:
            return 7
            
    def _calculate_future_potential(self, symbol: str, date: pd.Timestamp) -> float:
        """Gelecek potansiyel skoru"""
        # ML tahmin + teknik setup kalitesi
        df = self.load_data(symbol)
        if df.empty or date not in df.index:
            return 5.0
            
        hist = df[df.index <= date].tail(20)
        
        # Basit potansiyel hesabı
        sma20 = hist['close'].mean()
        current = hist['close'].iloc[-1]
        
        if current > sma20 * 1.02:  # %2 üstünde
            return 7.0
        elif current > sma20:
            return 6.0
        else:
            return 4.0
            
    def _evaluate_setup_quality(self, hist_data: pd.DataFrame) -> float:
        """Setup kalitesini değerlendir"""
        if len(hist_data) < 20:
            return 5.0
            
        close = hist_data['close']
        volume = hist_data['volume']
        
        # Consolidation breakout?
        volatility_20d = close.pct_change().std() * np.sqrt(252)
        volatility_5d = close.tail(5).pct_change().std() * np.sqrt(252)
        
        score = 5.0
        
        # Düşük volatilite sonrası hareket
        if volatility_5d > volatility_20d * 1.5:
            score += 2
            
        # Volume spike
        if volume.iloc[-1] > volume.mean() * 1.5:
            score += 1
            
        # Yeni high
        if close.iloc[-1] >= close.rolling(20).max().iloc[-2]:
            score += 2
            
        return min(10, max(0, score))
        
    def _get_ml_prediction(self, symbol: str, hist_data: pd.DataFrame) -> float:
        """ML tahmin skoru (basitleştirilmiş)"""
        # Gerçek ML modeli yerine basit momentum tahmini
        if len(hist_data) < 20:
            return 5.0
            
        returns = hist_data['close'].pct_change()
        recent_return = returns.tail(5).mean()
        
        if recent_return > 0.02:  # %2+ ortalama getiri
            return 8.0
        elif recent_return > 0:
            return 6.0
        else:
            return 4.0
            
    def _calculate_technical_strength(self, hist_data: pd.DataFrame) -> float:
        """Teknik güç hesapla"""
        if len(hist_data) < 20:
            return 5.0
            
        close = hist_data['close']
        sma20 = close.rolling(20).mean()
        
        # Trend gücü
        if close.iloc[-1] > sma20.iloc[-1]:
            trend_score = 6.0
        else:
            trend_score = 4.0
            
        # Momentum
        momentum = (close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
        if momentum > 0.03:
            trend_score += 2
            
        return min(10, max(0, trend_score))
        
    def _calculate_relative_performance(self, symbol: str, date: pd.Timestamp) -> float:
        """Endekse göre göreli performans"""
        # Basitleştirilmiş - sadece mutlak performans
        df = self.load_data(symbol)
        if df.empty or date not in df.index:
            return 5.0
            
        hist = df[df.index <= date].tail(20)
        if len(hist) < 20:
            return 5.0
            
        perf_20d = (hist['close'].iloc[-1] / hist['close'].iloc[0] - 1)
        
        if perf_20d > 0.10:  # %10+
            return 8.0
        elif perf_20d > 0.05:
            return 6.0
        elif perf_20d > 0:
            return 5.0
        else:
            return 3.0
            
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI hesapla"""
        if len(prices) < period:
            return 50.0
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        if loss.iloc[-1] == 0:
            return 100.0
            
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def load_data(self, symbol: str, index: bool = False) -> pd.DataFrame:
        """Hisse verisini yükle"""
        try:
            if index:
                # Index verisi için özel yol
                path = f"data/raw/XU100_1d_raw.csv"
            else:
                path = f"data/raw/{symbol}_1d_raw.csv"
                
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df
        except:
            pass
        return pd.DataFrame()
        
    def run_backtest(self, start_date: str = '2024-01-01', end_date: str = '2024-12-31'):
        """Backtest çalıştır"""
        logger.info(f"Starting Quality Upgrader backtest from {start_date} to {end_date}")
        logger.info(f"Initial capital: {self.initial_capital:,.0f} TL")
        
        # Trading günlerini al
        sample_symbol = list(self.stock_quality.keys())[0] if self.stock_quality else 'GARAN'
        sample_df = self.load_data(sample_symbol)
        
        if sample_df.empty:
            logger.error("No data available")
            return
            
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        trading_dates = sample_df[(sample_df.index >= start) & (sample_df.index <= end)].index
        
        logger.info(f"Trading days: {len(trading_dates)}")
        
        # Ana backtest döngüsü
        for i, date in enumerate(trading_dates):
            if i < 20:  # İlk 20 gün veri toplama
                continue
                
            # Günlük trading
            self.execute_daily_trading(date)
            
            # Progress update
            if (i - 20 + 1) % 20 == 0:
                regime = self.detect_market_regime(date)
                logger.info(f"Day {i-20+1}: Portfolio {self.total_value:,.0f} TL " +
                           f"(+{(self.total_value/self.initial_capital-1)*100:.1f}%), " +
                           f"Positions: {len(self.positions)}, Regime: {regime.value}")
                           
    def print_results(self):
        """Detaylı sonuçları yazdır"""
        if not self.portfolio_history:
            logger.error("No results to display")
            return
            
        df = pd.DataFrame(self.portfolio_history)
        
        initial = self.initial_capital
        final = self.total_value
        total_return = (final - initial) / initial * 100
        
        print(f"\n{'='*80}")
        print(f"QUALITY UPGRADER SYSTEM RESULTS")
        print(f"{'='*80}")
        print(f"Initial Capital: {initial:,.0f} TL")
        print(f"Final Value: {final:,.0f} TL")
        print(f"Total Return: {total_return:.1f}%")
        
        # İşlem analizi
        trades_df = pd.DataFrame(self.transaction_history)
        buys = trades_df[trades_df['type'] == 'BUY']
        sells = trades_df[trades_df['type'] == 'SELL']
        
        print(f"\n{'='*50}")
        print("TRANSACTION ANALYSIS:")
        print(f"Total buys: {len(buys)}")
        print(f"Total sells: {len(sells)}")
        
        # Satış nedenleri
        if 'reason' in sells.columns:
            sell_reasons = sells['reason'].value_counts()
            print(f"\nSell reasons:")
            for reason, count in sell_reasons.items():
                print(f"  {reason}: {count}")
                
        # Kalite yükseltme analizi
        if self.quality_upgrades:
            print(f"\n{'='*50}")
            print(f"QUALITY UPGRADES: {len(self.quality_upgrades)}")
            
            upgrades_df = pd.DataFrame(self.quality_upgrades)
            avg_score_improvement = (upgrades_df['new_score'] - upgrades_df['old_score']).mean()
            
            print(f"Average score improvement: {avg_score_improvement:.2f}")
            
            # Rejim bazlı dağılım
            regime_counts = upgrades_df['regime'].value_counts()
            print(f"\nUpgrades by regime:")
            for regime, count in regime_counts.items():
                print(f"  {regime}: {count}")
                
            # En iyi upgrade örnekleri
            print(f"\nTop 5 quality upgrades:")
            top_upgrades = upgrades_df.nlargest(5, 'new_score')
            for _, upgrade in top_upgrades.iterrows():
                print(f"  {upgrade['old_symbol']} ({upgrade['old_score']:.1f}) -> " +
                     f"{upgrade['new_symbol']} ({upgrade['new_score']:.1f})")
                     
        # Piyasa rejimi analizi
        regime_counts = df['regime'].value_counts()
        print(f"\n{'='*50}")
        print("MARKET REGIME DISTRIBUTION:")
        for regime, count in regime_counts.items():
            pct = count / len(df) * 100
            print(f"{regime}: {count} days ({pct:.1f}%)")
            
        # Rejim bazlı performans
        print(f"\nPERFORMANCE BY REGIME:")
        for regime in regime_counts.index:
            regime_data = df[df['regime'] == regime]
            if len(regime_data) > 0:
                start_val = regime_data['total_value'].iloc[0]
                end_val = regime_data['total_value'].iloc[-1]
                regime_return = (end_val / start_val - 1) * 100
                print(f"{regime}: {regime_return:+.1f}%")
                
        # Aylık getiriler
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly = df.groupby('month')['total_value'].agg(['first', 'last'])
        monthly['return'] = (monthly['last'] - monthly['first']) / monthly['first'] * 100
        
        print(f"\n{'='*50}")
        print("MONTHLY RETURNS:")
        for month, ret in monthly['return'].items():
            print(f"{month}: {ret:>6.1f}%")
            
        avg_monthly = monthly['return'].mean()
        print(f"\nAverage Monthly: {avg_monthly:.1f}%")
        print(f"Annualized: {avg_monthly * 12:.1f}%")
        
        # Risk metrikleri
        returns = df['total_value'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe = (total_return / 100) / (volatility / 100) * np.sqrt(252)
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        print(f"\n{'='*50}")
        print("RISK METRICS:")
        print(f"Volatility: {volatility:.1f}%")
        print(f"Max Drawdown: {max_drawdown:.1f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        
        print(f"{'='*80}")
        
    def plot_results(self):
        """Sonuçları görselleştir"""
        if not self.portfolio_history:
            return
            
        df = pd.DataFrame(self.portfolio_history)
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # 1. Portfolio değeri ve rejimler
        ax1 = axes[0]
        
        # Rejim renklerini ayarla
        colors = {'BOĞA': 'green', 'AYI': 'red', 'YATAY': 'gray'}
        
        # Portfolio çizgisi
        ax1.plot(df['date'], df['total_value'], 'b-', linewidth=2, label='Portfolio Value')
        
        # Rejim arka planları
        for regime, color in colors.items():
            regime_data = df[df['regime'] == regime]
            if len(regime_data) > 0:
                for i in range(len(regime_data)):
                    ax1.axvspan(regime_data['date'].iloc[i], 
                              regime_data['date'].iloc[i] + pd.Timedelta(days=1),
                              alpha=0.2, color=color)
                              
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5)
        ax1.set_title('Portfolio Value with Market Regimes')
        ax1.set_ylabel('Value (TL)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Pozisyon sayısı ve nakit oranı
        ax2 = axes[1]
        ax2.plot(df['date'], df['num_positions'], 'purple', label='Positions')
        ax2_twin = ax2.twinx()
        cash_pct = df['cash'] / df['total_value'] * 100
        ax2_twin.plot(df['date'], cash_pct, 'green', alpha=0.7, label='Cash %')
        ax2.set_ylabel('Number of Positions')
        ax2_twin.set_ylabel('Cash %')
        ax2.set_title('Position Count and Cash Allocation')
        ax2.grid(True, alpha=0.3)
        
        # 3. Kalite yükseltmeleri
        ax3 = axes[2]
        if self.quality_upgrades:
            upgrades_df = pd.DataFrame(self.quality_upgrades)
            upgrade_dates = pd.to_datetime(upgrades_df['date'])
            
            # Her upgrade için marker
            for date in upgrade_dates:
                ax3.axvline(x=date, color='orange', alpha=0.5, linestyle='--')
                
            # Score improvement
            ax3.scatter(upgrade_dates, 
                       upgrades_df['new_score'] - upgrades_df['old_score'],
                       color='green', s=100, alpha=0.7)
                       
        ax3.set_title('Quality Upgrade Events')
        ax3.set_ylabel('Score Improvement')
        ax3.grid(True, alpha=0.3)
        
        # 4. Drawdown
        ax4 = axes[3]
        returns = df['total_value'].pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        ax4.fill_between(df['date'], 0, drawdown, color='red', alpha=0.3)
        ax4.plot(df['date'], drawdown, 'red', linewidth=1)
        ax4.set_title('Drawdown %')
        ax4.set_ylabel('Drawdown %')
        ax4.grid(True, alpha=0.3)
        
        # Format
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
        plt.suptitle('Quality Upgrader System Performance', fontsize=16)
        plt.tight_layout()
        
        # Save
        plt.savefig('quality_upgrader_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Quality Upgrader System')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date')
    
    args = parser.parse_args()
    
    # Create and run system
    upgrader = QualityUpgrader(initial_capital=args.capital)
    
    try:
        upgrader.run_backtest(args.start_date, args.end_date)
        upgrader.print_results()
        upgrader.plot_results()
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()