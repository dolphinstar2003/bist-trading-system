#!/usr/bin/env python3
"""
Dinamik Risk Yönetim Sistemi
Piyasa koşullarına göre adaptif risk kontrolü
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
from collections import deque
import json
from pathlib import Path
from loguru import logger

# Proje imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.csv_data_manager import CSVDataManager


class DynamicRiskManager:
    """Adaptif risk yönetim sistemi"""
    
    def __init__(self, initial_capital: float = 50000, max_risk_per_trade: float = 0.02):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        
        self.csv_manager = CSVDataManager()
        
        # Risk parametreleri
        self.risk_parameters = {
            'base_risk': max_risk_per_trade,
            'max_portfolio_risk': 0.06,  # Toplam portföy riski
            'max_positions': 10,
            'min_position_size': 0.01,    # Minimum %1
            'max_position_size': 0.10,    # Maximum %10
            'correlation_limit': 0.7,      # Korelasyon limiti
            'max_sector_exposure': 0.3     # Sektör limiti
        }
        
        # Volatilite rejimleri
        self.volatility_regimes = {
            'low': {
                'atr_multiplier': 1.5,
                'position_size_multiplier': 1.0,
                'max_risk': 0.025
            },
            'normal': {
                'atr_multiplier': 2.0,
                'position_size_multiplier': 0.8,
                'max_risk': 0.02
            },
            'high': {
                'atr_multiplier': 2.5,
                'position_size_multiplier': 0.6,
                'max_risk': 0.015
            },
            'extreme': {
                'atr_multiplier': 3.0,
                'position_size_multiplier': 0.3,
                'max_risk': 0.01
            }
        }
        
        # Performans takibi
        self.trade_history = deque(maxlen=1000)
        self.daily_returns = deque(maxlen=252)  # 1 yıllık
        self.drawdown_history = deque(maxlen=100)
        
        # Mevcut pozisyonlar
        self.open_positions = {}
        
    def calculate_position_size(self, signal_data: Dict, symbol: str, 
                              price: float, timeframe: str = '1h') -> Dict:
        """Dinamik pozisyon büyüklüğü hesapla"""
        
        # 1. Temel risk hesabı
        base_risk = self._calculate_base_risk(signal_data)
        
        # 2. Volatilite ayarlaması
        vol_adjustment = self._calculate_volatility_adjustment(symbol, timeframe)
        
        # 3. Sinyal kalitesi ayarlaması
        signal_adjustment = self._calculate_signal_quality_adjustment(signal_data)
        
        # 4. Drawdown ayarlaması
        drawdown_adjustment = self._calculate_drawdown_adjustment()
        
        # 5. Korelasyon ayarlaması
        correlation_adjustment = self._calculate_correlation_adjustment(symbol)
        
        # Final risk yüzdesi
        risk_percentage = (
            base_risk * 
            vol_adjustment * 
            signal_adjustment * 
            drawdown_adjustment * 
            correlation_adjustment
        )
        
        # Min/max limitleri
        risk_percentage = max(
            self.risk_parameters['min_position_size'],
            min(risk_percentage, self.risk_parameters['max_position_size'])
        )
        
        # TL cinsinden pozisyon büyüklüğü
        position_value = self.current_capital * risk_percentage
        
        # Hisse adedi (lot = 100 adet)
        shares = int(position_value / price / 100) * 100
        
        # Stop loss hesabı
        stop_loss = self._calculate_stop_loss(symbol, price, timeframe)
        
        # Take profit hesabı (Risk/Reward oranı)
        take_profit = self._calculate_take_profit(price, stop_loss, signal_data)
        
        return {
            'position_size_pct': risk_percentage,
            'position_value': shares * price,
            'shares': shares,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': abs(price - stop_loss) * shares,
            'risk_reward_ratio': abs(take_profit - price) / abs(price - stop_loss) if stop_loss else 0
        }
    
    def _calculate_base_risk(self, signal_data: Dict) -> float:
        """Temel risk yüzdesini hesapla"""
        confidence = signal_data.get('confidence', 0.5)
        
        # Güven seviyesine göre risk ayarla
        if confidence > 0.8:
            return self.risk_parameters['base_risk'] * 1.2
        elif confidence > 0.6:
            return self.risk_parameters['base_risk']
        elif confidence > 0.4:
            return self.risk_parameters['base_risk'] * 0.8
        else:
            return self.risk_parameters['base_risk'] * 0.5
    
    def _calculate_volatility_adjustment(self, symbol: str, timeframe: str) -> float:
        """Volatilite bazlı ayarlama"""
        try:
            df = self.csv_manager.load_raw_data(symbol, timeframe)
            
            if df is None or len(df) < 20:
                return 0.8  # Veri yoksa muhafazakar
            
            # ATR hesapla
            atr = self._calculate_atr(df)
            atr_pct = atr / df['close'].iloc[-1]
            
            # Volatilite rejimini belirle
            if atr_pct < 0.01:
                regime = 'low'
            elif atr_pct < 0.02:
                regime = 'normal'
            elif atr_pct < 0.03:
                regime = 'high'
            else:
                regime = 'extreme'
            
            return self.volatility_regimes[regime]['position_size_multiplier']
            
        except Exception as e:
            logger.error(f"Volatilite hesaplama hatası: {e}")
            return 0.5  # Hata durumunda çok muhafazakar
    
    def _calculate_signal_quality_adjustment(self, signal_data: Dict) -> float:
        """Sinyal kalitesine göre ayarlama"""
        # Kategori sinyalleri kontrolü
        category_signals = signal_data.get('category_signals', {})
        
        # Kaç kategori aynı yönde sinyal veriyor?
        buy_categories = sum(1 for cat in category_signals.values() 
                           if cat.get('signal') == 'BUY')
        sell_categories = sum(1 for cat in category_signals.values() 
                            if cat.get('signal') == 'SELL')
        
        total_categories = len(category_signals)
        
        if total_categories == 0:
            return 0.5
        
        # Konsensus oranı
        consensus_ratio = max(buy_categories, sell_categories) / total_categories
        
        if consensus_ratio > 0.8:
            return 1.2
        elif consensus_ratio > 0.6:
            return 1.0
        elif consensus_ratio > 0.4:
            return 0.8
        else:
            return 0.6
    
    def _calculate_drawdown_adjustment(self) -> float:
        """Drawdown bazlı ayarlama"""
        if not self.drawdown_history:
            return 1.0
        
        current_drawdown = self.drawdown_history[-1] if self.drawdown_history else 0
        
        # Drawdown arttıkça riski azalt
        if current_drawdown < 0.05:
            return 1.0
        elif current_drawdown < 0.10:
            return 0.8
        elif current_drawdown < 0.15:
            return 0.6
        elif current_drawdown < 0.20:
            return 0.4
        else:
            return 0.2  # %20+ drawdown'da minimal risk
    
    def _calculate_correlation_adjustment(self, symbol: str) -> float:
        """Portföy korelasyonu bazlı ayarlama"""
        if not self.open_positions:
            return 1.0
        
        # Basit sektör bazlı korelasyon kontrolü
        # (Gerçek uygulamada fiyat korelasyonu hesaplanmalı)
        same_sector_positions = sum(1 for s in self.open_positions 
                                   if self._get_sector(s) == self._get_sector(symbol))
        
        if same_sector_positions >= 3:
            return 0.5
        elif same_sector_positions >= 2:
            return 0.7
        elif same_sector_positions >= 1:
            return 0.85
        else:
            return 1.0
    
    def _get_sector(self, symbol: str) -> str:
        """Sembol sektörünü belirle"""
        # Basit sektör tespiti
        banks = ['GARAN', 'AKBNK', 'ISCTR', 'YKBNK', 'VAKBN', 'HALKB', 'SKBNK']
        airlines = ['THYAO', 'PEGASUS']
        steel = ['EREGL', 'KRDMD']
        
        if symbol in banks:
            return 'banking'
        elif symbol in airlines:
            return 'airlines'
        elif symbol in steel:
            return 'steel'
        else:
            return 'other'
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Average True Range hesapla"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1])
            )
        )
        
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
        return atr
    
    def _calculate_stop_loss(self, symbol: str, entry_price: float, 
                           timeframe: str = '1h') -> float:
        """Dinamik stop loss hesapla"""
        try:
            df = self.csv_manager.load_raw_data(symbol, timeframe)
            
            if df is None or len(df) < 20:
                # Veri yoksa sabit %2 stop
                return entry_price * 0.98
            
            # ATR bazlı stop
            atr = self._calculate_atr(df)
            atr_pct = atr / df['close'].iloc[-1]
            
            # Volatilite rejimine göre ATR çarpanı
            if atr_pct < 0.01:
                multiplier = self.volatility_regimes['low']['atr_multiplier']
            elif atr_pct < 0.02:
                multiplier = self.volatility_regimes['normal']['atr_multiplier']
            elif atr_pct < 0.03:
                multiplier = self.volatility_regimes['high']['atr_multiplier']
            else:
                multiplier = self.volatility_regimes['extreme']['atr_multiplier']
            
            stop_distance = atr * multiplier
            stop_loss = entry_price - stop_distance
            
            # Minimum %1, maksimum %5 stop
            min_stop = entry_price * 0.95
            max_stop = entry_price * 0.99
            
            return max(min_stop, min(stop_loss, max_stop))
            
        except Exception as e:
            logger.error(f"Stop loss hesaplama hatası: {e}")
            return entry_price * 0.98
    
    def _calculate_take_profit(self, entry_price: float, stop_loss: float, 
                             signal_data: Dict) -> float:
        """Risk/Reward oranına göre hedef fiyat"""
        stop_distance = abs(entry_price - stop_loss)
        
        # Sinyal gücüne göre R:R oranı
        confidence = signal_data.get('confidence', 0.5)
        
        if confidence > 0.8:
            rr_ratio = 3.0
        elif confidence > 0.6:
            rr_ratio = 2.5
        elif confidence > 0.4:
            rr_ratio = 2.0
        else:
            rr_ratio = 1.5
        
        take_profit = entry_price + (stop_distance * rr_ratio)
        return take_profit
    
    def check_risk_limits(self) -> Dict[str, bool]:
        """Risk limitlerini kontrol et"""
        checks = {}
        
        # Toplam açık pozisyon riski
        total_risk = sum(pos.get('risk_pct', 0) for pos in self.open_positions.values())
        checks['total_risk_ok'] = total_risk <= self.risk_parameters['max_portfolio_risk']
        
        # Pozisyon sayısı
        checks['position_count_ok'] = len(self.open_positions) < self.risk_parameters['max_positions']
        
        # Drawdown kontrolü
        current_drawdown = self.calculate_current_drawdown()
        checks['drawdown_ok'] = current_drawdown < 0.20  # %20 maksimum
        
        # Günlük kayıp limiti
        daily_loss = self.calculate_daily_pnl()
        checks['daily_loss_ok'] = daily_loss > -self.current_capital * 0.02  # Günlük %2
        
        return checks
    
    def update_position(self, symbol: str, position_data: Dict):
        """Pozisyon güncelle"""
        self.open_positions[symbol] = position_data
    
    def close_position(self, symbol: str, exit_price: float, exit_reason: str):
        """Pozisyon kapat"""
        if symbol not in self.open_positions:
            return
        
        position = self.open_positions[symbol]
        entry_price = position['entry_price']
        shares = position['shares']
        
        # PnL hesapla
        pnl = (exit_price - entry_price) * shares
        pnl_pct = pnl / (entry_price * shares)
        
        # İşlem geçmişine ekle
        trade_result = {
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': exit_reason,
            'duration': (datetime.now() - position['entry_time']).total_seconds() / 3600,
            'timestamp': datetime.now().isoformat()
        }
        
        self.trade_history.append(trade_result)
        
        # Sermaye güncelle
        self.current_capital += pnl
        
        # Pozisyonu sil
        del self.open_positions[symbol]
    
    def calculate_current_drawdown(self) -> float:
        """Mevcut drawdown hesapla"""
        if not self.trade_history:
            return 0.0
        
        # Equity curve oluştur
        equity = [self.initial_capital]
        for trade in self.trade_history:
            equity.append(equity[-1] + trade['pnl'])
        
        # Maximum drawdown
        peak = max(equity)
        current = equity[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0
        
        self.drawdown_history.append(drawdown)
        return drawdown
    
    def calculate_daily_pnl(self) -> float:
        """Günlük PnL hesapla"""
        today = datetime.now().date()
        daily_pnl = 0.0
        
        for trade in self.trade_history:
            trade_date = datetime.fromisoformat(trade['timestamp']).date()
            if trade_date == today:
                daily_pnl += trade['pnl']
        
        return daily_pnl
    
    def get_risk_metrics(self) -> Dict:
        """Risk metriklerini al"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'current_drawdown': 0
            }
        
        trades = list(self.trade_history)
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
        
        profit_factor = abs(sum(t['pnl'] for t in wins) / sum(t['pnl'] for t in losses)) if losses else 0
        
        # Sharpe Ratio (basitleştirilmiş)
        returns = [t['pnl_pct'] for t in trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max(self.drawdown_history) if self.drawdown_history else 0,
            'current_drawdown': self.calculate_current_drawdown(),
            'open_positions': len(self.open_positions),
            'current_capital': self.current_capital
        }


def main():
    """Test fonksiyonu"""
    risk_manager = DynamicRiskManager(initial_capital=50000)
    
    # Test sinyali
    signal_data = {
        'signal': 'BUY',
        'confidence': 0.75,
        'category_signals': {
            'momentum': {'signal': 'BUY'},
            'trend': {'signal': 'BUY'},
            'ml_based': {'signal': 'BUY'},
            'volatility': {'signal': 'HOLD'}
        }
    }
    
    # Pozisyon hesapla
    position = risk_manager.calculate_position_size(
        signal_data=signal_data,
        symbol='THYAO',
        price=245.50,
        timeframe='1h'
    )
    
    print("Pozisyon Hesaplaması:")
    print(f"Pozisyon Büyüklüğü: %{position['position_size_pct']*100:.2f}")
    print(f"Pozisyon Değeri: {position['position_value']:,.2f} TL")
    print(f"Hisse Adedi: {position['shares']}")
    print(f"Stop Loss: {position['stop_loss']:.2f} TL")
    print(f"Hedef Fiyat: {position['take_profit']:.2f} TL")
    print(f"Risk Miktarı: {position['risk_amount']:,.2f} TL")
    print(f"Risk/Reward: 1:{position['risk_reward_ratio']:.2f}")
    
    # Risk limitleri
    limits = risk_manager.check_risk_limits()
    print("\nRisk Limitleri:")
    for check, status in limits.items():
        print(f"  {check}: {'✓' if status else '✗'}")
    
    # Risk metrikleri
    metrics = risk_manager.get_risk_metrics()
    print("\nRisk Metrikleri:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()