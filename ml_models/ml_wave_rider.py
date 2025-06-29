#!/usr/bin/env python3
"""
ML Wave Rider System - Multi-Timeframe Momentum Dalga Sörfü

Strateji:
1. Haftalık: Büyük trend filtreleme (sadece uptrend)
2. Günlük: Setup tespiti (Bollinger kırılımı, volume spike)
3. 4 Saatlik: Hassas giriş zamanlaması (MACD, momentum)
4. Pyramid: Kazanan pozisyonları büyütme
5. Kademeli çıkış: Momentum kaybında partial profit

Hedef: Aylık %15-20 (Banka faizinin 3-4 katı)
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


@dataclass
class TimeframeSignals:
    """Multi-timeframe sinyalleri"""
    weekly_trend: str  # UP, DOWN, SIDEWAYS
    weekly_rsi: float
    weekly_above_ma: bool
    
    daily_setup: bool
    daily_bb_breakout: bool
    daily_volume_spike: float
    daily_adx: float
    
    h4_momentum: bool
    h4_macd_signal: bool
    h4_stoch_oversold_exit: bool
    
    h1_atr: float
    h1_support: float
    

@dataclass 
class WavePosition:
    """Dalga pozisyon detayları"""
    symbol: str
    entry_date: pd.Timestamp
    initial_size: float  # İlk pozisyon büyüklüğü %
    current_size: float  # Mevcut pozisyon büyüklüğü %
    
    entries: List[Dict]  # Tüm giriş noktaları
    avg_price: float
    highest_price: float
    
    wave_stage: str  # INITIAL, BUILDING, RIDING, EXITING
    momentum_score: float
    

class WaveRider:
    """Multi-timeframe Momentum Wave Rider System"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.total_value = initial_capital
        
        # Sistem parametreleri
        self.initial_position_size = 0.05  # %5 ilk giriş
        self.pyramid_size = 0.05  # %5 ekleme
        self.max_position_size = 0.15  # %15 max
        self.commission = 0.002
        
        # Multi-timeframe parametreler
        self.timeframes = {
            'weekly': {'sma': 20, 'rsi': 14},
            'daily': {'bb_period': 20, 'bb_std': 2, 'adx': 14},
            '4h': {'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9},
            '1h': {'atr_period': 14, 'atr_multiplier': 3.5}  # 2'den 3.5'e çıkarıldı
        }
        
        # Minimum holding period
        self.min_holding_days = 3
        
        # Wave stages thresholds
        self.pyramid_threshold = 0.05  # %5 karda pyramid
        self.partial_exit_threshold = -0.02  # 4H momentum kaybı
        self.full_exit_threshold = -0.05  # Günlük trend kırılması
        
        # Portfolio
        self.positions = {}  # Active wave positions
        self.completed_waves = []  # Tamamlanan dalgalar
        self.transaction_history = []
        self.portfolio_history = []
        
        # Stock universe ve parametreler
        self.load_stock_parameters()
        self.load_timeframe_data()
        
    def load_stock_parameters(self):
        """Hisse parametrelerini yükle"""
        try:
            # RR oranı > 1.4 olan momentum stocks
            self.momentum_stocks = [
                'GARAN', 'PGSUS', 'YKBNK', 'TAVHL', 'THYAO',
                'DOHOL', 'BRSAN', 'ISCTR', 'FROTO', 'AKSEN',
                'ASELS', 'ARCLK', 'SAHOL', 'AEFES', 'TKFEN',
                'ENKAI', 'EKGYO', 'DOAS', 'AKBNK', 'TUPRS'
            ]
            
            # Risk/Reward metrikleri
            strategies_path = 'data/analysis/trading_strategies.csv'
            if os.path.exists(strategies_path):
                strategies_df = pd.read_csv(strategies_path, index_col=0)
                self.stock_rr = strategies_df['risk_reward_ratio'].to_dict()
            else:
                self.stock_rr = {stock: 1.0 for stock in self.momentum_stocks}
                
            # Optimal stop loss ve trailing parametreleri
            self.optimal_params = {}
            advanced_params_path = 'data/analysis/advanced_trading_parameters.csv'
            if os.path.exists(advanced_params_path):
                advanced_df = pd.read_csv(advanced_params_path)
                for _, row in advanced_df.iterrows():
                    symbol = row['symbol']
                    self.optimal_params[symbol] = {
                        'stop_loss': row['optimal_stop_loss'] / 100,  # Yüzde olarak
                        'trailing_stop': row['optimal_trailing_stop'],
                        'take_profit': row['optimal_take_profit'] / 100,
                        'volatility': row['annual_volatility'] / 100
                    }
                logger.info(f"Loaded optimal parameters for {len(self.optimal_params)} stocks")
            
            logger.info(f"Loaded {len(self.momentum_stocks)} momentum stocks")
            
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            
    def load_timeframe_data(self):
        """Multi-timeframe data kontrolü"""
        self.available_timeframes = {
            '1d': True,  # Raw data var
            '4h': False,  # Kontrol edilecek
            '1h': False,  # Kontrol edilecek
            '1w': False   # Hesaplanacak
        }
        
        # 4h ve 1h data kontrolü
        for tf in ['4h', '1h']:
            sample_path = f"data/raw/THYAO_{tf}_raw.csv"
            if os.path.exists(sample_path):
                self.available_timeframes[tf] = True
                logger.info(f"{tf} data available")
            else:
                logger.warning(f"{tf} data NOT available - will use daily data")
                
    def load_multi_timeframe_data(self, symbol: str, date: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """Bir hisse için multi-timeframe data yükle"""
        data = {}
        
        # Günlük data (temel)
        daily_df = self.load_data(symbol, '1d')
        if daily_df.empty:
            return data
            
        # Date'e kadar olan veriyi al
        daily_df = daily_df[daily_df.index <= date]
        data['daily'] = daily_df
        
        # Haftalık data (günlükten hesapla)
        if len(daily_df) > 20:
            weekly_df = daily_df.resample('W').agg({
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            data['weekly'] = weekly_df
            
        # 4H data
        if self.available_timeframes['4h']:
            h4_df = self.load_data(symbol, '4h')
            if not h4_df.empty:
                data['4h'] = h4_df[h4_df.index <= date]
        else:
            # Günlükten simüle et (basitleştirilmiş)
            data['4h'] = daily_df
            
        # 1H data  
        if self.available_timeframes['1h']:
            h1_df = self.load_data(symbol, '1h')
            if not h1_df.empty:
                data['1h'] = h1_df[h1_df.index <= date]
        else:
            # Günlükten simüle et
            data['1h'] = daily_df
            
        return data
        
    def calculate_timeframe_signals(self, symbol: str, date: pd.Timestamp) -> Optional[TimeframeSignals]:
        """Multi-timeframe sinyal hesaplama"""
        data = self.load_multi_timeframe_data(symbol, date)
        
        if not data or 'daily' not in data:
            return None
            
        signals = TimeframeSignals(
            weekly_trend="SIDEWAYS",
            weekly_rsi=50,
            weekly_above_ma=False,
            daily_setup=False,
            daily_bb_breakout=False,
            daily_volume_spike=1.0,
            daily_adx=20,
            h4_momentum=False,
            h4_macd_signal=False,
            h4_stoch_oversold_exit=False,
            h1_atr=0,
            h1_support=0
        )
        
        # HAFTALIK ANALİZ
        if 'weekly' in data and len(data['weekly']) >= 20:
            weekly = data['weekly']
            
            # Trend
            sma20 = weekly['close'].rolling(20).mean()
            current_close = weekly['close'].iloc[-1]
            signals.weekly_above_ma = current_close > sma20.iloc[-1]
            
            # RSI
            signals.weekly_rsi = self._calculate_rsi(weekly['close'])
            
            # Trend yönü
            if signals.weekly_above_ma and signals.weekly_rsi > 50:
                signals.weekly_trend = "UP"
            elif not signals.weekly_above_ma and signals.weekly_rsi < 50:
                signals.weekly_trend = "DOWN"
                
        # GÜNLÜK ANALİZ
        daily = data['daily']
        if len(daily) >= 20:
            # Bollinger Bands
            sma = daily['close'].rolling(20).mean()
            std = daily['close'].rolling(20).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            
            current_close = daily['close'].iloc[-1]
            prev_close = daily['close'].iloc[-2] if len(daily) > 1 else current_close
            
            # Breakout check
            signals.daily_bb_breakout = (prev_close < upper_band.iloc[-2] and 
                                       current_close > upper_band.iloc[-1])
            
            # Volume spike
            avg_volume = daily['volume'].rolling(20).mean().iloc[-1]
            current_volume = daily['volume'].iloc[-1]
            signals.daily_volume_spike = current_volume / avg_volume if avg_volume > 0 else 1
            
            # ADX
            signals.daily_adx = self._calculate_adx(daily)
            
            # Setup var mı? - DAHA ESNEK
            signals.daily_setup = ((signals.daily_bb_breakout or current_close > sma.iloc[-1] * 1.02) and 
                                 signals.daily_volume_spike > 1.5 and
                                 signals.daily_adx > 20)
                                 
        # 4 SAATLIK ANALİZ
        if '4h' in data and len(data['4h']) >= 26:
            h4 = data['4h']
            
            # MACD
            macd_line, signal_line, histogram = self._calculate_macd(h4['close'])
            
            # MACD crossover
            if len(histogram) >= 2:
                signals.h4_macd_signal = (histogram.iloc[-2] < 0 and histogram.iloc[-1] > 0)
                
            # Momentum (basit)
            returns = h4['close'].pct_change(5).iloc[-1]  # 5 period momentum
            signals.h4_momentum = returns > 0.01  # %1+ momentum (daha esnek)
            
            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(h4)
            if len(stoch_k) > 1:
                signals.h4_stoch_oversold_exit = (stoch_k.iloc[-2] < 20 and stoch_k.iloc[-1] > 20)
                
        # 1 SAATLIK ANALİZ (veya günlük kullan)
        if '1h' in data and len(data['1h']) >= 14:
            h1 = data['1h']
        else:
            # 1h yoksa günlük kullan
            h1 = data.get('daily', pd.DataFrame())
            
        if len(h1) >= 14:
            # ATR
            atr_value = self._calculate_atr(h1).iloc[-1]
            # ATR'yi fiyatın yüzdesi olarak normalleştir
            signals.h1_atr = (atr_value / h1['close'].iloc[-1]) * 100  # Yüzde olarak
            
            # Support level (son 20 bar low)
            signals.h1_support = h1['low'].rolling(20).min().iloc[-1] if len(h1) >= 20 else h1['low'].min()
        else:
            # Default değerler
            signals.h1_atr = 2.0  # %2 default ATR
            signals.h1_support = 0
            
        return signals
        
    def scan_wave_opportunities(self, date: pd.Timestamp) -> List[Tuple[str, TimeframeSignals, float]]:
        """Dalga fırsatlarını tara"""
        opportunities = []
        
        for symbol in self.momentum_stocks:
            # Zaten pozisyonda mı?
            if symbol in self.positions:
                continue
                
            # Multi-timeframe sinyalleri hesapla
            signals = self.calculate_timeframe_signals(symbol, date)
            
            if not signals:
                continue
                
            # Wave setup kriterleri - DAHA ESNEK
            # Haftalık uptrend VEYA günlük güçlü setup
            if ((signals.weekly_trend == "UP" or signals.weekly_rsi > 55) and  
                (signals.daily_setup or signals.daily_volume_spike > 1.5)):
                
                # Fırsat skoru
                score = self._calculate_wave_score(signals, symbol)
                opportunities.append((symbol, signals, score))
                
        # Debug için
        if len(opportunities) > 0 and np.random.random() < 0.1:  # %10 olasılıkla log
            logger.info(f"{date.date()} Found {len(opportunities)} opportunities, best: " +
                       f"{opportunities[0][0] if opportunities else 'None'}")
                
        # Skora göre sırala
        opportunities.sort(key=lambda x: x[2], reverse=True)
        
        return opportunities
        
    def _calculate_wave_score(self, signals: TimeframeSignals, symbol: str) -> float:
        """Dalga fırsat skoru"""
        score = 0
        
        # Haftalık güç
        if signals.weekly_rsi > 60:
            score += 2
        elif signals.weekly_rsi > 50:
            score += 1
            
        # Günlük setup kalitesi  
        if signals.daily_volume_spike > 3:
            score += 3
        elif signals.daily_volume_spike > 2:
            score += 2
            
        if signals.daily_adx > 30:
            score += 2
        elif signals.daily_adx > 25:
            score += 1
            
        # 4H momentum
        if signals.h4_macd_signal:
            score += 2
        if signals.h4_stoch_oversold_exit:
            score += 1
            
        # Risk/Reward faktörü
        rr = self.stock_rr.get(symbol, 1.0)
        if rr > 1.5:
            score += 2
        elif rr > 1.2:
            score += 1
            
        return score
        
    def enter_wave(self, symbol: str, date: pd.Timestamp, signals: TimeframeSignals) -> bool:
        """Dalgaya giriş yap"""
        df = self.load_data(symbol, '1d')
        if df.empty or date not in df.index:
            return False
            
        price = df.loc[date, 'close']
        
        # İlk pozisyon büyüklüğü
        position_value = self.total_value * self.initial_position_size
        
        if self.cash < position_value * (1 + self.commission):
            logger.warning(f"Insufficient cash for {symbol}")
            return False
            
        shares = int(position_value / price)
        if shares == 0:
            return False
            
        cost = shares * price * (1 + self.commission)
        
        # Wave position oluştur
        wave_pos = WavePosition(
            symbol=symbol,
            entry_date=date,
            initial_size=self.initial_position_size,
            current_size=self.initial_position_size,
            entries=[{
                'date': date,
                'price': price,
                'shares': shares,
                'size': self.initial_position_size,
                'reason': 'INITIAL_ENTRY'
            }],
            avg_price=price,
            highest_price=price,
            wave_stage='INITIAL',
            momentum_score=10.0
        )
        
        # Optimal stop loss kullan
        if symbol in self.optimal_params:
            optimal_sl = self.optimal_params[symbol]['stop_loss']
            # Minimum %10 stop loss
            stop_loss_pct = max(0.10, optimal_sl)
            logger.info(f"{symbol} using optimal SL: {stop_loss_pct*100:.1f}%")
        else:
            # Optimal veri yoksa ATR kullan (min %10)
            atr_stop_pct = signals.h1_atr * self.timeframes['1h']['atr_multiplier'] / 100
            stop_loss_pct = max(0.10, atr_stop_pct)
            logger.info(f"{symbol} using ATR SL: {stop_loss_pct*100:.1f}%")
            
        self.positions[symbol] = {
            'wave': wave_pos,
            'total_shares': shares,
            'total_cost': cost,
            'current_value': shares * price,
            'atr_stop': price * (1 - stop_loss_pct),
            'trail_stop': None,
            'optimal_trailing': self.optimal_params.get(symbol, {}).get('trailing_stop', 10)  # Default 10
        }
        
        self.cash -= cost
        
        self.transaction_history.append({
            'date': date,
            'type': 'WAVE_ENTRY',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'size_pct': self.initial_position_size * 100,
            'stage': 'INITIAL'
        })
        
        stop_price = self.positions[symbol]['atr_stop']
        stop_pct = (price - stop_price) / price * 100
        
        # Debug info
        optimal_info = ""
        if symbol in self.optimal_params:
            opt = self.optimal_params[symbol]
            optimal_info = f", Optimal SL: {opt['stop_loss']*100:.1f}%"
            
        logger.info(f"{date.date()} WAVE ENTRY {symbol} @ {price:.2f} " +
                   f"({shares} shares, {self.initial_position_size*100:.0f}% position, " +
                   f"Stop: {stop_price:.2f} [-{stop_pct:.1f}%]{optimal_info})")
        
        return True
        
    def manage_wave_positions(self, date: pd.Timestamp):
        """Mevcut dalga pozisyonlarını yönet"""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            wave = pos['wave']
            
            # Güncel fiyat
            df = self.load_data(symbol, '1d')
            if df.empty or date not in df.index:
                continue
                
            current_price = df.loc[date, 'close']
            
            # P&L hesapla
            avg_price = wave.avg_price
            pnl_pct = (current_price - avg_price) / avg_price
            
            # Güncel sinyaller
            signals = self.calculate_timeframe_signals(symbol, date)
            if not signals:
                continue
                
            # Stop loss kontrolü
            if current_price < pos['atr_stop']:
                self._exit_wave(symbol, date, current_price, "STOP_LOSS")
                continue
                
            # Wave stage yönetimi
            if wave.wave_stage == 'INITIAL':
                # Pyramid opportunity?
                if pnl_pct > self.pyramid_threshold and wave.current_size < self.max_position_size:
                    if signals.h4_momentum and signals.daily_volume_spike > 1.5:
                        self._pyramid_position(symbol, date, current_price)
                        wave.wave_stage = 'BUILDING'
                        
            elif wave.wave_stage == 'BUILDING':
                # Tam pozisyona ulaştık mı?
                if wave.current_size >= self.max_position_size * 0.9:
                    wave.wave_stage = 'RIDING'
                    # Optimal trailing stop aktif et
                    trailing_pct = pos.get('optimal_trailing', 10) / 100
                    pos['trail_stop'] = current_price * (1 - trailing_pct)
                    
                # Daha fazla pyramid?
                elif pnl_pct > self.pyramid_threshold * 2 and signals.h4_momentum:
                    self._pyramid_position(symbol, date, current_price)
                    
            elif wave.wave_stage == 'RIDING':
                # Trailing stop güncelle
                if current_price > wave.highest_price:
                    wave.highest_price = current_price
                    # Optimal trailing kullan
                    trailing_pct = pos.get('optimal_trailing', 10) / 100 * 0.7  # %30 daha sıkı
                    pos['trail_stop'] = current_price * (1 - trailing_pct)
                    
                # Trailing stop hit?
                if current_price < pos['trail_stop']:
                    self._exit_wave(symbol, date, current_price, "TRAILING_STOP")
                    continue
                    
                # Momentum kaybı - partial exit
                if not signals.h4_momentum and wave.momentum_score > 5:
                    wave.momentum_score -= 2
                    if wave.momentum_score <= 5:
                        self._partial_exit(symbol, date, current_price, 0.5)
                        wave.wave_stage = 'EXITING'
                        
            elif wave.wave_stage == 'EXITING':
                # Günlük trend kırılması - full exit
                if signals.weekly_trend != "UP" or not signals.daily_setup:
                    self._exit_wave(symbol, date, current_price, "TREND_BREAK")
                    
                # Stop güncelle (daha sıkı)
                if current_price > wave.highest_price:
                    wave.highest_price = current_price
                pos['trail_stop'] = current_price * 0.97  # %3 sıkı trailing
                
                if current_price < pos['trail_stop']:
                    self._exit_wave(symbol, date, current_price, "TIGHT_TRAIL")
                    
    def _pyramid_position(self, symbol: str, date: pd.Timestamp, price: float):
        """Pozisyonu büyüt (pyramid)"""
        pos = self.positions[symbol]
        wave = pos['wave']
        
        # Ekleme büyüklüğü
        add_value = self.total_value * self.pyramid_size
        
        if self.cash < add_value * (1 + self.commission):
            return
            
        shares = int(add_value / price)
        if shares == 0:
            return
            
        cost = shares * price * (1 + self.commission)
        
        # Pozisyonu güncelle
        old_total = pos['total_shares'] * wave.avg_price
        new_total = old_total + (shares * price)
        pos['total_shares'] += shares
        pos['total_cost'] += cost
        wave.avg_price = new_total / pos['total_shares']
        wave.current_size += self.pyramid_size
        
        # Entry log
        wave.entries.append({
            'date': date,
            'price': price,
            'shares': shares,
            'size': self.pyramid_size,
            'reason': 'PYRAMID'
        })
        
        self.cash -= cost
        
        self.transaction_history.append({
            'date': date,
            'type': 'PYRAMID',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'size_pct': wave.current_size * 100,
            'total_shares': pos['total_shares']
        })
        
        logger.info(f"{date.date()} PYRAMID {symbol} @ {price:.2f} " +
                   f"(+{shares} shares, total size: {wave.current_size*100:.0f}%)")
                   
    def _partial_exit(self, symbol: str, date: pd.Timestamp, price: float, exit_pct: float):
        """Kısmi çıkış"""
        pos = self.positions[symbol]
        wave = pos['wave']
        
        exit_shares = int(pos['total_shares'] * exit_pct)
        if exit_shares == 0:
            return
            
        revenue = exit_shares * price * (1 - self.commission)
        partial_cost = pos['total_cost'] * exit_pct
        profit = revenue - partial_cost
        
        # Pozisyonu güncelle
        pos['total_shares'] -= exit_shares
        pos['total_cost'] -= partial_cost
        wave.current_size *= (1 - exit_pct)
        
        self.cash += revenue
        
        self.transaction_history.append({
            'date': date,
            'type': 'PARTIAL_EXIT',
            'symbol': symbol,
            'shares': exit_shares,
            'price': price,
            'profit': profit,
            'exit_pct': exit_pct * 100
        })
        
        pnl_pct = (price - wave.avg_price) / wave.avg_price * 100
        logger.info(f"{date.date()} PARTIAL EXIT {symbol} @ {price:.2f} " +
                   f"({exit_pct*100:.0f}% position, {pnl_pct:+.1f}% gain)")
                   
    def _exit_wave(self, symbol: str, date: pd.Timestamp, price: float, reason: str):
        """Dalgadan tamamen çık"""
        pos = self.positions[symbol]
        wave = pos['wave']
        
        revenue = pos['total_shares'] * price * (1 - self.commission)
        profit = revenue - pos['total_cost']
        pnl_pct = (price - wave.avg_price) / wave.avg_price * 100
        
        # Wave tamamlandı
        self.completed_waves.append({
            'symbol': symbol,
            'entry_date': wave.entry_date,
            'exit_date': date,
            'days_held': (date - wave.entry_date).days,
            'entries': len(wave.entries),
            'max_size': max(e['size'] for e in wave.entries) * 100,
            'avg_price': wave.avg_price,
            'exit_price': price,
            'profit': profit,
            'pnl_pct': pnl_pct,
            'exit_reason': reason
        })
        
        self.cash += revenue
        del self.positions[symbol]
        
        self.transaction_history.append({
            'date': date,
            'type': 'WAVE_EXIT', 
            'symbol': symbol,
            'shares': pos['total_shares'],
            'price': price,
            'profit': profit,
            'reason': reason
        })
        
        logger.info(f"{date.date()} WAVE EXIT {symbol} @ {price:.2f} " +
                   f"({pnl_pct:+.1f}%, Profit: {profit:+.0f} TL, Reason: {reason})")
                   
    def find_new_waves(self, date: pd.Timestamp):
        """Yeni dalga fırsatları bul"""
        # Max pozisyon kontrolü
        if len(self.positions) >= 7:  # Max 7 concurrent waves
            return
            
        # Fırsatları tara
        opportunities = self.scan_wave_opportunities(date)
        
        # En iyi fırsatları al
        for symbol, signals, score in opportunities:
            if score >= 7:  # Minimum dalga skoru - YÜKSELTİLDİ
                # Ek filtreler
                if signals.weekly_rsi > 50 and signals.daily_adx > 25:
                    if self.enter_wave(symbol, date, signals):
                        break  # Günde max 1 yeni dalga
                    
    def execute_daily_trading(self, date: pd.Timestamp):
        """Günlük trading döngüsü"""
        # 1. Mevcut pozisyonları yönet
        self.manage_wave_positions(date)
        
        # 2. Yeni dalgalar ara
        self.find_new_waves(date)
        
        # 3. Portfolio değerini güncelle
        self._update_portfolio_value(date)
        
    def _update_portfolio_value(self, date: pd.Timestamp):
        """Portfolio değerini güncelle"""
        positions_value = 0
        
        for symbol, pos in self.positions.items():
            df = self.load_data(symbol, '1d')
            if not df.empty and date in df.index:
                current_price = df.loc[date, 'close']
                pos['current_value'] = pos['total_shares'] * current_price
                positions_value += pos['current_value']
            else:
                positions_value += pos['current_value']
                
        self.total_value = self.cash + positions_value
        
        # Portfolio history
        self.portfolio_history.append({
            'date': date,
            'cash': self.cash,
            'positions_value': positions_value, 
            'total_value': self.total_value,
            'num_positions': len(self.positions),
            'active_waves': [(s, p['wave'].wave_stage) for s, p in self.positions.items()]
        })
        
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
        
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """ADX hesapla (basitleştirilmiş)"""
        if len(df) < period * 2:
            return 20.0
            
        # True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        
        # Directional Movement
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']
        
        pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        pos_di = 100 * (pos_dm.rolling(period).mean() / atr)
        neg_di = 100 * (neg_dm.rolling(period).mean() / atr)
        
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 20.0
        
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD hesapla"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14, smooth: int = 3):
        """Stochastic hesapla"""
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(smooth).mean()
        
        return k_percent, d_percent
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ATR hesapla"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
        
    def load_data(self, symbol: str, timeframe: str = '1d') -> pd.DataFrame:
        """Timeframe bazlı data yükle"""
        try:
            path = f"data/raw/{symbol}_{timeframe}_raw.csv"
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
        logger.info(f"Starting Wave Rider backtest from {start_date} to {end_date}")
        logger.info(f"Initial capital: {self.initial_capital:,.0f} TL")
        logger.info(f"Strategy: Multi-timeframe momentum wave riding")
        
        # Trading günlerini al
        sample_df = self.load_data(self.momentum_stocks[0], '1d')
        if sample_df.empty:
            logger.error("No data available")
            return
            
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        trading_dates = sample_df[(sample_df.index >= start) & (sample_df.index <= end)].index
        
        logger.info(f"Trading days: {len(trading_dates)}")
        
        # Ana backtest döngüsü
        for i, date in enumerate(trading_dates):
            if i < 30:  # İlk 30 gün veri toplama
                continue
                
            # Günlük trading
            self.execute_daily_trading(date)
            
            # Progress update
            if (i - 30 + 1) % 20 == 0:
                logger.info(f"Day {i-30+1}: Portfolio {self.total_value:,.0f} TL " +
                           f"(+{(self.total_value/self.initial_capital-1)*100:.1f}%), " +
                           f"Active waves: {len(self.positions)}")
                           
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
        print(f"WAVE RIDER SYSTEM RESULTS")
        print(f"{'='*80}")
        print(f"Initial Capital: {initial:,.0f} TL")
        print(f"Final Value: {final:,.0f} TL")
        print(f"Total Return: {total_return:.1f}%")
        
        # Wave analizi
        if self.completed_waves:
            waves_df = pd.DataFrame(self.completed_waves)
            
            print(f"\n{'='*50}")
            print(f"WAVE ANALYSIS:")
            print(f"Total waves completed: {len(waves_df)}")
            print(f"Average wave duration: {waves_df['days_held'].mean():.1f} days")
            print(f"Average profit per wave: {waves_df['profit'].mean():,.0f} TL")
            print(f"Average return per wave: {waves_df['pnl_pct'].mean():.1f}%")
            
            # Win rate
            winners = len(waves_df[waves_df['profit'] > 0])
            win_rate = winners / len(waves_df) * 100
            print(f"Win rate: {win_rate:.1f}%")
            
            # Best waves
            print(f"\n{'='*50}")
            print("TOP 5 WAVES:")
            print(f"{'Symbol':<8} {'Days':>6} {'Entries':>8} {'Max Size':>10} {'Return':>10}")
            print(f"{'-'*50}")
            
            top_waves = waves_df.nlargest(5, 'pnl_pct')
            for _, wave in top_waves.iterrows():
                print(f"{wave['symbol']:<8} {wave['days_held']:>6} {wave['entries']:>8} " +
                     f"{wave['max_size']:>9.0f}% {wave['pnl_pct']:>9.1f}%")
                     
            # Exit reasons
            print(f"\n{'='*50}")
            print("EXIT REASONS:")
            exit_counts = waves_df['exit_reason'].value_counts()
            for reason, count in exit_counts.items():
                pct = count / len(waves_df) * 100
                print(f"{reason}: {count} ({pct:.1f}%)")
                
        # Transaction analizi
        if self.transaction_history:
            trades_df = pd.DataFrame(self.transaction_history)
            
            print(f"\n{'='*50}")
            print("TRANSACTION SUMMARY:")
            print(f"Total transactions: {len(trades_df)}")
            
            trade_types = trades_df['type'].value_counts()
            for ttype, count in trade_types.items():
                print(f"{ttype}: {count}")
        else:
            print(f"\n{'='*50}")
            print("TRANSACTION SUMMARY:")
            print("No transactions executed")
            
        # Pyramid başarısı
        if self.transaction_history:
            trades_df = pd.DataFrame(self.transaction_history)
            pyramids = trades_df[trades_df['type'] == 'PYRAMID']
            if len(pyramids) > 0:
                print(f"\nPyramid trades: {len(pyramids)}")
                print(f"Average pyramid size: {pyramids['size_pct'].mean():.1f}%")
            
        # Aylık getiriler
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly = df.groupby('month')['total_value'].agg(['first', 'last'])
        monthly['return'] = (monthly['last'] - monthly['first']) / monthly['first'] * 100
        
        print(f"\n{'='*50}")
        print("MONTHLY RETURNS:")
        for month, ret in monthly['return'].items():
            status = "✓✓" if ret >= 15 else "✓" if ret >= 8 else "○"
            print(f"{month}: {ret:>6.1f}% {status}")
            
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
        
        # Banka faizi karşılaştırması
        bank_rate = 0.50  # %50 yıllık
        bank_monthly = (1 + bank_rate) ** (1/12) - 1
        print(f"\n{'='*50}")
        print("BANK COMPARISON:")
        print(f"Bank annual rate: {bank_rate*100:.0f}%")
        print(f"Bank monthly equivalent: {bank_monthly*100:.1f}%")
        print(f"Strategy monthly avg: {avg_monthly:.1f}%")
        print(f"Outperformance: {avg_monthly - bank_monthly*100:.1f}% per month")
        
        print(f"{'='*80}")
        
    def plot_results(self):
        """Sonuçları görselleştir"""
        if not self.portfolio_history:
            return
            
        df = pd.DataFrame(self.portfolio_history)
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        # 1. Portfolio value with waves
        ax1 = axes[0]
        ax1.plot(df['date'], df['total_value'], 'b-', linewidth=2, label='Portfolio Value')
        ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5)
        
        # Wave entries and exits
        trades_df = pd.DataFrame(self.transaction_history)
        entries = trades_df[trades_df['type'] == 'WAVE_ENTRY']
        exits = trades_df[trades_df['type'] == 'WAVE_EXIT']
        
        for _, entry in entries.iterrows():
            ax1.axvline(x=entry['date'], color='green', alpha=0.3, linestyle=':')
            
        for _, exit in exits.iterrows():
            ax1.axvline(x=exit['date'], color='red', alpha=0.3, linestyle=':')
            
        ax1.set_title('Portfolio Value with Wave Entries/Exits')
        ax1.set_ylabel('Value (TL)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Active waves and stages
        ax2 = axes[1]
        ax2.plot(df['date'], df['num_positions'], 'purple', linewidth=2)
        ax2.fill_between(df['date'], 0, df['num_positions'], alpha=0.3, color='purple')
        ax2.set_title('Active Wave Positions')
        ax2.set_ylabel('Number of Waves')
        ax2.set_ylim(0, 8)
        ax2.grid(True, alpha=0.3)
        
        # 3. Position sizes
        ax3 = axes[2]
        
        # Track position sizes over time
        for _, trade in trades_df.iterrows():
            if 'size_pct' in trade and pd.notna(trade['size_pct']):
                ax3.scatter(trade['date'], trade['size_pct'], 
                          color='green' if trade['type'] == 'PYRAMID' else 'blue',
                          alpha=0.7, s=50)
                          
        ax3.set_title('Position Sizes (Pyramiding)')
        ax3.set_ylabel('Position Size %')
        ax3.grid(True, alpha=0.3)
        
        # 4. Monthly returns comparison
        ax4 = axes[3]
        
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly = df.groupby('month')['total_value'].agg(['first', 'last'])
        monthly['return'] = (monthly['last'] - monthly['first']) / monthly['first'] * 100
        
        months = [str(m) for m in monthly.index]
        returns = monthly['return'].values
        colors = ['darkgreen' if r >= 15 else 'green' if r >= 8 else 'orange' if r >= 0 else 'red' for r in returns]
        
        bars = ax4.bar(months, returns, color=colors, alpha=0.7)
        ax4.axhline(y=4.17, color='red', linestyle='--', alpha=0.5, label='Bank Rate (50% annual)')
        ax4.axhline(y=15, color='darkgreen', linestyle='--', alpha=0.5, label='Target (15%)')
        
        # Value labels
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ret:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
                    
        ax4.set_title('Monthly Returns vs Bank Rate')
        ax4.set_ylabel('Return %')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Format
        for ax in axes[:-1]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.suptitle('Wave Rider Multi-Timeframe System', fontsize=16)
        plt.tight_layout()
        
        # Save
        plt.savefig('wave_rider_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wave Rider System')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date')
    
    args = parser.parse_args()
    
    # Create and run system
    rider = WaveRider(initial_capital=args.capital)
    
    try:
        rider.run_backtest(args.start_date, args.end_date)
        rider.print_results()
        rider.plot_results()
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()