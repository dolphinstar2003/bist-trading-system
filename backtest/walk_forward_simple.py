#!/usr/bin/env python3
"""
Walk-Forward Simple Backtest
3'lü Onay Sistemi ile dinamik hisse seçimi
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
from loguru import logger
import json
from collections import defaultdict

# Proje imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS


class WalkForwardSimple:
    """Simple 3-confirmation system with walk-forward dynamic selection"""
    
    def __init__(self, initial_capital: float = 50000, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.stop_loss_pct = 0.08  # %8 stop loss
        self.take_profit_pct = 0.20  # %20 take profit
        
        self.csv_manager = CSVDataManager()
        self.positions = {}  # Açık pozisyonlar
        self.trades = []     # Tamamlanmış işlemler
        self.daily_snapshots = []
        
        # Ana indikatörler
        self.primary_indicators = ['supertrend', 'squeeze_momentum']
        # Onay indikatörleri
        self.confirmation_indicators = ['macd', 'wavetrend', 'adx_di', 'lorentzian', 'trend_vanguard']
        
        # Scoring weights
        self.scoring_weights = {
            'primary_alignment': 0.40,     # Ana indikatör uyumu
            'confirmations': 0.25,         # Onay sayısı
            'momentum_strength': 0.15,     # Momentum gücü
            'trend_strength': 0.10,        # Trend gücü
            'volume': 0.10                 # Volume
        }
        
        # İstatistikler
        self.stats = {
            'max_drawdown': 0,
            'peak_capital': initial_capital,
            'total_commission': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'confirmation_performance': {ind: {'trades': 0, 'wins': 0} for ind in self.confirmation_indicators}
        }
        
    def get_timeframe_choice(self) -> str:
        """Kullanıcıdan timeframe seçimi al"""
        print("\n" + "="*60)
        print("WALK-FORWARD SIMPLE BACKTEST")
        print("3'lü Onay Sistemi + Dinamik Hisse Seçimi")
        print("="*60)
        print("Timeframe seçin:")
        print("1. 1d  (Günlük)")
        print("2. 4h  (4 Saatlik)") 
        print("3. 1h  (Saatlik)")
        print("="*60)
        
        while True:
            choice = input("Seçiminiz (1-3): ")
            mapping = {'1': '1d', '2': '4h', '3': '1h'}
            if choice in mapping:
                return mapping[choice]
            print("Hatalı seçim! 1-3 arası seçin.")
    
    def preload_all_data(self, timeframe: str) -> Dict:
        """Tüm sembollerin verilerini önceden yükle"""
        all_data = {}
        
        print("\nVeriler yükleniyor...")
        for symbol in ASSETS:
            try:
                # Fiyat verisi
                price_data = self.csv_manager.load_raw_data(symbol, timeframe)
                if price_data is None or len(price_data) < 100:
                    continue
                
                signals = pd.DataFrame(index=price_data.index)
                
                # Supertrend
                st_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'supertrend')
                if st_data is not None and 'buy_signal' in st_data.columns:
                    signals['supertrend_buy'] = st_data['buy_signal'].astype(int)
                    signals['supertrend_sell'] = st_data.get('sell_signal', 0).astype(int)
                    signals['supertrend'] = signals['supertrend_buy'] - signals['supertrend_sell']
                
                # Squeeze Momentum
                sqz_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'squeeze_momentum')
                if sqz_data is not None and 'sqz_buy_signal' in sqz_data.columns:
                    signals['sqz_buy'] = sqz_data['sqz_buy_signal'].astype(int)
                    signals['sqz_sell'] = sqz_data.get('sqz_sell_signal', 0).astype(int)
                    signals['squeeze_momentum'] = signals['sqz_buy'] - signals['sqz_sell']
                    if 'squeeze_on' in sqz_data.columns:
                        signals['squeeze_active'] = sqz_data['squeeze_on'].astype(int)
                    if 'momentum' in sqz_data.columns:
                        signals['momentum_value'] = sqz_data['momentum']
                
                # MACD
                macd_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'macd')
                if macd_data is not None and 'macd_buy_signal' in macd_data.columns:
                    signals['macd_buy'] = macd_data['macd_buy_signal'].astype(int)
                    signals['macd_sell'] = macd_data.get('macd_sell_signal', 0).astype(int)
                    signals['macd'] = signals['macd_buy'] - signals['macd_sell']
                    if 'macd_histogram' in macd_data.columns:
                        signals['macd_histogram'] = macd_data['macd_histogram']
                
                # WaveTrend
                wt_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'wavetrend')
                if wt_data is not None and 'wt_buy_signal' in wt_data.columns:
                    signals['wt_buy'] = wt_data['wt_buy_signal'].astype(int)
                    signals['wt_sell'] = wt_data.get('wt_sell_signal', 0).astype(int)
                    signals['wavetrend'] = signals['wt_buy'] - signals['wt_sell']
                    if 'wt1' in wt_data.columns:
                        signals['wt_value'] = wt_data['wt1']
                
                # ADX/DI
                adx_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'adx_di')
                if adx_data is not None and 'adx_buy_signal' in adx_data.columns:
                    signals['adx_buy'] = adx_data['adx_buy_signal'].astype(int)
                    signals['adx_sell'] = adx_data.get('adx_sell_signal', 0).astype(int)
                    signals['adx_di'] = signals['adx_buy'] - signals['adx_sell']
                    if 'adx' in adx_data.columns:
                        signals['adx_value'] = adx_data['adx']
                
                # Lorentzian
                lor_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'lorentzian')
                if lor_data is not None and 'signal' in lor_data.columns:
                    signals['lorentzian'] = lor_data['signal']
                    if 'confidence' in lor_data.columns:
                        signals['lor_confidence'] = lor_data['confidence']
                
                # Trend Vanguard
                tv_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'trend_vanguard')
                if tv_data is not None and 'signal' in tv_data.columns:
                    signals['trend_vanguard'] = tv_data['signal']
                    if 'strength' in tv_data.columns:
                        signals['tv_strength'] = tv_data['strength']
                    if 'confidence' in tv_data.columns:
                        signals['tv_confidence'] = tv_data['confidence']
                    if 'market_regime' in tv_data.columns:
                        signals['tv_regime'] = tv_data['market_regime']
                
                # Technical indicators
                price_data['returns'] = price_data['close'].pct_change()
                price_data['volatility'] = price_data['returns'].rolling(20).std()
                price_data['volume_ratio'] = price_data['volume'] / price_data['volume'].rolling(20).mean()
                price_data['rsi'] = self.calculate_rsi(price_data['close'])
                
                # SMA trends
                price_data['sma20'] = price_data['close'].rolling(20).mean()
                price_data['sma50'] = price_data['close'].rolling(50).mean()
                
                if not signals.empty:
                    all_data[symbol] = {
                        'price': price_data,
                        'signals': signals
                    }
                    
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
                continue
        
        print(f"{len(all_data)} sembol yüklendi.")
        return all_data
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI hesapla"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def check_entry_conditions(self, signals: pd.Series) -> Tuple[str, List[str]]:
        """Giriş koşullarını kontrol et ve onay veren indikatörleri döndür"""
        # Ana koşul: Supertrend + Squeeze Momentum ikisi de BUY
        primary_buy = (
            signals.get('supertrend', 0) > 0 and 
            signals.get('squeeze_momentum', 0) > 0
        )
        
        primary_sell = (
            signals.get('supertrend', 0) < 0 and 
            signals.get('squeeze_momentum', 0) < 0
        )
        
        if not primary_buy and not primary_sell:
            return 'HOLD', []
        
        confirmations = []
        
        # BUY için kontrol
        if primary_buy:
            # Hiçbir onay indikatörü SELL vermemeli
            no_sell_signals = all(
                signals.get(ind, 0) >= 0 
                for ind in self.confirmation_indicators
            )
            
            # Onay veren indikatörleri topla
            for ind in self.confirmation_indicators:
                if signals.get(ind, 0) > 0:
                    confirmations.append(ind)
            
            if no_sell_signals and len(confirmations) >= 1:
                return 'BUY', confirmations
            
        # SELL için kontrol  
        elif primary_sell:
            # Hiçbir onay indikatörü BUY vermemeli
            no_buy_signals = all(
                signals.get(ind, 0) <= 0 
                for ind in self.confirmation_indicators
            )
            
            # SELL onaylarını topla
            for ind in self.confirmation_indicators:
                if signals.get(ind, 0) < 0:
                    confirmations.append(ind)
            
            if no_buy_signals and len(confirmations) >= 1:
                return 'SELL', confirmations
        
        return 'HOLD', []
    
    def score_symbol_simple(self, symbol: str, data: Dict, current_idx: int) -> Tuple[float, Dict]:
        """Simple sistem ile sembol skorla"""
        try:
            signals_df = data['signals']
            price_df = data['price']
            
            if current_idx >= len(signals_df):
                return -999, {}
            
            current_signals = signals_df.iloc[current_idx]
            signal_type, confirmations = self.check_entry_conditions(current_signals)
            
            if signal_type != 'BUY':
                return -999, {}
            
            score = 0
            details = {
                'signal_type': signal_type,
                'confirmations': confirmations,
                'confirmation_count': len(confirmations)
            }
            
            # 1. Primary alignment (0.40)
            # Both primary indicators must be positive
            primary_strength = 0
            if current_signals.get('supertrend', 0) > 0:
                primary_strength += 0.5
            if current_signals.get('squeeze_momentum', 0) > 0:
                primary_strength += 0.5
                
                # Momentum value bonus
                if 'momentum_value' in current_signals:
                    mom_val = current_signals['momentum_value']
                    if not pd.isna(mom_val) and mom_val > 0:
                        primary_strength += min(mom_val / 200, 0.2)
            
            score += primary_strength * self.scoring_weights['primary_alignment']
            details['primary_strength'] = primary_strength
            
            # 2. Confirmations (0.25)
            # More confirmations = better
            confirmation_score = len(confirmations) / len(self.confirmation_indicators)
            
            # Extra weight for strong confirmations
            if 'lorentzian' in confirmations and 'lor_confidence' in current_signals:
                confidence = current_signals['lor_confidence']
                if not pd.isna(confidence):
                    confirmation_score += confidence * 0.2
            
            # Trend Vanguard bonus
            if 'trend_vanguard' in confirmations:
                if 'tv_strength' in current_signals:
                    strength = current_signals['tv_strength']
                    if not pd.isna(strength):
                        confirmation_score += strength * 0.15
                
                # Market regime bonus
                if 'tv_regime' in current_signals:
                    regime = current_signals['tv_regime']
                    if regime == 'bullish':
                        confirmation_score += 0.1
            
            score += confirmation_score * self.scoring_weights['confirmations']
            details['confirmation_score'] = confirmation_score
            
            # 3. Momentum strength (0.15)
            momentum_score = 0
            if 'momentum_value' in current_signals:
                momentum = current_signals['momentum_value']
                if not pd.isna(momentum):
                    # Normalize momentum
                    momentum_score = np.tanh(momentum / 100)
            
            if 'macd_histogram' in current_signals:
                macd_hist = current_signals['macd_histogram']
                if not pd.isna(macd_hist) and macd_hist > 0:
                    momentum_score += 0.3
            
            score += momentum_score * self.scoring_weights['momentum_strength']
            details['momentum_score'] = momentum_score
            
            # 4. Trend strength (0.10)
            trend_score = 0
            if current_idx >= 50 and current_idx < len(price_df):
                # Price above moving averages
                current_price = price_df['close'].iloc[current_idx]
                
                if 'sma20' in price_df.columns:
                    sma20 = price_df['sma20'].iloc[current_idx]
                    if not pd.isna(sma20) and current_price > sma20:
                        trend_score += 0.5
                
                if 'sma50' in price_df.columns:
                    sma50 = price_df['sma50'].iloc[current_idx]
                    if not pd.isna(sma50) and current_price > sma50:
                        trend_score += 0.5
                
                # Recent price trend
                price_20_ago = price_df['close'].iloc[current_idx-20]
                recent_trend = (current_price - price_20_ago) / price_20_ago
                if recent_trend > 0:
                    trend_score += min(recent_trend * 5, 0.3)
            
            score += trend_score * self.scoring_weights['trend_strength']
            details['trend_score'] = trend_score
            
            # 5. Volume (0.10)
            volume_score = 0
            if current_idx < len(price_df) and 'volume_ratio' in price_df.columns:
                vol_ratio = price_df['volume_ratio'].iloc[current_idx]
                if not pd.isna(vol_ratio):
                    # Higher volume is better (up to 2x normal)
                    volume_score = min(vol_ratio / 2, 1)
            
            score += volume_score * self.scoring_weights['volume']
            details['volume_score'] = volume_score
            
            # Additional factors
            
            # Squeeze bonus
            if 'squeeze_active' in current_signals:
                if current_signals['squeeze_active']:
                    score += 0.05
                    details['squeeze_bonus'] = True
            
            # RSI adjustment
            if current_idx < len(price_df) and 'rsi' in price_df.columns:
                rsi = price_df['rsi'].iloc[current_idx]
                if not pd.isna(rsi):
                    if rsi < 30:
                        score += 0.05  # Oversold bonus
                    elif rsi > 70:
                        score -= 0.10  # Overbought penalty
                    details['rsi'] = rsi
            
            # ADX strength bonus
            if 'adx_value' in current_signals:
                adx = current_signals['adx_value']
                if not pd.isna(adx) and adx > 25:
                    score += 0.05  # Strong trend bonus
                    details['adx_bonus'] = True
            
            return score, details
            
        except Exception as e:
            logger.error(f"Error scoring {symbol}: {e}")
            return -999, {}
    
    def select_best_opportunities_simple(self, all_data: Dict, current_time: pd.Timestamp, 
                                       max_selections: int = 5) -> List[Tuple[str, float, Dict]]:
        """En iyi fırsatları seç - Simple sistem"""
        opportunities = []
        
        for symbol, data in all_data.items():
            if symbol in self.positions:
                continue
            
            price_df = data['price']
            if current_time not in price_df.index:
                continue
            
            current_idx = price_df.index.get_loc(current_time)
            score, details = self.score_symbol_simple(symbol, data, current_idx)
            
            if score > 0.3:  # Minimum threshold
                current_price = price_df['close'].iloc[current_idx]
                opportunities.append((symbol, score, details, current_price))
        
        # Sort by score
        opportunities.sort(key=lambda x: x[1], reverse=True)
        
        return [(sym, score, details) for sym, score, details, _ in opportunities[:max_selections]]
    
    def check_exit_conditions_simple(self, symbol: str, data: Dict, current_idx: int, 
                                   position: Dict, current_price: float) -> Tuple[bool, str]:
        """Çıkış koşulları - Simple sistem"""
        entry_price = position['entry_price']
        pnl_pct = (current_price - entry_price) / entry_price
        
        # 1. Stop Loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, 'STOP_LOSS'
        
        # 2. Take Profit
        if pnl_pct >= self.take_profit_pct:
            return True, 'TAKE_PROFIT'
        
        # 3. Signal-based exit
        signals_df = data['signals']
        if current_idx < len(signals_df):
            current_signals = signals_df.iloc[current_idx]
            signal_type, _ = self.check_entry_conditions(current_signals)
            
            if signal_type == 'SELL':
                return True, 'SIGNAL'
        
        # 4. Primary indicators divergence
        if current_idx < len(signals_df):
            # If either primary indicator turns negative
            if (current_signals.get('supertrend', 0) < 0 or 
                current_signals.get('squeeze_momentum', 0) < 0):
                # But only if we're in profit or held for a while
                bars_held = current_idx - position.get('entry_idx', current_idx)
                if pnl_pct > 0.05 or bars_held > 10:
                    return True, 'PRIMARY_EXIT'
        
        # 5. Trailing stop for big winners
        if pnl_pct > 0.15:  # 15%+ profit
            # Tighten stop to 5%
            if 'momentum_value' in signals_df.columns:
                momentum = signals_df['momentum_value'].iloc[current_idx]
                if not pd.isna(momentum) and momentum < -50:
                    return True, 'TRAILING_STOP'
        
        return False, ''
    
    def calculate_position_size_simple(self, price: float, score: float, details: Dict) -> int:
        """Score ve onay sayısına göre pozisyon boyutu"""
        available_cash = self.current_capital
        
        # Base size
        base_position_value = min(
            available_cash * 0.95,
            (self.initial_capital + self.current_capital) / (2 * self.max_positions)
        )
        
        # Score multiplier (0.7x to 1.3x)
        score_multiplier = 0.7 + (min(score, 1) * 0.6)
        
        # Confirmation bonus
        confirmation_count = details.get('confirmation_count', 0)
        if confirmation_count >= 3:
            score_multiplier *= 1.15
        elif confirmation_count >= 2:
            score_multiplier *= 1.05
        
        position_value = base_position_value * score_multiplier
        
        # Commission
        commission_rate = 0.002
        position_value_after_commission = position_value / (1 + commission_rate)
        
        shares = int(position_value_after_commission / price)
        
        if shares < 1 or (shares * price * (1 + commission_rate)) > available_cash:
            return 0
        
        return shares
    
    def run_walk_forward(self, timeframe: str):
        """Walk-forward backtest çalıştır - Simple sistem"""
        logger.info(f"\nWalk-Forward Simple Backtest başlıyor - Timeframe: {timeframe}")
        logger.info(f"Başlangıç Sermayesi: {self.initial_capital:,.0f} TL")
        logger.info(f"3'lü Onay Sistemi\n")
        
        # Load all data
        all_data = self.preload_all_data(timeframe)
        if not all_data:
            logger.error("No data loaded!")
            return
        
        # Get timeline
        all_dates = set()
        for symbol, data in all_data.items():
            all_dates.update(data['price'].index)
        timeline = sorted(list(all_dates))
        
        start_idx = 100
        logger.info(f"Test period: {timeline[start_idx]} to {timeline[-1]}")
        logger.info(f"Total bars: {len(timeline) - start_idx}\n")
        
        # Main loop
        for i, current_time in enumerate(timeline[start_idx:], start_idx):
            
            # Check exits
            positions_to_close = []
            
            for symbol, position in self.positions.items():
                if symbol not in all_data:
                    continue
                
                data = all_data[symbol]
                price_df = data['price']
                
                if current_time not in price_df.index:
                    continue
                
                current_idx = price_df.index.get_loc(current_time)
                current_price = price_df['close'].iloc[current_idx]
                
                should_exit, exit_reason = self.check_exit_conditions_simple(
                    symbol, data, current_idx, position, current_price
                )
                
                if should_exit:
                    positions_to_close.append((symbol, current_price, exit_reason))
            
            # Close positions
            for symbol, exit_price, exit_reason in positions_to_close:
                position = self.positions[symbol]
                commission = exit_price * position['shares'] * 0.002
                exit_value = (exit_price * position['shares']) - commission
                pnl = exit_value - position['cost']
                
                trade = {
                    'symbol': symbol,
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl / position['cost'],
                    'exit_reason': exit_reason,
                    'entry_score': position.get('entry_score', 0),
                    'confirmations': position.get('confirmations', [])
                }
                
                self.trades.append(trade)
                self.current_capital += exit_value
                del self.positions[symbol]
                
                if pnl > 0:
                    self.stats['winning_trades'] += 1
                else:
                    self.stats['losing_trades'] += 1
                
                # Track confirmation performance
                for conf in position.get('confirmations', []):
                    self.stats['confirmation_performance'][conf]['trades'] += 1
                    if pnl > 0:
                        self.stats['confirmation_performance'][conf]['wins'] += 1
                
                logger.debug(f"{current_time} - {symbol} {exit_reason}: "
                           f"{pnl/position['cost']*100:.1f}%")
            
            # Find new opportunities
            if len(self.positions) < self.max_positions:
                opportunities = self.select_best_opportunities_simple(
                    all_data, current_time,
                    max_selections=self.max_positions - len(self.positions)
                )
                
                for symbol, score, details in opportunities:
                    if len(self.positions) >= self.max_positions:
                        break
                    
                    data = all_data[symbol]
                    price_df = data['price']
                    current_idx = price_df.index.get_loc(current_time)
                    current_price = price_df['close'].iloc[current_idx]
                    
                    shares = self.calculate_position_size_simple(current_price, score, details)
                    if shares > 0:
                        commission = shares * current_price * 0.002
                        cost = (shares * current_price) + commission
                        
                        if cost <= self.current_capital * 0.95:
                            self.positions[symbol] = {
                                'entry_time': current_time,
                                'entry_idx': current_idx,
                                'entry_price': current_price,
                                'shares': shares,
                                'cost': cost,
                                'entry_score': score,
                                'confirmations': details.get('confirmations', []),
                                'details': details
                            }
                            self.current_capital -= cost
                            self.stats['total_commission'] += commission
                            
                            logger.debug(f"{current_time} - BUY {symbol}: "
                                       f"Score: {score:.2f}, "
                                       f"Confirmations: {details.get('confirmations', [])}")
            
            # Portfolio snapshot
            if i % 10 == 0:
                portfolio_value = self.current_capital
                for symbol, pos in self.positions.items():
                    if symbol in all_data and current_time in all_data[symbol]['price'].index:
                        idx = all_data[symbol]['price'].index.get_loc(current_time)
                        current_price = all_data[symbol]['price']['close'].iloc[idx]
                        portfolio_value += current_price * pos['shares']
                
                self.daily_snapshots.append({
                    'time': current_time,
                    'portfolio_value': portfolio_value,
                    'cash': self.current_capital,
                    'positions': len(self.positions)
                })
                
                # Update drawdown
                if portfolio_value > self.stats['peak_capital']:
                    self.stats['peak_capital'] = portfolio_value
                else:
                    drawdown = (self.stats['peak_capital'] - portfolio_value) / self.stats['peak_capital']
                    self.stats['max_drawdown'] = max(self.stats['max_drawdown'], drawdown)
        
        # Close remaining positions
        final_time = timeline[-1]
        for symbol, position in list(self.positions.items()):
            if symbol in all_data and final_time in all_data[symbol]['price'].index:
                idx = all_data[symbol]['price'].index.get_loc(final_time)
                final_price = all_data[symbol]['price']['close'].iloc[idx]
                
                commission = final_price * position['shares'] * 0.002
                exit_value = (final_price * position['shares']) - commission
                pnl = exit_value - position['cost']
                
                trade = {
                    'symbol': symbol,
                    'entry_time': position['entry_time'],
                    'exit_time': final_time,
                    'entry_price': position['entry_price'],
                    'exit_price': final_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl / position['cost'],
                    'exit_reason': 'END_TEST',
                    'confirmations': position.get('confirmations', [])
                }
                
                self.trades.append(trade)
                self.current_capital += exit_value
                
                if pnl > 0:
                    self.stats['winning_trades'] += 1
                else:
                    self.stats['losing_trades'] += 1
        
        # Results
        self.print_results(timeframe)
        self.save_results(timeframe)
    
    def print_results(self, timeframe: str):
        """Sonuçları yazdır"""
        print("\n" + "="*80)
        print(f"WALK-FORWARD SIMPLE (3 ONAY) SONUÇLARI - {timeframe}")
        print("="*80)
        
        final_value = self.current_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        print(f"\nPORTFOLIO ÖZET:")
        print(f"Başlangıç Sermayesi: {self.initial_capital:,.0f} TL")
        print(f"Final Sermaye: {final_value:,.0f} TL")
        print(f"Toplam Getiri: {total_return*100:.1f}%")
        
        years = 3
        annual_return = (((final_value/self.initial_capital)**(1/years))-1)*100
        print(f"Yıllık Getiri: {annual_return:.1f}%")
        print(f"Maksimum Drawdown: {self.stats['max_drawdown']*100:.1f}%")
        
        # Trade statistics
        total_trades = len(self.trades)
        if total_trades > 0:
            win_rate = self.stats['winning_trades'] / total_trades
            
            print(f"\nİŞLEM İSTATİSTİKLERİ:")
            print(f"Toplam İşlem: {total_trades}")
            print(f"Başarılı: {self.stats['winning_trades']} ({win_rate*100:.1f}%)")
            print(f"Başarısız: {self.stats['losing_trades']}")
            
            # Confirmation indicator performance
            print(f"\nONAY İNDİKATÖR PERFORMANSI:")
            for ind, stats in self.stats['confirmation_performance'].items():
                if stats['trades'] > 0:
                    ind_win_rate = stats['wins'] / stats['trades']
                    print(f"{ind}: {stats['trades']} işlem, {ind_win_rate*100:.1f}% başarı")
            
            # Exit reasons
            exit_reasons = defaultdict(int)
            for trade in self.trades:
                exit_reasons[trade['exit_reason']] += 1
            
            print(f"\nÇIKIŞ NEDENLERİ:")
            for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"{reason}: {count} ({count/total_trades*100:.1f}%)")
            
            # Confirmation distribution
            conf_dist = defaultdict(int)
            for trade in self.trades:
                conf_count = len(trade.get('confirmations', []))
                conf_dist[conf_count] += 1
            
            print(f"\nONAY SAYISI DAĞILIMI:")
            for conf_count, trades in sorted(conf_dist.items()):
                if trades > 0:
                    print(f"{conf_count} onay: {trades} işlem")
            
            # Top performers
            symbol_performance = defaultdict(lambda: {'pnl': 0, 'trades': 0})
            for trade in self.trades:
                symbol_performance[trade['symbol']]['pnl'] += trade['pnl']
                symbol_performance[trade['symbol']]['trades'] += 1
            
            print(f"\nEN KARLI 10 HİSSE:")
            sorted_symbols = sorted(symbol_performance.items(), 
                                  key=lambda x: x[1]['pnl'], reverse=True)
            
            for symbol, perf in sorted_symbols[:10]:
                if perf['pnl'] > 0:
                    avg_pnl = perf['pnl'] / perf['trades']
                    print(f"{symbol}: {perf['trades']} işlem, "
                          f"{perf['pnl']:,.2f} TL toplam, "
                          f"{avg_pnl:,.2f} TL ortalama")
    
    def save_results(self, timeframe: str):
        """Sonuçları kaydet"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        summary = {
            'strategy': 'Walk-Forward Simple (3 Confirmation)',
            'timeframe': timeframe,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'max_drawdown': self.stats['max_drawdown'],
            'total_trades': len(self.trades),
            'win_rate': self.stats['winning_trades'] / len(self.trades) if self.trades else 0,
            'confirmation_performance': self.stats['confirmation_performance'],
            'timestamp': timestamp
        }
        
        summary_file = Path(f"backtest/walk_forward_simple_{timeframe}_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            # Convert confirmations list to string
            if 'confirmations' in trades_df.columns:
                trades_df['confirmations'] = trades_df['confirmations'].apply(
                    lambda x: ','.join(x) if isinstance(x, list) else x
                )
            trades_file = Path(f"backtest/walk_forward_simple_trades_{timeframe}_{timestamp}.csv")
            trades_df.to_csv(trades_file, index=False)
        
        logger.info(f"\nSonuçlar kaydedildi: {summary_file}")


def main():
    backtest = WalkForwardSimple(initial_capital=50000, max_positions=10)
    timeframe = backtest.get_timeframe_choice()
    backtest.run_walk_forward(timeframe)


if __name__ == "__main__":
    main()