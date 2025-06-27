#!/usr/bin/env python3
"""
Walk-Forward Sequential (Sıralı) Backtest
4 Fazlı sistem ile dinamik hisse seçimi
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


class WalkForwardSequential:
    """4 Fazlı sıralı sistem ile walk-forward backtest"""
    
    def __init__(self, initial_capital: float = 50000, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.stop_loss_pct = 0.08  # %8 stop loss
        self.take_profit_pct = 0.25  # %25 take profit
        
        self.csv_manager = CSVDataManager()
        self.positions = {}  # Açık pozisyonlar
        self.trades = []     # Tamamlanmış işlemler
        self.daily_snapshots = []
        
        # İndikatör fazları
        self.phases = {
            'phase1': ['volume_ratio', 'pattern_signal'],  # Volume + Patterns
            'phase2': ['wavetrend', 'momentum'],           # WaveTrend + Momentum  
            'phase3': ['macd', 'squeeze_momentum'],        # MACD + Squeeze
            'phase4': ['adx_di', 'supertrend']            # ADX + Supertrend
        }
        
        # Faz ağırlıkları skorlama için
        self.phase_weights = {
            'phase1': 0.15,  # Volume/Pattern
            'phase2': 0.25,  # WaveTrend/Momentum
            'phase3': 0.30,  # MACD/Squeeze
            'phase4': 0.30   # ADX/Supertrend
        }
        
        # İstatistikler
        self.stats = {
            'max_drawdown': 0,
            'peak_capital': initial_capital,
            'total_commission': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'phase_performance': {p: {'trades': 0, 'wins': 0} for p in self.phases}
        }
        
    def get_timeframe_choice(self) -> str:
        """Kullanıcıdan timeframe seçimi al"""
        print("\n" + "="*60)
        print("WALK-FORWARD SEQUENTIAL (SIRALI) BACKTEST")
        print("4 Fazlı Sistem + Dinamik Hisse Seçimi")
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
                
                # İndikatör verileri
                indicators = {}
                
                # Volume ratio
                if 'volume' in price_data.columns:
                    indicators['volume_ratio'] = price_data['volume'] / price_data['volume'].rolling(20).mean()
                
                # Pattern signal
                indicators['pattern_signal'] = self.calculate_pattern_signal(price_data)
                
                # WaveTrend
                wt_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'wavetrend')
                if wt_data is not None and 'wt_buy_signal' in wt_data.columns:
                    indicators['wavetrend'] = wt_data['wt_buy_signal'].astype(int) - wt_data.get('wt_sell_signal', 0).astype(int)
                    if 'wt1' in wt_data.columns:
                        indicators['wt_value'] = wt_data['wt1']
                
                # Momentum (from squeeze)
                sqz_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'squeeze_momentum')
                if sqz_data is not None:
                    if 'momentum' in sqz_data.columns:
                        indicators['momentum'] = sqz_data['momentum']
                    if 'sqz_buy_signal' in sqz_data.columns:
                        indicators['squeeze_momentum'] = sqz_data['sqz_buy_signal'].astype(int) - sqz_data.get('sqz_sell_signal', 0).astype(int)
                    if 'squeeze_on' in sqz_data.columns:
                        indicators['squeeze_on'] = sqz_data['squeeze_on']
                
                # MACD
                macd_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'macd')
                if macd_data is not None and 'macd_buy_signal' in macd_data.columns:
                    indicators['macd'] = macd_data['macd_buy_signal'].astype(int) - macd_data.get('macd_sell_signal', 0).astype(int)
                    if 'macd_histogram' in macd_data.columns:
                        indicators['macd_histogram'] = macd_data['macd_histogram']
                
                # ADX/DI
                adx_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'adx_di')
                if adx_data is not None:
                    if 'adx_buy_signal' in adx_data.columns:
                        indicators['adx_di'] = adx_data['adx_buy_signal'].astype(int) - adx_data.get('adx_sell_signal', 0).astype(int)
                    if 'adx' in adx_data.columns:
                        indicators['adx_value'] = adx_data['adx']
                    if 'plus_di' in adx_data.columns and 'minus_di' in adx_data.columns:
                        indicators['di_diff'] = adx_data['plus_di'] - adx_data['minus_di']
                
                # Supertrend
                st_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'supertrend')
                if st_data is not None and 'buy_signal' in st_data.columns:
                    indicators['supertrend'] = st_data['buy_signal'].astype(int) - st_data.get('sell_signal', 0).astype(int)
                
                # Technical indicators
                price_data['returns'] = price_data['close'].pct_change()
                price_data['volatility'] = price_data['returns'].rolling(20).std()
                price_data['rsi'] = self.calculate_rsi(price_data['close'])
                
                if indicators:
                    indicators_df = pd.DataFrame(indicators)
                    all_data[symbol] = {
                        'price': price_data,
                        'indicators': indicators_df
                    }
                    
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
                continue
        
        print(f"{len(all_data)} sembol yüklendi.")
        return all_data
    
    def calculate_pattern_signal(self, df: pd.DataFrame) -> pd.Series:
        """Pattern sinyali hesapla"""
        body = abs(df['close'] - df['open'])
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        
        # Hammer
        hammer = (lower_wick > 2 * body) & (df['close'] > df['open'])
        
        # Shooting star
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        shooting_star = (upper_wick > 2 * body) & (df['close'] < df['open'])
        
        signal = pd.Series(0, index=df.index)
        signal[hammer] = 1
        signal[shooting_star] = -1
        
        return signal
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI hesapla"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def check_phase_signals(self, indicators: pd.DataFrame, idx: int, phase: str) -> Tuple[float, float]:
        """Bir fazın sinyal gücünü hesapla (0-1 arası)"""
        phase_indicators = self.phases[phase]
        
        buy_strength = 0
        sell_strength = 0
        valid_indicators = 0
        
        for ind in phase_indicators:
            if ind in indicators.columns and idx < len(indicators):
                value = indicators[ind].iloc[idx]
                if not pd.isna(value):
                    valid_indicators += 1
                    
                    if ind == 'volume_ratio':
                        if value > 1.5:
                            buy_strength += min(value / 2, 1)
                        elif value < 0.7:
                            sell_strength += (1 - value) / 0.3
                    
                    elif ind == 'momentum':
                        # Normalize momentum
                        norm_momentum = np.tanh(value / 100)
                        if norm_momentum > 0:
                            buy_strength += norm_momentum
                        else:
                            sell_strength += abs(norm_momentum)
                    
                    elif ind in ['wavetrend', 'macd', 'squeeze_momentum', 'adx_di', 'supertrend', 'pattern_signal']:
                        if value > 0:
                            buy_strength += 1
                        elif value < 0:
                            sell_strength += 1
        
        if valid_indicators > 0:
            buy_strength /= valid_indicators
            sell_strength /= valid_indicators
        
        return buy_strength, sell_strength
    
    def score_symbol_sequential(self, symbol: str, data: Dict, current_idx: int) -> Tuple[float, Dict]:
        """4 fazlı sistem ile sembol skorla"""
        try:
            indicators_df = data['indicators']
            price_df = data['price']
            
            if current_idx >= len(indicators_df):
                return -999, {}
            
            phase_scores = {}
            total_score = 0
            
            # Her faz için skor hesapla
            for phase in ['phase1', 'phase2', 'phase3', 'phase4']:
                buy_strength, sell_strength = self.check_phase_signals(indicators_df, current_idx, phase)
                
                # Net score for phase
                phase_score = buy_strength - sell_strength
                phase_scores[phase] = {
                    'score': phase_score,
                    'buy_strength': buy_strength,
                    'sell_strength': sell_strength
                }
                
                # Weighted contribution
                total_score += phase_score * self.phase_weights[phase]
            
            # Additional factors
            
            # 1. Trend strength bonus
            if current_idx >= 50:
                short_trend = (price_df['close'].iloc[current_idx] - price_df['close'].iloc[current_idx-20]) / price_df['close'].iloc[current_idx-20]
                medium_trend = (price_df['close'].iloc[current_idx] - price_df['close'].iloc[current_idx-50]) / price_df['close'].iloc[current_idx-50]
                
                if short_trend > 0 and medium_trend > 0:
                    trend_bonus = min((short_trend + medium_trend) * 2, 0.2)
                    total_score += trend_bonus
            
            # 2. Momentum bonus
            if 'momentum' in indicators_df.columns:
                momentum = indicators_df['momentum'].iloc[current_idx]
                if not pd.isna(momentum) and momentum > 0:
                    mom_bonus = min(momentum / 500, 0.1)
                    total_score += mom_bonus
            
            # 3. Squeeze bonus
            if 'squeeze_on' in indicators_df.columns:
                squeeze_on = indicators_df['squeeze_on'].iloc[current_idx]
                if squeeze_on:
                    total_score += 0.05  # Squeeze aktifse bonus
            
            # 4. RSI penalty
            if current_idx < len(price_df):
                rsi = price_df['rsi'].iloc[current_idx]
                if not pd.isna(rsi):
                    if rsi > 70:
                        total_score -= 0.1  # Overbought penalty
                    elif rsi < 30:
                        total_score += 0.05  # Oversold bonus
            
            # Phase alignment bonus
            positive_phases = sum(1 for p in phase_scores.values() if p['score'] > 0)
            if positive_phases >= 3:
                total_score += 0.1 * (positive_phases - 2)
            
            return total_score, phase_scores
            
        except Exception as e:
            logger.error(f"Error scoring {symbol}: {e}")
            return -999, {}
    
    def select_best_opportunities_sequential(self, all_data: Dict, current_time: pd.Timestamp, 
                                           max_selections: int = 5) -> List[Tuple[str, float, Dict]]:
        """En iyi fırsatları seç - Sıralı sistem"""
        opportunities = []
        
        for symbol, data in all_data.items():
            if symbol in self.positions:
                continue
            
            price_df = data['price']
            if current_time not in price_df.index:
                continue
            
            current_idx = price_df.index.get_loc(current_time)
            score, phase_scores = self.score_symbol_sequential(symbol, data, current_idx)
            
            # Minimum 3 faz pozitif olmalı
            positive_phases = sum(1 for p in phase_scores.values() if p['score'] > 0)
            
            if score > 0.2 and positive_phases >= 3:  # Threshold
                current_price = price_df['close'].iloc[current_idx]
                opportunities.append((symbol, score, phase_scores, current_price))
        
        # Sort by score
        opportunities.sort(key=lambda x: x[1], reverse=True)
        
        return [(sym, score, phases) for sym, score, phases, _ in opportunities[:max_selections]]
    
    def check_exit_conditions_sequential(self, symbol: str, data: Dict, current_idx: int, 
                                       position: Dict, current_price: float) -> Tuple[bool, str]:
        """Çıkış koşulları - Sıralı sistem"""
        entry_price = position['entry_price']
        pnl_pct = (current_price - entry_price) / entry_price
        
        # 1. Stop Loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, 'STOP_LOSS'
        
        # 2. Take Profit
        if pnl_pct >= self.take_profit_pct:
            return True, 'TAKE_PROFIT'
        
        # 3. Phase-based exit
        _, phase_scores = self.score_symbol_sequential(symbol, data, current_idx)
        
        # Count negative phases
        negative_phases = sum(1 for p in phase_scores.values() if p['score'] < -0.2)
        
        # Exit if 3+ phases turn negative
        if negative_phases >= 3:
            return True, 'PHASE_EXIT'
        
        # 4. Trailing stop for profitable positions
        if pnl_pct > 0.10:  # 10%+ profit
            # Check if momentum is reversing
            if 'momentum' in data['indicators'].columns:
                momentum = data['indicators']['momentum'].iloc[current_idx]
                if not pd.isna(momentum) and momentum < -50:
                    return True, 'TRAILING_STOP'
        
        return False, ''
    
    def calculate_position_size_sequential(self, price: float, score: float, phase_scores: Dict) -> int:
        """Faz bazlı pozisyon boyutu hesapla"""
        available_cash = self.current_capital
        
        # Base size
        base_position_value = min(
            available_cash * 0.95,
            (self.initial_capital + self.current_capital) / (2 * self.max_positions)
        )
        
        # Score multiplier (0.6x to 1.4x)
        score_multiplier = 0.6 + (min(score, 1) * 0.8)
        
        # Phase alignment bonus
        positive_phases = sum(1 for p in phase_scores.values() if p['score'] > 0)
        if positive_phases == 4:
            score_multiplier *= 1.2  # All phases aligned
        
        position_value = base_position_value * score_multiplier
        
        # Commission
        commission_rate = 0.002
        position_value_after_commission = position_value / (1 + commission_rate)
        
        shares = int(position_value_after_commission / price)
        
        if shares < 1 or (shares * price * (1 + commission_rate)) > available_cash:
            return 0
        
        return shares
    
    def run_walk_forward(self, timeframe: str):
        """Walk-forward backtest çalıştır - Sıralı sistem"""
        logger.info(f"\nWalk-Forward Sequential Backtest başlıyor - Timeframe: {timeframe}")
        logger.info(f"Başlangıç Sermayesi: {self.initial_capital:,.0f} TL")
        logger.info(f"4 Fazlı Sıralı Sistem\n")
        
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
                
                should_exit, exit_reason = self.check_exit_conditions_sequential(
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
                    'phases_aligned': position.get('phases_aligned', 0)
                }
                
                self.trades.append(trade)
                self.current_capital += exit_value
                del self.positions[symbol]
                
                if pnl > 0:
                    self.stats['winning_trades'] += 1
                else:
                    self.stats['losing_trades'] += 1
                
                # Track phase performance
                for phase in position.get('positive_phases', []):
                    self.stats['phase_performance'][phase]['trades'] += 1
                    if pnl > 0:
                        self.stats['phase_performance'][phase]['wins'] += 1
                
                logger.debug(f"{current_time} - {symbol} {exit_reason}: "
                           f"{pnl/position['cost']*100:.1f}%")
            
            # Find new opportunities
            if len(self.positions) < self.max_positions:
                opportunities = self.select_best_opportunities_sequential(
                    all_data, current_time,
                    max_selections=self.max_positions - len(self.positions)
                )
                
                for symbol, score, phase_scores in opportunities:
                    if len(self.positions) >= self.max_positions:
                        break
                    
                    data = all_data[symbol]
                    price_df = data['price']
                    current_idx = price_df.index.get_loc(current_time)
                    current_price = price_df['close'].iloc[current_idx]
                    
                    shares = self.calculate_position_size_sequential(current_price, score, phase_scores)
                    if shares > 0:
                        commission = shares * current_price * 0.002
                        cost = (shares * current_price) + commission
                        
                        if cost <= self.current_capital * 0.95:
                            positive_phases = [p for p, s in phase_scores.items() if s['score'] > 0]
                            
                            self.positions[symbol] = {
                                'entry_time': current_time,
                                'entry_idx': current_idx,
                                'entry_price': current_price,
                                'shares': shares,
                                'cost': cost,
                                'entry_score': score,
                                'phase_scores': phase_scores,
                                'positive_phases': positive_phases,
                                'phases_aligned': len(positive_phases)
                            }
                            self.current_capital -= cost
                            self.stats['total_commission'] += commission
                            
                            logger.debug(f"{current_time} - BUY {symbol}: "
                                       f"Score: {score:.2f}, Phases: {len(positive_phases)}/4")
            
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
                    'phases_aligned': position.get('phases_aligned', 0)
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
        print(f"WALK-FORWARD SEQUENTIAL (4 FAZ) SONUÇLARI - {timeframe}")
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
            
            # Phase performance
            print(f"\nFAZ PERFORMANSLARI:")
            for phase, stats in self.stats['phase_performance'].items():
                if stats['trades'] > 0:
                    phase_win_rate = stats['wins'] / stats['trades']
                    print(f"{phase}: {stats['trades']} işlem, {phase_win_rate*100:.1f}% başarı")
            
            # Exit reasons
            exit_reasons = defaultdict(int)
            for trade in self.trades:
                exit_reasons[trade['exit_reason']] += 1
            
            print(f"\nÇIKIŞ NEDENLERİ:")
            for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"{reason}: {count} ({count/total_trades*100:.1f}%)")
            
            # Phase alignment distribution
            phase_dist = defaultdict(int)
            for trade in self.trades:
                phases = trade.get('phases_aligned', 0)
                phase_dist[phases] += 1
            
            print(f"\nFAZ UYUMU DAĞILIMI (Girişteki):")
            for phases, count in sorted(phase_dist.items()):
                if count > 0:
                    print(f"{phases} faz uyumlu: {count} işlem")
    
    def save_results(self, timeframe: str):
        """Sonuçları kaydet"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        summary = {
            'strategy': 'Walk-Forward Sequential (4 Phase)',
            'timeframe': timeframe,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'max_drawdown': self.stats['max_drawdown'],
            'total_trades': len(self.trades),
            'win_rate': self.stats['winning_trades'] / len(self.trades) if self.trades else 0,
            'phase_performance': self.stats['phase_performance'],
            'timestamp': timestamp
        }
        
        summary_file = Path(f"backtest/walk_forward_sequential_{timeframe}_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = Path(f"backtest/walk_forward_sequential_trades_{timeframe}_{timestamp}.csv")
            trades_df.to_csv(trades_file, index=False)
        
        logger.info(f"\nSonuçlar kaydedildi: {summary_file}")


def main():
    backtest = WalkForwardSequential(initial_capital=50000, max_positions=10)
    timeframe = backtest.get_timeframe_choice()
    backtest.run_walk_forward(timeframe)


if __name__ == "__main__":
    main()