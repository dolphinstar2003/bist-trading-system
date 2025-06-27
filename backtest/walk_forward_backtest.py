#!/usr/bin/env python3
"""
Walk-Forward Backtest Sistemi
Dinamik hisse seçimi ile gerçek zamanlı portfolio yönetimi
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


class WalkForwardBacktest:
    """Walk-forward backtest with dynamic stock selection"""
    
    def __init__(self, initial_capital: float = 50000, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.stop_loss_pct = 0.08  # %8 stop loss
        self.take_profit_pct = 0.20  # %20 take profit
        
        self.csv_manager = CSVDataManager()
        self.positions = {}  # Açık pozisyonlar
        self.trades = []     # Tamamlanmış işlemler
        self.daily_snapshots = []  # Günlük portfolio durumu
        
        # Scoring weights for stock selection
        self.scoring_weights = {
            'signal_strength': 0.3,    # Sinyal gücü
            'momentum': 0.2,           # Momentum
            'volume': 0.15,            # Volume
            'volatility': 0.15,        # Volatilite (düşük tercih)
            'trend_alignment': 0.2     # Trend uyumu
        }
        
        # İstatistikler
        self.stats = {
            'max_drawdown': 0,
            'peak_capital': initial_capital,
            'total_commission': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
    def get_timeframe_choice(self) -> str:
        """Kullanıcıdan timeframe seçimi al"""
        print("\n" + "="*50)
        print("WALK-FORWARD BACKTEST SİSTEMİ")
        print("Dinamik Hisse Seçimi")
        print("="*50)
        print("Timeframe seçin:")
        print("1. 1d  (Günlük)")
        print("2. 4h  (4 Saatlik)") 
        print("3. 1h  (Saatlik)")
        print("="*50)
        
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
            # Fiyat verisi
            price_data = self.csv_manager.load_raw_data(symbol, timeframe)
            if price_data is None or len(price_data) < 100:
                continue
            
            # İndikatör verileri
            indicators = {}
            
            # Supertrend
            st_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'supertrend')
            if st_data is not None and 'buy_signal' in st_data.columns:
                indicators['supertrend_buy'] = st_data['buy_signal'].astype(int)
                indicators['supertrend_sell'] = st_data.get('sell_signal', 0).astype(int)
                indicators['supertrend'] = indicators['supertrend_buy'] - indicators['supertrend_sell']
            
            # Squeeze Momentum
            sqz_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'squeeze_momentum')
            if sqz_data is not None and 'sqz_buy_signal' in sqz_data.columns:
                indicators['sqz_buy'] = sqz_data['sqz_buy_signal'].astype(int)
                indicators['sqz_sell'] = sqz_data.get('sqz_sell_signal', 0).astype(int)
                indicators['squeeze_momentum'] = indicators['sqz_buy'] - indicators['sqz_sell']
                if 'momentum' in sqz_data.columns:
                    indicators['momentum_value'] = sqz_data['momentum']
            
            # MACD
            macd_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'macd')
            if macd_data is not None and 'macd_buy_signal' in macd_data.columns:
                indicators['macd_buy'] = macd_data['macd_buy_signal'].astype(int)
                indicators['macd_sell'] = macd_data.get('macd_sell_signal', 0).astype(int)
                indicators['macd'] = indicators['macd_buy'] - indicators['macd_sell']
                if 'macd' in macd_data.columns:
                    indicators['macd_value'] = macd_data['macd']
                if 'macd_histogram' in macd_data.columns:
                    indicators['macd_histogram'] = macd_data['macd_histogram']
            
            # ADX/DI
            adx_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'adx_di')
            if adx_data is not None:
                if 'adx' in adx_data.columns:
                    indicators['adx'] = adx_data['adx']
                if 'plus_di' in adx_data.columns and 'minus_di' in adx_data.columns:
                    indicators['di_diff'] = adx_data['plus_di'] - adx_data['minus_di']
            
            # Lorentzian
            lor_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'lorentzian')
            if lor_data is not None and 'signal' in lor_data.columns:
                indicators['lorentzian'] = lor_data['signal']
                if 'confidence' in lor_data.columns:
                    indicators['lor_confidence'] = lor_data['confidence']
            
            if indicators:
                # İndikatörleri DataFrame'e çevir
                indicators_df = pd.DataFrame(indicators)
                
                # Technical indicators from price
                price_data['returns'] = price_data['close'].pct_change()
                price_data['volatility'] = price_data['returns'].rolling(20).std()
                price_data['volume_ratio'] = price_data['volume'] / price_data['volume'].rolling(20).mean()
                price_data['rsi'] = self.calculate_rsi(price_data['close'])
                
                all_data[symbol] = {
                    'price': price_data,
                    'indicators': indicators_df
                }
        
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
    
    def score_symbol(self, symbol: str, data: Dict, current_idx: int) -> float:
        """Bir sembolü skorla - yüksek skor = daha iyi alım fırsatı"""
        try:
            price_df = data['price']
            indicators_df = data['indicators']
            
            if current_idx >= len(price_df) or current_idx >= len(indicators_df):
                return -999
            
            score = 0
            weights_sum = 0
            
            # 1. Signal Strength (0.3)
            signal_score = 0
            signal_count = 0
            
            # Supertrend
            if 'supertrend' in indicators_df.columns:
                st_signal = indicators_df['supertrend'].iloc[current_idx]
                if st_signal > 0:
                    signal_score += 1
                    signal_count += 1
                elif st_signal < 0:
                    return -999  # Supertrend SELL ise hemen ele
            
            # Squeeze Momentum
            if 'squeeze_momentum' in indicators_df.columns:
                sqz_signal = indicators_df['squeeze_momentum'].iloc[current_idx]
                if sqz_signal > 0:
                    signal_score += 1
                    signal_count += 1
                elif sqz_signal < 0:
                    signal_score -= 0.5
                    signal_count += 1
            
            # MACD
            if 'macd' in indicators_df.columns:
                macd_signal = indicators_df['macd'].iloc[current_idx]
                if macd_signal > 0:
                    signal_score += 0.8
                    signal_count += 1
            
            # Lorentzian
            if 'lorentzian' in indicators_df.columns:
                lor_signal = indicators_df['lorentzian'].iloc[current_idx]
                if lor_signal > 0:
                    signal_score += 0.7
                    signal_count += 1
                    
                    # Confidence bonus
                    if 'lor_confidence' in indicators_df.columns:
                        confidence = indicators_df['lor_confidence'].iloc[current_idx]
                        signal_score += confidence * 0.3
            
            if signal_count > 0:
                signal_strength = signal_score / signal_count
                score += signal_strength * self.scoring_weights['signal_strength']
                weights_sum += self.scoring_weights['signal_strength']
            
            # 2. Momentum (0.2)
            if 'momentum_value' in indicators_df.columns:
                momentum = indicators_df['momentum_value'].iloc[current_idx]
                if not pd.isna(momentum):
                    # Normalize momentum (-1 to 1)
                    mom_norm = np.tanh(momentum / 100)  
                    score += mom_norm * self.scoring_weights['momentum']
                    weights_sum += self.scoring_weights['momentum']
            
            # 3. Volume (0.15)
            if current_idx < len(price_df):
                volume_ratio = price_df['volume_ratio'].iloc[current_idx]
                if not pd.isna(volume_ratio):
                    # Yüksek volume iyi (1.5x normal = iyi)
                    vol_score = min(volume_ratio / 2, 1)  # Max 1
                    score += vol_score * self.scoring_weights['volume']
                    weights_sum += self.scoring_weights['volume']
            
            # 4. Volatility (0.15) - Düşük volatilite tercih
            if current_idx < len(price_df):
                volatility = price_df['volatility'].iloc[current_idx]
                if not pd.isna(volatility):
                    # Düşük volatilite iyi
                    vol_score = 1 - min(volatility * 10, 1)  # 0.1 = max score
                    score += vol_score * self.scoring_weights['volatility']
                    weights_sum += self.scoring_weights['volatility']
            
            # 5. Trend Alignment (0.2)
            trend_score = 0
            if current_idx >= 50:
                # Short term trend (20 bars)
                short_trend = (price_df['close'].iloc[current_idx] - price_df['close'].iloc[current_idx-20]) / price_df['close'].iloc[current_idx-20]
                # Medium term trend (50 bars)
                medium_trend = (price_df['close'].iloc[current_idx] - price_df['close'].iloc[current_idx-50]) / price_df['close'].iloc[current_idx-50]
                
                # Both positive = good
                if short_trend > 0 and medium_trend > 0:
                    trend_score = min((short_trend + medium_trend) * 5, 1)
                else:
                    trend_score = max((short_trend + medium_trend) * 2, -1)
                
                score += trend_score * self.scoring_weights['trend_alignment']
                weights_sum += self.scoring_weights['trend_alignment']
            
            # RSI bonus/penalty
            if current_idx < len(price_df):
                rsi = price_df['rsi'].iloc[current_idx]
                if not pd.isna(rsi):
                    if rsi < 30:  # Oversold
                        score += 0.1
                    elif rsi > 70:  # Overbought
                        score -= 0.2
            
            # Normalize by weights actually used
            if weights_sum > 0:
                score = score / weights_sum
            
            return score
            
        except Exception as e:
            logger.error(f"Error scoring {symbol}: {e}")
            return -999
    
    def select_best_opportunities(self, all_data: Dict, current_time: pd.Timestamp, 
                                max_selections: int = 5) -> List[Tuple[str, float]]:
        """En iyi alım fırsatlarını seç"""
        opportunities = []
        
        for symbol, data in all_data.items():
            # Skip if already have position
            if symbol in self.positions:
                continue
            
            # Find current index
            price_df = data['price']
            if current_time not in price_df.index:
                continue
                
            current_idx = price_df.index.get_loc(current_time)
            
            # Score the symbol
            score = self.score_symbol(symbol, data, current_idx)
            
            if score > 0:  # Only positive scores
                current_price = price_df['close'].iloc[current_idx]
                opportunities.append((symbol, score, current_price))
        
        # Sort by score descending
        opportunities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return [(sym, score) for sym, score, _ in opportunities[:max_selections]]
    
    def check_exit_conditions(self, symbol: str, data: Dict, current_idx: int, 
                            position: Dict, current_price: float) -> Tuple[bool, str]:
        """Çıkış koşullarını kontrol et"""
        entry_price = position['entry_price']
        pnl_pct = (current_price - entry_price) / entry_price
        
        # 1. Stop Loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, 'STOP_LOSS'
        
        # 2. Take Profit
        if pnl_pct >= self.take_profit_pct:
            return True, 'TAKE_PROFIT'
        
        # 3. Signal-based exit
        indicators_df = data['indicators']
        
        exit_signals = 0
        total_signals = 0
        
        # Supertrend SELL
        if 'supertrend' in indicators_df.columns and current_idx < len(indicators_df):
            if indicators_df['supertrend'].iloc[current_idx] < 0:
                exit_signals += 2  # Supertrend has more weight
            total_signals += 2
        
        # Squeeze Momentum SELL
        if 'squeeze_momentum' in indicators_df.columns and current_idx < len(indicators_df):
            if indicators_df['squeeze_momentum'].iloc[current_idx] < 0:
                exit_signals += 1
            total_signals += 1
        
        # MACD SELL
        if 'macd' in indicators_df.columns and current_idx < len(indicators_df):
            if indicators_df['macd'].iloc[current_idx] < 0:
                exit_signals += 1
            total_signals += 1
        
        # Exit if majority signals are SELL
        if total_signals > 0 and exit_signals / total_signals >= 0.6:
            return True, 'SIGNAL'
        
        # 4. Time-based exit (optional)
        # Hold for minimum 5 bars before considering weak exit signals
        bars_held = current_idx - position['entry_idx']
        if bars_held > 20 and pnl_pct < 0.02:  # 20 bars and less than 2% profit
            return True, 'TIME_EXIT'
        
        return False, ''
    
    def calculate_position_size(self, price: float, score: float) -> int:
        """Score bazlı pozisyon büyüklüğü hesapla"""
        # Base position size
        available_cash = self.current_capital
        base_position_value = min(
            available_cash * 0.95,  # Leave some cash
            (self.initial_capital + self.current_capital) / (2 * self.max_positions)
        )
        
        # Adjust by score (higher score = larger position)
        score_multiplier = 0.7 + (score * 0.6)  # 0.7x to 1.3x
        position_value = base_position_value * score_multiplier
        
        # Commission
        commission_rate = 0.002
        position_value_after_commission = position_value / (1 + commission_rate)
        
        shares = int(position_value_after_commission / price)
        
        # Check constraints
        if shares < 1 or (shares * price * (1 + commission_rate)) > available_cash:
            return 0
        
        return shares
    
    def run_walk_forward(self, timeframe: str):
        """Walk-forward backtest çalıştır"""
        logger.info(f"\nWalk-Forward Backtest başlıyor - Timeframe: {timeframe}")
        logger.info(f"Başlangıç Sermayesi: {self.initial_capital:,.0f} TL")
        logger.info(f"Maksimum Pozisyon: {self.max_positions}")
        logger.info(f"Stop Loss: %{self.stop_loss_pct*100}, Take Profit: %{self.take_profit_pct*100}\n")
        
        # Preload all data
        all_data = self.preload_all_data(timeframe)
        if not all_data:
            logger.error("No data loaded!")
            return
        
        # Get common timeline
        all_dates = set()
        for symbol, data in all_data.items():
            all_dates.update(data['price'].index)
        
        timeline = sorted(list(all_dates))
        
        # Minimum 100 bars warm-up
        start_idx = 100
        
        logger.info(f"Backtest period: {timeline[start_idx]} to {timeline[-1]}")
        logger.info(f"Total days/bars: {len(timeline) - start_idx}\n")
        
        # Main loop through time
        for i, current_time in enumerate(timeline[start_idx:], start_idx):
            
            # Update positions and check exits
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
                
                # Check exit conditions
                should_exit, exit_reason = self.check_exit_conditions(
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
                    'holding_periods': current_idx - position['entry_idx']
                }
                
                self.trades.append(trade)
                self.current_capital += exit_value
                del self.positions[symbol]
                
                if pnl > 0:
                    self.stats['winning_trades'] += 1
                else:
                    self.stats['losing_trades'] += 1
                
                logger.debug(f"{current_time} - {symbol} {exit_reason}: "
                           f"{position['entry_price']:.2f} -> {exit_price:.2f} "
                           f"({pnl/position['cost']*100:.1f}%)")
            
            # Select new opportunities if we have room
            if len(self.positions) < self.max_positions:
                opportunities = self.select_best_opportunities(
                    all_data, current_time, 
                    max_selections=self.max_positions - len(self.positions)
                )
                
                for symbol, score in opportunities:
                    if len(self.positions) >= self.max_positions:
                        break
                    
                    data = all_data[symbol]
                    price_df = data['price']
                    current_idx = price_df.index.get_loc(current_time)
                    current_price = price_df['close'].iloc[current_idx]
                    
                    shares = self.calculate_position_size(current_price, score)
                    if shares > 0:
                        commission = shares * current_price * 0.002
                        cost = (shares * current_price) + commission
                        
                        if cost <= self.current_capital * 0.95:  # Don't use all capital
                            self.positions[symbol] = {
                                'entry_time': current_time,
                                'entry_idx': current_idx,
                                'entry_price': current_price,
                                'shares': shares,
                                'cost': cost,
                                'entry_score': score
                            }
                            self.current_capital -= cost
                            self.stats['total_commission'] += commission
                            
                            logger.debug(f"{current_time} - BUY {symbol}: "
                                       f"{shares} @ {current_price:.2f} "
                                       f"(Score: {score:.2f})")
            
            # Track daily snapshot
            if i % 10 == 0:  # Every 10 bars
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
                    'entry_score': position.get('entry_score', 0)
                }
                
                self.trades.append(trade)
                self.current_capital += exit_value
                
                if pnl > 0:
                    self.stats['winning_trades'] += 1
                else:
                    self.stats['losing_trades'] += 1
        
        # Final results
        self.print_results(timeframe)
        self.save_results(timeframe)
    
    def print_results(self, timeframe: str):
        """Sonuçları yazdır"""
        print("\n" + "="*80)
        print(f"WALK-FORWARD BACKTEST SONUÇLARI - {timeframe}")
        print("="*80)
        
        # Portfolio summary
        final_value = self.current_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        print(f"\nPORTFOLIO ÖZET:")
        print(f"Başlangıç Sermayesi: {self.initial_capital:,.0f} TL")
        print(f"Final Sermaye: {final_value:,.0f} TL")
        print(f"Toplam Getiri: {total_return*100:.1f}%")
        
        # Calculate annualized return (assuming 3 years)
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
            print(f"Başarılı İşlem: {self.stats['winning_trades']} ({win_rate*100:.1f}%)")
            print(f"Başarısız İşlem: {self.stats['losing_trades']}")
            
            # Average returns
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] < 0]
            
            if winning_trades:
                avg_win = np.mean([t['pnl_pct'] for t in winning_trades])
                print(f"Ortalama Kazanç: {avg_win*100:.2f}%")
            
            if losing_trades:
                avg_loss = np.mean([t['pnl_pct'] for t in losing_trades])
                print(f"Ortalama Kayıp: {avg_loss*100:.2f}%")
            
            # Profit factor
            if losing_trades:
                total_wins = sum(t['pnl'] for t in winning_trades)
                total_losses = abs(sum(t['pnl'] for t in losing_trades))
                profit_factor = total_wins / total_losses if total_losses > 0 else 0
                print(f"Profit Factor: {profit_factor:.2f}")
            
            # Exit reason breakdown
            exit_reasons = defaultdict(int)
            for trade in self.trades:
                exit_reasons[trade['exit_reason']] += 1
            
            print(f"\nÇIKIŞ NEDENLERİ:")
            for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"{reason}: {count} ({count/total_trades*100:.1f}%)")
            
            # Top traded symbols
            symbol_pnl = defaultdict(float)
            symbol_trades = defaultdict(int)
            
            for trade in self.trades:
                symbol_pnl[trade['symbol']] += trade['pnl']
                symbol_trades[trade['symbol']] += 1
            
            print(f"\nEN KARLI 10 HİSSE:")
            print(f"{'Sembol':<8} {'İşlem':<8} {'Toplam K/Z':<15} {'Ort K/Z'}")
            print("-" * 50)
            
            sorted_symbols = sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)
            for symbol, total_pnl in sorted_symbols[:10]:
                if total_pnl > 0:
                    avg_pnl = total_pnl / symbol_trades[symbol]
                    print(f"{symbol:<8} {symbol_trades[symbol]:<8} "
                          f"{total_pnl:>12,.2f} TL {avg_pnl:>10,.2f} TL")
            
            # Average holding period
            holding_periods = [t.get('holding_periods', 0) for t in self.trades if 'holding_periods' in t]
            if holding_periods:
                avg_holding = np.mean(holding_periods)
                print(f"\nOrtalama Tutma Süresi: {avg_holding:.1f} bar")
    
    def save_results(self, timeframe: str):
        """Sonuçları kaydet"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Summary
        summary = {
            'strategy': 'Walk-Forward Dynamic Selection',
            'timeframe': timeframe,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'max_drawdown': self.stats['max_drawdown'],
            'total_trades': len(self.trades),
            'win_rate': self.stats['winning_trades'] / len(self.trades) if self.trades else 0,
            'scoring_weights': self.scoring_weights,
            'timestamp': timestamp
        }
        
        # Save summary
        summary_file = Path(f"backtest/walk_forward_{timeframe}_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = Path(f"backtest/walk_forward_trades_{timeframe}_{timestamp}.csv")
            trades_df.to_csv(trades_file, index=False)
        
        # Save daily snapshots
        if self.daily_snapshots:
            snapshots_df = pd.DataFrame(self.daily_snapshots)
            snapshots_file = Path(f"backtest/walk_forward_snapshots_{timeframe}_{timestamp}.csv")
            snapshots_df.to_csv(snapshots_file, index=False)
        
        logger.info(f"\nSonuçlar kaydedildi:")
        logger.info(f"  Özet: {summary_file}")
        if self.trades:
            logger.info(f"  İşlemler: {trades_file}")
        if self.daily_snapshots:
            logger.info(f"  Portfolio: {snapshots_file}")


def main():
    backtest = WalkForwardBacktest(initial_capital=50000, max_positions=10)
    
    # Timeframe seçimi
    timeframe = backtest.get_timeframe_choice()
    
    # Run walk-forward backtest
    backtest.run_walk_forward(timeframe)


if __name__ == "__main__":
    main()