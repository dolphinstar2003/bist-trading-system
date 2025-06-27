#!/usr/bin/env python3
"""
Sıralı Backtest Sistemi - Fixed Version
4 Fazlı İndikatör Onay Sistemi ile Backtest
Düzeltilmiş sermaye yönetimi
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
from loguru import logger
import json

# Proje imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS


class SiraliBacktestFixed:
    """4 Fazlı sıralı indikatör backtest sistemi - Düzeltilmiş sermaye yönetimi"""
    
    def __init__(self, initial_capital: float = 50000, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.stop_loss_pct = 0.08  # %8 stop loss
        
        self.csv_manager = CSVDataManager()
        self.positions = {}  # Açık pozisyonlar
        self.trades = []     # Tamamlanmış işlemler
        self.portfolio_values = []
        self.capital_history = []  # Sermaye takibi için
        
        # İndikatör fazları
        self.phases = {
            'phase1': ['volume_ratio', 'pattern_signal'],  # Volume + Patterns
            'phase2': ['wavetrend', 'momentum'],           # WaveTrend + Momentum  
            'phase3': ['macd', 'squeeze_momentum'],        # MACD + Squeeze
            'phase4': ['adx_di', 'supertrend']            # ADX + Supertrend
        }
        
        # İstatistikler
        self.stats = {
            'max_drawdown': 0,
            'peak_capital': initial_capital,
            'total_commission': 0
        }
        
    def get_timeframe_choice(self) -> str:
        """Kullanıcıdan timeframe seçimi al"""
        print("\n" + "="*50)
        print("SIRALI BACKTEST SİSTEMİ - FIXED")
        print("="*50)
        print("Timeframe seçin:")
        print("1. 1d  (Günlük)")
        print("2. 4h  (4 Saatlik)") 
        print("3. 1h  (Saatlik)")
        print("4. 15m (15 Dakika)")
        print("="*50)
        
        while True:
            choice = input("Seçiminiz (1-4): ")
            mapping = {'1': '1d', '2': '4h', '3': '1h', '4': '15m'}
            if choice in mapping:
                return mapping[choice]
            print("Hatalı seçim! 1-4 arası seçin.")
    
    def load_all_data(self, symbol: str, timeframe: str) -> Dict:
        """Bir sembol için tüm verileri yükle"""
        try:
            # Ham veri
            price_data = self.csv_manager.load_raw_data(symbol, timeframe)
            if price_data is None or len(price_data) < 100:
                return None
            
            # İndikatör verileri
            indicators = {}
            
            # Volume (fiyat verisinden hesapla)
            if 'volume' in price_data.columns:
                indicators['volume_ratio'] = price_data['volume'] / price_data['volume'].rolling(20).mean()
            
            # Pattern sinyali (basit örnek - gerçek pattern tanıma eklenebilir)
            indicators['pattern_signal'] = self.calculate_pattern_signal(price_data)
            
            # Diğer indikatörleri yükle
            indicator_files = {
                'wavetrend': 'wavetrend',
                'momentum': 'squeeze_momentum',
                'macd': 'macd',
                'squeeze_momentum': 'squeeze_momentum',
                'adx_di': 'adx_di',
                'supertrend': 'supertrend'
            }
            
            for ind_name, file_name in indicator_files.items():
                ind_data = self.csv_manager.load_indicator_data(symbol, timeframe, file_name)
                if ind_data is not None:
                    if ind_name == 'wavetrend' and 'wt_buy_signal' in ind_data.columns:
                        indicators['wavetrend'] = ind_data['wt_buy_signal'].astype(int) - ind_data.get('wt_sell_signal', 0).astype(int)
                    elif ind_name == 'momentum' and 'momentum' in ind_data.columns:
                        indicators['momentum'] = ind_data['momentum']
                    elif ind_name == 'macd' and 'macd_buy_signal' in ind_data.columns:
                        indicators['macd'] = ind_data['macd_buy_signal'].astype(int) - ind_data.get('macd_sell_signal', 0).astype(int)
                    elif ind_name == 'squeeze_momentum' and 'sqz_buy_signal' in ind_data.columns:
                        indicators['squeeze_momentum'] = ind_data['sqz_buy_signal'].astype(int) - ind_data.get('sqz_sell_signal', 0).astype(int)
                    elif ind_name == 'adx_di' and 'adx_buy_signal' in ind_data.columns:
                        indicators['adx_di'] = ind_data['adx_buy_signal'].astype(int) - ind_data.get('adx_sell_signal', 0).astype(int)
                    elif ind_name == 'supertrend' and 'buy_signal' in ind_data.columns:
                        indicators['supertrend'] = ind_data['buy_signal'].astype(int) - ind_data.get('sell_signal', 0).astype(int)
            
            # Tüm veriyi birleştir
            result = {
                'price': price_data,
                'indicators': pd.DataFrame(indicators, index=price_data.index)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def calculate_pattern_signal(self, df: pd.DataFrame) -> pd.Series:
        """Basit pattern sinyali hesapla"""
        # Örnek: Hammer pattern benzeri
        body = abs(df['close'] - df['open'])
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        
        # Hammer: küçük gövde, uzun alt fitil
        hammer = (lower_wick > 2 * body) & (df['close'] > df['open'])
        
        # Shooting star: küçük gövde, uzun üst fitil  
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        shooting_star = (upper_wick > 2 * body) & (df['close'] < df['open'])
        
        signal = pd.Series(0, index=df.index)
        signal[hammer] = 1
        signal[shooting_star] = -1
        
        return signal
    
    def check_phase_signals(self, indicators: pd.DataFrame, idx: int, phase: str) -> Tuple[bool, bool]:
        """Bir fazın sinyallerini kontrol et"""
        phase_indicators = self.phases[phase]
        
        buy_signals = []
        sell_signals = []
        
        for ind in phase_indicators:
            if ind in indicators.columns and idx < len(indicators):
                value = indicators[ind].iloc[idx]
                if not pd.isna(value):
                    if ind in ['volume_ratio']:
                        # Volume yüksekse pozitif
                        buy_signals.append(value > 1.5)
                        sell_signals.append(value < 0.7)
                    elif ind == 'momentum':
                        # Momentum pozitif/negatif
                        buy_signals.append(value > 0)
                        sell_signals.append(value < 0)
                    else:
                        # Diğerleri için sinyal değeri
                        buy_signals.append(value > 0)
                        sell_signals.append(value < 0)
        
        # En az bir pozitif sinyal varsa True
        phase_buy = any(buy_signals) if buy_signals else False
        phase_sell = any(sell_signals) if sell_signals else False
        
        return phase_buy, phase_sell
    
    def check_all_phases(self, indicators: pd.DataFrame, idx: int, 
                        lookback: int = 10) -> Tuple[str, Dict]:
        """Tüm fazları kontrol et"""
        # Son N mum için fazları kontrol et
        phase_status = {}
        
        for i in range(max(0, idx - lookback + 1), idx + 1):
            for phase in ['phase1', 'phase2', 'phase3', 'phase4']:
                buy, sell = self.check_phase_signals(indicators, i, phase)
                
                if phase not in phase_status:
                    phase_status[phase] = {'buy': False, 'sell': False, 'last_idx': None}
                
                if buy:
                    phase_status[phase]['buy'] = True
                    phase_status[phase]['last_idx'] = i
                elif sell:
                    phase_status[phase]['sell'] = True
                    phase_status[phase]['last_idx'] = i
        
        # Sıralı onay kontrolü - Daha katı kurallar
        # En az 3 fazın onayı gerekli
        buy_count = sum(1 for status in phase_status.values() if status['buy'])
        sell_count = sum(1 for status in phase_status.values() if status['sell'])
        
        if buy_count >= 3 and sell_count == 0:
            return 'BUY', phase_status
        elif sell_count >= 3 and buy_count == 0:
            return 'SELL', phase_status
        else:
            return 'HOLD', phase_status
    
    def calculate_position_size(self, price: float) -> int:
        """Pozisyon büyüklüğü hesapla - Düzeltilmiş"""
        # Mevcut nakit sermaye
        available_cash = self.current_capital
        
        # Açık pozisyonların değerini hesapla
        open_positions_value = 0
        for pos in self.positions.values():
            open_positions_value += pos['cost']
        
        # Gerçek kullanılabilir sermaye
        total_portfolio_value = available_cash + open_positions_value
        
        # Her pozisyon için max %10 kullan (10 pozisyon = %100)
        position_value = min(
            available_cash,  # Eldeki nakitten fazla kullanma
            total_portfolio_value * 0.1  # Toplam portföyün %10'u
        )
        
        # Komisyon dahil hesapla (%0.2)
        commission_rate = 0.002
        position_value_after_commission = position_value / (1 + commission_rate)
        
        shares = int(position_value_after_commission / price)
        
        # Minimum 1 lot kontrolü ve maksimum sermaye kontrolü
        if shares < 1 or (shares * price * (1 + commission_rate)) > available_cash:
            return 0
        
        return shares
    
    def run_backtest(self, symbol: str, data: Dict) -> Dict:
        """Tek sembol için backtest çalıştır - Düzeltilmiş sermaye yönetimi"""
        price_df = data['price']
        indicators = data['indicators']
        
        symbol_trades = []
        
        # Ana döngü
        for idx in range(50, len(price_df)):  # İlk 50 mum skip (indikatör hesaplama)
            current_time = price_df.index[idx]
            current_price = price_df['close'].iloc[idx]
            
            # Sermaye takibi
            self.capital_history.append({
                'time': current_time,
                'capital': self.current_capital,
                'positions': len(self.positions)
            })
            
            # Pozisyon var mı kontrol et
            if symbol in self.positions:
                # Stop loss kontrolü
                position = self.positions[symbol]
                loss_pct = (current_price - position['entry_price']) / position['entry_price']
                
                if loss_pct <= -self.stop_loss_pct:
                    # Stop loss
                    commission = current_price * position['shares'] * 0.002
                    exit_value = (current_price * position['shares']) - commission
                    pnl = exit_value - position['cost']
                    
                    trade = {
                        'symbol': symbol,
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'shares': position['shares'],
                        'pnl': pnl,
                        'pnl_pct': pnl / position['cost'],
                        'exit_reason': 'STOP_LOSS',
                        'commission': position['entry_commission'] + commission
                    }
                    
                    symbol_trades.append(trade)
                    self.trades.append(trade)
                    self.current_capital += exit_value
                    self.stats['total_commission'] += commission
                    del self.positions[symbol]
                    
                    logger.debug(f"{symbol} Stop Loss: {position['entry_price']:.2f} -> {current_price:.2f} ({loss_pct:.1%})")
                    continue
            
            # Sinyal kontrolü
            signal, phase_status = self.check_all_phases(indicators, idx)
            
            # Pozisyon açma
            if signal == 'BUY' and symbol not in self.positions and len(self.positions) < self.max_positions:
                shares = self.calculate_position_size(current_price)
                
                if shares > 0:
                    commission = shares * current_price * 0.002
                    cost = (shares * current_price) + commission
                    
                    if cost <= self.current_capital:
                        self.positions[symbol] = {
                            'entry_time': current_time,
                            'entry_price': current_price,
                            'shares': shares,
                            'cost': cost,
                            'phases': phase_status,
                            'entry_commission': commission
                        }
                        self.current_capital -= cost
                        self.stats['total_commission'] += commission
                        
                        logger.debug(f"{symbol} BUY: {shares} @ {current_price:.2f} (Cost: {cost:.2f})")
            
            # Pozisyon kapatma
            elif signal == 'SELL' and symbol in self.positions:
                position = self.positions[symbol]
                commission = current_price * position['shares'] * 0.002
                exit_value = (current_price * position['shares']) - commission
                pnl = exit_value - position['cost']
                
                trade = {
                    'symbol': symbol,
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'shares': position['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl / position['cost'],
                    'exit_reason': 'SIGNAL',
                    'commission': position['entry_commission'] + commission
                }
                
                symbol_trades.append(trade)
                self.trades.append(trade)
                self.current_capital += exit_value
                self.stats['total_commission'] += commission
                del self.positions[symbol]
                
                logger.debug(f"{symbol} SELL: {position['shares']} @ {current_price:.2f} (PnL: {pnl:.2f})")
            
            # Drawdown hesapla
            if self.current_capital > self.stats['peak_capital']:
                self.stats['peak_capital'] = self.current_capital
            else:
                drawdown = (self.stats['peak_capital'] - self.current_capital) / self.stats['peak_capital']
                self.stats['max_drawdown'] = max(self.stats['max_drawdown'], drawdown)
        
        # Açık pozisyonları kapat
        if symbol in self.positions:
            position = self.positions[symbol]
            final_price = price_df['close'].iloc[-1]
            commission = final_price * position['shares'] * 0.002
            exit_value = (final_price * position['shares']) - commission
            pnl = exit_value - position['cost']
            
            trade = {
                'symbol': symbol,
                'entry_time': position['entry_time'],
                'exit_time': price_df.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': final_price,
                'shares': position['shares'],
                'pnl': pnl,
                'pnl_pct': pnl / position['cost'],
                'exit_reason': 'END_TEST',
                'commission': position['entry_commission'] + commission
            }
            
            symbol_trades.append(trade)
            self.trades.append(trade)
            self.current_capital += exit_value
            self.stats['total_commission'] += commission
            del self.positions[symbol]
        
        return symbol_trades
    
    def run_all_symbols(self, timeframe: str):
        """Tüm semboller için backtest çalıştır"""
        logger.info(f"\nSıralı Backtest (Fixed) başlıyor - Timeframe: {timeframe}")
        logger.info(f"Başlangıç Sermayesi: {self.initial_capital:,.0f} TL")
        logger.info(f"Maksimum Pozisyon: {self.max_positions}")
        logger.info(f"Stop Loss: %{self.stop_loss_pct*100}")
        logger.info(f"Komisyon: %0.2")
        logger.info(f"Test edilecek hisse sayısı: {len(ASSETS)}\n")
        
        results = []
        
        for i, symbol in enumerate(ASSETS):
            logger.info(f"[{i+1}/{len(ASSETS)}] {symbol} test ediliyor...")
            
            # Veri yükle
            data = self.load_all_data(symbol, timeframe)
            if data is None:
                logger.warning(f"{symbol} - Veri bulunamadı, atlanıyor")
                continue
            
            # Backtest çalıştır
            trades = self.run_backtest(symbol, data)
            
            if trades:
                total_pnl = sum(t['pnl'] for t in trades)
                win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
                avg_pnl_pct = np.mean([t['pnl_pct'] for t in trades])
                
                results.append({
                    'symbol': symbol,
                    'trades': len(trades),
                    'total_pnl': total_pnl,
                    'win_rate': win_rate,
                    'avg_pnl_pct': avg_pnl_pct,
                    'total_commission': sum(t['commission'] for t in trades)
                })
                
                logger.info(f"  İşlem: {len(trades)}, Kar/Zarar: {total_pnl:,.2f} TL, Başarı: {win_rate:.1%}")
        
        # Final sonuçlar
        self.print_final_results(results, timeframe)
        self.save_results(results, timeframe)
    
    def print_final_results(self, results: List[Dict], timeframe: str):
        """Final sonuçları yazdır - Detaylı"""
        print("\n" + "="*80)
        print(f"SIRALI BACKTEST SONUÇLARI (FIXED) - {timeframe}")
        print("="*80)
        
        # Genel özet
        total_trades = sum(r['trades'] for r in results)
        total_pnl = sum(r['total_pnl'] for r in results)
        final_capital = self.current_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        print(f"\nGENEL ÖZET:")
        print(f"Başlangıç Sermayesi: {self.initial_capital:,.0f} TL")
        print(f"Final Sermaye: {final_capital:,.0f} TL")
        print(f"Toplam Kar/Zarar: {total_pnl:,.2f} TL")
        print(f"Toplam Getiri: {total_return:.1%}")
        print(f"Yıllık Getiri (3 yıl): {(((final_capital/self.initial_capital)**(1/3))-1)*100:.1f}%")
        print(f"Toplam İşlem: {total_trades}")
        print(f"Toplam Komisyon: {self.stats['total_commission']:,.2f} TL")
        print(f"Maksimum Drawdown: {self.stats['max_drawdown']:.1%}")
        
        # Başarılı sembol sayısı
        profitable_symbols = sum(1 for r in results if r['total_pnl'] > 0)
        print(f"Karlı Sembol Sayısı: {profitable_symbols}/{len(results)}")
        
        # En karlı hisseler
        sorted_results = sorted(results, key=lambda x: x['total_pnl'], reverse=True)
        
        print(f"\nEN KARLI 10 HİSSE:")
        print(f"{'Sembol':<10} {'İşlem':<10} {'Kar/Zarar':<15} {'Başarı %':<10} {'Ort PnL%'}")
        print("-" * 60)
        
        for r in sorted_results[:10]:
            if r['total_pnl'] > 0:
                print(f"{r['symbol']:<10} {r['trades']:<10} {r['total_pnl']:>12,.2f} TL "
                      f"{r['win_rate']:>8.1%} {r['avg_pnl_pct']*100:>8.1f}%")
        
        # En zararlı hisseler
        print(f"\nEN ZARARLI 5 HİSSE:")
        print(f"{'Sembol':<10} {'İşlem':<10} {'Kar/Zarar':<15} {'Başarı %'}")
        print("-" * 50)
        
        for r in sorted_results[-5:]:
            if r['total_pnl'] < 0:
                print(f"{r['symbol']:<10} {r['trades']:<10} {r['total_pnl']:>12,.2f} TL {r['win_rate']:>8.1%}")
        
        # İşlem istatistikleri
        if self.trades:
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] < 0]
            
            print(f"\nİŞLEM İSTATİSTİKLERİ:")
            print(f"Toplam İşlem: {len(self.trades)}")
            print(f"Başarılı İşlem: {len(winning_trades)} ({len(winning_trades)/len(self.trades)*100:.1f}%)")
            print(f"Başarısız İşlem: {len(losing_trades)} ({len(losing_trades)/len(self.trades)*100:.1f}%)")
            
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
    
    def save_results(self, results: List[Dict], timeframe: str):
        """Sonuçları kaydet"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Özet dosyası
        summary = {
            'strategy': 'Sıralı 4 Faz - Fixed',
            'timeframe': timeframe,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'annual_return': (((self.current_capital/self.initial_capital)**(1/3))-1)*100,
            'total_trades': len(self.trades),
            'max_drawdown': self.stats['max_drawdown'],
            'total_commission': self.stats['total_commission'],
            'symbol_results': results,
            'timestamp': timestamp
        }
        
        summary_file = Path(f"backtest/sirali_backtest_fixed_{timeframe}_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Detaylı trade listesi
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_file = Path(f"backtest/sirali_trades_fixed_{timeframe}_{timestamp}.csv")
            trades_df.to_csv(trades_file, index=False)
        
        logger.info(f"\nSonuçlar kaydedildi: {summary_file}")


def main():
    backtest = SiraliBacktestFixed(initial_capital=50000, max_positions=10)
    
    # Timeframe seçimi
    timeframe = backtest.get_timeframe_choice()
    
    # Backtest çalıştır
    backtest.run_all_symbols(timeframe)


if __name__ == "__main__":
    main()