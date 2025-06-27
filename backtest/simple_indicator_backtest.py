#!/usr/bin/env python3
"""
Simple Indicator Backtest
3'lü Onay Sistemi ile Backtest
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


class SimpleIndicatorBacktest:
    """3'lü onay sistemi ile basit backtest"""
    
    def __init__(self, initial_capital: float = 50000, max_positions: int = 10):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_positions = max_positions
        self.stop_loss_pct = 0.08  # %8 stop loss
        
        self.csv_manager = CSVDataManager()
        self.positions = {}  # Açık pozisyonlar
        self.trades = []     # Tamamlanmış işlemler
        self.portfolio_values = []
        
        # Ana indikatörler
        self.primary_indicators = ['supertrend', 'squeeze_momentum']
        # Onay indikatörleri
        self.confirmation_indicators = ['macd', 'wavetrend', 'adx_di', 'lorentzian', 'trend_vanguard']
        
    def get_timeframe_choice(self) -> str:
        """Kullanıcıdan timeframe seçimi al"""
        print("\n" + "="*50)
        print("SIMPLE INDICATOR BACKTEST")
        print("3'lü Onay Sistemi")
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
    
    def load_indicator_signals(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Tüm indikatör sinyallerini yükle"""
        signals = pd.DataFrame()
        
        # Supertrend
        supertrend_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'supertrend')
        if supertrend_data is not None and 'buy_signal' in supertrend_data.columns:
            signals['supertrend_buy'] = supertrend_data['buy_signal'].astype(int)
            signals['supertrend_sell'] = supertrend_data.get('sell_signal', 0).astype(int)
            signals['supertrend'] = signals['supertrend_buy'] - signals['supertrend_sell']
        
        # Squeeze Momentum
        sqz_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'squeeze_momentum')
        if sqz_data is not None and 'sqz_buy_signal' in sqz_data.columns:
            signals['sqz_buy'] = sqz_data['sqz_buy_signal'].astype(int)
            signals['sqz_sell'] = sqz_data.get('sqz_sell_signal', 0).astype(int)
            signals['squeeze_momentum'] = signals['sqz_buy'] - signals['sqz_sell']
            # Squeeze on/off durumu
            if 'squeeze_on' in sqz_data.columns:
                signals['squeeze_active'] = sqz_data['squeeze_on'].astype(int)
        
        # MACD
        macd_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'macd')
        if macd_data is not None and 'macd_buy_signal' in macd_data.columns:
            signals['macd_buy'] = macd_data['macd_buy_signal'].astype(int)
            signals['macd_sell'] = macd_data.get('macd_sell_signal', 0).astype(int)
            signals['macd'] = signals['macd_buy'] - signals['macd_sell']
        
        # WaveTrend
        wt_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'wavetrend')
        if wt_data is not None and 'wt_buy_signal' in wt_data.columns:
            signals['wt_buy'] = wt_data['wt_buy_signal'].astype(int)
            signals['wt_sell'] = wt_data.get('wt_sell_signal', 0).astype(int)
            signals['wavetrend'] = signals['wt_buy'] - signals['wt_sell']
        
        # ADX/DI
        adx_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'adx_di')
        if adx_data is not None and 'adx_buy_signal' in adx_data.columns:
            signals['adx_buy'] = adx_data['adx_buy_signal'].astype(int)
            signals['adx_sell'] = adx_data.get('adx_sell_signal', 0).astype(int)
            signals['adx_di'] = signals['adx_buy'] - signals['adx_sell']
        
        # Lorentzian
        lor_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'lorentzian')
        if lor_data is not None:
            if 'is_bullish' in lor_data.columns:
                signals['lor_bullish'] = lor_data['is_bullish'].astype(int)
                signals['lor_bearish'] = lor_data.get('is_bearish', 0).astype(int)
                signals['lorentzian'] = signals['lor_bullish'] - signals['lor_bearish']
            elif 'signal' in lor_data.columns:
                signals['lorentzian'] = lor_data['signal'].astype(int)
        
        # Trend Vanguard
        tv_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'trend_vanguard')
        if tv_data is not None and 'signal' in tv_data.columns:
            signals['tv_signal'] = tv_data['signal'].astype(int)
            signals['trend_vanguard'] = signals['tv_signal']
            # Market regime bilgisi de eklenebilir
            if 'market_regime' in tv_data.columns:
                signals['tv_regime'] = tv_data['market_regime']
        
        return signals
    
    def check_entry_conditions(self, signals: pd.Series) -> str:
        """Giriş koşullarını kontrol et"""
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
            return 'HOLD'
        
        # BUY için kontrol
        if primary_buy:
            # Hiçbir onay indikatörü SELL vermemeli
            no_sell_signals = all(
                signals.get(ind, 0) >= 0 
                for ind in self.confirmation_indicators
            )
            
            # En az 1 onay indikatörü BUY vermeli
            has_confirmation = any(
                signals.get(ind, 0) > 0 
                for ind in self.confirmation_indicators
            )
            
            if no_sell_signals and has_confirmation:
                return 'STRONG_BUY'
            
        # SELL için kontrol  
        elif primary_sell:
            # Hiçbir onay indikatörü BUY vermemeli
            no_buy_signals = all(
                signals.get(ind, 0) <= 0 
                for ind in self.confirmation_indicators
            )
            
            # En az 1 onay indikatörü SELL vermeli
            has_confirmation = any(
                signals.get(ind, 0) < 0 
                for ind in self.confirmation_indicators
            )
            
            if no_buy_signals and has_confirmation:
                return 'STRONG_SELL'
        
        return 'HOLD'
    
    def calculate_position_size(self, price: float) -> int:
        """Pozisyon büyüklüğü hesapla"""
        position_value = self.current_capital / self.max_positions
        shares = int(position_value / price)
        return max(shares, 1)
    
    def run_backtest(self, symbol: str, price_data: pd.DataFrame, 
                    signal_data: pd.DataFrame) -> List[Dict]:
        """Tek sembol için backtest"""
        symbol_trades = []
        
        # Her mum için döngü
        for idx in range(50, len(price_data)):
            current_time = price_data.index[idx]
            current_price = price_data['close'].iloc[idx]
            
            # Stop loss kontrolü
            if symbol in self.positions:
                position = self.positions[symbol]
                loss_pct = (current_price - position['entry_price']) / position['entry_price']
                
                if loss_pct <= -self.stop_loss_pct:
                    # Stop loss tetiklendi
                    exit_value = current_price * position['shares']
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
                        'entry_signal': position['signal']
                    }
                    
                    symbol_trades.append(trade)
                    self.trades.append(trade)
                    self.current_capital += exit_value
                    del self.positions[symbol]
                    
                    logger.debug(f"{symbol} STOP LOSS @ {current_price:.2f} (Loss: {loss_pct:.1%})")
                    continue
            
            # Sinyal kontrolü
            if idx < len(signal_data):
                current_signals = signal_data.iloc[idx]
                signal = self.check_entry_conditions(current_signals)
                
                # Pozisyon aç
                if signal == 'STRONG_BUY' and symbol not in self.positions:
                    if len(self.positions) < self.max_positions:
                        shares = self.calculate_position_size(current_price)
                        cost = shares * current_price
                        
                        if cost <= self.current_capital:
                            # Hangi indikatörler onay verdi?
                            confirmations = [
                                ind for ind in self.confirmation_indicators
                                if current_signals.get(ind, 0) > 0
                            ]
                            
                            self.positions[symbol] = {
                                'entry_time': current_time,
                                'entry_price': current_price,
                                'shares': shares,
                                'cost': cost,
                                'signal': signal,
                                'confirmations': confirmations
                            }
                            self.current_capital -= cost
                            
                            logger.debug(f"{symbol} BUY: {shares} @ {current_price:.2f} (Confirmations: {confirmations})")
                
                # Pozisyon kapat
                elif signal == 'STRONG_SELL' and symbol in self.positions:
                    position = self.positions[symbol]
                    exit_value = current_price * position['shares']
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
                        'entry_signal': position['signal'],
                        'confirmations': position['confirmations']
                    }
                    
                    symbol_trades.append(trade)
                    self.trades.append(trade)
                    self.current_capital += exit_value
                    del self.positions[symbol]
                    
                    logger.debug(f"{symbol} SELL: {position['shares']} @ {current_price:.2f} (PnL: {pnl:.2f})")
        
        # Test sonu açık pozisyonları kapat
        if symbol in self.positions:
            position = self.positions[symbol]
            final_price = price_data['close'].iloc[-1]
            exit_value = final_price * position['shares']
            pnl = exit_value - position['cost']
            
            trade = {
                'symbol': symbol,
                'entry_time': position['entry_time'],
                'exit_time': price_data.index[-1],
                'entry_price': position['entry_price'],
                'exit_price': final_price,
                'shares': position['shares'],
                'pnl': pnl,
                'pnl_pct': pnl / position['cost'],
                'exit_reason': 'END_TEST',
                'entry_signal': position['signal'],
                'confirmations': position['confirmations']
            }
            
            symbol_trades.append(trade)
            self.trades.append(trade)
            self.current_capital += exit_value
            del self.positions[symbol]
        
        return symbol_trades
    
    def run_all_symbols(self, timeframe: str):
        """Tüm semboller için backtest"""
        logger.info(f"\nSimple Indicator Backtest başlıyor - Timeframe: {timeframe}")
        logger.info(f"Başlangıç Sermayesi: {self.initial_capital:,.0f} TL")
        logger.info(f"Test edilecek hisse sayısı: {len(ASSETS)}")
        logger.info(f"Strateji: Supertrend + Squeeze Momentum + Onay\n")
        
        results = []
        successful_symbols = 0
        
        for i, symbol in enumerate(ASSETS):
            logger.info(f"[{i+1}/{len(ASSETS)}] {symbol} test ediliyor...")
            
            # Fiyat verisi
            price_data = self.csv_manager.load_raw_data(symbol, timeframe)
            if price_data is None or len(price_data) < 100:
                logger.warning(f"{symbol} - Yetersiz veri, atlanıyor")
                continue
            
            # Sinyal verisi
            signal_data = self.load_indicator_signals(symbol, timeframe)
            if signal_data.empty:
                logger.warning(f"{symbol} - İndikatör verisi bulunamadı")
                continue
            
            # İndeksleri hizala
            common_index = price_data.index.intersection(signal_data.index)
            price_data = price_data.loc[common_index]
            signal_data = signal_data.loc[common_index]
            
            # Backtest
            trades = self.run_backtest(symbol, price_data, signal_data)
            
            if trades:
                total_pnl = sum(t['pnl'] for t in trades)
                win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
                avg_pnl = total_pnl / len(trades)
                
                results.append({
                    'symbol': symbol,
                    'trades': len(trades),
                    'total_pnl': total_pnl,
                    'avg_pnl': avg_pnl,
                    'win_rate': win_rate,
                    'trades_detail': trades
                })
                
                successful_symbols += 1
                logger.info(f"  ✓ İşlem: {len(trades)}, Kar/Zarar: {total_pnl:,.2f} TL, Başarı: {win_rate:.1%}")
            else:
                logger.info(f"  - Hiç işlem yapılmadı")
        
        logger.info(f"\nToplam {successful_symbols} sembolde işlem yapıldı")
        
        # Sonuçları göster ve kaydet
        self.print_final_results(results, timeframe)
        self.save_results(results, timeframe)
    
    def print_final_results(self, results: List[Dict], timeframe: str):
        """Detaylı sonuçları yazdır"""
        print("\n" + "="*80)
        print(f"SIMPLE INDICATOR BACKTEST SONUÇLARI - {timeframe}")
        print("Strateji: Supertrend + Squeeze Momentum + Onay Sistemi")
        print("="*80)
        
        # Genel istatistikler
        total_trades = sum(r['trades'] for r in results)
        total_pnl = sum(r['total_pnl'] for r in results)
        final_capital = self.current_capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        profitable_symbols = sum(1 for r in results if r['total_pnl'] > 0)
        avg_win_rate = np.mean([r['win_rate'] for r in results]) if results else 0
        
        print(f"\nGENEL PERFORMANS:")
        print(f"Başlangıç Sermayesi: {self.initial_capital:,.0f} TL")
        print(f"Final Sermaye: {final_capital:,.0f} TL")
        print(f"Toplam Getiri: {total_return:.1%}")
        print(f"\nToplam İşlem: {total_trades}")
        print(f"Toplam Kar/Zarar: {total_pnl:,.2f} TL")
        print(f"Karlı Sembol Sayısı: {profitable_symbols}/{len(results)}")
        print(f"Ortalama Başarı Oranı: {avg_win_rate:.1%}")
        
        # En karlı semboller
        sorted_results = sorted(results, key=lambda x: x['total_pnl'], reverse=True)
        
        print(f"\nEN KARLI 10 SEMBOL:")
        print(f"{'Sembol':<8} {'İşlem':<8} {'Toplam K/Z':<15} {'Ort. K/Z':<12} {'Başarı'}")
        print("-" * 60)
        
        for r in sorted_results[:10]:
            if r['total_pnl'] > 0:
                print(f"{r['symbol']:<8} {r['trades']:<8} "
                      f"{r['total_pnl']:>12,.2f} TL {r['avg_pnl']:>9,.2f} TL "
                      f"{r['win_rate']:>7.1%}")
        
        # En çok işlem yapan semboller
        print(f"\nEN AKTİF 10 SEMBOL (İşlem Sayısı):")
        print(f"{'Sembol':<8} {'İşlem':<8} {'Toplam K/Z':<15} {'Başarı'}")
        print("-" * 50)
        
        active_sorted = sorted(results, key=lambda x: x['trades'], reverse=True)
        for r in active_sorted[:10]:
            print(f"{r['symbol']:<8} {r['trades']:<8} "
                  f"{r['total_pnl']:>12,.2f} TL {r['win_rate']:>7.1%}")
        
        # Onay indikatör istatistikleri
        if self.trades:
            confirmation_stats = {}
            for trade in self.trades:
                if 'confirmations' in trade:
                    for conf in trade['confirmations']:
                        if conf not in confirmation_stats:
                            confirmation_stats[conf] = {'count': 0, 'pnl': 0}
                        confirmation_stats[conf]['count'] += 1
                        confirmation_stats[conf]['pnl'] += trade['pnl']
            
            print(f"\nONAY İNDİKATÖR İSTATİSTİKLERİ:")
            print(f"{'İndikatör':<15} {'Kullanım':<10} {'Toplam K/Z'}")
            print("-" * 40)
            
            for ind, stats in sorted(confirmation_stats.items(), 
                                   key=lambda x: x[1]['count'], reverse=True):
                print(f"{ind:<15} {stats['count']:<10} {stats['pnl']:>12,.2f} TL")
    
    def save_results(self, results: List[Dict], timeframe: str):
        """Sonuçları dosyaya kaydet"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ana özet
        summary = {
            'strategy': 'Simple Indicator - 3 Onay',
            'timeframe': timeframe,
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': (self.current_capital - self.initial_capital) / self.initial_capital,
            'total_trades': len(self.trades),
            'primary_indicators': self.primary_indicators,
            'confirmation_indicators': self.confirmation_indicators,
            'symbol_count': len(results),
            'profitable_symbols': sum(1 for r in results if r['total_pnl'] > 0),
            'timestamp': timestamp
        }
        
        # Symbol bazlı özet
        symbol_summary = []
        for r in results:
            symbol_summary.append({
                'symbol': r['symbol'],
                'trades': r['trades'],
                'total_pnl': r['total_pnl'],
                'avg_pnl': r['avg_pnl'],
                'win_rate': r['win_rate']
            })
        summary['symbol_results'] = symbol_summary
        
        # JSON olarak kaydet
        summary_file = Path(f"backtest/simple_indicator_{timeframe}_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Detaylı trade listesi CSV
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            # Confirmations listesini string'e çevir
            if 'confirmations' in trades_df.columns:
                trades_df['confirmations'] = trades_df['confirmations'].apply(
                    lambda x: ','.join(x) if isinstance(x, list) else x
                )
            
            trades_file = Path(f"backtest/simple_indicator_trades_{timeframe}_{timestamp}.csv")
            trades_df.to_csv(trades_file, index=False)
            
            logger.info(f"\nSonuçlar kaydedildi:")
            logger.info(f"  Özet: {summary_file}")
            logger.info(f"  İşlemler: {trades_file}")


def main():
    # Backtest nesnesi oluştur
    backtest = SimpleIndicatorBacktest(initial_capital=50000, max_positions=10)
    
    # Timeframe seçimi
    timeframe = backtest.get_timeframe_choice()
    
    # Tüm semboller için backtest çalıştır
    backtest.run_all_symbols(timeframe)


if __name__ == "__main__":
    main()