#!/usr/bin/env python3
"""
Momentum Burst Catcher System - Kısa süreli güçlü momentumları yakala

Strateji:
1. Bollinger Band sıkışması sonrası patlama
2. Volume spike (3x+) 
3. RSI momentum burst (50'den 65+'e hızlı çıkış)
4. Hızlı kar al (%5-10 arası)
5. Sıkı stop (%3-5)

Hedef: Aylık %5-8 (Banka faizinin 2 katı)
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

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')


@dataclass
class BurstSignal:
    """Momentum burst sinyali"""
    symbol: str
    date: pd.Timestamp
    entry_price: float
    
    # Sinyal güçleri
    bb_squeeze_days: int  # Kaç gündür sıkışmada
    volume_spike: float   # Volume çarpanı
    rsi_change: float    # RSI değişimi (5 gün)
    price_breakout: float # BB üst bandı kırılım %
    
    # Risk parametreleri
    stop_loss: float     # Entry'den % olarak
    take_profit: float   # Entry'den % olarak
    
    # Skor
    burst_score: float


class MomentumBurstCatcher:
    """Kısa süreli momentum patlamalarını yakala"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.total_value = initial_capital
        
        # Sistem parametreleri
        self.position_size_pct = 0.20  # %20 pozisyon (konsantre)
        self.max_positions = 5         # Max 5 pozisyon
        self.commission = 0.002
        
        # Burst parametreleri
        self.min_bb_squeeze_days = 3   # En az 3 gün BB daralması
        self.min_volume_spike = 2.0    # En az 2x volume
        self.min_rsi_change = 7        # RSI 7+ puan artış
        self.min_burst_score = 6.0     # Minimum giriş skoru
        
        # Risk parametreleri
        self.base_stop_loss = 0.04     # %4 base stop
        self.base_take_profit = 0.08   # %8 base target
        self.trailing_activation = 0.05 # %5'te trailing aktif
        self.trailing_distance = 0.025  # %2.5 trailing mesafe
        
        # Portfolio
        self.positions = {}
        self.pending_signals = []
        self.completed_trades = []
        self.transaction_history = []
        self.portfolio_history = []
        
        # Hisse evreni ve parametreler
        self.load_universe()
        
    def load_universe(self):
        """Yüksek momentum potansiyelli hisseler"""
        # Volatil ama likit hisseler
        self.universe = [
            'THYAO', 'GARAN', 'AKBNK', 'ISCTR', 'YKBNK',  # Bankalar
            'ASELS', 'TUPRS', 'EREGL', 'SAHOL', 'PGSUS',  # Sanayi
            'SISE', 'ARCLK', 'FROTO', 'TOASO', 'PETKM',   # Üretim
            'EKGYO', 'SASA', 'KOZAL', 'BIMAS', 'TCELL',   # Diğer
            'DOHOL', 'TAVHL', 'KCHOL', 'ENKAI', 'AEFES'   # Holding
        ]
        
        # Optimal parametreler
        self.load_optimal_parameters()
        
    def load_optimal_parameters(self):
        """Stop loss ve volatilite parametrelerini yükle"""
        self.stock_params = {}
        
        try:
            # Advanced parameters
            path = 'data/analysis/advanced_trading_parameters.csv'
            if os.path.exists(path):
                df = pd.read_csv(path)
                
                for _, row in df.iterrows():
                    symbol = row['symbol']
                    if symbol in self.universe:
                        self.stock_params[symbol] = {
                            'volatility': row['annual_volatility'] / 100,
                            'avg_rally': row['avg_rally'] / 100,
                            'optimal_stop': row['optimal_stop_loss'] / 100,
                            'tight_stop': row['tight_stop_loss'] / 100
                        }
                        
                logger.info(f"Loaded parameters for {len(self.stock_params)} stocks")
                
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            
    def scan_for_bursts(self, date: pd.Timestamp) -> List[BurstSignal]:
        """Momentum burst fırsatlarını tara"""
        signals = []
        
        for symbol in self.universe:
            # Pozisyonda mı?
            if symbol in self.positions:
                continue
                
            # Data yükle
            df = self.load_data(symbol)
            if df.empty or date not in df.index:
                continue
                
            # Son 30 günlük data
            hist = df[df.index <= date].tail(30)
            if len(hist) < 20:
                continue
                
            # Burst sinyali ara
            signal = self.detect_burst_signal(symbol, hist, date)
            if signal and signal.burst_score >= self.min_burst_score:
                signals.append(signal)
                
        # Skora göre sırala
        signals.sort(key=lambda x: x.burst_score, reverse=True)
        
        return signals
        
    def detect_burst_signal(self, symbol: str, hist: pd.DataFrame, date: pd.Timestamp) -> Optional[BurstSignal]:
        """Momentum burst tespiti"""
        try:
            current = hist.iloc[-1]
            
            # 1. Bollinger Band analizi
            sma20 = hist['close'].rolling(20).mean()
            std20 = hist['close'].rolling(20).std()
            upper_band = sma20 + (std20 * 2)
            lower_band = sma20 - (std20 * 2)
            
            # BB genişliği (squeeze tespiti)
            bb_width = (upper_band - lower_band) / sma20
            squeeze_days = 0
            
            # Son 10 günde kaç gün daralma var?
            for i in range(-10, -1):
                if i >= -len(bb_width):
                    if bb_width.iloc[i] < bb_width.iloc[-11]:  # Daralmada
                        squeeze_days += 1
                        
            # Kırılım kontrolü
            price_vs_upper = (current['close'] - upper_band.iloc[-1]) / upper_band.iloc[-1]
            if price_vs_upper <= 0:  # Üst bandı kırmamış
                return None
                
            # 2. Volume analizi
            avg_volume = hist['volume'].rolling(20).mean().iloc[-1]
            volume_spike = current['volume'] / avg_volume if avg_volume > 0 else 1
            
            if volume_spike < self.min_volume_spike:
                return None
                
            # 3. RSI momentum
            rsi_current = self.calculate_rsi(hist['close'])
            rsi_5d_ago = self.calculate_rsi(hist['close'].iloc[:-5]) if len(hist) > 5 else 50
            rsi_change = rsi_current - rsi_5d_ago
            
            if rsi_change < self.min_rsi_change:
                return None
                
            # 4. Price momentum
            momentum_5d = (current['close'] / hist['close'].iloc[-6] - 1) if len(hist) > 5 else 0
            
            # 5. Burst skoru hesapla
            score = 0
            
            # Squeeze kalitesi (ne kadar sıkışmış)
            if squeeze_days >= 7:
                score += 3
            elif squeeze_days >= 5:
                score += 2
            elif squeeze_days >= 3:
                score += 1
                
            # Volume gücü
            if volume_spike >= 4:
                score += 3
            elif volume_spike >= 3:
                score += 2
            elif volume_spike >= 2.5:
                score += 1
                
            # RSI momentum
            if rsi_change >= 20:
                score += 3
            elif rsi_change >= 15:
                score += 2
            elif rsi_change >= 10:
                score += 1
                
            # Price breakout gücü
            if price_vs_upper >= 0.02:  # %2+ kırılım
                score += 2
            elif price_vs_upper > 0:
                score += 1
                
            # Momentum bonus
            if momentum_5d >= 0.05:  # 5 günde %5+
                score += 1
                
            # Risk ayarlaması
            params = self.stock_params.get(symbol, {})
            volatility = params.get('volatility', 0.4)
            
            # Stop loss: Tight stop kullan (burst için)
            stop_loss = params.get('tight_stop', self.base_stop_loss)
            stop_loss = min(stop_loss, self.base_stop_loss * 1.5)  # Max %6
            
            # Take profit: Volatiliteye göre ayarla
            if volatility < 0.3:  # Düşük volatilite
                take_profit = self.base_take_profit * 0.8
            elif volatility > 0.5:  # Yüksek volatilite
                take_profit = self.base_take_profit * 1.2
            else:
                take_profit = self.base_take_profit
                
            # Minimum kar hedefi
            take_profit = max(take_profit, stop_loss * 1.5)  # Min 1.5 R/R
            
            return BurstSignal(
                symbol=symbol,
                date=date,
                entry_price=current['close'],
                bb_squeeze_days=squeeze_days,
                volume_spike=volume_spike,
                rsi_change=rsi_change,
                price_breakout=price_vs_upper * 100,
                stop_loss=stop_loss,
                take_profit=take_profit,
                burst_score=score
            )
            
        except Exception as e:
            logger.error(f"Error detecting burst for {symbol}: {e}")
            return None
            
    def enter_position(self, signal: BurstSignal) -> bool:
        """Burst pozisyonuna gir"""
        # Sermaye kontrolü
        position_value = self.total_value * self.position_size_pct
        if self.cash < position_value * (1 + self.commission):
            return False
            
        price = signal.entry_price
        shares = int(position_value / price)
        if shares == 0:
            return False
            
        cost = shares * price * (1 + self.commission)
        
        # Pozisyon oluştur
        self.positions[signal.symbol] = {
            'shares': shares,
            'entry_price': price,
            'entry_date': signal.date,
            'cost': cost,
            'stop_price': price * (1 - signal.stop_loss),
            'target_price': price * (1 + signal.take_profit),
            'trail_active': False,
            'trail_stop': None,
            'highest_price': price,
            'signal': signal
        }
        
        self.cash -= cost
        
        # Log
        logger.info(f"{signal.date.date()} BURST ENTRY {signal.symbol} @ {price:.2f} " +
                   f"(Score: {signal.burst_score:.1f}, Vol: {signal.volume_spike:.1f}x, " +
                   f"RSI: +{signal.rsi_change:.0f}, Stop: -{signal.stop_loss*100:.1f}%, " +
                   f"Target: +{signal.take_profit*100:.1f}%)")
                   
        self.transaction_history.append({
            'date': signal.date,
            'type': 'BURST_ENTRY',
            'symbol': signal.symbol,
            'shares': shares,
            'price': price,
            'signal_score': signal.burst_score
        })
        
        return True
        
    def manage_positions(self, date: pd.Timestamp):
        """Pozisyon yönetimi"""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            
            # Güncel fiyat
            df = self.load_data(symbol)
            if df.empty or date not in df.index:
                continue
                
            current_price = df.loc[date, 'close']
            entry_price = pos['entry_price']
            pnl_pct = (current_price - entry_price) / entry_price
            
            # Stop loss kontrolü
            if current_price <= pos['stop_price']:
                self.exit_position(symbol, date, current_price, "STOP_LOSS")
                continue
                
            # Target kontrolü
            if current_price >= pos['target_price']:
                self.exit_position(symbol, date, current_price, "TARGET_HIT")
                continue
                
            # Trailing stop yönetimi
            if pnl_pct >= self.trailing_activation and not pos['trail_active']:
                # Trailing aktif et
                pos['trail_active'] = True
                pos['trail_stop'] = current_price * (1 - self.trailing_distance)
                logger.info(f"{date.date()} {symbol} trailing stop activated at {pos['trail_stop']:.2f}")
                
            if pos['trail_active']:
                # Yeni high?
                if current_price > pos['highest_price']:
                    pos['highest_price'] = current_price
                    pos['trail_stop'] = current_price * (1 - self.trailing_distance)
                    
                # Trailing stop hit?
                if current_price <= pos['trail_stop']:
                    self.exit_position(symbol, date, current_price, "TRAILING_STOP")
                    continue
                    
            # Momentum kaybı kontrolü (opsiyonel hızlı çıkış)
            if pnl_pct > 0.03:  # %3+ karda
                # Son 3 günlük momentum
                recent = df[df.index <= date].tail(3)
                if len(recent) == 3:
                    momentum = (recent['close'].iloc[-1] / recent['close'].iloc[0] - 1)
                    if momentum < -0.02:  # 3 günde %2 kayıp
                        self.exit_position(symbol, date, current_price, "MOMENTUM_LOSS")
                        
    def exit_position(self, symbol: str, date: pd.Timestamp, price: float, reason: str):
        """Pozisyondan çık"""
        pos = self.positions[symbol]
        
        revenue = pos['shares'] * price * (1 - self.commission)
        profit = revenue - pos['cost']
        pnl_pct = (price - pos['entry_price']) / pos['entry_price'] * 100
        days_held = (date - pos['entry_date']).days
        
        self.cash += revenue
        
        # Trade kaydı
        self.completed_trades.append({
            'symbol': symbol,
            'entry_date': pos['entry_date'],
            'exit_date': date,
            'entry_price': pos['entry_price'],
            'exit_price': price,
            'shares': pos['shares'],
            'profit': profit,
            'pnl_pct': pnl_pct,
            'days_held': days_held,
            'exit_reason': reason,
            'signal_score': pos['signal'].burst_score
        })
        
        self.transaction_history.append({
            'date': date,
            'type': 'BURST_EXIT',
            'symbol': symbol,
            'shares': pos['shares'],
            'price': price,
            'profit': profit,
            'reason': reason
        })
        
        del self.positions[symbol]
        
        logger.info(f"{date.date()} BURST EXIT {symbol} @ {price:.2f} " +
                   f"({pnl_pct:+.1f}% in {days_held} days, Reason: {reason})")
                   
    def execute_daily_trading(self, date: pd.Timestamp):
        """Günlük trading"""
        # 1. Mevcut pozisyonları yönet
        self.manage_positions(date)
        
        # 2. Yeni burst sinyalleri ara
        if len(self.positions) < self.max_positions:
            signals = self.scan_for_bursts(date)
            
            # En iyi sinyalleri al
            for signal in signals:
                if len(self.positions) >= self.max_positions:
                    break
                    
                # Giriş yap
                if self.enter_position(signal):
                    # Günde max 2 yeni pozisyon
                    if sum(1 for t in self.transaction_history 
                          if t['date'] == date and t['type'] == 'BURST_ENTRY') >= 2:
                        break
                        
        # 3. Portfolio değerini güncelle
        self.update_portfolio_value(date)
        
    def update_portfolio_value(self, date: pd.Timestamp):
        """Portfolio değerini hesapla"""
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
            'num_positions': len(self.positions)
        })
        
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
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
        
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Hisse verisini yükle"""
        try:
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
        logger.info(f"Starting Momentum Burst backtest from {start_date} to {end_date}")
        logger.info(f"Initial capital: {self.initial_capital:,.0f} TL")
        logger.info(f"Strategy: Short-term momentum bursts with tight stops")
        
        # Trading günleri
        sample_df = self.load_data(self.universe[0])
        if sample_df.empty:
            logger.error("No data available")
            return
            
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        trading_dates = sample_df[(sample_df.index >= start) & (sample_df.index <= end)].index
        
        logger.info(f"Trading days: {len(trading_dates)}")
        
        # Backtest döngüsü
        for i, date in enumerate(trading_dates):
            if i < 20:  # İlk 20 gün veri toplama
                continue
                
            # Günlük trading
            self.execute_daily_trading(date)
            
            # Progress
            if (i - 20 + 1) % 20 == 0:
                logger.info(f"Day {i-20+1}: Portfolio {self.total_value:,.0f} TL " +
                           f"(+{(self.total_value/self.initial_capital-1)*100:.1f}%), " +
                           f"Positions: {len(self.positions)}/{self.max_positions}")
                           
    def print_results(self):
        """Sonuçları yazdır"""
        if not self.portfolio_history:
            logger.error("No results to display")
            return
            
        df = pd.DataFrame(self.portfolio_history)
        
        initial = self.initial_capital
        final = self.total_value
        total_return = (final - initial) / initial * 100
        
        print(f"\n{'='*80}")
        print(f"MOMENTUM BURST CATCHER RESULTS")
        print(f"{'='*80}")
        print(f"Initial Capital: {initial:,.0f} TL")
        print(f"Final Value: {final:,.0f} TL")
        print(f"Total Return: {total_return:.1f}%")
        
        # Trade analizi
        if self.completed_trades:
            trades_df = pd.DataFrame(self.completed_trades)
            
            print(f"\n{'='*50}")
            print(f"TRADE ANALYSIS:")
            print(f"Total trades: {len(trades_df)}")
            print(f"Average holding period: {trades_df['days_held'].mean():.1f} days")
            print(f"Average return per trade: {trades_df['pnl_pct'].mean():.1f}%")
            
            # Win rate
            winners = len(trades_df[trades_df['profit'] > 0])
            win_rate = winners / len(trades_df) * 100 if len(trades_df) > 0 else 0
            print(f"Win rate: {win_rate:.1f}%")
            
            # Exit reasons
            print(f"\n{'='*50}")
            print("EXIT REASONS:")
            exit_counts = trades_df['exit_reason'].value_counts()
            for reason, count in exit_counts.items():
                pct = count / len(trades_df) * 100
                print(f"{reason}: {count} ({pct:.1f}%)")
                
            # Best trades
            print(f"\n{'='*50}")
            print("TOP 5 TRADES:")
            print(f"{'Symbol':<8} {'Entry':>8} {'Exit':>8} {'Days':>6} {'Return':>8}")
            print(f"{'-'*50}")
            
            top_trades = trades_df.nlargest(5, 'pnl_pct')
            for _, trade in top_trades.iterrows():
                print(f"{trade['symbol']:<8} {trade['entry_price']:>8.2f} " +
                     f"{trade['exit_price']:>8.2f} {trade['days_held']:>6} " +
                     f"{trade['pnl_pct']:>7.1f}%")
                     
        # Aylık getiriler
        df['month'] = pd.to_datetime(df['date']).dt.to_period('M')
        monthly = df.groupby('month')['total_value'].agg(['first', 'last'])
        monthly['return'] = (monthly['last'] - monthly['first']) / monthly['first'] * 100
        
        print(f"\n{'='*50}")
        print("MONTHLY RETURNS:")
        for month, ret in monthly['return'].items():
            status = "✓✓" if ret >= 5 else "✓" if ret >= 3 else "○"
            print(f"{month}: {ret:>6.1f}% {status}")
            
        avg_monthly = monthly['return'].mean()
        print(f"\nAverage Monthly: {avg_monthly:.1f}%")
        print(f"Annualized: {avg_monthly * 12:.1f}%")
        
        # Risk
        returns = df['total_value'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe = (total_return / 100) / (volatility / 100) * np.sqrt(252)
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        print(f"\n{'='*50}")
        print("RISK METRICS:")
        print(f"Volatility: {volatility:.1f}%")
        print(f"Max Drawdown: {max_drawdown:.1f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        
        # Banka karşılaştırması
        print(f"\n{'='*50}")
        print("BANK COMPARISON:")
        bank_monthly = 3.4
        print(f"Bank monthly rate: {bank_monthly:.1f}%")
        print(f"Strategy monthly avg: {avg_monthly:.1f}%")
        print(f"Outperformance: {avg_monthly - bank_monthly:.1f}% per month")
        
        print(f"{'='*80}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Momentum Burst Catcher')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='End date')
    
    args = parser.parse_args()
    
    # Create and run system
    catcher = MomentumBurstCatcher(initial_capital=args.capital)
    
    try:
        catcher.run_backtest(args.start_date, args.end_date)
        catcher.print_results()
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()