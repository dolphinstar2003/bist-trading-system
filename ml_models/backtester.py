#!/usr/bin/env python3
"""
Backtesting Module
ML modellerinin performansını geçmiş veri üzerinde test eder
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime, timedelta
import json
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Proje imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_models.predictor import Predictor
from ml_models.model_trainer import ModelTrainer
from utils.csv_data_manager import CSVDataManager


class Backtester:
    """ML stratejileri için backtesting sınıfı"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.predictor = Predictor()
        self.csv_manager = CSVDataManager()
        
        # Başlangıç parametreleri
        self.initial_capital = initial_capital
        self.commission_rate = 0.001  # %0.1
        self.slippage = 0.0005  # %0.05
        
        # Sonuçlar
        self.results_dir = Path("data/backtest_results")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Risk yönetimi
        self.max_position_size = 0.1  # Sermayenin maksimum %10'u
        self.stop_loss = 0.02  # %2 stop loss
        self.take_profit = 0.04  # %4 take profit
    
    def calculate_position_size(self, capital: float, price: float, 
                               signal_strength: float = 1.0) -> int:
        """Pozisyon büyüklüğünü hesapla"""
        # Maksimum pozisyon değeri
        max_value = capital * self.max_position_size
        
        # Sinyal gücüne göre ayarla
        position_value = max_value * min(signal_strength, 1.0)
        
        # Lot sayısı (100'lük lotlar)
        lots = int(position_value / (price * 100)) * 100
        
        return max(lots, 100)  # Minimum 1 lot
    
    def apply_commission_and_slippage(self, price: float, side: str) -> float:
        """Komisyon ve slippage uygula"""
        if side == 'BUY':
            # Alımda daha yüksek fiyat
            return price * (1 + self.commission_rate + self.slippage)
        else:
            # Satımda daha düşük fiyat
            return price * (1 - self.commission_rate - self.slippage)
    
    def run_backtest(self, symbol: str, timeframe: str, 
                    start_date: Optional[str] = None, 
                    end_date: Optional[str] = None,
                    model_type: Optional[str] = None) -> Dict:
        """Backtest çalıştır"""
        try:
            logger.info(f"Starting backtest for {symbol} {timeframe}")
            
            # Veriyi yükle
            df = self.csv_manager.load_raw_data(symbol, timeframe)
            if df is None or len(df) < 200:
                logger.error(f"Insufficient data for {symbol} {timeframe}")
                return {}
            
            # Tarih filtresi
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Modeli yükle
            model_data, metadata = self.predictor.load_latest_model(symbol, timeframe, model_type)
            if model_data is None:
                logger.error(f"No model found for {symbol} {timeframe}")
                return {}
            
            # Özellikleri hazırla
            features = self.predictor.prepare_live_features(symbol, timeframe)
            if features.empty:
                return {}
            
            # Veri ve özellikleri hizala
            common_index = df.index.intersection(features.index)
            df = df.loc[common_index]
            features = features.loc[common_index]
            
            # Backtest değişkenleri
            positions = []  # Açık pozisyonlar
            trades = []     # Tamamlanmış işlemler
            portfolio_value = [self.initial_capital]
            cash = self.initial_capital
            
            # Her zaman dilimi için
            for i in range(200, len(df)):  # İlk 200 bar feature hesaplama için
                current_time = df.index[i]
                current_price = df.iloc[i]['close']
                
                # Tahmin yap (son veriyle)
                X = features.iloc[:i+1]
                predictions = self.predictor.predict_single(
                    model_data, X, last_n=1
                )
                
                if predictions.empty:
                    continue
                
                # Sinyal üret
                signals = self.predictor.generate_signals(predictions)
                signal = signals.iloc[-1]['signal']
                signal_strength = signals.iloc[-1]['signal_strength']
                
                # Pozisyon yönetimi
                if signal == 1 and len(positions) == 0:  # AL
                    # Pozisyon büyüklüğü
                    size = self.calculate_position_size(cash, current_price, signal_strength)
                    
                    # Maliyetli fiyat
                    entry_price = self.apply_commission_and_slippage(current_price, 'BUY')
                    
                    # Pozisyon aç
                    if cash >= entry_price * size:
                        position = {
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'size': size,
                            'stop_loss': entry_price * (1 - self.stop_loss),
                            'take_profit': entry_price * (1 + self.take_profit),
                            'signal_strength': signal_strength
                        }
                        positions.append(position)
                        cash -= entry_price * size
                        
                        logger.debug(f"BUY: {size} @ {entry_price:.2f} (signal: {signal_strength:.2f})")
                
                elif signal == -1 and len(positions) > 0:  # SAT
                    # Tüm pozisyonları kapat
                    for position in positions:
                        exit_price = self.apply_commission_and_slippage(current_price, 'SELL')
                        
                        # Kar/Zarar hesapla
                        pnl = (exit_price - position['entry_price']) * position['size']
                        pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                        
                        trade = {
                            'symbol': symbol,
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'size': position['size'],
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'signal_strength': position['signal_strength'],
                            'holding_period': (current_time - position['entry_time']).days
                        }
                        trades.append(trade)
                        cash += exit_price * position['size']
                        
                        logger.debug(f"SELL: {position['size']} @ {exit_price:.2f} (PnL: {pnl:.2f})")
                    
                    positions = []
                
                # Stop Loss / Take Profit kontrolü
                positions_to_close = []
                for i, position in enumerate(positions):
                    if (current_price <= position['stop_loss'] or 
                        current_price >= position['take_profit']):
                        positions_to_close.append(i)
                
                # SL/TP tetiklenen pozisyonları kapat
                for idx in reversed(positions_to_close):
                    position = positions.pop(idx)
                    exit_price = self.apply_commission_and_slippage(current_price, 'SELL')
                    
                    pnl = (exit_price - position['entry_price']) * position['size']
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    
                    reason = 'STOP_LOSS' if current_price <= position['stop_loss'] else 'TAKE_PROFIT'
                    
                    trade = {
                        'symbol': symbol,
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'signal_strength': position['signal_strength'],
                        'holding_period': (current_time - position['entry_time']).days,
                        'exit_reason': reason
                    }
                    trades.append(trade)
                    cash += exit_price * position['size']
                    
                    logger.debug(f"{reason}: {position['size']} @ {exit_price:.2f} (PnL: {pnl:.2f})")
                
                # Portföy değerini hesapla
                position_value = sum(p['size'] * current_price for p in positions)
                total_value = cash + position_value
                portfolio_value.append(total_value)
            
            # Kalan pozisyonları kapat
            if positions:
                final_price = df.iloc[-1]['close']
                for position in positions:
                    exit_price = self.apply_commission_and_slippage(final_price, 'SELL')
                    
                    pnl = (exit_price - position['entry_price']) * position['size']
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
                    
                    trade = {
                        'symbol': symbol,
                        'entry_time': position['entry_time'],
                        'exit_time': df.index[-1],
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'size': position['size'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'signal_strength': position['signal_strength'],
                        'holding_period': (df.index[-1] - position['entry_time']).days,
                        'exit_reason': 'END_OF_BACKTEST'
                    }
                    trades.append(trade)
            
            # Performans metrikleri hesapla
            results = self.calculate_performance_metrics(
                trades, portfolio_value, self.initial_capital, df
            )
            
            # Model bilgilerini ekle
            results['model_info'] = {
                'model_name': model_data.get('model_name', 'unknown'),
                'model_type': model_data.get('model_type', 'unknown'),
                'features_used': metadata.get('features', [])
            }
            
            # Sonuçları kaydet
            self.save_results(results, symbol, timeframe)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def calculate_performance_metrics(self, trades: List[Dict], 
                                    portfolio_value: List[float],
                                    initial_capital: float,
                                    price_data: pd.DataFrame) -> Dict:
        """Performans metriklerini hesapla"""
        
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'message': 'No trades executed'
            }
        
        # Temel metrikler
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Kar/Zarar
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Return metrikleri
        final_value = portfolio_value[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # Portföy değeri serisi
        portfolio_series = pd.Series(portfolio_value[1:], index=price_data.index[200:])
        returns = portfolio_series.pct_change().dropna()
        
        # Sharpe Ratio (yıllık)
        if len(returns) > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Holding period istatistikleri
        avg_holding_period = trades_df['holding_period'].mean()
        max_holding_period = trades_df['holding_period'].max()
        
        # Sinyal gücü analizi
        avg_signal_strength_winners = trades_df[trades_df['pnl'] > 0]['signal_strength'].mean() if winning_trades > 0 else 0
        avg_signal_strength_losers = trades_df[trades_df['pnl'] < 0]['signal_strength'].mean() if losing_trades > 0 else 0
        
        results = {
            # Genel
            'symbol': trades_df['symbol'].iloc[0] if len(trades_df) > 0 else '',
            'start_date': str(price_data.index[200]),
            'end_date': str(price_data.index[-1]),
            'initial_capital': initial_capital,
            'final_capital': final_value,
            
            # İşlem istatistikleri
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            
            # Kar/Zarar
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            
            # Return metrikleri
            'total_return': total_return,
            'annualized_return': total_return * 252 / len(price_data[200:]) if len(price_data[200:]) > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            
            # Holding period
            'avg_holding_days': avg_holding_period,
            'max_holding_days': max_holding_period,
            
            # Sinyal analizi
            'avg_signal_strength_winners': avg_signal_strength_winners,
            'avg_signal_strength_losers': avg_signal_strength_losers,
            
            # Detaylı işlemler
            'trades': trades
        }
        
        return results
    
    def save_results(self, results: Dict, symbol: str, timeframe: str):
        """Backtest sonuçlarını kaydet"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Ana sonuçlar
        results_file = self.results_dir / f"{symbol}_{timeframe}_backtest_{timestamp}.json"
        
        # Trade detaylarını ayrı kaydet
        trades = results.pop('trades', [])
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Trades CSV
        if trades:
            trades_df = pd.DataFrame(trades)
            trades_file = self.results_dir / f"{symbol}_{timeframe}_trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
        
        logger.info(f"Backtest results saved: {results_file}")
        
        # Özet rapor
        self.print_summary(results)
    
    def print_summary(self, results: Dict):
        """Backtest özetini yazdır"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Symbol: {results.get('symbol', 'N/A')}")
        print(f"Period: {results['start_date']} to {results['end_date']}")
        print(f"\nCapital:")
        print(f"  Initial: ${results['initial_capital']:,.0f}")
        print(f"  Final:   ${results['final_capital']:,.0f}")
        print(f"  Return:  {results['total_return']:.1%}")
        print(f"\nTrading Statistics:")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate']:.1%}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.1%}")
        print(f"\nAverage Trade:")
        print(f"  PnL: ${results['avg_pnl']:.2f}")
        print(f"  Win: ${results['avg_win']:.2f}")
        print(f"  Loss: ${results['avg_loss']:.2f}")
        print(f"  Holding Period: {results['avg_holding_days']:.1f} days")
        print("="*60)
    
    def run_multi_backtest(self, symbols: List[str], timeframes: List[str],
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """Birden fazla sembol/timeframe için backtest"""
        all_results = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                logger.info(f"\nRunning backtest for {symbol} {timeframe}")
                
                results = self.run_backtest(symbol, timeframe, start_date, end_date)
                
                if results and results.get('total_trades', 0) > 0:
                    # Özet satır
                    summary = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'total_return': results['total_return'],
                        'win_rate': results['win_rate'],
                        'sharpe_ratio': results['sharpe_ratio'],
                        'max_drawdown': results['max_drawdown'],
                        'total_trades': results['total_trades'],
                        'profit_factor': results['profit_factor']
                    }
                    all_results.append(summary)
        
        # Sonuçları DataFrame'e dönüştür
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # En iyileri sırala
            results_df = results_df.sort_values('sharpe_ratio', ascending=False)
            
            # Rapor kaydet
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = self.results_dir / f"multi_backtest_report_{timestamp}.csv"
            results_df.to_csv(report_file, index=False)
            
            print("\n" + "="*80)
            print("MULTI-BACKTEST SUMMARY")
            print("="*80)
            print(results_df.to_string())
            print("="*80)
            
            return results_df
        
        return pd.DataFrame()


def main():
    """Test backtesting"""
    backtester = Backtester(initial_capital=100000)
    
    # Tek backtest
    symbol = "AKBNK"
    timeframe = "1h"
    
    results = backtester.run_backtest(
        symbol, timeframe,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    
    # Multi backtest
    print("\nRunning multi-backtest...")
    
    symbols = ["AKBNK", "THYAO", "GARAN"]
    timeframes = ["1h", "4h"]
    
    multi_results = backtester.run_multi_backtest(
        symbols, timeframes,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )


if __name__ == "__main__":
    main()