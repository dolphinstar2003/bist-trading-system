"""
Portfolio Manager Module
Portfolio yönetimi, pozisyon takibi ve risk kontrolü
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from loguru import logger
import json
from pathlib import Path
from collections import defaultdict
import asyncio


class PortfolioManager:
    """Portfolio ve pozisyon yönetimi"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Portfolio parametreleri
        self.initial_capital = config['portfolio']['initial_capital']
        self.current_capital = self.initial_capital
        self.max_positions = config['portfolio']['max_positions']
        self.max_risk_per_trade = config['risk']['max_risk_per_trade']
        self.max_portfolio_risk = config['risk']['max_portfolio_risk']
        self.max_correlation = config['risk']['max_correlation']
        
        # Pozisyonlar
        self.positions = {}  # symbol -> position dict
        self.closed_positions = []  # Geçmiş pozisyonlar
        self.pending_orders = {}  # Bekleyen emirler
        
        # Performance metrics
        self.performance = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'peak_equity': self.initial_capital,
            'sharpe_ratio': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
        
        # Risk tracking
        self.daily_pnl = []
        self.equity_curve = [self.initial_capital]
        self.last_update = datetime.now()
        
        # Persistence
        self.portfolio_file = Path("data/portfolio_state.json")
        self.trades_file = Path("data/trade_history.csv")
        self._load_state()
        
        logger.info(f"PortfolioManager başlatıldı - Capital: {self.initial_capital:,.0f}")
    
    def _load_state(self):
        """Portfolio durumunu yükle"""
        try:
            if self.portfolio_file.exists():
                with open(self.portfolio_file, 'r') as f:
                    state = json.load(f)
                    
                self.current_capital = state.get('current_capital', self.initial_capital)
                self.positions = state.get('positions', {})
                self.performance = state.get('performance', self.performance)
                self.equity_curve = state.get('equity_curve', [self.initial_capital])
                
                logger.info(f"Portfolio state yüklendi - {len(self.positions)} açık pozisyon")
                
        except Exception as e:
            logger.error(f"Portfolio state yükleme hatası: {e}")
    
    def _save_state(self):
        """Portfolio durumunu kaydet"""
        try:
            state = {
                'current_capital': self.current_capital,
                'positions': self.positions,
                'performance': self.performance,
                'equity_curve': self.equity_curve[-100:],  # Son 100 nokta
                'last_update': self.last_update.isoformat()
            }
            
            self.portfolio_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.portfolio_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Portfolio state kaydetme hatası: {e}")
    
    async def process_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Sinyal işle ve emir oluştur"""
        symbol = signal['symbol']
        
        # Pozisyon kontrolü
        if symbol in self.positions:
            logger.debug(f"{symbol} için zaten pozisyon var")
            return None
        
        # Max pozisyon kontrolü
        if len(self.positions) >= self.max_positions:
            logger.debug(f"Max pozisyon sayısına ulaşıldı: {self.max_positions}")
            return None
        
        # Risk kontrolü
        if not self._check_risk_limits(signal):
            logger.debug(f"{symbol} risk limitleri aşıldı")
            return None
        
        # Korelasyon kontrolü
        if not await self._check_correlation(symbol):
            logger.debug(f"{symbol} korelasyon limiti aşıldı")
            return None
        
        # Pozisyon büyüklüğü hesapla
        position_size = self._calculate_position_size(signal)
        
        if position_size <= 0:
            logger.debug(f"{symbol} pozisyon büyüklüğü yetersiz")
            return None
        
        # Emir oluştur
        order = {
            'symbol': symbol,
            'side': 'buy' if signal['direction'] == 'buy' else 'sell',
            'quantity': position_size,
            'order_type': 'limit',
            'price': signal['entry_price'],
            'stop_loss': signal['stop_loss'],
            'take_profit': signal['target_1'],
            'signal': signal,
            'created_at': datetime.now(),
            'status': 'pending'
        }
        
        # Pending order olarak kaydet
        self.pending_orders[symbol] = order
        
        logger.info(f"Emir oluşturuldu: {symbol} {order['side']} {position_size} @ {order['price']:.2f}")
        
        return order
    
    def _calculate_position_size(self, signal: Dict[str, Any]) -> int:
        """Kelly Criterion ile pozisyon büyüklüğü hesapla"""
        try:
            # Risk miktarı (sermayenin %1'i)
            risk_amount = self.current_capital * self.max_risk_per_trade
            
            # Stop mesafesi
            entry_price = signal['entry_price']
            stop_loss = signal['stop_loss']
            stop_distance = abs(entry_price - stop_loss)
            
            if stop_distance == 0:
                return 0
            
            # Pozisyon büyüklüğü
            position_value = risk_amount / (stop_distance / entry_price)
            
            # Kelly fraction uygula (raporda %25 önerilmiş)
            kelly_fraction = 0.25
            win_rate = self.performance.get('win_rate', 0.5)
            avg_win = self.performance.get('avg_win', 2.0)
            avg_loss = self.performance.get('avg_loss', 1.0)
            
            if avg_loss > 0:
                kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_pct = max(0, min(kelly_pct, 0.25))  # 0-25% arası
            else:
                kelly_pct = 0.01  # Default %1
            
            # Final pozisyon büyüklüğü
            position_value = position_value * kelly_fraction * kelly_pct
            
            # Lot sayısına çevir (100'lük lotlar)
            lot_size = int(position_value / entry_price / 100) * 100
            
            # Min/Max kontrol
            min_lot = 100
            max_lot = int(self.current_capital * 0.1 / entry_price / 100) * 100  # Max %10 sermaye
            
            return max(min_lot, min(lot_size, max_lot))
            
        except Exception as e:
            logger.error(f"Pozisyon hesaplama hatası: {e}")
            return 0
    
    def _check_risk_limits(self, signal: Dict[str, Any]) -> bool:
        """Risk limitlerini kontrol et"""
        # Mevcut açık risk
        current_risk = self._calculate_portfolio_risk()
        
        # Yeni pozisyonla toplam risk
        new_risk = abs(signal['entry_price'] - signal['stop_loss']) / signal['entry_price']
        total_risk = current_risk + new_risk * self.max_risk_per_trade
        
        # Portfolio risk limiti
        if total_risk > self.max_portfolio_risk:
            logger.warning(f"Portfolio risk limiti aşılıyor: {total_risk:.1%} > {self.max_portfolio_risk:.1%}")
            return False
        
        # Drawdown kontrolü
        if self.performance['current_drawdown'] > 0.08:  # %8 drawdown (raporda aylık limit)
            logger.warning(f"Drawdown limiti aşıldı: {self.performance['current_drawdown']:.1%}")
            return False
        
        return True
    
    def _calculate_portfolio_risk(self) -> float:
        """Mevcut portfolio riskini hesapla"""
        total_risk = 0.0
        
        for symbol, position in self.positions.items():
            # Her pozisyonun riski
            position_risk = abs(position['current_price'] - position['stop_loss']) / position['current_price']
            position_weight = (position['quantity'] * position['current_price']) / self.current_capital
            total_risk += position_risk * position_weight
        
        return total_risk
    
    async def _check_correlation(self, symbol: str) -> bool:
        """Pozisyonlar arası korelasyon kontrolü"""
        if len(self.positions) == 0:
            return True
        
        try:
            # Basit sektör bazlı korelasyon kontrolü
            # (Gerçek implementasyonda historical correlation hesaplanmalı)
            
            # Aynı sektörden max 2 pozisyon
            sector_counts = defaultdict(int)
            
            # Sektör mapping (örnek)
            sectors = {
                'AKBNK': 'banking', 'GARAN': 'banking', 'ISCTR': 'banking', 'YKBNK': 'banking',
                'THYAO': 'transport', 'TCELL': 'telecom', 'TTKOM': 'telecom',
                'EREGL': 'steel', 'KRDMD': 'steel',
                # ... diğer semboller
            }
            
            current_sector = sectors.get(symbol, 'other')
            
            for pos_symbol in self.positions:
                pos_sector = sectors.get(pos_symbol, 'other')
                sector_counts[pos_sector] += 1
            
            if sector_counts[current_sector] >= 2:
                logger.debug(f"{symbol} - Aynı sektörde çok fazla pozisyon")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Korelasyon kontrol hatası: {e}")
            return True
    
    def execute_order(self, order: Dict[str, Any], execution_price: float) -> bool:
        """Emri execute et ve pozisyon oluştur"""
        try:
            symbol = order['symbol']
            
            # Pozisyon oluştur
            position = {
                'symbol': symbol,
                'side': order['side'],
                'quantity': order['quantity'],
                'entry_price': execution_price,
                'current_price': execution_price,
                'stop_loss': order['stop_loss'],
                'take_profit': order['take_profit'],
                'opened_at': datetime.now(),
                'pnl': 0.0,
                'pnl_percent': 0.0,
                'status': 'open',
                'signal': order['signal']
            }
            
            # Portfolio güncelle
            self.positions[symbol] = position
            self.performance['total_trades'] += 1
            
            # Capital güncelle
            position_value = order['quantity'] * execution_price
            self.current_capital -= position_value  # Nakit azalt
            
            # Pending'den kaldır
            if symbol in self.pending_orders:
                del self.pending_orders[symbol]
            
            # State kaydet
            self._save_state()
            
            logger.info(f"Pozisyon açıldı: {symbol} {order['side']} {order['quantity']} @ {execution_price:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Order execution hatası: {e}")
            return False
    
    def update_position(self, symbol: str, current_price: float, 
                       volume: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Pozisyon güncelle"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position['current_price'] = current_price
        
        # PnL hesapla
        if position['side'] == 'buy':
            pnl = (current_price - position['entry_price']) * position['quantity']
            pnl_percent = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl = (position['entry_price'] - current_price) * position['quantity']
            pnl_percent = (position['entry_price'] - current_price) / position['entry_price']
        
        position['pnl'] = pnl
        position['pnl_percent'] = pnl_percent
        
        # Stop/Target kontrol
        action = None
        
        if position['side'] == 'buy':
            if current_price <= position['stop_loss']:
                action = 'stop_loss'
            elif current_price >= position['take_profit']:
                action = 'take_profit'
        else:
            if current_price >= position['stop_loss']:
                action = 'stop_loss'
            elif current_price <= position['take_profit']:
                action = 'take_profit'
        
        if action:
            return self.close_position(symbol, current_price, action)
        
        return position
    
    def close_position(self, symbol: str, close_price: float, 
                      reason: str = 'manual') -> Optional[Dict[str, Any]]:
        """Pozisyonu kapat"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Final PnL
        if position['side'] == 'buy':
            pnl = (close_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - close_price) * position['quantity']
        
        position['pnl'] = pnl
        position['close_price'] = close_price
        position['closed_at'] = datetime.now()
        position['close_reason'] = reason
        position['status'] = 'closed'
        
        # Performance güncelle
        self.performance['total_pnl'] += pnl
        
        if pnl > 0:
            self.performance['winning_trades'] += 1
        else:
            self.performance['losing_trades'] += 1
        
        # Capital güncelle
        position_value = position['quantity'] * close_price
        self.current_capital += position_value
        
        # Equity curve güncelle
        self.equity_curve.append(self.current_capital)
        
        # Performance metrics güncelle
        self._update_performance_metrics()
        
        # Pozisyonu geçmişe taşı
        self.closed_positions.append(position)
        del self.positions[symbol]
        
        # Trade history'ye kaydet
        self._save_trade_history(position)
        
        # State kaydet
        self._save_state()
        
        logger.info(f"Pozisyon kapatıldı: {symbol} - PnL: {pnl:.2f} ({pnl/position['quantity']/position['entry_price']*100:.1f}%) - Reason: {reason}")
        
        return position
    
    def _update_performance_metrics(self):
        """Performance metriklerini güncelle"""
        if self.performance['total_trades'] == 0:
            return
        
        # Win rate
        self.performance['win_rate'] = self.performance['winning_trades'] / self.performance['total_trades']
        
        # Average win/loss
        winning_pnls = [p['pnl'] for p in self.closed_positions if p['pnl'] > 0]
        losing_pnls = [p['pnl'] for p in self.closed_positions if p['pnl'] <= 0]
        
        if winning_pnls:
            self.performance['avg_win'] = np.mean(winning_pnls)
        
        if losing_pnls:
            self.performance['avg_loss'] = abs(np.mean(losing_pnls))
        
        # Profit factor
        if self.performance['avg_loss'] > 0:
            self.performance['profit_factor'] = self.performance['avg_win'] / self.performance['avg_loss']
        
        # Drawdown
        peak = max(self.equity_curve) if self.equity_curve else self.initial_capital
        current_dd = (peak - self.current_capital) / peak
        self.performance['current_drawdown'] = current_dd
        self.performance['max_drawdown'] = max(self.performance['max_drawdown'], current_dd)
        
        # Sharpe ratio (simplified)
        if len(self.equity_curve) > 2:
            returns = pd.Series(self.equity_curve).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                self.performance['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
    
    def _save_trade_history(self, position: Dict[str, Any]):
        """Trade geçmişini kaydet"""
        try:
            trade_data = {
                'symbol': position['symbol'],
                'side': position['side'],
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'exit_price': position['close_price'],
                'pnl': position['pnl'],
                'pnl_percent': position['pnl'] / (position['quantity'] * position['entry_price']) * 100,
                'opened_at': position['opened_at'],
                'closed_at': position['closed_at'],
                'duration': (position['closed_at'] - position['opened_at']).total_seconds() / 3600,
                'close_reason': position['close_reason']
            }
            
            # CSV'ye append
            df = pd.DataFrame([trade_data])
            
            if self.trades_file.exists():
                df.to_csv(self.trades_file, mode='a', header=False, index=False)
            else:
                self.trades_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(self.trades_file, index=False)
                
        except Exception as e:
            logger.error(f"Trade history kaydetme hatası: {e}")
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Portfolio durumu özeti"""
        # Açık pozisyonların PnL'i
        open_pnl = sum(pos['pnl'] for pos in self.positions.values())
        
        # Toplam equity
        total_equity = self.current_capital + sum(
            pos['quantity'] * pos['current_price'] for pos in self.positions.values()
        )
        
        status = {
            'capital': self.current_capital,
            'total_equity': total_equity,
            'open_positions': len(self.positions),
            'open_pnl': open_pnl,
            'total_pnl': self.performance['total_pnl'],
            'total_return': (total_equity - self.initial_capital) / self.initial_capital * 100,
            'win_rate': self.performance['win_rate'] * 100,
            'profit_factor': self.performance['profit_factor'],
            'max_drawdown': self.performance['max_drawdown'] * 100,
            'current_drawdown': self.performance['current_drawdown'] * 100,
            'sharpe_ratio': self.performance['sharpe_ratio'],
            'positions': self.positions
        }
        
        return status
    
    def risk_check(self) -> Dict[str, Any]:
        """Risk durumu kontrolü"""
        portfolio_risk = self._calculate_portfolio_risk()
        
        risk_status = {
            'portfolio_risk': portfolio_risk * 100,
            'risk_limit': self.max_portfolio_risk * 100,
            'within_limit': portfolio_risk <= self.max_portfolio_risk,
            'current_drawdown': self.performance['current_drawdown'] * 100,
            'drawdown_limit': 8.0,  # %8 monthly limit
            'can_open_positions': (
                len(self.positions) < self.max_positions and
                portfolio_risk < self.max_portfolio_risk and
                self.performance['current_drawdown'] < 0.08
            )
        }
        
        return risk_status