#!/usr/bin/env python3
"""
Entegre Adaptif Trading Sistemi
Tüm bileşenleri birleştiren ana sistem
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
import asyncio
from pathlib import Path
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Proje imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from strategies.adaptive_ensemble_system import AdaptiveEnsembleSystem
from strategies.market_regime_detector import MarketRegimeDetector
from risk.dynamic_risk_manager import DynamicRiskManager
from ml_models.ensemble_ml_predictor import EnsembleMLPredictor
from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS


class IntegratedTradingSystem:
    """Tüm bileşenleri entegre eden ana trading sistemi"""
    
    def __init__(self, initial_capital: float = 50000, mode: str = 'paper'):
        self.initial_capital = initial_capital
        self.mode = mode  # 'paper' veya 'live'
        
        # Alt sistemler
        self.ensemble_system = AdaptiveEnsembleSystem(initial_capital)
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = DynamicRiskManager(initial_capital)
        self.ml_predictor = None  # Lazy load
        
        # CSV manager
        self.csv_manager = CSVDataManager()
        
        # Sistem durumu
        self.active_positions = {}
        self.pending_orders = {}
        self.system_state = 'running'
        
        # Performans takibi
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'current_drawdown': 0.0
        }
        
        # Adaptasyon parametreleri
        self.adaptation_config = {
            'weight_update_interval': 7,  # gün
            'ml_retrain_interval': 30,    # gün
            'regime_check_interval': 1,    # saat
            'risk_check_interval': 15      # dakika
        }
        
        # Log ayarları
        self._setup_logging()
        
    def _setup_logging(self):
        """Loglama ayarları"""
        log_dir = Path("logs/integrated_system")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log"
        logger.add(log_file, rotation="1 day", retention="30 days", level="INFO")
    
    def initialize_ml_models(self, train: bool = False):
        """ML modellerini başlat"""
        self.ml_predictor = EnsembleMLPredictor()
        
        model_dir = 'ml_models/trained/ensemble'
        
        if train or not Path(model_dir).exists():
            logger.info("ML modelleri eğitiliyor...")
            # En likit semboller ile eğit
            training_symbols = ['THYAO', 'GARAN', 'AKBNK', 'EREGL', 'ASELS']
            self.ml_predictor.train_models(training_symbols)
            self.ml_predictor.save_models()
        else:
            logger.info("ML modelleri yükleniyor...")
            self.ml_predictor.load_models()
    
    def generate_trading_decision(self, symbol: str, timeframe: str = '1h') -> Dict:
        """Entegre trading kararı üret"""
        try:
            # 1. Piyasa rejimini tespit et
            regime, regime_analysis = self.regime_detector.detect_regime(symbol, timeframe)
            logger.info(f"{symbol} - Rejim: {regime} (güven: {regime_analysis['confidence']:.3f})")
            
            # 2. Ensemble sinyal al
            ensemble_signal = self.ensemble_system.generate_ensemble_signal(symbol, timeframe)
            logger.info(f"{symbol} - Ensemble: {ensemble_signal['signal']} (güven: {ensemble_signal['confidence']:.3f})")
            
            # 3. ML tahmini (opsiyonel)
            ml_signal = {'signal': 'HOLD', 'confidence': 0.0}
            if self.ml_predictor and self.ml_predictor.is_fitted:
                try:
                    ml_signal = self.ml_predictor.predict(symbol, timeframe)
                    logger.info(f"{symbol} - ML: {ml_signal['signal']} (güven: {ml_signal['confidence']:.3f})")
                except Exception as e:
                    logger.warning(f"ML tahmin hatası: {e}")
            
            # 4. Sinyalleri birleştir
            combined_signal = self._combine_signals(
                ensemble_signal, 
                ml_signal, 
                regime
            )
            
            # 5. Risk kontrolü
            if combined_signal['signal'] != 'HOLD':
                # Güncel fiyat al
                df = self.csv_manager.load_raw_data(symbol, timeframe)
                if df is not None and len(df) > 0:
                    current_price = df['close'].iloc[-1]
                    
                    # Risk limitleri kontrol et
                    risk_checks = self.risk_manager.check_risk_limits()
                    
                    if not all(risk_checks.values()):
                        logger.warning(f"Risk limitleri aşıldı: {risk_checks}")
                        combined_signal['signal'] = 'HOLD'
                        combined_signal['risk_blocked'] = True
                    else:
                        # Pozisyon boyutu hesapla
                        position_data = self.risk_manager.calculate_position_size(
                            combined_signal,
                            symbol,
                            current_price,
                            timeframe
                        )
                        combined_signal['position_data'] = position_data
            
            # 6. Final karar
            decision = {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': combined_signal['signal'],
                'confidence': combined_signal['confidence'],
                'regime': regime,
                'regime_confidence': regime_analysis['confidence'],
                'ensemble_details': ensemble_signal,
                'ml_details': ml_signal,
                'position_data': combined_signal.get('position_data', {}),
                'risk_blocked': combined_signal.get('risk_blocked', False),
                'timestamp': datetime.now().isoformat()
            }
            
            return decision
            
        except Exception as e:
            logger.error(f"{symbol} karar üretme hatası: {e}")
            return {
                'symbol': symbol,
                'signal': 'HOLD',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _combine_signals(self, ensemble_signal: Dict, ml_signal: Dict, 
                        regime: str) -> Dict:
        """Sinyalleri birleştir"""
        # Rejime göre ağırlıklar
        regime_weights = {
            'strong_trend_up': {'ensemble': 0.7, 'ml': 0.3},
            'strong_trend_down': {'ensemble': 0.8, 'ml': 0.2},
            'weak_trend': {'ensemble': 0.6, 'ml': 0.4},
            'ranging': {'ensemble': 0.5, 'ml': 0.5},
            'volatile': {'ensemble': 0.8, 'ml': 0.2},
            'squeeze': {'ensemble': 0.6, 'ml': 0.4}
        }
        
        weights = regime_weights.get(regime, {'ensemble': 0.7, 'ml': 0.3})
        
        # Sinyal skorları
        signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        # Ensemble skoru
        if ensemble_signal['signal'] == 'BUY':
            signal_scores['BUY'] += ensemble_signal['confidence'] * weights['ensemble']
        elif ensemble_signal['signal'] == 'SELL':
            signal_scores['SELL'] += ensemble_signal['confidence'] * weights['ensemble']
        else:
            signal_scores['HOLD'] += 0.5 * weights['ensemble']
        
        # ML skoru
        if ml_signal['signal'] == 'BUY':
            signal_scores['BUY'] += ml_signal['confidence'] * weights['ml']
        elif ml_signal['signal'] == 'SELL':
            signal_scores['SELL'] += ml_signal['confidence'] * weights['ml']
        else:
            signal_scores['HOLD'] += 0.5 * weights['ml']
        
        # En yüksek skorlu sinyali seç
        max_signal = max(signal_scores.items(), key=lambda x: x[1])
        
        # Minimum güven seviyesi
        min_confidence = 0.6
        
        if max_signal[1] < min_confidence:
            return {'signal': 'HOLD', 'confidence': max_signal[1]}
        
        return {
            'signal': max_signal[0],
            'confidence': max_signal[1],
            'scores': signal_scores
        }
    
    def execute_decision(self, decision: Dict):
        """Trading kararını uygula"""
        if self.mode != 'live':
            logger.info(f"[PAPER] Karar: {decision['symbol']} - {decision['signal']}")
            return
        
        symbol = decision['symbol']
        signal = decision['signal']
        
        # Mevcut pozisyon kontrolü
        if symbol in self.active_positions:
            if signal == 'SELL' and self.active_positions[symbol]['side'] == 'long':
                self._close_position(symbol, 'signal_reversal')
            elif signal == 'BUY' and self.active_positions[symbol]['side'] == 'short':
                self._close_position(symbol, 'signal_reversal')
        else:
            # Yeni pozisyon
            if signal in ['BUY', 'SELL'] and 'position_data' in decision:
                self._open_position(symbol, signal, decision['position_data'])
    
    def _open_position(self, symbol: str, signal: str, position_data: Dict):
        """Pozisyon aç"""
        # Bu fonksiyon gerçek uygulamada broker API'si ile emir gönderecek
        logger.info(f"Pozisyon açılıyor: {symbol} - {signal}")
        logger.info(f"Detaylar: {position_data}")
        
        self.active_positions[symbol] = {
            'side': 'long' if signal == 'BUY' else 'short',
            'entry_price': position_data.get('entry_price', 0),
            'shares': position_data.get('shares', 0),
            'stop_loss': position_data.get('stop_loss', 0),
            'take_profit': position_data.get('take_profit', 0),
            'entry_time': datetime.now()
        }
        
        # Risk manager'a bildir
        self.risk_manager.update_position(symbol, self.active_positions[symbol])
    
    def _close_position(self, symbol: str, reason: str):
        """Pozisyon kapat"""
        if symbol not in self.active_positions:
            return
        
        logger.info(f"Pozisyon kapatılıyor: {symbol} - Sebep: {reason}")
        
        # Risk manager'a bildir
        # Gerçek uygulamada güncel fiyat alınacak
        exit_price = 0  
        self.risk_manager.close_position(symbol, exit_price, reason)
        
        del self.active_positions[symbol]
    
    def scan_all_symbols(self, symbols: Optional[List[str]] = None, 
                        timeframe: str = '1h') -> List[Dict]:
        """Tüm sembolleri tara"""
        if symbols is None:
            symbols = ASSETS[:20]  # İlk 20 sembol
        
        decisions = []
        
        for symbol in symbols:
            try:
                decision = self.generate_trading_decision(symbol, timeframe)
                decisions.append(decision)
                
                # Rate limiting
                import time
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"{symbol} tarama hatası: {e}")
        
        # Güven skoruna göre sırala
        decisions.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return decisions
    
    def update_system_weights(self):
        """Sistem ağırlıklarını güncelle"""
        logger.info("Sistem ağırlıkları güncelleniyor...")
        
        # Ensemble sistem ağırlıklarını güncelle
        self.ensemble_system.update_weights()
        
        # ML modellerini yeniden eğit (gerekirse)
        # Bu genelde daha uzun aralıklarla yapılır
        
        logger.info("Ağırlık güncellemesi tamamlandı")
    
    def get_system_status(self) -> Dict:
        """Sistem durumunu al"""
        return {
            'state': self.system_state,
            'active_positions': len(self.active_positions),
            'pending_orders': len(self.pending_orders),
            'current_capital': self.risk_manager.current_capital,
            'performance': self.performance_metrics,
            'risk_metrics': self.risk_manager.get_risk_metrics(),
            'ensemble_stats': self.ensemble_system.get_system_stats(),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_state(self, filepath: str = 'data/system_state.json'):
        """Sistem durumunu kaydet"""
        state = {
            'active_positions': self.active_positions,
            'performance_metrics': self.performance_metrics,
            'risk_manager_state': self.risk_manager.get_risk_metrics(),
            'ensemble_state': self.ensemble_system.get_system_stats(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    async def run_continuous(self, symbols: List[str], timeframe: str = '1h'):
        """Sürekli çalışma modu"""
        logger.info("Sürekli trading modu başlatılıyor...")
        
        while self.system_state == 'running':
            try:
                # Sembolleri tara
                decisions = self.scan_all_symbols(symbols, timeframe)
                
                # En iyi sinyalleri uygula
                for decision in decisions[:5]:  # En iyi 5 sinyal
                    if decision['signal'] != 'HOLD' and decision['confidence'] > 0.7:
                        self.execute_decision(decision)
                
                # Mevcut pozisyonları kontrol et
                self._check_existing_positions()
                
                # Sistem durumunu logla
                status = self.get_system_status()
                logger.info(f"Sistem durumu: {status['active_positions']} açık pozisyon, "
                           f"Sermaye: {status['current_capital']:,.2f} TL")
                
                # Periyodik güncellemeler
                await self._periodic_updates()
                
                # Bekleme
                await asyncio.sleep(60)  # 1 dakika
                
            except Exception as e:
                logger.error(f"Ana döngü hatası: {e}")
                await asyncio.sleep(10)
    
    def _check_existing_positions(self):
        """Mevcut pozisyonları kontrol et"""
        for symbol, position in list(self.active_positions.items()):
            # Stop loss / take profit kontrolü
            # Gerçek uygulamada güncel fiyat alınacak
            pass
    
    async def _periodic_updates(self):
        """Periyodik güncellemeler"""
        now = datetime.now()
        
        # Ağırlık güncellemesi
        if hasattr(self, '_last_weight_update'):
            if (now - self._last_weight_update).days >= self.adaptation_config['weight_update_interval']:
                self.update_system_weights()
                self._last_weight_update = now
        else:
            self._last_weight_update = now
    
    def shutdown(self):
        """Sistemi kapat"""
        logger.info("Sistem kapatılıyor...")
        
        # Tüm pozisyonları kapat
        for symbol in list(self.active_positions.keys()):
            self._close_position(symbol, 'system_shutdown')
        
        # Durumu kaydet
        self.save_state()
        
        self.system_state = 'stopped'
        logger.info("Sistem kapatıldı")


def main():
    """Test fonksiyonu"""
    # Sistemi başlat
    system = IntegratedTradingSystem(initial_capital=50000, mode='paper')
    
    # ML modellerini başlat (eğitim olmadan)
    # system.initialize_ml_models(train=False)
    
    # Test sembolleri
    test_symbols = ['THYAO', 'GARAN', 'AKBNK']
    
    # Tek seferlik tarama
    print("\nSembol Taraması:")
    print("="*80)
    
    decisions = system.scan_all_symbols(test_symbols)
    
    for decision in decisions:
        if 'error' not in decision:
            print(f"\n{decision['symbol']}:")
            print(f"  Sinyal: {decision['signal']}")
            print(f"  Güven: {decision['confidence']:.3f}")
            print(f"  Rejim: {decision['regime']} ({decision['regime_confidence']:.3f})")
            
            if decision['signal'] != 'HOLD' and 'position_data' in decision:
                pos = decision['position_data']
                print(f"  Pozisyon: %{pos['position_size_pct']*100:.1f} = {pos['position_value']:,.0f} TL")
                print(f"  Stop Loss: {pos['stop_loss']:.2f}")
                print(f"  Hedef: {pos['take_profit']:.2f}")
    
    # Sistem durumu
    print("\n\nSistem Durumu:")
    print("="*80)
    status = system.get_system_status()
    print(json.dumps(status, indent=2, default=str))
    
    # Asyncio ile sürekli çalıştırma örneği
    # asyncio.run(system.run_continuous(test_symbols))


if __name__ == "__main__":
    main()