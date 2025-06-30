"""
Signal Generator Module
ML/DL modellerden sinyal üretimi
Multi-timeframe analiz ve ensemble predictions
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from loguru import logger
import asyncio
from pathlib import Path
import sys

# Parent path ekle
sys.path.append(str(Path(__file__).parent.parent))

from models.simple_gru_model import SimpleMultiTimeframeGRU
from core.feature_engineering import FeatureEngineering
from indicators.indicator_calculator import IndicatorCalculator


class SignalGenerator:
    """Trading sinyalleri üretir"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Feature engineering
        self.feature_engineer = FeatureEngineering(config)
        
        # Indicator calculator
        self.indicator_calc = IndicatorCalculator()
        
        # GRU model
        self.model = None
        self.model_path = Path("models/saved/gru_multi_timeframe.pth")
        self._load_model()
        
        # Signal thresholds
        self.confidence_threshold = config['signals']['confidence_threshold']
        self.min_profit_target = config['signals']['min_profit_target']
        
        # Signal cache
        self.signal_cache = {}
        self.cache_ttl = 300  # 5 dakika
        
        logger.info("SignalGenerator başlatıldı")
    
    def _load_model(self):
        """Model yükle veya yeni oluştur"""
        try:
            # Model config
            model_config = {
                'input_size': 50,
                'hidden_size': 50,
                'num_layers': 1
            }
            
            if self.model_path.exists():
                # Mevcut modeli yükle
                checkpoint = torch.load(self.model_path, map_location='cpu')
                
                # Model parametrelerini güncelle
                if 'config' in checkpoint:
                    model_config.update(checkpoint['config'])
                
                # Model oluştur
                self.model = SimpleMultiTimeframeGRU(
                    input_size=model_config['input_size'],
                    hidden_size=model_config['hidden_size'],
                    num_layers=model_config['num_layers']
                )
                
                # Ağırlıkları yükle
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                logger.info("Mevcut model yüklendi")
            else:
                # Yeni model oluştur
                self.model = SimpleMultiTimeframeGRU(
                    input_size=model_config['input_size'],
                    hidden_size=model_config['hidden_size'],
                    num_layers=model_config['num_layers']
                )
                self.model.eval()
                logger.warning("Yeni model oluşturuldu - Eğitim gerekli!")
                
        except Exception as e:
            logger.error(f"Model yükleme hatası: {e}")
            # Default model
            self.model = SimpleMultiTimeframeGRU(
                input_size=50,
                hidden_size=50,
                num_layers=1
            )
            self.model.eval()
    
    async def generate_signals(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Verilen semboller için sinyal üret"""
        signals = []
        
        # Paralel işlem için task'ler
        tasks = []
        for symbol in symbols:
            task = self._generate_signal_for_symbol(symbol)
            tasks.append(task)
        
        # Tüm sinyalleri topla
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"{symbols[i]} sinyal hatası: {result}")
            elif result is not None:
                signals.append(result)
        
        # Sinyalleri filtrele ve sırala
        filtered_signals = self._filter_and_rank_signals(signals)
        
        logger.info(f"{len(symbols)} sembol için {len(filtered_signals)} sinyal üretildi")
        
        return filtered_signals
    
    async def _generate_signal_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Tek bir sembol için sinyal üret"""
        # Cache kontrolü
        cache_key = f"signal_{symbol}"
        if cache_key in self.signal_cache:
            cached_time, cached_signal = self.signal_cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_ttl:
                return cached_signal
        
        try:
            # Multi-timeframe veri topla
            from core.data_collector import UnifiedDataCollector
            collector = UnifiedDataCollector(self.config)
            data = await collector.collect_multi_timeframe_data(symbol)
            
            if not data or 'indicators' not in data:
                logger.debug(f"{symbol} için veri bulunamadı")
                return None
            
            # Feature'ları oluştur
            features = self.feature_engineer.create_features(data, symbol)
            
            if not features:
                logger.debug(f"{symbol} için feature oluşturulamadı")
                return None
            
            # Model tahmini
            prediction = self._predict_with_model(features)
            
            # MACD bazlı sinyal kontrolü (raporda en güvenli)
            macd_signal = self._check_macd_signal(data)
            
            # Sinyal kombinasyonu
            signal = self._combine_signals(
                symbol, prediction, macd_signal, features, data
            )
            
            # Cache'e kaydet
            if signal:
                self.signal_cache[cache_key] = (datetime.now(), signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"{symbol} sinyal üretim hatası: {e}")
            return None
    
    def _predict_with_model(self, features: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Model ile tahmin yap"""
        predictions = {}
        
        try:
            # Her timeframe için model girişi hazırla
            model_inputs = {}
            for tf in self.config['timeframes']['analysis']:
                if tf in features:
                    # Son 30 bar'ı al ve normalize et
                    input_data = self.feature_engineer.prepare_model_input(
                        features, tf, sequence_length=30
                    )
                    model_inputs[tf] = torch.FloatTensor(input_data).unsqueeze(0)
            
            # Model inference
            with torch.no_grad():
                if all(tf in model_inputs for tf in ['15m', '1h', '4h', '1d', '1w']):
                    output, attention_weights = self.model(
                        model_inputs['15m'],
                        model_inputs['1h'],
                        model_inputs['4h'],
                        model_inputs['1d'],
                        model_inputs['1w']
                    )
                    
                    # Sigmoid ile 0-1 arası değere çevir
                    probability = torch.sigmoid(output).item()
                    
                    predictions['probability'] = probability
                    predictions['direction'] = 'buy' if probability > 0.5 else 'sell'
                    predictions['confidence'] = abs(probability - 0.5) * 2  # 0-1 arası
                    
                    # Attention weights
                    predictions['attention'] = {
                        '15m': attention_weights[0, 0].item(),
                        '1h': attention_weights[0, 1].item(),
                        '4h': attention_weights[0, 2].item(),
                        '1d': attention_weights[0, 3].item(),
                        '1w': attention_weights[0, 4].item()
                    }
            
        except Exception as e:
            logger.error(f"Model tahmin hatası: {e}")
            predictions = {
                'probability': 0.5,
                'direction': 'neutral',
                'confidence': 0.0
            }
        
        return predictions
    
    def _check_macd_signal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """MACD sinyali kontrol et"""
        macd_signals = {}
        
        for tf in ['1h', '4h', '1d']:
            if tf in data.get('indicators', {}):
                df = data['indicators'][tf]
                
                if 'macd' in df.columns and 'macd_signal' in df.columns:
                    # Son değerler
                    macd_current = df['macd'].iloc[-1]
                    signal_current = df['macd_signal'].iloc[-1]
                    macd_prev = df['macd'].iloc[-2]
                    signal_prev = df['macd_signal'].iloc[-2]
                    
                    # Kesişim kontrolü
                    bullish_cross = (macd_prev <= signal_prev) and (macd_current > signal_current)
                    bearish_cross = (macd_prev >= signal_prev) and (macd_current < signal_current)
                    
                    # Histogram momentum
                    if 'macd_hist' in df.columns:
                        hist_momentum = df['macd_hist'].iloc[-1] - df['macd_hist'].iloc[-5]
                    else:
                        hist_momentum = 0
                    
                    macd_signals[tf] = {
                        'bullish_cross': bullish_cross,
                        'bearish_cross': bearish_cross,
                        'histogram_momentum': hist_momentum,
                        'position': 'above' if macd_current > signal_current else 'below'
                    }
        
        # Genel MACD sinyali
        bullish_count = sum(1 for tf in macd_signals.values() if tf['bullish_cross'])
        bearish_count = sum(1 for tf in macd_signals.values() if tf['bearish_cross'])
        
        overall_signal = {
            'direction': 'buy' if bullish_count > bearish_count else 'sell' if bearish_count > 0 else 'neutral',
            'strength': max(bullish_count, bearish_count) / len(macd_signals) if macd_signals else 0,
            'details': macd_signals
        }
        
        return overall_signal
    
    def _combine_signals(self, symbol: str, ml_prediction: Dict, macd_signal: Dict,
                        features: Dict, data: Dict) -> Optional[Dict[str, Any]]:
        """Tüm sinyalleri birleştir"""
        try:
            # Temel kontroller
            if ml_prediction['confidence'] < self.confidence_threshold:
                return None
            
            # MACD ve ML uyumu
            if ml_prediction['direction'] != macd_signal['direction']:
                # Uyumsuzluk varsa, MACD'ye güven (raporda daha güvenli bulunmuş)
                if macd_signal['strength'] < 0.5:
                    return None
            
            # Son fiyat bilgisi
            current_data = data.get('indicators', {}).get('1h', pd.DataFrame())
            if current_data.empty:
                return None
            
            current_price = current_data['close'].iloc[-1]
            
            # ATR bazlı stop loss (raporda önerilen 2x ATR)
            atr = current_data.get('atr', pd.Series()).iloc[-1] if 'atr' in current_data.columns else current_price * 0.02
            stop_loss = current_price - (2 * atr) if ml_prediction['direction'] == 'buy' else current_price + (2 * atr)
            
            # Risk/Reward hesapla
            risk = abs(current_price - stop_loss)
            target_1 = current_price + (risk * 2)  # 2:1 R/R
            target_2 = current_price + (risk * 3)  # 3:1 R/R
            
            # Volatilite kontrolü
            volatility_ok = self._check_volatility(current_data)
            if not volatility_ok:
                return None
            
            # Volume kontrolü
            volume_ok = self._check_volume(current_data)
            if not volume_ok:
                logger.debug(f"{symbol} volume yetersiz")
                return None
            
            # Final sinyal
            signal = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'direction': ml_prediction['direction'],
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'target_1': target_1,
                'target_2': target_2,
                'confidence': ml_prediction['confidence'],
                'ml_probability': ml_prediction['probability'],
                'macd_strength': macd_signal['strength'],
                'attention_weights': ml_prediction.get('attention', {}),
                'risk_reward': 2.0,  # Minimum R/R
                'position_size_pct': self._calculate_position_size(ml_prediction['confidence']),
                'timeframe_signals': {
                    '1h': macd_signal['details'].get('1h', {}),
                    '4h': macd_signal['details'].get('4h', {}),
                    '1d': macd_signal['details'].get('1d', {})
                }
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Sinyal birleştirme hatası: {e}")
            return None
    
    def _check_volatility(self, df: pd.DataFrame) -> bool:
        """Volatilite kontrolü"""
        if 'atr_percent' in df.columns:
            # Son 20 bar ortalama volatilite
            avg_volatility = df['atr_percent'].tail(20).mean()
            current_volatility = df['atr_percent'].iloc[-1]
            
            # Çok yüksek volatilite varsa pas geç
            if current_volatility > avg_volatility * 2:
                return False
            
            # Minimum volatilite olmalı
            if current_volatility < 0.5:  # %0.5'ten az volatilite
                return False
        
        return True
    
    def _check_volume(self, df: pd.DataFrame) -> bool:
        """Volume kontrolü"""
        if 'volume_ratio' in df.columns:
            # Son bar volume ratio
            volume_ratio = df['volume_ratio'].iloc[-1]
            
            # Minimum 0.8x ortalama volume
            if volume_ratio < 0.8:
                return False
        
        return True
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Confidence'a göre pozisyon büyüklüğü"""
        # Base position size %1 (raporda önerilen)
        base_size = 0.01
        
        # Confidence'a göre ayarla (0.5-1.5x)
        multiplier = 0.5 + confidence  # confidence 0-1 arası
        
        return min(base_size * multiplier, 0.02)  # Max %2
    
    def _filter_and_rank_signals(self, signals: List[Dict]) -> List[Dict]:
        """Sinyalleri filtrele ve sırala"""
        # Boş sinyalleri temizle
        valid_signals = [s for s in signals if s is not None]
        
        # Confidence'a göre sırala
        sorted_signals = sorted(
            valid_signals, 
            key=lambda x: x['confidence'] * x['macd_strength'], 
            reverse=True
        )
        
        # Max sinyal sayısı
        max_signals = self.config.get('max_concurrent_signals', 5)
        
        return sorted_signals[:max_signals]
    
    def get_signal_summary(self, signal: Dict) -> str:
        """Sinyal özeti oluştur"""
        summary = f"""
        Symbol: {signal['symbol']}
        Direction: {signal['direction'].upper()}
        Entry: {signal['entry_price']:.2f}
        Stop Loss: {signal['stop_loss']:.2f} ({abs(signal['entry_price']-signal['stop_loss'])/signal['entry_price']*100:.1f}%)
        Target 1: {signal['target_1']:.2f} ({abs(signal['target_1']-signal['entry_price'])/signal['entry_price']*100:.1f}%)
        Target 2: {signal['target_2']:.2f} ({abs(signal['target_2']-signal['entry_price'])/signal['entry_price']*100:.1f}%)
        Confidence: {signal['confidence']:.1%}
        Position Size: {signal['position_size_pct']:.1%}
        
        Timeframe Weights:
        """
        
        for tf, weight in signal['attention_weights'].items():
            summary += f"  {tf}: {weight:.1%}\n"
        
        return summary.strip()
    
    async def update_existing_signals(self, open_positions: List[Dict]) -> List[Dict]:
        """Mevcut pozisyonlar için sinyal güncellemesi"""
        updates = []
        
        for position in open_positions:
            symbol = position['symbol']
            
            # Güncel veri al
            from core.data_collector import UnifiedDataCollector
            collector = UnifiedDataCollector(self.config)
            data = await collector.collect_multi_timeframe_data(symbol)
            
            if not data:
                continue
            
            # Trailing stop hesapla
            current_price = data.get('indicators', {}).get('1h', pd.DataFrame()).get('close', pd.Series()).iloc[-1]
            
            if position['direction'] == 'buy':
                # Long pozisyon için trailing stop
                new_stop = self._calculate_trailing_stop_long(
                    position['entry_price'],
                    current_price,
                    position['stop_loss'],
                    data
                )
                
                # Take profit kontrolü
                if current_price >= position['target_1'] and not position.get('target_1_hit'):
                    updates.append({
                        'symbol': symbol,
                        'action': 'partial_close',
                        'percentage': 0.5,  # %50 kapat
                        'reason': 'target_1_reached'
                    })
            else:
                # Short pozisyon için trailing stop
                new_stop = self._calculate_trailing_stop_short(
                    position['entry_price'],
                    current_price,
                    position['stop_loss'],
                    data
                )
            
            # Stop update gerekli mi?
            if new_stop != position['stop_loss']:
                updates.append({
                    'symbol': symbol,
                    'action': 'update_stop',
                    'new_stop': new_stop,
                    'reason': 'trailing_stop'
                })
        
        return updates
    
    def _calculate_trailing_stop_long(self, entry: float, current: float, 
                                     current_stop: float, data: Dict) -> float:
        """Long pozisyon için trailing stop"""
        # Profit'te mi?
        if current <= entry:
            return current_stop
        
        # ATR bazlı trailing
        atr = data.get('indicators', {}).get('1h', pd.DataFrame()).get('atr', pd.Series())
        if not atr.empty:
            atr_value = atr.iloc[-1]
            new_stop = current - (2 * atr_value)
            
            # Sadece yukarı hareket ettir
            return max(new_stop, current_stop)
        
        return current_stop
    
    def _calculate_trailing_stop_short(self, entry: float, current: float,
                                      current_stop: float, data: Dict) -> float:
        """Short pozisyon için trailing stop"""
        # Profit'te mi?
        if current >= entry:
            return current_stop
        
        # ATR bazlı trailing
        atr = data.get('indicators', {}).get('1h', pd.DataFrame()).get('atr', pd.Series())
        if not atr.empty:
            atr_value = atr.iloc[-1]
            new_stop = current + (2 * atr_value)
            
            # Sadece aşağı hareket ettir
            return min(new_stop, current_stop)
        
        return current_stop