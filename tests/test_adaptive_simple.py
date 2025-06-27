#!/usr/bin/env python3
"""
Basit Adaptif Sistem Testi
ML olmadan temel bileşenleri test eder
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Proje imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from strategies.adaptive_ensemble_system import AdaptiveEnsembleSystem
from strategies.market_regime_detector import MarketRegimeDetector
from risk.dynamic_risk_manager import DynamicRiskManager


def main():
    print("\n" + "="*80)
    print("BASİT ADAPTİF SİSTEM TESTİ")
    print("="*80)
    
    # Test sembolü
    symbol = 'THYAO'
    timeframe = '1h'
    
    # 1. Market Regime Detector
    print("\n1. Market Regime Detector:")
    print("-"*40)
    
    try:
        detector = MarketRegimeDetector()
        regime, analysis = detector.detect_regime(symbol, timeframe)
        
        print(f"Sembol: {symbol}")
        print(f"Rejim: {regime}")
        print(f"Güven: {analysis['confidence']:.3f}")
        print(f"Açıklama: {detector.regimes[regime]['description']}")
        
        # Öneriler
        recommendations = detector.get_regime_recommendation(regime)
        print(f"\nÖnerilen Strateji: {recommendations.get('strategy', 'N/A')}")
        
    except Exception as e:
        print(f"Market Regime Hatası: {e}")
    
    # 2. Adaptive Ensemble System
    print("\n\n2. Adaptive Ensemble System:")
    print("-"*40)
    
    try:
        ensemble = AdaptiveEnsembleSystem()
        signal = ensemble.generate_ensemble_signal(symbol, timeframe)
        
        print(f"Sembol: {symbol}")
        print(f"Sinyal: {signal['signal']}")
        print(f"Güven: {signal['confidence']:.3f}")
        print(f"Buy Score: {signal['buy_score']:.3f}")
        print(f"Sell Score: {signal['sell_score']:.3f}")
        
        # Kategori detayları
        print("\nKategori Sinyalleri:")
        for cat, data in signal['category_signals'].items():
            print(f"  {cat}: {data['signal']} (güç: {data['strength']:.3f})")
            
    except Exception as e:
        print(f"Ensemble Hatası: {e}")
    
    # 3. Risk Manager
    print("\n\n3. Dynamic Risk Manager:")
    print("-"*40)
    
    try:
        risk_manager = DynamicRiskManager(initial_capital=50000)
        
        # Test sinyali
        test_signal = {
            'signal': 'BUY',
            'confidence': 0.75,
            'category_signals': {
                'momentum': {'signal': 'BUY'},
                'trend': {'signal': 'BUY'},
                'ml_based': {'signal': 'BUY'},
                'volatility': {'signal': 'HOLD'}
            }
        }
        
        # Pozisyon hesapla
        position = risk_manager.calculate_position_size(
            signal_data=test_signal,
            symbol=symbol,
            price=245.50,
            timeframe=timeframe
        )
        
        print(f"Sembol: {symbol}")
        print(f"Giriş Fiyatı: 245.50 TL")
        print(f"\nPozisyon Detayları:")
        print(f"  Büyüklük: %{position['position_size_pct']*100:.2f}")
        print(f"  Değer: {position['position_value']:,.2f} TL")
        print(f"  Hisse: {position['shares']} adet")
        print(f"  Stop Loss: {position['stop_loss']:.2f} TL")
        print(f"  Hedef: {position['take_profit']:.2f} TL")
        print(f"  Risk/Reward: 1:{position['risk_reward_ratio']:.2f}")
        
        # Risk limitleri
        limits = risk_manager.check_risk_limits()
        print("\nRisk Limitleri:")
        for check, ok in limits.items():
            status = "✓" if ok else "✗"
            print(f"  {status} {check}")
            
    except Exception as e:
        print(f"Risk Manager Hatası: {e}")
    
    # 4. Entegre Sinyal
    print("\n\n4. Entegre Sinyal Örneği:")
    print("-"*40)
    
    try:
        # Rejim ve ensemble sinyallerini birleştir
        if 'regime' in locals() and 'signal' in locals():
            print(f"Rejim: {regime}")
            print(f"Ensemble Sinyal: {signal['signal']}")
            print(f"Birleşik Güven: {(analysis['confidence'] + signal['confidence']) / 2:.3f}")
            
            # Rejime göre strateji önerisi
            if regime == 'strong_trend_up' and signal['signal'] == 'BUY':
                print("\n✓ Güçlü AL sinyali! Trend ve sinyal uyumlu.")
            elif regime == 'volatile' and signal['signal'] in ['BUY', 'SELL']:
                print("\n⚠️ Dikkat! Volatil piyasada küçük pozisyon önerilir.")
            elif regime == 'ranging':
                print("\n📊 Yatay piyasa. Destek/direnç seviyeleri önemli.")
                
    except Exception as e:
        print(f"Entegre Sinyal Hatası: {e}")
    
    print("\n" + "="*80)
    print("Test tamamlandı!")


if __name__ == "__main__":
    main()