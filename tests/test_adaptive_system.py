#!/usr/bin/env python3
"""
Adaptif Ensemble Trading Sistemi Test Script
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from loguru import logger

# Proje imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from strategies.integrated_trading_system import IntegratedTradingSystem
from strategies.adaptive_ensemble_system import AdaptiveEnsembleSystem
from strategies.market_regime_detector import MarketRegimeDetector
from risk.dynamic_risk_manager import DynamicRiskManager


def test_individual_components():
    """Bileşenleri tek tek test et"""
    print("\n" + "="*80)
    print("BİLEŞEN TESTLERİ")
    print("="*80)
    
    # 1. Market Regime Detector
    print("\n1. Market Regime Detector Testi:")
    print("-"*40)
    
    detector = MarketRegimeDetector()
    symbol = 'THYAO'
    
    try:
        regime, analysis = detector.detect_regime(symbol, '1h')
        print(f"Sembol: {symbol}")
        print(f"Rejim: {regime}")
        print(f"Güven: {analysis['confidence']:.3f}")
        print("\nRejim Skorları:")
        for r, score in analysis['scores'].items():
            print(f"  {r}: {score:.3f}")
        
        recommendations = detector.get_regime_recommendation(regime)
        print(f"\nÖnerilen Strateji: {recommendations.get('strategy', 'N/A')}")
        print(f"Önerilen İndikatörler: {', '.join(recommendations.get('indicators', []))}")
    except Exception as e:
        print(f"Hata: {e}")
    
    # 2. Adaptive Ensemble System
    print("\n\n2. Adaptive Ensemble System Testi:")
    print("-"*40)
    
    ensemble = AdaptiveEnsembleSystem()
    
    try:
        signal = ensemble.generate_ensemble_signal(symbol, '1h')
        print(f"Sembol: {symbol}")
        print(f"Sinyal: {signal['signal']}")
        print(f"Güven: {signal['confidence']:.3f}")
        print(f"Buy Score: {signal['buy_score']:.3f}")
        print(f"Sell Score: {signal['sell_score']:.3f}")
        
        print("\nKategori Detayları:")
        for cat, data in signal['category_signals'].items():
            print(f"  {cat}: {data['signal']} (güç: {data['strength']:.3f}, ağırlık: {data['weight']:.3f})")
    except Exception as e:
        print(f"Hata: {e}")
    
    # 3. Dynamic Risk Manager
    print("\n\n3. Dynamic Risk Manager Testi:")
    print("-"*40)
    
    risk_manager = DynamicRiskManager(initial_capital=50000)
    
    try:
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
        
        position = risk_manager.calculate_position_size(
            signal_data=test_signal,
            symbol=symbol,
            price=245.50,
            timeframe='1h'
        )
        
        print(f"Sembol: {symbol}")
        print(f"Giriş Fiyatı: 245.50 TL")
        print(f"Pozisyon Büyüklüğü: %{position['position_size_pct']*100:.2f}")
        print(f"Pozisyon Değeri: {position['position_value']:,.2f} TL")
        print(f"Hisse Adedi: {position['shares']}")
        print(f"Stop Loss: {position['stop_loss']:.2f} TL")
        print(f"Hedef Fiyat: {position['take_profit']:.2f} TL")
        print(f"Risk/Reward: 1:{position['risk_reward_ratio']:.2f}")
    except Exception as e:
        print(f"Hata: {e}")


def test_integrated_system():
    """Entegre sistemi test et"""
    print("\n\n" + "="*80)
    print("ENTEGRE SİSTEM TESTİ")
    print("="*80)
    
    # Sistemi başlat
    system = IntegratedTradingSystem(initial_capital=50000, mode='paper')
    
    # Test sembolleri
    test_symbols = ['THYAO', 'GARAN', 'AKBNK', 'EREGL', 'ASELS']
    
    print("\nSistem başlatıldı.")
    print(f"Başlangıç Sermayesi: 50,000 TL")
    print(f"Test Sembolleri: {', '.join(test_symbols)}")
    
    # Her sembol için karar üret
    print("\n\nTRADING KARARLARI:")
    print("-"*80)
    
    all_decisions = []
    
    for symbol in test_symbols:
        print(f"\n{symbol} analiz ediliyor...")
        
        try:
            decision = system.generate_trading_decision(symbol, '1h')
            all_decisions.append(decision)
            
            if 'error' not in decision:
                print(f"  Rejim: {decision['regime']} (güven: {decision['regime_confidence']:.3f})")
                print(f"  Sinyal: {decision['signal']} (güven: {decision['confidence']:.3f})")
                
                # Detaylı ensemble bilgisi
                ensemble = decision['ensemble_details']
                print(f"  Ensemble - Buy: {ensemble['buy_score']:.3f}, Sell: {ensemble['sell_score']:.3f}")
                
                # Pozisyon bilgisi
                if decision['signal'] != 'HOLD' and 'position_data' in decision:
                    pos = decision['position_data']
                    print(f"  Önerilen Pozisyon:")
                    print(f"    - Büyüklük: %{pos['position_size_pct']*100:.1f} ({pos['position_value']:,.0f} TL)")
                    print(f"    - Stop Loss: {pos['stop_loss']:.2f} TL")
                    print(f"    - Hedef: {pos['take_profit']:.2f} TL")
                    print(f"    - Risk/Reward: 1:{pos['risk_reward_ratio']:.2f}")
                elif decision.get('risk_blocked'):
                    print("  ⚠️ Risk limitleri nedeniyle işlem engellendi")
            else:
                print(f"  ❌ Hata: {decision.get('error', 'Bilinmeyen hata')}")
                
        except Exception as e:
            print(f"  ❌ Analiz hatası: {e}")
    
    # En güçlü sinyaller
    print("\n\nEN GÜÇLÜ SİNYALLER:")
    print("-"*80)
    
    # Sinyalleri güven skoruna göre sırala
    valid_decisions = [d for d in all_decisions if 'error' not in d and d['signal'] != 'HOLD']
    valid_decisions.sort(key=lambda x: x['confidence'], reverse=True)
    
    if valid_decisions:
        for i, decision in enumerate(valid_decisions[:3], 1):
            print(f"\n{i}. {decision['symbol']} - {decision['signal']}")
            print(f"   Güven: {decision['confidence']:.3f}")
            print(f"   Rejim: {decision['regime']}")
            if 'position_data' in decision:
                print(f"   Pozisyon: {decision['position_data']['position_value']:,.0f} TL")
    else:
        print("Aktif sinyal bulunamadı.")
    
    # Sistem durumu
    print("\n\nSİSTEM DURUMU:")
    print("-"*80)
    
    status = system.get_system_status()
    print(f"Durum: {status['state']}")
    print(f"Aktif Pozisyonlar: {status['active_positions']}")
    print(f"Mevcut Sermaye: {status['current_capital']:,.2f} TL")
    
    # Risk metrikleri
    risk_metrics = status['risk_metrics']
    print(f"\nRisk Metrikleri:")
    print(f"  Toplam İşlem: {risk_metrics['total_trades']}")
    print(f"  Kazanma Oranı: %{risk_metrics['win_rate']*100:.1f}")
    print(f"  Mevcut Drawdown: %{risk_metrics['current_drawdown']*100:.1f}")
    
    # Ensemble istatistikleri
    ensemble_stats = status['ensemble_stats']
    print(f"\nEnsemble İstatistikleri:")
    print(f"  Toplam Sinyal: {ensemble_stats['total_signals']}")
    print(f"  Kategori Ağırlıkları:")
    for cat, weight in ensemble_stats['category_weights'].items():
        print(f"    - {cat}: %{weight*100:.1f}")


def save_test_results(results: dict):
    """Test sonuçlarını kaydet"""
    filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = Path("logs") / filename
    
    filepath.parent.mkdir(exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nTest sonuçları kaydedildi: {filepath}")


def main():
    """Ana test fonksiyonu"""
    print("\n" + "="*80)
    print("ADAPTİF ENSEMBLE TRADING SİSTEMİ TESTİ")
    print("="*80)
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test sonuçları
    results = {
        'test_date': datetime.now().isoformat(),
        'components': {},
        'integrated_system': {}
    }
    
    # Bileşen testleri
    test_individual_components()
    
    # Entegre sistem testi
    test_integrated_system()
    
    # Sonuçları kaydet
    # save_test_results(results)
    
    print("\n\nTest tamamlandı!")


if __name__ == "__main__":
    main()