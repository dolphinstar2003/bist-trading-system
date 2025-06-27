#!/usr/bin/env python3
"""
Entegre Sistem Basit Testi
ML olmadan çalışır
"""

import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from strategies.integrated_trading_system import IntegratedTradingSystem

def main():
    print("\n" + "="*80)
    print("ENTEGRE TRADİNG SİSTEMİ TESTİ")
    print("="*80)
    print(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Sistemi başlat (ML olmadan)
    system = IntegratedTradingSystem(initial_capital=50000, mode='paper')
    
    # Test sembolleri
    test_symbols = ['THYAO', 'GARAN', 'AKBNK']
    
    print(f"\nBaşlangıç Sermayesi: 50,000 TL")
    print(f"Test Sembolleri: {', '.join(test_symbols)}")
    print("\n" + "-"*80)
    
    # Her sembol için karar üret
    for symbol in test_symbols:
        print(f"\n{symbol} Analizi:")
        print("-"*40)
        
        try:
            decision = system.generate_trading_decision(symbol, '1h')
            
            if 'error' not in decision:
                print(f"Sinyal: {decision['signal']}")
                print(f"Güven: {decision['confidence']:.3f}")
                print(f"Rejim: {decision['regime']} (güven: {decision['regime_confidence']:.3f})")
                
                # Ensemble detayları
                ensemble = decision['ensemble_details']
                print(f"\nEnsemble Detayları:")
                print(f"  Buy Score: {ensemble.get('buy_score', 0):.3f}")
                print(f"  Sell Score: {ensemble.get('sell_score', 0):.3f}")
                
                # Kategori sinyalleri
                if 'category_signals' in ensemble:
                    print("\nKategori Sinyalleri:")
                    for cat, data in ensemble['category_signals'].items():
                        print(f"  {cat}: {data['signal']} (güç: {data['strength']:.3f})")
                
                # Pozisyon önerisi
                if decision['signal'] != 'HOLD' and 'position_data' in decision:
                    pos = decision['position_data']
                    print(f"\nPozisyon Önerisi:")
                    print(f"  Büyüklük: %{pos['position_size_pct']*100:.1f}")
                    print(f"  Değer: {pos['position_value']:,.0f} TL")
                    print(f"  Stop Loss: {pos['stop_loss']:.2f}")
                    print(f"  Hedef: {pos['take_profit']:.2f}")
            else:
                print(f"❌ Hata: {decision.get('error', 'Bilinmeyen hata')}")
                
        except Exception as e:
            print(f"❌ Sistem hatası: {e}")
    
    # Sistem durumu
    print("\n\n" + "="*80)
    print("SİSTEM DURUMU")
    print("="*80)
    
    try:
        status = system.get_system_status()
        print(f"Durum: {status['state']}")
        print(f"Aktif Pozisyonlar: {status['active_positions']}")
        print(f"Mevcut Sermaye: {status['current_capital']:,.2f} TL")
        
        # Risk metrikleri
        risk = status['risk_metrics']
        print(f"\nRisk Metrikleri:")
        print(f"  Toplam İşlem: {risk['total_trades']}")
        print(f"  Kazanma Oranı: {risk['win_rate']*100:.1f}%")
        print(f"  Sharpe Ratio: {risk['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {risk['max_drawdown']*100:.1f}%")
        
        # Ensemble istatistikleri
        ensemble_stats = status['ensemble_stats']
        print(f"\nEnsemble İstatistikleri:")
        print(f"  Toplam Sinyal: {ensemble_stats['total_signals']}")
        
        if 'category_weights' in ensemble_stats:
            print("\nKategori Ağırlıkları:")
            for cat, weight in ensemble_stats['category_weights'].items():
                perf = ensemble_stats['category_performance'].get(cat, {})
                print(f"  {cat}: %{weight*100:.1f} (skor: {perf.get('score', 1.0):.3f})")
                
    except Exception as e:
        print(f"Sistem durumu hatası: {e}")
    
    print("\n" + "="*80)
    print("Test tamamlandı!")


if __name__ == "__main__":
    main()