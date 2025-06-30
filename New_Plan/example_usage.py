"""
Hybrid Trading System - Örnek Kullanım
Sistemin nasıl kullanılacağını gösteren pratik örnekler
"""

import asyncio
import json
from datetime import datetime
from loguru import logger
import pandas as pd

# Sistem modülleri
from core.csv_data_manager import CSVDataManager
from core.data_collector import UnifiedDataCollector
from core.feature_engineering import FeatureEngineering
from core.signal_generator import SignalGenerator
from core.portfolio_manager import PortfolioManager
from indicators.indicator_calculator import IndicatorCalculator


# Konfigurasyon yükle
def load_config():
    """Konfigurasyon dosyasını yükle"""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except:
        logger.warning("config.json bulunamadı, default config kullanılıyor")
        return {
            'api': {'finnhub': {'api_key': ''}, 'alpha_vantage': {'api_key': ''}},
            'data': {'cache_ttl': 900},
            'timeframes': {'analysis': ['1h', '4h', '1d']},
            'signals': {'confidence_threshold': 0.65, 'min_profit_target': 0.02},
            'portfolio': {'initial_capital': 100000, 'max_positions': 5},
            'risk': {
                'max_risk_per_trade': 0.01,
                'max_portfolio_risk': 0.05,
                'max_correlation': 0.7
            }
        }


# ÖRNEK 1: Mevcut verileri kontrol et
def check_available_data():
    """Mevcut CSV verilerini kontrol et"""
    print("\n=== MEVCUT VERİ KONTROLÜ ===")
    
    csv_manager = CSVDataManager()
    symbols = csv_manager.get_available_symbols()
    
    print(f"Toplam {len(symbols)} sembol bulundu\n")
    
    # İlk 5 sembol için detay
    for symbol in symbols[:5]:
        timeframes = csv_manager.get_available_timeframes(symbol)
        print(f"{symbol}: {', '.join(timeframes)}")
        
        # 1 günlük veri kontrolü
        integrity = csv_manager.check_data_integrity(symbol, '1d')
        if integrity['exists']:
            print(f"  - Satır sayısı: {integrity['rows']}")
            print(f"  - Tarih aralığı: {integrity['start_date'].date()} - {integrity['end_date'].date()}")


# ÖRNEK 2: İndikatör hesapla
async def calculate_indicators_example():
    """Bir sembol için indikatör hesapla"""
    print("\n=== İNDİKATÖR HESAPLAMA ===")
    
    calculator = IndicatorCalculator()
    symbol = 'AKBNK'
    timeframe = '1h'
    
    print(f"{symbol} {timeframe} için indikatörler hesaplanıyor...")
    
    # Hesapla (kaydetmeden)
    indicators = calculator.calculate_all_indicators(symbol, timeframe, save=False)
    
    if not indicators.empty:
        print(f"\nHesaplanan indikatörler ({len(indicators.columns)} adet):")
        
        # Son değerler
        latest = indicators.iloc[-1]
        print(f"  - MACD: {latest.get('macd', 'N/A'):.4f}")
        print(f"  - RSI: {latest.get('rsi', 'N/A'):.2f}")
        print(f"  - ATR: {latest.get('atr', 'N/A'):.2f}")
        print(f"  - ADX: {latest.get('adx', 'N/A'):.2f}")


# ÖRNEK 3: Sinyal üret
async def generate_signals_example():
    """Trading sinyalleri üret"""
    print("\n=== SİNYAL ÜRETİMİ ===")
    
    config = load_config()
    generator = SignalGenerator(config)
    
    # Test sembolleri
    symbols = ['THYAO', 'GARAN', 'EREGL', 'AKBNK', 'ASELS']
    
    print(f"{len(symbols)} sembol için sinyal üretiliyor...")
    signals = await generator.generate_signals(symbols)
    
    if signals:
        print(f"\n{len(signals)} sinyal üretildi:\n")
        
        for i, signal in enumerate(signals, 1):
            print(f"{i}. {signal['symbol']} - {signal['direction'].upper()}")
            print(f"   Entry: {signal['entry_price']:.2f}")
            print(f"   Stop: {signal['stop_loss']:.2f} ({abs(signal['entry_price']-signal['stop_loss'])/signal['entry_price']*100:.1f}%)")
            print(f"   Target: {signal['target_1']:.2f} ({abs(signal['target_1']-signal['entry_price'])/signal['entry_price']*100:.1f}%)")
            print(f"   Confidence: {signal['confidence']:.1%}")
            print(f"   Position Size: {signal['position_size_pct']:.1%}\n")
    else:
        print("Sinyal üretilemedi")
    
    return signals


# ÖRNEK 4: Portfolio yönetimi
def portfolio_management_example(signals):
    """Portfolio yönetimi örneği"""
    print("\n=== PORTFOLIO YÖNETİMİ ===")
    
    config = load_config()
    pm = PortfolioManager(config)
    
    # Mevcut durum
    status = pm.get_portfolio_status()
    print(f"Başlangıç Sermaye: {pm.initial_capital:,.0f} TRY")
    print(f"Mevcut Sermaye: {status['capital']:,.0f} TRY")
    print(f"Açık Pozisyonlar: {status['open_positions']}")
    
    # Risk durumu
    risk = pm.risk_check()
    print(f"\nRisk Durumu:")
    print(f"  - Portfolio Risk: {risk['portfolio_risk']:.1f}%")
    print(f"  - Risk Limiti: {risk['risk_limit']:.1f}%")
    print(f"  - Yeni Pozisyon Açılabilir: {'EVET' if risk['can_open_positions'] else 'HAYIR'}")
    
    # Sinyal işleme simülasyonu
    if signals and risk['can_open_positions']:
        print("\n--- Sinyal İşleme Simülasyonu ---")
        
        for signal in signals[:2]:  # İlk 2 sinyal
            order = asyncio.run(pm.process_signal(signal))
            
            if order:
                print(f"\nEmir oluşturuldu:")
                print(f"  Symbol: {order['symbol']}")
                print(f"  Side: {order['side']}")
                print(f"  Quantity: {order['quantity']}")
                print(f"  Price: {order['price']:.2f}")
                
                # Simüle execution
                pm.execute_order(order, order['price'])
                print(f"  ✓ Pozisyon açıldı")
    
    # Final durum
    final_status = pm.get_portfolio_status()
    print(f"\nFinal Durum:")
    print(f"  - Açık Pozisyonlar: {final_status['open_positions']}")
    print(f"  - Toplam Equity: {final_status['total_equity']:,.0f} TRY")


# ÖRNEK 5: Multi-timeframe analiz
async def multi_timeframe_analysis():
    """Multi-timeframe veri analizi"""
    print("\n=== MULTI-TIMEFRAME ANALİZ ===")
    
    config = load_config()
    collector = UnifiedDataCollector(config)
    
    symbol = 'THYAO'
    print(f"{symbol} için multi-timeframe veri toplanıyor...")
    
    # Veri topla
    data = await collector.collect_multi_timeframe_data(symbol)
    
    if data:
        print(f"\nToplanan veriler:")
        for tf in ['1h', '4h', '1d']:
            if tf in data:
                df = data[tf]
                print(f"\n{tf}:")
                print(f"  - Satır sayısı: {len(df)}")
                print(f"  - Son kapanış: {df['close'].iloc[-1]:.2f}")
                
                if 'indicators' in data and tf in data['indicators']:
                    ind_df = data['indicators'][tf]
                    if 'macd' in ind_df.columns:
                        print(f"  - MACD: {ind_df['macd'].iloc[-1]:.4f}")
                    if 'rsi' in ind_df.columns:
                        print(f"  - RSI: {ind_df['rsi'].iloc[-1]:.2f}")


# ÖRNEK 6: Feature engineering
async def feature_engineering_example():
    """Feature engineering örneği"""
    print("\n=== FEATURE ENGINEERING ===")
    
    config = load_config()
    fe = FeatureEngineering(config)
    collector = UnifiedDataCollector(config)
    
    symbol = 'GARAN'
    
    # Veri topla
    data = await collector.collect_multi_timeframe_data(symbol)
    
    if data:
        # Feature oluştur
        features = fe.create_features(data, symbol)
        
        print(f"{symbol} için feature'lar oluşturuldu:")
        
        for tf, feat_df in features.items():
            if tf in ['1h', '4h', '1d']:
                print(f"\n{tf}: {feat_df.shape[0]} satır, {feat_df.shape[1]} feature")
                
                # Örnek feature'lar
                print("  Örnek feature'lar:")
                sample_features = ['returns_1', 'macd_signal', 'rsi_overbought', 
                                 'trend_strength', 'momentum_score']
                
                for feat in sample_features:
                    if feat in feat_df.columns:
                        value = feat_df[feat].iloc[-1]
                        print(f"    - {feat}: {value:.4f}")


# ANA PROGRAM
async def main():
    """Tüm örnekleri çalıştır"""
    print("=" * 60)
    print("HYBRID TRADING SYSTEM - ÖRNEK KULLANIM")
    print("=" * 60)
    
    try:
        # 1. Veri kontrolü
        check_available_data()
        
        # 2. İndikatör hesaplama
        await calculate_indicators_example()
        
        # 3. Multi-timeframe analiz
        await multi_timeframe_analysis()
        
        # 4. Feature engineering
        await feature_engineering_example()
        
        # 5. Sinyal üretimi
        signals = await generate_signals_example()
        
        # 6. Portfolio yönetimi
        if signals:
            portfolio_management_example(signals)
        
        print("\n" + "=" * 60)
        print("TÜM ÖRNEKLER TAMAMLANDI")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Async event loop çalıştır
    asyncio.run(main())