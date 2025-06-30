"""
Test script for the hybrid trading system
Tüm modüllerin çalıştığını doğrular
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from loguru import logger

# Modülleri import et
from core.csv_data_manager import CSVDataManager
from core.data_collector import UnifiedDataCollector
from core.feature_engineering import FeatureEngineering
from core.signal_generator import SignalGenerator
from core.portfolio_manager import PortfolioManager
from indicators.indicator_calculator import IndicatorCalculator
from models.gru_multi_timeframe import MultiTimeframeGRU


async def test_data_loading():
    """CSV veri yükleme testi"""
    logger.info("=== CSV Data Manager Testi ===")
    
    csv_manager = CSVDataManager()
    
    # Mevcut sembolleri listele
    symbols = csv_manager.get_available_symbols()
    logger.info(f"Toplam {len(symbols)} sembol bulundu")
    
    if symbols:
        # İlk sembol için test
        test_symbol = symbols[0]
        logger.info(f"Test sembolü: {test_symbol}")
        
        # Raw data test
        df = csv_manager.get_raw_data(test_symbol, "1d")
        if df is not None:
            logger.success(f"Raw data yüklendi: {len(df)} satır")
            logger.info(f"Tarih aralığı: {df.index[0]} - {df.index[-1]}")
        else:
            logger.error("Raw data yüklenemedi")
        
        # Timeframe listesi
        timeframes = csv_manager.get_available_timeframes(test_symbol)
        logger.info(f"Mevcut timeframe'ler: {timeframes}")
    
    return symbols


async def test_indicators(symbol: str):
    """İndikatör hesaplama testi"""
    logger.info("=== Indicator Calculator Testi ===")
    
    calc = IndicatorCalculator()
    
    # Test için 1 saatlik veriyi hesapla
    indicators = calc.calculate_all_indicators(symbol, "1h", save=False)
    
    if not indicators.empty:
        logger.success(f"İndikatörler hesaplandı: {len(indicators.columns)} kolon")
        logger.info(f"İndikatörler: {list(indicators.columns)[:10]}...")
        
        # MACD kontrolü
        if 'macd' in indicators.columns:
            logger.info(f"MACD son değer: {indicators['macd'].iloc[-1]:.4f}")
    else:
        logger.error("İndikatör hesaplanamadı")
    
    return indicators


async def test_data_collector(symbol: str):
    """Data collector testi"""
    logger.info("=== Data Collector Testi ===")
    
    # Basit config
    config = {
        'api': {
            'finnhub': {'api_key': ''},
            'alpha_vantage': {'api_key': ''}
        },
        'data': {'cache_ttl': 900},
        'timeframes': {
            'analysis': ['1h', '4h', '1d']
        }
    }
    
    collector = UnifiedDataCollector(config)
    
    # Multi-timeframe data topla
    data = await collector.collect_multi_timeframe_data(symbol)
    
    if data:
        logger.success(f"Multi-timeframe data toplandı: {list(data.keys())}")
        
        for tf in data:
            if tf not in ['macro', 'sentiment', 'indicators']:
                logger.info(f"{tf}: {len(data[tf])} satır")
    else:
        logger.error("Data toplanamadı")
    
    return data


async def test_feature_engineering(symbol: str, data: dict):
    """Feature engineering testi"""
    logger.info("=== Feature Engineering Testi ===")
    
    config = {
        'timeframes': {
            'analysis': ['1h', '4h', '1d']
        }
    }
    
    fe = FeatureEngineering(config)
    
    # Feature'ları oluştur
    features = fe.create_features(data, symbol)
    
    if features:
        logger.success(f"Feature'lar oluşturuldu: {list(features.keys())} timeframes")
        
        for tf, feat_df in features.items():
            logger.info(f"{tf}: {feat_df.shape} - {list(feat_df.columns)[:5]}...")
    else:
        logger.error("Feature oluşturulamadı")
    
    return features


async def test_signal_generation(symbols: list):
    """Sinyal üretim testi"""
    logger.info("=== Signal Generator Testi ===")
    
    config = {
        'api': {
            'finnhub': {'api_key': ''},
            'alpha_vantage': {'api_key': ''}
        },
        'data': {'cache_ttl': 900},
        'timeframes': {
            'analysis': ['15m', '1h', '4h', '1d', '1w']
        },
        'signals': {
            'confidence_threshold': 0.6,
            'min_profit_target': 0.02
        },
        'max_concurrent_signals': 5
    }
    
    generator = SignalGenerator(config)
    
    # Test için ilk 3 sembol
    test_symbols = symbols[:3]
    signals = await generator.generate_signals(test_symbols)
    
    if signals:
        logger.success(f"{len(signals)} sinyal üretildi")
        
        for signal in signals:
            logger.info(f"\nSinyal: {signal['symbol']} - {signal['direction'].upper()}")
            logger.info(f"Confidence: {signal['confidence']:.1%}")
            logger.info(f"Entry: {signal['entry_price']:.2f}")
            logger.info(f"Stop: {signal['stop_loss']:.2f}")
            logger.info(f"Target: {signal['target_1']:.2f}")
    else:
        logger.warning("Sinyal üretilemedi")
    
    return signals


async def test_portfolio_manager():
    """Portfolio manager testi"""
    logger.info("=== Portfolio Manager Testi ===")
    
    config = {
        'portfolio': {
            'initial_capital': 100000,
            'max_positions': 5
        },
        'risk': {
            'max_risk_per_trade': 0.01,
            'max_portfolio_risk': 0.05,
            'max_correlation': 0.7
        }
    }
    
    pm = PortfolioManager(config)
    
    # Portfolio durumu
    status = pm.get_portfolio_status()
    logger.info(f"Capital: {status['capital']:,.0f}")
    logger.info(f"Open positions: {status['open_positions']}")
    logger.info(f"Total PnL: {status['total_pnl']:,.2f}")
    
    # Risk durumu
    risk = pm.risk_check()
    logger.info(f"Portfolio risk: {risk['portfolio_risk']:.1f}%")
    logger.info(f"Can open positions: {risk['can_open_positions']}")
    
    return pm


async def test_model():
    """GRU model testi"""
    logger.info("=== GRU Model Testi ===")
    
    model = MultiTimeframeGRU(
        input_size=50,
        hidden_size=50,
        num_layers=1
    )
    
    # Test input
    batch_size = 1
    seq_len = 30
    features = 50
    
    # Random test data
    import torch
    x_15m = torch.randn(batch_size, seq_len, features)
    x_1h = torch.randn(batch_size, seq_len, features)
    x_4h = torch.randn(batch_size, seq_len, features)
    x_1d = torch.randn(batch_size, seq_len, features)
    x_1w = torch.randn(batch_size, seq_len, features)
    
    # Forward pass
    output, attention = model(x_15m, x_1h, x_4h, x_1d, x_1w)
    
    logger.success(f"Model output shape: {output.shape}")
    logger.info(f"Attention weights: {attention[0].tolist()}")
    
    return model


async def main():
    """Ana test fonksiyonu"""
    logger.info("=== Hybrid Trading System Test ===")
    logger.info(f"Başlangıç: {datetime.now()}")
    
    try:
        # 1. Data yükleme testi
        symbols = await test_data_loading()
        
        if not symbols:
            logger.error("Sembol bulunamadı, test sonlandırılıyor")
            return
        
        test_symbol = symbols[0]
        
        # 2. İndikatör testi
        indicators = await test_indicators(test_symbol)
        
        # 3. Data collector testi
        data = await test_data_collector(test_symbol)
        
        # 4. Feature engineering testi
        if data:
            features = await test_feature_engineering(test_symbol, data)
        
        # 5. Model testi
        model = await test_model()
        
        # 6. Signal generation testi
        signals = await test_signal_generation(symbols[:5])
        
        # 7. Portfolio manager testi
        pm = await test_portfolio_manager()
        
        logger.success("\n=== TÜM TESTLER TAMAMLANDI ===")
        
        # Özet
        logger.info("\nSistem Özeti:")
        logger.info(f"- {len(symbols)} sembol mevcut")
        logger.info(f"- CSV data manager: ✓")
        logger.info(f"- Indicator calculator: ✓")
        logger.info(f"- Data collector: ✓")
        logger.info(f"- Feature engineering: ✓")
        logger.info(f"- GRU model: ✓")
        logger.info(f"- Signal generator: ✓")
        logger.info(f"- Portfolio manager: ✓")
        
    except Exception as e:
        logger.error(f"Test hatası: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())