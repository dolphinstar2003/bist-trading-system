#!/usr/bin/env python3
"""
Simple ML Signal Test
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.ml_trading_system_fixed import MLTradingSystem
from ml_models.ml_signal_generator import MLSignalGenerator
from config.assets import ASSETS


def test_simple():
    """Simple test with just raw predictions"""
    logger.info("Testing raw ML predictions...")
    
    # Create ML system and load saved models
    ml_system = MLTradingSystem()
    
    # Load the models
    timeframe = '1d'
    success = ml_system.load_models(timeframe, timestamp=None)
    
    if not success:
        logger.error("Failed to load models")
        return
    
    # Test all symbols
    test_symbols = ASSETS
    
    logger.info("\nRaw ML Predictions (no threshold):")
    logger.info("="*60)
    
    buy_count = 0
    sell_count = 0
    hold_count = 0
    
    for symbol in test_symbols:
        prediction = ml_system.predict_ensemble(symbol, timeframe)
        if prediction:
            signal = prediction['ensemble_prediction']
            conf = prediction['confidence']
            
            if signal == 1:
                buy_count += 1
                signal_text = "BUY"
            elif signal == -1:
                sell_count += 1
                signal_text = "SELL"
            else:
                hold_count += 1
                signal_text = "HOLD"
            
            logger.info(f"{symbol}: {signal_text} (Confidence: {conf:.1%}) - Models: {prediction['individual_predictions']}")
    
    logger.info("\nSummary:")
    logger.info(f"Buy signals: {buy_count}")
    logger.info(f"Sell signals: {sell_count}")
    logger.info(f"Hold signals: {hold_count}")
    
    # Now test with signal generator
    logger.info("\n" + "="*60)
    logger.info("Testing with Signal Generator (0.4 threshold):")
    logger.info("="*60)
    
    generator = MLSignalGenerator()
    generator.ml_system = ml_system
    generator.min_confidence = 0.4  # Lower threshold
    
    signals = []
    for symbol in test_symbols:
        signal = generator.generate_signal(symbol, timeframe)
        if signal:
            signals.append(signal)
            direction = 'BUY' if signal.signal > 0 else 'SELL' if signal.signal < 0 else 'HOLD'
            logger.info(f"{symbol}: {direction} (Confidence: {signal.confidence:.1%})")
    
    # Summary
    buy_signals = sum(1 for s in signals if s.signal > 0)
    sell_signals = sum(1 for s in signals if s.signal < 0)
    hold_signals = sum(1 for s in signals if s.signal == 0)
    
    logger.info("\nWith 0.4 threshold:")
    logger.info(f"Buy signals: {buy_signals}")
    logger.info(f"Sell signals: {sell_signals}")
    logger.info(f"Hold signals: {hold_signals}")


if __name__ == "__main__":
    test_simple()