#!/usr/bin/env python3
"""
Test ML Signals with Lower Confidence Threshold
"""

import sys
from pathlib import Path
from loguru import logger
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.ml_trading_system_fixed import MLTradingSystem
from ml_models.ml_signal_generator import MLSignalGenerator
from config.assets import ASSETS


def test_signals_with_different_thresholds():
    """Test signal generation with various confidence thresholds"""
    logger.info("Testing ML signals with different confidence thresholds...")
    
    # Create ML system and load saved models
    ml_system = MLTradingSystem()
    
    # Load the models
    timeframe = '1d'
    success = ml_system.load_models(timeframe, timestamp=None)
    
    if not success:
        logger.error("Failed to load models")
        return
    
    # Test different confidence thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6]
    test_symbols = ASSETS[:30]  # Test with more symbols
    
    for threshold in thresholds:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with confidence threshold: {threshold:.0%}")
        logger.info(f"{'='*60}")
        
        # Create signal generator with custom threshold
        generator = MLSignalGenerator()
        generator.ml_system = ml_system
        generator.min_confidence = threshold
        
        # Generate signals
        buy_signals = []
        sell_signals = []
        hold_signals = []
        all_confidences = []
        
        for symbol in test_symbols:
            signal = generator.generate_signal(symbol, timeframe)
            if signal:
                all_confidences.append(signal.confidence)
                
                if signal.signal > 0:
                    buy_signals.append((symbol, signal.confidence))
                elif signal.signal < 0:
                    sell_signals.append((symbol, signal.confidence))
                else:
                    hold_signals.append((symbol, signal.confidence))
        
        # Show results
        logger.info(f"\nResults for threshold {threshold:.0%}:")
        logger.info(f"Total signals: {len(all_confidences)}")
        logger.info(f"Buy signals: {len(buy_signals)}")
        logger.info(f"Sell signals: {len(sell_signals)}")
        logger.info(f"Hold signals: {len(hold_signals)}")
        
        if all_confidences:
            avg_confidence = sum(all_confidences) / len(all_confidences)
            logger.info(f"Average confidence: {avg_confidence:.1%}")
            logger.info(f"Max confidence: {max(all_confidences):.1%}")
            logger.info(f"Min confidence: {min(all_confidences):.1%}")
        
        # Show top buy signals
        if buy_signals:
            logger.info("\nTop BUY signals:")
            for symbol, conf in sorted(buy_signals, key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  {symbol}: {conf:.1%}")
        
        # Show top sell signals
        if sell_signals:
            logger.info("\nTop SELL signals:")
            for symbol, conf in sorted(sell_signals, key=lambda x: x[1], reverse=True)[:5]:
                logger.info(f"  {symbol}: {conf:.1%}")
    
    # Test with very low threshold to see raw ML predictions
    logger.info(f"\n{'='*60}")
    logger.info("Raw ML predictions (threshold = 0.1):")
    logger.info(f"{'='*60}")
    
    generator.min_confidence = 0.1
    raw_predictions = {}
    
    for symbol in test_symbols[:10]:  # Just first 10 symbols
        prediction = ml_system.predict_ensemble(symbol, timeframe)
        if prediction:
            raw_predictions[symbol] = {
                'signal': prediction['ensemble_prediction'],
                'confidence': prediction['confidence'],
                'models': prediction['individual_predictions']
            }
    
    # Show raw predictions
    for symbol, pred in raw_predictions.items():
        logger.info(f"\n{symbol}:")
        logger.info(f"  Ensemble signal: {pred['signal']}")
        logger.info(f"  Confidence: {pred['confidence']:.1%}")
        logger.info(f"  Model votes: {pred['models']}")


def main():
    """Run the test"""
    test_signals_with_different_thresholds()


if __name__ == "__main__":
    main()