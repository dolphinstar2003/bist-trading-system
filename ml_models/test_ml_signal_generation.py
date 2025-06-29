#!/usr/bin/env python3
"""
Test ML Signal Generation with Trained Models
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.ml_trading_system_fixed import MLTradingSystem
from ml_models.ml_signal_generator import MLSignalGenerator
from config.assets import ASSETS


def test_signal_generation_with_trained_models():
    """Test signal generation using pre-trained models"""
    logger.info("Testing Signal Generation with Pre-trained Models...")
    
    # Create ML system and load saved models
    ml_system = MLTradingSystem()
    
    # Load the models we just trained
    timeframe = '1d'
    success = ml_system.load_models(timeframe, timestamp=None)  # Will find latest
    
    if not success:
        logger.error("Failed to load models")
        return
    
    logger.info(f"Successfully loaded models for {timeframe}")
    
    # Create signal generator
    generator = MLSignalGenerator()
    generator.ml_system = ml_system
    
    # Test with main symbols
    test_symbols = ASSETS[:20]
    
    # Generate individual signals
    logger.info(f"\nGenerating signals for {len(test_symbols)} symbols...")
    signals = []
    
    for symbol in test_symbols:
        signal = generator.generate_signal(symbol, timeframe)
        if signal:
            signals.append(signal)
            direction = 'BUY' if signal.signal > 0 else 'SELL' if signal.signal < 0 else 'HOLD'
            logger.info(f"{symbol}: {direction} (Confidence: {signal.confidence:.1%})")
    
    # Generate portfolio signals
    logger.info(f"\nGenerating portfolio signals...")
    portfolio = generator.generate_portfolio_signals(test_symbols, timeframe)
    
    # Print report
    report = generator.create_trading_report(portfolio)
    print("\n" + report)
    
    # Save signals
    generator.save_signals(portfolio, "ml_models/test_signals_1d.json")
    logger.info("Signals saved to ml_models/test_signals_1d.json")
    
    # Show summary
    logger.info(f"\nSummary:")
    logger.info(f"Total symbols analyzed: {len(test_symbols)}")
    logger.info(f"Signals generated: {len(signals)}")
    logger.info(f"Buy signals: {sum(1 for s in signals if s.signal > 0)}")
    logger.info(f"Sell signals: {sum(1 for s in signals if s.signal < 0)}")
    logger.info(f"Hold signals: {sum(1 for s in signals if s.signal == 0)}")


def main():
    """Run the test"""
    test_signal_generation_with_trained_models()


if __name__ == "__main__":
    main()