"""
Calculate and save indicators for training symbols
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from indicators.indicator_calculator import IndicatorCalculator
from loguru import logger

# Initialize calculator
calc = IndicatorCalculator()

# Symbols to process
symbols = ['THYAO', 'GARAN', 'AKBNK', 'EREGL', 'ASELS']
timeframes = ['1h', '4h', '1d']

logger.info("Starting indicator calculation...")

for symbol in symbols:
    logger.info(f"\nProcessing {symbol}...")
    
    for tf in timeframes:
        logger.info(f"  Calculating {tf} indicators...")
        
        try:
            # Calculate and save indicators
            indicators = calc.calculate_all_indicators(symbol, tf, save=True)
            
            if not indicators.empty:
                logger.info(f"    ✓ Calculated {len(indicators.columns)} indicators")
            else:
                logger.warning(f"    ✗ No indicators calculated")
                
        except Exception as e:
            logger.error(f"    ✗ Error: {e}")

logger.info("\nIndicator calculation completed!")