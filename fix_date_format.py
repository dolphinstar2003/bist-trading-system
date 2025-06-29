#!/usr/bin/env python3
"""
Fix date format for newly downloaded stocks
"""

import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stocks that need fixing
stocks_to_fix = [
    'TCELL', 'TOASO', 'TUPRS', 'VAKBN', 'VESTL', 'YKBNK', 
    'ALARK', 'ALBRK', 'ANHYT', 'AYGAZ', 'BAGFS', 'BRSAN', 
    'CCOLA', 'CEMTS', 'CIMSA', 'GENIL', 'GLYHO', 'GOZDE', 
    'AYEN', 'ZOREN', 'KLNMA', 'YYAPI', 'KUYAS', 'ANELE', 'TURSG'
]

def fix_date_format(symbol):
    """Fix date format for a stock"""
    try:
        path = f"data/raw/{symbol}_1d_raw.csv"
        if os.path.exists(path):
            # Read CSV
            df = pd.read_csv(path)
            
            # Check if 'date' column exists (lowercase)
            if 'date' in df.columns:
                # Rename to 'Date' with capital D
                df = df.rename(columns={'date': 'Date'})
                
                # Ensure date format
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                
                # Save back
                df.to_csv(path, index=False)
                logger.info(f"Fixed {symbol}")
                return True
            elif 'Date' in df.columns:
                # Already has correct column name, just ensure format
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                df.to_csv(path, index=False)
                logger.info(f"Format updated for {symbol}")
                return True
            else:
                logger.warning(f"No date column found in {symbol}")
                return False
    except Exception as e:
        logger.error(f"Error fixing {symbol}: {e}")
        return False

def main():
    """Fix all stocks"""
    logger.info(f"Fixing date format for {len(stocks_to_fix)} stocks...")
    
    successful = 0
    for symbol in stocks_to_fix:
        if fix_date_format(symbol):
            successful += 1
            
    logger.info(f"Fixed {successful}/{len(stocks_to_fix)} stocks")
    
    # Also check column names
    print("\nColumn check:")
    for symbol in stocks_to_fix[:5]:  # Check first 5
        path = f"data/raw/{symbol}_1d_raw.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, nrows=1)
            print(f"{symbol}: {list(df.columns)}")

if __name__ == "__main__":
    main()