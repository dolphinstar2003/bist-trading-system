#!/usr/bin/env python3
"""
Download data for missing stocks
"""

import os
import sys
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Missing stocks
missing_stocks = [
    'TCELL', 'TOASO', 'TUPRS', 'VAKBN', 'VESTL', 'YKBNK', 
    'ALARK', 'ALBRK', 'ANHYT', 'AYGAZ', 'BAGFS', 'BRSAN', 
    'CCOLA', 'CEMTS', 'CIMSA', 'GENIL', 'GLYHO', 'GOZDE', 
    'AYEN', 'ZOREN', 'QNBFB', 'KLNMA', 'YYAPI', 'KUYAS', 
    'ANELE', 'TURSG'
]

def download_stock_data(symbol: str, start_date: str = "2018-01-01"):
    """Download historical data for a single stock"""
    try:
        # Add .IS suffix for Istanbul Stock Exchange
        ticker = f"{symbol}.IS"
        
        logger.info(f"Downloading {symbol}...")
        
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=datetime.now().strftime('%Y-%m-%d'), interval='1d')
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return False
            
        # Prepare data
        df = df.reset_index()
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        
        # Keep only OHLCV columns
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Save to CSV
        output_dir = f"data/raw"
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{symbol}_1d_raw.csv")
        df.to_csv(output_path, index=False)
        
        logger.info(f"Downloaded {symbol}: {len(df)} days of data")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading {symbol}: {e}")
        return False


def main():
    """Download all missing stocks"""
    logger.info(f"Starting download for {len(missing_stocks)} missing stocks...")
    
    successful = []
    failed = []
    
    for i, symbol in enumerate(missing_stocks, 1):
        logger.info(f"\nProcessing {i}/{len(missing_stocks)}: {symbol}")
        
        if download_stock_data(symbol):
            successful.append(symbol)
        else:
            failed.append(symbol)
            
        # Small delay to avoid rate limiting
        if i < len(missing_stocks):
            time.sleep(1)
    
    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total stocks: {len(missing_stocks)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessfully downloaded ({len(successful)}):")
        print(", ".join(successful))
        
    if failed:
        print(f"\nFailed to download ({len(failed)}):")
        print(", ".join(failed))
        
    print("\nNext step: Run advanced_stop_loss_analyzer.py to analyze all stocks")


if __name__ == "__main__":
    main()