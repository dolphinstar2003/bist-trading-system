"""
Check data availability for backtest
"""

from core.csv_data_manager import CSVDataManager
from indicators.indicator_calculator import IndicatorCalculator
import pandas as pd
from datetime import datetime

# Initialize
csv_manager = CSVDataManager()
calc = IndicatorCalculator()

# Test symbols
symbols = ['THYAO', 'GARAN', 'AKBNK']
timeframes = ['1h', '4h', '1d']

print("Checking data availability for backtest...\n")

for symbol in symbols:
    print(f"\n{symbol}:")
    print("-" * 40)
    
    for tf in timeframes:
        # Get raw data
        df = csv_manager.get_raw_data(symbol, tf)
        
        if df is not None:
            # Date range
            start_date = df.index[0]
            end_date = df.index[-1]
            
            # Check for 2024 data
            df_2024 = df[(df.index >= '2024-01-01') & (df.index <= '2024-12-31')]
            
            print(f"  {tf}: {len(df)} bars total")
            print(f"       Date range: {start_date.date()} to {end_date.date()}")
            print(f"       2024 data: {len(df_2024)} bars")
            
            # Calculate a quick MACD
            if len(df) > 50:
                ema_12 = df['close'].ewm(span=12, adjust=False).mean()
                ema_26 = df['close'].ewm(span=26, adjust=False).mean()
                macd = ema_12 - ema_26
                
                # Check for NaN
                nan_count = macd.isna().sum()
                print(f"       MACD NaN count: {nan_count}")
        else:
            print(f"  {tf}: No data available")

print("\n\nRecommended backtest period based on data availability:")
print("2024-01-01 to 2024-12-31")