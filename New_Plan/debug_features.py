"""
Debug feature engineering issue
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
import numpy as np
from core.csv_data_manager import CSVDataManager
from indicators.indicator_calculator import IndicatorCalculator
from core.feature_engineering import FeatureEngineering
import json

# Load config
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize components
csv_manager = CSVDataManager()
indicator_calc = IndicatorCalculator()
feature_eng = FeatureEngineering(config)

# Test with THYAO
symbol = 'THYAO'
print(f"\nTesting {symbol}...")

# Load data for each timeframe
data = {'indicators': {}}
for tf in config['timeframes']['analysis']:
    print(f"\nLoading {tf} data...")
    df = csv_manager.get_raw_data(symbol, tf)
    if df is None:
        print(f"  No raw data found for {tf}")
        continue
    
    print(f"  Raw data shape: {df.shape}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    # Calculate indicators
    print(f"  Calculating indicators...")
    indicators = indicator_calc.calculate_all_indicators(symbol, tf, save=False)
    
    if indicators.empty:
        print(f"  No indicators calculated")
    else:
        print(f"  Indicators shape: {indicators.shape}")
        print(f"  Indicator columns: {list(indicators.columns)[:5]}...")
        
        # Merge
        df = pd.concat([df, indicators], axis=1)
    
    data['indicators'][tf] = df
    print(f"  Final shape: {df.shape}")

# Add dummy macro/sentiment
data['macro'] = {'vix': 20, 'usdtry': 30}
data['sentiment'] = {'score': 0, 'count': 0}

# Check what columns we have
print("\nChecking available columns in data:")
for tf, df in data['indicators'].items():
    print(f"  {tf}: {list(df.columns)}")

# Try to create features
print("\nCreating features...")
features = feature_eng.create_features(data, symbol)

print(f"\nFeatures type: {type(features)}")
if features:
    print(f"Number of timeframes in features: {len(features)}")
    for tf, feat_df in features.items():
        print(f"  {tf}: shape={feat_df.shape if hasattr(feat_df, 'shape') else 'N/A'}")
else:
    print("No features created!")
    print("\nDebugging: Check if timeframes match")
    print(f"Config timeframes: {config['timeframes']['analysis']}")
    print(f"Data indicators keys: {list(data['indicators'].keys())}")