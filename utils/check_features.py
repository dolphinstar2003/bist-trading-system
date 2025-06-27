#!/usr/bin/env python3
"""
Feature'ları kontrol et
"""

import sys
from pathlib import Path

# Proje ana dizinini Python path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_models.feature_engineering import FeatureEngineering
from indicators.indicator_calculator import IndicatorCalculator
import pandas as pd

fe = FeatureEngineering()
calc = IndicatorCalculator()

# İndikatör verisini kontrol et
print("Checking indicator data for AKBNK 1h...\n")

indicators = [
    'williams_vix_fix', 'wavetrend', 'squeeze_momentum', 
    'adx_di', 'supertrend', 'macd', 'lorentzian', 'trend_vanguard'
]

for ind in indicators:
    try:
        data = calc.get_indicator_data('AKBNK', '1h', ind)
        if data is not None:
            print(f"\n{ind}:")
            print(f"  Columns: {list(data.columns)}")
            
            # String kolonları bul
            string_cols = []
            for col in data.columns:
                if data[col].dtype == 'object' or data[col].dtype == 'O':
                    string_cols.append(col)
                    print(f"  ⚠️  {col} is string type!")
                    print(f"      Unique values: {data[col].unique()[:5]}")
            
            if not string_cols:
                print("  ✓ All columns are numeric")
    except Exception as e:
        print(f"  ❌ Error: {e}")

# Feature matrix'i kontrol et
print("\n\nCreating feature matrix...")
features, targets = fe.create_feature_matrix('AKBNK', '1h')

if not features.empty:
    print(f"\nFeature matrix shape: {features.shape}")
    
    # String kolonları kontrol et
    string_features = []
    for col in features.columns:
        if features[col].dtype == 'object':
            string_features.append(col)
    
    if string_features:
        print(f"\n⚠️  Found {len(string_features)} string features:")
        for col in string_features:
            print(f"  - {col}: {features[col].unique()[:5]}")
    else:
        print("\n✓ All features are numeric")