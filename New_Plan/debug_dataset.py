"""Debug dataset issue"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from train_model import TradingDataset

# Create dummy data
features = {
    '1h': pd.DataFrame(np.random.randn(100, 10)),
    '4h': pd.DataFrame(np.random.randn(100, 10)),
    '1d': pd.DataFrame(np.random.randn(100, 10))
}

labels = np.random.randint(0, 2, 100)

# Create dataset
dataset = TradingDataset(features, labels, sequence_length=30)

print(f"Dataset length: {len(dataset)}")

# Test a few samples
for i in [0, 10, 29, 30, 31, 99]:
    try:
        sample, label = dataset[i]
        print(f"\nIndex {i}:")
        for tf, tensor in sample.items():
            print(f"  {tf}: shape={tensor.shape}")
    except Exception as e:
        print(f"\nIndex {i}: ERROR - {e}")