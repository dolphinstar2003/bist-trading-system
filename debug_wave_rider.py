#!/usr/bin/env python3
"""
Debug Wave Rider System
"""

import pandas as pd
import os

# Check optimal parameters
advanced_params_path = 'data/analysis/advanced_trading_parameters.csv'
if os.path.exists(advanced_params_path):
    df = pd.read_csv(advanced_params_path)
    
    # Wave Rider stocks
    wave_stocks = ['GARAN', 'PGSUS', 'YKBNK', 'TAVHL', 'THYAO',
                   'DOHOL', 'BRSAN', 'ISCTR', 'FROTO', 'AKSEN',
                   'ASELS', 'ARCLK', 'SAHOL', 'AEFES', 'TKFEN',
                   'ENKAI', 'EKGYO', 'DOAS', 'AKBNK', 'TUPRS']
    
    print("WAVE RIDER STOCKS - OPTIMAL PARAMETERS:")
    print("-" * 80)
    print(f"{'Symbol':<8} {'Stop Loss':>10} {'Trailing':>10} {'Take Profit':>12} {'R/R Ratio':>10}")
    print("-" * 80)
    
    for stock in wave_stocks:
        if stock in df['symbol'].values:
            row = df[df['symbol'] == stock].iloc[0]
            print(f"{stock:<8} {row['optimal_stop_loss']:>9.1f}% {row['optimal_trailing_stop']:>10} "
                  f"{row['optimal_take_profit']:>11.1f}% {row['risk_reward_ratio']:>10.2f}")
    
    # Failed stocks from results
    print("\n\nFAILED TRADES ANALYSIS:")
    print("-" * 80)
    failed = ['PGSUS', 'AEFES', 'DOAS', 'GARAN', 'ISCTR']
    
    for stock in failed:
        if stock in df['symbol'].values:
            row = df[df['symbol'] == stock].iloc[0]
            print(f"\n{stock}:")
            print(f"  Optimal Stop Loss: {row['optimal_stop_loss']:.1f}%")
            print(f"  Volatility: {row['annual_volatility']:.1f}%")
            print(f"  Avg Drawdown: {row['avg_drawdown']:.1f}%")
            print(f"  Max Drawdown: {row['max_drawdown']:.1f}%")

# Check 4h data availability
print("\n\n4H DATA CHECK:")
print("-" * 50)
for stock in ['GARAN', 'PGSUS', 'AEFES']:
    path_4h = f"data/raw/{stock}_4h_raw.csv"
    path_1h = f"data/raw/{stock}_1h_raw.csv"
    
    has_4h = os.path.exists(path_4h)
    has_1h = os.path.exists(path_1h)
    
    print(f"{stock}: 4h={has_4h}, 1h={has_1h}")
    
    if has_4h:
        df_4h = pd.read_csv(path_4h)
        print(f"  4h data points: {len(df_4h)}")