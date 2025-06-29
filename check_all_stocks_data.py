#!/usr/bin/env python3
"""
Check data availability for all stocks in settings.json
"""

import os
import json
import pandas as pd
from datetime import datetime

def check_stock_data():
    """Check data availability for all stocks"""
    
    # Load stock list from settings
    with open('settings.json', 'r') as f:
        settings = json.load(f)
    
    all_symbols = settings['trading']['symbols']
    print(f"Total symbols in settings: {len(all_symbols)}\n")
    
    # Check each stock
    stock_info = []
    
    for symbol in all_symbols:
        path = f"data/raw/{symbol}_1d_raw.csv"
        
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df['Date'] = pd.to_datetime(df['Date'])
                
                if len(df) > 0:
                    first_date = df['Date'].min()
                    last_date = df['Date'].max()
                    total_days = len(df)
                    
                    # Check if data after 2020
                    df_2020 = df[df['Date'] >= '2020-01-01']
                    days_after_2020 = len(df_2020)
                    
                    stock_info.append({
                        'symbol': symbol,
                        'first_date': first_date,
                        'last_date': last_date,
                        'total_days': total_days,
                        'days_after_2020': days_after_2020,
                        'status': 'OK' if days_after_2020 >= 200 else 'LIMITED'
                    })
                else:
                    stock_info.append({
                        'symbol': symbol,
                        'first_date': None,
                        'last_date': None,
                        'total_days': 0,
                        'days_after_2020': 0,
                        'status': 'EMPTY'
                    })
            except Exception as e:
                stock_info.append({
                    'symbol': symbol,
                    'first_date': None,
                    'last_date': None,
                    'total_days': 0,
                    'days_after_2020': 0,
                    'status': f'ERROR: {str(e)}'
                })
        else:
            stock_info.append({
                'symbol': symbol,
                'first_date': None,
                'last_date': None,
                'total_days': 0,
                'days_after_2020': 0,
                'status': 'NO_FILE'
            })
    
    # Create DataFrame
    df_info = pd.DataFrame(stock_info)
    
    # Summary
    print("=" * 80)
    print("DATA AVAILABILITY SUMMARY")
    print("=" * 80)
    
    status_counts = df_info['status'].value_counts()
    for status, count in status_counts.items():
        print(f"{status}: {count} stocks")
    
    print("\n" + "=" * 80)
    print("STOCKS WITH SUFFICIENT DATA (200+ days after 2020)")
    print("=" * 80)
    
    good_stocks = df_info[df_info['status'] == 'OK'].sort_values('symbol')
    print(f"\nTotal: {len(good_stocks)} stocks")
    print(f"Symbols: {', '.join(good_stocks['symbol'].tolist())}")
    
    print("\n" + "=" * 80)
    print("STOCKS WITH LIMITED DATA")
    print("=" * 80)
    
    limited_stocks = df_info[df_info['status'] == 'LIMITED']
    for _, row in limited_stocks.iterrows():
        print(f"{row['symbol']}: {row['days_after_2020']} days after 2020")
    
    print("\n" + "=" * 80)
    print("STOCKS WITH NO DATA")
    print("=" * 80)
    
    no_data = df_info[df_info['status'].isin(['NO_FILE', 'EMPTY'])]
    print(f"\nTotal: {len(no_data)} stocks")
    print(f"Symbols: {', '.join(no_data['symbol'].tolist())}")
    
    # Save to CSV
    output_path = 'data/analysis/stock_data_availability.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_info.to_csv(output_path, index=False)
    print(f"\n\nDetailed report saved to: {output_path}")
    
    # Create list of all stocks with any data after 2018
    print("\n" + "=" * 80)
    print("STOCKS WITH ANY DATA AFTER 2018")
    print("=" * 80)
    
    stocks_with_data = []
    for _, row in df_info.iterrows():
        if row['total_days'] > 100:  # At least 100 days of data
            stocks_with_data.append(row['symbol'])
    
    print(f"\nTotal: {len(stocks_with_data)} stocks")
    print(f"Symbols: {stocks_with_data}")
    
    return stocks_with_data


if __name__ == "__main__":
    stocks_with_data = check_stock_data()