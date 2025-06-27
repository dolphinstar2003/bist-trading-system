#!/usr/bin/env python3
"""
Backtest Test Script
"""

import sys
from pathlib import Path

# Proje kökünü ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtest.simple_indicator_backtest import SimpleIndicatorBacktest
from backtest.sirali_backtest import SiraliBacktest

print("""
========================================
BACKTEST TEST MENÜSÜ
========================================
1. Simple Indicator Backtest (3'lü Onay)
2. Sıralı Backtest (4 Fazlı)
3. İkisini de çalıştır
========================================
""")

choice = input("Seçiminiz (1-3): ")

if choice in ['1', '3']:
    print("\n--- SIMPLE INDICATOR BACKTEST ---")
    simple_bt = SimpleIndicatorBacktest(initial_capital=50000, max_positions=10)
    
    # Timeframe seçimi
    print("\nTimeframe: 1) 1d  2) 4h  3) 1h  4) 15m")
    tf_choice = input("Seçim (1-4): ")
    tf_map = {'1': '1d', '2': '4h', '3': '1h', '4': '15m'}
    timeframe = tf_map.get(tf_choice, '1h')
    
    simple_bt.run_all_symbols(timeframe)

if choice in ['2', '3']:
    print("\n--- SIRALI BACKTEST ---")
    sirali_bt = SiraliBacktest(initial_capital=50000, max_positions=10)
    
    # Timeframe seçimi
    print("\nTimeframe: 1) 1d  2) 4h  3) 1h  4) 15m")
    tf_choice = input("Seçim (1-4): ")
    tf_map = {'1': '1d', '2': '4h', '3': '1h', '4': '15m'}
    timeframe = tf_map.get(tf_choice, '1h')
    
    sirali_bt.run_all_symbols(timeframe)

print("\nBacktest tamamlandı!")