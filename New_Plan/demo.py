"""
Hybrid Trading System Demo
Shows the working system with actual calculations
"""

import asyncio
import json
from datetime import datetime
from loguru import logger
import pandas as pd
import numpy as np

# System modules
from core.csv_data_manager import CSVDataManager
from indicators.indicator_calculator import IndicatorCalculator

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg), level="INFO")


def demo_data_and_indicators():
    """Demo: Load data and calculate indicators"""
    print("\n" + "="*60)
    print("HYBRID TRADING SYSTEM DEMO")
    print("Based on Hibrit Alım-Satım Algoritması Research Report")
    print("="*60)
    
    # Initialize
    csv_manager = CSVDataManager()
    calc = IndicatorCalculator()
    
    # Get available symbols
    symbols = csv_manager.get_available_symbols()
    print(f"\n✓ Found {len(symbols)} symbols in the system")
    
    # Demo with THYAO (Turkish Airlines)
    symbol = 'THYAO'
    timeframe = '1h'
    
    print(f"\n📊 DEMO: {symbol} Analysis")
    print("-" * 40)
    
    # 1. Load raw data
    df = csv_manager.get_raw_data(symbol, timeframe)
    if df is None:
        print(f"No data found for {symbol}")
        return
    
    print(f"✓ Loaded {len(df)} {timeframe} bars")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Last close: {df['close'].iloc[-1]:.2f} TRY")
    
    # 2. Calculate indicators
    print("\n📈 Calculating Technical Indicators...")
    
    # MACD (most reliable according to research)
    trend_indicators = calc.calculate_trend_indicators(df)
    
    macd = trend_indicators['macd'].iloc[-1]
    macd_signal = trend_indicators['macd_signal'].iloc[-1]
    macd_hist = trend_indicators['macd_hist'].iloc[-1]
    
    print(f"\nMACD Analysis:")
    print(f"  MACD Line: {macd:.4f}")
    print(f"  Signal Line: {macd_signal:.4f}")
    print(f"  Histogram: {macd_hist:.4f}")
    print(f"  Position: {'BULLISH' if macd > macd_signal else 'BEARISH'} 📊")
    
    # RSI
    momentum_indicators = calc.calculate_momentum_indicators(df)
    rsi = momentum_indicators['rsi'].iloc[-1]
    
    print(f"\nMomentum Analysis:")
    print(f"  RSI(14): {rsi:.2f}")
    if rsi > 70:
        print(f"  Status: OVERBOUGHT ⚠️")
    elif rsi < 30:
        print(f"  Status: OVERSOLD 🔥")
    else:
        print(f"  Status: NEUTRAL ➖")
    
    # ATR for stop loss (2x ATR recommended)
    volatility_indicators = calc.calculate_volatility_indicators(df)
    atr = volatility_indicators['atr'].iloc[-1]
    atr_pct = volatility_indicators['atr_percent'].iloc[-1]
    
    print(f"\nRisk Management:")
    print(f"  ATR: {atr:.2f} ({atr_pct:.2f}%)")
    print(f"  Recommended Stop Loss: {2 * atr:.2f} points (2x ATR)")
    print(f"  Stop Loss %: {2 * atr_pct:.2f}%")
    
    # Volume analysis
    volume_indicators = calc.calculate_volume_indicators(df)
    volume_ratio = volume_indicators['volume_ratio'].iloc[-1]
    
    print(f"\nVolume Analysis:")
    print(f"  Volume Ratio: {volume_ratio:.2f}x average")
    print(f"  Status: {'HIGH VOLUME 📊' if volume_ratio > 1.5 else 'NORMAL' if volume_ratio > 0.8 else 'LOW VOLUME ⚠️'}")
    
    # 3. Multi-timeframe overview
    print(f"\n🔄 Multi-Timeframe Analysis for {symbol}:")
    print("-" * 40)
    
    timeframes_to_check = ['15m', '1h', '4h', '1d']
    mtf_signals = {}
    
    for tf in timeframes_to_check:
        tf_df = csv_manager.get_raw_data(symbol, tf)
        if tf_df is not None and len(tf_df) > 26:
            # Quick MACD calculation
            ema_12 = tf_df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = tf_df['close'].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()
            
            current_signal = 'BUY' if macd.iloc[-1] > signal.iloc[-1] else 'SELL'
            mtf_signals[tf] = current_signal
            
            print(f"  {tf:>4}: {current_signal:>4} | Close: {tf_df['close'].iloc[-1]:>7.2f}")
    
    # 4. Trading recommendation
    print(f"\n💡 TRADING RECOMMENDATION:")
    print("-" * 40)
    
    buy_signals = sum(1 for s in mtf_signals.values() if s == 'BUY')
    total_signals = len(mtf_signals)
    
    if buy_signals > total_signals * 0.6:
        print("  Signal: STRONG BUY ✅")
        print(f"  Entry: {df['close'].iloc[-1]:.2f}")
        print(f"  Stop Loss: {df['close'].iloc[-1] - 2*atr:.2f} (-{2*atr_pct:.1f}%)")
        print(f"  Target 1: {df['close'].iloc[-1] + 4*atr:.2f} (+{4*atr_pct:.1f}%) - 2:1 R/R")
        print(f"  Target 2: {df['close'].iloc[-1] + 6*atr:.2f} (+{6*atr_pct:.1f}%) - 3:1 R/R")
        print(f"  Position Size: 1% risk (Kelly 25% fraction)")
    elif buy_signals < total_signals * 0.4:
        print("  Signal: AVOID/SELL ❌")
        print("  Reason: Bearish multi-timeframe alignment")
    else:
        print("  Signal: NEUTRAL ➖")
        print("  Reason: Mixed signals across timeframes")
    
    # 5. System features
    print(f"\n🚀 SYSTEM FEATURES:")
    print("-" * 40)
    print("✓ Multi-timeframe analysis (15m, 1h, 4h, 1d, 1w)")
    print("✓ MACD-focused strategy (most reliable per research)")
    print("✓ 1% risk management with 2x ATR stops")
    print("✓ Kelly Criterion position sizing (25% fraction)")
    print("✓ Target: 8-9% monthly returns")
    print("✓ CSV data integration with existing system")
    print("✓ GRU neural network for signal confirmation")
    print("✓ Algolab WebSocket ready for live trading")
    
    return symbol, df


def show_other_opportunities(csv_manager, calc):
    """Show other potential trading opportunities"""
    print(f"\n📊 SCANNING OTHER OPPORTUNITIES...")
    print("-" * 40)
    
    # Top BIST stocks to check
    watchlist = ['THYAO', 'GARAN', 'AKBNK', 'EREGL', 'ASELS', 'SASA', 'KCHOL']
    opportunities = []
    
    for symbol in watchlist:
        df = csv_manager.get_raw_data(symbol, '1d')
        if df is not None and len(df) > 50:
            # Quick analysis
            ema_9 = df['close'].ewm(span=9, adjust=False).mean()
            ema_21 = df['close'].ewm(span=21, adjust=False).mean()
            
            # Simple momentum
            momentum = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
            
            # MACD
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9, adjust=False).mean()
            
            if macd.iloc[-1] > signal.iloc[-1] and ema_9.iloc[-1] > ema_21.iloc[-1]:
                opportunities.append({
                    'symbol': symbol,
                    'close': df['close'].iloc[-1],
                    'momentum': momentum,
                    'volume_ratio': df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
                })
    
    if opportunities:
        print("\n🔥 BULLISH OPPORTUNITIES (Daily):")
        opportunities.sort(key=lambda x: x['momentum'], reverse=True)
        
        for opp in opportunities[:5]:
            print(f"  {opp['symbol']:>6}: {opp['close']:>7.2f} TRY | "
                  f"Momentum: {opp['momentum']:>5.1f}% | "
                  f"Volume: {opp['volume_ratio']:.1f}x")


async def main():
    """Main demo function"""
    try:
        # Run the demo
        symbol, df = demo_data_and_indicators()
        
        # Show other opportunities
        csv_manager = CSVDataManager()
        calc = IndicatorCalculator()
        show_other_opportunities(csv_manager, calc)
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print("\n📝 NEXT STEPS:")
        print("1. Run full backtests with: python backtest/run_backtest.py")
        print("2. Train the GRU model with: python train_model.py")
        print("3. Start paper trading with: python main.py --mode paper")
        print("4. Monitor performance with: python monitoring/dashboard.py")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())