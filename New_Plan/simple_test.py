"""
Simple test to verify system components
"""

import asyncio
from core.csv_data_manager import CSVDataManager
from indicators.indicator_calculator import IndicatorCalculator
from loguru import logger

# Disable debug logs for cleaner output
logger.remove()
logger.add(lambda msg: print(msg), level="INFO")


def test_csv_manager():
    """Test CSV data loading"""
    print("\n=== CSV DATA MANAGER TEST ===")
    
    csv_manager = CSVDataManager()
    symbols = csv_manager.get_available_symbols()
    print(f"Found {len(symbols)} symbols")
    
    # Test with first symbol
    if symbols:
        symbol = symbols[0]
        print(f"\nTesting with: {symbol}")
        
        # Test different timeframes
        for tf in ['1d', '1h', '15m']:
            df = csv_manager.get_raw_data(symbol, tf)
            if df is not None:
                print(f"  {tf}: {len(df)} rows, columns: {list(df.columns)}")
            else:
                print(f"  {tf}: No data")
    
    return symbols


def test_indicators():
    """Test indicator calculation"""
    print("\n=== INDICATOR CALCULATOR TEST ===")
    
    calculator = IndicatorCalculator()
    
    # Test with a symbol
    symbol = 'THYAO'
    timeframe = '1d'
    
    print(f"\nCalculating indicators for {symbol} {timeframe}...")
    
    # Get raw data first
    df = calculator.csv_manager.get_raw_data(symbol, timeframe)
    if df is None:
        print("No raw data found")
        return
    
    print(f"Raw data: {len(df)} rows")
    
    # Calculate trend indicators only
    trend_indicators = calculator.calculate_trend_indicators(df)
    print(f"\nTrend indicators calculated: {list(trend_indicators.keys())}")
    
    # Check MACD values
    if 'macd' in trend_indicators:
        macd = trend_indicators['macd']
        print(f"MACD last value: {macd.iloc[-1]:.4f}")
        print(f"MACD signal last value: {trend_indicators['macd_signal'].iloc[-1]:.4f}")


def test_multi_timeframe():
    """Test multi-timeframe data loading"""
    print("\n=== MULTI-TIMEFRAME TEST ===")
    
    csv_manager = CSVDataManager()
    symbol = 'AKBNK'
    
    print(f"\nLoading multi-timeframe data for {symbol}...")
    
    timeframes = ['15m', '1h', '4h', '1d']
    data = csv_manager.get_multi_timeframe_data(symbol, timeframes)
    
    for tf, df in data.items():
        print(f"  {tf}: {len(df)} rows, last close: {df['close'].iloc[-1]:.2f}")


async def test_basic_signal_generation():
    """Test basic signal generation flow"""
    print("\n=== BASIC SIGNAL TEST ===")
    
    from core.signal_generator import SignalGenerator
    
    # Minimal config
    config = {
        'api': {'finnhub': {'api_key': ''}, 'alpha_vantage': {'api_key': ''}},
        'data': {'cache_ttl': 900},
        'timeframes': {'analysis': ['1h', '4h', '1d']},
        'signals': {'confidence_threshold': 0.5, 'min_profit_target': 0.02},
        'max_concurrent_signals': 3
    }
    
    generator = SignalGenerator(config)
    
    # Test with one symbol
    symbols = ['THYAO']
    print(f"\nGenerating signals for: {symbols}")
    
    try:
        signals = await generator.generate_signals(symbols)
        
        if signals:
            print(f"Generated {len(signals)} signals")
            for signal in signals:
                print(f"\n{signal['symbol']} - {signal['direction']}")
                print(f"  Entry: {signal['entry_price']:.2f}")
                print(f"  Stop: {signal['stop_loss']:.2f}")
                print(f"  Confidence: {signal['confidence']:.1%}")
        else:
            print("No signals generated")
            
    except Exception as e:
        print(f"Signal generation error: {e}")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("HYBRID TRADING SYSTEM - SIMPLE TEST")
    print("=" * 60)
    
    try:
        # Test 1: CSV Manager
        symbols = test_csv_manager()
        
        # Test 2: Indicators
        test_indicators()
        
        # Test 3: Multi-timeframe
        test_multi_timeframe()
        
        # Test 4: Basic signal
        await test_basic_signal_generation()
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())