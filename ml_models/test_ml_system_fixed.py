#!/usr/bin/env python3
"""
Test ML Trading System - Fixed Version
"""

import sys
from pathlib import Path
import pandas as pd
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_models.ml_trading_system import MLTradingSystem
from ml_models.ml_signal_generator import MLSignalGenerator
from ml_models.feature_engineering_advanced import AdvancedFeatureEngineering
from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS


def test_feature_engineering():
    """Test feature engineering"""
    logger.info("Testing Feature Engineering...")
    
    csv_manager = CSVDataManager()
    fe = AdvancedFeatureEngineering()
    
    # Test with one symbol
    symbol = 'THYAO'
    timeframe = '1d'  # Changed to 1d for daily model training
    
    # Load data
    df = csv_manager.load_raw_data(symbol, timeframe)
    if df is None:
        logger.error(f"Could not load data for {symbol}")
        return
    
    logger.info(f"Loaded {len(df)} rows for {symbol}")
    
    # Load indicator data
    indicator_data = {
        'lorentzian': csv_manager.load_indicator_data(symbol, timeframe, 'lorentzian'),
        'trend_vanguard': csv_manager.load_indicator_data(symbol, timeframe, 'trend_vanguard')
    }
    
    # Apply feature engineering
    df_features = fe.engineer_all_features(df.copy(), indicator_data)
    
    logger.info(f"Created {len(df_features.columns)} features")
    logger.info(f"Feature categories: {fe.feature_stats}")
    
    return df_features


def test_ml_training(timeframe='1d'):
    """Test ML model training"""
    logger.info(f"\nTesting ML Model Training for {timeframe} timeframe...")
    
    ml_system = MLTradingSystem()
    
    # Use main symbols for training
    test_symbols = ['THYAO', 'GARAN', 'SAHOL', 'AKBNK', 'EREGL', 'KCHOL', 'SISE', 'FROTO', 'ASELS', 'BIMAS']
    
    # Check what features are expected vs available
    logger.info("Expected features:")
    expected_features = ml_system.get_feature_list()
    logger.info(f"Total expected: {len(expected_features)}")
    
    # Load one symbol to check available features
    test_df = ml_system.load_and_prepare_data(test_symbols[0], timeframe)
    if test_df is not None:
        logger.info(f"\nAvailable features in data: {len(test_df.columns)}")
        logger.info(f"Target distribution: {test_df['target'].value_counts().to_dict()}")
        
        # Check missing features
        missing = [f for f in expected_features if f not in test_df.columns]
        if missing:
            logger.warning(f"Missing features: {missing[:10]}...")
        
        # Check extra features
        extra = [f for f in test_df.columns if f not in expected_features and 
                f not in ['open', 'high', 'low', 'close', 'volume', 'target', 'future_return']]
        if extra:
            logger.info(f"Extra features available: {extra[:10]}...")
    
    # Train models
    try:
        ml_system.train_ensemble_models(test_symbols, timeframe)
        logger.info(f"Model training completed successfully for {timeframe}!")
        
        # Save the trained models
        ml_system.save_models(timeframe)
        logger.info(f"Models saved for {timeframe}")
        
        # Test prediction
        for symbol in test_symbols[:3]:
            prediction = ml_system.predict_ensemble(symbol, timeframe)
            if prediction:
                logger.info(f"\nTest prediction for {symbol}:")
                logger.info(f"Signal: {prediction['ensemble_prediction']}")
                logger.info(f"Confidence: {prediction['confidence']:.2f}")
                logger.info(f"Individual models: {prediction['individual_predictions']}")
        
        return ml_system
        
    except Exception as e:
        logger.error(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_signal_generation(timeframe='1d'):
    """Test signal generation"""
    logger.info(f"\nTesting Signal Generation for {timeframe}...")
    
    # First train a model
    ml_system = test_ml_training(timeframe)
    if ml_system is None:
        logger.error("Model training failed, cannot test signal generation")
        return
    
    # Create signal generator with trained model
    generator = MLSignalGenerator()
    generator.ml_system = ml_system
    
    # Test with main symbols
    test_symbols = ASSETS[:20]
    
    # Generate individual signals
    for symbol in test_symbols[:5]:
        signal = generator.generate_signal(symbol, timeframe)
        if signal:
            logger.info(f"\n{symbol} Signal:")
            logger.info(f"  Direction: {signal.signal} ({'BUY' if signal.signal > 0 else 'SELL' if signal.signal < 0 else 'HOLD'})")
            logger.info(f"  Confidence: {signal.confidence:.2%}")
            logger.info(f"  Risk metrics: Vol={signal.risk_metrics['volatility']:.3f}, DD={signal.risk_metrics['max_drawdown']:.2%}")
    
    # Generate portfolio signals
    portfolio = generator.generate_portfolio_signals(test_symbols, timeframe)
    
    # Print report
    report = generator.create_trading_report(portfolio)
    print("\n" + report)
    
    # Save signals
    generator.save_signals(portfolio, "ml_models/test_signals.json")
    logger.info("Test signals saved to ml_models/test_signals.json")


def check_indicator_data(timeframe='1d'):
    """Check available indicator data"""
    logger.info(f"\nChecking Indicator Data Availability for {timeframe}...")
    
    csv_manager = CSVDataManager()
    symbol = 'THYAO'
    
    indicators = [
        'supertrend', 'squeeze_momentum', 'macd', 'wavetrend', 
        'adx_di', 'lorentzian', 'trend_vanguard'
    ]
    
    for indicator in indicators:
        data = csv_manager.load_indicator_data(symbol, timeframe, indicator)
        if data is not None:
            logger.info(f"\n{indicator}:")
            logger.info(f"  Rows: {len(data)}")
            logger.info(f"  Columns: {list(data.columns)}")
            
            # Check for specific columns
            if indicator == 'macd':
                has_hist = 'macd_histogram' in data.columns or 'macd_hist' in data.columns
                has_signal = 'macd_buy_signal' in data.columns
                logger.info(f"  Has histogram: {has_hist}")
                logger.info(f"  Has buy signal: {has_signal}")
        else:
            logger.warning(f"{indicator}: No data found")


def main():
    """Run all tests"""
    logger.info("Starting ML System Tests (Fixed Version)...")
    
    # Test for both 1h and 1d timeframes
    for timeframe in ['1d', '1h']:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {timeframe} timeframe")
        logger.info(f"{'='*60}")
        
        # 1. Check indicator data
        check_indicator_data(timeframe)
        
        # 2. Test feature engineering
        test_feature_engineering()
        
        # 3. Test ML training and signal generation
        test_signal_generation(timeframe)
    
    logger.info("\nAll tests completed!")


if __name__ == "__main__":
    main()