#!/usr/bin/env python3
"""
Test ML Trading System
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
    timeframe = '1h'
    
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


def test_ml_training():
    """Test ML model training"""
    logger.info("\nTesting ML Model Training...")
    
    ml_system = MLTradingSystem()
    
    # Use fewer symbols for testing
    test_symbols = ['THYAO', 'GARAN', 'SAHOL']
    timeframe = '1h'
    
    # Check what features are expected vs available
    logger.info("Expected features:")
    expected_features = ml_system.get_feature_list()
    logger.info(f"Total expected: {len(expected_features)}")
    
    # Load one symbol to check available features
    test_df = ml_system.load_and_prepare_data(test_symbols[0], timeframe)
    if test_df is not None:
        logger.info(f"\nAvailable features in data: {len(test_df.columns)}")
        
        # Check missing features
        missing = [f for f in expected_features if f not in test_df.columns]
        if missing:
            logger.warning(f"Missing features: {missing}")
        
        # Check extra features
        extra = [f for f in test_df.columns if f not in expected_features and 
                f not in ['open', 'high', 'low', 'close', 'volume', 'target', 'future_return']]
        if extra:
            logger.info(f"Extra features available: {extra[:10]}...")
    
    # Train models
    try:
        ml_system.train_ensemble_models(test_symbols, timeframe)
        logger.info("Model training completed successfully!")
        
        # Test prediction
        prediction = ml_system.predict_ensemble(test_symbols[0], timeframe)
        if prediction:
            logger.info(f"\nTest prediction for {test_symbols[0]}:")
            logger.info(f"Signal: {prediction['ensemble_prediction']}")
            logger.info(f"Confidence: {prediction['confidence']:.2f}")
            logger.info(f"Individual models: {prediction['individual_predictions']}")
        
        return ml_system
        
    except Exception as e:
        logger.error(f"Error in training: {e}")
        return None


def test_signal_generation():
    """Test signal generation"""
    logger.info("\nTesting Signal Generation...")
    
    # First train a model
    ml_system = test_ml_training()
    if ml_system is None:
        logger.error("Model training failed, cannot test signal generation")
        return
    
    # Create signal generator with trained model
    generator = MLSignalGenerator()
    generator.ml_system = ml_system
    
    # Test with a few symbols
    test_symbols = ['THYAO', 'GARAN', 'SAHOL']
    timeframe = '1h'
    
    # Generate individual signals
    for symbol in test_symbols:
        signal = generator.generate_signal(symbol, timeframe)
        if signal:
            logger.info(f"\n{symbol} Signal:")
            logger.info(f"  Direction: {signal.signal} ({'BUY' if signal.signal > 0 else 'SELL' if signal.signal < 0 else 'HOLD'})")
            logger.info(f"  Confidence: {signal.confidence:.2%}")
            logger.info(f"  Risk metrics: Vol={signal.risk_metrics['volatility']:.3f}, DD={signal.risk_metrics['max_drawdown']:.2%}")
    
    # Generate portfolio signals
    portfolio = generator.generate_portfolio_signals(ASSETS[:20], '1d')
    
    # Print report
    report = generator.create_trading_report(portfolio)
    print("\n" + report)
    
    # Save signals
    generator.save_signals(portfolio, "ml_models/test_signals.json")
    logger.info("Test signals saved to ml_models/test_signals.json")


def check_indicator_data():
    """Check available indicator data"""
    logger.info("\nChecking Indicator Data Availability...")
    
    csv_manager = CSVDataManager()
    symbol = 'THYAO'
    timeframe = '1h'
    
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
                has_hist = 'macd_histogram' in data.columns
                has_signal = 'macd_buy_signal' in data.columns
                logger.info(f"  Has histogram: {has_hist}")
                logger.info(f"  Has buy signal: {has_signal}")
        else:
            logger.warning(f"{indicator}: No data found")


def main():
    """Run all tests"""
    logger.info("Starting ML System Tests...")
    
    # 1. Check indicator data
    check_indicator_data()
    
    # 2. Test feature engineering
    test_feature_engineering()
    
    # 3. Test ML training and signal generation
    test_signal_generation()
    
    logger.info("\nAll tests completed!")


if __name__ == "__main__":
    main()