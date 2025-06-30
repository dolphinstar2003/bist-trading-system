#!/usr/bin/env python3
"""
Debug script for Feature Engineering
Diagnoses why feature engineering is not working properly
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from loguru import logger
import json
import traceback

# Add paths
sys.path.append(str(Path(__file__).parent))
from core.feature_engineering import FeatureEngineering
from core.csv_data_manager import CSVDataManager

# Configure logging
logger.remove()
logger.add(sys.stdout, level="DEBUG", format="{time:HH:mm:ss} | {level} | {message}")


def debug_feature_engineering():
    """Main debug function"""
    logger.info("=== Starting Feature Engineering Debug ===")
    
    # Test configuration
    config = {
        'timeframes': {
            'analysis': ['1h', '4h', '1d']
        },
        'symbols': ['THYAO']
    }
    
    symbol = 'THYAO'
    
    # Initialize managers
    logger.info("1. Initializing managers...")
    try:
        csv_manager = CSVDataManager()
        feature_eng = FeatureEngineering(config)
        logger.success("Managers initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize managers: {e}")
        traceback.print_exc()
        return
    
    # Load raw data
    logger.info("\n2. Loading raw data for all timeframes...")
    raw_data = {}
    for tf in config['timeframes']['analysis']:
        logger.info(f"Loading {symbol} {tf} data...")
        try:
            df = csv_manager.get_raw_data(symbol, tf)
            if df is not None and not df.empty:
                raw_data[tf] = df
                logger.success(f"✓ {tf}: Shape {df.shape}, Date range: {df.index[0]} to {df.index[-1]}")
                logger.debug(f"  Columns: {list(df.columns)[:10]}...")
                logger.debug(f"  Sample data:\n{df.head(2)}")
            else:
                logger.warning(f"✗ {tf}: No data found")
        except Exception as e:
            logger.error(f"✗ {tf}: Error loading - {e}")
            traceback.print_exc()
    
    if not raw_data:
        logger.error("No raw data could be loaded!")
        return
    
    # Load indicator data
    logger.info("\n3. Loading indicator data...")
    indicator_data = {}
    for tf in config['timeframes']['analysis']:
        if tf not in raw_data:
            continue
            
        logger.info(f"Loading indicators for {symbol} {tf}...")
        try:
            # Get all indicators combined with raw data
            ind_df = csv_manager.get_all_indicators(symbol, tf)
            if ind_df is not None and not ind_df.empty:
                indicator_data[tf] = ind_df
                logger.success(f"✓ {tf}: Shape {ind_df.shape}")
                
                # Show indicator columns
                indicator_cols = [col for col in ind_df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
                logger.info(f"  Found {len(indicator_cols)} indicator columns")
                if indicator_cols:
                    logger.debug(f"  Indicator columns: {indicator_cols[:10]}...")
                else:
                    logger.warning(f"  No indicator columns found!")
            else:
                logger.warning(f"✗ {tf}: No indicator data")
        except Exception as e:
            logger.error(f"✗ {tf}: Error loading indicators - {e}")
            traceback.print_exc()
    
    # Prepare data structure for feature engineering
    logger.info("\n4. Preparing data structure...")
    data = {
        'raw': raw_data,
        'indicators': indicator_data
    }
    
    # Debug the data structure
    logger.debug("Data structure:")
    logger.debug(f"- raw: {list(data['raw'].keys())}")
    logger.debug(f"- indicators: {list(data['indicators'].keys())}")
    
    # Create features
    logger.info("\n5. Creating features...")
    try:
        features = feature_eng.create_features(data, symbol)
        
        if features:
            logger.success(f"Features created for {len(features)} timeframes")
            
            # Analyze features for each timeframe
            for tf, feat_df in features.items():
                logger.info(f"\n{tf} Features:")
                logger.info(f"  Shape: {feat_df.shape}")
                logger.info(f"  Memory usage: {feat_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                # Feature groups
                feature_groups = {
                    'price': [col for col in feat_df.columns if 'price' in col or 'returns' in col],
                    'trend': [col for col in feat_df.columns if 'trend' in col or 'macd' in col or 'ema' in col],
                    'momentum': [col for col in feat_df.columns if 'momentum' in col or 'rsi' in col],
                    'volatility': [col for col in feat_df.columns if 'atr' in col or 'bb_' in col],
                    'volume': [col for col in feat_df.columns if 'volume' in col or 'obv' in col],
                    'pattern': [col for col in feat_df.columns if 'pattern' in col or 'candle' in col],
                    'multi_tf': [col for col in feat_df.columns if any(x in col for x in ['1h_', '4h_', '1d_', '1w_'])]
                }
                
                for group, cols in feature_groups.items():
                    if cols:
                        logger.info(f"  {group}: {len(cols)} features - {cols[:5]}...")
                
                # Check for NaN values
                nan_counts = feat_df.isna().sum()
                if nan_counts.any():
                    logger.warning(f"  NaN values found in {nan_counts[nan_counts > 0].shape[0]} columns")
                    logger.debug(f"  Columns with NaN: {list(nan_counts[nan_counts > 0].index[:10])}...")
                
                # Sample features
                logger.debug(f"\n  Sample features (last 5 rows):")
                logger.debug(f"{feat_df.tail()}")
                
                # Feature statistics
                logger.info(f"\n  Feature statistics:")
                stats = feat_df.describe()
                logger.debug(f"{stats[stats.columns[:5]]}")
                
        else:
            logger.error("No features were created!")
            
    except Exception as e:
        logger.error(f"Error creating features: {e}")
        traceback.print_exc()
        return
    
    # Test model input preparation
    logger.info("\n6. Testing model input preparation...")
    for tf in features.keys():
        try:
            model_input = feature_eng.prepare_model_input(features, tf, sequence_length=30)
            logger.success(f"✓ {tf}: Model input shape {model_input.shape}")
            logger.debug(f"  Input stats - Mean: {model_input.mean():.4f}, Std: {model_input.std():.4f}")
        except Exception as e:
            logger.error(f"✗ {tf}: Failed to prepare model input - {e}")
    
    # Save debug info
    logger.info("\n7. Saving debug information...")
    debug_info = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'timeframes': list(features.keys()) if 'features' in locals() else [],
        'raw_data_shapes': {tf: df.shape for tf, df in raw_data.items()},
        'indicator_data_shapes': {tf: df.shape for tf, df in indicator_data.items()},
        'feature_shapes': {tf: df.shape for tf, df in features.items()} if 'features' in locals() else {},
        'feature_columns': {tf: list(df.columns) for tf, df in features.items()} if 'features' in locals() else {}
    }
    
    debug_file = Path('debug_feature_engineering_report.json')
    with open(debug_file, 'w') as f:
        json.dump(debug_info, f, indent=2)
    logger.success(f"Debug info saved to {debug_file}")
    
    # Additional diagnostics
    logger.info("\n8. Additional diagnostics...")
    
    # Check if indicator files exist
    logger.info("Checking indicator files...")
    indicators_to_check = ['macd', 'rsi', 'bb', 'ema', 'adx', 'atr', 'obv', 'stoch']
    for tf in config['timeframes']['analysis']:
        logger.info(f"\n{tf} indicator files:")
        for indicator in indicators_to_check:
            indicator_file = csv_manager.indicators_path / f"{symbol}_{tf}_{indicator}.csv"
            if indicator_file.exists():
                logger.success(f"  ✓ {indicator}: {indicator_file}")
            else:
                logger.warning(f"  ✗ {indicator}: File not found")
    
    logger.info("\n=== Debug Complete ===")
    
    return features if 'features' in locals() else None


def check_raw_data_structure():
    """Check the structure of raw data files"""
    logger.info("\n=== Checking Raw Data Structure ===")
    
    csv_manager = CSVDataManager()
    symbol = 'THYAO'
    
    for tf in ['1h', '4h', '1d']:
        file_path = csv_manager.raw_data_path / f"{symbol}_{tf}_raw.csv"
        logger.info(f"\nChecking {file_path}...")
        
        if file_path.exists():
            # Read first few lines
            df = pd.read_csv(file_path, nrows=5)
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Sample:\n{df}")
            
            # Check date column
            if 'Date' in df.columns:
                logger.success("✓ Date column found")
            elif 'date' in df.columns:
                logger.warning("⚠ 'date' column found (lowercase)")
            elif 'Datetime' in df.columns:
                logger.warning("⚠ 'Datetime' column found (should be 'Date')")
            else:
                logger.error("✗ No date column found!")
        else:
            logger.error(f"✗ File not found: {file_path}")


def list_all_indicator_files():
    """List all available indicator files"""
    logger.info("\n=== Available Indicator Files ===")
    
    csv_manager = CSVDataManager()
    
    # Check both paths
    for path_name, path in [("Original", csv_manager.indicators_path), 
                            ("New_Plan", csv_manager.new_indicators_path)]:
        logger.info(f"\n{path_name} indicators path: {path}")
        
        if path.exists():
            indicator_files = sorted(path.glob("*.csv"))
            if indicator_files:
                logger.success(f"Found {len(indicator_files)} indicator files:")
                for f in indicator_files[:20]:  # Show first 20
                    logger.info(f"  - {f.name}")
                if len(indicator_files) > 20:
                    logger.info(f"  ... and {len(indicator_files) - 20} more")
            else:
                logger.warning("No indicator files found")
        else:
            logger.error(f"Path does not exist: {path}")


if __name__ == "__main__":
    # First check raw data structure
    check_raw_data_structure()
    
    # List available indicator files
    list_all_indicator_files()
    
    # Then run main debug
    features = debug_feature_engineering()
    
    if features:
        logger.success("\n✓ Feature engineering debug completed successfully!")
    else:
        logger.error("\n✗ Feature engineering debug failed!")