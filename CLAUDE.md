# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development and Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run main trading system
python main.py --mode paper  # Paper trading (default)
python main.py --mode live   # Live trading (requires confirmation)
python main.py --mode backtest --start 2024-01-01 --end 2024-12-31
python main.py --mode update-data  # Update market data only

# Run backtests
python backtest/simple_indicator_backtest.py  # 3-confirmation system backtest
python backtest/walk_forward_backtest.py       # Walk-forward backtest
python backtest/walk_forward_sequential.py     # Sequential walk-forward
python backtest/trimode_backtest.py           # TriMode orchestrator backtest
python backtest/trimode_enhanced_backtest.py  # Enhanced with quick wins
python backtest/trimode_aggressive_backtest.py # Aggressive mode testing

# Run ML system tests
python ml_models/test_ml_system.py
python ml_models/run_ml_pipeline.py
python ml_models/ml_trading_system_fixed.py --train --timeframe 1d

# Calculate indicators
python indicators/indicator_calculator_optimized.py --symbols all --timeframe 1h

# Test individual components
python tests/test_indicators.py
python tests/test_lorentzian_optimized.py
python tests/test_trend_vanguard_optimized.py
python tests/test_adaptive_system.py

# Data management
python download_data_yahoo_proper.py  # Download Yahoo Finance data
python data/download_data_multi_source.py  # Multi-source data download
python data/data_download_incremental.py  # Incremental updates

# Check data status
python utils/check_data_status.py

# Debug tools
python backtest/debug_signals.py  # Debug signal generation issues
```

### Linting and Code Quality

```bash
# Format code with black
black . --line-length 120

# Run flake8 linter
flake8 . --max-line-length=120 --exclude=algolab,trading

# Type checking with mypy
mypy . --ignore-missing-imports
```

## High-Level Architecture

### Core System Design

The trading system follows a modular architecture with clear separation of concerns:

1. **Data Layer** (`data/`, `utils/csv_data_manager.py`)
   - CSV-based data storage with caching for performance
   - Multi-source data acquisition (Yahoo Finance, Alpha Vantage)
   - Incremental data updates to minimize API calls
   - Raw data → Processed data → Indicator data pipeline

2. **Indicator Layer** (`indicators/`)
   - Technical indicators optimized with Numba JIT compilation
   - Lorentzian Classification (ML-based) - ~12x optimized
   - Trend Vanguard (market regime detection) - ~35x optimized
   - Standard indicators: Supertrend, Squeeze Momentum, MACD, WaveTrend, ADX/DI
   - Parallel calculation support for multiple symbols/timeframes

3. **ML System** (`ml_models/`)
   - Feature engineering with 200+ features across multiple categories
   - Ensemble learning: Random Forest, Gradient Boosting, XGBoost
   - Time-series aware training with proper train/test splits
   - Dynamic model selection based on market conditions
   - Signal generation with confidence scoring and risk metrics
   - XGBoost label fix: Convert -1/1 to 0/1 labels

4. **Strategy Layer** (`strategies/`)
   - Adaptive ensemble system with dynamic weight adjustment
   - Market regime detection (bullish/bearish/neutral)
   - Integrated trading system combining ML + indicators
   - Risk-adjusted position sizing
   - TriMode Orchestrator: Aggressive/Balanced/Defensive modes
   - EMA cross optimization with reasonable parameters

5. **Risk Management** (`risk/`)
   - Dynamic risk manager with volatility-based adjustments
   - Portfolio correlation limits
   - Maximum drawdown protection
   - Position sizing with Kelly Criterion
   - ATR-based trailing stops
   - Partial profit taking (50% at targets)

6. **Backtesting** (`backtest/`)
   - Simple indicator backtest: 3-confirmation system
   - Walk-forward backtest: Dynamic stock selection
   - Sequential backtest: Order-based execution
   - TriMode backtests: Mode-switching strategy testing
   - Realistic slippage and commission modeling

### Key Design Patterns

1. **CSV Data Architecture**
   - All market data stored in CSV format for portability
   - Directory structure: `data/symbols/{SYMBOL}/{TIMEFRAME}.csv`
   - Indicator data: `data/indicators/{INDICATOR}/{SYMBOL}_{TIMEFRAME}.csv`
   - Efficient caching with `.cache/` directory

2. **Indicator Optimization Strategy**
   - Vectorized operations with NumPy
   - Numba JIT compilation for compute-intensive loops
   - Caching of intermediate calculations
   - Parallel processing for multiple symbols

3. **ML Pipeline**
   - Feature groups: price_features, trend_indicators, momentum_indicators, ml_indicators, pattern_features
   - XGBoost label mapping: -1 (sell) → 0, 1 (buy) → 1
   - Ensemble predictions with confidence weighting
   - Walk-forward validation to prevent look-ahead bias

4. **Trading Signal Flow**
   ```
   Raw Data → Indicators → Feature Engineering → ML Models → Signal Generation → Risk Management → Order Execution
   ```

### Critical Implementation Details

1. **AlgoLab Integration** (`algolab/`)
   - DO NOT modify files in this directory
   - SMS authentication flow handled in `main.py`
   - WebSocket connection for real-time data

2. **Data Synchronization**
   - Yahoo Finance as primary source
   - Alpha Vantage for validation
   - Incremental updates track last processed dates
   - Missing data detection and backfill

3. **Performance Optimizations**
   - Lorentzian: Uses KDTree for neighbor search, vectorized distance calculations
   - Trend Vanguard: Optimized pivot detection, cached regime calculations
   - CSV Manager: In-memory caching, batch operations

4. **Common Issues and Solutions**
   - `MACD histogram`: Use 'macd_hist' not 'macd_histogram' in CSV data
   - XGBoost labels: Must be 0/1, not -1/1
   - True range calculation: Proper handling of shifted close prices
   - Array length checks before argpartition operations
   - EMA optimization: Avoid overly long periods (>50) for slow EMA

### TriMode Orchestrator System

The TriMode system dynamically switches between three trading modes based on market conditions:

1. **Aggressive Mode**
   - Position size: 15% per trade
   - Stop loss: 12% with 4x ATR trailing
   - Max positions: 10
   - Minimal filters for more signals
   - Default mode in favorable conditions

2. **Balanced Mode**
   - Position size: 10% per trade
   - Stop loss: 8% with 3.5x ATR trailing
   - Max positions: 8
   - Moderate confirmation requirements
   - Used in normal market conditions

3. **Defensive Mode**
   - Position size: 5% per trade
   - Stop loss: 5% with 3x ATR trailing
   - Max positions: 5
   - Strict ML-based confirmations
   - Activated during high volatility or drawdowns

### Configuration

Main configuration in `settings.json`:
- Trading symbols list (59 BIST stocks)
- Risk parameters (max position size, stop loss, drawdown limits)
- ML model configurations and ensemble weights
- Indicator parameters
- API credentials (stored separately in .env)

Optimal EMA parameters stored in:
- `backtest/optimal_ema_params.json` (full optimization results)
- `backtest/reasonable_ema_params.json` (filtered reasonable values)

### Testing Strategy

1. Unit tests for individual indicators
2. Integration tests for ML pipeline
3. Backtest validation on historical data
4. Paper trading before live deployment

### Recent Enhancements

1. **Quick Win Features**
   - ATR-based trailing stops per mode
   - Partial profit taking (40-60% at targets)
   - Volume spike filter (RVOL > 1.5)
   - Optimal EMA parameter storage
   - Volatility-based position sizing

2. **ML System Fixes**
   - Fixed NaN conversion errors with `.fillna(0)`
   - Corrected XGBoost label mapping
   - Improved feature engineering pipeline

3. **Strategy Improvements**
   - More lenient mode switching conditions
   - Wider stop losses to reduce premature exits
   - Lower profit targets for quicker gains
   - Simplified signal generation logic

## Important Notes

- The system is designed for BIST (Turkish stock market) with TRY currency
- Market hours: 09:10 - 18:00 Turkish time, Monday-Friday
- All timestamps are in Turkish timezone
- Minimum recommended capital: 50,000 TRY for effective diversification
- Target monthly return: 10% (aggressive but achievable with proper risk management)