# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development and Testing

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r dl_models/requirements_dl.txt  # Deep Learning dependencies

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

# Run ML trading systems
python ml_models/ml_trading_system_fixed.py --train --timeframe 1d
python ml_models/ml_price_prediction_ensemble.py  # 8-model ensemble
python ml_models/ml_portfolio_optimizer.py        # Portfolio optimization
python ml_models/ml_aggressive_trader.py          # 2x leverage aggressive
python ml_models/ml_profit_accumulator.py         # Fixed position accumulator
python ml_models/ml_wave_rider.py                 # Multi-timeframe momentum
python ml_models/ml_quality_upgrader.py           # Adaptive rotation system
python ml_models/ml_momentum_burst.py             # Short-term burst catcher

# Run Deep Learning trading systems
python dl_models/dl_lstm_price_predictor.py      # LSTM time series prediction
python dl_models/dl_cnn_pattern_detector.py       # CNN candlestick patterns
python dl_models/dl_transformer_predictor.py      # Transformer market analysis
python dl_models/dl_reinforcement_trader.py       # RL portfolio optimization
python dl_models/dl_hybrid_ensemble.py            # ML+DL hybrid ensemble

# DL model backtesting
python dl_models/dl_backtest_framework.py         # Comprehensive DL backtesting

# Optimal stop loss analysis
python optimal_stop_loss_finder.py                # Find optimal stops
python advanced_stop_loss_analyzer.py             # Advanced analysis with trailing

# Calculate indicators
python indicators/indicator_calculator_optimized.py --symbols all --timeframe 1h

# Data management
python download_data_yahoo_proper.py              # Download Yahoo Finance data
python data/download_data_multi_source.py         # Multi-source data download
python data/data_download_incremental.py          # Incremental updates
python download_missing_stocks.py                 # Download missing stocks
python fix_date_format.py                         # Fix date format issues

# Check data status
python utils/check_data_status.py
python check_all_stocks_data.py                   # Check stock availability

# Debug tools
python backtest/debug_signals.py                  # Debug signal generation
python debug_wave_rider.py                        # Debug wave rider parameters
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
   - Multi-timeframe support: 1d, 4h, 1h data available
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
   - Ensemble learning: Random Forest, Gradient Boosting, XGBoost, LightGBM
   - Time-series aware training with proper train/test splits
   - Dynamic model selection based on market conditions
   - Signal generation with confidence scoring and risk metrics
   - XGBoost label fix: Convert -1/1 to 0/1 labels

4. **Deep Learning System** (`dl_models/`)
   - LSTM with attention for temporal pattern recognition
   - CNN for visual pattern detection in candlestick charts
   - Transformer models for capturing long-range dependencies
   - Reinforcement Learning agents (DQN, PPO, A2C) for adaptive trading
   - Hybrid ML+DL ensemble with performance-based weighting
   - GPU acceleration support for faster training

5. **Strategy Layer** (`strategies/`)
   - Adaptive ensemble system with dynamic weight adjustment
   - Market regime detection (bullish/bearish/neutral)
   - Integrated trading system combining ML + indicators
   - Risk-adjusted position sizing
   - TriMode Orchestrator: Aggressive/Balanced/Defensive modes
   - EMA cross optimization with reasonable parameters

6. **Risk Management** (`risk/`)
   - Dynamic risk manager with volatility-based adjustments
   - Portfolio correlation limits
   - Maximum drawdown protection
   - Position sizing with Kelly Criterion
   - ATR-based trailing stops
   - Partial profit taking (50% at targets)
   - Optimal stop loss parameters from historical analysis

7. **Backtesting** (`backtest/`)
   - Simple indicator backtest: 3-confirmation system
   - Walk-forward backtest: Dynamic stock selection
   - Sequential backtest: Order-based execution
   - TriMode backtests: Mode-switching strategy testing
   - Realistic slippage and commission modeling

### Key Design Patterns

1. **CSV Data Architecture**
   - All market data stored in CSV format for portability
   - Directory structure: `data/raw/{SYMBOL}_{TIMEFRAME}_raw.csv`
   - Indicator data: `data/indicators/{SYMBOL}_{TIMEFRAME}_{INDICATOR}.csv`
   - Analysis data: `data/analysis/` (stop losses, parameters, strategies)
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
   Raw Data → Indicators → Feature Engineering → ML/DL Models → Ensemble → Risk Management → Order Execution
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
   - Date format standardization (capital 'Date' column)

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
   - Date format: Ensure 'Date' column with capital D

### Advanced Trading Systems

1. **ML Profit Accumulator** (`ml_models/ml_profit_accumulator.py`)
   - Fixed position sizing (8,000 TL default)
   - Dynamic profit targets based on historical rally/drawdown patterns
   - Portfolio expansion every 50,000 TL milestone
   - Stock-specific targets using R/R ratios

2. **Wave Rider** (`ml_models/ml_wave_rider.py`)
   - Multi-timeframe analysis (Weekly/Daily/4H/1H)
   - Pyramid position building on winners
   - Optimal stop loss integration from analysis data
   - Wave stages: Initial → Building → Riding → Exiting

3. **Quality Upgrader** (`ml_models/ml_quality_upgrader.py`)
   - Market regime adaptive (Bull/Bear/Sideways)
   - Quality-based stock rotation
   - Blacklist management for recently sold stocks
   - Dynamic position limits based on regime

4. **Momentum Burst Catcher** (`ml_models/ml_momentum_burst.py`)
   - Bollinger Band squeeze breakout detection
   - Volume spike requirements (2x+ average)
   - RSI momentum burst detection
   - Tight stops (3-5%) with quick targets (5-10%)

### Optimal Parameters

Stored in `data/analysis/`:
- `advanced_trading_parameters.csv`: Per-stock optimal stop loss, trailing stops, take profits
- `trading_strategies.csv`: Risk/reward ratios and strategy recommendations
- `optimal_stop_losses.csv`: Historical drawdown-based stop losses

Key parameter files:
- `backtest/optimal_ema_params.json`: Full EMA optimization results
- `backtest/reasonable_ema_params.json`: Filtered reasonable EMA values

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

### Testing Strategy

1. Unit tests for individual indicators
2. Integration tests for ML pipeline
3. Backtest validation on historical data
4. Paper trading before live deployment

### Recent Enhancements

1. **Expanded Universe**
   - Increased from 33 to 58 stocks
   - Added missing stocks: TCELL, TOASO, TUPRS, etc.
   - Data validation and format standardization

2. **ML System Improvements**
   - Fixed NaN conversion errors with `.fillna(0)`
   - Corrected XGBoost label mapping
   - Improved feature engineering pipeline
   - Added ML ensemble systems

3. **Strategy Enhancements**
   - Stock-specific optimal parameters
   - Dynamic profit targets based on historical patterns
   - Multi-timeframe confirmation systems
   - Market regime adaptation

4. **Deep Learning Integration**
   - LSTM/GRU models for time series prediction with attention mechanism
   - CNN for candlestick pattern recognition (ResNet option available)
   - Transformer architecture for long-range market dependencies
   - Reinforcement Learning (DQN/PPO/A2C) for portfolio optimization
   - Hybrid ML+DL ensemble with adaptive weighting
   - Comprehensive DL backtesting framework with walk-forward analysis

## Important Notes

- The system is designed for BIST (Turkish stock market) with TRY currency
- Market hours: 09:10 - 18:00 Turkish time, Monday-Friday
- All timestamps are in Turkish timezone
- Minimum recommended capital: 50,000 TRY for effective diversification
- Target monthly return: 10% (aggressive but achievable with proper risk management)
- Bank deposit benchmark: 50% annual (3.4% monthly)